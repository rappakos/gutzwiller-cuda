#!/usr/bin/env python3
"""
Time-dependent Gutzwiller (TDGW) CPU reference solver  --  TRUSTED ORACLE
=========================================================================

Small-lattice, high-precision reference for the Bose-Hubbard mean-field
dynamics of Rapp, PRA 90, 053607 (2014), Eq. (14)-(15).

This file is *deliberately* slow, vectorised-NumPy, and obvious. It is NOT
meant to scale -- it is the ground truth the CUDA code is validated against
on small lattices. Two roles:

  * the "reference vs SciPy" leg of the validation ladder (this script IS
    that reference; it integrates with an adaptive RK45 at tight tolerance),
  * a generator of analytic-limit checks the GPU code must also pass.

Equations of motion (per site j, Fock amplitude f_m, hbar = 1):

    i d/dt f_m(j) = eps_m(j) f_m(j)
                    - conj(Phi_j) sqrt(m+1) f_{m+1}(j)
                    -      Phi_j  sqrt(m)   f_{m-1}(j)

    eps_m(j) = (U/2) m (m-1) + (V0 r_j^2 - mu0) m
    Phi_j    = sum_{k in nbr(j)} J_{jk} <b_k>
    <b_j>    = sum_m sqrt(m+1) conj(f_m(j)) f_{m+1}(j)

Conserved by the exact flow:
    * per-site norm   sum_m |f_m(j)|^2  = 1   (generator is Hermitian)
    * total number    N = sum_{j,m} m |f_m(j)|^2
    * total energy    E = <H> (paper's E_tot - mu0 N; mu0 term drops out)
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


# --------------------------------------------------------------------------
# Lattice: square LxL, open boundaries. Returns an (N,N) hopping matrix A
# with A[j,k] = J_dir for nearest neighbours, used as Phi = A @ psi.
# Swapping this one function for a triangular (z=6) or cubic builder is the
# ONLY change needed to move to those geometries -- same EOM, same code.
# --------------------------------------------------------------------------
def square_lattice(L, Jx=1.0, Jy=1.0):
    N = L * L
    A = np.zeros((N, N))
    def idx(x, y):
        return x + L * y
    for y in range(L):
        for x in range(L):
            j = idx(x, y)
            if x + 1 < L: A[j, idx(x + 1, y)] = Jx
            if x - 1 >= 0: A[j, idx(x - 1, y)] = Jx
            if y + 1 < L: A[j, idx(x, y + 1)] = Jy
            if y - 1 >= 0: A[j, idx(x, y - 1)] = Jy
    # r^2 from the trap centre, in lattice units
    cx = cy = (L - 1) / 2.0
    r2 = np.array([(x - cx) ** 2 + (y - cy) ** 2
                   for y in range(L) for x in range(L)], dtype=float)
    return A, r2


def triangular_lattice(L, J1=1.0, J2=1.0, J3=1.0):
    """Triangular lattice (z=6), open boundaries; matches tdgw make_triangular.
    Directions +-a1 (J1), +-a2 (J2), +-(a1+a2) (J3), with a1=(1,0),
    a2=(-1/2, sqrt3/2) (unit NN spacing, 120 deg). r^2 is the physical (oblique)
    distance from the trap centre."""
    N = L * L
    A = np.zeros((N, N))
    def idx(x, y): return x + L * y
    a1 = np.array([1.0, 0.0]); a2 = np.array([-0.5, np.sqrt(3.0) / 2.0])
    dirs = [((1, 0), J1), ((-1, 0), J1), ((0, 1), J2),
            ((0, -1), J2), ((1, 1), J3), ((-1, -1), J3)]
    r2 = np.zeros(N); c = (L - 1) / 2.0
    for y in range(L):
        for x in range(L):
            j = idx(x, y)
            Rj = (x - c) * a1 + (y - c) * a2
            r2[j] = float(Rj @ Rj)
            for (dx, dy), Jw in dirs:
                xx, yy = x + dx, y + dy
                if 0 <= xx < L and 0 <= yy < L:
                    A[j, idx(xx, yy)] = Jw
    return A, r2


# --------------------------------------------------------------------------
# Right-hand side and conserved quantities (vectorised over all sites).
# --------------------------------------------------------------------------
class TDGW:
    def __init__(self, A, r2, U, V0, mu0, D):
        self.A = A
        self.r2 = r2
        self.U = U
        self.V0 = V0
        self.mu0 = mu0
        self.N = A.shape[0]
        self.D = D                          # number of Fock states, m = 0..D-1
        m = np.arange(D)
        self.m = m
        self.cup = np.sqrt(np.arange(1, D))             # sqrt(m+1), len D-1
        self.eps = (0.5 * U * (m * (m - 1)))[None, :] \
                   + (V0 * r2 - mu0)[:, None] * m[None, :]   # (N,D)
        self.ekin = (0.5 * U * (m * (m - 1)))[None, :] \
                    + (V0 * r2)[:, None] * m[None, :]        # energy w/o mu0

    def psi(self, f):                                       # <b_j>, shape (N,)
        return np.sum(np.conj(f[:, :-1]) * f[:, 1:] * self.cup[None, :], axis=1)

    def dfdt(self, f):
        phi = self.A @ self.psi(f)                          # (N,)
        fup = np.zeros_like(f); fup[:, :-1] = f[:, 1:] * self.cup[None, :]   # sqrt(m+1) f_{m+1}
        fdn = np.zeros_like(f); fdn[:, 1:] = f[:, :-1] * self.cup[None, :]   # sqrt(m)   f_{m-1}
        return -1j * (self.eps * f
                      - np.conj(phi)[:, None] * fup
                      - phi[:, None] * fdn)

    # diagnostics ----------------------------------------------------------
    def site_norm(self, f):
        return np.sum(np.abs(f) ** 2, axis=1)
    def Ntot(self, f):
        return float(np.sum(self.m[None, :] * np.abs(f) ** 2))
    def Etot(self, f):
        psi = self.psi(f)
        local = np.sum(self.ekin * np.abs(f) ** 2)
        bond = -np.real(np.sum(np.conj(psi) * (self.A @ psi)))
        return float(local + bond)

    # packing for scipy (real solver) -------------------------------------
    def pack(self, f):
        return np.concatenate([f.real.ravel(), f.imag.ravel()])
    def unpack(self, y):
        h = self.N * self.D
        return (y[:h] + 1j * y[h:]).reshape(self.N, self.D)
    def rhs(self, t, y):
        return self.pack(self.dfdt(self.unpack(y)))

    def evolve(self, f0, T, n_eval=21, rtol=1e-10, atol=1e-12):
        t_eval = np.linspace(0.0, T, n_eval)
        sol = solve_ivp(self.rhs, [0.0, T], self.pack(f0), t_eval=t_eval,
                        method="RK45", rtol=rtol, atol=atol)
        fs = [self.unpack(sol.y[:, k]) for k in range(sol.y.shape[1])]
        return t_eval, fs


# --------------------------------------------------------------------------
# Helpers to build initial states.
# --------------------------------------------------------------------------
def coherent_state(alpha, D):
    m = np.arange(D)
    c = np.exp(-0.5 * abs(alpha) ** 2) * (alpha ** m) / np.sqrt(
        np.array([math.factorial(int(k)) for k in m], dtype=float))
    return c / np.linalg.norm(c)

def random_normalized_state(N, D, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((N, D)) + 1j * rng.standard_normal((N, D))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


# ==========================================================================
# TEST A -- conservation of per-site norm, N_tot, E_tot under full dynamics
# ==========================================================================
def test_conservation():
    print("=" * 70)
    print("TEST A: conservation under full nonlinear post-quench dynamics")
    print("=" * 70)
    L, D = 6, 7
    A, r2 = square_lattice(L, Jx=1.0, Jy=1.0)
    model = TDGW(A, r2, U=-0.5, V0=-2.5e-3, mu0=0.0, D=D)   # attractive U, anti-trap
    f0 = random_normalized_state(model.N, D, seed=1)
    T = 5.0
    t, fs = model.evolve(f0, T, n_eval=11)

    n0, e0 = model.Ntot(f0), model.Etot(f0)
    max_norm_dev = max(np.max(np.abs(model.site_norm(f) - 1.0)) for f in fs)
    max_N_dev = max(abs(model.Ntot(f) - n0) for f in fs)
    max_E_dev = max(abs(model.Etot(f) - e0) for f in fs)
    print(f"  lattice {L}x{L}, D={D}, U={model.U}, V0={model.V0}, T={T}")
    print(f"  N_tot(0) = {n0:.10f}     E_tot(0) = {e0:.10f}")
    print(f"  max |per-site norm - 1|   = {max_norm_dev:.3e}")
    print(f"  max |N_tot(t) - N_tot(0)| = {max_N_dev:.3e}")
    print(f"  max |E_tot(t) - E_tot(0)| = {max_E_dev:.3e}")
    ok = max_norm_dev < 1e-8 and max_N_dev < 1e-7 and max_E_dev < 1e-7
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


# ==========================================================================
# TEST B -- U=0 limit reproduces EXACT single-particle tight-binding flow.
# Coherent states stay coherent; <b_j>(t) must equal expm(-i H_sp t) psi0,
# H_sp = -A (the hopping matrix). This is a code-free analytic oracle.
# ==========================================================================
def test_free_limit():
    print("=" * 70)
    print("TEST B: U=0 reproduces exact free tight-binding evolution of <b>")
    print("=" * 70)
    L, D = 8, 9
    A, r2 = square_lattice(L, Jx=1.0, Jy=1.0)
    model = TDGW(A, r2, U=0.0, V0=0.0, mu0=0.0, D=D)

    # coherent bump in the centre, small alpha so D=9 truncation is negligible
    cx = cy = (L - 1) / 2.0
    alphas = np.array([0.8 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 4.0)
                       for y in range(L) for x in range(L)])
    f0 = np.array([coherent_state(a, D) for a in alphas])

    psi0 = model.psi(f0)
    Hsp = -A                                   # i d/dt psi = Hsp psi
    T = 1.5
    t, fs = model.evolve(f0, T, n_eval=16)

    max_err = 0.0
    for tk, f in zip(t, fs):
        psi_exact = expm(-1j * Hsp * tk) @ psi0
        psi_ga = model.psi(f)
        max_err = max(max_err, np.max(np.abs(psi_ga - psi_exact)))
    rel = max_err / np.max(np.abs(psi0))
    print(f"  lattice {L}x{L}, D={D}, alpha_max={alphas.max():.2f}, T={T}")
    print(f"  max |<b>_GA(t) - <b>_exact(t)|       = {max_err:.3e}")
    print(f"  relative to |psi0|_max               = {rel:.3e}")
    ok = rel < 1e-3
    print(f"  --> {'PASS' if ok else 'FAIL'}  (residual = Fock truncation + RK45 tol)")
    return ok


if __name__ == "__main__":
    a = test_conservation()
    print()
    b = test_free_limit()
    print()
    print("ALL PASS" if (a and b) else "SOME FAILED")