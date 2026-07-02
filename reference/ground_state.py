#!/usr/bin/env python3
"""
Compressed-Mott Gutzwiller ground state on a trapped square lattice.

Self-consistent GA: at each site solve the local (DxD) mean-field Hamiltonian
    h_j[m,m]   = (U/2) m(m-1) + (V0 r_j^2 - mu0) m
    h_j[m,m+1] = -conj(Phi_j) sqrt(m+1),   Phi_j = sum_nbr J <b_nbr>
take its ground eigenvector as f_j, recompute <b_j> and Phi, iterate (mixing the
order parameter) to self-consistency. At deep-lattice params (J/U ~ 0.0023,
mu0/U ~ 0.15) this gives an n=1 Mott core with a thin superfluid shell at the
trap edge -- the t0 = 20 ms initial state of Rapp PRA 87, 043611 (2013).

Key validation: the converged state is a *stationary* solution of the TDGW EOM.
Each site's whole Fock vector rotates by a single phase e^{-i E_j t}, so <b_j>
(hence Phi, n_j, everything observable) is time-independent. We confirm n_j and
N_tot do not drift under real-time evolution.
"""
import argparse
import numpy as np
from tdgw_reference import square_lattice, triangular_lattice, TDGW
from fieldio import write_field


def gs_solve(A, r2, U, J_unused, mu0, V0, D, iters=600, mix=0.3, tol=1e-10, seed=0.05):
    N = A.shape[0]
    m = np.arange(D)
    eps = (0.5 * U * (m * (m - 1)))[None, :] + (V0 * r2 - mu0)[:, None] * m[None, :]  # (N,D)
    cup = np.sqrt(np.arange(1, D))                                  # sqrt(m+1)

    # LDA initial guess: Fock |n_j> at the atomic-limit minimum, + small seed.
    # TODO: a standalone lda_state() (homogeneous n(mu_eff) builder) could provide this
    # guess directly; LDA nails the density but misses the proximity SF shell, so the
    # self-consistency sweeps below are still needed for the (dynamically important) shell.
    f = np.zeros((N, D))
    n0 = np.argmin(eps, axis=1)
    f[np.arange(N), n0] = 1.0
    for j in range(N):
        if n0[j] + 1 < D: f[j, n0[j] + 1] += seed
        if n0[j] - 1 >= 0: f[j, n0[j] - 1] += seed
    f /= np.linalg.norm(f, axis=1, keepdims=True)

    def order_param(f):
        return np.sum(f[:, :-1] * cup[None, :] * f[:, 1:], axis=1)  # real f -> real psi

    psi = order_param(f)
    for it in range(iters):
        phi = A @ psi                                              # real
        fnew = np.empty_like(f)
        for j in range(N):
            h = np.diag(eps[j]).astype(float)
            for mm in range(D - 1):
                off = -phi[j] * cup[mm]                            # real phi -> real h
                h[mm, mm + 1] = off; h[mm + 1, mm] = off
            w, v = np.linalg.eigh(h)
            gv = v[:, 0]
            if gv[np.argmax(np.abs(gv))] < 0: gv = -gv             # fix sign
            fnew[j] = gv
        psi_new = order_param(fnew)
        dpsi = np.max(np.abs(psi_new - psi))
        psi = (1 - mix) * psi + mix * psi_new
        f = fnew
        if dpsi < tol:
            print(f"  converged in {it+1} sweeps (max|dpsi|={dpsi:.2e})")
            break
    else:
        print(f"  reached {iters} sweeps (max|dpsi|={dpsi:.2e})")
    return f.astype(complex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compressed-Mott GA ground state")
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--D", type=int, default=7)
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--J", type=float, default=0.0023)
    ap.add_argument("--mu0", type=float, default=0.15)
    ap.add_argument("--V0", type=float, default=0.0023, help="trap (compression); >0 confines")
    ap.add_argument("--seed", type=float, default=0.05)
    ap.add_argument("--lattice", choices=["square", "triangular"], default="square")
    ap.add_argument("--dump", help="write the initial state to a field file (for tdgw --load)")
    ap.add_argument("--no-check", action="store_true", help="skip the stationarity check")
    args = ap.parse_args()
    L, D, U, J, mu0, V0 = args.L, args.D, args.U, args.J, args.mu0, args.V0
    # Deep lattice: isotropic hopping J on every bond (Mott core is geometry-agnostic).
    A, r2 = (triangular_lattice(L, J, J, J) if args.lattice == "triangular"
             else square_lattice(L, J, J))
    print(f"deep-lattice ground state [{args.lattice}]: {L}x{L}, D={D}, "
          f"J/U={J/U:.4f}, mu0/U={mu0:.3f}, V0={V0}")
    f = gs_solve(A, r2, U, J, mu0, V0, D, seed=args.seed)

    n = np.sum(np.arange(D)[None, :] * np.abs(f) ** 2, axis=1)
    psi = np.sum(np.conj(f[:, :-1]) * np.sqrt(np.arange(1, D))[None, :] * f[:, 1:], axis=1)
    Ntot = float(np.sum(n)); N0 = float(np.sum(np.abs(psi) ** 2))
    Rcloud = float(np.sqrt(np.sum(r2 * n) / Ntot)); nc = n.reshape(L, L)[L // 2, L // 2]
    print(f"  central n={nc:.4f}  N_tot={Ntot:.1f}  N0={N0:.3f}  N0/N={N0/Ntot:.2e}  R_rms={Rcloud:.2f}")
    print(f"  n=1 sites={int(np.sum(np.round(n)==1))}  shell |psi|>1e-3 sites={int(np.sum(np.abs(psi)>1e-3))}")

    if args.dump:
        write_field(args.dump, f.astype(complex), L, D)
        print(f"  dumped initial state -> {args.dump}")

    if not args.no_check and L <= 32:
        print("  stationarity check (real-time TDGW, same params, T=2):")
        model = TDGW(A, r2, U, V0, mu0, D)
  