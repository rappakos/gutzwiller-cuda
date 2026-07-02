#!/usr/bin/env python3
"""
Time-of-flight (TOF) image generator for the TDGW square-lattice solver.

The only transform needed is a 2D FFT of the order-parameter field <b_j>, which
is L x L (tiny) -- so plain numpy.fft, no cuFFT. Following Rapp PRA 87, 043611
(2013), Eq. (11)-(12):

    I_TOF(k)  ~  |w(k)|^2 * G(k),     G(k) = N_tot - N_0 + |b(k)|^2
    b(k)      =  sum_j exp(-i k.r_j) <b_j>            (2D DFT of the field)
    |w(k)|^2  ~  exp(-k^2 / (sqrt(s) k_L^2))          (Wannier envelope, depth s)

Momentum is reported in units of k_L. For a square optical lattice the spacing is
a = lambda_L/2 so k_L = pi/a: the first Brillouin zone is k in [-k_L, k_L) and its
corners sit at Q = (+-k_L, +-k_L). The negative-T signature is a checkerboard field
<b_j> ~ (-1)^(x+y) <b_0>, whose transform piles weight onto exactly those corners.

Usage
-----
    python tof.py --demo --out tof_demo.png            # synthetic protocol-(a) field
    python tof.py --in psi_dump.bin --out tof.png      # field dumped by tdgw --dump

Field format consumed by --in is the shared fieldio format (see reference/fieldio.py):
    int32 L, int32 D, then N*D complex64 as f[m*N + j]. Written by tdgw --dump and
    by ground_state.py --dump. tof reduces f to psi = <b_j> and forms I_TOF.
"""
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
from fieldio import read_field


# --------------------------------------------------------------------------
def load_field(path):
    """Read a field file (fieldio format) and reduce to (psi[y,x], N_tot, N0)."""
    f, L, D = read_field(path)                       # f: (N, D), index f[j, m]
    cup = np.sqrt(np.arange(1, D))
    psi = np.sum(np.conj(f[:, :-1]) * cup[None, :] * f[:, 1:], axis=1)   # <b_j>, (N,)
    n = np.sum(np.arange(D)[None, :] * np.abs(f) ** 2, axis=1)
    Ntot = float(np.sum(n)); N0 = float(np.sum(np.abs(psi) ** 2))
    return psi.reshape(L, L), Ntot, N0               # psi[y, x] (j = x + L*y)


def synth_field(L=128, sigma_frac=0.18, n_center=1.0, condensate_frac=0.7):
    """Synthetic protocol-(a) negative-T field: a trap-localized cloud with a
    checkerboard (pi,pi) phase -- the thing whose TOF shows four corner peaks."""
    c = (L - 1) / 2.0
    y, x = np.mgrid[0:L, 0:L]
    r2 = (x - c) ** 2 + (y - c) ** 2
    amp = np.sqrt(n_center) * np.exp(-r2 / (2 * (sigma_frac * L) ** 2))
    psi = amp * (-1.0) ** (x + y)                        # checkerboard phase
    N0 = float(np.sum(np.abs(psi) ** 2))                 # condensate occupation
    Ntot = N0 / condensate_frac                          # add incoherent background
    return psi.astype(np.complex64), Ntot, N0


# --------------------------------------------------------------------------
def tof_intensity(psi, Ntot, N0, lattice_depth=6.0, reps=3, pad=1, normalize=True):
    """Return (kx, ky, I) with momenta in units of k_L.

    The lattice DFT is periodic over the reciprocal lattice, so we tile the
    first-BZ power spectrum across `reps` (odd) BZs in each direction, centred
    on k=0, to show the corner peaks Q=(+-1,+-1) and their envelope-damped
    replicas. The momentum axis is tiled in lockstep with the data so the
    Nyquist (checkerboard) component stays pinned to k=+-1.
    """
    L = psi.shape[0]
    if pad > 1:                                          # zero-pad the (localized) cloud ->
        Lp = pad * L; o = (Lp - L) // 2                  # finer k-grid = true finite-cloud peak shape
        pp = np.zeros((Lp, Lp), dtype=complex); pp[o:o+L, o:o+L] = psi
        psi = pp; L = Lp
    b = np.fft.fftshift(np.fft.fft2(psi))                # b(k), first BZ
    G = (Ntot - N0) + np.abs(b) ** 2                     # incoherent bg + |b|^2

    reps = reps + 1 if reps % 2 == 0 else reps           # force odd -> symmetric
    # fftshifted first-BZ axis in units of k_L: kappa = 2 * fftshift(fftfreq(L))
    k1 = np.fft.fftshift(np.fft.fftfreq(L)) * 2.0        # spans [-1, 1)
    offs = 2 * (np.arange(reps) - reps // 2)             # ..., -2, 0, 2, ...
    k = np.concatenate([k1 + o for o in offs])           # data & axis tiled alike
    G_tiled = np.tile(G, (reps, reps))

    KX, KY = np.meshgrid(k, k)
    W = np.exp(-(KX ** 2 + KY ** 2) / np.sqrt(lattice_depth))   # |w(k)|^2 envelope
    I = W * G_tiled
    if normalize:
        I /= I.max()
    return k, k, I


def visibility(psi, Ntot, N0, lattice_depth=6.0, reps=3, pad=2, win=0.18):
    """TOF visibility V=(I_A-I_B)/(I_A+I_B), Rapp PRA 87 Eq. below (12):
    I_A integrated around the BZ corner Q=(k_L,k_L), I_B around (sqrt2 k_L, 0).
    V>0 => corner (negative-T, (pi,pi)) weight dominates."""
    kx, ky, I = tof_intensity(psi, Ntot, N0, lattice_depth, reps, pad, normalize=False)
    KX, KY = np.meshgrid(kx, ky)
    def wsum(cx, cy):
        return float(I[((KX - cx) ** 2 + (KY - cy) ** 2) <= win * win].sum())
    IA = wsum(1.0, 1.0); IB = wsum(np.sqrt(2.0), 0.0)
    return (IA - IB) / (IA + IB) if (IA + IB) > 0 else 0.0


# ==========================================================================
# Triangular lattice: hexagonal reciprocal lattice.
# --------------------------------------------------------------------------
# A plain fft2 of psi[y,x] samples b(k) on the OBLIQUE reciprocal grid
#   k = (m_x/L) B1 + (m_y/L) B2 ,   a_i . B_j = 2*pi*delta_ij ,
# for lattice vectors a1=(1,0), a2=(-1/2, sqrt3/2) (matches make_triangular).
# We map those samples to physical (kx,ky), tile over neighbouring BZs, apply
# the Wannier envelope, and interpolate onto a regular grid for display. The
# condensate peaks sit at the single-particle band MAXIMA (K-points), whose
# location MOVES with the hopping anisotropy J1:J2:J3 -- the frustration signal
# of Rapp, PRA 90, 053607 (2014). Momenta are reported in units of k_L = pi/a.
# ==========================================================================
_A1 = np.array([1.0, 0.0]); _A2 = np.array([-0.5, np.sqrt(3.0) / 2.0]); _A3 = _A1 + _A2
_B1 = 2 * np.pi * np.array([1.0, 1.0 / np.sqrt(3.0)])
_B2 = 2 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])


def band_max_k(J1, J2, J3, ngrid=361, span=1.7, tol=3e-3, min_sep=0.7):
    """Distinct momenta of the band maxima eps_k = -2 sum_j J_j cos(k.a_j)
    within (a little beyond) the first Brillouin zone -- the frustrated
    K-points -- returned in units of k_L = pi."""
    g = np.linspace(-span, span, ngrid) * np.pi; KX, KY = np.meshgrid(g, g)
    E = -2 * (J1 * np.cos(KX * _A1[0] + KY * _A1[1])
              + J2 * np.cos(KX * _A2[0] + KY * _A2[1])
              + J3 * np.cos(KX * _A3[0] + KY * _A3[1]))
    Emax = E.max(); m = E > Emax - tol * abs(Emax)
    cand = np.column_stack([KX[m], KY[m]]) / np.pi
    peaks = []
    for p in cand:
        if all(np.hypot(*(p - q)) > min_sep for q in peaks):
            peaks.append(p)
    return np.array(peaks), Emax


def _tri_scatter(psi, reps):
    """b(k) samples in physical k (units of k_L=pi): returns (kx,ky,b) flat."""
    L = psi.shape[0]
    b = np.fft.fftshift(np.fft.fft2(psi))
    fx = np.fft.fftshift(np.fft.fftfreq(L)); FX, FY = np.meshgrid(fx, fx)  # FX~m_x, FY~m_y
    kxs, kys, bs = [], [], []
    rr = range(-(reps // 2), reps // 2 + 1)
    for n1 in rr:
        for n2 in rr:
            kx = ((FX + n1) * _B1[0] + (FY + n2) * _B2[0]) / np.pi   # k_L units
            ky = ((FX + n1) * _B1[1] + (FY + n2) * _B2[1]) / np.pi
            kxs.append(kx.ravel()); kys.append(ky.ravel()); bs.append(b.ravel())
    return np.concatenate(kxs), np.concatenate(kys), np.concatenate(bs)


def tof_intensity_tri(psi, Ntot, N0, lattice_depth=6.0, reps=3, pad=2,
                      ngrid=420, normalize=True):
    """(kx,ky,I) for a triangular field on a regular physical k-grid (k_L units)."""
    from scipy.interpolate import griddata
    L = psi.shape[0]
    if pad > 1:
        Lp = pad * L; o = (Lp - L) // 2
        pp = np.zeros((Lp, Lp), dtype=complex); pp[o:o+L, o:o+L] = psi
        psi = pp
    reps = reps + 1 if reps % 2 == 0 else reps
    kx, ky, b = _tri_scatter(psi, reps)
    G = (Ntot - N0) + np.abs(b) ** 2
    gx = np.linspace(-3.0, 3.0, ngrid); KXg, KYg = np.meshgrid(gx, gx)
    Gg = griddata(np.column_stack([kx, ky]), G, (KXg, KYg),
                  method="linear", fill_value=0.0)
    W = np.exp(-(KXg ** 2 + KYg ** 2) / np.sqrt(lattice_depth))   # k_L units, as square
    I = W * Gg
    if normalize and I.max() > 0: I /= I.max()
    return gx, gx, I


def kpoint_coherence(psi, J1, J2, J3, reps=3, win=0.35):
    """Fraction of coherent weight |b(k)|^2 lying within `win` (k_L) of a band-max
    K-point. Large => sharp condensate at the frustrated K-point (isotropic/rhombic);
    small => weight smeared / off the K-points (frustrated, no coherence). At L=24
    an incoherent Mott gives ~0.01, the negative-T condensate ~0.2 (peaks sharpen
    with L, so the contrast grows on the GPU-scale lattices)."""
    peaks, _ = band_max_k(J1, J2, J3)
    reps = reps + 1 if reps % 2 == 0 else reps
    kx, ky, b = _tri_scatter(psi, reps)
    w = np.abs(b) ** 2
    near = np.zeros(kx.shape, bool)
    for p in peaks:
        near |= (kx - p[0]) ** 2 + (ky - p[1]) ** 2 <= win * win
    tot = w.sum()
    return float(w[near].sum() / tot) if tot > 0 else 0.0


# --------------------------------------------------------------------------
def plot_tof_tri(kx, ky, I, out, peaks=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(I, origin="lower", extent=[kx[0], kx[-1], ky[0], ky[-1]],
                   cmap="inferno", interpolation="bilinear")
    if peaks is not None:
        for p in peaks:
            ax.plot(p[0], p[1], "o", mfc="none", mec="cyan", ms=12, mew=1.2)
    ax.set_xlabel(r"$k_x / k_L$"); ax.set_ylabel(r"$k_y / k_L$")
    ax.set_aspect("equal"); ax.set_title(title or "TOF intensity  $I(k)$  (triangular)")
    fig.colorbar(im, ax=ax, label="normalized intensity")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


# --------------------------------------------------------------------------
def plot_tof(kx, ky, I, out, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(I, origin="lower", extent=[kx[0], kx[-1], ky[0], ky[-1]],
                   cmap="inferno", interpolation="bilinear")
    # mark the four BZ corners Q = (+-1, +-1) in units of k_L
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot(sx, sy, "o", mfc="none", mec="cyan", ms=12, mew=1.2)
    ax.set_xlabel(r"$k_x / k_L$"); ax.set_ylabel(r"$k_y / k_L$")
    ax.set_title(title or "TOF intensity  $I(k)$")
    fig.colorbar(im, ax=ax, label="normalized intensity")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TDGW time-of-flight image generator")
    ap.add_argument("--in", dest="infile", help="field dump from tdgw --dump")
    ap.add_argument("--demo", action="store_true",
                    help="use a synthetic protocol-(a) checkerboard field")
    ap.add_argument("--out", default="results/tof.png")
    ap.add_argument("--lattice-depth", type=float, default=6.0,
                    help="final lattice depth s (sets Wannier envelope width)")
    ap.add_argument("--reps", type=int, default=2, help="BZ tiling for display")
    ap.add_argument("--pad", type=int, default=2, help="zero-pad factor (smooth finite-cloud peaks)")
    ap.add_argument("--lattice", choices=["square", "triangular"], default="square")
    ap.add_argument("--J1", type=float, default=1.0)
    ap.add_argument("--J2", type=float, default=1.0)
    ap.add_argument("--J3", type=float, default=1.0)
    args = ap.parse_args()

    if args.demo or not args.infile:
        psi, Ntot, N0 = synth_field()
        title = "TOF (synthetic protocol-a field)"
    else:
        psi, Ntot, N0 = load_field(args.infile)
        title = f"TOF  (N0/Ntot = {N0/Ntot:.2f})"

    if args.lattice == "triangular":
        kx, ky, I = tof_intensity_tri(psi, Ntot, N0, args.lattice_depth, args.reps, args.pad)
        peaks, _ = band_max_k(args.J1, args.J2, args.J3)
        C = kpoint_coherence(psi, args.J1, args.J2, args.J3)
        print(f"L={psi.shape[0]}  N_tot={Ntot:.1f}  N0={N0:.1f}  "
              f"condensate fraction={N0/Ntot:.2f}  K-point coherence={C:.3f}  "
              f"({len(peaks)} band-max K-points)")
        title += f"   [tri J={args.J1:g}:{args.J2:g}:{args.J3:g}]  C_K={C:.2f}"
        plot_tof_tri(kx, ky, I, args.out, peaks, title)
    else:
        kx, ky, I = tof_intensity(psi, Ntot, N0, args.lattice_depth, args.reps, args.pad)
        V = visibility(psi, Ntot, N0, args.lattice_depth, args.reps, args.pad)
        print(f"L={psi.shape[0]}  N_tot={Ntot:.1f}  N0={N0:.1f}  "
              f"condensate fraction={N0/Ntot:.2f}  visibility V={V:+.3f}")
        title += f"   V={V:+.2f}"
        plot_tof(kx, ky, I, args.out, title)
