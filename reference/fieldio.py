#!/usr/bin/env python3
"""Shared binary field format for the TDGW pipeline (Python <-> CUDA).

Layout (little-endian):
    int32   L              # lattice linear size; N = L*L
    int32   D              # number of Fock states
    complex64[N*D]         # f, ordered f[m*N + j] (component-planar, matches the
                           # CUDA SoA layout); each value is (re, im) float32.

ground_state.py --dump writes it; tdgw --load/--dump read/write it; tof.py reads it.
"""
import numpy as np


def write_field(path, f, L, D):
    """f: complex array shape (N, D) with N = L*L, index f[j, m]."""
    N = L * L
    assert f.shape == (N, D), f"expected ({N},{D}), got {f.shape}"
    with open(path, "wb") as fh:
        np.array([L, D], dtype="<i4").tofile(fh)
        # (N,D) -> (D,N) -> ravel C-order gives index m*N + j
        f.T.astype("<c8").ravel(order="C").tofile(fh)


def read_field(path):
    """Returns (f, L, D) with f complex shape (N, D), index f[j, m]."""
    with open(path, "rb") as fh:
        hdr = np.fromfile(fh, dtype="<i4", count=2)
        L, D = int(hdr[0]), int(hdr[1])
        body = np.fromfile(fh, dtype="<c8", count=L * L * D)
    f = body.reshape(D, L * L).T          # -> (N, D)
    return np.ascontiguousarray(f), L, D
