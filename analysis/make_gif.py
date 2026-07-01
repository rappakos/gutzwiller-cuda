#!/usr/bin/env python3
"""Assemble a GIF of TOF frames from tdgw `--dump-every` field files.

    tdgw --load init.bin --U -2.2 --V0 -8e-4 --steps 20000 --dt 0.002 \
         --dump-every 500 --dump-prefix results/frame
    python analysis/make_gif.py "results/frame_*.bin" --out results/tof.gif --fps 10

Frames share one global intensity scale so the condensate is seen to *grow*
(per-frame normalization would hide that). Title shows the condensate fraction.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tof
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description="GIF of TOF frames from tdgw --dump-every")
    ap.add_argument("pattern", help='glob for frames, e.g. "results/frame_*.bin"')
    ap.add_argument("--out", default="results/tof.gif")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--lattice-depth", type=float, default=6.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--pad", type=int, default=2)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"no files match {args.pattern}")

    Is, meta = [], []
    for f in files:
        psi, Ntot, N0 = tof.load_field(f)
        kx, ky, I = tof.tof_intensity(psi, Ntot, N0, args.lattice_depth,
                                      args.reps, args.pad, normalize=False)
        V = tof.visibility(psi, Ntot, N0, args.lattice_depth, args.reps, args.pad)
        Is.append(I); meta.append((kx, ky, N0 / Ntot, V, os.path.basename(f)))
    gmax = max(float(I.max()) for I in Is) or 1.0

    frames = []
    for I, (kx, ky, frac, V, name) in zip(Is, meta):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.imshow(I / gmax, origin="lower", extent=[kx[0], kx[-1], ky[0], ky[-1]],
                  cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        for sx in (-1, 1):
            for sy in (-1, 1):
                ax.plot(sx, sy, "o", mfc="none", mec="cyan", ms=10, mew=1.0)
        ax.set_xlabel(r"$k_x/k_L$"); ax.set_ylabel(r"$k_y/k_L$")
        ax.set_title(f"{name}   N0/N={frac:.2f}  V={V:+.2f}")
        fig.tight_layout(); fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
        plt.close(fig); frames.append(Image.fromarray(buf))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out}  ({len(frames)} frames, global vmax={gmax:.3g})")


if __name__ == "__main__":
    main()
