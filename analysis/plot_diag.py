#!/usr/bin/env python3
"""Plot tdgw diagnostics vs step from a captured run log.

    tdgw --load init.bin --U -2.2 --V0 -8e-4 --steps 20000 --dt 0.002 > run.log
    python analysis/plot_diag.py run.log --out results/diag.png

Parses lines like:
    step  19800   N=1963.42  N0/N=0.6258  K=+3454.789  R=38.0  |norm-1|=3.6e-07  dN=4.3e-01
"""
import argparse, re, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"step\s+(\d+)\s+N=([\d.eE+-]+)\s+N0/N=([\d.eE+-]+)"
                 r"\s+K=([+\-][\d.eE+-]+)\s+R=([\d.eE+-]+)")

def main():
    ap = argparse.ArgumentParser(description="Plot tdgw diagnostics (N0/N, K, R) vs step")
    ap.add_argument("log", help="captured tdgw stdout ('-' for stdin)")
    ap.add_argument("--out", default="results/diag.png")
    args = ap.parse_args()
    text = sys.stdin.read() if args.log == "-" else open(args.log).read()
    rows = [(int(m[1]), float(m[3]), float(m[4]), float(m[5])) for m in
            (PAT.search(l) for l in text.splitlines()) if m]
    if not rows:
        sys.exit("no diagnostic lines matched")
    step, n0, K, R = np.array(rows).T
    fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    ax[0].plot(step, n0);        ax[0].set_ylabel("N0 / N")
    ax[1].plot(step, K, "C1");   ax[1].axhline(0, color="k", lw=0.6, ls=":"); ax[1].set_ylabel("K (bond)")
    ax[2].plot(step, R, "C2");   ax[2].set_ylabel("R (sites)"); ax[2].set_xlabel("step")
    ax[0].set_title("TDGW diagnostics: condensate, kinetic (K>0 = inverse pop.), cloud radius")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=130); print("wrote", args.out)

if __name__ == "__main__":
    main()
