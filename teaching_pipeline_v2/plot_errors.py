#!/usr/bin/env python3
"""Diagnose HOW the network makes errors (needs run_all.py --save-predictions).

    python run_all.py --preset quick --epochs 40 --save-predictions --out results/diag
    python plot_errors.py results/diag --channels 32 16 4

Per channel count it draws
  left : predicted x vs true x for hits on the central row of the test grid. If the network "snaps" to the
         calibration grid, the points form plateaus at the grid positions with jumps in between (a staircase);
         a well-interpolating network follows the diagonal.
  right: histograms of the x error at a central and an edge test position. Two separate peaks = flipping between
         neighbouring grid points; a single peak = smooth interpolation.
and prints a 'snapping index': the share of predictions lying within 0.15 mm of a calibration position
with 2 mm pitch evenly spread predictions give about 0.15 (= 2*0.15/2); clearly higher values mean snapping.
The index is only a hint: judge it together with the left-hand plot and the histograms.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np


def load(folder, ch):
    d = np.load(os.path.join(folder, f"pred_{ch:02d}ch.npz"))
    return d["pred"], d["true"], d["pos_id"], d["test_pos"]


def snapping_index(pred, pitch, tol=0.15, half=20.0):
    """Share of x and y predictions within `tol` mm of a calibration coordinate (cell centres at pitch/2 + k*pitch)."""
    centres = (np.arange(int(round(2 * half / pitch))) + 0.5) * pitch - half
    out = []
    for ax in (0, 1):
        d = np.abs(pred[:, ax][:, None] - centres[None, :]).min(1)
        out.append((d < tol).mean())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 16, 4])
    ap.add_argument("--pitch", type=float, default=2.0, help="training-grid pitch in mm (quick/medium: 2, paper: 1)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(args.channels)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.6 * n), squeeze=False)
    for row, ch in enumerate(args.channels):
        pred, true, pid, tpos = load(args.folder, ch)
        # central row of the test grid: y closest to 0
        ys = np.unique(tpos[:, 1]); y0 = ys[np.argmin(np.abs(ys))]
        sel = np.abs(true[:, 1] - y0) < 1e-3
        a = axes[row, 0]
        a.plot([-20, 20], [-20, 20], color="#7A8B99", lw=1)
        a.scatter(true[sel, 0] + np.random.default_rng(0).uniform(-0.05, 0.05, sel.sum()), pred[sel, 0],
                  s=3, alpha=0.35, color="#1C7293")
        a.set_xlim(-6, 6); a.set_ylim(-6, 6)
        a.set_xlabel("true x [mm]"); a.set_ylabel("predicted x [mm]")
        snap = snapping_index(pred, args.pitch)
        a.set_title(f"{ch} channels, central row (zoom)   snapping index {snap:.2f}", fontsize=9)
        b = axes[row, 1]
        for label, target, color in [("centre", (0.0, 0.0), "#1C7293"), ("edge", (0.0, -18.0), "#F2A541")]:
            d = np.linalg.norm(tpos - np.array(target), axis=1); k = int(np.argmin(d))
            e = pred[pid == k, 0] - true[pid == k, 0]
            b.hist(e, bins=np.linspace(-4, 4, 65), alpha=0.6, color=color,
                   label=f"{label} ({tpos[k, 0]:+.1f}, {tpos[k, 1]:+.1f})")
        b.set_xlabel("x error [mm]"); b.set_ylabel("events"); b.legend(fontsize=8)
        print(f"{ch:3d} channels: snapping index {snap:.2f}")
    fig.tight_layout()
    out = os.path.join(args.folder, "error_diagnostics.png")
    fig.savefig(out, dpi=150); print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
