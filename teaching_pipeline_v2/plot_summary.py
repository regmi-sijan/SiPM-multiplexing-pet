#!/usr/bin/env python3
"""Redraw 'average FWHM vs readout channels' WITH error bars from an existing summary.csv, next to the paper's
Table 2 values, and write a comparison table.

    python plot_summary.py results/medium            # -> results/medium/fwhm_line.png + paper_comparison.csv
    python plot_summary.py results/medium --no-paper # without the paper reference
"""
import argparse, csv, os
from src import plots
from src.paper_reference import TABLE2_OPTIMAL


def read(folder):
    with open(os.path.join(folder, "summary.csv")) as f:
        rows = list(csv.DictReader(f))
    g = lambda k: [float(r[k]) for r in rows]
    return [int(r["channels"]) for r in rows], g("fwhm_x_mm"), g("fwhm_y_mm"), g("fwhm_x_std"), g("fwhm_y_std")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="run folder containing summary.csv")
    ap.add_argument("--no-paper", action="store_true")
    ap.add_argument("--out", help="output PNG (default <folder>/fwhm_line.png)")
    a = ap.parse_args()
    ch, fx, fy, sx, sy = read(a.folder)
    out = a.out or os.path.join(a.folder, "fwhm_line.png")
    plots.plot_fwhm_line(ch, fx, fy, out, std_x=sx, std_y=sy, paper_ref=not a.no_paper)
    print(f"Figure written to {out}")
    path = os.path.join(a.folder, "paper_comparison.csv")
    hdr = "channels,ours_fwhm_y,ours_sd_y,paper_fwhm_y,paper_sd_y,ours_fwhm_x,ours_sd_x,paper_fwhm_x,paper_sd_x,ratio_ours_over_paper_fwhm"
    lines = [hdr]
    print("\nchannels |   FWHM Y  (ours +- sd | paper +- sd)   |   FWHM X  (ours +- sd | paper +- sd)   | ours/paper")
    for c, x, y, ex, ey in sorted(zip(ch, fx, fy, sx, sy)):
        p = TABLE2_OPTIMAL.get(c)
        if p is None:
            continue
        ratio = (x + y) / (p[0] + p[2])
        lines.append(f"{c},{y:.3f},{ey:.3f},{p[0]:.2f},{p[1]:.2f},{x:.3f},{ex:.3f},{p[2]:.2f},{p[3]:.2f},{ratio:.2f}")
        print(f"{c:8d} | {y:5.2f} +- {ey:4.2f} | {p[0]:4.2f} +- {p[1]:4.2f}   | {x:5.2f} +- {ex:4.2f} | {p[2]:4.2f} +- {p[3]:4.2f}   | {ratio:5.2f}x")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Table written to {path}")


if __name__ == "__main__":
    main()
