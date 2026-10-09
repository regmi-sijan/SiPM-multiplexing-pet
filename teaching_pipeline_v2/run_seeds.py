#!/usr/bin/env python3
"""Repeat run_all.py with several random seeds and report mean +/- spread, or compare training variants.

Examples
  # baseline, 3 seeds, 40 epochs  -> results/q40/seed_1 ... seed_3 + mean table
  python run_seeds.py --seeds 1 2 3 --out results/q40 -- --preset quick --epochs 40

  # compare variants (each variant = its own folder, same seeds; simulated data is cached per seed, so cheap)
  python run_seeds.py --variants base cosine soft best all --seeds 1 2 3 --out results/cmp -- --preset quick --epochs 40

  # only (re)build the table/plot from folders that already exist
  python run_seeds.py --compare results/cmp/base results/cmp/cosine

Everything after a lone `--` is passed to run_all.py unchanged (preset, epochs, --device, ...).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import glob
import json
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {   # name -> extra run_all.py arguments
    "base": [],
    "cosine": ["--lr-schedule", "cosine"],
    "soft": ["--soft-label-sigma", "1.0"],
    "best": ["--select-best", "offgrid"],
    "all": ["--lr-schedule", "cosine", "--soft-label-sigma", "1.0", "--select-best", "offgrid"],
}
METRICS = [("rms_mm", "RMS error (mm)"), ("fwhm", "FWHM (mm, mean of x and y)"),
           ("bias_abs_mean", "mean |bias| (mm)")]


def load_run(folder):
    """{channels: {metric: value}} for one run folder (reads summary.json)."""
    with open(os.path.join(folder, "summary.json")) as f:
        raw = json.load(f)
    out = {}
    for ch, s in raw.items():
        out[int(ch)] = dict(rms_mm=s["rms_mm"], fwhm=0.5 * (s["fwhm_x"] + s["fwhm_y"]),
                            bias_abs_mean=s["bias_abs_mean"], bias_abs_max=s["bias_abs_max"])
    return out


def aggregate(folders):
    """Combine several seed folders -> {channels: {metric: (mean, std, n)}}."""
    runs = [load_run(f) for f in folders]
    chans = sorted(set.intersection(*[set(r) for r in runs]), reverse=True)
    agg = {}
    for c in chans:
        agg[c] = {}
        for m in runs[0][c]:
            v = np.array([r[c][m] for r in runs])
            agg[c][m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))
    return agg


def seed_folders(root):
    return sorted(glob.glob(os.path.join(root, "seed_*")))


def write_table(tables, path_csv):
    """tables: {label: agg}. Prints a table and writes a CSV."""
    labels = list(tables)
    chans = sorted(set.intersection(*[set(t) for t in tables.values()]), reverse=True)
    lines = ["variant,channels,n_seeds,rms_mean,rms_std,fwhm_mean,fwhm_std,bias_mean,bias_std"]
    print(f"\n{'variant':14s} {'ch':>3s} {'n':>2s} | {'RMS error':>16s} | {'FWHM':>16s} | {'mean |bias|':>16s}   (mm, mean +/- std over seeds)")
    for lab in labels:
        for c in chans:
            a = tables[lab][c]
            f = lambda m: f"{a[m][0]:6.2f} +/- {a[m][1]:4.2f}"
            print(f"{lab:14s} {c:3d} {a['rms_mm'][2]:2d} | {f('rms_mm')} | {f('fwhm')} | {f('bias_abs_mean')}")
            lines.append(f"{lab},{c},{a['rms_mm'][2]},{a['rms_mm'][0]:.4f},{a['rms_mm'][1]:.4f},"
                         f"{a['fwhm'][0]:.4f},{a['fwhm'][1]:.4f},{a['bias_abs_mean'][0]:.4f},{a['bias_abs_mean'][1]:.4f}")
    with open(path_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTable written to {path_csv}")


def plot(tables, path_png, per_run=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#1C7293", "#F2A541", "#C8553D", "#5FA8A0", "#7A8B99", "#0B2A3C"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 3.8))
    for ax, (m, title) in zip(axes, METRICS):
        for i, (lab, agg) in enumerate(tables.items()):
            ch = sorted(agg)
            mu = [agg[c][m][0] for c in ch]
            sd = [agg[c][m][1] for c in ch]
            off = (i - (len(tables) - 1) / 2) * 0.3
            ax.errorbar(np.array(ch) + off, mu, yerr=sd, fmt="o-", ms=4, capsize=3, lw=1.2,
                        color=colors[i % len(colors)], label=lab)
        ax.set_xlabel("Readout channels"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean +/- std over seeds")
    fig.tight_layout(); fig.savefig(path_png, dpi=150); plt.close(fig)
    print(f"Plot written to {path_png}")


def run_one(extra, seed, out, passthrough):
    cmd = [sys.executable, os.path.join(HERE, "run_all.py")] + passthrough + extra + ["--seed", str(seed), "--out", out]
    print("\n$ " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"run_all.py failed (exit code {r.returncode}); see the output above. "
                         f"Run `python check_env.py` if it looks like an environment problem.")


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--"); passthrough, argv = argv[i + 1:], argv[:i]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="results/seeds", help="output root")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), help="compare these training variants")
    ap.add_argument("--compare", nargs="+", metavar="DIR", help="only aggregate these existing root folders")
    args = ap.parse_args(argv)

    if args.compare:
        tables = {}
        for root in args.compare:
            fs = seed_folders(root) or [root]
            tables[os.path.basename(os.path.normpath(root))] = aggregate(fs)
        out = os.path.join(os.path.dirname(os.path.normpath(args.compare[0])) or ".", "comparison")
        write_table(tables, out + ".csv"); plot(tables, out + ".png")
        return

    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
    tables = {}
    for name in (args.variants or ["base"]):
        root = os.path.join(args.out, name) if args.variants else args.out
        for s in args.seeds:
            folder = os.path.join(root, f"seed_{s}")
            if os.path.exists(os.path.join(HERE, folder, "summary.json")):
                print(f"\n(skipping {folder}: already done)")
                continue
            run_one(VARIANTS[name] if args.variants else [], s, folder, passthrough)
        tables[name] = aggregate([os.path.join(HERE, f) for f in seed_folders(os.path.join(HERE, root))])
    out = os.path.join(HERE, args.out, "seeds_summary")
    write_table(tables, out + ".csv"); plot(tables, out + ".png")


if __name__ == "__main__":
    main()
