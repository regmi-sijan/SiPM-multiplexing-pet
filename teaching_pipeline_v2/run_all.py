#!/usr/bin/env python3
"""End-to-end pipeline: simulate -> multiplex -> train CNN -> evaluate -> plot.

    python run_all.py --preset quick
    python run_all.py --preset medium --channels 32 16 8 4
    python run_all.py --preset quick --device cpu          # force the CPU
    python run_all.py --preset medium --device mps         # Apple GPU (default is --device auto)
"""
import os
# macOS workaround: PyTorch and NumPy/SciPy can each bundle their own libomp, which makes OpenMP abort
# ("OMP: Error #15"). Must be set BEFORE numpy/torch are imported. Harmless on other systems.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Apple GPU (MPS): run an unsupported op on the CPU instead of crashing. Must be set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from src.envcheck import startup_warnings
startup_warnings()          # warn about Rosetta / duplicate OpenMP BEFORE numpy and torch are loaded
import argparse
import json
import time
from dataclasses import replace
import numpy as np
import torch

from src.config import get_config
from src.device import resolve_device, describe
from src.geometry import train_grid, test_grid
from src.dataset import get_datasets, get_offgrid_validation
from src.multiplex import summing_matrix, validate_scheme
from src.train import train_model, predict
from src.evaluate import evaluate, summarize
from src import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="quick", choices=["quick", "medium", "paper"])
    ap.add_argument("--channels", type=int, nargs="*", help="override channel counts")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="compute device for training/prediction (auto = Apple GPU > NVIDIA GPU > CPU)")
    ap.add_argument("--sim-backend", default="auto", choices=["auto", "numpy", "torch"],
                    help="optical simulation: numpy (CPU) or torch (runs on --device). auto = torch on a GPU, numpy on CPU")
    ap.add_argument("--lr-schedule", choices=["const", "cosine"], help="learning-rate schedule (default const)")
    ap.add_argument("--soft-label-sigma", type=float,
                    help="Gaussian soft labels over neighbouring grid points, sigma in mm (default 0 = one-hot)")
    ap.add_argument("--select-best", choices=["none", "offgrid"],
                    help="'offgrid': keep the epoch with the lowest error on off-grid validation hits")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save raw test predictions (pred_XXch.npz) for plot_errors.py")
    ap.add_argument("--batch-size", type=int, help="training batch size (larger batches use a GPU better)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    if args.channels: cfg = replace(cfg, channels=tuple(args.channels))
    if args.epochs: cfg = replace(cfg, epochs=args.epochs)
    if args.seed is not None: cfg = replace(cfg, seed=args.seed)
    if args.batch_size: cfg = replace(cfg, batch_size=args.batch_size)
    if args.lr_schedule: cfg = replace(cfg, lr_schedule=args.lr_schedule)
    if args.soft_label_sigma is not None: cfg = replace(cfg, soft_label_sigma_mm=args.soft_label_sigma)
    if args.select_best: cfg = replace(cfg, select_best=args.select_best)
    device = resolve_device(args.device)
    backend = args.sim_backend if args.sim_backend != "auto" else ("torch" if device.type != "cpu" else "numpy")
    out = args.out or os.path.join("results", args.preset)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Device: {describe(device)} | simulation backend: {backend}")
    variants = [f"lr {cfg.lr_schedule}"] if cfg.lr_schedule != "const" else []
    if cfg.soft_label_sigma_mm > 0: variants.append(f"soft labels sigma={cfg.soft_label_sigma_mm} mm")
    if cfg.select_best != "none": variants.append(f"best epoch by {cfg.select_best} validation")
    if variants: print("Training variants: " + ", ".join(variants))
    print(f"Preset '{args.preset}': channels {cfg.channels}, train pitch {cfg.train_pitch_mm} mm, "
          f"filters {cfg.n_filters}, epochs {cfg.epochs}")

    for c in cfg.channels:
        validate_scheme(c)

    # 1. simulate (cached) ------------------------------------------------------------
    print("[1/4] Optical Monte Carlo")
    tr, te = get_datasets(cfg, cache_dir=os.path.join("results", "cache"),
                           backend=backend, device=device.type)
    train_pos, n_tr = train_grid(cfg)
    test_pos, n_te = test_grid(cfg)
    print(f"  training grid {n_tr}x{n_tr} = {len(train_pos)} classes, test grid {n_te}x{n_te} = {len(test_pos)} points")

    va = None
    if cfg.select_best == "offgrid":
        va = get_offgrid_validation(cfg, os.path.join("results", "cache"), backend=backend, device=device.type)

    # sanity figure: light patterns for a middle / edge / corner position (Fig. 2)
    pick = {"Middle": (0.0, 0.0), "Edge": (0.0, -cfg.crystal_xy_mm / 2 + 1.0),
            "Corner": (-cfg.crystal_xy_mm / 2 + 1.0, -cfg.crystal_xy_mm / 2 + 1.0)}
    pats = []
    for xy in pick.values():
        d = np.linalg.norm(te["xy"] - np.array(xy), axis=1)
        pats.append(te["counts"][d < d.min() + 1e-3].mean(0))
    plots.plot_light_patterns(pats, list(pick), os.path.join(out, "light_patterns.png"))

    # 2-3. multiplex, train, evaluate ---------------------------------------------------
    summary = {}
    for c in cfg.channels:
        print(f"[2-3/4] {c}-channel readout")
        M = summing_matrix(c)
        xtr, xte = tr["counts"] @ M, te["counts"] @ M
        model, scale, hist = train_model(xtr, tr["pos_id"], train_pos, cfg, seed=cfg.seed, device=device,
                                         offgrid=None if va is None else (va["counts"] @ M, va["xy"]))
        pred = predict(model, scale, xte, train_pos, device=device)
        if args.save_predictions:
            np.savez_compressed(os.path.join(out, f"pred_{c:02d}ch.npz"), pred=pred.astype(np.float32),
                                true=te["xy"], pos_id=te["pos_id"], test_pos=test_pos.astype(np.float32))
        res = evaluate(pred, te["xy"], te["pos_id"], len(test_pos), cfg)
        summary[c] = summarize(res)
        s = summary[c]
        print(f"  -> FWHM x/y = {s['fwhm_x']:.2f}/{s['fwhm_y']:.2f} mm,  "
              f"mean|bias| = {s['bias_abs_mean']:.2f} mm,  max|bias| = {s['bias_abs_max']:.2f} mm,  RMS error = {s['rms_mm']:.2f} mm")
        plots.plot_maps(res, n_te, f"{c}-channel readout", os.path.join(out, f"maps_{c:02d}ch.png"))
        np.savez(os.path.join(out, f"per_position_{c:02d}ch.npz"), **res)

    # 4. summary ----------------------------------------------------------------------------
    print("[4/4] Summary")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    with open(os.path.join(out, "summary.csv"), "w") as f:
        f.write("channels,fwhm_x_mm,fwhm_y_mm,fwhm_x_std,fwhm_y_std,bias_abs_mean_mm,bias_abs_max_mm,rms_error_mm\n")
        for c in sorted(summary, reverse=True):
            s = summary[c]
            f.write(f"{c},{s['fwhm_x']:.3f},{s['fwhm_y']:.3f},{s['fwhm_x_std']:.3f},{s['fwhm_y_std']:.3f},"
                    f"{s['bias_abs_mean']:.3f},{s['bias_abs_max']:.3f},{s['rms_mm']:.3f}\n")
    plots.plot_fwhm_vs_channels(summary, os.path.join(out, "fwhm_vs_channels.png"))
    plots.plot_bias_vs_channels(summary, os.path.join(out, "bias_vs_channels.png"))
    _ch = sorted(summary)
    plots.plot_fwhm_line(_ch, [summary[c]["fwhm_x"] for c in _ch], [summary[c]["fwhm_y"] for c in _ch],
                         os.path.join(out, "fwhm_line.png"),
                         std_x=[summary[c]["fwhm_x_std"] for c in _ch], std_y=[summary[c]["fwhm_y_std"] for c in _ch])
    print(f"\nchannels | FWHM x | FWHM y | mean|bias| | max|bias| | RMS err   (mm)")
    for c in sorted(summary, reverse=True):
        s = summary[c]
        print(f"{c:8d} | {s['fwhm_x']:6.2f} | {s['fwhm_y']:6.2f} | {s['bias_abs_mean']:9.2f} | {s['bias_abs_max']:8.2f} | {s['rms_mm']:7.2f}")
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
