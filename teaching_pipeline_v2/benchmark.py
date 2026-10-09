#!/usr/bin/env python3
"""Compare CPU and GPU speed on your machine (about 1 to 2 minutes).

    python benchmark.py
    python benchmark.py --events 12000 --epochs 4
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import time
from dataclasses import replace
import numpy as np
import torch

from src.config import get_config
from src.device import resolve_device, describe, synchronize
from src.geometry import train_grid
from src.optics_mc import simulate_events
from src.train import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=9600, help="events for the simulation benchmark")
    ap.add_argument("--epochs", type=int, default=3, help="timed training epochs")
    args = ap.parse_args()

    cfg = replace(get_config("medium"), epochs=args.epochs)
    gpu = resolve_device("auto")
    print(f"Best device found: {describe(gpu)}")
    if gpu.type == "cpu":
        print("No GPU available to this Python; only CPU numbers will be shown. On an M1 check that "
              "`python -c \"import platform; print(platform.machine())\"` prints arm64.")

    # ---- simulation ------------------------------------------------------------------------
    xy = np.random.default_rng(0).uniform(-19, 19, (args.events, 2))
    rows = []
    t = time.time(); simulate_events(xy, cfg, np.random.default_rng(1), backend="numpy")
    rows.append(("simulation, numpy on CPU", time.time() - t))
    for name, dev in [("cpu", torch.device("cpu"))] + ([("gpu", gpu)] if gpu.type != "cpu" else []):
        simulate_events(xy[:300], cfg, np.random.default_rng(1), backend="torch", device=dev.type)  # warm-up
        t = time.time(); simulate_events(xy, cfg, np.random.default_rng(1), backend="torch", device=dev.type)
        synchronize(dev)
        rows.append((f"simulation, torch on {dev.type}", time.time() - t))

    # ---- training (the 'medium' network: 64 filters, 400 classes) ----------------------------------
    pos, _ = train_grid(cfg)
    sim_xy = np.repeat(pos, 30, axis=0)
    counts, _ = simulate_events(sim_xy, cfg, np.random.default_rng(2))
    labels = np.repeat(np.arange(len(pos)), 30)
    for dev in [torch.device("cpu")] + ([gpu] if gpu.type != "cpu" else []):
        train_model(counts, labels, pos, replace(cfg, epochs=1), verbose=False, device=dev)   # warm-up
        t = time.time(); train_model(counts, labels, pos, cfg, verbose=False, device=dev)
        synchronize(dev)
        rows.append((f"training ({args.epochs} epochs), {dev.type}", time.time() - t))

    print(f"\n{'step':38s} {'seconds':>8s}")
    for r in rows:
        print(f"{r[0]:38s} {r[1]:8.1f}")
    cpu_tr = [r[1] for r in rows if r[0].startswith("training") and r[0].endswith("cpu")]
    gpu_tr = [r[1] for r in rows if r[0].startswith("training") and not r[0].endswith("cpu")]
    if cpu_tr and gpu_tr:
        print(f"\nTraining speedup on GPU: {cpu_tr[0] / gpu_tr[0]:.1f}x")
        print("(If this is below 1x the network is too small for the GPU to pay off; try --device cpu for "
              "the quick preset and --device mps for medium/paper, ideally with --batch-size 256.)")


if __name__ == "__main__":
    main()
