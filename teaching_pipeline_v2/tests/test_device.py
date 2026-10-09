"""Checks for the device handling and the torch simulation backend.
Run:  python tests/test_device.py   (or: python -m pytest -q)
These pass on a machine without a GPU too (everything then runs on the CPU)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import torch
from src.config import Config
from src.device import resolve_device, describe
from src.optics_mc import simulate_events
from src.train import train_model, predict
from src.geometry import train_grid


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "mps", "cuda")
    assert resolve_device("mps").type in ("cpu", "mps")        # falls back to cpu if there is no Apple GPU
    assert isinstance(describe(resolve_device("auto")), str)


def test_torch_simulation_matches_numpy_on_cpu():
    """Same physics, different random generators: the mean light pattern must agree statistically."""
    cfg = Config()
    for pos in [(0.0, 0.0), (10.0, -15.0), (-19.0, 19.0)]:
        xy = np.tile(np.array([pos]), (1500, 1))
        a, _ = simulate_events(xy, cfg, np.random.default_rng(1), backend="numpy")
        b, _ = simulate_events(xy, cfg, np.random.default_rng(2), backend="torch", device="cpu")
        ma, mb = a.mean(0), b.mean(0)
        assert abs(ma.sum() - mb.sum()) / ma.sum() < 0.03, (pos, ma.sum(), mb.sum())
        assert np.abs(ma - mb).max() / ma.sum() < 0.01, pos


def test_torch_simulation_on_best_device_runs():
    cfg = Config()
    dev = resolve_device("auto")
    xy = np.tile([[0.0, 15.0]], (200, 1))
    c, _ = simulate_events(xy, cfg, np.random.default_rng(0), backend="torch", device=dev.type)
    assert c.shape == (200, 32) and c.dtype == np.int32
    side = c.reshape(200, 4, 8).sum((0, 2))
    assert side[0] == side.max() and 800 < c.sum(1).mean() < 6000


def test_training_smoke_on_best_device():
    cfg = replace(Config(), train_pitch_mm=4.0, n_filters=8, epochs=3, batch_size=64)
    pos, _ = train_grid(cfg)                                    # 10 x 10 = 100 classes
    rng = np.random.default_rng(0)
    xy = np.repeat(pos, 20, axis=0)
    counts, _ = simulate_events(xy, cfg, rng)
    labels = np.repeat(np.arange(len(pos)), 20)
    dev = resolve_device("auto")
    model, scale, hist = train_model(counts, labels, pos, cfg, seed=0, verbose=False, device=dev)
    assert np.isfinite(hist[-1][1]) and hist[-1][1] < hist[0][1] + 1e-6   # validation loss did not blow up
    p = predict(model, scale, counts[:50], pos, device=dev)
    assert isinstance(p, np.ndarray) and p.shape == (50, 2) and np.isfinite(p).all()


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_") and callable(fn) and fn.__module__ == "__main__":
            fn(); print("ok", n)
