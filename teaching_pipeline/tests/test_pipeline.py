"""Fast sanity checks. Run:  python -m pytest -q   (or: python tests/test_pipeline.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from src.config import Config
from src.multiplex import SCHEMES, validate_scheme, summing_matrix
from src.geometry import sipm_centres, train_grid as _train_grid, test_grid as _test_grid
from src.optics_mc import simulate_events
from src.evaluate import fit_error_distribution


def test_schemes_are_valid():
    for c in SCHEMES:
        assert validate_scheme(c)


def test_grids():
    cfg = Config()
    tr, n = _train_grid(cfg); te, m = _test_grid(cfg)
    assert (n, len(tr)) == (40, 1600) and (m, len(te)) == (39, 1521)       # [paper]
    assert np.allclose(np.unique(te[:, 0])[0] - np.unique(tr[:, 0])[0], 0.5)  # half-pitch shift


def test_sipm_layout():
    c = sipm_centres(Config())
    assert c.shape == (32, 2)
    assert np.allclose(c[0], [-16.1, 20]) and np.allclose(c[7], [16.1, 20])   # S1, S8 (top, left->right)
    assert np.allclose(c[8], [20, 16.1])                                      # S9 (right, top)
    assert np.allclose(c[16], [16.1, -20])                                    # S17 (bottom, right)
    assert np.allclose(c[24], [-20, -16.1])                                   # S25 (left, bottom)


def test_light_goes_to_nearest_side_and_is_symmetric():
    cfg = Config(); rng = np.random.default_rng(0)
    xy = np.tile([[0.0, 15.0]], (300, 1))                # near the top edge
    counts, _ = simulate_events(xy, cfg, rng)
    side = counts.reshape(300, 4, 8).sum((0, 2))
    assert side[0] == side.max()                          # top side sees most light
    mean = counts.mean(0)
    assert abs(mean[:8][:4].sum() - mean[:8][4:].sum()) / mean[:8].sum() < 0.1   # left-right symmetry
    assert 800 < counts.sum(1).mean() < 6000              # plausible collected photons


def test_fit():
    rng = np.random.default_rng(1)
    e = rng.normal(0.2, 0.25, 4000)
    f, b = fit_error_distribution(e, method="gauss")
    assert abs(f - 2.355 * 0.25) < 0.08 and abs(b - 0.2) < 0.03
    f2, b2 = fit_error_distribution(e[:80], method="robust")
    assert abs(b2 - 0.2) < 0.15


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_") and callable(fn) and fn.__module__ == "__main__":
            fn(); print("ok", n)
