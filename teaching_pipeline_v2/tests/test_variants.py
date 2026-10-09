"""Tests for the optional training variants and the multi-seed tools.  python tests/test_variants.py"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import torch
from src.config import Config
from src.geometry import train_grid
from src.optics_mc import simulate_events
from src.train import train_model, predict, soft_target_matrix


def _tiny(**kw):
    cfg = replace(Config(), train_pitch_mm=4.0, n_filters=8, epochs=4, batch_size=64, **kw)
    pos, _ = train_grid(cfg)                                   # 10 x 10 = 100 classes
    rng = np.random.default_rng(0)
    counts, _ = simulate_events(np.repeat(pos, 20, axis=0), cfg, rng)
    labels = np.repeat(np.arange(len(pos)), 20)
    off_xy = rng.uniform(-16, 16, (300, 2))
    off_counts, _ = simulate_events(off_xy, cfg, rng)
    return cfg, pos, counts, labels, off_counts, off_xy


def test_soft_targets():
    calib = torch.from_numpy(train_grid(replace(Config(), train_pitch_mm=4.0))[0].astype(np.float32))
    S = soft_target_matrix(calib, 2.0)
    assert torch.allclose(S.sum(1), torch.ones(len(calib)), atol=1e-5)
    assert (S.argmax(1) == torch.arange(len(calib))).all()                       # peak on the true position
    assert torch.allclose(soft_target_matrix(calib, 1e-3), torch.eye(len(calib)), atol=1e-6)   # sigma -> 0 = one-hot
    assert S[0, 1] > S[0, 50]                                                    # neighbours weigh more than far points


def test_cosine_schedule_decays_and_const_does_not():
    cfg, pos, c, y, *_ = _tiny(lr_schedule="cosine")
    _, _, h = train_model(c, y, pos, cfg, verbose=False)
    lrs = [r[4] for r in h]
    assert abs(lrs[0] - cfg.lr) < 1e-9 and all(a > b for a, b in zip(lrs, lrs[1:]))
    cfg2 = replace(cfg, lr_schedule="const")
    _, _, h2 = train_model(c, y, pos, cfg2, verbose=False)
    assert len({r[4] for r in h2}) == 1


def test_soft_labels_train_and_predict():
    cfg, pos, c, y, *_ = _tiny(soft_label_sigma_mm=3.0)
    m, s, h = train_model(c, y, pos, cfg, verbose=False)
    p = predict(m, s, c[:40], pos)
    assert p.shape == (40, 2) and np.isfinite(p).all() and np.isfinite(h[-1][0])


def test_best_epoch_selection_restores_the_best_weights():
    cfg, pos, c, y, oc, oxy = _tiny(select_best="offgrid")
    m, s, h = train_model(c, y, pos, cfg, verbose=False, offgrid=(oc, oxy))
    best = min(r[3] for r in h)
    p = predict(m, s, oc, pos)
    rms = float(np.sqrt(((p - oxy) ** 2).sum(1).mean()))
    assert abs(rms - best) < 1e-3, (rms, best)             # the restored model IS the best epoch
    try:
        train_model(c, y, pos, cfg, verbose=False)         # no off-grid data -> clear error
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_default_has_no_variants():
    cfg = Config()
    assert cfg.lr_schedule == "const" and cfg.soft_label_sigma_mm == 0 and cfg.select_best == "none"


def test_seed_aggregation():
    import run_seeds
    with tempfile.TemporaryDirectory() as d:
        for s, shift in [(1, 0.0), (2, 0.2)]:
            f = os.path.join(d, f"seed_{s}"); os.makedirs(f)
            json.dump({str(c): dict(fwhm_x=0.6 + shift, fwhm_y=0.6 + shift, bias_abs_mean=0.3, bias_abs_max=2.0,
                                    rms_mm=0.5 + shift) for c in (32, 16)}, open(os.path.join(f, "summary.json"), "w"))
        agg = run_seeds.aggregate(run_seeds.seed_folders(d))
        mu, sd, n = agg[32]["rms_mm"]
        assert n == 2 and abs(mu - 0.6) < 1e-9
        assert abs(sd - 0.2 / np.sqrt(2)) < 1e-9                  # sample std (ddof=1) of [0.5, 0.7]
        assert sorted(agg, reverse=True) == [32, 16]


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_") and callable(fn) and fn.__module__ == "__main__":
            fn(); print("ok", n)
