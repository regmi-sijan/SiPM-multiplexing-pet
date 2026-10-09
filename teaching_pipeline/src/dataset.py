"""Generate (and cache) the 32-channel training and test data sets."""
import os
import time
import numpy as np
from .geometry import train_grid, test_grid
from .optics_mc import simulate_events


def _make(positions, events_per_pos, cfg, seed, label):
    rng = np.random.default_rng(seed)
    xy = np.repeat(positions, events_per_pos, axis=0)
    pos_id = np.repeat(np.arange(len(positions)), events_per_pos)
    t0 = time.time()
    counts, z = simulate_events(xy, cfg, rng)
    print(f"  simulated {len(xy):,} {label} events in {time.time() - t0:.1f} s "
          f"(mean detected photons/event: {counts.sum(1).mean():.0f})")
    return dict(counts=counts, xy=xy.astype(np.float32), pos_id=pos_id.astype(np.int32), z=z)


def get_datasets(cfg, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    tag = (f"p{cfg.train_pitch_mm}_tr{cfg.events_per_train_pos}_te{cfg.events_per_test_pos}"
           f"_ly{cfg.light_yield_per_kev}_s{cfg.seed}")
    f = os.path.join(cache_dir, f"data_{tag}.npz")
    if os.path.exists(f):
        print(f"  loading cached data {f}")
        d = np.load(f)
        tr = {k[3:]: d[k] for k in d.files if k.startswith("tr_")}
        te = {k[3:]: d[k] for k in d.files if k.startswith("te_")}
        return tr, te
    train_pos, _ = train_grid(cfg)
    test_pos, _ = test_grid(cfg)
    tr = _make(train_pos, cfg.events_per_train_pos, cfg, cfg.seed, "training")
    te = _make(test_pos, cfg.events_per_test_pos, cfg, cfg.seed + 1, "test")
    np.savez_compressed(f, **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"te_{k}": v for k, v in te.items()})
    return tr, te
