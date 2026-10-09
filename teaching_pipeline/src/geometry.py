"""Crystal and SiPM layout.

SiPM numbering follows Fig. 8 of the paper (clockwise, S1..S32):
  S1..S8    top side,    left  -> right
  S9..S16   right side,  top   -> bottom
  S17..S24  bottom side, right -> left
  S25..S32  left side,   bottom -> top
In code the channels are 0-based (S1 = index 0).
"""
import numpy as np


def sipm_centres(cfg):
    """Return (32, 2) array of SiPM-centre (x, y) positions in mm (on the crystal edge)."""
    n, p, h = cfg.n_sipm_per_side, cfg.sipm_pitch_mm, cfg.crystal_xy_mm / 2
    offs = (np.arange(n) - (n - 1) / 2) * p
    top = np.stack([offs, np.full(n, h)], 1)
    right = np.stack([np.full(n, h), -offs], 1)
    bottom = np.stack([-offs, np.full(n, -h)], 1)
    left = np.stack([np.full(n, -h), offs], 1)
    return np.concatenate([top, right, bottom, left])


def grid_positions(cfg, pitch=None, offset=0.0):
    """Gamma-source (x, y) grid.

    The paper's training grid is 40x40 points with 1 mm pitch (points at the centre of
    1 mm cells); the test grid is shifted by half a pitch, giving 39x39 points.
    Returns (positions (N,2), nx).
    """
    pitch = pitch or cfg.train_pitch_mm
    L = cfg.crystal_xy_mm
    n_train = int(round(L / pitch))
    if offset == 0.0:
        axis = (np.arange(n_train) + 0.5) * pitch - L / 2
    else:
        axis = (np.arange(n_train - 1) + 1.0) * pitch - L / 2
    xx, yy = np.meshgrid(axis, axis, indexing="xy")   # row = y, col = x
    return np.stack([xx.ravel(), yy.ravel()], 1), len(axis)


def train_grid(cfg):
    return grid_positions(cfg, offset=0.0)


def test_grid(cfg):
    return grid_positions(cfg, offset=1.0)
