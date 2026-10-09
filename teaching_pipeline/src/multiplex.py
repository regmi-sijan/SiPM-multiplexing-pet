"""Multiplexing (analog summing) schemes of Fig. 8 of the paper.

SiPMs are numbered S1..S32 (1-based, as in the paper). A scheme is a list of groups; the
signals of the SiPMs in a group are summed into one readout channel. Only the best scheme
per channel count reported in the paper is included; add your own to SCHEMES to explore.
"""
import numpy as np


def _per_side(side, groups):
    """Replicate a per-side group pattern (given with 1..8 indices) on all 4 sides."""
    off = 8 * side
    return [[off + i for i in g] for g in groups]


def _all_sides(groups):
    return [g for side in range(4) for g in _per_side(side, groups)]


SCHEMES = {
    32: [[i] for i in range(1, 33)],
    # one pair in the middle of every side
    28: _all_sides([[1], [2], [3], [4, 5], [6], [7], [8]]),
    # a pair at both ends of every side
    24: _all_sides([[1, 2], [3], [4], [5], [6], [7, 8]]),
    # alternating pair / single on every side
    20: _all_sides([[1, 2], [3], [4, 5], [6], [7, 8]]),
    # Config16b: neighbouring pairs, shifted by one so that pairs straddle the corners
    16: [[32, 1]] + [[2 * i, 2 * i + 1] for i in range(1, 16)],
    # Config12a: 4-SiPM corner groups, 2-SiPM groups elsewhere
    12: [[1, 2, 31, 32], [3, 4], [5, 6], [7, 8, 9, 10], [11, 12], [13, 14],
         [15, 16, 17, 18], [19, 20], [21, 22], [23, 24, 25, 26], [27, 28], [29, 30]],
    # 8 groups of four, straddling the corners
    8: [[1, 2, 31, 32], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14],
        [15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26], [27, 28, 29, 30]],
    # one channel per side-quarter, centred on the corners
    4: [[1, 2, 3, 4, 29, 30, 31, 32], list(range(5, 13)),
        list(range(13, 21)), list(range(21, 29))],
}


def summing_matrix(n_channels, schemes=SCHEMES, n_sipm=32):
    """(32, C) 0/1 matrix M so that readout = counts @ M."""
    groups = schemes[n_channels]
    M = np.zeros((n_sipm, len(groups)), dtype=np.float32)
    for c, g in enumerate(groups):
        for s in g:
            M[s - 1, c] = 1.0
    return M


def validate_scheme(n_channels, schemes=SCHEMES, n_sipm=32):
    """Every SiPM must feed exactly one channel."""
    M = summing_matrix(n_channels, schemes, n_sipm)
    assert M.shape[1] == n_channels, f"scheme {n_channels}: got {M.shape[1]} channels"
    assert (M.sum(1) == 1).all(), f"scheme {n_channels}: a SiPM is unused or used twice"
    return True


def apply(counts, n_channels):
    return counts @ summing_matrix(n_channels)
