"""Vectorised optical Monte Carlo of a single monolithic LSO plate with edge SiPMs.

This replaces the GATE/Geant4 simulation of the paper with a transparent, fast model:

* A 511 keV gamma is absorbed at (x, y, z). x, y = beam position; z drawn from the
  truncated exponential attenuation law (gamma enters from the top face).
  [simplification] full-energy deposit only: no Compton scatter, no escape, no energy window.
* N ~ Poisson(light_yield * 511 keV) optical photons are emitted isotropically.
* Top and bottom faces carry ESR: specular reflection with probability 0.98, else absorbed.
  [paper]  The reflection count of every photon is obtained analytically by "unfolding" the
  z-direction, so there is no per-bounce loop.
* Photons reaching a side wall are detected only if they land on a 4 mm SiPM window
  (0.6 mm gaps between SiPMs and the strips beyond the array ends absorb them [assumed]),
  with probability PDE = 0.5 [paper].
* Detected photons per SiPM are counted -> shot (Poisson) noise is inherent, and is the
  only noise source, exactly as in the paper (no electronic noise).
"""
import numpy as np


def _simulate_chunk(xy, z, n_ph, cfg, rng):
    E = xy.shape[0]
    L = cfg.crystal_xy_mm / 2
    T = cfg.crystal_z_mm
    nmax = int(n_ph.max())
    shape = (E, nmax)
    valid = np.arange(nmax)[None, :] < n_ph[:, None]

    cos_t = rng.uniform(-1, 1, shape).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, shape).astype(np.float32)
    s = np.sqrt(1 - cos_t ** 2)
    ux, uy, uz = s * np.cos(phi), s * np.sin(phi), cos_t

    x0 = xy[:, 0:1].astype(np.float32)
    y0 = xy[:, 1:2].astype(np.float32)
    z0 = z[:, None].astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (np.where(ux > 0, L, -L) - x0) / ux
        ty = (np.where(uy > 0, L, -L) - y0) / uy
    tx = np.where(np.isfinite(tx), tx, np.inf)
    ty = np.where(np.isfinite(ty), ty, np.inf)
    hit_x = tx < ty
    t = np.where(hit_x, tx, ty)

    # number of ESR reflections on the way to the side wall (z-unfolding)
    z_end = z0 + uz * t
    n_refl = np.abs(np.floor(z_end / T))
    alive = rng.random(shape, dtype=np.float32) < np.power(cfg.esr_reflectance, n_refl)

    # position along the wall in "clockwise" coordinates, and side index
    #   top (uy>0): s = x   right (ux>0): s = -y   bottom (uy<0): s = -x   left (ux<0): s = y
    with np.errstate(invalid="ignore"):          # t is inf on the unused axis when u == 0
        xw = np.nan_to_num(x0 + ux * t, nan=0.0, posinf=0.0, neginf=0.0)
        yw = np.nan_to_num(y0 + uy * t, nan=0.0, posinf=0.0, neginf=0.0)
    side = np.where(hit_x, np.where(ux > 0, 1, 3), np.where(uy > 0, 0, 2))
    s_cw = np.select([side == 0, side == 1, side == 2], [xw, -yw, -xw], default=yw)

    n, p, w = cfg.n_sipm_per_side, cfg.sipm_pitch_mm, cfg.sipm_size_mm
    k = np.rint(s_cw / p + (n - 1) / 2).astype(np.int16)
    centre = (k - (n - 1) / 2) * p
    in_window = (k >= 0) & (k < n) & (np.abs(s_cw - centre) <= w / 2)
    detected = alive & in_window & (rng.random(shape, dtype=np.float32) < cfg.sipm_pde) & valid

    chan = side * n + np.clip(k, 0, n - 1)
    ev = np.broadcast_to(np.arange(E)[:, None], shape)
    flat = (ev * cfg.n_sipm + chan)[detected]
    counts = np.bincount(flat, minlength=E * cfg.n_sipm).reshape(E, cfg.n_sipm)
    return counts.astype(np.int32)


def simulate_events(xy, cfg, rng, chunk_events=48, backend="numpy", device="cpu"):
    """Simulate one gamma event for every row of ``xy`` (E, 2) in mm.

    backend 'numpy' runs on the CPU; 'torch' runs on ``device`` ('cpu', 'mps' or 'cuda').
    Returns (counts (E, 32) int32, depth_z (E,) float32). Chunked to bound memory.
    """
    E = xy.shape[0]
    z = _sample_depth(E, cfg, rng)
    n_ph = rng.poisson(cfg.n_photons_mean, E)
    if backend == "torch":
        from .optics_mc_torch import simulate_events_torch
        return simulate_events_torch(xy, z, n_ph, cfg, rng, device), z.astype(np.float32)
    if backend != "numpy":
        raise ValueError(f"unknown simulation backend {backend!r}")
    out = np.empty((E, cfg.n_sipm), dtype=np.int32)
    for i in range(0, E, chunk_events):
        j = min(E, i + chunk_events)
        out[i:j] = _simulate_chunk(xy[i:j], z[i:j], n_ph[i:j], cfg, rng)
    return out, z.astype(np.float32)


def _sample_depth(E, cfg, rng):
    """z of interaction (0 = bottom, T = top). Gamma enters from the top."""
    T, mu = cfg.crystal_z_mm, cfg.mu_per_mm
    u = rng.random(E)
    d = -np.log(1 - u * (1 - np.exp(-mu * T))) / mu       # depth below the top face
    return T - d
