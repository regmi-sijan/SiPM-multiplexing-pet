"""PyTorch version of the optical Monte Carlo in ``optics_mc.py``.

Same physics and the same formulas, written with torch tensors so that it can run on the Apple GPU
(MPS) or an NVIDIA GPU. On the CPU it is statistically equivalent to the NumPy version (see
tests/test_device.py) but not bit-identical, because the random number generators differ.
"""
import math
import numpy as np
import torch


def _chunk(xy, z, n_ph, cfg, dev):
    E = xy.shape[0]
    L, T = cfg.crystal_xy_mm / 2, cfg.crystal_z_mm
    n, p, w = cfg.n_sipm_per_side, cfg.sipm_pitch_mm, cfg.sipm_size_mm
    nmax = int(n_ph.max())
    shape = (E, nmax)

    n_ph_d = n_ph.to(dev)
    valid = torch.arange(nmax, device=dev)[None, :] < n_ph_d[:, None]

    cos_t = torch.rand(shape, device=dev) * 2 - 1
    phi = torch.rand(shape, device=dev) * (2 * math.pi)
    s = torch.sqrt(torch.clamp(1 - cos_t ** 2, min=0))
    ux, uy, uz = s * torch.cos(phi), s * torch.sin(phi), cos_t

    x0 = xy[:, 0:1].to(dev)
    y0 = xy[:, 1:2].to(dev)
    z0 = z[:, None].to(dev)

    inf = torch.tensor(float("inf"), device=dev)
    wall_x = torch.where(ux > 0, torch.full_like(ux, L), torch.full_like(ux, -L))
    wall_y = torch.where(uy > 0, torch.full_like(uy, L), torch.full_like(uy, -L))
    tx = torch.nan_to_num((wall_x - x0) / ux, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    ty = torch.nan_to_num((wall_y - y0) / uy, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    hit_x = tx < ty
    t = torch.where(hit_x, tx, ty)

    # ESR reflections on the way to the side wall (z-unfolding)
    n_refl = torch.floor((z0 + uz * t) / T).abs()
    alive = torch.rand(shape, device=dev) < torch.pow(cfg.esr_reflectance, n_refl)

    xw = torch.nan_to_num(x0 + ux * t, nan=0.0, posinf=0.0, neginf=0.0)
    yw = torch.nan_to_num(y0 + uy * t, nan=0.0, posinf=0.0, neginf=0.0)
    side = torch.where(hit_x, 1 + 2 * (ux <= 0).long(), 2 * (uy <= 0).long())   # 0 top,1 right,2 bottom,3 left
    s_cw = torch.where(side == 0, xw, torch.where(side == 1, -yw, torch.where(side == 2, -xw, yw)))

    k = torch.round(s_cw / p + (n - 1) / 2).long()
    centre = (k.float() - (n - 1) / 2) * p
    in_window = (k >= 0) & (k < n) & ((s_cw - centre).abs() <= w / 2)
    detected = alive & in_window & (torch.rand(shape, device=dev) < cfg.sipm_pde) & valid

    chan = side * n + k.clamp(0, n - 1)
    flat = (torch.arange(E, device=dev)[:, None] * cfg.n_sipm + chan).reshape(-1)
    counts = torch.zeros(E * cfg.n_sipm, device=dev)
    counts.scatter_add_(0, flat, detected.reshape(-1).float())
    return counts.reshape(E, cfg.n_sipm).round().to(torch.int32).cpu().numpy()


def simulate_events_torch(xy, z, n_ph, cfg, rng, device, chunk_events=192):
    """xy (E,2), z (E,), n_ph (E,) -> counts (E, 32) int32. Depth and photon numbers are drawn by the
    caller (NumPy) so that both back ends share the same event definition."""
    dev = torch.device(device)
    torch.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    xy_t = torch.from_numpy(np.asarray(xy, dtype=np.float32))
    z_t = torch.from_numpy(np.asarray(z, dtype=np.float32))
    n_t = torch.from_numpy(np.asarray(n_ph, dtype=np.int64))
    E = xy_t.shape[0]
    out = np.empty((E, cfg.n_sipm), dtype=np.int32)
    for i in range(0, E, chunk_events):
        j = min(E, i + chunk_events)
        out[i:j] = _chunk(xy_t[i:j], z_t[i:j], n_t[i:j], cfg, dev)
    return out
