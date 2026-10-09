"""Train the position CNN for one readout configuration (CPU or GPU).

Optional variants (all off by default, so the default is the plain baseline):
  * cfg.lr_schedule = "cosine"       learning rate decays smoothly to 1 % of its start value
  * cfg.soft_label_sigma_mm > 0      the target is a Gaussian over neighbouring grid points instead of one class
  * cfg.select_best = "offgrid"      keep the weights of the epoch with the lowest error on off-grid validation hits
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from .model import PositionCNN, centre_of_mass


def to_images(x, scale, device="cpu"):
    """(E, C) counts -> (E, 1, 1, C) float32 tensor 'grayscale image', globally scaled."""
    t = torch.from_numpy((x / scale).astype(np.float32)).reshape(-1, 1, 1, x.shape[1])
    return t.to(device)


def soft_target_matrix(calib, sigma_mm):
    """(N, N) matrix; row i = Gaussian (sigma in mm) over the calibration positions around position i, rows sum to 1.
    With sigma -> 0 it becomes the identity, i.e. ordinary one-hot labels."""
    d2 = (calib[:, None, :] - calib[None, :, :]).pow(2).sum(-1)
    S = torch.exp(-d2 / (2 * sigma_mm ** 2))
    return S / S.sum(1, keepdim=True)


def train_model(x_train, label_train, calib_xy, cfg, seed=0, verbose=True, device="cpu", offgrid=None):
    """x_train: (E, C) readout counts, label_train: (E,) index into calib_xy.

    offgrid: optional (x_val (E2, C), xy_val (E2, 2) in mm). Needed for cfg.select_best == 'offgrid'.
    Returns (model, scale, hist); hist rows = (train loss, val loss, val mean error mm, off-grid RMS mm, lr).
    """
    device = torch.device(device)
    if device.type == "cpu":
        # Very small ("denormal") floats make CPU training noticeably slower as the weights settle;
        # flushing them to zero was ~40 % faster here with identical accuracy. No-op if unsupported.
        torch.set_flush_denormal(True)
    if cfg.select_best == "offgrid" and offgrid is None:
        raise ValueError("select_best='offgrid' needs the off-grid validation set (offgrid=...)")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(np.percentile(x_train, 99.9)) + 1e-6
    n = len(x_train)
    idx = rng.permutation(n)
    n_val = int(cfg.val_fraction * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]            # 10 % validation [paper]
    # the whole (small) data set lives on the device, so no per-batch host->GPU copies
    X = to_images(x_train, scale, device)
    y = torch.from_numpy(label_train.astype(np.int64)).to(device)
    calib = torch.from_numpy(calib_xy.astype(np.float32)).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)
    X_off = off_xy = None
    if offgrid is not None:
        X_off = to_images(offgrid[0], scale, device)
        off_xy = torch.from_numpy(np.asarray(offgrid[1], dtype=np.float32)).to(device)

    model = PositionCNN(x_train.shape[1], len(calib_xy), cfg.n_filters, cfg.kernel_size).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    sched = None
    if cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.01 * cfg.lr)
    elif cfg.lr_schedule != "const":
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")
    soft = soft_target_matrix(calib, cfg.soft_label_sigma_mm) if cfg.soft_label_sigma_mm > 0 else None
    loss_fn = nn.CrossEntropyLoss()
    hist, best_err, best_state = [], float("inf"), None
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        lr_now = opt.param_groups[0]["lr"]
        perm = torch.from_numpy(rng.permutation(tr_idx)).to(device)
        tot = torch.zeros((), device=device)
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            logits = model(X[b])
            if soft is None:
                loss = loss_fn(logits, y[b])
            else:
                loss = -(soft[y[b]] * torch.log_softmax(logits, dim=1)).sum(1).mean()
            loss.backward()
            opt.step()
            tot += loss.detach() * len(b)                 # no .item() per step -> no GPU sync stalls
        if sched is not None:
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = model(X[val_idx_t])
            val_loss = loss_fn(vl, y[val_idx_t]).item()
            val_err = (centre_of_mass(vl, calib) - calib[y[val_idx_t]]).norm(dim=1).mean().item()
            off_rms = float("nan")
            if X_off is not None:
                d = centre_of_mass(model(X_off), calib) - off_xy
                off_rms = d.pow(2).sum(1).mean().sqrt().item()
        if cfg.select_best == "offgrid" and off_rms < best_err:
            best_err = off_rms
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
        hist.append((tot.item() / len(perm), val_loss, val_err, off_rms, lr_now))
        if verbose:
            extra = f"  off-grid RMS {off_rms:.2f} mm" if X_off is not None else ""
            print(f"    epoch {ep + 1:2d}/{cfg.epochs}  train loss {hist[-1][0]:.3f}  "
                  f"val loss {val_loss:.3f}  val mean error {val_err:.2f} mm{extra}  "
                  f"[{time.time() - t0:.0f}s]")
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"    -> kept the epoch with the lowest off-grid RMS ({best_err:.2f} mm)")
    return model, scale, hist


def predict(model, scale, x, calib_xy, batch=8192, device=None):
    """Return (E, 2) predicted positions as a NumPy array (always on the CPU)."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    calib = torch.from_numpy(calib_xy.astype(np.float32)).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(centre_of_mass(model(to_images(x[i:i + batch], scale, device)), calib).cpu())
    return torch.cat(out).numpy()
