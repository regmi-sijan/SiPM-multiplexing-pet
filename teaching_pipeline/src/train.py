"""Train the position CNN for one readout configuration."""
import time
import numpy as np
import torch
import torch.nn as nn
from .model import PositionCNN, centre_of_mass


def to_images(x, scale):
    """(E, C) counts -> (E, 1, 1, C) float tensor 'grayscale image', globally scaled."""
    return torch.from_numpy((x / scale).astype(np.float32)).reshape(-1, 1, 1, x.shape[1])


def train_model(x_train, label_train, calib_xy, cfg, seed=0, verbose=True):
    """x_train: (E, C) readout counts, label_train: (E,) index into calib_xy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(np.percentile(x_train, 99.9)) + 1e-6
    n = len(x_train)
    idx = rng.permutation(n)
    n_val = int(cfg.val_fraction * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]            # 10 % validation [paper]
    X = to_images(x_train, scale)
    y = torch.from_numpy(label_train.astype(np.int64))
    calib = torch.from_numpy(calib_xy.astype(np.float32))

    model = PositionCNN(x_train.shape[1], len(calib_xy), cfg.n_filters, cfg.kernel_size)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(tr_idx))
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = model(X[val_idx])
            val_loss = loss_fn(vl, y[val_idx]).item()
            val_err = (centre_of_mass(vl, calib) - calib[y[val_idx]]).norm(dim=1).mean().item()
        hist.append((tot / len(perm), val_loss, val_err))
        if verbose:
            print(f"    epoch {ep + 1:2d}/{cfg.epochs}  train loss {hist[-1][0]:.3f}  "
                  f"val loss {val_loss:.3f}  val mean error {val_err:.2f} mm  "
                  f"[{time.time() - t0:.0f}s]")
    return model, scale, hist


def predict(model, scale, x, calib_xy, batch=4096):
    model.eval()
    calib = torch.from_numpy(calib_xy.astype(np.float32))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(centre_of_mass(model(to_images(x[i:i + batch], scale)), calib))
    return torch.cat(out).numpy()
