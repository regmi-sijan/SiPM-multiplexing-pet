"""Spatial resolution (FWHM) and bias per test position (paper Section 2.3)."""
import numpy as np
from scipy.optimize import curve_fit

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548


def _gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_error_distribution(err, window=3.0, method="auto"):
    """Return (fwhm, bias) of a 1-D error sample (mm).

    'gauss'  : histogram + Gaussian fit, FWHM = 2.355 sigma, bias = centroid   [as in paper]
    'robust' : median and 1.4826*MAD (same quantities, stable for few events)
    'auto'   : gauss if there are >= 150 events, else robust
    """
    med = np.median(err)
    mad_sigma = 1.4826 * np.median(np.abs(err - med)) + 1e-6
    if method == "auto":
        method = "gauss" if len(err) >= 150 else "robust"
    if method == "robust":
        return FWHM_FACTOR * mad_sigma, med
    # clip to the window, histogram, fit
    e = err[np.abs(err - med) < window]
    nb = max(15, int(np.sqrt(len(e)) * 1.5))
    h, edges = np.histogram(e, bins=nb)
    c = 0.5 * (edges[1:] + edges[:-1])
    try:
        p, _ = curve_fit(_gauss, c, h, p0=[h.max(), med, mad_sigma], maxfev=2000)
        if not np.isfinite(p).all() or p[2] <= 0:
            raise RuntimeError
        return FWHM_FACTOR * abs(p[2]), p[1]
    except Exception:
        return FWHM_FACTOR * mad_sigma, med


def evaluate(pred_xy, true_xy, pos_id, n_pos, cfg):
    """Return dict of per-position arrays: fwhm_x, fwhm_y, bias_x, bias_y (shape (n_pos,))."""
    err = pred_xy - true_xy
    order = np.argsort(pos_id, kind="stable")
    err, pid = err[order], pos_id[order]
    bounds = np.searchsorted(pid, np.arange(n_pos + 1))
    out = {k: np.zeros(n_pos) for k in ("fwhm_x", "fwhm_y", "bias_x", "bias_y", "rms_x", "rms_y")}
    for i in range(n_pos):
        e = err[bounds[i]:bounds[i + 1]]
        out["rms_x"][i], out["rms_y"][i] = np.sqrt((e ** 2).mean(0))   # total error, robust to 'snapping'
        for ax, name in ((0, "x"), (1, "y")):
            f, b = fit_error_distribution(e[:, ax], cfg.hist_window_mm, cfg.fit)
            out[f"fwhm_{name}"][i], out[f"bias_{name}"][i] = f, b
    return out


def summarize(res):
    return dict(
        fwhm_x=float(res["fwhm_x"].mean()), fwhm_y=float(res["fwhm_y"].mean()),
        fwhm_x_std=float(res["fwhm_x"].std()), fwhm_y_std=float(res["fwhm_y"].std()),
        bias_x_mean=float(res["bias_x"].mean()), bias_y_mean=float(res["bias_y"].mean()),
        bias_abs_max=float(max(np.abs(res["bias_x"]).max(), np.abs(res["bias_y"]).max())),
        bias_abs_mean=float(0.5 * (np.abs(res["bias_x"]).mean() + np.abs(res["bias_y"]).mean())),
        rms_mm=float(0.5 * (res["rms_x"].mean() + res["rms_y"].mean())),
    )
