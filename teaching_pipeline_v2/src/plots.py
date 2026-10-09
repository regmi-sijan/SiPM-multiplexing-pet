"""Figures that mirror the paper: light patterns (Fig. 2), per-position maps (Figs 9/11),
and FWHM vs channel number (Fig. 12)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, AMBER, NAVY = "#1C7293", "#F2A541", "#0B2A3C"


def plot_light_patterns(counts_by_pos, labels, path):
    """Mean 1x32 light pattern for a middle / edge / corner position (cf. Fig. 2)."""
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 1.0 * len(labels) + 0.8))
    for ax, c, lab in zip(np.atleast_1d(axes), counts_by_pos, labels):
        ax.imshow(c[None, :], aspect="auto", cmap="gray", vmin=0)
        ax.set_yticks([]); ax.set_ylabel(lab, rotation=0, labelpad=30, va="center")
        ax.set_xticks(range(0, 32, 4)); ax.set_xticklabels([f"S{i + 1}" for i in range(0, 32, 4)])
    fig.suptitle("Mean detected-photon pattern on the 32 SiPMs")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_maps(res, nx, title, path):
    """FWHM and bias maps for X and Y over the test grid (cf. Figs 9 and 11)."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    spec = [("fwhm_y", "FWHM along Y [mm]", "viridis"), ("fwhm_x", "FWHM along X [mm]", "viridis"),
            ("bias_y", "Bias along Y [mm]", "coolwarm"), ("bias_x", "Bias along X [mm]", "coolwarm")]
    for ax, (k, t, cm) in zip(axes.ravel(), spec):
        v = res[k].reshape(nx, nx)
        kw = {}
        if k.startswith("bias"):
            m = max(0.2, np.abs(v).max()); kw = dict(vmin=-m, vmax=m)
        im = ax.imshow(v, origin="lower", cmap=cm, **kw)
        ax.set_title(f"{t}\navg {v.mean():.2f}  [min {v.min():.2f}, max {v.max():.2f}]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_fwhm_vs_channels(summary, path):
    ch = sorted(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, ax_name in zip(axes, ("y", "x")):
        m = [summary[c][f"fwhm_{ax_name}"] for c in ch]
        s = [summary[c][f"fwhm_{ax_name}_std"] for c in ch]
        ax.errorbar(ch, m, yerr=s, fmt="o", color="#C8553D", ecolor=TEAL, capsize=3)
        ax.set_xlabel("Readout channels"); ax.set_ylabel(f"Average FWHM ({ax_name.upper()}) [mm]")
        ax.grid(alpha=0.3)
    fig.suptitle("Spatial resolution vs number of readout channels (bars: std over test grid)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_bias_vs_channels(summary, path):
    ch = sorted(summary)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(ch, [summary[c]["bias_abs_mean"] for c in ch], "o-", color=TEAL, label="mean |bias|")
    ax.plot(ch, [summary[c]["bias_abs_max"] for c in ch], "s--", color=AMBER, label="max |bias|")
    ax.plot(ch, [summary[c]["rms_mm"] for c in ch], "^:", color="#C8553D", label="RMS error")
    ax.set_xlabel("Readout channels"); ax.set_ylabel("Bias [mm]"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_fwhm_line(channels, fwhm_x, fwhm_y, path, std_x=None, std_y=None, paper_ref=True):
    """Average FWHM vs readout channels with error bars, Y (left) and X (right) like the paper's Fig. 12.
    Error bar = standard deviation of the per-position FWHM over the test grid (same definition as the paper).
    Paper values (Table 2, optimal schemes) are drawn as grey squares, shifted slightly left of each channel count."""
    from .paper_reference import TABLE2_OPTIMAL
    ch = np.asarray(channels)
    order = np.argsort(ch)
    ch = ch[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name, mean, sd, pi in ((axes[0], "Y", fwhm_y, std_y, 0), (axes[1], "X", fwhm_x, std_x, 2)):
        mean = np.asarray(mean)[order]
        yerr = None if sd is None else np.asarray(sd)[order]
        if paper_ref:
            pc = sorted(TABLE2_OPTIMAL)
            ax.errorbar(np.array(pc) - 0.6, [TABLE2_OPTIMAL[c][pi] for c in pc],
                        yerr=[TABLE2_OPTIMAL[c][pi + 1] for c in pc], fmt="s", color="#777777", ecolor="#BBBBBB",
                        capsize=3, ms=5, label="Paper (Table 2)")
        ax.errorbar(ch + 0.6, mean, yerr=yerr, fmt="o", color=TEAL, ecolor=AMBER, elinewidth=1.8, capsize=3,
                    ms=6, label="This run")
        ax.set_xticks(ch)
        _top = max(float(np.max(mean + (0 if yerr is None else yerr))), 1.0)
        ax.set_ylim(0, _top * 1.05)
        ax.set_xlabel("Readout channels"); ax.set_ylabel(f"Average FWHM ({name}) [mm]")
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Average FWHM vs readout channels (bars: std of FWHM over the test grid)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
