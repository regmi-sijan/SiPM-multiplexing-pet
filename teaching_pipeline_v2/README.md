# Teaching version (v2, GPU-enabled): multiplexing SiPM signals in a monolithic PET detector

> **This is v2 of a two-version repository** — see the [repository README](../README.md).
> v2 = v1 + GPU support: same physics, same pipeline, plus `--device auto|cpu|mps|cuda` (Apple GPU via Metal,
> NVIDIA via CUDA, or CPU), a torch optical simulation that runs on the GPU, an RMS-error metric, a
> paper-comparison plot, multi-seed and variant tooling, error diagnostics, and a CPU speed fix.
> [`CHANGELOG.md`](CHANGELOG.md) lists every change from [v1](../teaching_pipeline/).
>
> **Just want to run it?** [`HOW_TO_RUN.md`](HOW_TO_RUN.md) is the step-by-step guide: setup, every CLI option,
> the output layout, timings and troubleshooting. This README covers the design and the assumptions.

Re-implementation (for learning, not a bit-exact reproduction) of

> Subedi, S. K., Cherry, S. R., Qiang, Y., & Peng, P. (2025).
> *Feasibility study of multiplexing analog signals from SiPMs for a single layer monolithic PET detector design.*
> **Radiation Measurements 182**, 107399. <https://doi.org/10.1016/j.radmeas.2025.107399>
>
> This code was written from the earlier preprint, [SSRN 4865116](https://ssrn.com/abstract=4865116), so **all
> section, figure and table numbers in this repository refer to the preprint** — they differ from the published
> article. See the [numbering map](../README.md#which-version-of-the-paper-is-referenced).

**Question:** a 40 x 40 x 4 mm LSO plate is read out by 32 SiPMs on its four edges. How many
readout channels can you remove by summing SiPM signals *before* digitisation without losing
position accuracy? The paper finds the average resolution stays about 0.5 mm down to 16 channels
and degrades below that, with growing corner bias at 12 channels and fewer.

## Pipeline

```
 gamma hit (x, y, z)                                   1 x C "grayscale image"
        │                                                      │
 optics_mc.py ──► 32 SiPM photon counts ──► multiplex.py ──► model.py (CNN) ──► softmax over N calibration
 (replaces GATE)        (Poisson noise)      (sum groups)      conv→BN→ReLU→FC    positions → centre of mass
                                                                                         │
                                                                evaluate.py: per-position FWHM + bias
```

| Step | File | Paper section |
|---|---|---|
| Geometry, SiPM numbering S1..S32 | `src/geometry.py` | 2.1, Fig. 1, Fig. 8 |
| Optical Monte Carlo (replaces GATE) | `src/optics_mc.py` | 2.1 |
| 40x40 training grid, offset 39x39 test grid | `src/geometry.py`, `src/dataset.py` | 2.2, Fig. 4 |
| Multiplexing schemes 28 ... 4 channels | `src/multiplex.py` | 2.4, Fig. 8 |
| CNN, softmax centre-of-mass (Eq. 1) | `src/model.py`, `src/train.py` | 2.2, Fig. 3 |
| Gaussian-fit FWHM and bias | `src/evaluate.py` | 2.3 |
| Figures | `src/plots.py` | Figs 2, 9, 11, 12 |
| Device selection (MPS / CUDA / CPU) | `src/device.py` | (new in v2) |
| Torch version of the simulation (GPU-capable) | `src/optics_mc_torch.py` | (new in v2) |
| CPU-vs-GPU timing | `benchmark.py` | (new in v2) |
| Optional training variants | `src/train.py`, `src/config.py` | (new in v2, not in the paper) |
| Multi-seed runs and variant comparison | `run_seeds.py` | (new in v2) |
| Error diagnostics (snapping check) | `plot_errors.py` | (new in v2) |
| The paper's published FWHM / MSE numbers | `src/paper_reference.py` | Tables 1, 2 (new in v2) |
| FWHM-vs-channels plot with error bars, against the paper | `plot_summary.py` | Fig. 12 (new in v2) |
| Everything in one go | `run_all.py` | |

## Run it

Recommended on macOS (builds a clean, pip-only `.venv` — see
[Troubleshooting](#troubleshooting-the-python-environment-macos-omp-error-15-segmentation-faults)):

```bash
./setup_env.sh                           # one time: build .venv and test it
.venv/bin/python check_env.py            # expect "All checks passed"
./run.sh --preset quick                  # same arguments as run_all.py below
```

Or, in an environment you already trust:

```bash
pip install -r requirements.txt
python -m pytest -q tests                # all 18 tests
python tests/test_pipeline.py            # 5 quick sanity checks (~10 s)
python tests/test_device.py              # 4 device / GPU-code checks
python tests/test_variants.py            # 6 checks for the training variants and multi-seed tools
python run_all.py --preset quick         # ~2 min on a laptop CPU: channels 32, 16, 8, 4
python run_all.py --preset medium        # all 8 channel counts, 2 mm class grid (tens of minutes)
python run_all.py --preset paper         # 1 mm grid / 1600 classes / 200 filters (GPU or hours)
python run_all.py --preset quick --channels 32 20 12 --epochs 25 --seed 3
```

Output goes to `results/<preset>/`: `summary.csv`, `fwhm_vs_channels.png`, `bias_vs_channels.png`,
`maps_XXch.png` (FWHM and bias maps like paper Figs 9/11), `light_patterns.png` (like Fig. 2),
`fwhm_line.png` (FWHM with error bars against the paper's Table 2, like Fig. 12) and raw per-position
arrays (`per_position_XXch.npz`). Simulated data is cached in `results/cache/`.

## GPU support (new in v2)

```bash
python benchmark.py                       # ~1-2 min: times simulation and training on CPU vs GPU
python run_all.py --preset medium         # --device auto picks the Apple GPU (mps) if available
python run_all.py --preset medium --device cpu    # force the CPU
python run_all.py --preset paper --device mps --batch-size 256
python tests/test_device.py               # device + torch-simulation checks (also pass without a GPU)
```

* **What runs on the GPU:** CNN training and prediction, and (with `--sim-backend torch`, the default when a GPU
  is used) the optical Monte Carlo. Evaluation (Gaussian fits) and plotting stay on the CPU.
* **Apple Silicon requirement:** Python must run natively on arm64, otherwise PyTorch cannot see the GPU. Check with
  `python -c "import platform; print(platform.machine())"` (must print `arm64`, not `x86_64`) and
  `python -c "import torch; print(torch.backends.mps.is_available())"` (must print `True`).
  If it is not available, the code prints a warning and runs on the CPU.
* **Unsupported GPU operations** fall back to the CPU automatically (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
* **The GPU is not always faster.** The `quick` and `medium` networks are tiny, so launch overhead can eat the
  gain; the GPU pays off mainly for the `paper` preset (1600 classes, 200 filters) and for the simulation.
  Run `benchmark.py` and use whichever device is faster. Larger `--batch-size` (e.g. 256) helps GPUs but changes
  training dynamics slightly versus the default 128.
* **Results differ slightly between devices and backends** because random number generators and floating-point
  rounding differ. Compare trends and use the same `--seed` and settings when you compare runs.
* **Cached data is separate per simulation backend** (`..._numpy.npz` vs `..._torch.npz`).
* **CPU speed fix:** denormal floats are flushed to zero during CPU training (about 40 % faster in my test,
  identical accuracy).
* **Tested where:** the torch simulation and the training path were tested on a Linux CPU (they match the NumPy
  version statistically; see `tests/test_device.py`). The Apple GPU path itself could not be tested in my environment,
  so run `python tests/test_device.py` and `python benchmark.py` on your Mac first.

## Training variants and multi-seed experiments (new in v2)

The baseline trains exactly as described in the paper (one-hot labels, fixed learning rate, last epoch kept). Three
optional switches, **my additions and not from the paper**, are off by default:

| Flag | What it does |
|---|---|
| `--lr-schedule cosine` | learning rate decays smoothly to 1 % over the epochs |
| `--soft-label-sigma 1.0` | target = Gaussian (sigma in mm) over neighbouring grid points instead of one class, which teaches interpolation between grid points |
| `--select-best offgrid` | keep the epoch with the lowest error on validation hits at random, off-grid positions (the normal 10 % validation split sits on the training grid points, so it cannot see interpolation errors) |

```bash
# several seeds of one setting: mean +/- std table and plot
python run_seeds.py --seeds 1 2 3 --out results/q40 -- --preset quick --epochs 40
# compare variants (5 variants x 3 seeds; simulated data is cached per seed)
python run_seeds.py --variants base cosine soft best all --seeds 1 2 3 --out results/cmp -- --preset quick --epochs 40
# rebuild the table/plot from existing folders
python run_seeds.py --compare results/cmp/base results/cmp/soft
# how are the errors made? (needs --save-predictions)
python run_all.py --preset quick --epochs 40 --save-predictions --out results/diag
python plot_errors.py results/diag --channels 32 16 4
```
Everything after a lone `--` goes to `run_all.py` unchanged (add `--device cpu`, `--channels ...`, etc.).

### Example result: quick preset, 40 epochs, 3 seeds (CPU)

RMS error in mm, mean over seeds (seed-to-seed std was 0.00 to 0.05); `example_results/variants_cmp40.csv/png`:

| Variant | 32 ch | 16 ch | 8 ch | 4 ch |
|---|---|---|---|---|
| base (paper-like) | 0.61 | 0.50 | 0.56 | 1.21 |
| cosine LR | 0.49 | 0.37 | 0.50 | 1.45 |
| soft labels | 0.33 | 0.34 | 0.48 | 1.16 |
| best off-grid epoch | 0.46 | 0.44 | 0.56 | 1.19 |
| all three | **0.28** | **0.30** | 0.47 | 1.41 |

What to take from it:

* **The baseline error is dominated by interpolation between the calibration grid points**, not by the readout. With the
  2 mm class grid the test hits (half a pitch = 1 mm off-grid) are hard for a classifier trained on one-hot labels. Soft
  labels roughly halve the error at 32 and 16 channels. Longer training with one-hot labels made the 32/16-channel test
  results worse even though the on-grid validation loss kept improving (a misleading validation signal).
* **Once that is reduced** (soft labels / "all"), the trend resembles the paper's: about the same error at 32 and 16
  channels (0.28 vs 0.30 mm), clearly worse at 8 (0.47) and much worse at 4 (about 1.2 to 1.4 mm). In the baseline the
  8-channel loss was hidden under the interpolation error.
* **The 4-channel numbers are not final.** Cosine decay makes 4 channels worse (1.45 vs 1.21): the 4-channel network is
  still improving at epoch 40 and needs more epochs. Do not read a precise number for 4 channels from these runs.
* **"Snapping" (the network jumping to grid points) is only part of the story.** `plot_errors.py` shows a mild effect at
  32 channels after 40 epochs (snapping index 0.26 vs about 0.15 for evenly spread predictions) and none at 16 or 4.
* **Cautions:** coarse 2 mm grid, small network, simplified optics, FWHM from a median/MAD estimate with 60 events per
  position. Compare trends, not absolute values with the paper. Soft labels and the learning-rate schedule are not
  what the paper used, so a "better" number from them is not a better reproduction.

## Troubleshooting the Python environment (macOS: OMP Error #15, segmentation faults)

Symptoms: `OMP: Error #15 ... libomp.dylib already initialized`, or `zsh: segmentation fault` when NumPy and
PyTorch run together. Cause: the environment (often conda) contains two different OpenMP runtimes. The workaround
variable `KMP_DUPLICATE_LIB_OK=TRUE` (already set by the scripts) hides the first message but cannot make that
combination safe. The reliable fix is a clean, pip-only environment:

```bash
./setup_env.sh              # builds .venv with a native Python 3.10-3.13 (not conda) and tests it
./run.sh --preset quick     # runs run_all.py inside .venv, nothing to activate
python3 check_env.py        # diagnosis: runs each risky step in its own process and reports which one crashes
python3 check_env.py --full  # also runs the project's test files
```

* `setup_env.sh` needs a native Python. If it finds none it tells you to run `brew install python@3.12`
  (or use the python.org installer). `./setup_env.sh --recreate` rebuilds `.venv` from scratch.
* `run_all.py` prints an **ENVIRONMENT WARNING** at start-up if it detects Rosetta (Intel Python on an M1) or several
  OpenMP runtimes in the environment.
* If you must stay in the broken environment, `check_env.py` tests whether single-threaded math libraries avoid the
  crash and prints the exact `export ...` line; this is slower and still fragile.
* `tests/test_env.py` unit-tests these checks.

## What is taken from the paper and what is assumed

Taken from the paper: crystal size, ESR reflectance 0.98 on top/bottom, 32 SiPMs of 4x4 mm at 4.6 mm
pitch, SiPM efficiency 0.5, 511 keV pencil beam at normal incidence, shot noise only (no
electronics noise), 1 mm training grid (1600 positions), test grid shifted by half a pitch (1521 positions),
CNN layout (conv with 200 filters, batch norm, ReLU, fully connected layer with N nodes, softmax),
centre-of-mass position estimate, SGD with momentum, 10 % validation split, the multiplexing
groupings of Fig. 8, and the Gaussian-fit FWHM / centroid-bias evaluation.

Assumed or simplified (all in `src/config.py`, marked `[assumed]`):

* **No GATE/Geant4.** Optical transport is an analytic ray-trace: isotropic photons, specular ESR
  reflections counted by unfolding the thickness direction, absorbing side walls except at the SiPM windows.
* **Full-energy absorption only**: no Compton scatter, no escape, no 20 % energy window.
  Interaction depth follows the exponential attenuation law (mu = 0.088 /mm).
* **Light yield 30 photons/keV**, no absorption or Rayleigh scattering inside the crystal, no
  refractive-index/critical-angle effects (the paper's UNIFIED-model details are not fully specified).
* Fewer events per position than the paper (about 550 to 950 there); CNN kernel size, epochs, batch size
  and momentum are my choices.
* Only the best scheme per channel count from the paper is included, so the study of alternative
  schemes at 16, 12, 8 and 4 channels is not reproduced. Add entries to `SCHEMES` in `src/multiplex.py`.
* The paper's "bias-corrected" FWHM step (Eq. 2) is not applied: FWHM from a Gaussian fit is already
  independent of the bias (the centroid).


## Comparing with the paper (new in v2)

`src/paper_reference.py` holds the paper's published numbers, transcribed from the PDF:

* `TABLE2_OPTIMAL` — average FWHM and standard deviation (X and Y) for the best multiplexing scheme at each
  channel count, over the paper's 1521 test positions (Table 2).
* `TABLE1_CORNER_MSE` — mean square error at the detector corners (Table 1). **Not** directly comparable with
  the RMS error reported here, which is computed over the whole test grid, and whose corner-grid size varies
  per row in the paper.

`run_all.py` writes `fwhm_line.png` automatically; `plot_summary.py` redraws it for any existing run folder and
also writes a side-by-side table:

```bash
python plot_summary.py results/medium              # -> fwhm_line.png + paper_comparison.csv
python plot_summary.py results/medium --no-paper   # our numbers only
```

The error bar is the **standard deviation of the per-position FWHM across the test grid** — the same definition
the paper uses for the error bars in its Figs. 12 and 22. It describes variation *across the detector face*, not
run-to-run uncertainty; for the latter, use `run_seeds.py`.

## Example result (preset `medium`, seed 1, Apple M1 GPU, about 1 minute)

Our FWHM ± σ against the paper's Table 2, from `paper_comparison.csv`:

| Channels | This code, FWHM Y | Paper, FWHM Y | This code, FWHM X | Paper, FWHM X | ours / paper | RMS error |
|---|---|---|---|---|---|---|
| 32 | 0.75 ± 0.46 | 0.50 ± 0.24 | 0.75 ± 0.48 | 0.50 ± 0.24 | 1.50× | 0.63 |
| 28 | 0.65 ± 0.35 | 0.51 ± 0.23 | 0.65 ± 0.34 | 0.52 ± 0.23 | 1.26× | 0.57 |
| 24 | 0.80 ± 0.37 | 0.50 ± 0.21 | 0.78 ± 0.34 | 0.50 ± 0.21 | 1.58× | 0.53 |
| 20 | 0.72 ± 0.31 | 0.51 ± 0.21 | 0.71 ± 0.31 | 0.51 ± 0.21 | 1.41× | 0.51 |
| 16 | 0.82 ± 0.29 | 0.51 ± 0.19 | 0.83 ± 0.28 | 0.51 ± 0.20 | 1.61× | 0.51 |
| 12 | 0.72 ± 0.29 | 0.52 ± 0.18 | 0.72 ± 0.32 | 0.53 ± 0.18 | 1.37× | 0.51 |
| 8 | 0.76 ± 0.29 | 0.58 ± 0.17 | 0.77 ± 0.30 | 0.58 ± 0.17 | 1.32× | 0.59 |
| 4 | 1.05 ± 0.48 | 0.69 ± 0.25 | 1.09 ± 0.53 | 0.69 ± 0.24 | 1.55× | 1.14 |

(All values in mm. Figure: `example_results/fwhm_line_medium.png`.)

How it compares with the paper:

* **The qualitative story matches.** Resolution is essentially flat over a wide range of channel counts, then
  degrades clearly at the lowest count, and bias grows at 4 channels.
* **Absolute FWHM is 1.3–1.6× the paper's** throughout: coarser 2 mm class grid, fewer events, smaller network,
  simplified optics. The error bars overlap the paper's at every channel count, because the spread across the
  detector face is large in both.
* **The "knee" is in a different place.** The paper degrades below 16 channels; here the flat region extends
  down to 8 and only 4 is clearly worse. With soft labels (see above) the 8-channel loss becomes visible, which
  suggests the baseline knee is partly hidden under interpolation error rather than being a real difference.
* **The zig-zag between neighbouring channel counts is noise.** This is one seed; differences of 0.05–0.1 mm are
  within seed-to-seed variation. Use `run_seeds.py` before reading anything into the ordering.
* **Scheme caveat.** Only the 12-channel scheme is matched to the paper's optimal configuration (Config 12a) by
  name. The 16- and 4-channel schemes here have not been verified against the paper's Config 16b and 4b, so part
  of the gap at those counts may be a different grouping rather than a different simulation.

## Reading the results honestly

* Expect the **trend**, not the exact numbers: roughly flat resolution from 32 down to about 16
  channels, then worse, and bias that grows at the lowest channel counts.
* With the coarse **2 mm class grid** (quick/medium presets) the classifier can "snap" onto one
  calibration point, which makes the FWHM at that position look unrealistically small (even about 0 mm)
  and inflates bias. That is why the summary also reports the **RMS error** (bias and spread combined),
  which does not have this artifact. The 1 mm `paper` preset reduces it.
* With few events per position the fit falls back from a histogram Gaussian fit to median/MAD
  (`fit="auto"` in the config); both estimate the same quantities.
* Results vary a little with the random seed. Try `--seed` several times before trusting small differences.

## Ideas to extend

Add Compton scattering and an energy window; model crystal absorption and the SiPM electronic noise;
try other summing schemes (e.g. non-adjacent SiPMs) and rank them by RMS error; add depth-of-interaction
with stacked layers; swap the CNN for a plain MLP or a regression head.
