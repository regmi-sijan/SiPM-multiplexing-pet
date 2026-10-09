# Teaching version (v1, CPU): multiplexing SiPM signals in a monolithic PET detector

> **This is v1 of a two-version repository** — see the [repository README](../README.md).
> v1 runs on the CPU only. [`teaching_pipeline_v2/`](../teaching_pipeline_v2/) has the **same physics and the
> same pipeline** plus Apple/NVIDIA GPU support, an RMS-error metric, a paper-comparison plot, multi-seed runs
> and error diagnostics. Every difference is listed in [v2's CHANGELOG](../teaching_pipeline_v2/CHANGELOG.md).
> **Use v2 unless you specifically want the smaller, simpler code.**
>
> **New to this code?** [`GUIDE.md`](GUIDE.md) is the step-by-step introduction: what it is, how it works, how
> to run it. This README covers the design decisions and the assumptions.

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
| Everything in one go | `run_all.py` | |
| Step-by-step introduction and run guide | [`GUIDE.md`](GUIDE.md) | |

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
python -m pytest -q tests                # 8 tests
python tests/test_pipeline.py            # 5 quick sanity checks (~10 s)
python run_all.py --preset quick         # ~2 min on a laptop CPU: channels 32, 16, 8, 4
python run_all.py --preset medium        # all 8 channel counts, 2 mm class grid (tens of minutes)
python run_all.py --preset paper         # 1 mm grid / 1600 classes / 200 filters (GPU or hours)
python run_all.py --preset quick --channels 32 20 12 --epochs 25 --seed 3
```

Output goes to `results/<preset>/`: `summary.csv`, `fwhm_vs_channels.png`, `bias_vs_channels.png`,
`maps_XXch.png` (FWHM and bias maps like paper Figs 9/11), `light_patterns.png` (like Fig. 2) and
raw per-position arrays (`per_position_XXch.npz`). Simulated data is cached in `results/cache/`.

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


## Example result (preset `medium`, seed 1, about 25 min on 2 CPU cores)

| Channels | FWHM x / y (mm) | mean abs bias (mm) | RMS error (mm) |
|---|---|---|---|
| 32 | 0.74 / 0.72 | 0.49 | 0.63 |
| 28 | 0.65 / 0.64 | 0.44 | 0.57 |
| 24 | 0.77 / 0.75 | 0.39 | 0.54 |
| 20 | 0.72 / 0.71 | 0.37 | 0.52 |
| 16 | 0.82 / 0.80 | 0.35 | 0.51 |
| 12 | 0.72 / 0.72 | 0.35 | 0.51 |
| 8 | 0.79 / 0.77 | 0.43 | 0.60 |
| 4 | 1.08 / 1.04 | 0.94 | 1.14 |

How it compares with the paper: the qualitative story matches (resolution is essentially unchanged over a wide
range of channel counts, then degrades clearly at the lowest counts, with bias growing at 4 channels).
The numbers differ: the absolute FWHM is about 0.7 mm here versus about 0.5 mm in the paper (coarser 2 mm class
grid, fewer events, smaller network, simplified optics), and the "knee" appears lower here (between 8 and 4)
than the paper's 16. Differences of 0.05 to 0.1 mm between neighbouring channel counts are within seed-to-seed
noise, so do not read structure into them. Use `--seed` and the `paper` preset to tighten this up.

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
