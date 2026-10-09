# Guide to `teaching_pipeline` (v1, CPU version)

What this code is about, what it does, and how to run it.

This is the **original** version. It runs on the CPU only. The newer `teaching_pipeline_v2` adds Apple-GPU support, an RMS metric and optional training switches. See `teaching_pipeline_v2/CHANGELOG.md` for the differences.

---

## 1. What this project is about

The code is a small, runnable replication of the study published as:

> Subedi, S. K., Cherry, S. R., Qiang, Y., & Peng, P. (2025).
> *Feasibility study of multiplexing analog signals from SiPMs for a single layer monolithic PET detector design.*
> **Radiation Measurements 182**, 107399. <https://doi.org/10.1016/j.radmeas.2025.107399>

This code was written from the earlier preprint (SSRN 4865116, `ssrn-4865116.pdf`), so **every section, figure and
table number quoted below refers to the preprint**, not to the published article. The
[repository README](../README.md#which-version-of-the-paper-is-referenced) maps the two.

**The detector.** A PET scanner detects pairs of 511 keV gamma rays. This detector is one flat LSO crystal plate, 40 x 40 x 4 mm. A gamma that hits the plate makes a burst of scintillation light. Thirty-two small light sensors (SiPMs, 4 x 4 mm, 8 on each side of the plate) catch some of that light. The light pattern across the 32 sensors depends on where the gamma interacted, so a neural network can learn to turn the pattern back into an (x, y) position.

**The problem.** 32 sensors need 32 readout channels, which is expensive. Multiplexing adds the signals of several sensors together so fewer channels are needed (28, 24, 20, 16, 12, 8 or 4). The cost is that position information is mixed.

**The question the paper asks.** How much position accuracy (FWHM resolution and bias) is lost as the number of readout channels goes down? The paper's answer is that you can go down to roughly 8 - 16 channels with little loss, and that 4 channels degrades clearly.

**What this code does.** It reproduces that experiment end to end, at reduced size so it runs on a laptop:

1. Simulate gamma interactions and the light that reaches each of the 32 SiPMs.
2. Apply each multiplexing scheme to get fewer channels.
3. Train a CNN per scheme to predict the interaction position.
4. Measure FWHM, bias and error on a separate test set, and plot the results against channel count.

**What it is not.** The paper used the GATE Monte Carlo toolkit. This code replaces it with a simple optical Monte Carlo that I wrote. It also uses fewer events and (in the `quick` and `medium` presets) a coarser 2 mm training grid. The *trends* should resemble the paper. The exact numbers will not match it.

---

## 2. How the pipeline works

```
positions on a grid
        |
        v
 optical Monte Carlo  ->  32 SiPM counts per event            (src/optics_mc.py)
        |
        v
 multiplexing          ->  N-channel signal (N = 32 ... 4)    (src/multiplex.py)
        |
        v
 CNN position decoder  ->  softmax over grid points           (src/model.py, src/train.py)
        |                  position = centre of mass
        v
 evaluation            ->  FWHM and bias per position         (src/evaluate.py)
        |
        v
 plots / tables        ->  results/<preset>/                  (src/plots.py)
```

### Step by step

| Step | What happens | Where | Paper |
|---|---|---|---|
| Geometry | A training grid covers the 40 x 40 mm plate (1 mm pitch in the paper = 1600 points; 2 mm in `quick`/`medium`). The test grid is shifted by half a pitch (39 x 39 = 1521 points in the paper), so test positions are never training positions. SiPMs are numbered S1 to S32 clockwise. | `src/geometry.py` | 2.1, 2.2, Fig. 4 |
| Simulation | For each event, photons leave the interaction point in random directions. Reflections off the top and bottom ESR surfaces (reflectance 0.98) are handled analytically. Photons that reach a side wall inside a SiPM window are detected with probability 0.5. Counts are Poisson-noised. The gamma deposits its full energy at an exponentially distributed depth. | `src/optics_mc.py`, `src/config.py` | 2.1 |
| Multiplexing | A fixed 0/1 summing matrix maps 32 SiPM signals to N channels. Schemes follow the paper's Fig. 8. | `src/multiplex.py` | 2.4, Fig. 8 |
| Model | The N channels are a 1 x N "image". Conv (200 filters in the paper) -> BatchNorm -> ReLU -> fully connected layer with one node per grid point -> softmax. The predicted position is the probability-weighted centre of mass of the grid points (paper Eq. 1). | `src/model.py` | 2.2, Fig. 3 |
| Training | SGD with momentum, cross-entropy loss, 10 % validation split. One model is trained per channel count. | `src/train.py` | 2.2 |
| Evaluation | At each test position, the error distribution of the predicted position is fitted (Gaussian, or median/MAD when a fit is unreliable). FWHM = width, bias = centroid offset. | `src/evaluate.py` | 2.3 |
| Plots | FWHM and bias versus channels, per-position maps, example light patterns. | `src/plots.py` | Figs. 2, 9, 11, 12 |
| Driver | Runs everything for the chosen preset. | `run_all.py` | |

### Presets

| Preset | Grid | Events per training / test position | Filters | Epochs | Channel counts | Approximate time on CPU |
|---|---|---|---|---|---|---|
| `quick` | 2 mm (400 classes) | 40 / 60 | 32 | 15 | 32, 16, 8, 4 | a few minutes |
| `medium` | 2 mm | 100 / 100 | 64 | 20 | all 8 | tens of minutes (about 25 min on 2 cores in my test) |
| `paper` | 1 mm (1600 classes) | 300 / 200 | 200 | 20 | all 8 | hours; **not run by me** |

---

## 3. Folder contents

```
teaching_pipeline/
├── GUIDE.md             this file
├── README.md            short overview, assumptions, troubleshooting
├── requirements.txt     numpy, scipy, matplotlib, torch, pytest
├── run_all.py           main entry point
├── check_env.py         diagnoses Python/OpenMP problems
├── setup_env.sh         builds a clean .venv
├── run.sh               runs run_all.py inside .venv
├── src/
│   ├── config.py        parameters and presets ([paper] vs [assumed] marked)
│   ├── geometry.py      grids and SiPM positions
│   ├── optics_mc.py     optical Monte Carlo
│   ├── multiplex.py     multiplexing schemes (32 ... 4 channels)
│   ├── dataset.py       generates and caches the data sets
│   ├── model.py         the CNN and the centre-of-mass decoder
│   ├── train.py         training and prediction
│   ├── evaluate.py      FWHM and bias
│   ├── plots.py         figures
│   └── envcheck.py      environment checks used at start-up
├── tests/
│   ├── test_pipeline.py   5 sanity tests of the pipeline
│   └── test_env.py        3 tests of the environment checks
├── example_results/     reference outputs (medium preset, one seed)
└── results/             your own outputs and the data cache
```

---

## 4. How to run it

All commands are run from inside this folder:

```bash
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1/teaching_pipeline
```

### 4.1 One-time setup (recommended on your Mac)

On your machine the conda environment contains two OpenMP libraries, which caused `OMP: Error #15`, a segmentation fault, and a hang at "32-channel readout". Use a clean environment:

```bash
./setup_env.sh                  # builds .venv with native Python 3.10 - 3.13 and tests it
.venv/bin/python check_env.py      # expect: all checks passed
```

Use `./setup_env.sh --recreate` to rebuild it from scratch.

Then run the project with `./run.sh ...`, which is the same as `.venv/bin/python run_all.py ...`. You do not need to activate anything.

If you prefer not to use the script, any clean, native, non-conda environment works:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 4.2 Sanity checks

```bash
.venv/bin/python tests/test_pipeline.py      # about 10 seconds, expect five "ok" lines
.venv/bin/python tests/test_env.py
```

What they check: light never exceeds the generated photons, multiplexing conserves total signal, the model outputs one probability per grid point, and FWHM of a known Gaussian is recovered.

### 4.3 Fast smoke run (under a minute)

```bash
./run.sh --preset quick --channels 4 --epochs 2 --out results/smoke
```

### 4.4 The real runs

```bash
./run.sh --preset quick                                # a few minutes
./run.sh --preset medium                               # tens of minutes
./run.sh --preset paper                                # hours; CPU only in this version
```

Options:

| Option | Meaning |
|---|---|
| `--preset` (quick, medium or paper) | Size of the experiment. |
| `--channels 32 16 8 4` | Choose which channel counts to run. |
| `--epochs N` | Training epochs. |
| `--seed N` | Random seed. |
| `--out DIR` | Output folder (default `results/<preset>`). |

Examples:

```bash
./run.sh --preset quick --channels 32 20 12 --epochs 25 --seed 3
./run.sh --preset medium --out results/medium 2>&1 | tee results/medium.log
caffeinate -i ./run.sh --preset medium                 # keep the Mac awake
```

Stop a run with a single **Ctrl-C**. Do not use Ctrl-Z, which only suspends the process.

### 4.5 What you get

Files in `results/<preset>/`:

| File | Content |
|---|---|
| `summary.csv` / `summary.json` | Median FWHM and mean bias per channel count. |
| `fwhm_vs_channels.png` | FWHM versus channel count (the central result). |
| `bias_vs_channels.png` | Bias versus channel count. |
| `maps_XXch.png` | FWHM and bias over the crystal for each channel count. |
| `light_patterns.png` | Example SiPM light patterns. |
| `per_position_XXch.npz` | Raw per-position numbers. |

Simulated data is cached in `results/cache/`, so a repeated run skips the simulation. Delete that folder to force fresh data.

### 4.6 How to read the results

- Look at the **trend**: error should grow slowly from 32 down to about 8 channels, then more clearly at 4.
- On the 2 mm presets, FWHM can read close to 0 mm at some positions. Predictions snap onto calibration points, so that is an artifact and not a real resolution. This version has no RMS metric, which is the reason `teaching_pipeline_v2` adds one.
- `example_results/` has a reference run to compare with.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OMP: Error #15 ... libomp already initialized` | Two OpenMP copies, usually in conda. | Use `.venv` (section 4.1). |
| `zsh: segmentation fault` | Same problem. | Same fix. |
| Run appears stuck at "32-channel readout" | Same problem (seen in your conda run). | Ctrl-C, use `.venv`. |
| `No .venv yet: run ./setup_env.sh first.` | Environment missing. | `./setup_env.sh`. |
| `permission denied` on a `.sh` file | Executable bit lost (e.g. copied via a zip or a Windows share). | `chmod +x *.sh`, or prefix the command with `bash`. |
| Intel/Rosetta warning | Python is `x86_64` on an M1. | Install native Python, rebuild with `./setup_env.sh --recreate`. |

`KMP_DUPLICATE_LIB_OK=TRUE` is set in the scripts as a workaround, but it only hides the error message and is not safe in a broken environment.

---

## 6. Known limits of this version

- CPU only. It does not use the Apple GPU, and `paper` is slow on a CPU.
- No RMS metric, so FWHM on the 2 mm presets can look better than the real accuracy.
- A single fixed training recipe and single runs. There is no multi-seed script.
- The simulator is simpler than GATE, so absolute numbers differ from the paper.
- Several settings are assumptions, not from the paper (marked `[assumed]` in `src/config.py`): events per position, kernel size, number of epochs and batch size.

For the GPU, RMS metric, training variants and multi-seed tools, use `teaching_pipeline_v2`.
