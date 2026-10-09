# How to run `teaching_pipeline_v2` (step by step)

This code reproduces the study published as Subedi, Cherry, Qiang & Peng (2025), *Radiation Measurements* **182**,
107399, <https://doi.org/10.1016/j.radmeas.2025.107399>. Section, figure and table numbers used anywhere in this
repository refer to the earlier preprint ([SSRN 4865116](https://ssrn.com/abstract=4865116)), which this code was
written from. Please cite the published article — see [`../README.md`](../README.md#citing).

This guide assumes a Mac with Apple Silicon (M1, 16 GB) and `zsh`. It also notes what to change on Linux/Windows/NVIDIA.
Every command is run from inside the `teaching_pipeline_v2` folder:

```bash
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1/teaching_pipeline_v2
```

Contents

1. [Quick path (copy-paste)](#1-quick-path-copy-paste)
2. [Step 0 - what you need](#2-step-0---what-you-need)
3. [Step 1 - create the clean environment](#3-step-1---create-the-clean-environment)
4. [Step 2 - check the environment](#4-step-2---check-the-environment)
5. [Step 3 - run the tests](#5-step-3---run-the-tests)
6. [Step 4 - benchmark CPU vs GPU](#6-step-4---benchmark-cpu-vs-gpu-optional)
7. [Step 5 - first real run (`quick`)](#7-step-5---first-real-run-quick)
8. [Reading the results](#8-reading-the-results)
9. [Step 6 - longer runs (`medium`, `paper`)](#9-step-6---longer-runs-medium-paper)
10. [Step 7 - training variants](#10-step-7---optional-training-variants)
11. [Step 8 - several seeds and comparing variants](#11-step-8---several-seeds-and-comparing-variants)
12. [Step 9 - error diagnostics](#12-step-9---error-diagnostics)
13. [All command-line options](#13-all-command-line-options)
14. [Output folder layout](#14-output-folder-layout)
15. [Caching and re-running](#15-caching-and-re-running)
16. [Troubleshooting](#16-troubleshooting)
17. [How long things take](#17-how-long-things-take)
18. [Suggested order for a first session](#18-suggested-order-for-a-first-session)

---

## 1. Quick path (copy-paste)

```bash
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1/teaching_pipeline_v2

./setup_env.sh                 # once: builds .venv and tests it
.venv/bin/python check_env.py     # should end with "All checks passed"
.venv/bin/python tests/test_pipeline.py
.venv/bin/python tests/test_device.py
.venv/bin/python benchmark.py     # optional, ~1-2 min
./run.sh --preset quick        # the real run (results/quick/)
```

If every step prints OK, open `results/quick/` and look at `fwhm_vs_channels.png` and `summary.csv`.

`./run.sh ...` is just a shortcut for `.venv/bin/python run_all.py ...`. You never need to activate the environment, and you can ignore any conda `(python)` prompt prefix. `run.sh` uses `.venv/bin/python` directly.

---

## 2. Step 0 - what you need

| Need | Details |
|---|---|
| macOS on Apple Silicon | M1/M2/M3. The GPU is used through PyTorch **MPS**. |
| Native Python 3.10 - 3.13 | From Homebrew (`brew install python@3.12`) or python.org. **Not** conda/miniforge, **not** an Intel (Rosetta) Python. |
| Internet | Only for the one-time `pip install` (numpy, scipy, matplotlib, torch, pytest). |
| Disk | About 1 GB for `.venv` (PyTorch is large) plus a few hundred MB for cached data. |

To check which Python you have:

```bash
which -a python3.12 python3.11 python3
python3 -c "import platform,sys; print(sys.version); print(platform.machine())"   # must print arm64
```

If it prints `x86_64` on an M1, that Python runs under Rosetta. Install a native one and re-run `setup_env.sh`.

---

## 3. Step 1 - create the clean environment

```bash
./setup_env.sh
```

What it does:

1. Looks for a **native, non-conda** Python 3.10 - 3.13 (Homebrew, python.org, then PATH).
2. Creates `.venv/` in this folder.
3. `pip install -r requirements.txt` (numpy, scipy, matplotlib, torch, pytest).
4. Runs `check_env.py` on the new environment.

Useful variants:

```bash
./setup_env.sh --recreate        # delete .venv and rebuild from scratch
SKIP_INSTALL=1 ./setup_env.sh    # only create/check the venv, skip pip (offline)
```

Why this exists: in a conda environment, NumPy and PyTorch can each ship their own OpenMP library (`libomp`). Loading two copies causes `OMP: Error #15` or a segfault. The `KMP_DUPLICATE_LIB_OK` workaround only hides the message and is not safe. A pip-only `.venv` has one copy and avoids the problem.

If the script says it cannot find a suitable Python, install one (`brew install python@3.12`) and run it again.

---

## 4. Step 2 - check the environment

```bash
.venv/bin/python check_env.py
```

Expected output (abridged):

```
Python      : 3.12.x at .../.venv/bin/python
Machine     : arm64 ...
Conda env   : no
OpenMP libs : 1 found
...
All checks passed
```

What it checks:

| Check | Pass condition |
|---|---|
| Native architecture | `arm64` on Apple Silicon (not Rosetta). |
| Not conda | No `CONDA_PREFIX`. |
| OpenMP libraries | At most one copy in `site-packages`. |
| NumPy + PyTorch import together | No abort. |
| A small matrix multiply on CPU | Runs. |
| MPS | `MPS: available` (Mac), or CUDA on NVIDIA. |

If a check fails, the script prints what to do. Read its last lines first (see [Troubleshooting](#16-troubleshooting)).

---

## 5. Step 3 - run the tests

The tests are small self-checks of the code, not of the paper's numbers. They finish in about a minute altogether.

```bash
.venv/bin/python tests/test_pipeline.py     # physics + data + model + metric sanity (5 tests)
.venv/bin/python tests/test_device.py       # device selection, MPS/CPU consistency (4 tests)
.venv/bin/python tests/test_env.py          # environment-detection helpers (3 tests)
.venv/bin/python tests/test_variants.py     # soft labels, cosine LR, best-epoch (6 tests)
```

Or all of them with pytest:

```bash
.venv/bin/python -m pytest -q tests
```

Each test prints `ok <name>`. A segfault or `OMP: Error #15` here means the environment is wrong (go back to Step 1).

What the tests cover:

| File | Examples |
|---|---|
| `test_pipeline.py` | Light never exceeds the number of generated photons, the summing matrices conserve total signal, the model outputs one probability per grid point, FWHM of a known Gaussian is recovered. |
| `test_device.py` | `--device auto` picks the right device, falls back to CPU with a warning, CPU and MPS give the same numbers within tolerance. |
| `test_env.py` | Rosetta/conda/duplicate-OpenMP detection works. |
| `test_variants.py` | Soft-label matrix rows sum to 1, cosine schedule decays, `select_best='offgrid'` without data raises an error. |

---

## 6. Step 4 - benchmark CPU vs GPU (optional)

```bash
.venv/bin/python benchmark.py
.venv/bin/python benchmark.py --events 20000 --epochs 3    # custom size
```

It prints how long the simulation and a few training epochs take on CPU and on the GPU. On your M1 it measured:

| Part | CPU | M1 GPU (MPS) |
|---|---|---|
| Training (per fixed batch of work) | 1.0x | about 4.7x faster |
| Optical simulation | 9.6 s (NumPy) -> 5.1 s (torch CPU) | 4.0 s |

Use it to confirm the GPU is really being used. If MPS is slower than CPU on your machine, run with `--device cpu`.

---

## 7. Step 5 - first real run (`quick`)

```bash
./run.sh --preset quick
```

Equivalent: `.venv/bin/python run_all.py --preset quick`.

What `quick` means:

| Setting | Value |
|---|---|
| Class grid | 2 mm pitch (400 classes, 20 x 20) |
| Events | 40 per training position, 60 per test position |
| Filters | 32 |
| Epochs | 15 |
| Channel counts | 32, 16, 8, 4 |

What you will see, in order:

1. One line such as `Device: ... | simulation backend: torch`.
2. `Simulating training data ...` and `Simulating test data ...`. These are cached (see [Caching](#15-caching-and-re-running)).
3. For each channel count: a training log, one line per epoch with train loss, val loss, val mean error (mm), off-grid RMS (mm), and the learning rate.
4. A table with FWHM, bias and RMS per channel count.
5. A list of the files written to `results/quick/`.

Typical numbers (15 epochs, from your M1 run): RMS error about 0.51 / 0.44 / 0.60 / 1.71 mm for 32 / 16 / 8 / 4 channels. They vary a little from run to run.

Useful variations:

```bash
./run.sh --preset quick --channels 32 16            # only some channel counts
./run.sh --preset quick --epochs 40                 # train longer
./run.sh --preset quick --seed 5 --out results/q_seed5
./run.sh --preset quick --device cpu                # force CPU (to compare)
./run.sh --preset quick --save-predictions          # keep raw predictions for plot_errors.py
```

---

## 8. Reading the results

Open `results/quick/`.

| File | What it shows |
|---|---|
| `summary.csv` / `summary.json` | One row per channel count: median FWHM (mm), mean bias (mm), RMS error (mm). |
| `fwhm_vs_channels.png` | FWHM versus number of readout channels (the paper's central result). |
| `bias_vs_channels.png` | Bias versus number of channels. |
| `light_patterns.png` | Example SiPM light patterns from the simulation, to check the physics looks right. |
| `maps_32ch.png`, `maps_16ch.png`, ... | Maps over the crystal of FWHM and bias per position (like the paper's Figs. 9 - 10). |
| `per_position_XXch.npz` | Raw per-position numbers behind the maps. |
| `pred_XXch.npz` | Only with `--save-predictions`: raw predictions for the diagnostics script. |

Which number to trust:

- **RMS error** is the most reliable metric for the 2 mm presets. It combines spread and bias and cannot be fooled by quantization.
- **FWHM** can read about 0 mm at some positions on the coarse grid, because predictions snap onto calibration points. This is a known artifact of the teaching setup. It is not a real resolution of 0 mm.
- **Trend** is what matters: error should rise slowly from 32 down to about 8 channels, then jump at 4. This matches the paper's conclusion that heavy multiplexing is possible down to roughly 8 - 16 channels.

Absolute numbers are not the paper's. The simulator is simpler than GATE, there are fewer events per position, and the grid is coarser.

---

## 9. Step 6 - longer runs (`medium`, `paper`)

```bash
./run.sh --preset medium --out results/medium
./run.sh --preset paper  --out results/paper
```

| Preset | Grid | Events (train / test per pos.) | Filters | Epochs | Channels | Rough time on M1 |
|---|---|---|---|---|---|---|
| `quick` | 2 mm | 40 / 60 | 32 | 15 | 32, 16, 8, 4 | a few minutes |
| `medium` | 2 mm | 100 / 100 | 64 | 20 | all 8 (32 ... 4) | tens of minutes |
| `paper` | 1 mm (1600 classes) | 300 / 200 | 200 | 20 | all 8 | hours; **not run yet** |

Notes:

- `paper` follows the paper's grid and filter count, but still uses far fewer events than the paper (they used about 550 - 950 per position). It has **not been run or checked** by me, so treat its time and results as unknown.
- 16 GB of unified memory should be enough for `medium`. For `paper`, watch memory. If it swaps, lower `--batch-size` or run fewer channel counts at a time (`--channels 32 16`).
- Leave the Mac plugged in and avoid sleep. Use `caffeinate -i ./run.sh --preset medium` to keep it awake.
- For long runs, write to a log: `./run.sh --preset medium 2>&1 | tee results/medium.log`.

---

## 10. Step 7 - optional training variants

These three switches are **not from the paper**. They are my additions to reduce the interpolation error between 2 mm calibration points, and all are **off by default**.

| Switch | What it does |
|---|---|
| `--lr-schedule cosine` | Learning rate decays smoothly to zero over the epochs instead of staying constant. |
| `--soft-label-sigma 2.0` | Training targets are Gaussian blobs over neighbouring grid points (sigma in mm) instead of one-hot. The model learns that nearby positions are similar. |
| `--select-best offgrid` | Keeps the epoch with the lowest error on separate validation hits drawn at random positions, not only at grid points. |

Examples:

```bash
./run.sh --preset quick --epochs 40 --lr-schedule cosine
./run.sh --preset quick --epochs 40 --soft-label-sigma 2.0
./run.sh --preset quick --epochs 40 --select-best offgrid
./run.sh --preset quick --epochs 40 --lr-schedule cosine --soft-label-sigma 2.0 --select-best offgrid
```

Results from earlier 40-epoch, 3-seed runs (RMS in mm):

| Variant | 32 ch | 16 ch | 8 ch | 4 ch |
|---|---|---|---|---|
| Baseline | 0.61 | 0.50 | 0.56 | 1.21 |
| Cosine | 0.49 | 0.37 | 0.50 | 1.45 |
| Soft labels | 0.33 | 0.34 | 0.48 | 1.16 |
| Best-epoch | 0.46 | 0.44 | 0.56 | 1.19 |
| All combined | 0.28 | 0.30 | 0.47 | 1.41 |

Do not mix these variants into any comparison with the paper without saying so. The 4-channel result had not converged at 40 epochs.

---

## 11. Step 8 - several seeds and comparing variants

One run is noisy. `run_seeds.py` repeats runs with different seeds and reports mean +/- spread.

Everything after a lone `--` is passed to `run_all.py`.

**A. Baseline, 3 seeds**

```bash
.venv/bin/python run_seeds.py --seeds 1 2 3 --out results/q40 -- --preset quick --epochs 40
```

Output: `results/q40/seed_1 ... seed_3/` plus `results/q40/seeds_summary.csv` and `seeds_summary.png`.

**B. Compare variants (same seeds)**

```bash
.venv/bin/python run_seeds.py --variants base cosine soft best all --seeds 1 2 3 \
    --out results/cmp -- --preset quick --epochs 40
```

Variants: `base` (no extras), `cosine`, `soft`, `best`, `all` (the three combined). Each variant gets its own folder, and the simulated data is cached per seed, so it is generated only once. Output: `results/cmp/comparison.csv` and `comparison.png`.

**C. Only rebuild the table/plot from folders that already exist**

```bash
.venv/bin/python run_seeds.py --compare results/cmp/base results/cmp/cosine
```

Cost: runs scale with the number of variants x seeds x epochs, so a 5-variant, 3-seed, 40-epoch comparison is a long job (plan on the order of an hour or more on the M1; the first run did this in the sandbox). Start with `--seeds 1 2` if you want a faster check.

---

## 12. Step 9 - error diagnostics

First save raw predictions, then plot:

```bash
./run.sh --preset quick --epochs 40 --save-predictions --out results/diag
.venv/bin/python plot_errors.py results/diag
.venv/bin/python plot_errors.py results/diag --channels 32 8 --pitch 2     # choose channels / grid pitch
```

`--pitch` must match the preset (2 for `quick` and `medium`, 1 for `paper`).

The figure (`error_diagnostics.png` in that folder) has:

- **Predicted vs true x position**: a staircase means predictions are snapping onto calibration points.
- **x-error histograms**: should be centred at 0 and narrow.
- **Snapping index**: the share of predictions within 0.15 mm of a calibration coordinate. About 0.15 is the "evenly spread" level for a 2 mm grid. Much higher means heavy snapping.

An example is in `example_results/error_diagnostics_40ep.png`.

---

## 13. All command-line options

### `run_all.py` (main pipeline)

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--preset` | `quick`, `medium`, `paper` | `quick` | Size of the experiment. |
| `--channels` | e.g. `32 16 8 4` | per preset | Override the list of readout-channel counts. |
| `--epochs` | integer | per preset | Training epochs. |
| `--seed` | integer | from the preset (`Config.seed`) | Random seed for simulation, split and initialisation. |
| `--device` | `auto`, `cpu`, `mps`, `cuda` | `auto` | Compute device. `auto` = Apple GPU > NVIDIA GPU > CPU. |
| `--sim-backend` | `auto`, `numpy`, `torch` | `auto` | Simulator: torch (on the device) or NumPy (CPU). `auto` = torch on a GPU, NumPy on CPU. |
| `--batch-size` | integer | 128 | Larger batches use the GPU better but need more memory. |
| `--lr-schedule` | `const`, `cosine` | `const` | Learning-rate schedule. |
| `--soft-label-sigma` | mm (float) | 0 (one-hot) | Gaussian soft labels. |
| `--select-best` | `none`, `offgrid` | `none` | Keep the best epoch on off-grid validation. |
| `--save-predictions` | flag | off | Save raw predictions for `plot_errors.py`. |
| `--out` | folder | `results/<preset>` | Output directory. |

### Other scripts

| Script | Purpose | Key options |
|---|---|---|
| `check_env.py` | Diagnose the Python environment. | none |
| `setup_env.sh` | Build/test `.venv`. | `--recreate`; env var `SKIP_INSTALL=1` |
| `run.sh` | Run `run_all.py` with `.venv`. | same as `run_all.py` |
| `run_seeds.py` | Multi-seed runs and variant comparison. | `--seeds`, `--variants`, `--out`, `--compare`, `--` passthrough |
| `plot_errors.py` | Error diagnostics from saved predictions. | `folder`, `--channels`, `--pitch` |
| `benchmark.py` | CPU vs GPU timing. | `--events`, `--epochs` |

Help for any script: `.venv/bin/python <script>.py --help`.

---

## 14. Output folder layout

```
teaching_pipeline_v2/
├── results/
│   ├── cache/                      # simulated datasets (.npz), safe to delete
│   ├── quick/                      # one run
│   │   ├── summary.csv / .json
│   │   ├── fwhm_vs_channels.png, bias_vs_channels.png
│   │   ├── light_patterns.png, maps_XXch.png
│   │   ├── per_position_XXch.npz
│   │   └── pred_XXch.npz           # only with --save-predictions
│   ├── q40/                        # run_seeds.py: seed_1/, seed_2/, ..., seeds_summary.csv/png
│   └── cmp/                        # run_seeds.py --variants: base/, cosine/, ..., comparison.csv/png
└── example_results/                # reference outputs shipped with the code
```

---

## 15. Caching and re-running

Simulated data is expensive, so it is cached in `results/cache/`.

- The cache name includes the preset settings, seed and backend (ending `_numpy` or `_torch`). Changing the backend or seed creates a new file, and changing the preset also does.
- The same command run again reuses the cache, so a second run skips simulation.
- Off-grid validation data is cached as `valoff_*.npz`.
- To force fresh simulation: `rm -rf results/cache`.
- Disk use: the quick cache is small; the medium and paper caches can be hundreds of MB.

Note that `numpy` and `torch` backends use different random number streams, so their data differ slightly, and results agree statistically, not bit for bit.

---

## 16. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OMP: Error #15 ... libomp already initialized` | Two OpenMP copies (usually conda). | Use the `.venv`: `./setup_env.sh --recreate`, then `./run.sh ...`. Do not rely on `KMP_DUPLICATE_LIB_OK`. |
| `zsh: segmentation fault` right after `import torch` / numpy | Same duplicate-OpenMP problem. | Same fix. |
| `check_env.py` says "Rosetta" / `x86_64` | Intel Python on Apple Silicon. | Install native Python (`brew install python@3.12`), then `./setup_env.sh --recreate`. |
| `No .venv yet: run ./setup_env.sh first.` | `.venv` missing. | `./setup_env.sh`. |
| `permission denied: ./setup_env.sh` | Executable bit lost (e.g. copied via a zip or a Windows share). | `chmod +x *.sh`, or prefix the command with `bash`. |
| `Device: cpu` although you have an M1 | PyTorch has no MPS, or the Python is not native. | `.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` must print `True`. Rebuild the venv with native Python. |
| `NotImplementedError: ... not currently implemented for the MPS device` | An operation that MPS lacks. | Already handled: `run_all.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. If you call the modules yourself, set it first. |
| Training is slow on CPU | Denormal numbers (already flushed in v2) or too few threads. | Use `--device mps`. Close heavy apps. |
| Out of memory / Mac becomes slow | Large preset, large batch. | Lower `--batch-size`, use fewer `--channels` per run. |
| Loss is `nan` | Rare; high LR or bad data. | Re-run with another `--seed`; report it to me with the log. |
| FWHM is about 0.00 at many positions | Snapping on the 2 mm grid. | Look at RMS instead; run `plot_errors.py`; try `--soft-label-sigma 2.0`. |
| Different numbers each run | Different seed or backend. | Fix `--seed`, and use `run_seeds.py` for mean +/- spread. |
| A run stuck on "32-channel readout" for minutes | The old conda environment hang. | Ctrl-C, use the `.venv`. |

Stopping a run: press **Ctrl-C** once. Do not use Ctrl-Z (it only suspends the process and keeps memory). If you did, run `fg` and then Ctrl-C, or `kill %1`.

When asking for help, send the output of:

```bash
.venv/bin/python check_env.py
.venv/bin/python run_all.py --preset quick --channels 4 --epochs 2    # a very fast smoke run
```

---

## 17. How long things take

All measured or estimated for an M1 with 16 GB. Treat them as rough guides. The benchmark script gives you your own numbers.

| Job | Approximate time |
|---|---|
| `setup_env.sh` (download PyTorch) | a few minutes, depends on the network |
| `check_env.py` | seconds |
| All tests | about a minute |
| `benchmark.py` | 1 - 2 min |
| Smoke run (`--channels 4 --epochs 2`) | under a minute |
| `--preset quick` (15 epochs, 4 channel counts) | a few minutes on the GPU; about 2 min on a fast CPU in the sandbox |
| `--preset quick --epochs 40` | roughly 2 - 3x the 15-epoch time |
| `--preset medium` | tens of minutes (about 25 min on 2 CPU cores for v1; the GPU should be faster) |
| `run_seeds.py` comparison, 5 variants x 3 seeds x 40 epochs | about an hour or more |
| `--preset paper` | hours; not measured |

---

## 18. Suggested order for a first session

1. `./setup_env.sh`
2. `.venv/bin/python check_env.py`
3. `.venv/bin/python tests/test_pipeline.py` and `tests/test_device.py`
4. `.venv/bin/python benchmark.py`
5. `./run.sh --preset quick --channels 4 --epochs 2` (smoke run)
6. `./run.sh --preset quick` and open `results/quick/`
7. `./run.sh --preset quick --epochs 40 --save-predictions --out results/diag`, then `plot_errors.py results/diag`
8. `run_seeds.py --variants base all --seeds 1 2 --out results/cmp -- --preset quick --epochs 40`
9. Only then `--preset medium`, and last `--preset paper`.

Where to read more: `README.md` (design and physics assumptions), `CHANGELOG.md` (what changed from v1 and why).
