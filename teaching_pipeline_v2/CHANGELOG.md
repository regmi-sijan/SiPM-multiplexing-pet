# Change log

Two folders exist in `Project_1/`:

* `teaching_pipeline/` = **v1**: CPU-only teaching version (plus later environment-robustness tooling).
* `teaching_pipeline_v2/` = **v2**: v1 plus GPU support, a faster simulation, optional training variants and
  experiment tools. Default behaviour (all options off) gives the same results as v1.

What was NOT changed anywhere (the science): detector geometry and SiPM numbering, optical-physics model, the
multiplexing groupings of paper Fig. 8, the CNN architecture, softmax centre-of-mass position estimate, the offset
test grid, and the FWHM / bias / RMS evaluation. A regression check confirmed that v2 with all options off gives
numerically identical results to the earlier baseline.

## Timeline

| # | Change | Why | Where | Type |
|---|---|---|---|---|
| 0 | **v1 created**: NumPy optical Monte Carlo, CNN (PyTorch, CPU), grids, 8 multiplexing schemes, evaluation, plots, presets `quick`/`medium`/`paper`, README, 5 sanity tests | the teaching reproduction | whole project | baseline |
| 1 | Added **RMS error** (bias and spread combined) to the metrics and plots | with the 2 mm class grid the network can "snap" to grid points, making FWHM near 0 mm at some positions | `src/evaluate.py`, `run_all.py`, `src/plots.py` | metric (in v1) |
| 2 | Set `KMP_DUPLICATE_LIB_OK=TRUE` at the top of all entry points | macOS "OMP: Error #15" | `run_all.py`, tests, `src/__init__.py` | workaround (v1 and v2) |
| 3 | **v2 created** as a copy of v1 | keep v1 untouched while adding GPU support | `teaching_pipeline_v2/` | new folder |
| 4 | **GPU support** (Apple MPS / NVIDIA CUDA / CPU): `--device auto|cpu|mps|cuda`; whole data set kept on the device; no per-step GPU sync | speed on the M1 | `src/device.py` (new), `src/train.py`, `run_all.py` | performance |
| 5 | **Torch version of the optical simulation** (same physics, can run on the GPU); `--sim-backend auto|numpy|torch`; statistically equal to NumPy (tested) | simulation was the other slow part | `src/optics_mc_torch.py` (new), `src/optics_mc.py`, `src/dataset.py` | performance |
| 6 | **Denormal flush** on CPU training | CPU epochs slowed down as weights settled (81 s to 49 s for 14 epochs, identical accuracy) | `src/train.py` | performance |
| 7 | **Environment tooling**: `check_env.py` (each risky step in its own process), `setup_env.sh` (clean pip-only `.venv`), `run.sh`, start-up warning for Rosetta / duplicate OpenMP, `PYTORCH_ENABLE_MPS_FALLBACK=1` | the conda environment segfaulted with NumPy + PyTorch | `check_env.py`, `setup_env.sh`, `run.sh`, `src/envcheck.py`, `tests/test_env.py` | reliability (v1 and v2) |
| 8 | `benchmark.py`: times simulation and training on CPU vs GPU | measure the speedup on your Mac (measured: training 4.7x, simulation about 2.4x) | `benchmark.py` | tool |
| 9 | **Optional training variants** (off by default): `--lr-schedule cosine`, `--soft-label-sigma`, `--select-best offgrid` (+ off-grid validation set) | the baseline error was dominated by interpolation between grid points; on-grid validation was misleading | `src/config.py`, `src/train.py`, `src/dataset.py`, `run_all.py` | **method (not in the paper)** |
| 10 | **Experiment tools**: `run_seeds.py` (mean +/- std over seeds, variant comparison), `plot_errors.py` (snapping diagnostic), `--save-predictions`, `tests/test_variants.py` | single runs are noisy and hid the variant effects | new files, `run_all.py` | tool |
| 11 | README, example results (`example_results/`), this change log | documentation | | docs |

## Behaviour changes to be aware of (v2 versus v1)

1. **Default device and backend:** `run_all.py` now defaults to `--device auto` (GPU if available) and a torch simulation when
   a GPU is used. v1 always used CPU + NumPy. Random numbers therefore differ, but results agree statistically
   (your M1 run matched the CPU run to 0.01 to 0.02 mm).
2. **Cache file names** now end in `_numpy` or `_torch`; v1 caches are not reused in v2.
3. **API changes:** `train_model(..., device=..., offgrid=...)`; its `hist` now has 5 columns (train loss, val loss, val error,
   off-grid RMS, learning rate); `predict(..., device=...)`; `simulate_events(..., backend=..., device=...)`;
   `get_datasets(..., backend=..., device=...)`.
4. **New config fields** (`lr_schedule`, `soft_label_sigma_mm`, `select_best`, `val_offgrid_events`); the defaults reproduce the baseline.
5. The tests grew from 5 to 20 checks (`test_pipeline` 5, `test_device` 4, `test_env` 3, `test_variants` 6; with 3 of them
   shared with v1 where noted in the tests folder).

## Results history (quick preset unless noted; RMS error in mm)

| Run | 32 ch | 16 ch | 8 ch | 4 ch | Note |
|---|---|---|---|---|---|
| v1, 15 epochs (sandbox CPU) | 0.51 | 0.44 | 0.60 | 1.71 | first working run |
| v2, 15 epochs on your M1 GPU | 0.51 | 0.44 | 0.60 | 1.71 | GPU path matches CPU |
| v2, 40 epochs, baseline | 0.61 | 0.50 | 0.56 | 1.21 | longer training made 32/16 ch worse on test |
| v2, 40 epochs, soft labels (3 seeds) | 0.33 | 0.34 | 0.48 | 1.16 | interpolation error roughly halved |
| v2, 40 epochs, all variants (3 seeds) | 0.28 | 0.30 | 0.47 | 1.41 | trend closest to the paper; 4 ch under-trained |
| `medium` preset v1, 20 epochs, 8 channel counts | 0.63 | 0.51 | 0.60 | 1.14 | all eight counts in `teaching_pipeline/example_results/` |

## Known limitations / open items

* The Apple-GPU path was verified on your Mac (tests and benchmark passed), not in my environment.
* The 4-channel result is not converged at 40 epochs; try 100+ epochs.
* The paper preset (1 mm grid, 200 filters) has not been run; it may reduce the grid-interpolation effect.
* Soft labels, the learning-rate schedule and off-grid epoch selection are not part of the paper's method.
