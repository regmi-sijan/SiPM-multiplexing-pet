# Command cheat sheet

Every command for both code versions, in one place. Copy and paste them into your Mac terminal.

- **v1** = `teaching_pipeline` (CPU only)
- **v2** = `teaching_pipeline_v2` (Apple GPU, RMS metric, optional training switches, multi-seed tools)

Commands for the two versions are the same unless noted. Run all of them from inside the version's own folder.

Replace `<V>` below with `teaching_pipeline` or `teaching_pipeline_v2` where needed.

---

## 1. Go to the folder

```bash
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1/teaching_pipeline       # v1
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1/teaching_pipeline_v2    # v2
```

## 2. One-time environment setup (both versions)

```bash
./setup_env.sh                    # build .venv (native Python 3.10-3.13, not conda) and test it
./setup_env.sh --recreate         # delete .venv and rebuild it
SKIP_INSTALL=1 ./setup_env.sh     # create/check the venv but skip pip install
.venv/bin/python check_env.py        # diagnose the environment; expect "All checks passed"
```

Check your Python before setup:

```bash
which -a python3.12 python3.11 python3
python3 -c "import platform; print(platform.machine())"      # must print arm64 on an M1
brew install python@3.12                                      # if you need a native Python
```

Check the GPU (v2):

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"    # expect True
```

## 3. Tests

```bash
.venv/bin/python tests/test_pipeline.py      # v1 and v2: pipeline sanity (5 tests)
.venv/bin/python tests/test_env.py           # v1 and v2: environment helpers (3 tests)
.venv/bin/python tests/test_device.py        # v2 only: device selection, MPS vs CPU (4 tests)
.venv/bin/python tests/test_variants.py      # v2 only: soft labels, cosine LR, best epoch (6 tests)
.venv/bin/python -m pytest -q tests          # all tests at once
```

## 4. Benchmark (v2 only)

```bash
.venv/bin/python benchmark.py                          # CPU vs GPU timing
.venv/bin/python benchmark.py --events 20000 --epochs 3
```

## 5. Main runs

`./run.sh ARGS` is a shortcut for `.venv/bin/python run_all.py ARGS`.

```bash
./run.sh --preset quick --channels 4 --epochs 2 --out results/smoke    # smoke run, under a minute
./run.sh --preset quick                                                # a few minutes
./run.sh --preset medium                                               # tens of minutes
./run.sh --preset paper                                                # hours; not run by me
```

Common options (v1 and v2):

```bash
./run.sh --preset quick --channels 32 16            # only these channel counts
./run.sh --preset quick --epochs 40                 # more epochs
./run.sh --preset quick --seed 5                    # different random seed
./run.sh --preset quick --out results/my_run        # choose the output folder
```

Options that exist **only in v2**:

```bash
./run.sh --preset quick --device auto               # default: Apple GPU > NVIDIA GPU > CPU
./run.sh --preset quick --device mps                # force the Apple GPU
./run.sh --preset quick --device cpu                # force the CPU
./run.sh --preset quick --sim-backend torch         # simulate with torch (numpy or auto also valid)
./run.sh --preset quick --batch-size 64             # default 128; lower it if memory is tight
./run.sh --preset quick --save-predictions          # keep raw predictions for plot_errors.py
```

Optional training switches, **v2 only, not from the paper**:

```bash
./run.sh --preset quick --epochs 40 --lr-schedule cosine
./run.sh --preset quick --epochs 40 --soft-label-sigma 2.0       # use about 1.0 for the 1 mm paper preset
./run.sh --preset quick --epochs 40 --select-best offgrid
./run.sh --preset quick --epochs 40 --lr-schedule cosine --soft-label-sigma 2.0 --select-best offgrid
```

Long runs:

```bash
caffeinate -i ./run.sh --preset medium                                  # keep the Mac awake
./run.sh --preset medium 2>&1 | tee results/medium.log                  # save a log
caffeinate -i ./run.sh --preset paper --out results/paper 2>&1 | tee results/paper.log
```

Time one epoch of the paper preset before committing to a full run:

```bash
./run.sh --preset paper --channels 32 --epochs 1 --out results/paper_probe
```

Stop a run with **Ctrl-C** once. Do not use Ctrl-Z.

## 6. Multi-seed runs and variant comparison (v2 only)

Everything after a lone `--` goes to `run_all.py`.

```bash
# baseline, 3 seeds
.venv/bin/python run_seeds.py --seeds 1 2 3 --out results/q40 -- --preset quick --epochs 40

# compare variants (base, cosine, soft, best, all) over the same seeds
.venv/bin/python run_seeds.py --variants base cosine soft best all --seeds 1 2 3 \
    --out results/cmp -- --preset quick --epochs 40

# quicker check
.venv/bin/python run_seeds.py --variants base all --seeds 1 2 --out results/cmp -- --preset quick --epochs 40

# rebuild the table and plot from folders that already exist
.venv/bin/python run_seeds.py --compare results/cmp/base results/cmp/cosine
```

## 7. Error diagnostics (v2 only)

```bash
./run.sh --preset quick --epochs 40 --save-predictions --out results/diag
.venv/bin/python plot_errors.py results/diag
.venv/bin/python plot_errors.py results/diag --channels 32 8 --pitch 2      # --pitch 1 for the paper preset
```

## 8. Look at the results

```bash
open results/quick                         # open the folder in Finder
open results/quick/fwhm_vs_channels.png    # open one figure
cat results/quick/summary.csv              # print the table
```

Clean FWHM-vs-channels line chart (v2 only; `run_all.py` now also writes `fwhm_line.png` automatically):

```bash
.venv/bin/python plot_summary.py results/medium                    # -> results/medium/fwhm_line.png
.venv/bin/python plot_summary.py results/medium --no-paper         # without the dashed paper reference
.venv/bin/python plot_summary.py results/a results/b --out cmp.png # several runs on one chart
```

## 9. Help for any script

```bash
.venv/bin/python run_all.py --help
.venv/bin/python run_seeds.py --help       # v2
.venv/bin/python plot_errors.py --help     # v2
.venv/bin/python benchmark.py --help       # v2
```

## 10. Housekeeping

```bash
rm -rf results/cache                       # force fresh simulated data on the next run
rm -rf results/smoke                       # remove a test run
du -sh results/* .venv                     # see how much disk space things use
```

## 11. If something goes wrong

| Problem | Command |
|---|---|
| `OMP: Error #15` or segmentation fault | `./setup_env.sh --recreate`, then always use `./run.sh ...` or `.venv/bin/python ...` |
| `No .venv yet` | `./setup_env.sh` |
| `permission denied` on a `.sh` file | `chmod +x *.sh` (or prefix the command with `bash`) |
| Not sure what is wrong | `.venv/bin/python check_env.py`, then send me its output |
| Suspended a run with Ctrl-Z | `fg`, then Ctrl-C (or `kill %1`) |

## 12. Suggested order

```bash
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1/teaching_pipeline_v2
./setup_env.sh
.venv/bin/python check_env.py
.venv/bin/python tests/test_pipeline.py
.venv/bin/python tests/test_device.py
.venv/bin/python benchmark.py
./run.sh --preset quick --channels 4 --epochs 2 --out results/smoke
./run.sh --preset quick
```

For more detail see `teaching_pipeline/GUIDE.md` (v1) and `teaching_pipeline_v2/HOW_TO_RUN.md` (v2).

---

This code reproduces Subedi, Cherry, Qiang & Peng (2025), *Radiation Measurements* **182**, 107399,
<https://doi.org/10.1016/j.radmeas.2025.107399>. Please cite that article; see
[`README.md`](README.md#citing).
