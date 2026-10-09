#!/usr/bin/env python3
"""Diagnose the Python environment for this project. Each risky step runs in its OWN subprocess, so a crash
(segmentation fault, OpenMP abort) is reported instead of killing this script.

    python check_env.py            # ~20 s
    python check_env.py --full     # also runs the project's own test files (~1-2 min)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import subprocess
import sys
import platform

from src.envcheck import find_openmp_libs, running_under_rosetta, is_conda, classify

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("numpy", "import numpy as np; a = np.random.rand(300, 300); (a @ a).sum()"),
    ("scipy + matplotlib", "import scipy.optimize, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"),
    ("import torch", "import torch; print(torch.__version__)"),
    ("torch CPU operation", "import torch; x = torch.rand(600, 600); (x @ x).sum().item()"),
    ("numpy + scipy + torch together",
     "import numpy as np, torch, scipy.optimize\n"
     "a = np.random.rand(400, 400); (a @ a).sum()\n"
     "t = torch.from_numpy(a); (t @ t).sum().item()\n"
     "scipy.optimize.curve_fit(lambda x, m: m * x, np.arange(10.), 2 * np.arange(10.))"),
    ("torch CNN training (CPU)",
     "import torch, torch.nn as nn\n"
     "m = nn.Sequential(nn.Conv2d(1, 8, (1, 3), padding=(0, 1)), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(), nn.Linear(256, 50))\n"
     "o = torch.optim.SGD(m.parameters(), lr=.01, momentum=.9)\n"
     "x = torch.rand(64, 1, 1, 32); y = torch.randint(0, 50, (64,))\n"
     "for _ in range(5):\n    o.zero_grad(); nn.functional.cross_entropy(m(x), y).backward(); o.step()"),
    ("Apple GPU (MPS) operation",
     "import sys, torch\n"
     "if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()): sys.exit(3)\n"
     "x = torch.rand(512, 512, device='mps'); (x @ x).sum().item()\n"
     "m = torch.nn.Conv2d(1, 8, (1, 3), padding=(0, 1)).to('mps'); m(torch.rand(32, 1, 1, 32, device='mps')).sum().item()"),
]
THREAD_FIXES = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def run(code, extra_env=None, timeout=180, args=None):
    env = dict(os.environ, **(extra_env or {}))
    cmd = [sys.executable] + (args if args else ["-c", code])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=HERE)
        return p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main():
    full = "--full" in sys.argv
    print(f"Python      : {sys.version.split()[0]} at {sys.executable}")
    print(f"Machine     : {platform.machine()} on {platform.platform()}")
    if running_under_rosetta():
        print("              !! Intel Python under Rosetta on an Apple-Silicon Mac: no GPU (MPS), slower.")
    print(f"Conda env   : {'yes' if is_conda() else 'no'}")
    libs = find_openmp_libs()
    print(f"OpenMP libs : {len(libs)} found" + ("" if not libs else ""))
    for p in libs:
        print(f"              {p}")

    print("\nStep                               result")
    failed, clash = [], False
    for name, code in STEPS:
        rc, err = run(code)
        if rc == 3 and "MPS" in name:
            print(f"{name:34s} SKIP (no Apple GPU visible to this Python)")
            continue
        verdict = classify(rc, err)
        if verdict:
            failed.append(name)
            clash = clash or "OpenMP" in verdict or "segmentation" in verdict or "aborted" in verdict
            print(f"{name:34s} FAIL: {verdict}")
        else:
            print(f"{name:34s} PASS")

    if full:
        for name, args in [("project tests", ["tests/test_pipeline.py"]), ("device tests", ["tests/test_device.py"]), ("variant tests", ["tests/test_variants.py"])]:
            if os.path.exists(os.path.join(HERE, args[0])):
                rc, err = run(None, args=args, timeout=600)
                verdict = classify(rc, err)
                print(f"{name:34s} {'FAIL: ' + verdict if verdict else 'PASS'}")
                if verdict:
                    failed.append(name)

    print()
    if not failed:
        print("VERDICT: environment looks healthy. Run: python run_all.py --preset quick")
        return 0
    print("VERDICT: problems found in: " + ", ".join(failed))
    # does a single-threaded setup avoid the crash?
    combo = dict(STEPS)["numpy + scipy + torch together"]
    rc, err = run(combo, extra_env=THREAD_FIXES)
    if not classify(rc, err) and "numpy + scipy + torch together" in failed:
        print("  * The crash goes away with single-threaded math libraries. Temporary workaround:\n"
              "      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1\n"
              "    (slower, and the environment itself is still fragile.)")
    print("  * Recommended fix: a clean environment with pip only (one OpenMP runtime):\n"
          "      ./setup_env.sh        then        ./run.sh --preset quick")
    return 1


if __name__ == "__main__":
    sys.exit(main())
