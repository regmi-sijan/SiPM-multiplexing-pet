"""Lightweight environment checks that need no numpy/torch, so they can run BEFORE those are imported.

Used by run_all.py (start-up warnings) and check_env.py (full diagnostic).
"""
import glob
import os
import platform
import subprocess
import sys

OMP_PATTERNS = ("libomp*.dylib", "libiomp5*.dylib", "libgomp*")


def running_under_rosetta():
    """True if this is an Intel (x86_64) Python running on an Apple-Silicon Mac."""
    if sys.platform != "darwin" or platform.machine() != "x86_64":
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "1"
    except Exception:
        return False


def find_openmp_libs(prefixes=None):
    """OpenMP runtime libraries installed in this Python environment (site-packages and <prefix>/lib).

    Two different copies (e.g. conda's libiomp5 plus PyTorch's libomp) are the classic cause of
    'OMP: Error #15' and of segmentation faults when NumPy and PyTorch run in the same process.
    """
    if prefixes is None:
        prefixes = {sys.prefix, sys.base_prefix}
    roots = set()
    for pre in prefixes:
        roots.add(os.path.join(pre, "lib"))
        roots.update(glob.glob(os.path.join(pre, "lib", "python*", "*-packages")) + glob.glob(os.path.join(pre, "local", "lib", "python*", "*-packages")))
    found = set()
    for root in roots:
        for pat in OMP_PATTERNS:
            found.update(glob.glob(os.path.join(root, pat)))
            found.update(glob.glob(os.path.join(root, "*", "lib", pat)))          # e.g. torch/lib/libomp.dylib
            found.update(glob.glob(os.path.join(root, "*", ".dylibs", pat)))      # wheels that bundle their own
    return sorted(found)


def is_conda():
    return bool(os.environ.get("CONDA_PREFIX")) or os.path.isdir(os.path.join(sys.prefix, "conda-meta"))


def startup_warnings(verbose=True):
    """Print actionable warnings; returns the list of messages."""
    msgs = []
    if running_under_rosetta():
        msgs.append("This Python is an Intel (x86_64) build running under Rosetta on an Apple-Silicon Mac. "
                    "PyTorch cannot use the GPU (MPS) from it and is much slower.")
    libs = find_openmp_libs()
    names = {os.path.basename(p).split(".")[0] for p in libs}
    if len(libs) >= 2 and len(names) >= 2:
        msgs.append("This environment contains several different OpenMP runtimes (" + ", ".join(sorted(names)) +
                    "). Mixing them can crash NumPy+PyTorch programs.")
    if is_conda() and sys.platform == "darwin":
        msgs.append("You are in a conda environment on macOS, where duplicate OpenMP libraries are common.")
    if msgs and verbose:
        print("ENVIRONMENT WARNING:")
        for m in msgs:
            print("  - " + m)
        print("  -> If you see 'OMP: Error #15' or a segmentation fault, run ./setup_env.sh once "
              "(builds a clean .venv), then use ./run.sh instead of python. "
              "Run `python check_env.py` for a full diagnosis.\n")
    return msgs


def classify(returncode, stderr=""):
    """Turn a subprocess result into a human-readable verdict ('' if it succeeded)."""
    text = stderr or ""
    if "OMP: Error #15" in text:
        return "OpenMP clash (two copies of libomp loaded)"
    if returncode == 0:
        return ""
    if "No module named" in text:
        mod = text.split("No module named")[-1].strip().split()[0].strip("'\"")
        return f"missing package '{mod}' (run ./setup_env.sh or: pip install -r requirements.txt)"
    if returncode in (-11, 139):
        return "segmentation fault"
    if returncode in (-6, 134):
        return "aborted (often an OpenMP/native library clash)"
    if returncode < 0:
        return f"killed by signal {-returncode}"
    return f"exited with code {returncode}"
