"""Unit tests for the environment checks (no numpy/torch needed).  python tests/test_env.py"""
import os, sys, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import envcheck


def test_classify():
    assert envcheck.classify(0, "") == ""
    assert "segmentation" in envcheck.classify(-11)
    assert "segmentation" in envcheck.classify(139)
    assert "aborted" in envcheck.classify(-6)
    assert "OpenMP" in envcheck.classify(-6, "OMP: Error #15: Initializing libomp.dylib")
    assert "code 2" in envcheck.classify(2)
    assert "missing package 'numpy'" in envcheck.classify(1, "ModuleNotFoundError: No module named 'numpy'")


def test_find_openmp_libs_in_fake_environment():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "lib", "python3.12", "site-packages")
        os.makedirs(os.path.join(sp, "torch", "lib")); os.makedirs(os.path.join(d, "lib"), exist_ok=True)
        open(os.path.join(sp, "torch", "lib", "libomp.dylib"), "w").close()
        open(os.path.join(d, "lib", "libiomp5.dylib"), "w").close()
        found = envcheck.find_openmp_libs({d})
        names = {os.path.basename(p) for p in found}
        assert names == {"libomp.dylib", "libiomp5.dylib"}, names


def test_startup_warning_mentions_fix_for_duplicate_openmp(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "lib", "python3.12", "site-packages", "torch", "lib")
        os.makedirs(sp)
        open(os.path.join(sp, "libomp.dylib"), "w").close()
        open(os.path.join(d, "lib", "libiomp5.dylib"), "w").close()
        orig = envcheck.find_openmp_libs
        envcheck.find_openmp_libs = lambda prefixes=None: orig({d})
        try:
            msgs = envcheck.startup_warnings(verbose=False)
        finally:
            envcheck.find_openmp_libs = orig
        assert any("OpenMP" in m for m in msgs)


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_") and callable(fn) and fn.__module__ == "__main__":
            fn(); print("ok", n)
