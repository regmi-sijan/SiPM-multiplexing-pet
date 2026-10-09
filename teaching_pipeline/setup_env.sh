#!/usr/bin/env bash
# Build a clean, native Python environment (.venv, pip only) for this project and test it.
#   ./setup_env.sh              create .venv (reuse if it already works)
#   ./setup_env.sh --recreate   delete and rebuild .venv
# Then run the project with ./run.sh (no need to activate anything), e.g.  ./run.sh --preset quick
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$*"; }

if [ "${1:-}" = "--recreate" ]; then rm -rf .venv; fi

pick_python() {
  # Native (arm64 on Apple Silicon), version 3.10-3.13, and NOT from conda/miniforge.
  local cands=(/opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.10
               /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.13
               /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
               /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
               python3.12 python3.11 python3.13 python3.10 python3)
  local apple_silicon=0
  [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ] && apple_silicon=1
  for c in "${cands[@]}"; do
    local p; p="$(command -v "$c" 2>/dev/null || true)"
    [ -n "$p" ] || continue
    case "$(cd "$(dirname "$p")" && pwd -P)/$(basename "$p")" in *conda*|*miniforge*|*mambaforge*|*anaconda*|*mamba*) continue;; esac
    if "$p" - "$apple_silicon" <<'PY' 2>/dev/null
import platform, sys
v, arm_hw = sys.version_info, sys.argv[1] == "1"
ok_ver = (3, 10) <= (v.major, v.minor) <= (3, 13)
ok_arch = (not arm_hw) or platform.machine() == "arm64"
sys.exit(0 if ok_ver and ok_arch else 1)
PY
    then echo "$p"; return 0; fi
  done
  return 1
}

if [ -x .venv/bin/python ] && .venv/bin/python check_env.py >/dev/null 2>&1; then
  say "Existing .venv already passes the checks."
else
  say "Looking for a native Python 3.10-3.13 (not conda)"
  if ! PY="$(pick_python)"; then
    cat <<'MSG'
No suitable Python found. Install one, then rerun this script:
  Homebrew:   brew install python@3.12
  or download the macOS installer for Python 3.12 from https://www.python.org/downloads/macos/
(On Apple Silicon it must be the native arm64 build; python.org's "universal2" installer is fine.)
MSG
    exit 1
  fi
  echo "Using $PY"; "$PY" -c "import platform,sys;print(sys.version.split()[0], platform.machine())"
  say "Creating .venv"
  rm -rf .venv
  "$PY" -m venv .venv
  if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    say "Installing packages (PyTorch is large; this can take a few minutes)"
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  fi
  say "Checking the new environment"
  .venv/bin/python check_env.py || { echo "The new environment still has problems; send me the output above."; exit 1; }
fi

cat <<'MSG'

Done. Run the project WITHOUT activating anything:
  ./run.sh --preset quick
  ./run.sh --preset medium
Other commands in this environment:
  .venv/bin/python tests/test_pipeline.py
  .venv/bin/python check_env.py --full
MSG
