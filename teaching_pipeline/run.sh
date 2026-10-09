#!/usr/bin/env bash
# Run the pipeline with the clean .venv created by ./setup_env.sh, e.g.  ./run.sh --preset quick
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "No .venv yet: run ./setup_env.sh first."; exit 1; fi
exec .venv/bin/python run_all.py "$@"
