#!/bin/bash
set -euo pipefail
cd "$HOME/Projects/lacot"
exec env OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench PYTHONUNBUFFERED=1 \
    /archive/cymaxwelllee/LaCoT/.venv/bin/python -u experiments/u_variability.py
