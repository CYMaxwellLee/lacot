#!/bin/bash
#
# Smoke test: confirm the compute node + the NFS-shared venv + GPU all work.
#
# Run as a batch job:   GPUS=1 ./slurm_scripts/submit.sh slurm_scripts/smoke.sh
# Or inside an interactive shell:   bash slurm_scripts/smoke.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

echo "host        : $(hostname)"
echo "FPO_ROOT    : $FPO_ROOT"
echo "nvidia-smi  :"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 | head || echo "  (no GPU allocated)"
echo "python check:"
uv run python - <<'PY'
import numpy, matplotlib
print("  numpy      ", numpy.__version__)
print("  matplotlib ", matplotlib.__version__)
try:
    import torch
    print("  torch      ", torch.__version__)
    print("  cuda avail ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device     ", torch.cuda.get_device_name(0))
        # Exercise an actual GPU kernel. Blackwell (sm_120) needs torch>=2.7;
        # an older torch prints a "no kernel image" error here.
        x = torch.randn(256, 256, device="cuda")
        y = float((x @ x).sum().item())
        print("  matmul ok  ", y == y)  # not NaN
except Exception as e:
    print("  torch ERROR:", repr(e))
PY
echo "smoke: done"
