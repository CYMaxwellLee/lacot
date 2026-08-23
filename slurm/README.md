# slurm_scripts — job submission for this cluster (Slurm, node `frieren`)

The original `bv_scripts/` use IBM LSF (`bsub`), which does **not** exist on this
cluster. This cluster uses **Slurm**, and the GPU compute node is **frieren**
(6× RTX PRO 6000 Blackwell, ~98 GB VRAM each). These wrappers submit there.

The venv and caches live under `$HOME` (see `env.sh`), which is NFS-mounted on
frieren, so `uv run ...` uses the *same* environment on every node — install once.

## Batch job

```bash
./slurm_scripts/submit.sh <script> [args...]
GPUS=2 TIME=02:00:00 ./slurm_scripts/submit.sh scripts/run_pipeline.sh
```

Defaults: **1 GPU**, partition `blackwell`, node `frieren`, 12 h wall-clock.
Output → `slurm_outputs/<jobid>.out` and `.err`. Every default is overridable by
an environment variable — see the header of `submit.sh` (`GPUS`, `PARTITION`,
`NODELIST`, `NODES`, `CPUS`, `MEM`, `TIME`, `JOB_NAME`, `ACCOUNT`).

## Interactive shell

```bash
./slurm_scripts/interactive.sh      # 1 GPU on frieren, drops you into bash
```

## Smoke test (verify frieren + venv + GPU in one shot)

```bash
GPUS=1 ./slurm_scripts/submit.sh slurm_scripts/smoke.sh
cat slurm_outputs/<jobid>.out
```
