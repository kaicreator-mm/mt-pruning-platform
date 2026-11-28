#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R=$HOME/.cache/mt-pruning
GPU=$HOME/miniconda3/envs/voxcpm/bin/python
export PYTHONPATH="$HOME/mt-pruning-platform" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
sha256sum -c source.sha256
exec 8>"$R/r23-admissions/pipeline.lock"; flock -n 8
exec 9>"$R/r23-admissions/batch.lock"; flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > gpu-samples.csv &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
echo "$(date -Is) START chunk-probe" >> stages.log
rc=0
CUDA_VISIBLE_DEVICES=0 timeout 1800 "$GPU" -B r23-cuda-chunk-probe.py > chunk-probe.log 2> chunk-probe.stderr || rc=$?
echo "$rc" > chunk-probe.exit
echo "$(date -Is) END chunk-probe rc=$rc" >> stages.log
exit "$rc"
