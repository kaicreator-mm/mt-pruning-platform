#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R=$HOME/.cache/mt-pruning
CPU=$HOME/miniconda3/envs/mt-pruning/bin/python
GPU=$HOME/miniconda3/envs/voxcpm/bin/python
export PYTHONPATH="$HOME/mt-pruning-platform" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
exec 8>"$R/r23-admissions/pipeline.lock"
flock -n 8
sha256sum -c source.sha256
stage() {
  local name=$1; shift
  printf '%s START %s\n' "$(date -Is)" "$name" >> stages.log
  local rc=0
  "$@" > "$name.log" 2> "$name.stderr" || rc=$?
  printf '%s\n' "$rc" > "$name.exit"
  printf '%s END %s rc=%s\n' "$(date -Is)" "$name" "$rc" >> stages.log
  return "$rc"
}
(
 exec 9>"$R/r23-admissions/batch.lock"
 flock -n 9
 stage capture env CUDA_VISIBLE_DEVICES="" timeout 1200 "$CPU" -B r23-gpu-search-capture-v2.py
 nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > gpu-samples.csv &
 MON=$!
 trap 'kill "$MON" 2>/dev/null || true' EXIT
 stage gpu-search env CUDA_VISIBLE_DEVICES=0 timeout 3600 "$GPU" -B r23-gpu-swap-search-v2.py
)
stage wave1 env CUDA_VISIBLE_DEVICES="" timeout 7200 "$CPU" -B r23-gpu-proposed-primary-v2.py 1
stage wave2 env CUDA_VISIBLE_DEVICES="" timeout 7200 "$CPU" -B r23-gpu-proposed-primary-v2.py 2
printf '%s\n' "PIPELINE_COMPLETE_PENDING_INDEPENDENT_REPLAY_NO_PROMOTION" > terminal.txt
