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
# Kernel probe is exploratory and must not block the exact CPU candidate search
# if the CUDA environment or numerical probe fails.
(
 exec 9>"$R/r23-admissions/batch.lock"
 flock -n 9
 if stage cpu-capture env CUDA_VISIBLE_DEVICES="" timeout 600 "$CPU" -B r23-cuda-expert-capture.py; then
   stage cuda-probe env CUDA_VISIBLE_DEVICES=0 timeout 900 "$GPU" -B r23-cuda-expert-probe.py || true
 fi
)
stage exact-primary env CUDA_VISIBLE_DEVICES="" timeout 7200 "$CPU" -B r23h-l42-l10-l15-primary.py
printf '%s\n' "PIPELINE_COMPLETE_PENDING_INDEPENDENT_REPLAY_NO_PROMOTION" > terminal.txt
