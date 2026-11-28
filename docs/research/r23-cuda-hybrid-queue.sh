#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R=$HOME/.cache/mt-pruning
CPU=$HOME/miniconda3/envs/mt-pruning/bin/python
GPU=$HOME/miniconda3/envs/voxcpm/bin/python
export PYTHONPATH="$HOME/mt-pruning-platform" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
sha256sum -c source.sha256
stage() {
 local n=$1; shift
 printf '%s START %s\n' "$(date -Is)" "$n" >> stages.log
 local rc=0
 "$@" > "$n.log" 2> "$n.stderr" || rc=$?
 printf '%s\n' "$rc" > "$n.exit"
 printf '%s END %s rc=%s\n' "$(date -Is)" "$n" "$rc" >> stages.log
 return "$rc"
}
# Existing pipeline owns the pipeline lock across both exact waves.
# Wait for release with a finite bound; never compete with the 60GB mmap oracle.
printf '%s WAIT_EXISTING_PIPELINE\n' "$(date -Is)" >> stages.log
exec 8>"$R/r23-admissions/pipeline.lock"
flock -w 15000 8
test -f "$R/r23-pipeline-v2/terminal.txt"
test "$(cat "$R/r23-pipeline-v2/wave1.exit")" = 0
test "$(cat "$R/r23-pipeline-v2/wave2.exit")" = 0
exec 9>"$R/r23-admissions/batch.lock"
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
stage cpu-reference env CUDA_VISIBLE_DEVICES="" timeout 1800 "$CPU" -B r23-cuda-hybrid-reference.py
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > gpu-samples.csv &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
stage cuda-full-forward env CUDA_VISIBLE_DEVICES=0 timeout 5400 "$GPU" -B r23-cuda-hybrid-real-probe.py
printf 'COMPLETE_CUDA_DIAGNOSTIC_NOT_ORACLE_REPLACEMENT\n' > terminal.txt
