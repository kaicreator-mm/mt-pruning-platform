#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R=$HOME/.cache/mt-pruning
CPU=$HOME/miniconda3/envs/mt-pruning/bin/python
GPU=$HOME/miniconda3/envs/voxcpm/bin/python
export PYTHONPATH="$HOME/mt-pruning-platform" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
sha256sum -c source.sha256
test "$(sha256sum "$R/r23-cuda-chunk-search-v1/raw-replay.json" | cut -d' ' -f1)" = 22af51cb2e47dbcb14e7774fee58eb62917a84e8842a28f6712ae892fbb1a07c
exec 8>"$R/r23-admissions/pipeline.lock"; flock -n 8
exec 9>"$R/r23-admissions/batch.lock"; flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
stage() {
 local n=$1; shift
 echo "$(date -Is) START $n" >> stages.log
 local rc=0
 "$@" > "$n.log" 2> "$n.stderr" || rc=$?
 echo "$rc" > "$n.exit"
 echo "$(date -Is) END $n rc=$rc" >> stages.log
 return "$rc"
}
stage cpu-capture env CUDA_VISIBLE_DEVICES="" timeout 1200 "$CPU" -B r23-drift-capture.py
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > gpu-samples.csv &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
stage gpu-drift env CUDA_VISIBLE_DEVICES=0 timeout 3600 "$GPU" -B r23-drift-probe.py
echo COMPLETE_DIAGNOSTIC_ONLY > terminal.txt
