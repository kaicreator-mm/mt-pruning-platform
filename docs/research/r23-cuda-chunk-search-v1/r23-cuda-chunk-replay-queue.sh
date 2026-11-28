#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R=$HOME/.cache/mt-pruning
exec 8>"$R/r23-admissions/pipeline.lock"; flock -w 18000 8
test -f terminal.txt
test "$(cat gpu-propagation.exit)" = 0
exec 9>"$R/r23-admissions/batch.lock"; flock -n 9
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 1800 "$HOME/miniconda3/envs/mt-pruning/bin/python" -B r23-cuda-chunk-raw-replay.py > raw-replay.log 2> raw-replay.stderr
echo COMPLETE_RAW_REPLAY_APPROXIMATE_ONLY > replay-terminal.txt
