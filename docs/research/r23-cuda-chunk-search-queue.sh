#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
R=$HOME/.cache/mt-pruning
GPU=$HOME/miniconda3/envs/voxcpm/bin/python
export PYTHONPATH="$HOME/mt-pruning-platform" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
sha256sum -c source.sha256
exec 8>"$R/r23-admissions/pipeline.lock"; flock -w 2400 8
exec 9>"$R/r23-admissions/batch.lock"; flock -n 9
test "$(cat "$R/r23-cuda-chunk-v1/chunk-probe.exit")" = 0
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > gpu-samples.csv &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
stage() {
 local n=$1;shift
 echo "$(date -Is) START $n" >> stages.log
 local rc=0
 "$@" > "$n.log" 2> "$n.stderr" || rc=$?
 echo "$rc" > "$n.exit"
 echo "$(date -Is) END $n rc=$rc" >> stages.log
 return "$rc"
}
stage fp16 env CUDA_VISIBLE_DEVICES=0 timeout 1800 "$GPU" -B r23-cuda-fp16-probe.py || true
# Choose FP16 only if both samples demonstrate speed improvement and bounded relative additional drift;
# otherwise run the FP32 GPU path. This is a performance choice, not a quality admission.
mode=$(python3 - <<'PY'
import json,pathlib
p=pathlib.Path("fp16-report.json");r=pathlib.Path.home()/".cache/mt-pruning/r23-cuda-chunk-v1/chunk-report.json"
if not p.is_file():print("fp32")
else:
 a=json.loads(p.read_text());b=json.loads(r.read_text())
 ok=all(x["cuda_seconds"]<y["cuda_seconds"] and all(m["nmse"]<=n["nmse"]+.003 for m,n in zip(x["metrics"],y["metrics"])) for x,y in zip(a["results"],b["results"]))
 print("fp16" if ok else "fp32")
PY
)
echo "$mode" > precision.txt
stage gpu-propagation env CUDA_VISIBLE_DEVICES=0 timeout 14400 "$GPU" -B r23-cuda-chunk-search.py "$mode"
echo COMPLETE_GPU_APPROXIMATE_ONLY > terminal.txt
