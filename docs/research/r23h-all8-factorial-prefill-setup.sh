#!/usr/bin/env bash
set -euo pipefail
R=$HOME/.cache/mt-pruning
RR=$R/r23-results
D=$R/r23h-pareto-screen-v1
USB=$R/r23g-recovery-ssd-38ed4798
SD=$RR/r23h-pareto-prefill-v1
PREP=$RR/r23h-all8-factorial-prefill-v2
PREV=$RR/r23h-all8-ablation-primary-v3
RUN=$R/r23h-all8-factorial-prefill-run-v4
PY=$HOME/miniconda3/envs/mt-pruning/bin/python

test ! -e "$RUN"
test -d "$PREP"
test "$(sha256sum "$PREP/plan.json" | awk '{print $1}')" = 7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac
test "$(git hash-object "$PREP/plan.json")" = 415f2526c82356e216d74105301771b3060f3461
test "$(git hash-object "$PREP/all8_revert10.json")" = faa4d6146a3df4192e913e2702a926afa3fb095d
test "$(git hash-object "$PREP/all8_revert15.json")" = 044a6f395d779489462df00a664cf7876732ab06
test "$(git hash-object "$PREP/all8_revert10_40_42.json")" = adc46aceb97c4f35c8489b1b0cb02fbdf0295eee
test "$(git hash-object "$PREP/all8_revert15_40_42.json")" = e453de1af1fef152b44f2446ecb5de6f9e18b4ae
test "$(sha256sum "$PREV/report.json" | awk '{print $1}')" = 8ab5c2fc607ad49a589636df39ab46f801327d7c39cbbf70ce29b0fb2b23d2af
test "$(sha256sum "$USB/recovery-receipt.json" | awk '{print $1}')" = ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad
test "$(sha256sum "$USB/expanded.json" | awk '{print $1}')" = 21b4a54f626c32727284f32f34f3b43b54ff6e68d56ac4499934ee652fb6d166
test "$(sha256sum "$USB/candidate.json" | awk '{print $1}')" = e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1
test "$(git hash-object "$D/original.json")" = d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed
test "$(sha256sum "$SD/plan.json" | awk '{print $1}')" = b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e
test "$(sha256sum "$SD/report.json" | awk '{print $1}')" = 8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b
test -f "$D/driver.py"
test -f "$D/collector.py"
test -f "$D/run.log"
test -f "$R/run-d01b-r19/corpus/corpus.jsonl"
test -f "$R/r23-admissions/batch.lock"
for i in $(seq -w 0 17); do test -f "$RR/r23g-l38-propagation-v1/$i.json"; done
for i in 00 01 02 03 04 05 06 07; do test -f "$SD/$i.json"; done
mkdir -m 700 "$RUN"
cd "$RUN"
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/4e13345b09352583cf90c4757daf6fcf6aaaaae6/docs/research/r23h-all8-factorial-prefill.py -o factorial.py
test "$(git hash-object factorial.py)" = cb717d66c639f7be6153ce77a41b86a0b60556be
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/parallel-r19-offline-analysis/docs/research/r23h-all8-ablation-primary-replay.json -o primary-replay.json
test "$(git hash-object primary-replay.json)" = 1ae914e4e1e0c9729c43b237100b6fc20a5fbefb
python3 -m py_compile factorial.py
"$PY" - <<'PY'
import torch,transformers
assert torch.__version__=="2.10.0+cpu",torch.__version__
assert transformers.__version__=="5.24.1",transformers.__version__
print("CPU_ENGINE",torch.__version__,"TRANSFORMERS",transformers.__version__)
PY
flock -n "$R/r23-admissions/batch.lock" -c true
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
cat > run.sh <<'RUN'
#!/bin/bash
set -uo pipefail
export PYTHONPATH="$HOME/mt-pruning-platform" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
"$HOME/miniconda3/envs/mt-pruning/bin/python" -B factorial.py >run.log 2>run.stderr
x=$?
printf "%s\n" "$x" >exit-code
exit "$x"
RUN
chmod 700 run.sh
nohup setsid nice -n 5 bash run.sh </dev/null >/dev/null 2>&1 &
echo R23H_ALL8_FACTORIAL_PREFILL_PID=$!
sleep 2
tail -n 5 run.log
cat run.stderr
