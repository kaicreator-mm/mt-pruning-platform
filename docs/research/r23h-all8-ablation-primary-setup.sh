#!/usr/bin/env bash
set -euo pipefail
R=$HOME/.cache/mt-pruning
SCREEN=$R/r23h-pareto-screen-v1
D=$R/r23h-all8-ablation-screen-run-v2
RUN=$R/r23h-all8-ablation-primary-run-v1
mkdir -m700 "$RUN"
cd "$RUN"
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/parallel-r19-offline-analysis/docs/research/r23h-all8-ablation-primary-batch.py -o primary.py
test "$(git hash-object primary.py)" = 88af652bf4c40a9361d7866671790515d68bcc3b
test -f "$SCREEN/driver.py"
test -f "$SCREEN/collector.py"
test -f "$SCREEN/prop.py"
test -f "$SCREEN/original.json"
test -f "$SCREEN/run.log"
test -f "$SCREEN/exit-code"
test "$(cat "$SCREEN/exit-code")" = 0
test -f "$D/exit-code"
test "$(cat "$D/exit-code")" = 0
test -s "$D/run.log"
test ! -s "$D/run.stderr"
test -f "$D/all8_revert40.json"
test -f "$D/all8_revert42.json"
test -f "$D/all8_revert40_42.json"
test -f "$R/r23h-pareto-all10-run-v2/exit-code"
test "$(cat "$R/r23h-pareto-all10-run-v2/exit-code")" = 0
test -f "$R/r23g-recovery-ssd-38ed4798/recovery-receipt.json"
test -f "$R/r23g-recovery-ssd-38ed4798/expanded.json"
test -f "$R/r23g-recovery-ssd-38ed4798/prop-plan.json"
test -f "$R/r23-results/r23g-l38-propagation-v1/17.json"
python3 -m py_compile primary.py
flock -n "$R/r23-admissions/batch.lock" -c true
cat > run.sh <<'RUN'
#!/bin/bash
set -uo pipefail
export PYTHONPATH="$HOME/mt-pruning-platform" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
"$HOME/miniconda3/envs/mt-pruning/bin/python" -B primary.py >run.log 2>run.stderr
x=$?
printf "%s\n" "$x" >exit-code
exit "$x"
RUN
chmod 700 run.sh
nohup setsid nice -n 5 bash run.sh </dev/null >/dev/null 2>&1 &
echo ALL8_ABLATION_PRIMARY_PID=$!
sleep 2
tail -n 4 run.log
cat run.stderr
