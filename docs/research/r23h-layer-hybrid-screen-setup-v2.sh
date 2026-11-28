#!/usr/bin/env bash
set -euo pipefail
R=$HOME/.cache/mt-pruning
D=$R/r23h-layer-hybrid-screen-run-v2
mkdir -m700 "$D"
cd "$D"
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/328db143ec18921609b753bce8a308493e9f47ba/docs/research/r23h-layer-hybrid-prefill-screen-v2.py -o screen.py
test "$(git hash-object screen.py)" = a143b365f535e1651267e25a1b8b155e5e3edf41
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/419fc3e388a20cf4d4aaeb04020072748ac2fd45/docs/research/r23h-layer-hybrid-minimax.json -o minimax.json
test "$(git hash-object minimax.json)" = e8a9b24d182cba9f543eb5087ca96752fc211f7f
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/725fdc8e644106bfb9162d9f3df38fb64df54905/docs/research/r23h-layer-hybrid-balanced.json -o balanced.json
test "$(git hash-object balanced.json)" = 4522d3a91f5383a0a655a7365c8807a2d8620639
python3 -m py_compile screen.py
test -f "$R/r23h-pareto-screen-v1/driver.py"
test -f "$R/r23h-pareto-screen-v1/collector.py"
flock -n "$R/r23-admissions/batch.lock" -c true
cat > run.sh <<'RUN'
#!/bin/bash
set -uo pipefail
export PYTHONPATH="$HOME/mt-pruning-platform" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
"$HOME/miniconda3/envs/mt-pruning/bin/python" -B screen.py . >run.log 2>run.stderr
x=$?
printf "%s\n" "$x" >exit-code
exit "$x"
RUN
chmod 700 run.sh
nohup setsid nice -n 5 bash run.sh </dev/null >/dev/null 2>&1 &
echo HYBRID_SCREEN_PID=$!
sleep 2
tail -n 4 run.log
cat run.stderr
