#!/usr/bin/env bash
set -euo pipefail
R=$HOME/.cache/mt-pruning
D=$R/r23h-all8-ablation-screen-run-v1
mkdir -m700 "$D"
cd "$D"
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/9baf59f8d237706dd06dbdf800b6303fe82538ca/docs/research/r23h-all8-ablation-prefill-screen.py -o screen.py
test "$(git hash-object screen.py)" = aea20d2b9230ae4878b70f5ebe394e7608d458b4
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/016608745ab8bd1c6faf4702baef6a65c2e08330/docs/research/all8_revert40.json -o all8_revert40.json
test "$(git hash-object all8_revert40.json)" = 255c95382a5888f9749d7626395182ba514c3c8d
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/684221cf7d3628249649aea0b6c8bee711f20476/docs/research/all8_revert42.json -o all8_revert42.json
test "$(git hash-object all8_revert42.json)" = f738ea900952ab19dbf6fd4cbca235dc40e56d05
curl -fsSL --max-time 30 https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/9f7049c25e5bab5119e2b3a4235d17d281a4da6e/docs/research/all8_revert40_42.json -o all8_revert40_42.json
test "$(git hash-object all8_revert40_42.json)" = 97809cb5092ee6b903a4def8ee0139902d24f17d
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
echo ALL8_ABLATION_SCREEN_PID=$!
sleep 2
tail -n 4 run.log
cat run.stderr
