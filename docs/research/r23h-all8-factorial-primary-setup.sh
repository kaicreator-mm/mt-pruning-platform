#!/usr/bin/env bash
set -euo pipefail

R=$HOME/.cache/mt-pruning
RR=$R/r23-results
SCREEN=$R/r23h-pareto-screen-v1
PREFILL_RUN=$R/r23h-all8-factorial-prefill-run-v5
PREFILL=$RR/r23h-all8-factorial-prefill-v2
MASK_DIR=$R/r23h-all8-factorial-masks-v1
RUN=$R/r23h-all8-factorial-primary-run-v1
REV=5ed6d9f30716fd4e3de7eb3c672b474fe3329521
BASE=https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/$REV/docs/research

test ! -e "$RUN"
test -f "$PREFILL_RUN/exit-code"
test "$(cat "$PREFILL_RUN/exit-code")" = 0
test ! -s "$PREFILL_RUN/run.stderr"
test -f "$PREFILL/run.log"
test -f "$PREFILL/report.json"
test "$(sha256sum "$PREFILL/plan.json" | cut -d' ' -f1)" = 7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac
test "$(sha256sum "$PREFILL/report.json" | cut -d' ' -f1)" = 3953ed75fb4275019b51df5d6e8e43b91a4e77c37747330be365c3cd67f8da05
test -f "$PREFILL_RUN/primary-replay.json"
test "$(git hash-object "$PREFILL_RUN/primary-replay.json")" = 1ae914e4e1e0c9729c43b237100b6fc20a5fbefb
test -f "$SCREEN/driver.py"
test -f "$SCREEN/collector.py"
test -f "$SCREEN/prop.py"
test -f "$SCREEN/original.json"
test -f "$SCREEN/run.log"
test "$(cat "$SCREEN/exit-code")" = 0
test -f "$RR/r23h-pareto-prefill-v1/plan.json"
test -f "$RR/r23h-pareto-prefill-v1/report.json"
test -f "$R/r23g-recovery-ssd-38ed4798/recovery-receipt.json"
test -f "$R/r23g-recovery-ssd-38ed4798/expanded.json"
test -f "$R/r23g-recovery-ssd-38ed4798/prop-plan.json"
test -f "$RR/r23g-l38-propagation-v1/17.json"
test -f "$R/run-d01b-r19/corpus/corpus.jsonl"
test -f "$R/run-d01b-r19/baseline/records.jsonl"
test -f "$R/r23h-pareto-all10-run-v2/exit-code"
test "$(cat "$R/r23h-pareto-all10-run-v2/exit-code")" = 0
test "$("$HOME/miniconda3/envs/mt-pruning/bin/python" -c 'import torch; print(torch.__version__)')" = "2.10.0+cpu"
test "$("$HOME/miniconda3/envs/mt-pruning/bin/python" -c 'import transformers; print(transformers.__version__)')" = "5.24.1"

mkdir -m700 "$RUN"
install -d -m700 "$MASK_DIR"
cd "$RUN"
curl -fsSL --max-time 45 "$BASE/r23h-all8-factorial-primary-cpu-batch.py" -o primary.py
test "$(git hash-object primary.py)" = 62054bff85c77e06010a3d9968338676632b69c1

for n in all8_revert10 all8_revert15 all8_revert10_40_42 all8_revert15_40_42; do
  curl -fsSL --max-time 45 "$BASE/r23h-all8-factorial-masks/$n.json" -o "$MASK_DIR/$n.json"
done
test "$(git hash-object "$MASK_DIR/all8_revert10.json")" = faa4d6146a3df4192e913e2702a926afa3fb095d
test "$(git hash-object "$MASK_DIR/all8_revert15.json")" = 044a6f395d779489462df00a664cf7876732ab06
test "$(git hash-object "$MASK_DIR/all8_revert10_40_42.json")" = adc46aceb97c4f35c8489b1b0cb02fbdf0295eee
test "$(git hash-object "$MASK_DIR/all8_revert15_40_42.json")" = e453de1af1fef152b44f2446ecb5de6f9e18b4ae

python3 -m py_compile primary.py
flock -n "$R/r23-admissions/batch.lock" -c true
cat > run.sh <<'RUN'
#!/usr/bin/env bash
set -uo pipefail
export PYTHONPATH="$HOME/mt-pruning-platform" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
"$HOME/miniconda3/envs/mt-pruning/bin/python" -B primary.py >run.log 2>run.stderr
x=$?
printf "%s\n" "$x" >exit-code
exit "$x"
RUN
chmod 700 run.sh
nohup setsid nice -n 5 bash run.sh </dev/null >/dev/null 2>&1 &
echo R23H_FACTORIAL_PRIMARY_PID=$!
sleep 2
tail -n 8 run.log
cat run.stderr
