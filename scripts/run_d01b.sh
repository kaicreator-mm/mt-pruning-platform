#!/usr/bin/env bash
# D01b real-weight validation: every stage, in order, on one machine.
#
# Safe to rerun: a stage whose output already exists is skipped, and the run stage
# resumes per sentence. Pruned checkpoints are deleted once their manifest,
# verification and comparisons are on disk; they can be rebuilt from the
# selection file.
#
# Defaults are the real run on CC-LAN200. For a dry run on a tiny checkpoint set
# MODEL_DIR, TOKENIZER_DIR, RUN, PRUNED, EVIDENCE_DIR, KEEP_LIST (first entry =
# the checkpoint's expert count), LIMIT and SKIP_WAIT.
set -uo pipefail

PY=${PY:-/home/cc/miniconda3/envs/mt-pruning/bin/python}
ROOT=${ROOT:-$HOME/.cache/mt-pruning}
REPO=$(cd "$(dirname "$0")/.." && pwd)
REV=08ae9acea91abe265c6236e757e49d8851e1ce06
HUB_DIR=${HUB_DIR:-$ROOT/hf_home/hub/models--tencent--Hy-MT3-30B-A3B}
MODEL_DIR=${MODEL_DIR:-$HUB_DIR/snapshots/$REV}
TARBALL=${TARBALL:-$ROOT/data/flores200_dataset.tar.gz}
TOKENIZER_DIR=${TOKENIZER_DIR:-$MODEL_DIR}
RUN=${RUN:-$ROOT/run-d01b}
PRUNED=${PRUNED:-$ROOT/pruned}
KEEP_LIST=${KEEP_LIST:-"128 96 64"}
TRACE_N=${TRACE_N:-8}            # traced sentences per direction
TRACE_LIMIT=${TRACE_LIMIT:-8}    # sentences per direction in the no-op / masked runs
MAX_NEW=${MAX_NEW:-640}
THREADS=${THREADS:-16}
LIMIT=${LIMIT:-}                 # sentences per split and direction; non-empty means dry run
SKIP_WAIT=${SKIP_WAIT:-}
EVIDENCE_DIR=${EVIDENCE_DIR:-$REPO/evidence/d01b}
mkdir -p "$RUN" "$PRUNED" "$EVIDENCE_DIR/inputs"
cd "$REPO"

step() { echo "[$(date -u +%FT%TZ)] $*"; }
{
  echo "captured $(date -u +%FT%TZ) on $(hostname)"
  lscpu | grep -E "^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|NUMA node\(s\)|L3 cache)"
  echo "flags of interest: $(lscpu | grep -o -E "avx[0-9a-z_]*|amx[a-z_]*" | sort -u | tr "
" " ")"
  free -g
  df -h / | sed -n '1p;2p'
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1
  ip -br addr | grep -v -E "^(lo|veth|br-|docker)"
  uptime
  "$PY" -c "import sys, torch, transformers, sacrebleu, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'sacrebleu', sacrebleu.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"
} > "$EVIDENCE_DIR/inputs/host.txt" 2>&1
LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")
COMMON=(--corpus "$RUN/corpus/corpus.jsonl" --max-new-tokens "$MAX_NEW" --threads "$THREADS")

if [ -z "$SKIP_WAIT" ]; then
  step "wait for the checkpoint download"
  while [ ! -f "$HUB_DIR/snapshots/DOWNLOAD_COMPLETE" ]; do sleep 60; done
  cp "$HUB_DIR/snapshots/download.state" "$EVIDENCE_DIR/inputs/download_state.txt"
fi

if [ ! -f "$RUN/corpus/corpus.jsonl" ]; then
  step "corpus"
  "$PY" -m mtpruning corpus --tarball "$TARBALL" --tokenizer "$TOKENIZER_DIR" --out "$RUN/corpus"
fi

step "baseline run: original model, routing statistics, traces"
"$PY" -m mtpruning run --model "$MODEL_DIR" --out "$RUN/baseline/records.jsonl" \
  --stats "$RUN/baseline/stats" --trace-dir "$RUN/baseline/traces" --trace-n "$TRACE_N" \
  "${COMMON[@]}" "${LIMIT_ARG[@]}"

SOURCE_EXPERTS=${KEEP_LIST%% *}
CANDIDATES=${KEEP_LIST#* }
if [ ! -f "$RUN/selection.json" ]; then
  step "select: direction-balanced expert sets"
  "$PY" -m mtpruning select --model "$MODEL_DIR" --stats "$RUN/baseline/stats" \
    --keep "$(echo "$CANDIDATES" | tr ' ' ',')" --out "$RUN/selection.json"
fi

for K in $KEEP_LIST; do
  D=$RUN/k$K
  P=$PRUNED/k$K
  mkdir -p "$D"
  if [ ! -f "$D/prune.json" ]; then
    step "prune $K: physical checkpoint + tensor-level verification of the reload"
    # Stop before filling the disk: the pruned checkpoint is roughly K/E of the source plus slack for traces.
    need=$("$PY" - "$MODEL_DIR" "$K" <<'EOF'
import json, sys
total = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["metadata"]["total_size"]
experts = json.load(open(sys.argv[1] + "/config.json"))["num_experts"]
print(int(total * int(sys.argv[2]) / experts * 1.05) + 3_000_000_000)
EOF
)
    avail=$(df --output=avail -B1 "$PRUNED" | tail -n 1)
    if [ "$avail" -lt "$need" ]; then step "need $need bytes free in $PRUNED, have $avail: stopping"; exit 1; fi
    rm -rf "$P"    # leftovers of an interrupted prune; the manifest is only written after verification
    "$PY" -m mtpruning prune --model "$MODEL_DIR" --selection "$RUN/selection.json" --keep "$K" \
      --out "$P" --out-manifest "$D/prune.json"
  fi
  SMALL=(--splits calibration --limit "$TRACE_LIMIT" --trace-n "$TRACE_N")
  if [ "$K" = "$SOURCE_EXPERTS" ]; then
    step "k$K: reload the identity prune and compare with the baseline traces"
    "$PY" -m mtpruning run --model "$P" --out "$D/records.jsonl" --trace-dir "$D/traces" \
      "${SMALL[@]}" "${COMMON[@]}"
    "$PY" -m mtpruning equiv --reference-traces "$RUN/baseline/traces" --candidate-traces "$D/traces" \
      --reference-records "$RUN/baseline/records.jsonl" --candidate-records "$D/records.jsonl" \
      --out "$D/equiv.json" || { step "k$K no-op check FAILED: stopping"; exit 1; }
    rm -rf "$D/traces"
  else
    step "k$K: masked original (traces)"
    "$PY" -m mtpruning run --model "$MODEL_DIR" --out "$D/masked/records.jsonl" \
      --mask "$RUN/selection.json" --mask-keep "$K" --trace-dir "$D/masked/traces" \
      "${SMALL[@]}" "${COMMON[@]}"
    step "k$K: reloaded physical prune (validation + test, traces)"
    "$PY" -m mtpruning run --model "$P" --out "$D/records.jsonl" --trace-dir "$D/traces" \
      --trace-n "$TRACE_N" "${COMMON[@]}" "${LIMIT_ARG[@]}"
    step "k$K: masked equivalence"
    "$PY" -m mtpruning equiv --reference-traces "$D/masked/traces" --candidate-traces "$D/traces" \
      --reference-records "$D/masked/records.jsonl" --candidate-records "$D/records.jsonl" \
      --out "$D/equiv.json" || step "k$K masked equivalence FAILED: recorded, continuing"
    rm -rf "$D/traces" "$D/masked/traces"
    step "k$K: unmodified reference runtime on the pruned checkpoint"
    "$PY" -m mtpruning stock-check --model "$P" --corpus "$RUN/corpus/corpus.jsonl" \
      --records "$D/records.jsonl" --out "$D/stock_check.json" --splits validation --limit 1 \
      --max-new-tokens "$MAX_NEW" --threads "$THREADS" || step "k$K stock check FAILED: recorded, continuing"
  fi
  if [ -f "$D/equiv.json" ]; then
    step "release $P"
    rm -rf "$P"
  fi
done

step "report"
VARIANTS="{\"baseline\": \"baseline\""
for K in $CANDIDATES; do VARIANTS="$VARIANTS, \"$K\": \"k$K\""; done
VARIANTS="$VARIANTS}"
"$PY" -m mtpruning report --run-dir "$RUN" --variants "$VARIANTS"
rm -rf "$RUN/baseline/traces"

step "collect evidence"
( cd "$RUN" && find . -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.npz" \) -not -path "*/traces/*" ) | while read -r f; do
  mkdir -p "$EVIDENCE_DIR/$(dirname "$f")"
  cp "$RUN/$f" "$EVIDENCE_DIR/$f"
done
step "done"
