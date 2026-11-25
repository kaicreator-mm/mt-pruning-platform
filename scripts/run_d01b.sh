#!/usr/bin/env bash
# D01b real-weight experiment on one machine, in stages. Safe to rerun: a stage whose
# output exists is skipped, and a run resumes from its journal.
#
# By default this script stops after the provisional decision and never runs the
# test split:
#   corpus check → checkpoint content → baseline (calibration + validation) → selection
#   → identity reload of the unpruned checkpoint (no-op), on validation sentences
#   → per candidate: physical prune, masked equivalence on validation sentences, reference
#     runtime, complete validation
#   → provisional decision (validation only) → report
#
# The mechanism checks run on held-out validation sentences, not on the calibration
# sentences the selection was built from: the pruned experts are the ones the calibration
# sentences rarely use, so on those the mask would hardly ever change a routing decision.
#
# There is no final test in this build (R11-P1-05): every execution path for the test
# split is removed. Setting FINAL_TEST to anything exits before any stage runs, and no
# review reference re-enables it. The separately authorized execution design (decision/
# candidate/corpus/protocol bound, atomic one-candidate ledger) is a later task.
#
# Environment (defaults are the real run on CC-LAN200; a dry run overrides them):
#   PY ROOT RUN PRUNED EVIDENCE_DIR CACHE_DIR HUB_DIR MODEL_DIR TOKENIZER_DIR
#   CORPUS_TARBALL MANIFEST_SHA256 CANDIDATES TRACE_LIMIT MAX_NEW THREADS LIMIT SKIP_WAIT
#   MODE (real | rehearsal; a dry run must set rehearsal)
set -euo pipefail

PY=${PY:-/home/cc/miniconda3/envs/mt-pruning/bin/python}
ROOT=${ROOT:-$HOME/.cache/mt-pruning}
REPO=$(cd "$(dirname "$0")/.." && pwd)
HUB_DIR=${HUB_DIR:-$ROOT/hf_home/hub/models--tencent--Hy-MT3-30B-A3B}
MODEL_DIR=${MODEL_DIR:-$HUB_DIR/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06}
TOKENIZER_DIR=${TOKENIZER_DIR:-$MODEL_DIR}
RUN=${RUN:-$ROOT/run-d01b}
PRUNED=${PRUNED:-$ROOT/pruned}
CACHE_DIR=${CACHE_DIR:-$ROOT/content-hashes}
EVIDENCE_DIR=${EVIDENCE_DIR:-$REPO/evidence/d01b}
CANDIDATES=${CANDIDATES:-"96 64"}
TRACE_LIMIT=${TRACE_LIMIT:-8}       # validation sentences per direction in the mechanism checks
MAX_NEW=${MAX_NEW:-640}
THREADS=${THREADS:-16}
SKIP_WAIT=${SKIP_WAIT:-}
MODE=${MODE:-real}
mkdir -p "$RUN" "$PRUNED" "$CACHE_DIR" "$EVIDENCE_DIR/inputs"
cd "$REPO"
CORPUS=$RUN/corpus/corpus.jsonl

step() { echo "[$(date -u +%FT%TZ)] $*"; }
# Checked first, before any stage: the test split is not exposed in this build at all.
if [ "${FINAL_TEST:-0}" != "0" ]; then
  echo "FINAL_TEST is disabled in this build (R12): the test split is not executable, and no"
  echo "review reference re-enables it. test_execution_authorized stays false (see decision.json)."
  exit 1
fi
PIN=(); [ -n "${MANIFEST_SHA256:-}" ] && PIN=(--manifest-sha256 "$MANIFEST_SHA256")
LIMITED=(); [ -n "${LIMIT:-}" ] && LIMITED=(--limit "$LIMIT")
COMMON=(--mode "$MODE" --threads "$THREADS" --cache-dir "$CACHE_DIR" "${PIN[@]}")

if [ -z "$SKIP_WAIT" ]; then
  step "wait for the checkpoint download"
  while [ ! -f "$HUB_DIR/snapshots/DOWNLOAD_COMPLETE" ]; do sleep 60; done
  cp "$HUB_DIR/snapshots/download.state" "$EVIDENCE_DIR/inputs/download_state.txt"
fi
{
  echo "captured $(date -u +%FT%TZ) on $(hostname)"
  lscpu | grep -E "^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|NUMA node\(s\)|L3 cache)"
  echo "flags of interest: $(lscpu | grep -o -E "avx[0-9a-z_]*|amx[a-z_]*" | sort -u | tr "\n" " ")"
  free -g
  df -h / | sed -n '1p;2p'
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1
  ip -br addr | grep -v -E "^(lo|veth|br-|docker)"
  uptime
  "$PY" -c "import sys, torch, transformers, sacrebleu, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'sacrebleu', sacrebleu.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"
} > "$EVIDENCE_DIR/inputs/host.txt" 2>&1

step "checkpoint content: hash every shard and compare with what the directory claims to be"
"$PY" -m mtpruning checkpoint-check --model "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
  --out "$EVIDENCE_DIR/inputs/checkpoint_content.json"

if [ ! -f "$CORPUS" ]; then
  [ -n "${CORPUS_TARBALL:-}" ] || { echo "no corpus in $RUN and CORPUS_TARBALL is not set"; exit 1; }
  step "build the corpus once; the check below decides whether it is the frozen one"
  "$PY" -m mtpruning corpus --tarball "$CORPUS_TARBALL" --tokenizer "$TOKENIZER_DIR" --out "$RUN/corpus"
fi
step "frozen corpus contract (a failure here is BLOCKED_CORPUS_CONTRACT: nothing is resampled)"
"$PY" -m mtpruning corpus-check --corpus-dir "$RUN/corpus" --out "$RUN/corpus_receipt.json" "${PIN[@]}"

step "baseline: calibration routing statistics and validation outputs"
"$PY" -m mtpruning run --model "$MODEL_DIR" --corpus "$CORPUS" --out "$RUN/baseline/records.jsonl" \
  --splits calibration,validation --stats "$RUN/baseline/stats" \
  --trace-split validation --trace-dir "$RUN/baseline/traces" --trace-n "$TRACE_LIMIT"   --max-new-tokens "$MAX_NEW" "${LIMITED[@]}" "${COMMON[@]}"
EXPERTS=$("$PY" -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['num_experts'])" "$RUN/baseline/records.jsonl")

step "selection from the calibration routing, after the coverage check"
"$PY" -m mtpruning select --run-dir "$RUN" --keep "$(echo "$CANDIDATES" | tr ' ' ',')" \
  --out "$RUN/selection.json" --cache-dir "$CACHE_DIR" "${PIN[@]}"

# Stop before filling the disk: a pruned checkpoint is about keep/experts of the source, plus slack.
needs_space() {
  local need avail
  need=$("$PY" - "$MODEL_DIR" "$1" <<'EOF'
import json, sys
total = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["metadata"]["total_size"]
experts = json.load(open(sys.argv[1] + "/config.json"))["num_experts"]
print(int(total * int(sys.argv[2]) / experts * 1.05) + 3_000_000_000)
EOF
)
  avail=$(df --output=avail -B1 "$PRUNED" | tail -n 1)
  if [ "$avail" -lt "$need" ]; then echo "need $need bytes free in $PRUNED, have $avail: stopping"; exit 1; fi
}

mechanism() {       # mechanism <keep> <model dir> <equiv args...>
  local keep=$1 model=$2 folder=$RUN/k$1
  shift 2
  "$PY" -m mtpruning run --model "$model" --corpus "$CORPUS" --out "$folder/mechanism/records.jsonl" \
    --splits validation --limit "$TRACE_LIMIT" --trace-dir "$folder/mechanism/traces" --trace-n "$TRACE_LIMIT" \
    --max-new-tokens "$MAX_NEW" "${COMMON[@]}"
  "$PY" -m mtpruning equiv --run-dir "$RUN" --keep "$keep" --candidate-traces "$folder/mechanism/traces" \
    --candidate-records "$folder/mechanism/records.jsonl" --prune-record "$folder/prune.json" \
    --trace-n "$TRACE_LIMIT" --out "$folder/equiv.json" "$@"
  # The raw witness traces stay on disk until the independent review is done: they are the
  # evidence the equivalence receipt binds by hash (R11-P1-03).
}

step "identity reload of the unpruned checkpoint ($EXPERTS experts) against the baseline traces"
needs_space "$EXPERTS"
rm -rf "$PRUNED/k$EXPERTS"    # leftovers of an interrupted prune; the record is only written after verification
"$PY" -m mtpruning prune --model "$MODEL_DIR" --keep "$EXPERTS" --out "$PRUNED/k$EXPERTS" \
  --out-manifest "$RUN/k$EXPERTS/prune.json" --cache-dir "$CACHE_DIR"
mechanism "$EXPERTS" "$PRUNED/k$EXPERTS" --kind noop --reference-traces "$RUN/baseline/traces" \
  --reference-records "$RUN/baseline/records.jsonl" "${PIN[@]}"
rm -rf "$PRUNED/k$EXPERTS"    # rebuildable from the selection; manifest, hashes and checks are on disk

for keep in $CANDIDATES; do
  folder=$RUN/k$keep
  step "k$keep: physical prune and tensor-level verification of the reload"
  needs_space "$keep"
  rm -rf "$PRUNED/k$keep"
  "$PY" -m mtpruning prune --model "$MODEL_DIR" --selection "$RUN/selection.json" --keep "$keep" \
    --out "$PRUNED/k$keep" --out-manifest "$folder/prune.json" --cache-dir "$CACHE_DIR"
  step "k$keep: masked original, then the reloaded prune, then their equivalence"
  "$PY" -m mtpruning run --model "$MODEL_DIR" --corpus "$CORPUS" --out "$folder/masked/records.jsonl" \
    --mask "$RUN/selection.json" --mask-keep "$keep" --splits validation --limit "$TRACE_LIMIT" \
    --trace-dir "$folder/masked/traces" --trace-n "$TRACE_LIMIT" --max-new-tokens "$MAX_NEW" "${COMMON[@]}"
  mechanism "$keep" "$PRUNED/k$keep" --kind masked --reference-traces "$folder/masked/traces" \
    --reference-records "$folder/masked/records.jsonl" --selection "$RUN/selection.json" "${PIN[@]}"
  step "k$keep: validation outputs of the reloaded prune (the test split is not run)"
  "$PY" -m mtpruning run --model "$PRUNED/k$keep" --corpus "$CORPUS" --out "$folder/records.jsonl" \
    --splits validation --max-new-tokens "$MAX_NEW" "${LIMITED[@]}" "${COMMON[@]}"
  step "k$keep: unmodified reference runtime on the pruned checkpoint"
  "$PY" -m mtpruning stock-check --model "$PRUNED/k$keep" --corpus "$CORPUS" --records "$folder/records.jsonl" \
    --out "$folder/stock_check.json" --splits validation --limit 1 --max-new-tokens "$MAX_NEW" --threads "$THREADS" "${PIN[@]}"
  rm -rf "$PRUNED/k$keep"      # rebuildable from the selection; manifest, hashes and checks are on disk
done

step "provisional decision from validation records only"
"$PY" -m mtpruning decide --run-dir "$RUN" --candidates "$(echo "$CANDIDATES" | tr ' ' ',')" \
  --out "$RUN/decision.json" --cache-dir "$CACHE_DIR" "${PIN[@]}"
"$PY" -c "import hashlib,sys; print('decision.json sha256', hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$RUN/decision.json"

step "final test: DISABLED in this build (R12); execution stays test_execution_authorized=false and the"
step "external authorization design is a separate later task, after this decision is reviewed"

step "report"
"$PY" -m mtpruning report --run-dir "$RUN" "${PIN[@]}"

step "collect evidence"
(cd "$RUN" && find . -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.npz" -o -name "*.sqlite" \
    -o -name "*.identity.json" -o -name "report.md" \) \
    -not -path "*/traces/*" -not -name "*.timings.jsonl") | while read -r f; do
  mkdir -p "$EVIDENCE_DIR/$(dirname "$f")"
  cp "$RUN/$f" "$EVIDENCE_DIR/$f"
done
step "done"
