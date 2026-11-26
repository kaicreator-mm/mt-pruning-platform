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
RUN=$(cd "$RUN" && pwd)
PRUNED=$(cd "$PRUNED" && pwd)
EVIDENCE_DIR=$(cd "$EVIDENCE_DIR" && pwd)
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
  df -h "$RUN" "$PRUNED" "$EVIDENCE_DIR"
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

# Live, per-stage free-space checks on the actual output filesystems. The estimate is
# deliberately simple and conservative: the largest identity (128-expert) copy, every
# retained witness trace/journal, and the later evidence export. Witnesses are never reduced.
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

disk_preflight() {
  "$PY" - "$MODEL_DIR" "$RUN" "$PRUNED" "$EVIDENCE_DIR" "$1" "$MAX_NEW" "$TRACE_LIMIT" <<'EOF'
import datetime, json, os, shutil, subprocess, sys
model_dir, run_dir, pruned_dir, evidence_dir, stage = sys.argv[1:6]
max_new, trace_limit = int(sys.argv[6]), int(sys.argv[7])
total = json.load(open(model_dir + "/model.safetensors.index.json"))["metadata"]["total_size"]
experts = json.load(open(model_dir + "/config.json"))["num_experts"]
prune_peak = int(total * experts / experts * 1.05) + 3_000_000_000  # largest: the 128-copy + slack
positions = 700 + max_new
witness_sentences = 6 * trace_limit
trace_runs = 1 + 1 + 2 * 2                         # baseline + noop + masked/mechanism for 96/64
trace_budget = witness_sentences * trace_runs * positions * 1024
journal_budget = (288 + 4 * 12) * 16_384
run_budget = trace_budget + journal_budget
def du(path):
    done = subprocess.run(["du", "-sb", path], capture_output=True, text=True)
    return int(done.stdout.split()[0]) if done.returncode == 0 else 0
evidence_budget = int((du(run_dir) + run_budget) * 12 // 10) + 1_000_000_000
requirements = (("RUN", run_dir, run_budget), ("PRUNED", pruned_dir, prune_peak),
                ("EVIDENCE_DIR", evidence_dir, evidence_budget))
filesystems = {}
for label, path, needed in requirements:
    device = os.stat(path).st_dev
    item = filesystems.setdefault(device, {"device": device, "paths": [], "needed_bytes": 0,
                                          "free_bytes": shutil.disk_usage(path).free})
    item["paths"].append({"role": label, "path": os.path.abspath(path), "needed_bytes": needed})
    item["needed_bytes"] += needed
for item in filesystems.values():
    item["sufficient"] = item["free_bytes"] >= item["needed_bytes"]
checks = sorted(filesystems.values(), key=lambda item: item["device"])
receipt = {"stage": stage, "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "needed_bytes": sum(item["needed_bytes"] for item in checks),
           "free_bytes": sum(item["free_bytes"] for item in checks),
           "sufficient": all(item["sufficient"] for item in checks), "filesystems": checks,
           "assumptions": {"largest_copy_experts": experts, "prune_peak_bytes": prune_peak,
                           "run_budget_bytes": run_budget, "trace_budget_bytes": trace_budget,
                           "trace_positions_per_sentence": positions, "trace_runs": trace_runs,
                           "journal_budget_bytes": journal_budget, "evidence_budget_bytes": evidence_budget,
                           "witness_sentences": witness_sentences, "witness_count_never_reduced": True}}
os.makedirs(evidence_dir + "/inputs", exist_ok=True)
for path in (evidence_dir + "/inputs/disk_preflight.json",
             evidence_dir + "/inputs/disk_preflight_" + stage + ".json"):
    with open(path, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
print(json.dumps(receipt))
if not receipt["sufficient"]:
    sys.exit(f"disk preflight {stage}: insufficient live free space on an actual output filesystem; "
             "witness counts are never reduced to fit")
EOF
}

step "baseline: calibration routing statistics and validation outputs"
disk_preflight baseline
"$PY" -m mtpruning run --model "$MODEL_DIR" --corpus "$CORPUS" --out "$RUN/baseline/records.jsonl" \
  --splits calibration,validation --stats "$RUN/baseline/stats" \
  --trace-split validation --trace-dir "$RUN/baseline/traces" --trace-n "$TRACE_LIMIT"   --max-new-tokens "$MAX_NEW" "${LIMITED[@]}" "${COMMON[@]}"
EXPERTS=$("$PY" -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['num_experts'])" "$RUN/baseline/records.jsonl")

step "selection from the calibration routing, after the coverage check"
"$PY" -m mtpruning select --run-dir "$RUN" --keep "$(echo "$CANDIDATES" | tr ' ' ',')" \
  --out "$RUN/selection.json" --cache-dir "$CACHE_DIR" "${PIN[@]}"

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
disk_preflight "k$EXPERTS"
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
  disk_preflight "k$keep"
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
    --out "$folder/stock_check.json" --splits validation --limit 1 --max-new-tokens "$MAX_NEW" --threads "$THREADS" \
    --cache-dir "$CACHE_DIR" "${PIN[@]}"
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
disk_preflight evidence
# A retained-evidence manifest first: relative path -> sha256 for EVERY file the run kept,
# witness traces included. The evidence directory carries the manifest plus the small
# artifacts themselves; the large raw traces stay in $RUN at their recorded location and
# can be re-verified against this manifest (R13-P2-01).
(cd "$RUN" && find . -type f -not -name "*.timings.jsonl" -not -name "retained_manifest.json" \
    -not -name "retained_manifest.sha256") | sort | while read -r f; do
  sha=$(sha256sum "$RUN/$f" | cut -d' ' -f1)
  printf '%s  %s\n' "$sha" "$f"
done > "$RUN/retained_manifest.sha256"
printf '%s\n' "$RUN" > "$EVIDENCE_DIR/inputs/retained_run_location.txt"
(cd "$RUN" && sha256sum -c retained_manifest.sha256) > "$EVIDENCE_DIR/inputs/retained_manifest.check.txt"
step "evidence copy (JSON/JSONL/NPZ/SQLite/manifests; raw traces stay in the run directory)"
(cd "$RUN" && find . -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.npz" -o -name "*.sqlite" \
    -o -name "*.identity.json" -o -name "report.md" -o -name "retained_manifest.sha256" \) \
    -not -path "*/traces/*" -not -name "*.timings.jsonl") | while read -r f; do
  mkdir -p "$EVIDENCE_DIR/$(dirname "$f")"
  cp "$RUN/$f" "$EVIDENCE_DIR/$f"
done
step "done"
