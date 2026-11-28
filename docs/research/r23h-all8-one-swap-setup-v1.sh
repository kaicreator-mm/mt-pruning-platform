#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/mt-pruning-platform"
PINNED_HEAD="03e9e876f7aed4d1c3fd6f2271215f67eb074141"
BRANCH="parallel-r19-offline-analysis"
R="$HOME/.cache/mt-pruning"
RR="$R/r23-results"
SOURCE_RUN="$R/r23h-layer38-single-swap-run-v1"
WORKTREE="$R/worktrees/r23h-all8-one-swap-${PINNED_HEAD:0:12}"
RUN="$R/r23h-all8-one-swap-run-v1"

test -d "$REPO/.git"
test -d "$SOURCE_RUN"
test ! -e "$WORKTREE"
test ! -e "$RUN"
test ! -e "$RR/r23h-all8-one-swap-prefill-v1"
mkdir -p "$R/worktrees" "$RR"

git -C "$REPO" fetch --no-tags origin "refs/heads/\${BRANCH}"
FETCHED="$(git -C "$REPO" rev-parse FETCH_HEAD)"
test "$FETCHED" = "$PINNED_HEAD"

git -C "$REPO" worktree add --detach "$WORKTREE" "$PINNED_HEAD"
mkdir -m 700 "$RUN"

cp "$WORKTREE/docs/research/r23h-all8-one-swap-prefill-v1.py" "$RUN/screen.py"
cp "$WORKTREE/docs/research/r23h-all8-one-swap-candidate-inventory-v1.json" "$RUN/candidate-inventory.json"
cp "$WORKTREE/docs/research/r23h-all8-factorial-masks/all8_revert10.json" "$RUN/all8_revert10.json"
cp "$WORKTREE/docs/research/r23h-all8-factorial-masks/all8_revert15.json" "$RUN/all8_revert15.json"
cp "$SOURCE_RUN/driver.py" "$RUN/driver.py"
cp "$SOURCE_RUN/collector.py" "$RUN/collector.py"
cp "$SOURCE_RUN/original.json" "$RUN/original.json"
chmod 600 "$RUN"/*

blob() { git -C "$WORKTREE" hash-object "$1"; }
test "$(blob "$RUN/screen.py")" = "0fc170a1861340153663bc0fdd8b9f65f0fc336a"
test "$(blob "$RUN/candidate-inventory.json")" = "65e57a3add8cff36979b06928af22bcdcefe3ce6"
test "$(blob "$RUN/all8_revert10.json")" = "faa4d6146a3df4192e913e2702a926afa3fb095d"
test "$(blob "$RUN/all8_revert15.json")" = "044a6f395d779489462df00a664cf7876732ab06"
test "$(blob "$RUN/driver.py")" = "0735238e15f13fd75e3f6939fc31aa834d8b5854"
test "$(blob "$RUN/collector.py")" = "02daf65bb7fe0526b2a700d17013054f7b0eea46"
test "$(blob "$RUN/original.json")" = "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mt-pruning
python -m py_compile "$RUN/screen.py"

MT_PRUNING_REPO="$WORKTREE" python - "$RUN" <<'PY'
import importlib.util, json, pathlib, sys
run = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("r23h_all8_one_swap", run / "screen.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows, context, previous, baselines, candidates, plan = module.prepare(run)
assert len(rows) == 16 and len(previous) == 16
assert len(candidates) == 16 and len(plan["candidates"]) == 16
assert set(baselines) == {"original96", "all8", "r23g"}
print(json.dumps({
    "status": "R23H_ALL8_ONE_SWAP_STATIC_PREPARATION_PASS",
    "pinned_worktree_head": "03e9e876f7aed4d1c3fd6f2271215f67eb074141",
    "rows": len(rows),
    "directions": {"zh-my": 8, "my-zh": 8},
    "candidate_count": len(candidates),
    "execution_started": False
}, sort_keys=True))
PY
