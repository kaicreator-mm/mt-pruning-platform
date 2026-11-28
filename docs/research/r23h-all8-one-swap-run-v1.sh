#!/usr/bin/env bash
set -euo pipefail

PINNED_HEAD="03e9e876f7aed4d1c3fd6f2271215f67eb074141"
R="$HOME/.cache/mt-pruning"
RUN="$R/r23h-all8-one-swap-run-v1"
WORKTREE="$R/worktrees/r23h-all8-one-swap-${PINNED_HEAD:0:12}"
OUT="$R/r23-results/r23h-all8-one-swap-prefill-v1"

test -d "$RUN"
test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$PINNED_HEAD"
test ! -e "$RUN/exit-v1-code"
test ! -e "$OUT"
test -f "$RUN/screen.py"
test -f "$RUN/driver.py"
test -f "$RUN/collector.py"
test -f "$RUN/original.json"
test -f "$RUN/candidate-inventory.json"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mt-pruning
python -m py_compile "$RUN/screen.py"

ENGINE_HEAD="$(git -C "$HOME/mt-pruning-platform" rev-parse HEAD)"
WORKTREE_HEAD="$(git -C "$WORKTREE" rev-parse HEAD)"
test "$ENGINE_HEAD" = "3212303e1e02994a5124db8c85195b91325c7a7c"
test "$WORKTREE_HEAD" = "$PINNED_HEAD"
export PYTHONPATH="$HOME/mt-pruning-platform${PYTHONPATH:+:$PYTHONPATH}"
printf '{"status":"R23H_ALL8_ONE_SWAP_RUNTIME_ENVIRONMENT","engine_repo_head":"%s","worktree_head":"%s","pythonpath":"%s"}\n' "$ENGINE_HEAD" "$WORKTREE_HEAD" "$PYTHONPATH" > "$RUN/run-v1.log"

set +e
MT_PRUNING_REPO="$WORKTREE" python -u "$RUN/screen.py" "$RUN" \
  >> "$RUN/run-v1.log" 2> "$RUN/run-v1.stderr"
rc=$?
printf '%s\n' "$rc" > "$RUN/exit-v1-code"
exit "$rc"
