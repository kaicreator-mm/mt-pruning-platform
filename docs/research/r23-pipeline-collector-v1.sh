#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
end=$((SECONDS+10800))
S=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -p 22022 cc@localhost)
while (( SECONDS < end )); do
  state=$(timeout 20 "${S[@]}" 'D=$HOME/.cache/mt-pruning/r23-pipeline-v1; if test -f "$D/exact-primary.exit"; then cat "$D/exact-primary.exit"; else echo WAIT; fi' 2>/dev/null) || state=UNREACHABLE
  case "$state" in
    0) break;;
    WAIT|UNREACHABLE) sleep 30;;
    *) printf 'PRIMARY_FAILED %s\n' "$state" > collector-terminal.txt; exit 1;;
  esac
done
test "$state" = 0
test ! -e primary-evidence.tgz
timeout 180 "${S[@]}" 'cd ~/.cache/mt-pruning; tar -czf - r23-pipeline-v1/{r23h-l42-l10-l15-primary.py,search-plan.json,replay.json,exact-primary.log,exact-primary.stderr,exact-primary.exit} r23-results/r23h-l42-l10-l15-primary-v1 r23g-recovery-ssd-38ed4798/{recovery-receipt.json,expanded.json,prop-plan.json} r23-results/r23g-l38-propagation-v1/{00,01,02,03,12,13,14,15,16,17}.json r23-results/r23h-all8-one-swap-prefill-v1/plan.json' > primary-evidence.tgz
H=$(sha256sum primary-evidence.tgz | cut -d' ' -f1)
python3 r23h-l10-l15-independent-replay.py primary-evidence.tgz "$H" > independent-result.json
python3 - <<'PY'
import json
x=json.load(open("independent-result.json"))
print("REPLAY_PASS",x["primary_gate_pass"])
PY
printf 'COMPLETE_INDEPENDENT_REPLAY_NO_PROMOTION\n' > collector-terminal.txt
