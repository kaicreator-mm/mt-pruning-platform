#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
end=$((SECONDS+19800))
S=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -p 22022 cc@localhost)
for W in 1 2; do
 state=WAIT
 while (( SECONDS < end )); do
   state=$(timeout 20 "${S[@]}" "D=\$HOME/.cache/mt-pruning/r23-pipeline-v2; for f in capture gpu-search wave1 wave2; do if test -f \"\$D/\$f.exit\" && test \"\$(cat \"\$D/\$f.exit\")\" != 0; then echo FAILED:\$f; exit; fi; done; if test -f \"\$D/wave$W.exit\"; then cat \"\$D/wave$W.exit\"; else echo WAIT; fi" 2>/dev/null) || state=UNREACHABLE
   case "$state" in
     0) break;;
     WAIT|UNREACHABLE) sleep 30;;
     *) printf '%s\n' "$state" > collector-terminal.txt; exit 1;;
   esac
 done
 test "$state" = 0
 timeout 240 "${S[@]}" "cd ~/.cache/mt-pruning; tar -czf - r23-pipeline-v2/{r23-gpu-proposed-primary-v2.py,wave$W-plan.json,replay.json,gpu-search-report.json,wave$W.log,wave$W.stderr,wave$W.exit} r23-results/r23h-gpu-proposed-wave$W-primary-v2 r23g-recovery-ssd-38ed4798/{recovery-receipt.json,expanded.json,prop-plan.json} r23-results/r23g-l38-propagation-v1/{00,01,02,03,12,13,14,15,16,17}.json r23-results/r23h-all8-one-swap-prefill-v1/plan.json" > "wave$W-evidence.tgz"
 H=$(sha256sum "wave$W-evidence.tgz"|cut -d' ' -f1)
 python3 r23-gpu-wave-independent-replay-v2.py "wave$W-evidence.tgz" "$H" "$W" > "wave$W-independent-result.json"
 echo "REPLAY_PASS wave$W"
done
printf 'COMPLETE_INDEPENDENT_REPLAY_NO_PROMOTION\n' > collector-terminal.txt
