#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
S=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -p 22022 cc@localhost)
end=$((SECONDS+22500))
state=WAIT
while ((SECONDS<end)); do
 state=$(timeout 20 "${S[@]}" 'D=$HOME/.cache/mt-pruning/r23-cuda-hybrid-v1; if test -f "$D/terminal.txt"; then echo COMPLETE; elif test -f "$D/cpu-reference.exit" && test "$(cat "$D/cpu-reference.exit")" != 0; then echo FAILED_CPU; elif test -f "$D/cuda-full-forward.exit" && test "$(cat "$D/cuda-full-forward.exit")" != 0; then echo FAILED_CUDA; elif ! kill -0 250972 2>/dev/null; then echo STOPPED; else echo WAIT; fi' 2>/dev/null) || state=UNREACHABLE
 case "$state" in COMPLETE) break;; WAIT|UNREACHABLE) sleep 30;; *) echo "$state" > collector-terminal.txt; exit 1;; esac
done
test "$state" = COMPLETE
timeout 300 "${S[@]}" 'cd ~/.cache/mt-pruning; tar -czf - r23-cuda-hybrid-v1/{r23_cuda_hybrid_engine.py,r23-cuda-hybrid-reference.py,r23-cuda-hybrid-real-probe.py,r23-cuda-hybrid-queue.sh,source.sha256,terminal.txt,cpu-reference.exit,cpu-reference.log,cpu-reference.stderr,cuda-full-forward.exit,cuda-full-forward.log,cuda-full-forward.stderr,cpu-reference.pt,hybrid-report.json,gpu-samples.csv}' > evidence.tgz
python3 r23-cuda-hybrid-replay.py evidence.tgz > replay.json
echo COMPLETE_EVIDENCE_REPLAY_NO_ADMISSION > collector-terminal.txt
