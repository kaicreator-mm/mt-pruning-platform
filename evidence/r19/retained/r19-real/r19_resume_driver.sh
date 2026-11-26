#!/usr/bin/env bash
set -uo pipefail
cd /home/cc/mt-pruning-platform || exit 90
RUN=/home/cc/.cache/mt-pruning/run-d01b-r19
PRUNED=/home/cc/.cache/mt-pruning/pruned-r19
EVID=/home/cc/mt-pruning-platform/evidence/d01b-r19
REAL=/home/cc/.cache/mt-pruning/r19-real
{
  printf 'RESUME_START_UTC='
  date -u +%FT%TZ
  printf 'TESTED_CODE_HEAD='
  git rev-parse HEAD
  printf 'BASELINE_RECORDS_SHA256='
  sha256sum "$RUN/baseline/records.jsonl" | cut -d' ' -f1
  printf 'BASELINE_JOURNAL_SHA256='
  sha256sum "$RUN/baseline/records.jsonl.sqlite" | cut -d' ' -f1
  printf '%s\n' 'RESUME_REASON=attached tee SIGPIPE after complete baseline export; no product exception and no selection started'
} > "$REAL/resume-command.txt"
(
  while :; do
    printf '\n=== SAMPLE '
    date -u +%FT%TZ
    printf ' ===\n'
    free -b
    df -B1 "$RUN" "$PRUNED" "$EVID"
    printf 'LOAD '
    cat /proc/loadavg
    ps -C python -o pid,ppid,etime,%cpu,%mem,rss,vsz,args --sort=-rss | head -8
    du -sb "$RUN" "$PRUNED" "$EVID"
    sleep 60
  done
) > "$REAL/resources-resume.log" 2>&1 &
MON=$!
printf '%s\n' "$MON" > "$REAL/resume-monitor.pid"
trap 'kill "$MON" 2>/dev/null || true; wait "$MON" 2>/dev/null || true' EXIT
/usr/bin/time -v env -u MTPRUNING_TEST_CRASH_AFTER -u LIMIT -u MANIFEST_SHA256 -u TOKENIZER_DIR -u CORPUS_TARBALL \
  PY=/home/cc/miniconda3/envs/mt-pruning/bin/python ROOT=/home/cc/.cache/mt-pruning \
  RUN="$RUN" PRUNED="$PRUNED" EVIDENCE_DIR="$EVID" MODE=real CANDIDATES="96 64" \
  TRACE_LIMIT=8 MAX_NEW=640 THREADS=16 FINAL_TEST=0 SKIP_WAIT= bash scripts/run_d01b.sh
RC=$?
{
  printf 'EXIT_CODE=%s\n' "$RC"
  printf 'END_UTC='
  date -u +%FT%TZ
  printf 'HEAD='
  git rev-parse HEAD
} > "$REAL/exit.txt"
exit "$RC"
