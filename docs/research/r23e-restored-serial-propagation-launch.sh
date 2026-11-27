#!/usr/bin/env bash
# Launch only the frozen diagnostic after the stable96 branch is fully audited.
# Run on the restored host with exact pinned inputs in the supplied directory.
set -euo pipefail
D=$(realpath "${1:?inputs directory required}")
COMMENT=${2:?admission comment required}
HEAD=${3:?source head required}
[[ "$COMMENT" =~ ^[1-9][0-9]*$ && "$HEAD" =~ ^[0-9a-f]{40}$ ]]
ROOT=$HOME/.cache/mt-pruning
P=$HOME/miniconda3/envs/mt-pruning/bin/python
M=$ROOT/r23-results/expanded-fit48-recovery-manifest.json
C=$ROOT/r23-results/expanded-stable96-v1.json
MH=2a9bff199b9a180e471f9baabd709fae3b932b2142a7dcf3d1b377ee80666423
CH=488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef
pin() { test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
pin "$M" "$MH"; pin "$C" "$CH"
pin "$D/stable.py" 1fd2e670bc55570668b32c18d179f497d2a8a50c1808b4c18e304668af9f291a
pin "$D/admit.py" 6a7d0571b9021b4e3d2a48f73efe3945de9001391c7120eddfc185cd376cc6e1
pin "$D/proxy.py" ced20ef343a4c40f5db2b14faa09cb206901d17e0d2e061dffaa9e0e8916efe7
pin "$D/propagate.py" 39cab8f8c655cc2a263bd96f99b18865dd77d3e153ec8e695f291b718eb934d4
pin "$D/prop-audit.py" 616fcd205a2f345bf4ca58718ffd19e7a1599cc83453ebeca7ca279f0b737025
DEC=$ROOT/r23-results/expanded-stable96-proxy-v1/fit/decision.json
AUD=$ROOT/r23-results/expanded-stable96-proxy-v1-replay-audit.json
PRE=$ROOT/r23-results/stable96-recovery-fit-24-48.log
L=$ROOT/r23-results/nonuniform-propagation-v1.log
E=$ROOT/r23-results/nonuniform-propagation-v1.stderr
OUT=$ROOT/r23-results/nonuniform-propagation-v1
REPLAY=$ROOT/r23-results/nonuniform-propagation-v1-exact-replay.json
for f in "$L" "$E" "$OUT" "$REPLAY"; do test ! -e "$f" && test ! -L "$f"; done
T=$(mktemp -d /tmp/r23e-launch.XXXXXX)
trap 'rm -rf "$T"' EXIT
# No waiting by default; do not occupy a scarce job slot during another experiment.
test -f "$DEC" && test -f "$PRE"
if ! grep -Fq '"status": "NO_ADMISSION_FROM_EXPANDED_FIT"' "$PRE"; then test -f "$AUD"; fi
flock -n "$ROOT/r23-admissions/batch.lock" -c true
python3 -B "$D/admit.py" --inputs "$D" --stage audit --candidate-sha "$CH" --manifest "$M" --manifest-sha "$MH" --out "$T/stable-audit.json" > "$T/stable-audit.log"
cat "$T/stable-audit.log"
if test -f "$T/stable-audit.json"; then cmp "$T/stable-audit.json" "$AUD"; else grep -Fq STOP_FIT_REJECTED_NO_FRESH "$T/stable-audit.log"; fi
python3 - "$D/prop-plan.json" "$T/prop-admission.json" "$COMMENT" "$HEAD" <<'PY'
import pathlib,json,hashlib,sys
p,out=map(pathlib.Path,sys.argv[1:3]);raw=p.read_bytes()
assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()=="cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde"
plan=json.loads(raw)
a=dict(status="R23E_NONUNIFORM_PROPAGATION_DIAGNOSTIC_ADMITTED",plan_blob="cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",teacher="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94",engine_head="3212303e1e02994a5124db8c85195b91325c7a7c",threads=8,mask_sha256="2339adfdbb8a438ecfa046384cced8f652fd1a523e722a66bf4480002a0500c9",ids=[r["id"] for r in plan["rows"]],admission_comment=int(sys.argv[3]),source_head=sys.argv[4])
b=json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
with out.open("xb") as f:f.write(b)
print(json.dumps(dict(status="NONUNIFORM_PROPAGATION_ADMISSION_READY",sha256=hashlib.sha256(b).hexdigest())))
PY
# The original driver rechecks identity/resources under its exclusive lock.
# Create log files exclusively; failures preserve receipts and logs.
(set -C; : > "$L"; : > "$E")
env PYTHONPATH=$HOME/mt-pruning-platform CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "$P" -B "$D/propagate.py" --inputs "$D" --admission "$T/prop-admission.json" --execute 2>> "$E" | tee -a "$L"
AH=$(sha256sum "$T/prop-admission.json" | cut -d' ' -f1)
python3 -B "$D/prop-audit.py" --batch "$D/propagate.py" --plan "$D/prop-plan.json" --folder "$OUT" --log "$L" --stderr "$E" --admission-sha "$AH" | tee "$T/prop-replay.json"
(set -C; cat "$T/prop-replay.json" > "$REPLAY")
sha256sum "$L" "$E" "$REPLAY"
