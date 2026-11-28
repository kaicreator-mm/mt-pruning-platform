#!/usr/bin/env python3
"""Pack R23H factorial primary-fit evidence for independent cross-host replay."""
import hashlib
import json
import pathlib
import tarfile

R = pathlib.Path.home() / ".cache/mt-pruning"
Q = R / "r23-results"
B = Q / "r23h-all8-factorial-primary-v3"
BR = R / "r23h-all8-factorial-primary-run-v3"
A = Q / "r23h-pareto-all10-propagation-v2"
AR = R / "r23h-pareto-all10-run-v2"
S = R / "r23h-pareto-screen-v1"
SD = Q / "r23h-pareto-prefill-v1"
F = R / "r23h-all8-factorial-prefill-run-v5"
FD = Q / "r23h-all8-factorial-prefill-v2"
FM = R / "r23h-all8-factorial-masks-v1"
U = R / "r23g-recovery-ssd-38ed4798"
P = Q / "r23g-l38-propagation-v1"
CORPUS = R / "run-d01b-r19/corpus/corpus.jsonl"
RECORDS = R / "run-d01b-r19/baseline/records.jsonl"
NAMES = ("all8_revert15_40_42", "all8_revert15")
SCREEN_MASKS = ("all8_revert10", "all8_revert15", "all8_revert10_40_42", "all8_revert15_40_42")

ad = json.loads((B / "admission.json").read_text())
assert ad["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_FIT_ONLY"
indices = ad["indices"]
assert len(indices) == 10 and len(set(indices)) == 10
assert ad["candidates"] == {
    "all8_revert15_40_42": "31c5eeb44b3dc8acb6eeae97a2078faf1b96f9e129a24fc27726232b2b5c8df2",
    "all8_revert15": "4c81baf0201fa0f085b8908625a2b46d9f4fb39bebc6cbef829138817a416c7f",
}
assert (BR / "exit-code").read_text().strip() == "0" and (BR / "run.stderr").read_bytes() == b""
assert (AR / "exit-code").read_text().strip() == "0" and (AR / "run.stderr").read_bytes() == b""
assert (F / "exit-code").read_text().strip() == "0" and (F / "run.stderr").read_bytes() == b""
assert (S / "exit-code").read_text().strip() == "0" and (S / "run.stderr").read_bytes() == b""

files = {}
def add(prefix, names):
    for name, path in names:
        assert path.is_file() and not path.is_symlink(), str(path)
        limit = 32 * 1024 * 1024 if name in ("corpus/corpus.jsonl", "corpus/records.jsonl") else 2 * 1024 * 1024
        assert path.stat().st_size <= limit, (str(path), path.stat().st_size)
        files[prefix + name] = path

add("batch/", [(x, B/x) for x in ("admission.json", "report.json")])
add("batch/", [(x, BR/x) for x in ("run.log", "run.stderr", "exit-code")])
for n in NAMES:
    for i in indices:
        name = f"{n}-{i:02d}.json"
        add("batch/", [(name, B/name)])

add("prior-all10/", [(x, A/x) for x in ("admission.json", "report.json")])
add("prior-all10/", [(x, AR/x) for x in ("run.log", "run.stderr", "exit-code")])
for i in range(18):
    name = f"{i:02d}.json"
    add("prior-all10/", [(name, A/name)])

add("screen/", [(x, FD/x) for x in ("plan.json", "report.json")])
add("screen/", [(x, F/x) for x in ("run.log", "run.stderr", "exit-code", "primary-replay.json")])
for i in range(8):
    name = f"{i:02d}.json"
    add("screen/", [(name, FD/name)])
for n in SCREEN_MASKS:
    add("screen/masks/", [(n + ".json", FM/(n + ".json"))])

add("old-screen/", [(x, SD/x) for x in ("plan.json", "report.json")])
add("old-screen/", [(x, S/x) for x in ("run.log", "run.stderr", "exit-code", "original.json")])
for i in range(8):
    name = f"{i:02d}.json"
    add("old-screen/", [(name, SD/name)])

add("recovery/", [(x, U/x) for x in ("recovery-receipt.json", "expanded.json", "prop-plan.json", "candidate.json")])
for i in range(18):
    name = f"{i:02d}.json"
    add("prior-r23g/", [(name, P/name)])

add("corpus/", [("corpus.jsonl", CORPUS), ("records.jsonl", RECORDS)])

dest = pathlib.Path("/tmp/r23h-all8-factorial-primary-v1.tgz")
with tarfile.open(dest, "w:gz") as t:
    for name, path in sorted(files.items()):
        t.add(path, arcname=name, recursive=False)
with tarfile.open(dest, "r:gz") as t:
    members = t.getmembers()
    assert len(members) == len(files)
    assert {m.name for m in members} == set(files)
raw = dest.read_bytes()
print("ARCHIVE_SHA256=" + hashlib.sha256(raw).hexdigest())
print("FILES=" + str(len(files)))
print("BYTES=" + str(len(raw)))
print("BATCH_INDICES=" + ",".join(map(str, indices)))
