"""Cross-machine independent reconciliation of R23H finite fit-only Pareto MILP."""
import hashlib
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(r"D:\mt-pruning-independent")
old_path = base / "input" / "r23h-pareto-v2.json"
new_path = base / "independent-pareto-report-v3.json"
script_path = base / "pareto-r23h-v3.py"
archive_path = base / "input.zip"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path):
    raw = path.read_bytes()
    if raw.endswith(bytes((92, 110))):
        raw = raw[:-2]  # tolerate the historical literal "\\n" suffix in v2 report
    elif raw.endswith(bytes((10,))):
        raw = raw[:-1]
    return json.loads(raw.decode("utf-8-sig"))

old = load_json(old_path)
new = load_json(new_path)
fields = ("mask_sha256", "original96_swaps", "folds_passed")
old_by_name = {x["name"]: x for x in old["candidates"]}
new_by_name = {x["name"]: x for x in new["candidates"]}

assert sha(script_path) == "7e2cfeb9699ae042493ff25a0151d1b529439cb920e1d0620e3c323e0c0dd385"
assert sha(archive_path) == "fbaab5f95fe80232aaba5128205fabb97b1c9c17951c854dd0adf086113d84d8"
assert sha(old_path) == "df87768617f17ed7c2c6756400a3ebb36c51269867659c3481a49445dec2bc98"
assert len(old_by_name) == len(new_by_name) == 10
assert set(old_by_name) == set(new_by_name)
assert old["constraints"] == new["constraints"] == 434
assert old["candidate_sha256"] == new["candidate_sha256"]
assert old["sensitivity_sha256"] == new["sensitivity_sha256"]

candidates = []
for name in sorted(old_by_name):
    a, b = old_by_name[name], new_by_name[name]
    assert all(a[k] == b[k] for k in fields), (name, "identity fields")
    assert abs(a["full_primary_gain"] - b["full_primary_gain"]) < 1e-12, (name, "full gain")
    assert abs(a["minimum_fold_primary_gain"] - b["minimum_fold_primary_gain"]) < 1e-12, (name, "fold gain")
    assert b["folds_passed"] == 31, (name, "fold count")
    candidates.append({
        "name": name,
        "swaps": b["original96_swaps"],
        "mask_sha256": b["mask_sha256"],
        "folds_passed": b["folds_passed"],
        "full_primary_gain": b["full_primary_gain"],
        "minimum_fold_primary_gain": b["minimum_fold_primary_gain"],
    })

result = {
    "status": "R23H_INDEPENDENT_CROSS_MACHINE_MILP_REPRODUCTION_PASS_NOT_MODEL_ADMISSION",
    "machine": "DESKTOP-5K3AHE4",
    "python": "3.14",
    "numpy": new["numpy"],
    "scipy": new["scipy"],
    "source_numpy": old["numpy"],
    "source_scipy": old["scipy"],
    "solver_script_sha256": sha(script_path),
    "input_archive_sha256": sha(archive_path),
    "source_report_sha256": sha(old_path),
    "independent_report_sha256": sha(new_path),
    "independent_solver_version": "r23h-pareto-robust-local-v3",
    "constraints": 434,
    "candidate_count": 10,
    "candidate_mask_and_metric_exact_matches": 10,
    "all_candidates_fold_checks": 31,
    "candidates": candidates,
    "limitations": "Independent CPU-side replay of finite fit-only MILP. Does not establish actual model quality, fresh/holdout validity, translation quality, or pruning admission.",
}
out = base / "independent-reconciliation.json"
raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
out.write_bytes(raw)
print(json.dumps(result, sort_keys=True))
print("RECON_SHA256=" + hashlib.sha256(raw).hexdigest())
