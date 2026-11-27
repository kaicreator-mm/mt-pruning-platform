"""Independent validation of R23H's frozen six-mask fit-prefill screen inputs."""
import ast
import hashlib
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(r"D:\mt-pruning-independent")
inputs = base / "input"
def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()
def sha_file(path):
    return sha_bytes(path.read_bytes())
def git_blob(path):
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
def load_json(path):
    raw = path.read_bytes()
    if raw.endswith(bytes((92, 110))):
        raw = raw[:-2]
    elif raw.endswith(bytes((10,))):
        raw = raw[:-1]
    return json.loads(raw.decode("utf-8-sig"))
def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

screen_path = inputs / "r23h-pareto-screen-input.json"
source_path = inputs / "r23h-pareto-v2.json"
screen_script = base / "screen.py"
assert sha_file(screen_path) == "cbdc65097cce87cfc6ddb8934604c690e744b9f3d8f7f52b3708bf035c20fa2f"
assert sha_file(source_path) == "df87768617f17ed7c2c6756400a3ebb36c51269867659c3481a49445dec2bc98"
assert git_blob(screen_script) == "9cc6218f0aa89dab335dfde8aa40ad5ec12c74fd"
ast.parse(screen_script.read_text(encoding="utf-8"))
screen = load_json(screen_path)
source = load_json(source_path)
expanded = load_json(inputs / "expanded.json")
original = load_json(inputs / "original.json")["candidates"]["96"]["kept"]
stable = load_json(inputs / "candidate.json")["kept"]

assert screen["status"] == "R23H_PARETO_ROBUST_FIT_PROXY_CANDIDATES_NOT_ADMISSION"
assert screen["constraints"] == 434
assert screen["candidate_sha256"] == source["candidate_sha256"]
assert screen["sensitivity_sha256"] == source["sensitivity_sha256"]
assert screen["source_report_sha256"] == sha_file(source_path)
source_by_name = {x["name"]: x for x in source["candidates"]}
screen_names = [x["name"] for x in screen["candidates"]]
expected_names = ["all8", "all10", "all12", "all14", "primary4_10", "primary4_14"]
assert screen_names == expected_names and len(screen_names) == 6
assert len(original) == len(stable) == 47
verified = []
for item in screen["candidates"]:
    name = item["name"]
    src = source_by_name[name]
    mask = item["kept"]
    assert sha_bytes(canon(mask)) == item["mask_sha256"] == src["mask_sha256"]
    assert set(mask) == set(original) == set(stable)
    assert item["original96_swaps"] == src["original96_swaps"]
    assert item["folds_passed"] == src["folds_passed"] == 31
    assert abs(item["full_primary_gain"] - src["full_primary_gain"]) < 1e-12
    assert abs(item["minimum_fold_primary_gain"] - src["minimum_fold_primary_gain"]) < 1e-12
    swaps = 0
    for layer in original:
        experts = mask[layer]
        assert len(experts) == 96 and experts == sorted(set(experts))
        assert all(type(x) is int and 0 <= x < 128 for x in experts)
        assert experts in (original[layer], stable[layer])
        swaps += len(set(experts) - set(original[layer]))
    assert swaps == item["original96_swaps"]
    verified.append({"name": name, "swaps": swaps, "mask_sha256": item["mask_sha256"], "folds_passed": 31})

fit = expanded["fit"]
holdout = expanded["holdout"]
fit_lines = {x["line"] for x in fit}
holdout_lines = {x["line"] for x in holdout}
primary = [x for x in fit if x["direction"] == "zh-my"]
assert len(fit) == 48 and len(fit_lines) == 30 and not (fit_lines & holdout_lines)
assert len(primary) == 8 and all(x["id"] in {r["id"] for r in fit} for x in primary)
assert "R23H_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION" in screen_script.read_text(encoding="utf-8")
assert "samples=8" in screen_script.read_text(encoding="utf-8")
assert "holdout" in screen_script.read_text(encoding="utf-8")

result = {
    "status": "R23H_PARETO_SCREEN_INPUT_INDEPENDENT_VALIDATION_PASS_NOT_MODEL_ADMISSION",
    "machine": "DESKTOP-5K3AHE4",
    "screen_script_git_blob": git_blob(screen_script),
    "screen_input_sha256": sha_file(screen_path),
    "source_report_sha256": sha_file(source_path),
    "fit_rows": 48,
    "fit_lines": 30,
    "primary_direction": "zh-my",
    "primary_fit_samples": 8,
    "holdout_lines_disjoint": True,
    "screen_candidates": 6,
    "mask_hash_and_swap_matches": 6,
    "candidate_folds_each": 31,
    "candidates": verified,
    "limits": "Static input/contract validation only; no model inference, decode, holdout, Fresh, translation-quality or pruning admission.",
}
out = base / "screen-input-audit.json"
raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
out.write_bytes(raw)
print(json.dumps(result, sort_keys=True))
print("AUDIT_SHA256=" + sha_bytes(raw))
