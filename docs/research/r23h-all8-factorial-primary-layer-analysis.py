#!/usr/bin/env python3
"""Layer-local analysis of the R23H factorial primary Fit receipts; no model execution."""
import hashlib
import io
import json
import math
import pathlib
import tarfile

ARCHIVE_SHA = "80f8dc3d8ba4e598652a286e68e8eecfc32e54046855a9a194c8b7e8869b95cd"
REPORT_SHA = "eba41d87b5d1f418923584e778a303bdab0dbd1f78f54702ceb86fdea08fb4da"
NAMES = ("all8_revert15_40_42", "all8_revert15")
DIRS = ("zh-my", "my-zh")
PHASES = ("prefill", "decode")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def avg(values):
    return math.fsum(values) / len(values) if values else None


def read_archive(path):
    raw = pathlib.Path(path).read_bytes()
    if sha(raw) != ARCHIVE_SHA:
        raise ValueError("archive SHA mismatch")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len({item.name for item in members}) != len(members):
            raise ValueError("duplicate tar member")
        return {item.name: archive.extractfile(item).read() for item in members}


def main():
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_factorial_primary.py <factorial-primary.tgz>")
    files = read_archive(sys.argv[1])
    report = strict(files["batch/report.json"])
    if sha(files["batch/report.json"]) != REPORT_SHA:
        raise ValueError("primary report SHA mismatch")
    admission = strict(files["batch/admission.json"])
    original = strict(files["old-screen/original.json"])["candidates"]["96"]["kept"]
    masks = {
        name: strict(files["screen/masks/" + name + ".json"])["mask"]
        for name in NAMES
    }
    recovery = strict(files["recovery/recovery-receipt.json"])
    expanded = strict(files["recovery/expanded.json"])
    prop_plan = strict(files["recovery/prop-plan.json"])
    rows = prop_plan["rows"] + [
        row for row in expanded["fit"]
        if row["direction"] == "zh-my" and row["id"] not in {item["id"] for item in prop_plan["rows"]}
    ]
    selected = [(index, row) for index, row in enumerate(rows) if row["direction"] in DIRS]
    if [index for index, _ in selected] != admission["indices"]:
        raise ValueError("selected index binding mismatch")
    receipts = {name: {} for name in NAMES}
    for index, row in selected:
        for name in NAMES:
            receipts[name][row["id"]] = strict(files["batch/" + name + "-" + f"{index:02d}" + ".json"])

    result = {
        "status": "R23H_FACTORIAL_PRIMARY_LAYER_DIAGNOSTICS_ONLY",
        "archive_sha256": ARCHIVE_SHA,
        "batch_report_sha256": REPORT_SHA,
        "samples": len(selected),
        "candidates": {},
        "pairwise": {},
        "limits": "Layer deltas are correlational diagnostics from fixed Fit trajectories; they do not establish causal layer importance or translation quality.",
    }

    for name in NAMES:
        candidate = masks[name]
        changed = []
        for layer in range(1, 48):
            before = set(original[str(layer)])
            after = set(candidate[str(layer)])
            if before != after:
                changed.append({
                    "layer": layer,
                    "added": sorted(after - before),
                    "removed": sorted(before - after),
                    "overlap": len(before & after),
                })
        directions = {}
        for direction in DIRS:
            directions[direction] = {}
            for phase in PHASES:
                layers = []
                for layer in range(1, 48):
                    stats = {}
                    for field in ("moe_output_nmse", "hidden_input_nmse", "mean_overlap"):
                        original_values = []
                        candidate_values = []
                        for _, row in selected:
                            if row["direction"] != direction:
                                continue
                            capture = receipts[name][row["id"]]
                            original_values.extend(
                                item[field] for item in capture["original96"]["layer"]
                                if item["layer"] == layer and item["phase"] == phase
                            )
                            candidate_values.extend(
                                item[field] for item in capture["candidate96"]["layer"]
                                if item["layer"] == layer and item["phase"] == phase
                            )
                        stats[field] = {
                            "original96": avg(original_values),
                            "candidate96": avg(candidate_values),
                            "delta": avg(candidate_values) - avg(original_values)
                            if original_values and candidate_values else None,
                            "positions": len(candidate_values),
                        }
                    original_exact = []
                    candidate_exact = []
                    for _, row in selected:
                        if row["direction"] != direction:
                            continue
                        capture = receipts[name][row["id"]]
                        original_exact.extend(
                            item["exact_sets"] for item in capture["original96"]["layer"]
                            if item["layer"] == layer and item["phase"] == phase
                        )
                        candidate_exact.extend(
                            item["exact_sets"] for item in capture["candidate96"]["layer"]
                            if item["layer"] == layer and item["phase"] == phase
                        )
                    stats["exact_sets"] = {
                        "original96": sum(original_exact),
                        "candidate96": sum(candidate_exact),
                        "delta": sum(candidate_exact) - sum(original_exact),
                        "positions": len(candidate_exact),
                    }
                    layers.append({"layer": layer, **stats})
                directions[direction][phase] = {
                    "largest_moe_output_regressions": sorted(
                        [x for x in layers if x["moe_output_nmse"]["delta"] is not None],
                        key=lambda x: x["moe_output_nmse"]["delta"], reverse=True
                    )[:12],
                    "largest_moe_output_improvements": sorted(
                        [x for x in layers if x["moe_output_nmse"]["delta"] is not None],
                        key=lambda x: x["moe_output_nmse"]["delta"]
                    )[:12],
                    "largest_hidden_input_regressions": sorted(
                        [x for x in layers if x["hidden_input_nmse"]["delta"] is not None],
                        key=lambda x: x["hidden_input_nmse"]["delta"], reverse=True
                    )[:12],
                    "all_layers": layers,
                }
        result["candidates"][name] = {
            "primary_groups": report["groups"][name],
            "violations": report["violations"][name],
            "changed_vs_original96": changed,
            "layer_diagnostics": directions,
        }

    a, b = NAMES
    pair = []
    for layer in range(1, 48):
        left, right = set(masks[a][str(layer)]), set(masks[b][str(layer)])
        if left != right:
            pair.append({
                "layer": layer,
                "only_in_first": sorted(left - right),
                "only_in_second": sorted(right - left),
                "overlap": len(left & right),
            })
    result["pairwise"] = {
        "first": a,
        "second": b,
        "different_layers": pair,
        "different_layer_count": len(pair),
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    output = pathlib.Path(sys.argv[1]).with_name("factorial-primary-layer-diagnostics.json")
    output.write_bytes(raw)
    print(json.dumps({
        "status": result["status"],
        "sha256": sha(raw),
        "pairwise_different_layers": len(pair),
        "candidates": {
            name: {
                "violations": result["candidates"][name]["violations"],
                "changed_vs_original96": [x["layer"] for x in result["candidates"][name]["changed_vs_original96"]],
                "zh-my_decode_top_regressions": [
                    (x["layer"], round(x["moe_output_nmse"]["delta"], 8))
                    for x in result["candidates"][name]["layer_diagnostics"]["zh-my"]["decode"]["largest_moe_output_regressions"][:8]
                ],
                "my-zh_prefill_top_regressions": [
                    (x["layer"], round(x["moe_output_nmse"]["delta"], 8))
                    for x in result["candidates"][name]["layer_diagnostics"]["my-zh"]["prefill"]["largest_moe_output_regressions"][:8]
                ],
            }
            for name in NAMES
        },
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
