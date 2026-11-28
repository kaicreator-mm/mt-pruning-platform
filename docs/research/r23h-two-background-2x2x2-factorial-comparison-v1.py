#!/usr/bin/env python3
"""Reconcile two distinct 2^3 screens without conflating their backgrounds."""
import argparse
import hashlib
import io
import itertools
import json
import math
import pathlib
import tarfile

PREVIOUS_ARCHIVE_SHA = "ae78a5b2eabab90b6ff09258fc87d4ee6d8e542068f9b97e69874f904810a4c9"
PREVIOUS_PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
PREVIOUS_REPORT_SHA = "e906a1e16b38902ea3282afc44b9afcc190b1673b3ffb9cdfc0d01d456b77fa1"
CURRENT_PLAN_SHA = "52bd4cfdde4025b327ed949ddc54577dc145f256f2580f4233c75d8b1a98a602"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2edecf0f9"
LAYERS = (10, 15, 34)
DIRECTIONS = ("zh-my", "my-zh")
METRICS = ("kl", "nmse", "argmax_mismatch")


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse(raw, label):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key in " + label + ": " + key)
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid JSON " + x)))


def read_archive(path, expected):
    raw = pathlib.Path(path).read_bytes()
    need(sha(raw) == expected, "archive SHA mismatch: " + str(path))
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            need(member.isfile(), "non-regular archive member " + member.name)
            name = member.name.replace("\\", "/")
            parts = pathlib.PurePosixPath(name).parts
            need(not name.startswith("/") and ".." not in parts, "unsafe archive path " + name)
            need(name not in files, "duplicate archive path " + name)
            stream = tar.extractfile(member)
            need(stream is not None, "unreadable archive member " + name)
            files[name] = stream.read()
    return files


def obj(files, path):
    need(path in files, "missing " + path)
    return parse(files[path], path)


def check_mask(mask, label):
    need(set(mask) == {str(i) for i in range(1, 48)}, label + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96, label + " expert count " + layer)
        need(experts == sorted(set(experts)) and all(type(x) is int and 0 <= x < 128 for x in experts),
             label + " expert ordering/range " + layer)


def aggregate(receipts, name, direction):
    group = [r for r in receipts if r["direction"] == direction]
    need(len(group) == 8, "direction count " + direction)
    return {
        "kl": math.fsum(r["metrics"][name]["kl"] for r in group) / 8,
        "nmse": math.fsum(r["metrics"][name]["nmse"] for r in group) / 8,
        "argmax_mismatch": sum(r["metrics"][name]["argmax_mismatch"] for r in group),
    }


def effects(cells, factors):
    result = {}
    for direction in DIRECTIONS:
        result[direction] = {}
        for metric in METRICS:
            values = {
                bits: cells[bits][direction][metric] / (8 if metric == "argmax_mismatch" else 1)
                for bits in itertools.product((0, 1), repeat=3)
            }
            terms = {}
            for size in range(4):
                for subset in itertools.combinations(range(3), size):
                    signed = math.fsum(
                        value * math.prod(1 if bits[i] else -1 for i in subset)
                        for bits, value in values.items()
                    ) / 8
                    label = "INTERCEPT" if not subset else ":".join(factors[i] for i in subset)
                    terms[label] = signed if not subset else 2 * signed
            result[direction][metric] = terms
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("previous_archive")
    ap.add_argument("--previous-sha256", default=PREVIOUS_ARCHIVE_SHA)
    ap.add_argument("current_archive")
    ap.add_argument("--current-sha256", required=True)
    args = ap.parse_args()

    A = read_archive(args.previous_archive, args.previous_sha256)
    B = read_archive(args.current_archive, args.current_sha256)
    pa_raw, pb_raw = A["batch/plan.json"], B["batch/plan.json"]
    need(sha(pa_raw) == PREVIOUS_PLAN_SHA, "previous plan SHA")
    need(sha(pb_raw) == CURRENT_PLAN_SHA, "current plan SHA")
    pa, pb = parse(pa_raw, "previous plan"), parse(pb_raw, "current plan")
    need(pa["status"] == "R23H_FACTORIAL_FIT_PREFILL_SCREEN_ONLY"
         and pb["status"] == "R23H_R23G_BACKGROUND_LAYER_REVERSION_PREFILL_SCREEN_ONLY",
         "two distinct screen contracts")
    need(pa["teacher"] == TEACHER and pb["teacher"] == TEACHER, "teacher identity")
    need(pa["holdout"] == "FORBIDDEN" and pb["holdout"] == "FORBIDDEN", "Holdout exclusion")
    need(pa["ids"] == pb["ids"] and len(pa["ids"]) == 16, "shared frozen Fit IDs")

    ar, br = obj(A, "batch/report.json"), obj(B, "batch/report.json")
    need(sha(A["batch/report.json"]) == PREVIOUS_REPORT_SHA, "previous report SHA")
    need(ar["status"] == "R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"
         and ar["plan_sha256"] == PREVIOUS_PLAN_SHA, "previous report scope")
    need(br["status"] == "R23H_R23G_BACKGROUND_REVERSION_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
         and br["plan_sha256"] == CURRENT_PLAN_SHA, "current report scope")

    common = ("input/expanded.json", "input/candidate.json", "input/recovery-receipt.json",
              "input/expanded-stable96-v1.json", "corpus/corpus.jsonl", "run/original.json")
    for path in common:
        need(A[path] == B[path], "cross-run source mismatch " + path)

    expanded = obj(A, "input/expanded.json")
    rows = [x for x in expanded["fit"] if x["direction"] in DIRECTIONS]
    need([x["id"] for x in rows] == pa["ids"], "Fit row order")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8
         and sum(x["direction"] == "my-zh" for x in rows) == 8, "balanced directions")
    holdout = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout), "Fit/Holdout overlap")

    original = obj(A, "run/original.json")["candidates"]["96"]["kept"]
    stable = obj(A, "input/expanded-stable96-v1.json")["kept"]
    r23g = obj(A, "input/candidate.json")["kept"]
    check_mask(original, "original96")
    check_mask(stable, "stable96")
    check_mask(r23g, "R23G")
    stable_changed = [int(layer) for layer in original if original[layer] != stable[layer]]
    r23g_changed = [int(layer) for layer in stable if stable[layer] != r23g[layer]]
    need(r23g_changed == [38], "R23G differs from Stable96 only at layer 38")
    need(sorted(set(stable["38"]) - set(r23g["38"])) == [41, 43]
         and sorted(set(r23g["38"]) - set(stable["38"])) == [89, 123],
         "layer-38 R23G expert identity")

    old_receipts = [obj(A, f"batch/{i:02d}.json") for i in range(16)]
    new_receipts = [obj(B, f"batch/{i:02d}.json") for i in range(16)]
    need([x["id"] for x in old_receipts] == pa["ids"]
         and [x["id"] for x in new_receipts] == pb["ids"], "receipt order")
    for old, new in zip(old_receipts, new_receipts):
        need(old["direction"] == new["direction"], "direction mismatch " + old["id"])
        need(old["teacher_sha256"] == new["teacher_sha256"] == TEACHER,
             "teacher hash mismatch " + old["id"])
        for baseline in ("original96", "r23g"):
            x, y = old["metrics"][baseline], new["metrics"][baseline]
            need(x["logits_sha256"] == y["logits_sha256"], "baseline logits mismatch " + baseline + "/" + old["id"])
            need(x["mask_sha256"] == y["mask_sha256"], "baseline mask mismatch " + baseline + "/" + old["id"])
            for metric in METRICS:
                need(x[metric] == y[metric], "baseline metric mismatch " + baseline + "/" + old["id"])

    # Screen A: original96 is the background; selected Stable96 layers are applied sparsely.
    old_masks = {x["name"]: x["kept"] for x in pa["screen"]}
    old_cells = {}
    old_cell_names = {}
    old_effects_data = {}
    for bits in itertools.product((0, 1), repeat=3):
        selected = [layer for layer, flag in zip(LAYERS, bits) if flag]
        if selected:
            name = "stable_" + "_".join(map(str, selected))
            mask = old_masks[name]
        else:
            name, mask = "original96", original
        expected = {
            str(layer): stable[str(layer)] if layer in selected else original[str(layer)]
            for layer in range(1, 48)
        }
        check_mask(mask, "sparse/" + name)
        need(mask == expected, "sparse background mask composition " + name)
        old_cells[bits] = {d: aggregate(old_receipts, name, d) for d in DIRECTIONS}
        old_cell_names[bits] = name

    # Screen B: R23G is the background; selected layers are reverted to Original96.
    new_masks = {x["name"]: x["kept"] for x in pb["candidates"]}
    new_cells = {}
    new_cell_names = {}
    for bits in itertools.product((0, 1), repeat=3):
        reverted = [layer for layer, flag in zip(LAYERS, bits) if flag]
        if reverted:
            name = "r23g_revert" + "_".join(map(str, reverted))
            mask = new_masks[name]
        else:
            name, mask = "r23g", r23g
        expected = {
            str(layer): original[str(layer)] if layer in reverted else r23g[str(layer)]
            for layer in range(1, 48)
        }
        check_mask(mask, "R23G-background/" + name)
        need(mask == expected, "R23G background mask composition " + name)
        new_cells[bits] = {d: aggregate(new_receipts, name, d) for d in DIRECTIONS}
        new_cell_names[bits] = name

    # Reconcile both shared baselines independently of their different screen backgrounds.
    baseline_metrics = {
        "original96": {d: aggregate(old_receipts, "original96", d) for d in DIRECTIONS},
        "r23g": {d: aggregate(old_receipts, "r23g", d) for d in DIRECTIONS},
    }
    candidates = []
    eligible = []
    for bits, name in old_cell_names.items():
        if name == "original96":
            continue
        q = old_cells[bits]
        candidates.append({"background": "sparse_original96", "name": name,
                           "factors": dict(zip(("L10_STABLE", "L15_STABLE", "L34_STABLE"), bits)),
                           "by_direction": q})
    for bits, name in new_cell_names.items():
        if name == "r23g":
            continue
        q = new_cells[bits]
        candidates.append({"background": "full_R23G", "name": name,
                           "factors": dict(zip(("L10_REVERT", "L15_REVERT", "L34_REVERT"), bits)),
                           "by_direction": q})
    for item in candidates:
        zh, zh0 = item["by_direction"]["zh-my"], baseline_metrics["original96"]["zh-my"]
        my, myg = item["by_direction"]["my-zh"], baseline_metrics["r23g"]["my-zh"]
        if (zh["kl"] < zh0["kl"]
            and zh["nmse"] <= zh0["nmse"] + 1e-12
            and zh["argmax_mismatch"] <= zh0["argmax_mismatch"]
            and my["kl"] <= myg["kl"] + 1e-12
            and my["nmse"] <= myg["nmse"] + 1e-12
            and my["argmax_mismatch"] <= myg["argmax_mismatch"]):
            eligible.append(item["background"] + ":" + item["name"])

    result = {
        "status": "R23H_TWO_BACKGROUND_PAIRED_2X2X2_FACTORIALS_RECONCILED",
        "previous_archive_sha256": args.previous_sha256,
        "current_archive_sha256": args.current_sha256,
        "previous_plan_sha256": PREVIOUS_PLAN_SHA,
        "current_plan_sha256": CURRENT_PLAN_SHA,
        "samples": 16,
        "directions": {"zh-my": 8, "my-zh": 8},
        "backgrounds": {
            "sparse_original96": {
                "definition": "Original96 in every layer except selected stable changes in layers 10/15/34",
                "stable96_layers_changed_vs_original": stable_changed,
                "factor_semantics": "1=apply Stable96 layer, 0=retain Original96 layer",
                "cells": [{"factors": dict(zip(("L10_STABLE", "L15_STABLE", "L34_STABLE"), bits)),
                           "mask_name": old_cell_names[bits], "by_direction": old_cells[bits]}
                          for bits in itertools.product((0, 1), repeat=3)],
                "factorial_effects": effects(old_cells, ("L10_STABLE", "L15_STABLE", "L34_STABLE")),
            },
            "full_R23G": {
                "definition": "Full R23G mask in every layer except selected reverts of layers 10/15/34 to Original96",
                "r23g_layers_changed_vs_stable96": r23g_changed,
                "layer38_stable_experts": stable["38"],
                "layer38_r23g_experts": r23g["38"],
                "factor_semantics": "1=revert selected layer to Original96, 0=retain R23G layer",
                "cells": [{"factors": dict(zip(("L10_REVERT", "L15_REVERT", "L34_REVERT"), bits)),
                           "mask_name": new_cell_names[bits], "by_direction": new_cells[bits]}
                          for bits in itertools.product((0, 1), repeat=3)],
                "factorial_effects": effects(new_cells, ("L10_REVERT", "L15_REVERT", "L34_REVERT")),
            },
        },
        "shared_baselines": baseline_metrics,
        "cross_background_candidate_count": len(candidates),
        "eligible_for_full_fit_propagation_only": eligible,
        "candidates": candidates,
        "interpretation_limit": (
            "These are two separate 2^3 factorials under materially different backgrounds, not one 2^4 factorial. "
            "The background contrast is confounded by all other Stable96/R23G layer changes and cannot be "
            "interpreted as an isolated layer-38 effect."
        ),
        "decision": (
            "Prefill-only comparative evidence. Eligible candidates, if any, may proceed only to a separately "
            "gated full Fit propagation; no Decode/Fresh/Holdout or product admission is implied."
        ),
        "checks": [
            "both archive hashes and exact plan/report bindings",
            "same frozen 16 Fit IDs, directions and source files",
            "Original96/R23G baseline logits, masks and metrics match exactly across runs",
            "all sparse Original96-background masks reconstructed",
            "all full R23G-background reversion masks reconstructed",
            "two independent 2^3 effect decompositions; no invalid 2^4 assumption",
            "shared-baseline bidirectional candidate gate recomputed",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
