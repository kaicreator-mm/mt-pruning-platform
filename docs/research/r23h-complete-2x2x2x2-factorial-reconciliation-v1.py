#!/usr/bin/env python3
"""Join the two complementary R23H screens into a complete 2^4 layer factorial."""
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
FACTORS = ["L10_STABLE_VS_ORIGINAL", "L15_STABLE_VS_ORIGINAL",
           "L34_STABLE_VS_ORIGINAL", "L38_R23G_VS_STABLE"]
DIRECTIONS = ["zh-my", "my-zh"]
METRICS = ["kl", "nmse", "argmax_mismatch"]
EXPECTED_IDS = 16


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


def archive(path, expected):
    raw = pathlib.Path(path).read_bytes()
    need(sha(raw) == expected, "archive SHA mismatch: " + str(path))
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            need(member.isfile(), "non-regular archive member " + member.name)
            name = member.name.replace("\\", "/")
            parts = pathlib.PurePosixPath(name).parts
            need(not name.startswith("/") and ".." not in parts, "unsafe path " + name)
            need(name not in files, "duplicate archive path " + name)
            stream = tar.extractfile(member)
            need(stream is not None, "unreadable archive member " + name)
            files[name] = stream.read()
    return files


def obj(files, path):
    need(path in files, "missing " + path)
    return parse(files[path], path)


def candidate_metrics(receipts, name):
    out = {}
    for direction in DIRECTIONS:
        group = [r for r in receipts if r["direction"] == direction]
        need(len(group) == 8, "direction row count " + direction)
        out[direction] = {
            "kl": math.fsum(r["metrics"][name]["kl"] for r in group) / 8,
            "nmse": math.fsum(r["metrics"][name]["nmse"] for r in group) / 8,
            "argmax_mismatch": sum(r["metrics"][name]["argmax_mismatch"] for r in group),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("previous_archive")
    ap.add_argument("--previous-sha256", default=PREVIOUS_ARCHIVE_SHA)
    ap.add_argument("current_archive")
    ap.add_argument("--current-sha256", required=True)
    args = ap.parse_args()

    A = archive(args.previous_archive, args.previous_sha256)
    B = archive(args.current_archive, args.current_sha256)
    pa_raw, pb_raw = A["batch/plan.json"], B["batch/plan.json"]
    need(sha(pa_raw) == PREVIOUS_PLAN_SHA, "previous plan SHA")
    need(sha(pb_raw) == CURRENT_PLAN_SHA, "current plan SHA")
    pa, pb = parse(pa_raw, "previous plan"), parse(pb_raw, "current plan")
    need(pa["teacher"] == TEACHER and pb["teacher"] == TEACHER, "teacher identity")
    need(pa["holdout"] == "FORBIDDEN" and pb["holdout"] == "FORBIDDEN", "Holdout exclusion")
    need(pa["ids"] == pb["ids"] and len(pa["ids"]) == EXPECTED_IDS, "shared fixed Fit IDs")

    previous_report = obj(A, "batch/report.json")
    current_report = obj(B, "batch/report.json")
    need(sha(A["batch/report.json"]) == PREVIOUS_REPORT_SHA, "previous report SHA")
    need(previous_report["status"] == "R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"
         and previous_report["plan_sha256"] == PREVIOUS_PLAN_SHA, "previous report binding")
    need(current_report["status"] == "R23H_R23G_BACKGROUND_REVERSION_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
         and current_report["plan_sha256"] == CURRENT_PLAN_SHA, "current report binding")

    # Both screens must use the same frozen prompt rows and pinned source corpus.
    for path in ("input/expanded.json", "input/candidate.json", "input/recovery-receipt.json",
                 "input/expanded-stable96-v1.json", "corpus/corpus.jsonl"):
        need(A[path] == B[path], "cross-run source mismatch " + path)
    rows = [x for x in obj(A, "input/expanded.json")["fit"] if x["direction"] in DIRECTIONS]
    need([x["id"] for x in rows] == pa["ids"], "Fit row order")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8
         and sum(x["direction"] == "my-zh" for x in rows) == 8, "balanced directions")
    holdout = {x["line"] for x in obj(A, "input/expanded.json")["holdout"]}
    need(not ({x["line"] for x in rows} & holdout), "Fit/Holdout overlap")

    original = obj(A, "run/original.json")["candidates"]["96"]["kept"]
    stable = obj(A, "input/expanded-stable96-v1.json")["kept"]
    r23g = obj(A, "input/candidate.json")["kept"]
    need(A["run/original.json"] == B["run/original.json"], "original mask source mismatch")
    need([int(k) for k in original if original[k] != stable[k]] == [10, 15, 34],
         "stable96 must differ from original only at layers 10/15/34")
    need([int(k) for k in stable if stable[k] != r23g[k]] == [38],
         "R23G must differ from stable96 only at layer 38")

    old_receipts = [obj(A, f"batch/{i:02d}.json") for i in range(EXPECTED_IDS)]
    new_receipts = [obj(B, f"batch/{i:02d}.json") for i in range(EXPECTED_IDS)]
    need([x["id"] for x in old_receipts] == pa["ids"]
         and [x["id"] for x in new_receipts] == pb["ids"], "receipt ID order")
    for old, new in zip(old_receipts, new_receipts):
        need(old["direction"] == new["direction"], "direction mismatch " + old["id"])
        for baseline in ("original96", "r23g"):
            a, b = old["metrics"][baseline], new["metrics"][baseline]
            need(a["logits_sha256"] == b["logits_sha256"], "baseline logits mismatch " + baseline + "/" + old["id"])
            need(a["mask_sha256"] == b["mask_sha256"], "baseline mask mismatch " + baseline + "/" + old["id"])
            for metric in METRICS:
                need(a[metric] == b[metric], "baseline metric mismatch " + baseline + "/" + old["id"])
        need(old["teacher_sha256"] == new["teacher_sha256"] == TEACHER,
             "teacher logits identity mismatch " + old["id"])

    # Build all 16 cells by mapping stable-layer factors and the independent layer-38 factor.
    previous_masks = {x["name"]: x["kept"] for x in pa["screen"]}
    current_masks = {x["name"]: x["kept"] for x in pb["candidates"]}
    masks_by_cell = {}
    receipts_by_cell = {}
    metrics_by_cell = {}
    for bits in itertools.product((0, 1), repeat=4):
        l10, l15, l34, l38 = bits
        selected = [layer for layer, flag in ((10, l10), (15, l15), (34, l34)) if flag]
        if l38 == 0:
            if not selected:
                name, source, mask, receipts = "original96", "previous", original, old_receipts
            else:
                name = "stable_" + "_".join(map(str, selected))
                source, mask, receipts = "previous", previous_masks[name], old_receipts
        else:
            reverted = [layer for layer, flag in ((10, l10), (15, l15), (34, l34)) if not flag]
            if not reverted:
                name, source, mask, receipts = "r23g", "current", r23g, new_receipts
            else:
                name = "r23g_revert" + "_".join(map(str, reverted))
                source, mask, receipts = "current", current_masks[name], new_receipts
        expected = {
            str(layer): (
                stable[str(layer)] if layer in (10, 15, 34) and bits[(10, 15, 34).index(layer)]
                else original[str(layer)] if layer in (10, 15, 34)
                else r23g[str(layer)] if layer == 38 and l38
                else stable[str(layer)] if layer == 38
                else original[str(layer)]
            )
            for layer in range(1, 48)
        }
        need(mask == expected, "2^4 exact mask composition " + str(bits))
        mask_hash = sha(canon(mask))
        need(mask_hash not in masks_by_cell.values(), "duplicate factorial mask " + str(bits))
        masks_by_cell[bits] = mask_hash
        receipts_by_cell[bits] = receipts
        metrics_by_cell[bits] = {
            direction: {
                "kl": math.fsum(r["metrics"][name]["kl"] for r in receipts if r["direction"] == direction) / 8,
                "nmse": math.fsum(r["metrics"][name]["nmse"] for r in receipts if r["direction"] == direction) / 8,
                "argmax_mismatch": sum(r["metrics"][name]["argmax_mismatch"] for r in receipts if r["direction"] == direction),
            }
            for direction in DIRECTIONS
        }

    need(len(masks_by_cell) == 16, "complete 16-mask factorial")
    baseline_original = metrics_by_cell[(0, 0, 0, 0)]
    baseline_r23g = metrics_by_cell[(1, 1, 1, 1)]
    eligible = []
    candidates = []
    for bits in itertools.product((0, 1), repeat=4):
        name = "original96" if bits == (0, 0, 0, 0) else "r23g" if bits == (1, 1, 1, 1) else (
            "stable_" + "_".join(str(x) for x, b in zip((10, 15, 34), bits[:3]) if b)
            if bits[3] == 0 else
            "r23g_revert" + "_".join(str(x) for x, b in zip((10, 15, 34), bits[:3]) if not b)
        )
        if bits[3] == 0 and bits[:3] == (0, 0, 0):
            name = "original96"
        if bits[3] == 1 and bits[:3] == (1, 1, 1):
            name = "r23g"
        q = metrics_by_cell[bits]
        if bits not in ((0, 0, 0, 0), (1, 1, 1, 1)):
            zh, zh0 = q["zh-my"], baseline_original["zh-my"]
            my, myg = q["my-zh"], baseline_r23g["my-zh"]
            if (zh["kl"] < zh0["kl"]
                and zh["nmse"] <= zh0["nmse"] + 1e-12
                and zh["argmax_mismatch"] <= zh0["argmax_mismatch"]
                and my["kl"] <= myg["kl"] + 1e-12
                and my["nmse"] <= myg["nmse"] + 1e-12
                and my["argmax_mismatch"] <= myg["argmax_mismatch"]):
                eligible.append(name)
            candidates.append({"name": name, "factors": dict(zip(FACTORS, bits)),
                               "mask_sha256": masks_by_cell[bits], "by_direction": q})

    # Full orthogonal decomposition: coefficient for each interaction term is the
    # signed mean; reported effect is twice that coefficient for non-empty terms.
    effects = {}
    for direction in DIRECTIONS:
        effects[direction] = {}
        for metric in METRICS:
            values = {bits: (metrics_by_cell[bits][direction][metric] /
                             (8 if metric == "argmax_mismatch" else 1))
                      for bits in itertools.product((0, 1), repeat=4)}
            terms = {}
            for size in range(0, 5):
                for subset in itertools.combinations(range(4), size):
                    signed = math.fsum(
                        value * math.prod((1 if bits[i] else -1) for i in subset)
                        for bits, value in values.items()
                    ) / 16
                    label = "INTERCEPT" if not subset else ":".join(FACTORS[i] for i in subset)
                    terms[label] = signed if not subset else 2 * signed
            effects[direction][metric] = terms

    result = {
        "status": "R23H_COMPLETE_2X2X2X2_LAYER_FACTORIAL_RECONCILED",
        "previous_archive_sha256": args.previous_sha256,
        "current_archive_sha256": args.current_sha256,
        "previous_plan_sha256": PREVIOUS_PLAN_SHA,
        "current_plan_sha256": CURRENT_PLAN_SHA,
        "samples": EXPECTED_IDS,
        "directions": {"zh-my": 8, "my-zh": 8},
        "factor_definitions": {
            FACTORS[0]: "layer 10 stable96 experts vs original96 experts",
            FACTORS[1]: "layer 15 stable96 experts vs original96 experts",
            FACTORS[2]: "layer 34 stable96 experts vs original96 experts",
            FACTORS[3]: "layer 38 R23G experts vs stable96 experts",
        },
        "cells": [{"factors": dict(zip(FACTORS, bits)), "mask_sha256": masks_by_cell[bits],
                   "by_direction": metrics_by_cell[bits]}
                  for bits in itertools.product((0, 1), repeat=4)],
        "eligible_for_full_fit_propagation_only": eligible,
        "factorial_effects": effects,
        "decision": (
            "Prefill-only factorial evidence. Eligible candidates, if any, may proceed only to a separately "
            "gated full Fit propagation; no Decode/Fresh/Holdout or product admission is implied."
        ),
        "checks": [
            "both evidence archives independently hash-pinned",
            "same frozen 16 Fit IDs, directions, corpus and source inputs",
            "previous/current original96 and R23G baseline logits hashes and metrics match exactly",
            "stable96 differs from original96 only at layers 10/15/34",
            "R23G differs from stable96 only at layer 38",
            "all 16 unique 47x96 masks reconstructed and checked",
            "all direction-specific metrics joined by identical sample ID",
            "full 2^4 orthogonal main-effect and interaction decomposition",
            "bidirectional candidate gate recomputed from the joined factorial cells",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
