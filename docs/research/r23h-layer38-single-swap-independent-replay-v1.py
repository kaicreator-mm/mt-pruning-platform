#!/usr/bin/env python3
"""Independent stdlib-only replay of R23H layer-38 single-expert-swap evidence."""
import argparse
import hashlib
import io
import itertools
import json
import math
import pathlib
import tarfile

TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
RUNNER_BLOB = "552ac03e1d31df2b1db419fd1ec854346483840e"
PINS = {
    "run/driver.py": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "run/collector.py": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
    "run/original.json": "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",
}
EXPANDED_BLOB = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
STABLE_SHA = "488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
ORIGINAL_MASK_SHA = "23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
R23G_MASK_SHA = "77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
PREVIOUS_PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
PREVIOUS_REPORT_SHA = "e906a1e16b38902ea3282afc44b9afcc190b1673b3ffb9cdfc0d01d456b77fa1"
NAMES = ["original96", "stable96", "r23g",
         "l38_swap_41_to_89", "l38_swap_41_to_123",
         "l38_swap_43_to_89", "l38_swap_43_to_123"]
SWAPS = {
    "l38_swap_41_to_89": (41, 89),
    "l38_swap_41_to_123": (41, 123),
    "l38_swap_43_to_89": (43, 89),
    "l38_swap_43_to_123": (43, 123),
}
DIRECTIONS = ("zh-my", "my-zh")


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse(raw, label):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key " + label + "/" + key)
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid JSON " + x)))


def archive(path, expected):
    raw = pathlib.Path(path).read_bytes()
    need(len(raw) <= 128 * 1024 * 1024, "archive size limit")
    need(sha(raw) == expected, "archive SHA-256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            need(member.isfile(), "non-regular archive member " + member.name)
            name = member.name.replace("\\", "/")
            parts = pathlib.PurePosixPath(name).parts
            need(not name.startswith("/") and ".." not in parts, "unsafe archive path " + name)
            need(member.size <= 64 * 1024 * 1024, "archive member size limit " + name)
            need(name not in files, "duplicate archive member " + name)
            stream = tar.extractfile(member)
            need(stream is not None, "unreadable archive member " + name)
            files[name] = stream.read()
    need(len(files) == 49, "archive member inventory")
    return files


def get(files, name):
    need(name in files, "missing archive member " + name)
    return files[name]


def jget(files, name):
    return parse(get(files, name), name)


def check_mask(mask, label):
    need(set(mask) == {str(i) for i in range(1, 48)}, label + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96, label + " expert count " + layer)
        need(experts == sorted(set(experts)) and all(type(x) is int and 0 <= x < 128 for x in experts),
             label + " expert order/range " + layer)


def aggregate(group, name):
    need(len(group) in (8, 16), "aggregate count " + name)
    return {
        "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / len(group),
        "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / len(group),
        "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--archive-sha256", required=True)
    args = ap.parse_args()
    F = archive(args.archive, args.archive_sha256)

    need(get(F, "run/exit-v2-code").decode().strip() == "0", "screen exit code")
    need(get(F, "run/run-v2.stderr") == b"", "screen stderr")
    need(blob(get(F, "run/screen.py")) == RUNNER_BLOB, "exact screen runner blob")
    for path, expected in PINS.items():
        need(blob(get(F, path)) == expected, "pinned source " + path)

    events = [parse(line, "run log") for line in get(F, "run/run-v2.log").splitlines()
              if line.startswith(b"{")]
    plan_events = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_PLAN_FROZEN"]
    need(len(plan_events) == 1, "single frozen plan event")
    plan_raw = get(F, "batch/plan.json")
    plan_sha = sha(plan_raw)
    need(plan_events[0]["sha256"] == plan_sha, "plan event binding")
    plan = parse(plan_raw, "plan")
    need(plan["status"] == "R23H_LAYER38_SINGLE_SWAP_PREFILL_SCREEN_ONLY", "plan scope")
    need(plan["teacher"] == TEACHER and plan["holdout"] == "FORBIDDEN", "teacher/Holdout scope")
    need(plan["directions"] == {"zh-my": 8, "my-zh": 8}, "direction counts")
    need(len(plan["ids"]) == 16 and len(set(plan["ids"])) == 16, "16 fixed Fit IDs")
    need([x["name"] for x in plan["candidates"]] == list(SWAPS), "candidate inventory/order")
    need([x["name"] for x in plan["baselines"]] == ["original96", "stable96", "r23g"],
         "baseline inventory")

    identity = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_TEACHER_IDENTITY"]
    preflight = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_RESOURCE_PREFLIGHT"]
    need(len(identity) == len(preflight) == 1, "single identity/resource preflight")
    need(identity[0]["teacher"] == TEACHER and identity[0]["num_experts"] == 128
         and identity[0]["identity"]["teacher"] == TEACHER, "exact model identity")
    need(preflight[0]["blockers"] == [] and preflight[0]["gpu_pids"] == []
         and preflight[0]["model_processes"] == [], "resource preflight")

    report_raw = get(F, "batch/report.json")
    report = parse(report_raw, "report")
    need(report["status"] == "R23H_LAYER38_SINGLE_SWAP_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",
         "report scope")
    need(report["plan_sha256"] == plan_sha and report["samples"] == 16, "report binding")
    need("no Decode" in report["limits"] and "Fresh" in report["limits"]
         and "Holdout" in report["limits"], "report limits")

    expanded_raw = get(F, "input/expanded.json")
    need(blob(expanded_raw) == EXPANDED_BLOB, "expanded Fit source blob")
    expanded = parse(expanded_raw, "expanded Fit")
    rows = [x for x in expanded["fit"] if x["direction"] in DIRECTIONS]
    need(len(rows) == 16 and [x["id"] for x in rows] == plan["ids"], "ordered Fit rows")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8
         and sum(x["direction"] == "my-zh" for x in rows) == 8, "direction balance")
    holdout_lines = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout_lines), "Fit/Holdout separation")

    original = jget(F, "run/original.json")["candidates"]["96"]["kept"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA, "original96 mask")
    stable = jget(F, "input/expanded-stable96-v1.json")["kept"]
    need(sha(get(F, "input/expanded-stable96-v1.json")) == STABLE_SHA, "stable96 source")
    r23g = jget(F, "input/candidate.json")["kept"]
    need(sha(get(F, "input/candidate.json")) == CANDIDATE_SHA
         and sha(canon(r23g)) == R23G_MASK_SHA, "R23G source/mask")
    for label, mask in (("original96", original), ("stable96", stable), ("r23g", r23g)):
        check_mask(mask, label)
    stable_changed_layers = [layer for layer in stable if stable[layer] != original[layer]]
    need(len(stable_changed_layers) == 42, "Stable96 changed-layer count")
    need(len([layer for layer in stable if stable[layer] != original[layer]]) == 42,
         "Stable96 changed-layer count")
    need([layer for layer in r23g if r23g[layer] != stable[layer]] == ["38"],
         "R23G/Stable96 layer-38 anchor")
    need(sorted(set(stable["38"]) - set(r23g["38"])) == [41, 43]
         and sorted(set(r23g["38"]) - set(stable["38"])) == [89, 123],
         "layer-38 expert delta")

    masks = {"original96": original, "stable96": stable, "r23g": r23g}
    baseline_hashes = {x["name"]: x["mask_sha256"] for x in plan["baselines"]}
    for name, mask in masks.items():
        need(baseline_hashes[name] == sha(canon(mask)), "baseline mask binding " + name)
    for item in plan["candidates"]:
        name = item["name"]
        removed, added = SWAPS[name]
        mask = item["kept"]
        check_mask(mask, name)
        layer38 = sorted(set(stable["38"]) - {removed} | {added})
        expected = {str(layer): layer38 if layer == 38 else stable[str(layer)]
                    for layer in range(1, 48)}
        need(mask == expected, name + " exact one-swap mask")
        need(item["removed_expert"] == removed and item["added_expert"] == added,
             name + " swap identity")
        need(item["mask_sha256"] == sha(canon(mask)), name + " mask SHA")
        need(sum(len(set(mask[str(layer)]) - set(stable[str(layer)])) for layer in range(1, 48)) == 1,
             name + " one-swap count")
        masks[name] = mask

    previous_plan_raw = get(F, "input/previous-factorial-plan.json")
    previous_report_raw = get(F, "input/previous-factorial-report.json")
    need(sha(previous_plan_raw) == PREVIOUS_PLAN_SHA
         and sha(previous_report_raw) == PREVIOUS_REPORT_SHA, "previous factorial provenance")
    previous_plan = parse(previous_plan_raw, "previous plan")
    need(previous_plan["ids"] == plan["ids"] and previous_plan["teacher"] == TEACHER,
         "previous plan identity")
    previous_events = [parse(line, "previous log") for line in
                       get(F, "input/previous-factorial-run.log").splitlines() if line.startswith(b"{")]
    previous_capture_events = [x for x in previous_events
                               if x.get("status") == "R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(previous_capture_events) == 16, "previous receipt events")
    previous_receipts = []
    for i, event in enumerate(previous_capture_events):
        raw = get(F, f"input/previous-factorial-receipts/{i:02d}.json")
        need(sha(raw) == event["sha256"], "previous receipt SHA " + str(i))
        item = parse(raw, "previous receipt")
        need(item["id"] == plan["ids"][i], "previous receipt ID " + str(i))
        previous_receipts.append(item)

    corpus_raw = get(F, "input/corpus.jsonl")
    need(sha(corpus_raw) == CORPUS_SHA, "corpus hash")
    corpus = {x["id"]: x for x in (parse(line, "corpus line") for line in corpus_raw.splitlines())
              if x.get("split") == "calibration" and x.get("id") in set(plan["ids"])}
    need(len(corpus) == 16, "corpus coverage")
    for row in rows:
        need(sha(canon(corpus[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"],
             "prompt identity " + row["id"])

    teacher_checks = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_TEACHER_HASH_CHECK"]
    captures = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_PREFILL_CAPTURED"]
    need(len(teacher_checks) == len(captures) == 16, "teacher/capture event count")
    need([x["index"] for x in captures] == list(range(16)), "capture event order")
    receipts = []
    for i, row in enumerate(rows):
        raw = get(F, f"batch/{i:02d}.json")
        need(sha(raw) == captures[i]["sha256"], "receipt SHA " + str(i))
        rec = parse(raw, "receipt")
        old = previous_receipts[i]
        need(rec["id"] == row["id"] == old["id"] and rec["direction"] == row["direction"],
             "receipt identity " + str(i))
        need(rec["index"] == i and rec["plan_sha256"] == plan_sha, "receipt plan/index " + str(i))
        need(rec["teacher_sha256"] == old["teacher_sha256"]
             and teacher_checks[i]["actual"] == teacher_checks[i]["expected"] == old["teacher_sha256"],
             "teacher logits hash " + str(i))
        need(set(rec["metrics"]) == set(NAMES), "metric inventory " + str(i))
        for name in NAMES:
            metric = rec["metrics"][name]
            need(metric["mask_sha256"] == sha(canon(masks[name])), "mask binding " + name)
            need(math.isfinite(metric["kl"]) and metric["kl"] >= 0
                 and math.isfinite(metric["nmse"]) and metric["nmse"] >= 0, "finite metric " + name)
            need(type(metric["argmax_mismatch"]) is int and metric["argmax_mismatch"] in (0, 1),
                 "argmax mismatch " + name)
        for baseline in ("original96", "r23g"):
            for metric in ("kl", "nmse", "argmax_mismatch", "logits_sha256", "mask_sha256"):
                need(rec["metrics"][baseline][metric] == old["metrics"][baseline][metric],
                     "historical baseline replay " + baseline + "/" + str(i) + "/" + metric)
        receipts.append(rec)

    overall = {name: aggregate(receipts, name) for name in NAMES}
    directional = {
        direction: {name: aggregate([x for x in receipts if x["direction"] == direction], name)
                    for name in NAMES}
        for direction in DIRECTIONS
    }
    for name in NAMES:
        for metric, value in overall[name].items():
            need(report["metrics"][name][metric] == value, "overall aggregate " + name + "/" + metric)
        for direction in DIRECTIONS:
            for metric, value in directional[direction][name].items():
                need(report["by_direction"][direction][name][metric] == value,
                     "direction aggregate " + direction + "/" + name + "/" + metric)

    eligible = []
    for name in SWAPS:
        zh, zh0 = directional["zh-my"][name], directional["zh-my"]["original96"]
        my, myg = directional["my-zh"][name], directional["my-zh"]["r23g"]
        if (zh["kl"] < zh0["kl"]
            and zh["nmse"] <= zh0["nmse"] + 1e-12
            and zh["argmax_mismatch"] <= zh0["argmax_mismatch"]
            and my["kl"] <= myg["kl"] + 1e-12
            and my["nmse"] <= myg["nmse"] + 1e-12
            and my["argmax_mismatch"] <= myg["argmax_mismatch"]):
            eligible.append(name)
    need(report["eligible_for_full_fit_propagation_only"] == eligible, "bidirectional gate")
    need(report["selected"] == (eligible[0] if eligible else None), "selected candidate")
    final_events = [x for x in events if x.get("status") == report["status"]]
    need(len(final_events) == 1 and final_events[0]["report_sha256"] == sha(report_raw),
         "report event binding")

    result = {
        "status": "R23H_LAYER38_SINGLE_SWAP_INDEPENDENT_REPLAY_PASS",
        "archive_sha256": args.archive_sha256,
        "plan_sha256": plan_sha,
        "report_sha256": sha(report_raw),
        "samples": 16,
        "directions": {"zh-my": 8, "my-zh": 8},
        "baselines": {name: directional["zh-my"][name] for name in ("original96", "stable96", "r23g")},
        "by_direction": directional,
        "eligible_for_full_fit_propagation_only": eligible,
        "selected": report["selected"],
        "checks": [
            "archive SHA, member inventory, path safety and exact source pins",
            "exact model identity, no-blocker preflight and no active GPU/model processes",
            "fixed balanced Fit rows, Holdout exclusion and prompt identity",
            "Stable96 differs from Original96 in 42 layers; R23G differs from Stable96 only at layer 38",
            "four candidate masks reconstructed as exact one-expert swaps",
            "16 teacher-logit hashes match previous independently captured receipts",
            "Original96 and R23G baseline logits/metrics replay exactly against prior receipts",
            "all seven candidate/baseline metrics and direction aggregates recomputed",
            "bidirectional advancement gate recomputed independently",
        ],
        "decision": "Prefill-only evidence. No Decode/Fresh/Holdout or product admission is implied."
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
