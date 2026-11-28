#!/usr/bin/env python3
"""Pack and validate R23H layer-38 one-expert-swap Prefill evidence."""
import hashlib
import json
import math
import pathlib
import tarfile

R = pathlib.Path.home() / ".cache/mt-pruning"
RR = R / "r23-results"
RUN = R / "r23h-layer38-single-swap-run-v1"
OUT = RR / "r23h-layer38-single-swap-prefill-v2"
RECOVERY = R / "r23g-recovery-ssd-38ed4798"
PREVIOUS = RR / "r23h-factorial-prefill-v5"
PREVIOUS_RUN = R / "r23h-factorial-run-v5"
CORPUS = R / "run-d01b-r19/corpus/corpus.jsonl"

TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
RUNNER_BLOB = "552ac03e1d31df2b1db419fd1ec854346483840e"
PINS = {
    "driver.py": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "collector.py": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
    "original.json": "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",
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


def read(path, limit=64 * 1024 * 1024):
    need(path.is_file() and not path.is_symlink(), "regular source " + str(path))
    need(path.stat().st_size <= limit, "source size " + str(path))
    return path.read_bytes()


def check_mask(mask, label):
    need(set(mask) == {str(i) for i in range(1, 48)}, label + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96, label + " expert count " + layer)
        need(experts == sorted(set(experts)) and all(type(x) is int and 0 <= x < 128 for x in experts),
             label + " expert order/range " + layer)


def aggregate(group, name):
    need(len(group) == 8, "aggregate count " + name)
    return {
        "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / 8,
        "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / 8,
        "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
    }


def same_metrics(a, b, label):
    for metric in ("kl", "nmse", "argmax_mismatch", "logits_sha256", "mask_sha256"):
        need(a[metric] == b[metric], label + "/" + metric)


def main():
    need((RUN / "exit-v2-code").read_text().strip() == "0", "screen exit code")
    need(read(RUN / "run-v2.stderr") == b"", "screen stderr")
    runner = read(RUN / "screen.py")
    need(blob(runner) == RUNNER_BLOB, "exact screen runner blob")
    for name, expected in PINS.items():
        need(blob(read(RUN / name)) == expected, "pinned dependency " + name)

    log_raw = read(RUN / "run-v2.log")
    events = [parse(line, "run log") for line in log_raw.splitlines() if line.startswith(b"{")]
    plan_events = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_PLAN_FROZEN"]
    need(len(plan_events) == 1, "single frozen plan event")
    plan_raw = read(OUT / "plan.json")
    plan_sha = sha(plan_raw)
    need(plan_events[0]["sha256"] == plan_sha, "frozen plan event binding")
    plan = parse(plan_raw, "plan")
    need(plan["status"] == "R23H_LAYER38_SINGLE_SWAP_PREFILL_SCREEN_ONLY", "plan scope")
    need(plan["teacher"] == TEACHER and plan["holdout"] == "FORBIDDEN", "teacher/Holdout scope")
    need(plan["directions"] == {"zh-my": 8, "my-zh": 8}, "direction counts")
    need(len(plan["ids"]) == 16 and len(set(plan["ids"])) == 16, "16 fixed Fit IDs")
    need([x["name"] for x in plan["candidates"]] == list(SWAPS), "candidate inventory/order")
    need([x["name"] for x in plan["baselines"]] == ["original96", "stable96", "r23g"], "baseline inventory")

    identity = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_TEACHER_IDENTITY"]
    preflight = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_RESOURCE_PREFLIGHT"]
    need(len(identity) == len(preflight) == 1, "single identity and resource preflight")
    need(identity[0]["teacher"] == TEACHER and identity[0]["num_experts"] == 128
         and identity[0]["identity"]["teacher"] == TEACHER, "exact model identity")
    need(preflight[0]["blockers"] == [] and preflight[0]["gpu_pids"] == []
         and preflight[0]["model_processes"] == [], "resource preflight")

    report_raw = read(OUT / "report.json")
    report = parse(report_raw, "report")
    need(report["status"] == "R23H_LAYER38_SINGLE_SWAP_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",
         "report scope")
    need(report["plan_sha256"] == plan_sha and report["samples"] == 16, "report binding")
    need("no Decode" in report["limits"] and "Fresh" in report["limits"]
         and "Holdout" in report["limits"], "report limits")

    expanded_raw = read(RECOVERY / "expanded.json")
    need(blob(expanded_raw) == EXPANDED_BLOB, "expanded Fit source blob")
    expanded = parse(expanded_raw, "expanded Fit")
    rows = [x for x in expanded["fit"] if x["direction"] in DIRECTIONS]
    need(len(rows) == 16 and [x["id"] for x in rows] == plan["ids"], "ordered Fit rows")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8
         and sum(x["direction"] == "my-zh" for x in rows) == 8, "Fit direction balance")
    holdout_lines = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout_lines), "Fit/Holdout separation")

    original_raw = read(RUN / "original.json")
    original = parse(original_raw, "original mask")["candidates"]["96"]["kept"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA, "original96 mask hash")
    stable_raw = read(RR / "expanded-stable96-v1.json")
    need(sha(stable_raw) == STABLE_SHA, "stable96 source hash")
    stable = parse(stable_raw, "stable96")["kept"]
    candidate_raw = read(RECOVERY / "candidate.json")
    need(sha(candidate_raw) == CANDIDATE_SHA, "R23G candidate source hash")
    r23g = parse(candidate_raw, "R23G candidate")["kept"]
    need(sha(canon(r23g)) == R23G_MASK_SHA, "R23G mask hash")
    for label, mask in (("original96", original), ("stable96", stable), ("r23g", r23g)):
        check_mask(mask, label)
    stable_changed = [int(layer) for layer in stable if stable[layer] != original[layer]]
    need(len(stable_changed) == 42, "Stable96 changed-layer inventory")
    need([layer for layer in r23g if r23g[layer] != stable[layer]] == ["38"],
         "R23G/Stable96 layer-38 anchor")
    need(sorted(set(stable["38"]) - set(r23g["38"])) == [41, 43]
         and sorted(set(r23g["38"]) - set(stable["38"])) == [89, 123],
         "layer-38 expert delta")

    masks = {"original96": original, "stable96": stable, "r23g": r23g}
    expected_baselines = {x["name"]: x["mask_sha256"] for x in plan["baselines"]}
    for name, mask in masks.items():
        need(expected_baselines[name] == sha(canon(mask)), "baseline mask binding " + name)
    for item in plan["candidates"]:
        name = item["name"]
        removed, added = SWAPS[name]
        mask = item["kept"]
        check_mask(mask, name)
        expected_layer = sorted(set(stable["38"]) - {removed} | {added})
        expected = {str(layer): expected_layer if layer == 38 else stable[str(layer)]
                    for layer in range(1, 48)}
        need(mask == expected, name + " exact one-swap mask")
        need(item["removed_expert"] == removed and item["added_expert"] == added,
             name + " swap identity")
        need(item["mask_sha256"] == sha(canon(mask)), name + " mask hash")
        need(sum(len(set(mask[str(layer)]) - set(stable[str(layer)])) for layer in range(1, 48)) == 1,
             name + " exact single expert swap")
        masks[name] = mask

    previous_plan_raw = read(PREVIOUS / "plan.json")
    previous_report_raw = read(PREVIOUS / "report.json")
    need(sha(previous_plan_raw) == PREVIOUS_PLAN_SHA and sha(previous_report_raw) == PREVIOUS_REPORT_SHA,
         "previous factorial provenance")
    previous_plan = parse(previous_plan_raw, "previous plan")
    need(previous_plan["ids"] == plan["ids"], "previous Fit IDs")
    previous_log = read(PREVIOUS_RUN / "run.log")
    previous_events = [parse(line, "previous run log") for line in previous_log.splitlines()
                       if line.startswith(b"{")]
    previous_capture_events = [x for x in previous_events
                                if x.get("status") == "R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(previous_capture_events) == 16, "previous receipt event inventory")
    previous_receipts = []
    for i, event in enumerate(previous_capture_events):
        raw = read(PREVIOUS / f"{i:02d}.json")
        need(sha(raw) == event["sha256"], "previous receipt hash " + str(i))
        item = parse(raw, "previous receipt")
        need(item["id"] == plan["ids"][i], "previous receipt ID " + str(i))
        previous_receipts.append(item)

    corpus_raw = read(CORPUS)
    need(sha(corpus_raw) == CORPUS_SHA, "calibration corpus")
    corpus = {x["id"]: x for x in (parse(line, "corpus line") for line in corpus_raw.splitlines())
              if x.get("split") == "calibration" and x.get("id") in set(plan["ids"])}
    need(len(corpus) == 16, "calibration prompt coverage")
    for row in rows:
        need(sha(canon(corpus[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"],
             "prompt identity " + row["id"])

    teacher_checks = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_TEACHER_HASH_CHECK"]
    captures = [x for x in events if x.get("status") == "R23H_LAYER38_SINGLE_SWAP_PREFILL_CAPTURED"]
    need(len(teacher_checks) == len(captures) == 16, "16 teacher checks and capture events")
    need([x["index"] for x in captures] == list(range(16)), "capture event order")
    receipts = []
    for i, row in enumerate(rows):
        raw = read(OUT / f"{i:02d}.json")
        need(sha(raw) == captures[i]["sha256"], "capture receipt SHA " + str(i))
        rec = parse(raw, "receipt " + str(i))
        old = previous_receipts[i]
        need(rec["id"] == row["id"] == old["id"] and rec["direction"] == row["direction"],
             "receipt ID/direction " + str(i))
        need(rec["plan_sha256"] == plan_sha and rec["index"] == i, "receipt plan/index " + str(i))
        need(rec["teacher_sha256"] == old["teacher_sha256"]
             and teacher_checks[i]["actual"] == teacher_checks[i]["expected"] == old["teacher_sha256"],
             "teacher logits hash " + str(i))
        need(set(rec["metrics"]) == set(NAMES), "candidate metrics inventory " + str(i))
        for name in NAMES:
            m = rec["metrics"][name]
            need(m["mask_sha256"] == sha(canon(masks[name])), "mask binding " + name + "/" + str(i))
            need(math.isfinite(m["kl"]) and m["kl"] >= 0 and math.isfinite(m["nmse"]) and m["nmse"] >= 0,
                 "finite metrics " + name + "/" + str(i))
            need(type(m["argmax_mismatch"]) is int and m["argmax_mismatch"] in (0, 1),
                 "argmax mismatch " + name + "/" + str(i))
        for baseline in ("original96", "r23g"):
            same_metrics(rec["metrics"][baseline], old["metrics"][baseline],
                         "historical exact replay " + baseline + "/" + str(i))
        receipts.append(rec)

    expected_metrics = {name: aggregate(receipts, name) for name in NAMES}
    expected_directional = {
        direction: {
            name: aggregate([x for x in receipts if x["direction"] == direction], name)
            for name in NAMES
        }
        for direction in DIRECTIONS
    }
    for name in NAMES:
        for metric, value in expected_metrics[name].items():
            need(report["metrics"][name][metric] == value, "overall aggregate " + name + "/" + metric)
        for direction in DIRECTIONS:
            for metric, value in expected_directional[direction][name].items():
                need(report["by_direction"][direction][name][metric] == value,
                     "direction aggregate " + direction + "/" + name + "/" + metric)

    eligible = []
    for name in SWAPS:
        zh, zh0 = expected_directional["zh-my"][name], expected_directional["zh-my"]["original96"]
        my, myg = expected_directional["my-zh"][name], expected_directional["my-zh"]["r23g"]
        if (zh["kl"] < zh0["kl"]
            and zh["nmse"] <= zh0["nmse"] + 1e-12
            and zh["argmax_mismatch"] <= zh0["argmax_mismatch"]
            and my["kl"] <= myg["kl"] + 1e-12
            and my["nmse"] <= myg["nmse"] + 1e-12
            and my["argmax_mismatch"] <= myg["argmax_mismatch"]):
            eligible.append(name)
    need(report["eligible_for_full_fit_propagation_only"] == eligible, "independent candidate gate")
    need(report["selected"] == (eligible[0] if eligible else None), "selected candidate")
    final_events = [x for x in events if x.get("status") == report["status"]]
    need(len(final_events) == 1 and final_events[0]["report_sha256"] == sha(report_raw),
         "final report event binding")

    files = {}
    def add(prefix, name, path):
        raw = read(path)
        target = prefix + name
        need(target not in files, "duplicate archive member " + target)
        files[target] = raw

    for name in ("screen.py", "driver.py", "collector.py", "original.json",
                 "run-v2.log", "run-v2.stderr", "exit-v2-code"):
        add("run/", name, RUN / name)
    add("batch/", "plan.json", OUT / "plan.json")
    add("batch/", "report.json", OUT / "report.json")
    for i in range(16):
        add("batch/", f"{i:02d}.json", OUT / f"{i:02d}.json")
        add("input/previous-factorial-receipts/", f"{i:02d}.json", PREVIOUS / f"{i:02d}.json")
    add("input/", "expanded.json", RECOVERY / "expanded.json")
    add("input/", "candidate.json", RECOVERY / "candidate.json")
    add("input/", "recovery-receipt.json", RECOVERY / "recovery-receipt.json")
    add("input/", "expanded-stable96-v1.json", RR / "expanded-stable96-v1.json")
    add("input/", "corpus.jsonl", CORPUS)
    add("input/", "previous-factorial-plan.json", PREVIOUS / "plan.json")
    add("input/", "previous-factorial-report.json", PREVIOUS / "report.json")
    add("input/", "previous-factorial-run.log", PREVIOUS_RUN / "run.log")

    archive_path = RR / "r23h-layer38-single-swap-prefill-v2.tgz"
    need(not archive_path.exists(), "refuse to overwrite evidence archive")
    with tarfile.open(archive_path, mode="w:gz", compresslevel=6) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mode = 0o600
            info.mtime = 0
            import io
            tar.addfile(info, io.BytesIO(files[name]))
    archive_sha = sha(archive_path.read_bytes())
    manifest = {
        "status": report["status"],
        "plan_sha256": plan_sha,
        "report_sha256": sha(report_raw),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "archive_files": len(files),
        "runner_git_blob": RUNNER_BLOB,
        "teacher": TEACHER,
        "samples": 16,
        "directions": {"zh-my": 8, "my-zh": 8},
        "baselines": ["original96", "stable96", "r23g"],
        "candidate_count": 4,
        "eligible_for_full_fit_propagation_only": eligible,
        "selected": report["selected"],
        "full_fit_propagation_started": False,
        "decision": "Prefill-only. Advance no candidate unless the independently replayed bidirectional gate passes."
    }
    manifest_path = RR / "r23h-layer38-single-swap-prefill-v2-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "R23H_LAYER38_SINGLE_SWAP_EVIDENCE_PACKED",
                      "manifest": manifest, "archive": str(archive_path)}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
