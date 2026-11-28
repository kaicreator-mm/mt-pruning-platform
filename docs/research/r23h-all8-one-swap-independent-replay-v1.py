#!/usr/bin/env python3
"""Independent stdlib-only replay of R23H all8 one-expert-swap evidence."""
import argparse, hashlib, io, json, math, pathlib, tarfile

TEACHER = "0d1fb958cefa104dfc2f187cbb74da2edecf0f9"
RUNNER_BLOB = "0fc170a1861340153663bc0fdd8b9f65f0fc336a"
PINS = {
    "run/driver.py": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "run/collector.py": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
    "run/original.json": "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",
    "run/all8_revert10.json": "faa4d6146a3df4192e913e2702a926afa3fb095d",
    "run/all8_revert15.json": "044a6f395d779489462df00a664cf7876732ab06",
    "run/candidate-inventory.json": "65e57a3add8cff36979b06928af22bcdcefe3ce6",
}
EXPANDED_BLOB = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_BLOB = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
PREVIOUS_PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
PREVIOUS_REPORT_SHA = "e906a1e16b38902ea3282afc44b9afcc190b1673b3ffb9cdfc0d01d456b77fa1"
ORIGINAL_MASK_SHA = "23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
ALL8_MASK_SHA = "1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b"
R23G_MASK_SHA = "77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
LAYERS = (10, 15, 40, 42)
EXPECTED_CHANGES = {
    10: ([42, 95], [78, 100]),
    15: ([6, 64], [4, 71]),
    40: ([13, 84], [58, 62]),
    42: ([47, 120], [63, 126]),
}
BASELINES = ["original96", "all8", "r23g"]
CANDIDATE_ORDER = [
    f"l{layer}_remove{remove}_add{restore}"
    for layer in LAYERS
    for remove in EXPECTED_CHANGES[layer][1]
    for restore in EXPECTED_CHANGES[layer][0]
]
ALL_NAMES = BASELINES + CANDIDATE_ORDER


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw, label):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key " + label + "/" + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("invalid JSON " + value)))


def archive(path, expected_sha):
    raw = pathlib.Path(path).read_bytes()
    need(len(raw) <= 128 * 1024 * 1024, "archive size limit")
    need(sha(raw) == expected_sha, "archive SHA-256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            need(member.isfile(), "non-regular archive member " + member.name)
            name = member.name.replace("\\", "/")
            parts = pathlib.PurePosixPath(name).parts
            need(not name.startswith("/") and ".." not in parts, "unsafe archive path " + name)
            need(member.size <= 64 * 1024 * 1024, "archive member size " + name)
            need(name not in files, "duplicate archive member " + name)
            stream = tar.extractfile(member)
            need(stream is not None, "unreadable archive member " + name)
            files[name] = stream.read()
    need(len(files) == 51, "archive member inventory")
    return files


def get(files, name):
    need(name in files, "missing archive member " + name)
    return files[name]


def jget(files, name):
    return strict_json(get(files, name), name)


def check_mask(mask, label):
    need(set(mask) == {str(i) for i in range(1, 48)}, label + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96, label + " expert count " + layer)
        need(experts == sorted(set(experts))
             and all(type(x) is int and 0 <= x < 128 for x in experts),
             label + " expert order/range " + layer)


def build_masks(files):
    original = jget(files, "run/original.json")["candidates"]["96"]["kept"]
    revert10 = jget(files, "run/all8_revert10.json")
    revert15 = jget(files, "run/all8_revert15.json")
    inventory = jget(files, "run/candidate-inventory.json")
    all8 = dict(revert10["mask"])
    all8["10"] = revert15["mask"]["10"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA, "original96 mask hash")
    need(sha(canon(all8)) == ALL8_MASK_SHA, "reconstructed all8 mask hash")
    check_mask(original, "original96")
    check_mask(all8, "all8")
    need([int(layer) for layer in all8 if all8[layer] != original[layer]] == list(LAYERS),
         "all8 changed-layer inventory")
    candidates = {}
    for layer in LAYERS:
        removed = sorted(set(original[str(layer)]) - set(all8[str(layer)]))
        added = sorted(set(all8[str(layer)]) - set(original[str(layer)]))
        need((removed, added) == EXPECTED_CHANGES[layer], "layer delta " + str(layer))
        for remove in added:
            for restore in removed:
                name = f"l{layer}_remove{remove}_add{restore}"
                mask = dict(all8)
                mask[str(layer)] = sorted((set(all8[str(layer)]) - {remove}) | {restore})
                check_mask(mask, name)
                need([int(x) for x in mask if mask[x] != all8[x]] == [layer],
                     name + " changed-layer scope")
                candidates[name] = dict(layer=layer, removed_expert=remove,
                                         restored_expert=restore, kept=mask,
                                         mask_sha256=sha(canon(mask)))
    need(inventory["status"] == "R23H_ALL8_ONE_SWAP_CANDIDATE_INVENTORY_FROZEN"
         and inventory["base_mask_sha256"] == ALL8_MASK_SHA
         and inventory["original96_mask_sha256"] == ORIGINAL_MASK_SHA
         and inventory["layers"] == list(LAYERS)
         and len(inventory["candidates"]) == 16
         and list(candidates) == CANDIDATE_ORDER, "candidate inventory header/order")
    for layer in LAYERS:
        delta = inventory["layer_changes"][str(layer)]
        need(delta["removed_from_original96"] == EXPECTED_CHANGES[layer][0]
             and delta["added_by_all8"] == EXPECTED_CHANGES[layer][1],
             "inventory layer delta " + str(layer))
    for frozen, name in zip(inventory["candidates"], CANDIDATE_ORDER):
        actual = candidates[name]
        need(frozen["name"] == name and frozen["layer"] == actual["layer"]
             and frozen["removed_expert"] == actual["removed_expert"]
             and frozen["restored_expert"] == actual["restored_expert"]
             and frozen["mask_sha256"] == actual["mask_sha256"],
             "candidate inventory mask " + name)
    return original, all8, candidates


def aggregate(group, name):
    need(len(group) in (8, 16), "aggregate sample count " + name)
    return {
        "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / len(group),
        "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / len(group),
        "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
    }


def eligibility(by_direction):
    result = []
    for name in CANDIDATE_ORDER:
        passes = True
        strict = False
        for direction in ("zh-my", "my-zh"):
            candidate = by_direction[direction][name]
            for baseline in ("original96", "all8"):
                for field in ("kl", "nmse", "argmax_mismatch"):
                    passes = passes and candidate[field] <= by_direction[direction][baseline][field] + 1e-12
            for field in ("kl", "nmse", "argmax_mismatch"):
                strict = strict or candidate[field] < by_direction[direction]["all8"][field] - 1e-12
        if passes and strict:
            result.append(name)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    F = archive(args.archive, args.archive_sha256)

    need(get(F, "run/exit-v1-code").decode().strip() == "0", "screen exit code")
    need(get(F, "run/run-v1.stderr") == b"", "screen stderr")
    need(blob(get(F, "run/screen.py")) == RUNNER_BLOB, "exact runner blob")
    for path, expected in PINS.items():
        need(blob(get(F, path)) == expected, "pinned source " + path)
    need(blob(get(F, "input/expanded.json")) == EXPANDED_BLOB, "expanded Fit source blob")
    need(blob(get(F, "input/candidate.json")) == CANDIDATE_BLOB, "R23G candidate source blob")
    need(sha(get(F, "input/corpus.jsonl")) == CORPUS_SHA, "calibration corpus SHA")
    need(sha(get(F, "input/previous-factorial-plan.json")) == PREVIOUS_PLAN_SHA
         and sha(get(F, "input/previous-factorial-report.json")) == PREVIOUS_REPORT_SHA,
         "previous factorial plan/report pins")

    original, all8, candidates = build_masks(F)
    r23g = jget(F, "input/candidate.json")["kept"]
    need(sha(canon(r23g)) == R23G_MASK_SHA, "R23G mask hash")
    check_mask(r23g, "R23G")
    masks = {"original96": original, "all8": all8, "r23g": r23g}
    masks.update({name: item["kept"] for name, item in candidates.items()})

    events = [strict_json(line, "run log") for line in get(F, "run/run-v1.log").splitlines()
              if line.startswith(b"{")]
    plan_events = [x for x in events if x.get("status") == "R23H_ALL8_ONE_SWAP_PLAN_FROZEN"]
    identity = [x for x in events if x.get("status") == "R23H_ALL8_ONE_SWAP_TEACHER_IDENTITY"]
    preflight = [x for x in events if x.get("status") == "R23H_ALL8_ONE_SWAP_RESOURCE_PREFLIGHT"]
    need(len(plan_events) == len(identity) == len(preflight) == 1,
         "single plan/identity/preflight")
    need(identity[0]["teacher"] == TEACHER and identity[0]["num_experts"] == 128
         and identity[0]["identity"]["teacher"] == TEACHER, "exact teacher identity")
    need(preflight[0]["blockers"] == [] and preflight[0]["gpu_pids"] == []
         and preflight[0]["model_processes"] == [], "resource preflight")

    plan_raw = get(F, "batch/plan.json")
    plan = strict_json(plan_raw, "plan")
    plan_sha = sha(plan_raw)
    need(plan_events[0]["sha256"] == plan_sha, "plan event binding")
    need(plan["status"] == "R23H_ALL8_ONE_SWAP_BIDIRECTIONAL_PREFILL_SCREEN_ONLY"
         and plan["teacher"] == TEACHER and plan["holdout"] == "FORBIDDEN",
         "plan scope")
    need(plan["directions"] == {"zh-my": 8, "my-zh": 8}
         and len(plan["ids"]) == 16 and len(set(plan["ids"])) == 16,
         "plan Fit inventory")
    need(plan["source_previous_plan_sha256"] == PREVIOUS_PLAN_SHA
         and plan["source_previous_report_sha256"] == PREVIOUS_REPORT_SHA
         and plan["source_original96_mask_sha256"] == ORIGINAL_MASK_SHA
         and plan["source_r23g_mask_sha256"] == R23G_MASK_SHA
         and plan["source_all8_mask_sha256"] == ALL8_MASK_SHA
         and plan["source_candidate_inventory_blob"] == PINS["run/candidate-inventory.json"]
         and plan["source_all8_revert10_blob"] == PINS["run/all8_revert10.json"]
         and plan["source_all8_revert15_blob"] == PINS["run/all8_revert15.json"],
         "plan source pins")
    need(plan["changed_layers"] == list(LAYERS), "plan changed layers")
    need([x["name"] for x in plan["baselines"]] == BASELINES
         and [x["name"] for x in plan["candidates"]] == CANDIDATE_ORDER,
         "plan mask inventory")
    for item in plan["baselines"]:
        need(item["mask_sha256"] == sha(canon(masks[item["name"]])),
             "plan baseline mask " + item["name"])
    for item in plan["candidates"]:
        name = item["name"]
        need(item["mask_sha256"] == sha(canon(masks[name]))
             and item["kept"] == masks[name], "plan candidate mask " + name)

    expanded = jget(F, "input/expanded.json")
    rows = [x for x in expanded["fit"] if x["direction"] in ("zh-my", "my-zh")]
    need(len(rows) == 16 and [x["id"] for x in rows] == plan["ids"], "ordered Fit rows")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8
         and sum(x["direction"] == "my-zh" for x in rows) == 8, "direction balance")
    holdout_lines = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout_lines), "Fit/Holdout separation")
    corpus = {
        x["id"]: x for x in
        (strict_json(line, "corpus line") for line in get(F, "input/corpus.jsonl").splitlines())
        if x.get("split") == "calibration" and x.get("id") in set(plan["ids"])
    }
    need(len(corpus) == 16, "corpus prompt coverage")
    for row in rows:
        need(sha(canon(corpus[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"],
             "prompt identity " + row["id"])

    previous_plan = jget(F, "input/previous-factorial-plan.json")
    previous_report = jget(F, "input/previous-factorial-report.json")
    need(previous_plan["status"] == "R23H_FACTORIAL_FIT_PREFILL_SCREEN_ONLY"
         and previous_report["status"] == "R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"
         and previous_plan["ids"] == plan["ids"] and previous_plan["teacher"] == TEACHER,
         "previous factorial provenance")
    previous_events = [strict_json(line, "previous run log")
                       for line in get(F, "input/previous-factorial-run.log").splitlines()
                       if line.startswith(b"{")]
    previous_captures = [x for x in previous_events
                         if x.get("status") == "R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(previous_captures) == 16
         and [x["index"] for x in previous_captures] == list(range(16)),
         "previous receipt event count/order")
    previous = []
    for index, event_row in enumerate(previous_captures):
        raw = get(F, f"input/previous-factorial-receipts/{index:02d}.json")
        need(sha(raw) == event_row["sha256"], "previous receipt SHA " + str(index))
        item = strict_json(raw, "previous receipt")
        need(item["id"] == plan["ids"][index], "previous receipt ID")
        previous.append(item)

    teacher_checks = [x for x in events
                      if x.get("status") == "R23H_ALL8_ONE_SWAP_TEACHER_HASH_CHECK"]
    captures = [x for x in events
                if x.get("status") == "R23H_ALL8_ONE_SWAP_PREFILL_CAPTURED"]
    need(len(teacher_checks) == len(captures) == 16, "teacher/capture event count")
    need([x["index"] for x in captures] == list(range(16)), "capture event order")
    receipts = []
    for index, row in enumerate(rows):
        raw = get(F, f"batch/{index:02d}.json")
        need(sha(raw) == captures[index]["sha256"], "current receipt SHA " + str(index))
        receipt = strict_json(raw, "current receipt")
        old = previous[index]
        need(receipt["id"] == row["id"] == old["id"]
             and receipt["direction"] == row["direction"]
             and captures[index]["id"] == row["id"]
             and teacher_checks[index]["id"] == row["id"]
             and teacher_checks[index]["index"] == index
             and teacher_checks[index]["match"] is True,
             "receipt identity " + str(index))
        need(receipt["index"] == index and receipt["plan_sha256"] == plan_sha,
             "receipt plan/index " + str(index))
        need(receipt["teacher_sha256"] == old["teacher_sha256"]
             and teacher_checks[index]["actual"] == teacher_checks[index]["expected"]
             == old["teacher_sha256"], "teacher logits hash " + str(index))
        need(set(receipt["metrics"]) == set(ALL_NAMES), "metric inventory " + str(index))
        for name in ALL_NAMES:
            metric = receipt["metrics"][name]
            need(metric["mask_sha256"] == sha(canon(masks[name])), "mask binding " + name)
            need(type(metric["kl"]) in (int, float) and math.isfinite(metric["kl"])
                 and metric["kl"] >= 0
                 and type(metric["nmse"]) in (int, float) and math.isfinite(metric["nmse"])
                 and metric["nmse"] >= 0
                 and type(metric["logits_sha256"]) is str
                 and len(metric["logits_sha256"]) == 64, "finite metric/hash " + name)
            need(type(metric["argmax_mismatch"]) is int
                 and metric["argmax_mismatch"] in (0, 1), "argmax mismatch " + name)
        for baseline in ("original96", "r23g"):
            for field in ("kl", "nmse", "argmax_mismatch", "logits_sha256", "mask_sha256"):
                need(receipt["metrics"][baseline][field] == old["metrics"][baseline][field],
                     "historical baseline replay " + baseline + "/" + str(index) + "/" + field)
        receipts.append(receipt)

    report_raw = get(F, "batch/report.json")
    report = strict_json(report_raw, "report")
    need(report["status"] == "R23H_ALL8_ONE_SWAP_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
         and report["plan_sha256"] == plan_sha and report["samples"] == 16
         and "no Decode" in report["limits"] and "Fresh" in report["limits"]
         and "Holdout" in report["limits"] and "physical model build" in report["limits"],
         "report scope/limits")
    overall = {name: aggregate(receipts, name) for name in ALL_NAMES}
    by_direction = {
        direction: {
            name: aggregate([x for x in receipts if x["direction"] == direction], name)
            for name in ALL_NAMES
        }
        for direction in ("zh-my", "my-zh")
    }
    for name in ALL_NAMES:
        for field, value in overall[name].items():
            need(report["metrics"][name][field] == value, "overall aggregate " + name + "/" + field)
        for direction in ("zh-my", "my-zh"):
            for field, value in by_direction[direction][name].items():
                need(report["by_direction"][direction][name][field] == value,
                     "direction aggregate " + direction + "/" + name + "/" + field)
    eligible = eligibility(by_direction)
    need(report["eligible_for_primary_fit_only"] == eligible
         and report["selected"] == (eligible[0] if eligible else None),
         "independent candidate gate")
    final_events = [x for x in events if x.get("status") == report["status"]]
    need(len(final_events) == 1 and final_events[0]["report_sha256"] == sha(report_raw),
         "terminal report event binding")

    result = {
        "status": "R23H_ALL8_ONE_SWAP_INDEPENDENT_RECEIPT_REPLAY_PASS",
        "archive_sha256": args.archive_sha256,
        "plan_sha256": plan_sha,
        "report_sha256": sha(report_raw),
        "samples": 16,
        "directions": {"zh-my": 8, "my-zh": 8},
        "candidate_count": 16,
        "metrics": overall,
        "by_direction": by_direction,
        "eligible_for_primary_fit_only": eligible,
        "selected": report["selected"],
        "checks": [
            "archive SHA, 51-file inventory, regular-file and path-safety validation",
            "exact runner/driver/collector/original/mask/inventory source pins",
            "47x96 Original96, all8 and R23G masks; exact four changed-layer deltas",
            "all 16 candidate masks independently reconstructed and hash-pinned",
            "balanced 16-row Fit inventory, Holdout exclusion and corpus prompt identity",
            "previous factorial plan/report/receipts and exact teacher identity",
            "16 current teacher-logit hashes, receipt hashes, mask bindings and event order",
            "historical Original96 and R23G per-row logits/metrics exact replay",
            "independent overall/directional aggregates and candidate gate recomputation",
            "clean process exit, empty stderr and terminal report binding",
        ],
        "decision": "Prefill-only. No Decode/Fresh/Holdout or product admission is implied."
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
