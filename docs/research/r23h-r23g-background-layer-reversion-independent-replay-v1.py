#!/usr/bin/env python3
"""Independent stdlib-only replay of R23H R23G-background reversion receipts."""
import argparse
import hashlib
import io
import json
import math
import pathlib
import tarfile

PLAN_SHA = "52bd4cfdde4025b327ed949ddc54577dc145f256f2580f4233c75d8b1a98a602"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
RUNNER_BLOB = "bfd31208cf0884fbea17d3b1b2b3d9e32f0158c9"
PINS = {
    "run/driver.py": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "run/collector.py": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
    "run/original.json": "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",
}
EXPANDED_BLOB = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
STABLE_SHA = "488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
ORIGINAL_MASK_SHA = "23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
R23G_MASK_SHA = "77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
PREVIOUS_PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
PREVIOUS_REPORT_SHA = "e906a1e16b38902ea3282afc44b9afcc190b1673b3ffb9cdfc0d01d456b77fa1"
NAMES = [
    "original96", "r23g",
    "r23g_revert10", "r23g_revert15", "r23g_revert34",
    "r23g_revert10_15", "r23g_revert10_34", "r23g_revert15_34",
    "r23g_revert10_15_34",
]
CANDIDATES = NAMES[2:]
REVERTIONS = {
    "r23g_revert10": [10],
    "r23g_revert15": [15],
    "r23g_revert34": [34],
    "r23g_revert10_15": [10, 15],
    "r23g_revert10_34": [10, 34],
    "r23g_revert15_34": [15, 34],
    "r23g_revert10_15_34": [10, 15, 34],
}


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
            need(key not in out, "duplicate JSON key in " + label + ": " + key)
            out[key] = value
        return out
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("invalid JSON constant " + value)),
    )


def read_archive(path, expected):
    raw = pathlib.Path(path).read_bytes()
    need(sha(raw) == expected, "archive SHA-256 mismatch")
    need(len(raw) <= 128 * 1024 * 1024, "archive exceeds size limit")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            need(member.isfile(), "non-regular archive member " + member.name)
            name = member.name.replace("\\", "/")
            parts = pathlib.PurePosixPath(name).parts
            need(not name.startswith("/") and ".." not in parts, "unsafe archive path " + name)
            need(member.size <= 32 * 1024 * 1024, "oversize archive member " + name)
            need(name not in files, "duplicate archive member " + name)
            stream = archive.extractfile(member)
            need(stream is not None, "unreadable archive member " + name)
            files[name] = stream.read()
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
    need(group, "empty aggregate group " + name)
    return {
        "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / len(group),
        "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / len(group),
        "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    F = read_archive(args.archive, args.archive_sha256)

    need(get(F, "run/exit-code").decode().strip() == "0", "screen exit code")
    need(get(F, "run/run.stderr") == b"", "screen stderr")
    need(blob(get(F, "run/screen.py")) == RUNNER_BLOB, "exact v3 runner blob")
    for path, expected in PINS.items():
        need(blob(get(F, path)) == expected, "pinned source " + path)

    plan_raw = get(F, "batch/plan.json")
    need(sha(plan_raw) == PLAN_SHA, "frozen plan SHA")
    plan = parse(plan_raw, "plan")
    need(plan["status"] == "R23H_R23G_BACKGROUND_LAYER_REVERSION_PREFILL_SCREEN_ONLY", "plan scope")
    need(plan["teacher"] == TEACHER and plan["holdout"] == "FORBIDDEN", "teacher/Holdout scope")
    need(plan["prior_recovery_sha256"] == RECOVERY_SHA, "R23G recovery binding")
    need(plan["source_previous_factorial_plan_sha256"] == PREVIOUS_PLAN_SHA, "previous plan binding")
    need(plan["source_previous_factorial_report_sha256"] == PREVIOUS_REPORT_SHA, "previous report binding")
    need(plan["reverted_layers"] == [10, 15, 34], "reversion layer set")
    need([x["name"] for x in plan["candidates"]] == CANDIDATES, "candidate order")
    need(len(plan["ids"]) == 16 and len(set(plan["ids"])) == 16, "Fit ID inventory")
    need(plan["directions"] == {"zh-my": 8, "my-zh": 8}, "direction inventory")

    report_raw = get(F, "batch/report.json")
    report = parse(report_raw, "report")
    need(report["status"] == "R23H_R23G_BACKGROUND_REVERSION_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",
         "report scope")
    need(report["plan_sha256"] == PLAN_SHA and report["samples"] == 16, "report binding")
    need(all(x in report["limits"] for x in ("no Decode", "no Fresh", "no Holdout")),
         "report limits")

    expanded_raw = get(F, "input/expanded.json")
    need(blob(expanded_raw) == EXPANDED_BLOB, "expanded Fit source blob")
    expanded = parse(expanded_raw, "expanded Fit")
    rows = [x for x in expanded["fit"] if x["direction"] in ("zh-my", "my-zh")]
    need(len(rows) == 16 and [x["id"] for x in rows] == plan["ids"], "ordered Fit rows")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8
         and sum(x["direction"] == "my-zh" for x in rows) == 8, "direction balance")
    holdout_lines = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout_lines), "Fit/Holdout separation")

    original = parse(get(F, "run/original.json"), "original mask")["candidates"]["96"]["kept"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA, "original96 mask hash")
    stable_raw = get(F, "input/expanded-stable96-v1.json")
    need(sha(stable_raw) == STABLE_SHA, "stable96 source hash")
    stable = jget(F, "input/expanded-stable96-v1.json")["kept"]
    candidate_raw = get(F, "input/candidate.json")
    need(sha(candidate_raw) == CANDIDATE_SHA, "R23G candidate source hash")
    r23g = jget(F, "input/candidate.json")["kept"]
    need(sha(canon(r23g)) == R23G_MASK_SHA, "R23G mask hash")
    need([layer for layer in r23g if r23g[layer] != stable[layer]] == ["38"],
         "R23G/stable96 layer-38 anchor")

    need(sha(get(F, "input/previous-factorial-plan.json")) == PREVIOUS_PLAN_SHA,
         "previous factorial plan artifact")
    need(sha(get(F, "input/previous-factorial-report.json")) == PREVIOUS_REPORT_SHA,
         "previous factorial report artifact")

    masks = {}
    for item in plan["candidates"]:
        name = item["name"]
        reverted = REVERTIONS[name]
        need(item["reverted_layers"] == reverted, name + " reverted layer identity")
        mask = item["kept"]
        check_mask(mask, name)
        expected = {
            str(layer): (original[str(layer)] if layer in reverted else r23g[str(layer)])
            for layer in range(1, 48)
        }
        need(mask == expected, name + " R23G-background mask composition")
        need(sha(canon(mask)) == item["mask_sha256"], name + " mask SHA")
        changed = [layer for layer in range(1, 48) if mask[str(layer)] != r23g[str(layer)]]
        need(changed == reverted, name + " changed-layer set")
        swaps = sum(len(set(mask[str(layer)]) - set(r23g[str(layer)])) for layer in range(1, 48))
        need(swaps == 2 * len(reverted) and swaps == item["swaps_from_r23g"], name + " swap count")
        masks[name] = mask

    recovery_raw = get(F, "input/recovery-receipt.json")
    need(sha(recovery_raw) == RECOVERY_SHA, "R23G recovery receipt hash")
    recovery = parse(recovery_raw, "R23G recovery")
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH"
         and recovery["primary_violations"] == ["zh-my/prefill/kl"]
         and len(recovery["receipt_sha256"]) == 18, "R23G recovery scope")
    prior = {}
    for index, expected in enumerate(recovery["receipt_sha256"]):
        raw = get(F, f"prior-r23g/{index:02d}.json")
        need(sha(raw) == expected, "historical R23G receipt " + str(index))
        item = parse(raw, "historical R23G receipt")
        prior[item["id"]] = item

    corpus_raw = get(F, "corpus/corpus.jsonl")
    need(sha(corpus_raw) == CORPUS_SHA, "calibration corpus hash")
    wanted = set(plan["ids"])
    corpus = {
        item["id"]: item
        for item in (parse(line, "corpus row") for line in corpus_raw.splitlines())
        if item["split"] == "calibration" and item["id"] in wanted
    }
    need(set(corpus) == wanted, "calibration corpus coverage")
    for row in rows:
        need(sha(canon(corpus[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"],
             "prompt identity " + row["id"])

    log_raw = get(F, "run/run.log")
    events = [parse(line, "run event") for line in log_raw.splitlines() if line.startswith(b"{")]
    plan_events = [x for x in events if x.get("status") == "R23H_R23G_REVERSION_PLAN_FROZEN"]
    need(len(plan_events) == 1 and plan_events[0]["sha256"] == PLAN_SHA, "plan event")
    need(len([x for x in events if x.get("status") == "R23H_R23G_REVERSION_LOCK_ACQUIRED"]) == 1,
         "exclusive lock event")
    preflight = [x for x in events if x.get("status") == "R23H_R23G_REVERSION_RESOURCE_PREFLIGHT"]
    need(len(preflight) == 1 and preflight[0]["blockers"] == []
         and preflight[0]["gpu_pids"] == [] and preflight[0]["model_processes"] == [],
         "resource preflight")
    identity = [x for x in events if x.get("status") == "R23H_R23G_REVERSION_TEACHER_IDENTITY"]
    need(len(identity) == 1 and identity[0]["teacher"] == TEACHER
         and identity[0]["num_experts"] == 128
         and identity[0]["identity"]["kind"] == "pinned-snapshot", "teacher identity")
    ready = [x for x in events if x.get("status") == "R23H_R23G_REVERSION_TEACHER_READY"]
    need(len(ready) == 1 and ready[0]["teacher"] == TEACHER, "teacher ready")
    checks = [x for x in events if x.get("status") == "R23H_R23G_REVERSION_TEACHER_HASH_CHECK"]
    expected_checks = sum(row["id"] in prior for row in rows)
    need(len(checks) == expected_checks and all(x["match"] and x["actual"] == x["expected"] for x in checks),
         "historical teacher-logit checks")
    captures = [x for x in events if x.get("status") == "R23H_R23G_REVERSION_PREFILL_CAPTURED"]
    need(len(captures) == 16 and [x["index"] for x in captures] == list(range(16))
         and [x["id"] for x in captures] == plan["ids"], "capture inventory/order")
    terminal = [x for x in events if x.get("status") == report["status"]]
    need(len(terminal) == 1 and terminal[0]["report_sha256"] == sha(report_raw), "terminal report event")

    receipts = []
    for index, row in enumerate(rows):
        raw = get(F, f"batch/{index:02d}.json")
        need(sha(raw) == captures[index]["sha256"], "receipt hash " + str(index))
        receipt = parse(raw, "receipt " + str(index))
        need(receipt["id"] == row["id"] and receipt["direction"] == row["direction"]
             and receipt["index"] == index and receipt["plan_sha256"] == PLAN_SHA,
             "receipt identity " + str(index))
        need(receipt["teacher_sha256"] == TEACHER, "teacher hash " + row["id"])
        need(set(receipt["metrics"]) == set(NAMES), "metric inventory " + row["id"])
        for name in NAMES:
            q = receipt["metrics"][name]
            need(type(q["kl"]) in (int, float) and math.isfinite(q["kl"]) and q["kl"] >= 0
                 and type(q["nmse"]) in (int, float) and math.isfinite(q["nmse"]) and q["nmse"] >= 0
                 and type(q["argmax_mismatch"]) is int and q["argmax_mismatch"] in (0, 1)
                 and len(q["logits_sha256"]) == 64, "metric validity " + name + "/" + row["id"])
        need(receipt["metrics"]["original96"]["mask_sha256"] == ORIGINAL_MASK_SHA,
             "original96 mask binding " + row["id"])
        need(receipt["metrics"]["r23g"]["mask_sha256"] == R23G_MASK_SHA, "R23G mask binding " + row["id"])
        for name, mask in masks.items():
            need(receipt["metrics"][name]["mask_sha256"] == sha(canon(mask)),
                 "candidate mask binding " + name + "/" + row["id"])
        if row["id"] in prior:
            old = prior[row["id"]]
            need(receipt["teacher_sha256"] == old["original96"]["teacher_hashes"][0]["sha256"],
                 "historical teacher identity " + row["id"])
            for name, old_name in (("original96", "original96"), ("r23g", "candidate96")):
                q = receipt["metrics"][name]
                old_q = old[old_name]["logits"][0]
                need(q["logits_sha256"] == old[old_name]["masked_hashes"][0]["sha256"]
                     and abs(q["kl"] - old_q["teacher_to_masked_kl"]) < 1e-12
                     and abs(q["nmse"] - old_q["nmse"]) < 1e-12
                     and q["argmax_mismatch"] == int(old_q["teacher_argmax"] != old_q["masked_argmax"]),
                     "historical exact logits replay " + name + "/" + row["id"])
        receipts.append(receipt)

    metrics = {name: aggregate(receipts, name) for name in NAMES}
    by_direction = {}
    for direction in ("zh-my", "my-zh"):
        group = [x for x in receipts if x["direction"] == direction]
        need(len(group) == 8, "direction sample count " + direction)
        by_direction[direction] = {name: aggregate(group, name) for name in NAMES}

    eligible = []
    for name in CANDIDATES:
        zh, zh0 = by_direction["zh-my"][name], by_direction["zh-my"]["original96"]
        my, myg = by_direction["my-zh"][name], by_direction["my-zh"]["r23g"]
        if (zh["kl"] < zh0["kl"]
            and zh["nmse"] <= zh0["nmse"] + 1e-12
            and zh["argmax_mismatch"] <= zh0["argmax_mismatch"]
            and my["kl"] <= myg["kl"] + 1e-12
            and my["nmse"] <= myg["nmse"] + 1e-12
            and my["argmax_mismatch"] <= myg["argmax_mismatch"]):
            eligible.append(name)
    need(report["metrics"] == metrics, "independent overall metrics")
    need(report["by_direction"] == by_direction, "independent direction metrics")
    need(report["eligible_for_full_fit_propagation_only"] == eligible, "independent bidirectional gate")
    need(report["selected"] == (eligible[0] if eligible else None), "independent selection")

    result = {
        "status": "R23H_R23G_REVERSION_BIDIRECTIONAL_PREFILL_INDEPENDENT_REPLAY_PASS",
        "archive_sha256": args.archive_sha256,
        "plan_sha256": PLAN_SHA,
        "report_sha256": sha(report_raw),
        "samples": 16,
        "directions": {"zh-my": 8, "my-zh": 8},
        "eligible_for_full_fit_propagation_only": eligible,
        "selected": report["selected"],
        "metrics": metrics,
        "by_direction": by_direction,
        "decision": (
            "Prefill-only result. Advance only the selected candidate, if any, to a separately gated "
            "full Fit propagation; no Decode/Fresh/Holdout or product admission is implied."
        ),
        "checks": [
            "archive SHA, member inventory, path safety and regular-file validation",
            "exact v3 runner and pinned driver/collector/original mask blobs",
            "plan/report scope, previous factorial provenance and recovery receipt binding",
            "all seven 47x96 R23G-background masks reconstructed from original96/R23G",
            "16 frozen Fit rows, balanced directions, Holdout exclusion and corpus prompt identity",
            "18 historical R23G receipt hashes and all available teacher/original96/R23G logit replays",
            "resource preflight, exact pinned teacher identity and clean run exit/stderr",
            "all 16 receipt hashes, event order, mask bindings and finite metric values",
            "independent overall/direction aggregates and candidate gate recomputation",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
