#!/usr/bin/env python3
"""Independent stdlib-only replay of R23H bidirectional factorial prefill receipts."""
import argparse
import hashlib
import io
import json
import math
import pathlib
import tarfile

PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2edecf0f9"  # replaced below by the full pinned identity
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2edecf0f9"
RUNNER_BLOB = "6d44161629838aa22a06fcbad4b6ed0e53a1234b"
PINS = {
    "run/proxy.py": "248737901db40b50efb0a26a9d5ef76d1928538b",
    "run/driver.py": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "run/collector.py": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
    "run/prop.py": "88468e8c2a5905628365da52e1575d5251516512",
    "run/original.json": "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",
}
MIN_SWAP_SHA = "31af9d8aa1f328df05601414533e0e7bf1259faf6a90091d8f823e07e1b6d1a4"
EXPANDED_BLOB = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
STABLE_SHA = "488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
R23G_MASK_SHA = "77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
ORIGINAL_MASK_SHA = "23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
CANDIDATES = [
    "stable_10", "stable_15", "stable_34",
    "stable_10_15", "stable_10_34", "stable_15_34",
    "stable_10_15_34",
]
NAMES = ["original96", "r23g", *CANDIDATES]


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
        need(experts == sorted(set(experts)), label + " expert ordering " + layer)
        need(all(type(x) is int and 0 <= x < 128 for x in experts), label + " expert range " + layer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    F = read_archive(args.archive, args.archive_sha256)

    plan_raw = get(F, "batch/plan.json")
    need(sha(plan_raw) == PLAN_SHA, "plan pin")
    plan = parse(plan_raw, "plan")
    need(plan["status"] == "R23H_FACTORIAL_FIT_PREFILL_SCREEN_ONLY", "plan scope")
    need(plan["holdout"] == "FORBIDDEN" and len(plan["ids"]) == 16 and len(set(plan["ids"])) == 16, "Fit-only ID inventory")
    need(plan["teacher"] == "0d1fb958cefa104dfc2f187cbb74da2edecf0f9", "teacher placeholder guard")
    need([x["name"] for x in plan["screen"]] == CANDIDATES, "candidate order")

    for path, expected in PINS.items():
        need(blob(get(F, path)) == expected, "pinned source " + path)
    runner = get(F, "run/screen.py")
    need(blob(runner) == RUNNER_BLOB, "screen runner source")
    need(b'set_num_threads(8)' in runner and b'threads=8' in runner, "eight-thread exact replay contract")
    need(get(F, "run/exit-code").decode().strip() == "0" and get(F, "run/run.stderr") == b"", "clean run exit/stderr")

    expanded_raw = get(F, "input/expanded.json")
    need(blob(expanded_raw) == EXPANDED_BLOB, "expanded Fit source blob")
    expanded = parse(expanded_raw, "expanded Fit")
    rows = [x for x in expanded["fit"] if x["direction"] in ("zh-my", "my-zh")]
    need(len(rows) == 16 and [x["id"] for x in rows] == plan["ids"], "ordered bidirectional Fit rows")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8 and sum(x["direction"] == "my-zh" for x in rows) == 8, "direction balance")
    holdout_lines = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout_lines), "Fit/Holdout line separation")

    original = parse(get(F, "run/original.json"), "original mask file")["candidates"]["96"]["kept"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA, "original96 mask")
    stable_file = parse(get(F, "input/expanded-stable96-v1.json"), "stable96 mask file")
    need(sha(get(F, "input/expanded-stable96-v1.json")) == STABLE_SHA, "stable96 source pin")
    stable = stable_file["kept"]
    candidate_file = parse(get(F, "input/candidate.json"), "R23G candidate")
    need(sha(get(F, "input/candidate.json")) == CANDIDATE_SHA, "R23G candidate pin")
    r23g = candidate_file["kept"]
    need(sha(canon(r23g)) == R23G_MASK_SHA, "R23G mask hash")

    minimum_raw = get(F, "run/minimum-swap.json")
    need(sha(minimum_raw) == MIN_SWAP_SHA, "minimum-swap report pin")
    minimum = parse(minimum_raw, "minimum-swap report")
    need(
        minimum["status"] == "MINIMUM_SWAP_ROBUST_FIT_LOCAL_DIAGNOSTIC_NOT_ADMISSION"
        and minimum["solver"]["proved_minimum"] is True
        and minimum["original96_swaps"] == 6
        and minimum["stable_layers"] == [10, 15, 34]
        and len(minimum["folds"]) == 31
        and all(x["pass_local_algebra"] for x in minimum["folds"]),
        "minimum-swap solver evidence",
    )

    masks = {}
    for item in plan["screen"]:
        name = item["name"]
        mask = item["kept"]
        check_mask(mask, name)
        layers = item["stable_layers"]
        expected = {str(layer): stable[str(layer)] if layer in layers else original[str(layer)] for layer in range(1, 48)}
        need(mask == expected, "candidate mask composition " + name)
        need(sha(canon(mask)) == item["mask_sha256"], "candidate mask SHA " + name)
        changed = [layer for layer in range(1, 48) if mask[str(layer)] != original[str(layer)]]
        need(changed == layers, "candidate changed-layer set " + name)
        need(sum(len(set(mask[str(layer)]) - set(original[str(layer)])) for layer in range(1, 48)) == 2 * len(layers), "candidate swap count " + name)
        masks[name] = mask

    recovery_raw = get(F, "input/recovery-receipt.json")
    need(sha(recovery_raw) == RECOVERY_SHA, "R23G recovery receipt pin")
    recovery = parse(recovery_raw, "R23G recovery receipt")
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(recovery["receipt_sha256"]) == 18, "R23G recovery scope")
    prior = {}
    for index, expected in enumerate(recovery["receipt_sha256"]):
        raw = get(F, f"prior-r23g/{index:02d}.json")
        need(sha(raw) == expected, "prior R23G receipt SHA " + str(index))
        item = parse(raw, "prior R23G receipt")
        prior[item["id"]] = item

    corpus_raw = get(F, "corpus/corpus.jsonl")
    need(sha(corpus_raw) == CORPUS_SHA, "calibration corpus pin")
    wanted = set(plan["ids"])
    corpus = {
        item["id"]: item
        for item in (parse(line, "corpus row") for line in corpus_raw.splitlines())
        if item["split"] == "calibration" and item["id"] in wanted
    }
    need(set(corpus) == wanted, "corpus prompt coverage")
    for row in rows:
        need(sha(canon(corpus[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"], "prompt identity " + row["id"])

    log_raw = get(F, "run/run.log")
    events = [parse(line, "run event") for line in log_raw.splitlines() if line.startswith(b"{")]
    plan_events = [x for x in events if x.get("status") == "R23H_FACTORIAL_PLAN_FROZEN"]
    need(len(plan_events) == 1 and plan_events[0]["sha256"] == PLAN_SHA, "plan event")
    need(len([x for x in events if x.get("status") == "R23H_LOCK_ACQUIRED"]) == 1, "lock event")
    ready = [x for x in events if x.get("status") == "R23H_TEACHER_READY"]
    need(len(ready) == 1 and ready[0]["teacher"] == plan["teacher"], "teacher ready event")
    checks = [x for x in events if x.get("status") == "R23H_TEACHER_LOGIT_HASH_CHECK"]
    need(len(checks) == 8 and all(x["match"] and x["actual"] == x["expected"] for x in checks), "teacher hash replay checks")
    captures = [x for x in events if x.get("status") == "R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(captures) == 16 and [x["index"] for x in captures] == list(range(16)), "capture event inventory")
    need([x["id"] for x in captures] == plan["ids"], "capture event order")

    report_raw = get(F, "batch/report.json")
    report = parse(report_raw, "factorial report")
    need(
        report["status"] == "R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"
        and report["plan_sha256"] == PLAN_SHA
        and report["samples"] == 16,
        "report scope",
    )
    need(report["limits"].find("no Fresh") >= 0 and report["limits"].find("no decode") >= 0, "report limits")
    terminal = [x for x in events if x.get("status") == report["status"]]
    need(len(terminal) == 1 and terminal[0]["report_sha256"] == sha(report_raw), "terminal report event")

    receipts = []
    for index, row in enumerate(rows):
        raw = get(F, f"batch/{index:02d}.json")
        need(sha(raw) == captures[index]["sha256"], "receipt hash " + str(index))
        receipt = parse(raw, "receipt " + str(index))
        need(
            receipt["id"] == row["id"] and receipt["direction"] == row["direction"]
            and receipt["index"] == index and receipt["plan_sha256"] == PLAN_SHA,
            "receipt identity " + str(index),
        )
        need(len(receipt["teacher_sha256"]) == 64, "teacher logits hash length")
        need(list(receipt["metrics"]) == NAMES, "metric inventory " + row["id"])
        for name in NAMES:
            q = receipt["metrics"][name]
            need(
                type(q["kl"]) in (int, float) and math.isfinite(q["kl"]) and q["kl"] >= 0
                and type(q["nmse"]) in (int, float) and math.isfinite(q["nmse"]) and q["nmse"] >= 0
                and type(q["argmax_mismatch"]) is int and q["argmax_mismatch"] in (0, 1)
                and len(q["logits_sha256"]) == 64,
                "metric validity " + name + "/" + row["id"],
            )
        need(receipt["metrics"]["original96"]["mask_sha256"] == ORIGINAL_MASK_SHA, "original96 metric mask")
        need(receipt["metrics"]["r23g"]["mask_sha256"] == R23G_MASK_SHA, "R23G metric mask")
        for name, mask in masks.items():
            need(receipt["metrics"][name]["mask_sha256"] == sha(canon(mask)), "candidate metric mask " + name)
        if row["direction"] == "zh-my":
            old = prior[row["id"]]
            need(receipt["teacher_sha256"] == old["original96"]["teacher_hashes"][0]["sha256"], "historical teacher logits " + row["id"])
            for name, old_name in (("original96", "original96"), ("r23g", "candidate96")):
                q = receipt["metrics"][name]
                old_q = old[old_name]["logits"][0]
                need(
                    q["logits_sha256"] == old[old_name]["masked_hashes"][0]["sha256"]
                    and abs(q["kl"] - old_q["teacher_to_masked_kl"]) < 1e-12
                    and abs(q["nmse"] - old_q["nmse"]) < 1e-12
                    and q["argmax_mismatch"] == int(old_q["teacher_argmax"] != old_q["masked_argmax"]),
                    "historical exact logits replay " + name + "/" + row["id"],
                )
        receipts.append(receipt)

    def agg(group, name):
        return {
            "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / len(group),
            "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / len(group),
            "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
        }

    metrics = {name: agg(receipts, name) for name in NAMES}
    by_direction = {}
    for direction in ("zh-my", "my-zh"):
        group = [x for x in receipts if x["direction"] == direction]
        need(len(group) == 8, "direction sample count " + direction)
        by_direction[direction] = {name: agg(group, name) for name in NAMES}
    eligible = [
        name for name in CANDIDATES
        if all(
            by_direction[direction][name][key] <= by_direction[direction]["original96"][key] + 1e-12
            for direction in by_direction for key in ("kl", "nmse", "argmax_mismatch")
        )
        and all(by_direction[direction][name]["kl"] < by_direction[direction]["r23g"]["kl"] for direction in by_direction)
    ]
    need(report["metrics"] == metrics and report["by_direction"] == by_direction, "independent aggregate replay")
    need(report["eligible_for_full_fit_propagation_only"] == eligible, "independent gate replay")
    need(report["selected"] == (eligible[0] if eligible else None), "independent selection replay")

    result = {
        "status": "R23H_FACTORIAL_BIDIRECTIONAL_PREFILL_INDEPENDENT_RECEIPT_REPLAY_PASS",
        "archive_sha256": args.archive_sha256,
        "plan_sha256": PLAN_SHA,
        "report_sha256": sha(report_raw),
        "samples": 16,
        "directions": {"zh-my": 8, "my-zh": 8},
        "metrics": metrics,
        "by_direction": by_direction,
        "eligible_for_full_fit_propagation_only": eligible,
        "selected": report["selected"],
        "decision": "prefill-only screen; only a candidate passing both directions may enter the bounded primary Fit propagation",
        "checks": [
            "exact runner/dependency Git blob pins",
            "minimum-swap solver 6-swap and 31-fold provenance",
            "all seven 47x96 candidate masks reconstructed from original96/stable96",
            "16 frozen zh-my/my-zh Fit rows and Holdout exclusion",
            "corpus prompt identity and 18 R23G recovery receipt hashes",
            "eight historical zh-my teacher/original96/R23G logits exact replays",
            "all 16 receipt hashes, candidate mask bindings, finite metrics and capture event order",
            "independent overall/direction aggregates and eligibility gate recomputation",
            "clean process exit, empty stderr, exact teacher identity and terminal report binding",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
