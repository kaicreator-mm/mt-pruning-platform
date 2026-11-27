#!/usr/bin/env python3
"""Pack exact R23H seven-way bidirectional factorial prefill evidence."""
import hashlib
import json
import pathlib
import tarfile

R = pathlib.Path.home() / ".cache/mt-pruning"
RR = R / "r23-results"
RUN = R / "r23h-factorial-run-v5"
OUT = RR / "r23h-factorial-prefill-v5"
RECOVERY_DIR = R / "r23g-recovery-ssd-38ed4798"
PRIOR = RR / "r23g-l38-propagation-v1"
CORPUS = R / "run-d01b-r19/corpus/corpus.jsonl"

PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
RUNNER_BLOB = "6d44161629838aa22a06fcbad4b6ed0e53a1234b"
PINS = {
    "proxy.py": "248737901db40b50efb0a26a9d5ef76d1928538b",
    "driver.py": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "collector.py": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
    "prop.py": "88468e8c2a5905628365da52e1575d5251516512",
    "original.json": "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",
}
MIN_SWAP_SHA = "31af9d8aa1f328df05601414533e0e7bf1259faf6a90091d8f823e07e1b6d1a4"
EXPANDED_BLOB = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
STABLE_SHA = "488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
R23G_MASK_SHA = "77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
ORIGINAL_MASK_SHA = "23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
CANDIDATE_ORDER = [
    "stable_10", "stable_15", "stable_34",
    "stable_10_15", "stable_10_34", "stable_15_34",
    "stable_10_15_34",
]


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def add(files, prefix, entries):
    for name, path in entries:
        need(path.is_file() and not path.is_symlink(), "regular source " + str(path))
        limit = 32 * 1024 * 1024 if name == "corpus.jsonl" else 4 * 1024 * 1024
        need(path.stat().st_size <= limit, "source size " + str(path))
        files[prefix + name] = path


def main():
    need((RUN / "exit-code").read_text().strip() == "0", "screen run exit code")
    need((RUN / "run.stderr").read_bytes() == b"", "screen stderr must be empty")
    need((RUN / "screen.py").is_file() and blob((RUN / "screen.py").read_bytes()) == RUNNER_BLOB, "runner blob")
    for name, expected in PINS.items():
        need(blob((RUN / name).read_bytes()) == expected, "pinned dependency " + name)

    plan_raw = (OUT / "plan.json").read_bytes()
    need(sha(plan_raw) == PLAN_SHA, "factorial plan pin")
    plan = json.loads(plan_raw)
    need(plan["status"] == "R23H_FACTORIAL_FIT_PREFILL_SCREEN_ONLY", "plan status")
    need(plan["teacher"] == TEACHER and plan["holdout"] == "FORBIDDEN", "plan scope")
    need(plan["prior_recovery_sha256"] == RECOVERY_SHA, "recovery binding")
    need([x["name"] for x in plan["screen"]] == CANDIDATE_ORDER, "candidate order")
    need(len(plan["ids"]) == 16 and len(set(plan["ids"])) == 16, "16 frozen Fit IDs")

    report_raw = (OUT / "report.json").read_bytes()
    report = json.loads(report_raw)
    need(report["status"] == "R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION", "report status")
    need(report["plan_sha256"] == PLAN_SHA and report["samples"] == 16, "report plan/sample binding")

    expanded_raw = (RECOVERY_DIR / "expanded.json").read_bytes()
    need(blob(expanded_raw) == EXPANDED_BLOB, "expanded Fit input blob")
    expanded = json.loads(expanded_raw)
    rows = [x for x in expanded["fit"] if x["direction"] in ("zh-my", "my-zh")]
    need(len(rows) == 16 and [x["id"] for x in rows] == plan["ids"], "ordered bidirectional Fit rows")
    holdout_lines = {x["line"] for x in expanded["holdout"]}
    need(not ({x["line"] for x in rows} & holdout_lines), "Fit/Holdout separation")
    need(sum(x["direction"] == "zh-my" for x in rows) == 8 and sum(x["direction"] == "my-zh" for x in rows) == 8, "direction balance")

    original_raw = (RUN / "original.json").read_bytes()
    original = json.loads(original_raw)["candidates"]["96"]["kept"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA, "original96 mask")
    stable_raw = (RR / "expanded-stable96-v1.json").read_bytes()
    need(sha(stable_raw) == STABLE_SHA, "stable96 source")
    stable = json.loads(stable_raw)["kept"]
    candidate_raw = (RECOVERY_DIR / "candidate.json").read_bytes()
    need(sha(candidate_raw) == CANDIDATE_SHA, "R23G candidate source")
    r23g = json.loads(candidate_raw)["kept"]
    need(sha(canon(r23g)) == R23G_MASK_SHA, "R23G candidate mask")
    minimum_raw = (RUN / "minimum-swap.json").read_bytes()
    need(sha(minimum_raw) == MIN_SWAP_SHA, "minimum-swap solver report")
    minimum = json.loads(minimum_raw)
    need(minimum["status"] == "MINIMUM_SWAP_ROBUST_FIT_LOCAL_DIAGNOSTIC_NOT_ADMISSION", "minimum-swap scope")
    need(minimum["solver"]["proved_minimum"] is True and minimum["original96_swaps"] == 6 and minimum["stable_layers"] == [10, 15, 34], "six-swap robust solver result")
    need(len(minimum["folds"]) == 31 and all(x["pass_local_algebra"] for x in minimum["folds"]), "minimum-swap folds")

    masks = {}
    for item in plan["screen"]:
        name = item["name"]
        mask = item["kept"]
        stable_layers = item["stable_layers"]
        need(set(mask) == {str(i) for i in range(1, 48)}, name + " layer inventory")
        for layer, experts in mask.items():
            need(type(experts) is list and len(experts) == 96, name + " expert count " + layer)
            need(experts == sorted(set(experts)) and all(type(e) is int and 0 <= e < 128 for e in experts), name + " expert ordering/range " + layer)
        expected = {str(layer): (stable[str(layer)] if layer in stable_layers else original[str(layer)]) for layer in range(1, 48)}
        need(mask == expected, name + " exact layer-mask composition")
        need(sha(canon(mask)) == item["mask_sha256"], name + " mask hash")
        changed = [layer for layer in range(1, 48) if mask[str(layer)] != original[str(layer)]]
        need(changed == stable_layers, name + " actual changed-layer inventory")
        need(sum(len(set(mask[str(layer)]) - set(original[str(layer)])) for layer in range(1, 48)) == 2 * len(stable_layers), name + " swap count")
        masks[name] = mask

    recovery_raw = (RECOVERY_DIR / "recovery-receipt.json").read_bytes()
    need(sha(recovery_raw) == RECOVERY_SHA, "R23G recovery receipt")
    recovery = json.loads(recovery_raw)
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(recovery["receipt_sha256"]) == 18, "R23G recovery scope")
    prior = {}
    for index, expected in enumerate(recovery["receipt_sha256"]):
        raw = (PRIOR / f"{index:02d}.json").read_bytes()
        need(sha(raw) == expected, "R23G prior receipt " + str(index))
        item = json.loads(raw)
        prior[item["id"]] = item

    corpus_raw = CORPUS.read_bytes()
    need(sha(corpus_raw) == CORPUS_SHA, "calibration corpus")
    wanted = set(plan["ids"])
    corpus = {
        item["id"]: item
        for item in (json.loads(line) for line in corpus_raw.splitlines())
        if item["split"] == "calibration" and item["id"] in wanted
    }
    need(set(corpus) == wanted, "corpus row coverage")
    for row in rows:
        need(sha(canon(corpus[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"], "prompt identity " + row["id"])

    log_raw = (RUN / "run.log").read_bytes()
    events = [json.loads(line) for line in log_raw.splitlines() if line.startswith(b"{")]
    plan_events = [x for x in events if x.get("status") == "R23H_FACTORIAL_PLAN_FROZEN"]
    need(len(plan_events) == 1 and plan_events[0]["sha256"] == PLAN_SHA, "plan event binding")
    need(len([x for x in events if x.get("status") == "R23H_LOCK_ACQUIRED"]) == 1, "exclusive lock event")
    ready = [x for x in events if x.get("status") == "R23H_TEACHER_READY"]
    need(len(ready) == 1 and ready[0]["teacher"] == TEACHER, "teacher ready event")
    checks = [x for x in events if x.get("status") == "R23H_TEACHER_LOGIT_HASH_CHECK"]
    need(len(checks) == 8 and all(x["match"] and x["actual"] == x["expected"] for x in checks), "eight exact teacher-logit checks")
    captures = [x for x in events if x.get("status") == "R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(captures) == 16, "capture event count")
    need([x["index"] for x in captures] == list(range(16)) and [x["id"] for x in captures] == plan["ids"], "capture order")
    terminal = [x for x in events if x.get("status") == report["status"]]
    need(len(terminal) == 1 and terminal[0]["report_sha256"] == sha(report_raw), "terminal report event")

    names = ["original96", "r23g", *CANDIDATE_ORDER]
    receipts = []
    for index, row in enumerate(rows):
        raw = (OUT / f"{index:02d}.json").read_bytes()
        receipt_sha = sha(raw)
        need(captures[index]["sha256"] == receipt_sha, "capture receipt hash " + str(index))
        receipt = json.loads(raw)
        need(
            receipt["id"] == row["id"] and receipt["direction"] == row["direction"]
            and receipt["index"] == index and receipt["plan_sha256"] == PLAN_SHA,
            "receipt identity " + str(index),
        )
        need(receipt["teacher_sha256"] == TEACHER or len(receipt["teacher_sha256"]) == 64, "teacher receipt hash")
        need(list(receipt["metrics"]) == names, "metric inventory " + row["id"])
        for name in names:
            metric = receipt["metrics"][name]
            need(
                type(metric["kl"]) in (int, float) and math.isfinite(metric["kl"]) and metric["kl"] >= 0
                and type(metric["nmse"]) in (int, float) and math.isfinite(metric["nmse"]) and metric["nmse"] >= 0
                and type(metric["argmax_mismatch"]) is int and metric["argmax_mismatch"] in (0, 1)
                and len(metric["logits_sha256"]) == 64,
                "metric validity " + name + "/" + row["id"],
            )
        need(receipt["metrics"]["original96"]["mask_sha256"] == ORIGINAL_MASK_SHA, "original96 binding " + row["id"])
        need(receipt["metrics"]["r23g"]["mask_sha256"] == R23G_MASK_SHA, "R23G binding " + row["id"])
        for name, mask in masks.items():
            need(receipt["metrics"][name]["mask_sha256"] == sha(canon(mask)), "candidate binding " + name + "/" + row["id"])

        if row["id"] in prior:
            old = prior[row["id"]]
            need(receipt["teacher_sha256"] == old["original96"]["teacher_hashes"][0]["sha256"], "teacher baseline " + row["id"])
            for name, old_name in (("original96", "original96"), ("r23g", "candidate96")):
                metric = receipt["metrics"][name]
                old_metric = old[old_name]["logits"][0]
                old_hash = old[old_name]["masked_hashes"][0]["sha256"]
                need(
                    metric["logits_sha256"] == old_hash
                    and abs(metric["kl"] - old_metric["teacher_to_masked_kl"]) < 1e-12
                    and abs(metric["nmse"] - old_metric["nmse"]) < 1e-12
                    and metric["argmax_mismatch"] == int(old_metric["teacher_argmax"] != old_metric["masked_argmax"]),
                    "historical actual logits replay " + name + "/" + row["id"],
                )
        receipts.append(receipt)

    def aggregate(selected):
        return {
            "kl": math.fsum(x["metrics"][selected]["kl"] for x in receipts) / len(receipts),
            "nmse": math.fsum(x["metrics"][selected]["nmse"] for x in receipts) / len(receipts),
            "argmax_mismatch": sum(x["metrics"][selected]["argmax_mismatch"] for x in receipts),
        }

    metrics = {name: aggregate(name) for name in names}
    by_direction = {}
    for direction in ("zh-my", "my-zh"):
        group = [x for x in receipts if x["direction"] == direction]
        by_direction[direction] = {}
        for name in names:
            by_direction[direction][name] = {
                "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / len(group),
                "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / len(group),
                "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
            }

    eligible = [
        name for name in CANDIDATE_ORDER
        if all(
            by_direction[direction][name][key] <= by_direction[direction]["original96"][key] + 1e-12
            for direction in by_direction for key in ("kl", "nmse", "argmax_mismatch")
        )
        and all(by_direction[direction][name]["kl"] < by_direction[direction]["r23g"]["kl"] for direction in by_direction)
    ]
    need(report["metrics"] == metrics, "independent aggregate metrics")
    need(report["by_direction"] == by_direction, "independent direction metrics")
    need(report["eligible_for_full_fit_propagation_only"] == eligible, "independent candidate gate")
    need(report["selected"] == (eligible[0] if eligible else None), "independent selection")

    archive = pathlib.Path("/tmp/r23h-factorial-prefill-v5.tgz")
    files = {}
    add(files, "batch/", [(x, OUT / x) for x in ("plan.json", "report.json", *[f"{i:02d}.json" for i in range(16)])])
    add(files, "run/", [(x, RUN / x) for x in ("screen.py", "proxy.py", "driver.py", "collector.py", "prop.py", "original.json", "minimum-swap.json", "run.log", "run.stderr", "exit-code")])
    add(files, "input/", [
        ("expanded.json", RECOVERY_DIR / "expanded.json"),
        ("candidate.json", RECOVERY_DIR / "candidate.json"),
        ("recovery-receipt.json", RECOVERY_DIR / "recovery-receipt.json"),
        ("expanded-stable96-v1.json", RR / "expanded-stable96-v1.json"),
    ])
    add(files, "corpus/", [("corpus.jsonl", CORPUS)])
    for index in range(18):
        add(files, "prior-r23g/", [(f"{index:02d}.json", PRIOR / f"{index:02d}.json")])
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        need(len(members) == len(files) and {x.name for x in members} == set(files), "archive inventory")
    print(json.dumps({
        "status": "R23H_FACTORIAL_BIDIRECTIONAL_PREFILL_EVIDENCE_PACKED",
        "plan_sha256": PLAN_SHA,
        "report_sha256": sha(report_raw),
        "archive": str(archive),
        "archive_sha256": sha(archive.read_bytes()),
        "files": len(files),
        "bytes": archive.stat().st_size,
        "samples": 16,
        "eligible_for_full_fit_propagation_only": eligible,
        "selected": report["selected"],
        "metrics": metrics,
        "by_direction": by_direction,
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    import math
    main()
