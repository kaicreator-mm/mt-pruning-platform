#!/usr/bin/env python3
"""Independent, stdlib-only replay of R23H factorial2 prefill evidence."""
import argparse
import hashlib
import io
import json
import math
import pathlib
import tarfile

PLAN_SHA = "6b04e24ef14bcd2c01a7843569b896e6d3c664446d3098eedf6f0d9fbfb18dfe"
PLAN_BLOB = "ec2a5095923c20355a6043c1455dedd37d4cca44"
FACTORIAL2_BLOB = "2789f50086ee5f343cc12bed2c99e153e2ec1e98"
PRIMARY_REPORT_SHA = "eba41d87b5d1f418923584e778a303bdab0dbd1f78f54702ceb86fdea08fb4da"
PRIMARY_REPLAY_SHA = "7f1b8a0b116192837e016711abca20d020265bafcafd41ed7d8bc861c02e93e3"
LAYER_SUMMARY_SHA = "9dc475eecf9ee68e2b30c65a38f10a75c5797d86831b8b11412044c30ee7d7a6"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
EXPANDED_SHA = "21b4a54f626c32727284f32f34f3b43b54ff6e68d56ac4499934ee652fb6d166"
CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
ORIGINAL_BLOB = "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
PARETO_PLAN_SHA = "b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
PARETO_REPORT_SHA = "8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
FACTORIAL_PLAN_SHA = "7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac"
FACTORIAL_REPORT_SHA = "3953ed75fb4275019b51df5d6e8e43b91a4e77c37730be365c3cd67f8da05"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PINS = {
    "driver": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "collector": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
}
MASKS = {
    "r23h_l40_l42_only": {
        "receipt_sha": "33cbec0828eef568abe6c77ffb586699ebe76cb54a2e399fcd5fff631252b5e1",
        "git_blob": "98b8b39d1b56db0e830c6fba9bb6667927ab92bc",
        "mask_sha": "9957f0df87344eb0a9b507d4bfb97ca6d0a9650203979e5b0165c75471ef8020",
        "changed": [40, 42],
    },
    "r23h_l40_only": {
        "receipt_sha": "819b635fc56ae2d539987a5288bd2c16683242e06b75f8759e4c5296fed61e0a",
        "git_blob": "d51357b9f2c9e05e93876aebf03efd8ace44fb30",
        "mask_sha": "65bcb31b4267f426bc550c0ec43d3d9be9d6612a3b833b5ae433a0928f263e6b",
        "changed": [40],
    },
    "r23h_l42_only": {
        "receipt_sha": "6f89cf26d2de7a3f0ceba026c08ce160490ccfd15167ce194c66704436fb0e47",
        "git_blob": "47fb3ec41345a48d219dabd40a5fd0a63dfb53f4",
        "mask_sha": "684ac86eaa17d17c388462e535bd8d3cc02b6c93a1cf230af902095717c69e52",
        "changed": [42],
    },
}
MASK_NAMES = ("original96", "r23g", "all8", "all10", *MASKS)


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
    need(sha(raw) == expected, "evidence archive SHA256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            parts = pathlib.PurePosixPath(name).parts
            need(not name.startswith("/") and ".." not in parts, "unsafe archive path: " + name)
            need(member.size <= 32 * 1024 * 1024, "oversize archive member: " + name)
            need(name not in files, "duplicate archive path: " + name)
            stream = archive.extractfile(member)
            need(stream is not None, "unreadable archive member: " + name)
            files[name] = stream.read()
    return files


def get(files, name):
    need(name in files, "missing archive member: " + name)
    return files[name]


def jget(files, name):
    return parse(get(files, name), name)


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_mask(mask, name):
    need(set(mask) == {str(i) for i in range(1, 48)}, name + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96, name + " expert count at " + layer)
        need(experts == sorted(set(experts)), name + " expert order/uniqueness at " + layer)
        need(all(type(x) is int and 0 <= x < 128 for x in experts), name + " expert range at " + layer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    F = read_archive(args.archive, args.archive_sha256)

    plan_raw = get(F, "batch/plan.json")
    need(sha(plan_raw) == PLAN_SHA and blob(plan_raw) == PLAN_BLOB, "factorial2 plan pin")
    plan = parse(plan_raw, "factorial2 plan")
    need(plan["status"] == "R23H_FACTORIAL2_PREFILL_SCREEN_ONLY" and plan["holdout"] == "FORBIDDEN", "plan scope")
    need(plan["candidate_order"] == list(MASKS) and len(plan["ids"]) == 8 and len(set(plan["ids"])) == 8, "plan candidate/row inventory")
    need(plan["teacher"] == TEACHER, "teacher pin")
    need(plan["source_primary_report_sha256"] == PRIMARY_REPORT_SHA, "primary report binding")
    need(plan["source_primary_independent_replay_sha256"] == PRIMARY_REPLAY_SHA, "primary replay binding")
    need(plan["source_layer_summary_sha256"] == LAYER_SUMMARY_SHA, "layer summary binding")
    need(plan["source_factorial_prefill_plan_sha256"] == FACTORIAL_PLAN_SHA, "prior factorial plan binding")
    need(plan["source_factorial_prefill_report_sha256"] == FACTORIAL_REPORT_SHA, "prior factorial report binding")
    need(plan["recovery_sha256"] == RECOVERY_SHA, "recovery binding")

    runner = get(F, "run/factorial2.py")
    need(blob(runner) == FACTORIAL2_BLOB, "factorial2 runner source blob")
    need(b'CUDA_VISIBLE_DEVICES"] = ""' in runner and b'OMP_NUM_THREADS"] = "8"' in runner and b'MKL_NUM_THREADS"] = "8"' in runner, "CPU-only thread/device contract")
    primary_replay_raw = get(F, "run/primary-replay.json")
    need(sha(primary_replay_raw) == PRIMARY_REPLAY_SHA, "primary replay file pin")
    primary_replay = parse(primary_replay_raw, "primary replay")
    need(
        primary_replay["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS"
        and primary_replay["archive_sha256"] == plan["source_archive_sha256"]
        and primary_replay["batch_report_sha256"] == PRIMARY_REPORT_SHA
        and primary_replay["eligible_for_secondary_fit_only"] == [],
        "primary independent replay state",
    )
    need(sha(get(F, "run/layer-summary.json")) == LAYER_SUMMARY_SHA, "layer summary pin")

    primary_report_raw = get(F, "primary/report.json")
    need(sha(primary_report_raw) == PRIMARY_REPORT_SHA, "primary report pin")
    primary_report = parse(primary_report_raw, "primary report")
    need(
        primary_report["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"
        and primary_report["eligible_for_secondary_fit_only"] == [],
        "primary batch remains rejected",
    )

    prior_factorial_plan = get(F, "prior-factorial/plan.json")
    prior_factorial_report = get(F, "prior-factorial/report.json")
    need(sha(prior_factorial_plan) == FACTORIAL_PLAN_SHA and sha(prior_factorial_report) == FACTORIAL_REPORT_SHA, "prior factorial pins")
    prior_factorial = parse(prior_factorial_report, "prior factorial report")
    need(
        prior_factorial["status"] == "R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
        and prior_factorial["samples"] == 8
        and set(prior_factorial["eligible_for_primary_fit_only"]) == {"all8_revert15_40_42", "all8_revert15"},
        "prior factorial prefill decision",
    )

    pareto_plan_raw = get(F, "pareto/plan.json")
    pareto_report_raw = get(F, "pareto/report.json")
    need(sha(pareto_plan_raw) == PARETO_PLAN_SHA and sha(pareto_report_raw) == PARETO_REPORT_SHA, "Pareto screen pins")
    pareto_plan = parse(pareto_plan_raw, "Pareto plan")
    pareto_report = parse(pareto_report_raw, "Pareto report")
    need(pareto_report["selected"] == "all8" and pareto_report["samples"] == 8, "Pareto screen decision")

    original_raw = get(F, "pareto-run/original.json")
    need(blob(original_raw) == ORIGINAL_BLOB, "original96 source blob")
    original_mask = parse(original_raw, "original mask")["candidates"]["96"]["kept"]
    r23g_raw = get(F, "recovery/candidate.json")
    need(sha(r23g_raw) == CANDIDATE_SHA, "R23G candidate pin")
    r23g_mask = parse(r23g_raw, "R23G candidate")["kept"]
    old_masks = {item["name"]: item["kept"] for item in pareto_plan["screen"]}
    masks = {"original96": original_mask, "r23g": r23g_mask, "all8": old_masks["all8"], "all10": old_masks["all10"]}
    for name, mask in masks.items():
        check_mask(mask, name)
    need(sha(canon(original_mask)) == plan["controls"]["original96"], "original96 control hash")
    need(sha(canon(r23g_mask)) == plan["controls"]["r23g"], "R23G control hash")
    need(sha(canon(old_masks["all8"])) == plan["controls"]["all8"], "all8 control hash")
    need(sha(canon(old_masks["all10"])) == plan["controls"]["all10"], "all10 control hash")

    candidate_masks = {}
    for name, pin in MASKS.items():
        raw = get(F, "batch/masks/" + name + ".json")
        need(sha(raw) == pin["receipt_sha"] and blob(raw) == pin["git_blob"], "candidate receipt pin " + name)
        receipt = parse(raw, "candidate mask " + name)
        mask = receipt["mask"]
        check_mask(mask, name)
        need(
            receipt["status"] == "R23H_FACTORIAL2_LAYER_ONLY_MASK_EXPLORATORY_NOT_ADMITTED"
            and receipt["method"] == name
            and receipt["mask_sha256"] == pin["mask_sha"]
            and sha(canon(mask)) == pin["mask_sha"]
            and receipt["changed_layers"] == pin["changed"],
            "candidate mask receipt " + name,
        )
        changed = [layer for layer in range(1, 48) if set(mask[str(layer)]) != set(original_mask[str(layer)])]
        need(changed == pin["changed"], "actual changed-layer inventory " + name)
        binding = plan["candidates"][name]
        need(
            binding["receipt_sha256"] == pin["receipt_sha"]
            and binding["git_blob_sha"] == pin["git_blob"]
            and binding["mask_sha256"] == pin["mask_sha"]
            and binding["changed_layers"] == pin["changed"],
            "plan candidate binding " + name,
        )
        candidate_masks[name] = mask
    masks.update(candidate_masks)

    recovery_raw = get(F, "recovery/recovery-receipt.json")
    need(sha(recovery_raw) == RECOVERY_SHA, "R23G recovery receipt pin")
    recovery = parse(recovery_raw, "recovery receipt")
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(recovery["receipt_sha256"]) == 18, "R23G recovery state")
    expanded_raw = get(F, "recovery/expanded.json")
    need(sha(expanded_raw) == EXPANDED_SHA, "expanded Fit plan pin")
    expanded = parse(expanded_raw, "expanded Fit plan")
    rows = [row for row in expanded["fit"] if row["direction"] == "zh-my"]
    need(len(rows) == 8 and [row["id"] for row in rows] == plan["ids"], "frozen zh-my Fit rows")
    holdout = {row["line"] for row in expanded["holdout"]}
    need(not ({row["line"] for row in rows} & holdout), "Holdout separation")

    corpus_raw = get(F, "corpus/corpus.jsonl")
    need(sha(corpus_raw) == CORPUS_SHA, "corpus pin")
    ids = set(plan["ids"])
    context = {
        item["id"]: item
        for item in (parse(line, "corpus row") for line in corpus_raw.splitlines())
        if item["split"] == "calibration" and item["id"] in ids
    }
    need(set(context) == ids, "calibration corpus coverage")
    for row in rows:
        need(sha(canon(context[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"], "prompt identity " + row["id"])

    prior = {}
    for index, expected in enumerate(recovery["receipt_sha256"]):
        raw = get(F, f"prior-r23g/{index:02d}.json")
        need(sha(raw) == expected, "prior R23G receipt pin " + str(index))
        item = parse(raw, "prior R23G receipt")
        prior[item["id"]] = item
    need(ids <= set(prior), "R23G prior receipt coverage")

    pareto_run = get(F, "pareto-run/run.log")
    old_events = [parse(line, "Pareto run event") for line in pareto_run.splitlines() if line.startswith(b"{")]
    captures = [item for item in old_events if item.get("status") == "R23H_PREFILL_CAPTURED"]
    need(len(captures) == 8, "prior Pareto capture count")
    old_screen = {}
    for index, item in enumerate(captures):
        raw = get(F, f"pareto/{index:02d}.json")
        need(sha(raw) == item["sha256"], "prior Pareto receipt SHA " + str(index))
        receipt = parse(raw, "prior Pareto receipt")
        need(receipt["id"] == rows[index]["id"] and receipt["plan_sha256"] == PARETO_PLAN_SHA, "prior Pareto receipt binding")
        old_screen[receipt["id"]] = receipt

    preflight_exit = get(F, "run/preflight-v3.exit-code").decode().strip()
    need(preflight_exit == "0" and get(F, "run/preflight-v3.stderr") == b"", "preflight exit/stderr")
    preflight_log = get(F, "run/preflight-v3.log")
    need(b"R23H_FACTORIAL2_PREFLIGHT_PASS" in preflight_log and b'"torch": "2.10.0+cpu"' in preflight_log, "preflight evidence")
    run_exit = get(F, "run/execute-v1.exit-code").decode().strip()
    need(run_exit == "0" and get(F, "run/execute-v1.stderr") == b"", "execution exit/stderr")
    run_log = get(F, "run/execute-v1.log")
    run_events = [parse(line, "factorial2 run event") for line in run_log.splitlines() if line.startswith(b"{")]
    captures = [item for item in run_events if item.get("status") == "R23H_FACTORIAL2_PREFILL_CAPTURED"]
    need(len(captures) == 8, "factorial2 capture count")
    need(any(item.get("status") == "R23H_FACTORIAL2_TEACHER_READY" for item in run_events), "teacher ready event")
    terminal = [item for item in run_events if item.get("status") == "R23H_FACTORIAL2_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(terminal) == 1, "factorial2 terminal event")

    receipt_values = []
    for index, row in enumerate(rows):
        raw = get(F, f"batch/{index:02d}.json")
        receipt_sha = sha(raw)
        need(captures[index]["id"] == row["id"] and captures[index]["index"] == index and captures[index]["receipt_sha256"] == receipt_sha, "capture event receipt binding " + str(index))
        receipt = parse(raw, "factorial2 receipt " + str(index))
        need(
            receipt["status"] == "R23H_FACTORIAL2_PREFILL_CAPTURED_NOT_ADMISSION"
            and receipt["id"] == row["id"]
            and receipt["index"] == index
            and receipt["plan_sha256"] == PLAN_SHA,
            "factorial2 receipt identity " + str(index),
        )
        old = prior[row["id"]]
        need(receipt["teacher_hash"] == old["original96"]["teacher_hashes"][0]["sha256"], "teacher logits baseline " + row["id"])
        need(set(receipt["metrics"]) == set(MASK_NAMES), "metric mask inventory " + row["id"])
        for name, mask in masks.items():
            metric = receipt["metrics"][name]
            need(
                finite_nonnegative(metric["kl"])
                and finite_nonnegative(metric["nmse"])
                and type(metric["argmax_mismatch"]) is int
                and metric["argmax_mismatch"] in (0, 1)
                and len(metric["logits_sha256"]) == 64
                and metric["mask_sha256"] == sha(canon(mask)),
                "metric validity/mask binding " + name + "/" + row["id"],
            )
        for name in ("original96", "all8", "all10"):
            need(receipt["metrics"][name] == old_screen[row["id"]]["metrics"][name], "exact old-screen replay " + name + "/" + row["id"])
        prior_metric = old["candidate96"]["logits"][0]
        prior_hash = old["candidate96"]["masked_hashes"][0]["sha256"]
        q = receipt["metrics"]["r23g"]
        need(
            q["logits_sha256"] == prior_hash
            and abs(q["kl"] - prior_metric["teacher_to_masked_kl"]) < 1e-12
            and abs(q["nmse"] - prior_metric["nmse"]) < 1e-12
            and q["argmax_mismatch"] == int(prior_metric["teacher_argmax"] != prior_metric["masked_argmax"]),
            "exact R23G replay " + row["id"],
        )
        receipt_values.append(receipt)

    aggregates = {}
    for name in MASK_NAMES:
        aggregates[name] = {
            "kl": math.fsum(item["metrics"][name]["kl"] for item in receipt_values) / 8,
            "nmse": math.fsum(item["metrics"][name]["nmse"] for item in receipt_values) / 8,
            "argmax_mismatch": sum(item["metrics"][name]["argmax_mismatch"] for item in receipt_values),
        }
    eligible = [
        name for name in MASKS
        if all(aggregates[name][key] <= aggregates["original96"][key] + 1e-12 for key in aggregates[name])
        and aggregates[name]["kl"] < aggregates["r23g"]["kl"]
    ]
    eligible.sort(key=lambda name: (aggregates[name]["kl"], aggregates[name]["nmse"], name))

    report_raw = get(F, "batch/report.json")
    report_sha = sha(report_raw)
    report = parse(report_raw, "factorial2 report")
    need(
        report["status"] == "R23H_FACTORIAL2_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
        and report["plan_sha256"] == PLAN_SHA
        and report["teacher_sha256"] == TEACHER
        and report["samples"] == 8,
        "factorial2 report scope",
    )
    need(report["metrics"] == aggregates, "independent aggregate replay")
    need(report["eligible_for_primary_fit_only"] == eligible and report["selected"] == (eligible[0] if eligible else None), "independent gate replay")
    need(terminal[0]["report_sha256"] == report_sha, "terminal report hash binding")
    terminal_report = terminal[0]["report"]
    need(terminal_report == report, "terminal report content binding")

    print(json.dumps({
        "status": "R23H_FACTORIAL2_PREFILL_INDEPENDENT_RECEIPT_REPLAY_PASS",
        "archive_sha256": args.archive_sha256,
        "plan_sha256": PLAN_SHA,
        "report_sha256": report_sha,
        "samples": 8,
        "metrics": aggregates,
        "eligible_for_primary_fit_only": eligible,
        "selected": report["selected"],
        "decision": "prefill-only screen; any passing candidate may enter the bounded primary Fit propagation only",
        "checks": [
            "exact plan and source pins",
            "three candidate mask receipt, Git blob, and 47x96 shape",
            "actual changed-layer inventory 40/42",
            "8 frozen zh-my Fit rows and Holdout exclusion",
            "teacher logits baseline and per-row receipt hashes",
            "original96/all8/all10 historical replay",
            "R23G historical replay",
            "independent aggregate metric and candidate-gate recomputation",
            "CPU-only preflight, clean exit, and terminal report binding",
        ],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
