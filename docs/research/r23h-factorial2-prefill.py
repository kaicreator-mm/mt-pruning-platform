#!/usr/bin/env python3
"""R23H factorial2: isolated Layer 40/42 prefill-only Fit screen.

This is a CPU-mmap oracle stage for the 60 GB HF teacher. It deliberately does
not load the full checkpoint onto the 16 GB V100. It performs no decode,
holdout, Fresh, free-running translation, training, or physical model build.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

P = pathlib.Path
R = P.home() / ".cache/mt-pruning"
REPO = P.home() / "mt-pruning-platform"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
RR = R / "r23-results"
D = R / "r23h-pareto-screen-v1"
USB = R / "r23g-recovery-ssd-38ed4798"
SD = RR / "r23h-pareto-prefill-v1"
OLD_FACTORIAL = RR / "r23h-all8-factorial-prefill-v2"
PREP = RR / "r23h-factorial2-prefill-v1"
RUN = R / "r23h-factorial2-prefill-run-v1"
PRIMARY_BATCH = RR / "r23h-all8-factorial-primary-v3"
PRIOR_R23G = RR / "r23g-l38-propagation-v1"
CORPUS = R / "run-d01b-r19/corpus/corpus.jsonl"

PLAN_SHA = "6b04e24ef14bcd2c01a7843569b896e6d3c664446d3098eedf6f0d9fbfb18dfe"
PLAN_BLOB = "ec2a5095923c20355a6043c1455dedd37d4cca44"
PARETO_PLAN_SHA = "b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
PARETO_REPORT_SHA = "8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
FACTORIAL_PLAN_SHA = "7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac"
FACTORIAL_REPORT_SHA = "3953ed75fb4275019b51df5d6e8e43b91a4e77c37747330be365c3cd67f8da05"
PRIMARY_REPORT_SHA = "eba41d87b5d1f418923584e778a303bdab0dbd1f78f54702ceb86fdea08fb4da"
PRIMARY_REPLAY_SHA = "7f1b8a0b116192837e016711abca20d020265bafcafd41ed7d8bc861c02e93e3"
LAYER_SUMMARY_SHA = "9dc475eecf9ee68e2b30c65a38f10a75c5797d86831b8b11412044c30ee7d7a6"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ORIGINAL_BLOB = "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
EXPANDED_SHA = "21b4a54f626c32727284f32f34f3b43b54ff6e68d56ac4499934ee652fb6d166"
R23G_CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
PINS = {
    "driver": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "collector": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
}
MASKS = {
    "r23h_l40_l42_only": (
        "33cbec0828eef568abe6c77ffb586699ebe76cb54a2e399fcd5fff631252b5e1",
        "98b8b39d1b56db0e830c6fba9bb6667927ab92bc",
        "9957f0df87344eb0a9b507d4bfb97ca6d0a9650203979e5b0165c75471ef8020",
    ),
    "r23h_l40_only": (
        "819b635fc56ae2d539987a5288bd2c16683242e06b75f8759e4c5296fed61e0a",
        "d51357b9f2c9e05e93876aebf03efd8ace44fb30",
        "65bcb31b4267f426bc550c0ec43d3d9be9d6612a3b833b5ae433a0928f263e6b",
    ),
    "r23h_l42_only": (
        "6f89cf26d2de7a3f0ceba026c08ce160490ccfd15167ce194c66704436fb0e47",
        "47fb3ec41345a48d219dabd40a5fd0a63dfb53f4",
        "684ac86eaa17d17c388462e535bd8d3cc02b6c93a1cf230af902095717c69e52",
    ),
}
EXPECTED_CHANGED = {
    "r23h_l40_l42_only": [40, 42],
    "r23h_l40_only": [40],
    "r23h_l42_only": [42],
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def pin(path, expected, git=False):
    need(path.is_file() and not path.is_symlink(), "regular file " + str(path))
    raw = path.read_bytes()
    actual = blob(raw) if git else sha(raw)
    need(actual == expected, "pin mismatch " + str(path) + " expected=" + expected + " actual=" + actual)
    return raw


def publish(path, value):
    raw = canon(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def event(status, **fields):
    print(json.dumps({"status": status, **fields}, sort_keys=True), flush=True)


def load_module(name):
    path = D / (name + ".py")
    pin(path, PINS[name], git=True)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mean(values):
    return math.fsum(values) / len(values)


def metric(teacher_logits, candidate_logits):
    teacher = teacher_logits.double()
    candidate = candidate_logits.double()
    p = teacher.log_softmax(-1)
    q = candidate.log_softmax(-1)
    kl = float((p.exp() * (p - q)).sum(dim=-1).mean())
    nmse = float((teacher - candidate).square().sum() / teacher.square().sum())
    need(math.isfinite(kl) and kl >= -1e-12 and math.isfinite(nmse), "non-finite metric")
    return {
        "kl": max(0.0, kl),
        "nmse": nmse,
        "argmax_mismatch": int(teacher[-1].argmax() != candidate[-1].argmax()),
    }


def check_mask(mask, name):
    need(set(mask) == {str(i) for i in range(1, 48)}, name + " layer inventory")
    for layer, experts in mask.items():
        need(
            type(experts) is list
            and len(experts) == 96
            and experts == sorted(set(experts))
            and all(type(x) is int and 0 <= x < 128 for x in experts),
            name + " expert count/order/range at layer " + layer,
        )


def validate_sources():
    need(PREP.is_dir() and RUN.is_dir(), "factorial2 staging directories")
    need((RUN / "factorial2.py").is_file(), "staged runner copy")
    need(not (PREP / "report.json").exists(), "output report already exists")

    plan_raw = pin(PREP / "plan.json", PLAN_SHA)
    need(blob(plan_raw) == PLAN_BLOB, "factorial2 plan Git blob")
    plan = strict(plan_raw)
    need(plan["status"] == "R23H_FACTORIAL2_PREFILL_SCREEN_ONLY" and plan["holdout"] == "FORBIDDEN", "factorial2 plan scope")
    need(plan["candidate_order"] == list(MASKS), "candidate order")
    need(plan["source_primary_report_sha256"] == PRIMARY_REPORT_SHA, "primary report binding")
    need(plan["source_primary_independent_replay_sha256"] == PRIMARY_REPLAY_SHA, "primary replay binding")
    need(plan["source_factorial_prefill_plan_sha256"] == FACTORIAL_PLAN_SHA, "prior factorial plan binding")
    need(plan["source_factorial_prefill_report_sha256"] == FACTORIAL_REPORT_SHA, "prior factorial report binding")
    need(plan["source_layer_summary_sha256"] == LAYER_SUMMARY_SHA, "layer summary binding")
    need(plan["recovery_sha256"] == RECOVERY_SHA, "recovery binding")
    need(plan["teacher"] == TEACHER, "teacher binding")

    recovery = strict(pin(USB / "recovery-receipt.json", RECOVERY_SHA))
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH", "R23G recovery state")

    primary_report = strict(pin(PRIMARY_BATCH / "report.json", PRIMARY_REPORT_SHA))
    need(
        primary_report["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"
        and primary_report["eligible_for_secondary_fit_only"] == [],
        "prior factorial primary batch must remain rejected",
    )
    primary_replay = strict(pin(RUN / "primary-replay.json", PRIMARY_REPLAY_SHA))
    need(
        primary_replay["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS"
        and primary_replay["archive_sha256"] == plan["source_archive_sha256"]
        and primary_replay["batch_report_sha256"] == PRIMARY_REPORT_SHA
        and primary_replay["eligible_for_secondary_fit_only"] == [],
        "prior factorial primary independent replay",
    )
    pin(RUN / "layer-summary.json", LAYER_SUMMARY_SHA)

    factorial_plan = strict(pin(OLD_FACTORIAL / "plan.json", FACTORIAL_PLAN_SHA))
    factorial_report = strict(pin(OLD_FACTORIAL / "report.json", FACTORIAL_REPORT_SHA))
    need(
        factorial_plan["status"] == "R23H_ALL8_FACTORIAL_PREFILL_SCREEN_ONLY"
        and factorial_plan["holdout"] == "FORBIDDEN"
        and factorial_report["status"] == "R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
        and factorial_report["samples"] == 8
        and set(factorial_report["eligible_for_primary_fit_only"]) == {"all8_revert15_40_42", "all8_revert15"},
        "prior factorial prefill source",
    )

    pareto_plan = strict(pin(SD / "plan.json", PARETO_PLAN_SHA))
    pareto_report = strict(pin(SD / "report.json", PARETO_REPORT_SHA))
    need(pareto_report["selected"] == "all8" and pareto_report["samples"] == 8, "prior Pareto screen")
    original = strict(pin(D / "original.json", ORIGINAL_BLOB, git=True))["candidates"]["96"]["kept"]
    old_masks = {item["name"]: item["kept"] for item in pareto_plan["screen"]}
    r23g = strict(pin(USB / "candidate.json", R23G_CANDIDATE_SHA))["kept"]
    need(sha(canon(original)) == plan["controls"]["original96"], "original96 control")
    need(sha(canon(r23g)) == plan["controls"]["r23g"], "R23G control")
    need(sha(canon(old_masks["all8"])) == plan["controls"]["all8"], "all8 control")
    need(sha(canon(old_masks["all10"])) == plan["controls"]["all10"], "all10 control")

    candidates = {}
    for name, (receipt_sha, git_blob, mask_sha) in MASKS.items():
        raw = pin(PREP / (name + ".json"), receipt_sha)
        need(blob(raw) == git_blob, "mask Git blob " + name)
        receipt = strict(raw)
        mask = receipt["mask"]
        need(
            receipt["status"] == "R23H_FACTORIAL2_LAYER_ONLY_MASK_EXPLORATORY_NOT_ADMITTED"
            and receipt["method"] == name
            and receipt["mask_sha256"] == mask_sha
            and sha(canon(mask)) == mask_sha
            and receipt["changed_layers"] == EXPECTED_CHANGED[name],
            "mask receipt " + name,
        )
        check_mask(mask, name)
        changed = [layer for layer in range(1, 48) if set(mask[str(layer)]) != set(original[str(layer)])]
        need(changed == EXPECTED_CHANGED[name], "actual changed-layer binding " + name)
        binding = plan["candidates"][name]
        need(
            binding["receipt_sha256"] == receipt_sha
            and binding["git_blob_sha"] == git_blob
            and binding["mask_sha256"] == mask_sha
            and binding["changed_layers"] == EXPECTED_CHANGED[name],
            "plan mask binding " + name,
        )
        candidates[name] = mask

    expanded = strict(pin(USB / "expanded.json", EXPANDED_SHA))
    rows = [row for row in expanded["fit"] if row["direction"] == "zh-my"]
    need(len(rows) == 8 and [row["id"] for row in rows] == plan["ids"], "eight frozen zh-my Fit rows")
    need(not ({row["line"] for row in rows} & {row["line"] for row in expanded["holdout"]}), "holdout separation")

    prior = {}
    recovery_receipt = recovery
    for index, receipt_sha in enumerate(recovery_receipt["receipt_sha256"]):
        item = strict(pin(PRIOR_R23G / f"{index:02d}.json", receipt_sha))
        prior[item["id"]] = item
    ids = {row["id"] for row in rows}
    need(ids <= set(prior), "R23G prior receipt coverage")

    old_log = pin(D / "run.log", sha((D / "run.log").read_bytes()))
    events = [strict(line) for line in old_log.splitlines() if line.startswith(b"{")]
    old_captures = [item for item in events if item.get("status") == "R23H_PREFILL_CAPTURED"]
    need(len(old_captures) == 8, "old Pareto screen capture inventory")
    old_screen = {}
    for index, item in enumerate(old_captures):
        receipt = strict(pin(SD / f"{index:02d}.json", item["sha256"]))
        need(receipt["id"] == rows[index]["id"] and receipt["plan_sha256"] == PARETO_PLAN_SHA, "old screen receipt binding")
        old_screen[receipt["id"]] = receipt

    corpus = pin(CORPUS, CORPUS_SHA)
    context = {
        item["id"]: item
        for item in map(strict, corpus.splitlines())
        if item["split"] == "calibration" and item["id"] in ids
    }
    need(set(context) == ids, "calibration corpus rows")
    for row in rows:
        need(sha(canon(context[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"], "prompt identity " + row["id"])

    masks = {
        "original96": original,
        "r23g": r23g,
        "all8": old_masks["all8"],
        "all10": old_masks["all10"],
        **candidates,
    }
    need(list(plan["controls"]) == ["all10", "all8", "original96", "r23g"] or set(plan["controls"]) == {"all10", "all8", "original96", "r23g"}, "control inventory")
    return plan, rows, context, prior, old_screen, masks


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan, rows, context, prior, old_screen, masks = validate_sources()
    driver = load_module("driver")
    collector = load_module("collector")
    collector.verify_code()
    preflight = driver.preflight()
    need(not preflight["blockers"], "runtime preflight blockers: " + json.dumps(preflight["blockers"], sort_keys=True))
    need(collector.torch.__version__ == "2.10.0+cpu", "pinned CPU mmap runtime required")
    identity = driver.checkpoint_identity(driver.SOURCE, cache_dir=None, fresh=True, exact_inventory=True)
    need(identity["teacher"] == TEACHER and identity["num_experts"] == 128, "teacher checkpoint identity")
    event(
        "R23H_FACTORIAL2_PREFLIGHT_PASS",
        plan_sha256=PLAN_SHA,
        teacher=TEACHER,
        torch=collector.torch.__version__,
        rows=len(rows),
        candidates=list(MASKS),
        runtime=preflight,
    )
    if args.preflight_only:
        return

    lock_path = R / "r23-admissions" / "batch.lock"
    need(lock_path.parent.is_dir(), "admission lock directory")
    with lock_path.open("a+") as lock:
        deadline = time.monotonic() + 7200
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                need(time.monotonic() < deadline, "admission lock wait exceeded")
                time.sleep(10)

        need(not driver.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]), "GPU compute process appeared")
        collector.torch.set_num_threads(8)
        collector.torch.set_num_interop_threads(1)
        engine = collector.engine_module.Engine(collector.engine_module.Checkpoint(driver.SOURCE), threads=8)
        event("R23H_FACTORIAL2_TEACHER_READY", teacher=TEACHER, threads=8)

        def prefill(prompt, mask):
            try:
                engine.set_mask(None if mask is None else {int(layer): experts for layer, experts in mask.items()})
                with collector.torch.inference_mode():
                    logits = engine.forward(prompt, engine.new_cache(len(prompt) + 9), "prefill")
                return collector.logits_sha(logits), logits[[len(prompt) // 2 - 1, len(prompt) - 1]].clone()
            finally:
                engine.set_mask(None)

        receipts = []
        for index, row in enumerate(rows):
            need(
                not driver.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]),
                "GPU compute process appeared during CPU oracle run",
            )
            prompt = context[row["id"]]["prompt_ids"]
            old = prior[row["id"]]
            teacher_hash, teacher_logits = prefill(prompt, None)
            need(teacher_hash == old["original96"]["teacher_hashes"][0]["sha256"], "teacher baseline " + row["id"])
            receipt = {
                "status": "R23H_FACTORIAL2_PREFILL_CAPTURED_NOT_ADMISSION",
                "id": row["id"],
                "index": index,
                "plan_sha256": PLAN_SHA,
                "teacher_hash": teacher_hash,
                "metrics": {},
            }
            for name, mask in masks.items():
                digest, logits = prefill(prompt, mask)
                values = metric(teacher_logits, logits)
                if name in ("original96", "all8", "all10"):
                    old_values = old_screen[row["id"]]["metrics"][name]
                    need(
                        digest == old_values["logits_sha256"]
                        and abs(values["kl"] - old_values["kl"]) < 1e-12
                        and abs(values["nmse"] - old_values["nmse"]) < 1e-12
                        and values["argmax_mismatch"] == old_values["argmax_mismatch"],
                        "old screen replay " + name + " " + row["id"],
                    )
                if name == "r23g":
                    old_values = old["candidate96"]["logits"][0]
                    need(
                        digest == old["candidate96"]["masked_hashes"][0]["sha256"]
                        and abs(values["kl"] - old_values["teacher_to_masked_kl"]) < 1e-12
                        and abs(values["nmse"] - old_values["nmse"]) < 1e-12
                        and values["argmax_mismatch"] == int(old_values["teacher_argmax"] != old_values["masked_argmax"]),
                        "R23G replay " + row["id"],
                    )
                receipt["metrics"][name] = {
                    **values,
                    "logits_sha256": digest,
                    "mask_sha256": sha(canon(mask)),
                }
            restored_hash, _ = prefill(prompt, None)
            need(restored_hash == teacher_hash, "teacher restoration " + row["id"])
            receipt_sha = publish(PREP / f"{index:02d}.json", receipt)
            receipts.append(receipt)
            event("R23H_FACTORIAL2_PREFILL_CAPTURED", index=index, id=row["id"], receipt_sha256=receipt_sha)

        names = list(masks)
        metrics = {
            name: {
                key: (sum if key == "argmax_mismatch" else mean)([item["metrics"][name][key] for item in receipts])
                for key in ("kl", "nmse", "argmax_mismatch")
            }
            for name in names
        }
        eligible = [
            name
            for name in MASKS
            if all(metrics[name][key] <= metrics["original96"][key] + 1e-12 for key in metrics[name])
            and metrics[name]["kl"] < metrics["r23g"]["kl"]
        ]
        eligible.sort(key=lambda name: (metrics[name]["kl"], metrics[name]["nmse"], name))
        report = {
            "status": "R23H_FACTORIAL2_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",
            "plan_sha256": PLAN_SHA,
            "teacher_sha256": TEACHER,
            "samples": len(receipts),
            "metrics": metrics,
            "eligible_for_primary_fit_only": eligible,
            "selected": eligible[0] if eligible else None,
            "limits": "Three isolated 96-expert masks retaining only Layer 40 and/or Layer 42 from all8; eight frozen zh-my calibration Fit Prefill rows only. No decode, my-zh, Fresh, holdout, free-running translation, product admission, or physical model build.",
        }
        report_sha = publish(PREP / "report.json", report)
        event("R23H_FACTORIAL2_PREFILL_COMPLETE_NOT_FRESH_ADMISSION", report_sha256=report_sha, report=report)


if __name__ == "__main__":
    main()
