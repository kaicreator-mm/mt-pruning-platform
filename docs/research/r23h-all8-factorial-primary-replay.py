#!/usr/bin/env python3
"""Independent exact-receipt replay for the R23H factorial primary Fit batch."""
import hashlib
import io
import json
import math
import pathlib
import sys
import tarfile

P = pathlib.Path
ARCHIVE_SHA = "80f8dc3d8ba4e598652a286e68e8eecfc32e54046855a9a194c8b7e8869b95cd"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
FACTORIAL_PLAN = "7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac"
FACTORIAL_REPORT = "3953ed75fb4275019b51df5d6e8e43b91a4e77c37747330be365c3cd67f8da05"
PRIMARY_REPLAY_SHA = "f6927cccf2279a56673cad99b31559f199252fb734ff2e58caa6c667b8df1ebc"
PRIMARY_REPLAY_BLOB = "1ae914e4e1e0c9729c43b237100b6fc20a5fbefb"
OLD_PLAN = "b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT = "8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
REC = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
RECORDS_SHA = "37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8"
ALL10_REPORT = "63a53755c176d7d6514f8a18d4908d9cadec622a9fb3492b2cc91cdbbea814fa"
ALL10_ADMISSION = "bfd5267613c493558d3e0deece69aa4472cce53ea6aea5ad859b5754117ded2f"
BATCH_ADMISSION = "7434ba31d589276793eb3790745cf6b0afeb4d25561e66d04b08a3d15db37612"
BATCH_REPORT = "eba41d87b5d1f418923584e778a303bdab0dbd1f78f54702ceb86fdea08fb4da"

NAMES = ("all8_revert15_40_42", "all8_revert15")
SCREEN_MASKS = ("all8_revert10", "all8_revert15", "all8_revert10_40_42", "all8_revert15_40_42")
MASKS = {
    "all8_revert10": "b16b87ddd1058e0dd00286d0f1f7c7aa57e4925277346858f42bb36857479e7f",
    "all8_revert15": "4c81baf0201fa0f085b8908625a2b46d9f4fb39bebc6cbef829138817a416c7f",
    "all8_revert10_40_42": "2ffd628021e714e5ecf9145f9651a101788790e57f56bdf156b927dae65472ac",
    "all8_revert15_40_42": "31c5eeb44b3dc8acb6eeae97a2078faf1b96f9e129a24fc27726232b2b5c8df2",
}
BLOBS = {
    "all8_revert10": "faa4d6146a3df4192e913e2702a926afa3fb095d",
    "all8_revert15": "044a6f395d779489462df00a664cf7876732ab06",
    "all8_revert10_40_42": "adc46aceb97c4f35c8489b1b0cb02fbdf0295eee",
    "all8_revert15_40_42": "e453de1af1fef152b44f2446ecb5de6f9e18b4ae",
}
RECEIPTS = {
    "all8_revert10": "7c7a154b17f56d896167c992e06f0f743e286688bf44714949058a2bacd9511d",
    "all8_revert15": "cdaebcfe274af03118167638ad482f46dd8bae5474701b2bba34de2b8d747498",
    "all8_revert10_40_42": "776146272acf7673bac2f3b1393c85cc59be0316acac09bc466f561a69d9cbbd",
    "all8_revert15_40_42": "138cd339a7f7f9408bebf325fd8258d889fee156431a5f5b6be451d4cdbbf1fa",
}
CHANGED = {
    "all8_revert10": [15, 40, 42],
    "all8_revert15": [10, 40, 42],
    "all8_revert10_40_42": [15],
    "all8_revert15_40_42": [10],
}
INDICES = (0, 1, 2, 3, 12, 13, 14, 15, 16, 17)
DIRS = ("zh-my", "my-zh")
SCREEN_NAMES = ("original96", "r23g", "all8", "all10", *SCREEN_MASKS)


def need(ok, message):
    if not ok:
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
            need(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def avg(values):
    return math.fsum(values) / len(values)


def read_tar(raw):
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        need(len(names) == len(members), "duplicate archive paths")
        need(all(member.isfile() and 0 <= member.size <= 32 * 1024 * 1024 for member in members),
             "archive member type/size")
        return {member.name: archive.extractfile(member).read() for member in members}


def check_mask(mask, name):
    need(set(mask) == {str(i) for i in range(1, 48)}, name + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96 and experts == sorted(set(experts)),
             name + " expert count/order at layer " + layer)
        need(all(type(expert) is int and 0 <= expert < 128 for expert in experts),
             name + " expert range at layer " + layer)


def check_capture(value, row, index, admission_sha, mask_sha):
    need(value["status"] == "PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY"
         and value["teacher_restored"] is True
         and value["index"] == index
         and value["id"] == row["id"]
         and value["direction"] == row["direction"]
         and value["line"] == row["line"]
         and value["consumed_prefix"] == row["teacher_prefix_tokens"]
         and value["teacher"] == TEACHER
         and value["mask_sha256"] == mask_sha
         and value["admission_sha256"] == admission_sha, "capture provenance")
    expected = {(layer, "prefill" if step == 0 else "decode", step)
                for layer in range(1, 48) for step in range(9)}
    for name in ("original96", "candidate96"):
        item = value[name]
        layers = item["layer"]
        need(len(layers) == len(expected)
             and {(x["layer"], x["phase"], x["step"]) for x in layers} == expected,
             "layer coverage")
        for layer in layers:
            need(all(finite(layer[key]) for key in
                     ("hidden_input_nmse", "moe_output_nmse", "mean_overlap"))
                 and layer["mean_overlap"] <= 8
                 and layer["tokens"] == (2 if layer["step"] == 0 else 1)
                 and type(layer["exact_sets"]) is int
                 and 0 <= layer["exact_sets"] <= layer["tokens"], "layer values")
        for key in ("teacher_hashes", "masked_hashes"):
            hashes = item[key]
            need(len(hashes) == 9
                 and all(x["step"] == step and len(x["sha256"]) == 64
                         and type(x["next"]) is int and x["next"] >= 0
                         for step, x in enumerate(hashes)), "token hashes")
        need(len(item["logits"]) == 9
             and all(x["step"] == step and finite(x["nmse"])
                     and finite(x["teacher_to_masked_kl"])
                     and x["teacher_argmax"] == item["teacher_hashes"][step]["next"]
                     and x["masked_argmax"] == item["masked_hashes"][step]["next"]
                     for step, x in enumerate(item["logits"])), "logits")
    need(value["original96"]["teacher_hashes"] == value["candidate96"]["teacher_hashes"],
         "teacher identity")


def aggregate(rows, captures, name):
    groups = {}
    for direction in DIRS:
        for phase in ("prefill", "decode"):
            group = direction + "/" + phase
            groups[group] = {}
            for candidate in ("original96", "candidate96"):
                values = [
                    item for row in rows if row["direction"] == direction
                    for item in captures[row["id"]][candidate]["logits"]
                    if (item["step"] == 0) == (phase == "prefill")
                ]
                need(len(values) == (8 if direction == "zh-my" else 2)
                     * (1 if phase == "prefill" else 8), "position coverage " + group)
                groups[group][candidate] = {
                    "kl": avg([x["teacher_to_masked_kl"] for x in values]),
                    "nmse": avg([x["nmse"] for x in values]),
                    "argmax_mismatch": sum(x["teacher_argmax"] != x["masked_argmax"] for x in values),
                    "positions": len(values),
                }
    return groups


def main():
    need(len(sys.argv) == 2, "usage: verify_factorial_primary.py <evidence-directory>")
    root = P(sys.argv[1])
    archive_raw = (root / "factorial-primary.tgz").read_bytes()
    need(sha(archive_raw) == ARCHIVE_SHA, "archive SHA256")
    files = read_tar(archive_raw)
    expected = {
        "batch/admission.json", "batch/report.json", "batch/run.log", "batch/run.stderr", "batch/exit-code",
        *("batch/" + name + "-" + f"{index:02d}" + ".json" for name in NAMES for index in INDICES),
        "prior-all10/admission.json", "prior-all10/report.json", "prior-all10/run.log",
        "prior-all10/run.stderr", "prior-all10/exit-code",
        *("prior-all10/" + f"{index:02d}.json" for index in range(18)),
        "screen/plan.json", "screen/report.json", "screen/run.log", "screen/run.stderr",
        "screen/exit-code", "screen/primary-replay.json",
        *("screen/" + f"{index:02d}.json" for index in range(8)),
        *("screen/masks/" + name + ".json" for name in SCREEN_MASKS),
        "old-screen/plan.json", "old-screen/report.json", "old-screen/run.log",
        "old-screen/run.stderr", "old-screen/exit-code", "old-screen/original.json",
        *("old-screen/" + f"{index:02d}.json" for index in range(8)),
        "recovery/recovery-receipt.json", "recovery/expanded.json", "recovery/prop-plan.json",
        "recovery/candidate.json",
        *("prior-r23g/" + f"{index:02d}.json" for index in range(18)),
        "corpus/corpus.jsonl", "corpus/records.jsonl",
    }
    need(set(files) == expected, "exact archive inventory")
    for prefix in ("batch", "prior-all10", "screen", "old-screen"):
        need(files[prefix + "/exit-code"].strip() == b"0"
             and files[prefix + "/run.stderr"] == b"", "terminal status " + prefix)

    def exact(path, expected_hash, git_blob=False):
        raw = files[path]
        need((blob(raw) if git_blob else sha(raw)) == expected_hash, "pin " + path)
        return strict(raw)

    recovery = exact("recovery/recovery-receipt.json", REC)
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH"
         and len(recovery["receipt_sha256"]) == 18, "recovery receipt")
    expanded = exact("recovery/expanded.json", "a1047d1a7c5624ed0e3a868bd5d76eef4c754915", True)
    prop_plan = exact("recovery/prop-plan.json", "cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde", True)
    rows = prop_plan["rows"] + [
        row for row in expanded["fit"]
        if row["direction"] == "zh-my" and row["id"] not in {x["id"] for x in prop_plan["rows"]}
    ]
    need(len(rows) == 18 and len({row["line"] for row in rows}) == 16
         and all(row in expanded["fit"] for row in rows)
         and not ({row["line"] for row in rows} & {row["line"] for row in expanded["holdout"]}),
         "Fit-only row inventory")
    selected = [(index, row) for index, row in enumerate(rows) if row["direction"] in DIRS]
    need([index for index, _ in selected] == list(INDICES) and len(selected) == 10,
         "primary Fit row indices")
    corpus_raw = files["corpus/corpus.jsonl"]
    records_raw = files["corpus/records.jsonl"]
    need(sha(corpus_raw) == CORPUS_SHA and sha(records_raw) == RECORDS_SHA,
         "calibration corpus/baseline pins")
    ids = {row["id"] for row in rows}
    corpus = {item["id"]: item for item in
              (strict(line) for line in corpus_raw.splitlines() if line)
              if item.get("split") == "calibration" and item["id"] in ids}
    baseline = {item["id"]: item for item in
                (strict(line) for line in records_raw.splitlines()[1:] if line)
                if item.get("split") == "calibration" and item["id"] in ids}
    need(set(corpus) == set(baseline) == ids, "calibration input inventory")
    for row in rows:
        item = corpus[row["id"]]
        previous = baseline[row["id"]]
        need(item["line"] == previous["line"] == row["line"]
             and item["direction"] == previous["direction"] == row["direction"]
             and len(item["prompt_ids"]) == previous["prompt_tokens"] == row["prompt_tokens"]
             and sha(canon(item["prompt_ids"])) == row["prompt_ids_sha256"]
             and previous["finish"] == "eos"
             and previous["tokens"][:8] == row["teacher_prefix_tokens"],
             "calibration input identity " + row["id"])

    # Verify the prior all10 failure and its complete 18-row evidence.
    prior_admission_raw = files["prior-all10/admission.json"]
    prior_admission = exact("prior-all10/admission.json", ALL10_ADMISSION)
    prior_report = exact("prior-all10/report.json", ALL10_REPORT)
    need(prior_admission["selected"] == "all10"
         and prior_admission["mask_sha256"] == "a9274a240cb1e990c77b110b9a6231e0ae1c905849a2b7e9295f795bd3a81fe2"
         and prior_report["status"] == "R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION"
         and prior_report["samples"] == 18 and prior_report["selected"] == "all10"
         and prior_report["primary_diagnostic_pass"] is False, "all10 predecessor")
    prior_events = [strict(line) for line in files["prior-all10/run.log"].splitlines() if line.startswith(b"{")]
    prior_captures = [event for event in prior_events if event["status"] == "R23H_PROPAGATION_CAPTURED"]
    prior_done = [event for event in prior_events if event["status"] == "R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(prior_captures) == 18 and len(prior_done) == 1
         and prior_done[0]["report_sha256"] == ALL10_REPORT, "all10 predecessor events")
    prior_all10 = {}
    for index, (event, row) in enumerate(zip(prior_captures, rows)):
        value = exact("prior-all10/" + f"{index:02d}.json", event["receipt_sha256"])
        need(event["index"] == index and event["id"] == row["id"] and value["id"] == row["id"]
             and value["mask_sha256"] == prior_admission["mask_sha256"]
             and value["admission_sha256"] == sha(prior_admission_raw), "all10 receipt binding")
        prior_all10[row["id"]] = value
    prior_groups = {}
    for direction in ("zh-my", "my-zh", "zh-en", "en-zh", "en-my", "my-en"):
        for phase in ("prefill", "decode"):
            group = direction + "/" + phase
            prior_groups[group] = {}
            for candidate in ("original96", "candidate96"):
                values = [
                    item for value in prior_all10.values() if value["direction"] == direction
                    for item in value[candidate]["logits"]
                    if (item["step"] == 0) == (phase == "prefill")
                ]
                need(len(values) == (8 if direction == "zh-my" else 2)
                     * (1 if phase == "prefill" else 8), "all10 position coverage")
                prior_groups[group][candidate] = {
                    "kl": avg([x["teacher_to_masked_kl"] for x in values]),
                    "nmse": avg([x["nmse"] for x in values]),
                    "argmax_mismatch": sum(x["teacher_argmax"] != x["masked_argmax"] for x in values),
                    "positions": len(values),
                }
    need(prior_groups == prior_report["groups"], "all10 aggregate replay")

    # Verify the old Pareto prefill controls and exact original96 mask.
    old_plan = exact("old-screen/plan.json", OLD_PLAN)
    old_report = exact("old-screen/report.json", OLD_REPORT)
    need(old_plan["status"] == "R23H_PARETO_FIT_PREFILL_SCREEN_ONLY"
         and old_plan["holdout"] == "FORBIDDEN"
         and old_report["selected"] == "all8" and old_report["samples"] == 8,
         "old Pareto screen")
    old_events = [strict(line) for line in files["old-screen/run.log"].splitlines() if line.startswith(b"{")]
    old_capture_events = [event for event in old_events if event["status"] == "R23H_PREFILL_CAPTURED"]
    need(len(old_capture_events) == 8, "old screen event inventory")
    old_screen = {}
    for index, event in enumerate(old_capture_events):
        value = exact("old-screen/" + f"{index:02d}.json", event["sha256"])
        need(event["index"] == index and value["index"] == index
             and event["id"] == value["id"] == old_plan["ids"][index]
             and value["plan_sha256"] == OLD_PLAN, "old screen receipt")
        old_screen[value["id"]] = value
    base_mask = exact("old-screen/original.json", "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed", True)["candidates"]["96"]["kept"]
    r23g_mask = exact("recovery/candidate.json", "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1")["kept"]
    old_masks = {item["name"]: item["kept"] for item in old_plan["screen"]}
    controls = {
        "original96": sha(canon(base_mask)),
        "r23g": sha(canon(r23g_mask)),
        "all8": sha(canon(old_masks["all8"])),
        "all10": sha(canon(old_masks["all10"])),
    }

    # Verify factorial prefill screen plan, masks, receipts, and exact control replay.
    plan = exact("screen/plan.json", FACTORIAL_PLAN)
    need(plan["status"] == "R23H_ALL8_FACTORIAL_PREFILL_SCREEN_ONLY"
         and plan["recovery_sha256"] == REC and plan["teacher"] == TEACHER
         and plan["holdout"] == "FORBIDDEN" and plan["ids"] == old_plan["ids"]
         and plan["controls"] == controls and set(plan["candidates"]) == set(SCREEN_MASKS),
         "factorial prefill plan")
    replay_raw = files["screen/primary-replay.json"]
    need(sha(replay_raw) == PRIMARY_REPLAY_SHA and blob(replay_raw) == PRIMARY_REPLAY_BLOB,
         "previous primary replay pin")
    previous_replay = strict(replay_raw)
    need(previous_replay["status"] == "R23H_ALL8_ABLATION_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS"
         and previous_replay["batch_report_sha256"] == "8ab5c2fc607ad49a589636df39ab46f801327d7c39cbbf70ce29b0fb2b23d2af"
         and previous_replay["eligible_for_secondary_fit_only"] == [], "previous primary replay")
    masks = {}
    for name in SCREEN_MASKS:
        raw_mask = files["screen/masks/" + name + ".json"]
        need(blob(raw_mask) == BLOBS[name], "mask Git blob " + name)
        receipt = strict(raw_mask)
        mask = receipt["mask"]
        need(receipt["status"] == "R23H_ALL8_FACTORIAL_LAYER_REVERSION_EXPLORATORY_NOT_ADMITTED"
             and receipt["method"] == name
             and receipt["mask_sha256"] == MASKS[name]
             and sha(canon(mask)) == MASKS[name]
             and receipt["changed_layers"] == CHANGED[name], "mask identity " + name)
        candidate = plan["candidates"][name]
        need(candidate["receipt_sha256"] == RECEIPTS[name]
             and candidate["git_blob_sha"] == BLOBS[name]
             and candidate["mask_sha256"] == MASKS[name]
             and candidate["changed_layers"] == CHANGED[name], "plan mask binding " + name)
        check_mask(mask, name)
        masks[name] = mask

    r23g_prior = {}
    for index, receipt_hash in enumerate(recovery["receipt_sha256"]):
        value = strict(files["prior-r23g/" + f"{index:02d}.json"])
        need(sha(files["prior-r23g/" + f"{index:02d}.json"]) == receipt_hash,
             "R23G receipt SHA " + str(index))
        r23g_prior[value["id"]] = value

    screen_events = [strict(line) for line in files["screen/run.log"].splitlines() if line.startswith(b"{")]
    screen_ready = [event for event in screen_events if event["status"] == "R23H_ALL8_FACTORIAL_TEACHER_READY"]
    screen_capture_events = [event for event in screen_events if event["status"] == "R23H_ALL8_FACTORIAL_PREFILL_CAPTURED"]
    screen_done = [event for event in screen_events if event["status"] == "R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(screen_ready) == 1 and screen_ready[0]["teacher"] == TEACHER
         and screen_ready[0]["threads"] == 8
         and len(screen_capture_events) == 8 and len(screen_done) == 1,
         "factorial screen event inventory")
    screen_caps = {}
    for index, event in enumerate(screen_capture_events):
        value = exact("screen/" + f"{index:02d}.json", event["receipt_sha256"])
        rid = plan["ids"][index]
        need(event["index"] == value["index"] == index and event["id"] == value["id"] == rid
             and value["plan_sha256"] == FACTORIAL_PLAN
             and value["teacher_sha256"] == r23g_prior[rid]["original96"]["teacher_hashes"][0]["sha256"],
             "factorial screen receipt provenance")
        need(set(value["metrics"]) == set(SCREEN_NAMES), "factorial screen metric inventory")
        for name, metric in value["metrics"].items():
            need(finite(metric["kl"]) and finite(metric["nmse"])
                 and type(metric["argmax_mismatch"]) is int and metric["argmax_mismatch"] in (0, 1)
                 and len(metric["logits_sha256"]) == 64
                 and metric["mask_sha256"] == (controls[name] if name in controls else MASKS[name]),
                 "factorial screen metric " + name)
        for name in ("original96", "all8", "all10"):
            need(value["metrics"][name] == old_screen[rid]["metrics"][name],
                 "old control replay " + name)
        old = r23g_prior[rid]["candidate96"]
        old_logit = old["logits"][0]
        metric = value["metrics"]["r23g"]
        need(metric["logits_sha256"] == old["masked_hashes"][0]["sha256"]
             and abs(metric["kl"] - old_logit["teacher_to_masked_kl"]) < 1e-12
             and abs(metric["nmse"] - old_logit["nmse"]) < 1e-12
             and metric["argmax_mismatch"] == int(old_logit["teacher_argmax"] != old_logit["masked_argmax"]),
             "R23G exact screen replay")
        screen_caps[rid] = value
    screen_metrics = {
        name: {
            key: (sum if key == "argmax_mismatch" else avg)([x["metrics"][name][key] for x in screen_caps.values()])
            for key in ("kl", "nmse", "argmax_mismatch")
        }
        for name in SCREEN_NAMES
    }
    screen_eligible = [
        name for name in SCREEN_MASKS
        if all(screen_metrics[name][key] <= screen_metrics["original96"][key] + 1e-12
               for key in screen_metrics[name])
        and screen_metrics[name]["kl"] < screen_metrics["r23g"]["kl"]
    ]
    screen_eligible.sort(key=lambda name: (screen_metrics[name]["kl"], screen_metrics[name]["nmse"], name))
    screen_report = exact("screen/report.json", FACTORIAL_REPORT)
    need(screen_report["status"] == "R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
         and screen_report["plan_sha256"] == FACTORIAL_PLAN
         and screen_report["samples"] == 8
         and screen_report["metrics"] == screen_metrics
         and screen_report["eligible_for_primary_fit_only"] == screen_eligible
         and screen_report["selected"] == (screen_eligible[0] if screen_eligible else None)
         and screen_eligible == list(NAMES)
         and screen_done[0]["report_sha256"] == FACTORIAL_REPORT,
         "factorial prefill aggregate replay")

    # Verify exact primary admission and all 20 fixed-trajectory receipts.
    admission_raw = files["batch/admission.json"]
    admission = exact("batch/admission.json", BATCH_ADMISSION)
    admission_sha = sha(admission_raw)
    need(admission["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_FIT_ONLY"
         and admission["screen_plan_sha256"] == FACTORIAL_PLAN
         and admission["screen_report_sha256"] == FACTORIAL_REPORT
         and admission["previous_recovery_sha256"] == REC
         and admission["prior_all10_report_sha256"] == ALL10_REPORT
         and admission["teacher"] == TEACHER
         and admission["holdout"] == "FORBIDDEN"
         and admission["indices"] == list(INDICES)
         and admission["ids"] == [row["id"] for _, row in selected]
         and admission["candidates"] == {name: MASKS[name] for name in NAMES},
         "factorial primary admission")
    batch_events = [strict(line) for line in files["batch/run.log"].splitlines() if line.startswith(b"{")]
    batch_ready = [event for event in batch_events if event["status"] == "R23H_FACTORIAL_PRIMARY_TEACHER_READY"]
    batch_captures = [event for event in batch_events if event["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_CAPTURED"]
    batch_done = [event for event in batch_events if event["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(batch_ready) == 1 and batch_ready[0]["teacher"] == TEACHER
         and len(batch_captures) == 10 and len(batch_done) == 1
         and batch_done[0]["report_sha256"] == BATCH_REPORT,
         "factorial primary event inventory")
    captures = {name: {} for name in NAMES}
    for event_index, (index, row) in enumerate(selected):
        event = batch_captures[event_index]
        need(event["index"] == index and event["id"] == row["id"]
             and set(event["receipt_sha256"]) == set(NAMES), "primary event row binding")
        for name in NAMES:
            path = "batch/" + name + "-" + f"{index:02d}" + ".json"
            value = exact(path, event["receipt_sha256"][name])
            check_capture(value, row, index, admission_sha, MASKS[name])
            need(value["original96"] == r23g_prior[row["id"]]["original96"],
                 "exact historical original96 baseline")
            if row["id"] in screen_caps:
                metric = screen_caps[row["id"]]["metrics"][name]
                prefill = value["candidate96"]["logits"][0]
                need(metric["logits_sha256"] == value["candidate96"]["masked_hashes"][0]["sha256"]
                     and abs(metric["kl"] - prefill["teacher_to_masked_kl"]) < 1e-12
                     and abs(metric["nmse"] - prefill["nmse"]) < 1e-12
                     and metric["argmax_mismatch"] == int(prefill["teacher_argmax"] != prefill["masked_argmax"]),
                     "prefill-to-primary exact replay")
            captures[name][row["id"]] = value

    groups = {name: aggregate([row for _, row in selected], captures[name], name) for name in NAMES}
    violations = {
        name: [
            group + "/" + metric
            for group in groups[name]
            for metric in ("kl", "nmse", "argmax_mismatch")
            if groups[name][group]["candidate96"][metric] > groups[name][group]["original96"][metric] + 1e-12
        ]
        for name in NAMES
    }
    eligible = sorted(
        (name for name in NAMES if not violations[name]),
        key=lambda name: (
            avg([
                groups[name][group]["candidate96"][metric] / max(groups[name][group]["original96"][metric], 1e-12)
                for group in groups[name] for metric in ("kl", "nmse")
            ]),
            name,
        ),
    )
    report = exact("batch/report.json", BATCH_REPORT)
    need(report["status"] == "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"
         and report["admission_sha256"] == admission_sha
         and report["samples_per_candidate"] == 10
         and report["candidates"] == admission["candidates"]
         and report["groups"] == groups
         and report["violations"] == violations
         and report["eligible_for_secondary_fit_only"] == eligible
         and report["selected"] == (eligible[0] if eligible else None),
         "factorial primary aggregate replay")
    need(batch_done[0]["report_sha256"] == BATCH_REPORT, "primary terminal report hash")

    result = {
        "status": "R23H_ALL8_FACTORIAL_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS",
        "archive_sha256": ARCHIVE_SHA,
        "factorial_plan_sha256": FACTORIAL_PLAN,
        "factorial_prefill_report_sha256": FACTORIAL_REPORT,
        "batch_admission_sha256": BATCH_ADMISSION,
        "batch_report_sha256": BATCH_REPORT,
        "samples_per_candidate": 10,
        "groups": groups,
        "violations": violations,
        "eligible_for_secondary_fit_only": eligible,
        "selected": eligible[0] if eligible else None,
        "checks": [
            "exact archive inventory and terminal exits",
            "18-row Fit-only inventory and Holdout exclusion",
            "all10 predecessor failure replay",
            "original96/R23G/all8/all10 historical control binding",
            "four exact 47-layer x 96-expert factorial masks",
            "eight-row factorial Prefill replay and independent aggregate",
            "ten-row bidirectional primary receipt replay for both candidates",
            "exact original96 baseline and Prefill-to-primary replay",
            "independent KL/NMSE/Argmax aggregation and secondary-Fit gate",
        ],
        "limits": "Fixed Fit trajectories only. No Fresh, Holdout, free-running translation, physical build, or product qualification.",
    }
    result_raw = canon(result) + b"\n"
    target = root / "factorial-primary-replay.json"
    with target.open("xb") as stream:
        stream.write(result_raw)
    print(json.dumps({"status": result["status"], "sha256": sha(result_raw),
                      "eligible_for_secondary_fit_only": eligible, "selected": result["selected"],
                      "violations": violations}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
