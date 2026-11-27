#!/usr/bin/env python3
import sys, tarfile, io, json, hashlib, math, pathlib

ARCHIVE_SHA = "94d253fca88a3be7697dea05243cc5889ef4ae2739b8e9acb4be50802c824a8e"
PLAN_SHA = "7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac"
PLAN_BLOB = "415f2526c82356e216d74105301771b3060f3461"
REPORT_SHA = "3953ed75fb4275019b51df5d6e8e43b91a4e77c37747330be365c3cd67f8da05"
FACTORIAL_BLOB = "da1231a116036dfce9eeaab946331e4b32859ce5"
PRIMARY_REPLAY_SHA = "f6927cccf2279a56673cad99b31559f199252fb734ff2e58caa6c667b8df1ebc"
PRIMARY_REPLAY_PLAN_SHA = "89ab4d14f73563ee6e1a05cdeeb2143cf67c277a43c78629c2468b310514c552"
PRIMARY_REPLAY_BLOB = "1ae914e4e1e0c9729c43b237100b6fc20a5fbefb"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
EXPANDED_SHA = "21b4a54f626c32727284f32f34f3b43b54ff6e68d56ac4499934ee652fb6d166"
CANDIDATE_SHA = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
ORIGINAL_BLOB = "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
OLD_PLAN_SHA = "b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT_SHA = "8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
PRIMARY_REPORT_SHA = "8ab5c2fc607ad49a589636df39ab46f801327d7c39cbbf70ce29b0fb2b23d2af"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
CANDIDATES = ("all8_revert10", "all8_revert15", "all8_revert10_40_42", "all8_revert15_40_42")
EXPECTED_CHANGED = {
    "all8_revert10": [15, 40, 42],
    "all8_revert15": [10, 40, 42],
    "all8_revert10_40_42": [15],
    "all8_revert15_40_42": [10],
}
MASK_NAMES = ("original96", "r23g", "all8", "all10", *CANDIDATES)

def need(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def blob(b):
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def pairs_no_dupes(pairs):
    out = {}
    for k, v in pairs:
        need(k not in out, "duplicate JSON key: " + k)
        out[k] = v
    return out

def parse(raw, label):
    return json.loads(raw, object_pairs_hook=pairs_no_dupes,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid JSON constant " + x)))

def finite_nonnegative(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0

def read_archive(path):
    raw = pathlib.Path(path).read_bytes()
    need(sha(raw) == ARCHIVE_SHA, "archive SHA256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.replace("\\", "/")
            need(not name.startswith("/") and ".." not in pathlib.PurePosixPath(name).parts, "unsafe archive path")
            need(m.size <= 25 * 1024 * 1024, "oversize archive member: " + name)
            need(name not in files, "duplicate archive path: " + name)
            f = tf.extractfile(m)
            need(f is not None, "unreadable archive member: " + name)
            files[name] = f.read()
    return files

def get(files, path):
    need(path in files, "missing archive file: " + path)
    return files[path]

def jget(files, path):
    return parse(get(files, path), path)

def events(raw, path):
    out = []
    for line in raw.splitlines():
        if line.startswith(b"{"):
            out.append(parse(line, path))
    return out

def check_mask(mask, name):
    need(set(mask) == {str(i) for i in range(1, 48)}, name + " layer inventory")
    for layer, ids in mask.items():
        need(type(ids) is list and len(ids) == 96 and ids == sorted(set(ids)), name + " expert count/order at layer " + layer)
        need(all(type(i) is int and 0 <= i < 128 for i in ids), name + " expert id range at layer " + layer)

def main():
    need(len(sys.argv) == 2, "usage: verify_r23h_all8_prefill.py <archive.tar.gz>")
    F = read_archive(sys.argv[1])
    prep = "r23-results/r23h-all8-factorial-prefill-v2/"
    run = "r23h-all8-factorial-prefill-run-v5/"
    oldsd = "r23-results/r23h-pareto-prefill-v1/"
    r23gdir = "r23-results/r23g-l38-propagation-v1/"

    need(get(F, run + "exit-code").decode().strip() == "0", "nonzero run exit")
    need(get(F, run + "run.stderr") == b"", "nonempty run stderr")
    sh = get(F, run + "run.sh").decode()
    need('CUDA_VISIBLE_DEVICES=""' in sh and "OMP_NUM_THREADS=8" in sh and "MKL_NUM_THREADS=8" in sh, "runtime thread/device contract")
    factorial = get(F, run + "factorial.py")
    need(blob(factorial) == FACTORIAL_BLOB, "factorial runner source blob")
    primary_replay = get(F, run + "primary-replay.json")
    need(sha(primary_replay) == PRIMARY_REPLAY_SHA and blob(primary_replay) == PRIMARY_REPLAY_BLOB, "prior primary replay pin")

    plan_raw = get(F, prep + "plan.json")
    need(sha(plan_raw) == PLAN_SHA and blob(plan_raw) == PLAN_BLOB, "factorial plan pin")
    plan = parse(plan_raw, prep + "plan.json")
    need(plan["source_primary_independent_replay_sha256"] == PRIMARY_REPLAY_PLAN_SHA, "prior primary replay plan binding")
    need(plan["status"] == "R23H_ALL8_FACTORIAL_PREFILL_SCREEN_ONLY" and plan["holdout"] == "FORBIDDEN", "plan scope")
    need(len(plan["ids"]) == 8 and len(set(plan["ids"])) == 8, "plan row count/uniqueness")
    need(set(plan["candidates"]) == set(CANDIDATES), "candidate inventory")

    recovery_raw = get(F, "r23g-recovery-ssd-38ed4798/recovery-receipt.json")
    need(sha(recovery_raw) == RECOVERY_SHA, "R23G recovery receipt pin")
    recovery = parse(recovery_raw, "recovery receipt")
    need(recovery["status"] == "R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(recovery["receipt_sha256"]) == 18, "R23G recovery status/inventory")
    expanded_raw = get(F, "r23g-recovery-ssd-38ed4798/expanded.json")
    need(sha(expanded_raw) == EXPANDED_SHA, "expanded fit plan pin")
    expanded = parse(expanded_raw, "expanded")
    fit_rows = [r for r in expanded["fit"] if r["direction"] == "zh-my"]
    need(len(fit_rows) == 8 and [r["id"] for r in fit_rows] == plan["ids"], "fit row binding")
    holdout_lines = {r["line"] for r in expanded["holdout"]}
    need(not ({r["line"] for r in fit_rows} & holdout_lines), "holdout contamination")
    candidate_raw = get(F, "r23g-recovery-ssd-38ed4798/candidate.json")
    need(sha(candidate_raw) == CANDIDATE_SHA, "R23G candidate mask pin")
    r23g_mask = parse(candidate_raw, "R23G candidate")["kept"]
    original = parse(get(F, "r23h-pareto-screen-v1/original.json"), "original mask")
    need(blob(get(F, "r23h-pareto-screen-v1/original.json")) == ORIGINAL_BLOB, "original mask blob")
    original_mask = original["candidates"]["96"]["kept"]
    check_mask(original_mask, "original96")
    check_mask(r23g_mask, "r23g")
    controls = {
        "original96": sha(canon(original_mask)),
        "r23g": sha(canon(r23g_mask)),
    }

    oldplan_raw = get(F, oldsd + "plan.json")
    oldreport_raw = get(F, oldsd + "report.json")
    need(sha(oldplan_raw) == OLD_PLAN_SHA and sha(oldreport_raw) == OLD_REPORT_SHA, "previous screen plan/report pins")
    oldplan = parse(oldplan_raw, "old plan")
    oldreport = parse(oldreport_raw, "old report")
    need(oldreport["selected"] == "all8" and oldreport["samples"] == 8, "previous screen outcome")
    oldmask = {x["name"]: x["kept"] for x in oldplan["screen"]}
    controls["all8"] = sha(canon(oldmask["all8"]))
    controls["all10"] = sha(canon(oldmask["all10"]))
    need(controls == plan["controls"], "control mask bindings")
    for n, m in oldmask.items():
        if n in ("all8", "all10"):
            check_mask(m, n)

    prev_report_raw = get(F, "r23-results/r23h-all8-ablation-primary-v3/report.json")
    need(sha(prev_report_raw) == PRIMARY_REPORT_SHA, "previous primary report pin")
    prev_report = parse(prev_report_raw, "previous primary report")
    need(prev_report["status"] == "R23H_ALL8_ABLATION_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"
         and prev_report["eligible_for_secondary_fit_only"] == [], "previous primary rejection gate")
    replay = parse(primary_replay, "primary replay")
    need(replay["status"] == "R23H_ALL8_ABLATION_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS"
         and replay["batch_report_sha256"] == PRIMARY_REPORT_SHA
         and replay["eligible_for_secondary_fit_only"] == [], "prior independent replay gate")

    # Verify all four mask receipts and their per-layer cardinality.
    for n in CANDIDATES:
        raw = get(F, prep + n + ".json")
        c = plan["candidates"][n]
        need(sha(raw) == c["receipt_sha256"] and blob(raw) == c["git_blob_sha"], "mask receipt pin " + n)
        x = parse(raw, n)
        need(x["status"] == "R23H_ALL8_FACTORIAL_LAYER_REVERSION_EXPLORATORY_NOT_ADMITTED"
             and x["method"] == n and x["changed_layers"] == EXPECTED_CHANGED[n], "mask receipt provenance " + n)
        m = x["mask"]
        check_mask(m, n)
        mh = sha(canon(m))
        need(mh == x["mask_sha256"] == c["mask_sha256"], "mask content hash " + n)

    # Reconstruct the old prefill controls and the historical R23G outputs.
    old_events = events(get(F, "r23h-pareto-screen-v1/run.log"), "old screen log")
    oldcaps = [e for e in old_events if e["status"] == "R23H_PREFILL_CAPTURED"]
    need(len(oldcaps) == 8, "old screen capture event count")
    oldscreen = {}
    for i, e in enumerate(oldcaps):
        raw = get(F, oldsd + f"{i:02d}.json")
        need(sha(raw) == e["sha256"], "old screen receipt SHA " + str(i))
        x = parse(raw, "old screen receipt")
        need(x["index"] == i and x["id"] == plan["ids"][i] and x["plan_sha256"] == OLD_PLAN_SHA, "old screen receipt binding")
        oldscreen[x["id"]] = x

    prior = {}
    for i, h in enumerate(recovery["receipt_sha256"]):
        raw = get(F, r23gdir + f"{i:02d}.json")
        need(sha(raw) == h, "R23G receipt SHA " + str(i))
        x = parse(raw, "R23G receipt")
        prior[x["id"]] = x

    corpus_raw = get(F, "run-d01b-r19/corpus/corpus.jsonl")
    need(sha(corpus_raw) == CORPUS_SHA, "calibration corpus pin")
    corpus = [parse(line, "corpus line") for line in corpus_raw.splitlines() if line]
    ctx = {x["id"]: x for x in corpus if x.get("split") == "calibration" and x.get("id") in set(plan["ids"])}
    need(set(ctx) == set(plan["ids"]), "calibration corpus rows")
    for r in fit_rows:
        need(sha(canon(ctx[r["id"]]["prompt_ids"])) == r["prompt_ids_sha256"], "prompt ids hash " + r["id"])
        need(ctx[r["id"]]["direction"] == "zh-my", "direction " + r["id"])
        need(r["id"] in prior, "missing R23G prior " + r["id"])
        need(len(prior[r["id"]]["original96"]["teacher_hashes"][0]["sha256"]) == 64, "teacher logit hash " + r["id"])

    # Verify terminal events and all eight current capture receipts.
    run_events = events(get(F, run + "run.log"), "factorial run log")
    ready = [e for e in run_events if e["status"] == "R23H_ALL8_FACTORIAL_TEACHER_READY"]
    captures = [e for e in run_events if e["status"] == "R23H_ALL8_FACTORIAL_PREFILL_CAPTURED"]
    done = [e for e in run_events if e["status"] == "R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(ready) == 1 and ready[0]["teacher"] == TEACHER and ready[0]["threads"] == 8, "teacher ready event")
    need(len(captures) == 8 and len(done) == 1, "current run event inventory")
    masks = {**controls, **{n: plan["candidates"][n]["mask_sha256"] for n in CANDIDATES}}
    names = list(MASK_NAMES)
    receipts = []
    for i, e in enumerate(captures):
        rid = plan["ids"][i]
        raw = get(F, prep + f"{i:02d}.json")
        need(sha(raw) == e["receipt_sha256"], "current receipt SHA " + str(i))
        x = parse(raw, "current receipt")
        need(x["status"] == "R23H_ALL8_FACTORIAL_PREFILL_CAPTURED_NOT_ADMISSION"
             and x["index"] == i and x["id"] == rid and e["id"] == rid
             and x["plan_sha256"] == PLAN_SHA
             and x["teacher_sha256"] == prior[rid]["original96"]["teacher_hashes"][0]["sha256"], "current receipt provenance " + str(i))
        need(set(x["metrics"]) == set(names), "metric inventory " + rid)
        for n, q in x["metrics"].items():
            need(finite_nonnegative(q["kl"]) and finite_nonnegative(q["nmse"]), "nonfinite KL/NMSE " + n + "/" + rid)
            need(type(q["argmax_mismatch"]) is int and q["argmax_mismatch"] in (0, 1), "argmax mismatch " + n + "/" + rid)
            need(len(q["logits_sha256"]) == 64 and q["mask_sha256"] == masks[n], "metric mask/logit binding " + n + "/" + rid)
        # Controls must replay byte-for-byte against the previous screen.
        for n in ("original96", "all8", "all10"):
            need(x["metrics"][n] == oldscreen[rid]["metrics"][n], "exact previous-screen replay " + n + "/" + rid)
        # R23G must replay against its historical candidate96 receipt.
        prior_row = prior[rid]
        old_metric = prior_row["candidate96"]["logits"][0]
        old_hash = prior_row["candidate96"]["masked_hashes"][0]["sha256"]
        q = x["metrics"]["r23g"]
        need(q["logits_sha256"] == old_hash
             and abs(q["kl"] - old_metric["teacher_to_masked_kl"]) < 1e-12
             and abs(q["nmse"] - old_metric["nmse"]) < 1e-12
             and q["argmax_mismatch"] == int(old_metric["teacher_argmax"] != old_metric["masked_argmax"]),
             "exact R23G replay " + rid)
        receipts.append(x)

    # Independently recompute aggregate means and the candidate gate.
    aggregates = {}
    for n in names:
        aggregates[n] = {
            "kl": math.fsum(x["metrics"][n]["kl"] for x in receipts) / 8,
            "nmse": math.fsum(x["metrics"][n]["nmse"] for x in receipts) / 8,
            "argmax_mismatch": sum(x["metrics"][n]["argmax_mismatch"] for x in receipts),
        }
    eligible = [
        n for n in CANDIDATES
        if all(aggregates[n][k] <= aggregates["original96"][k] + 1e-12 for k in aggregates[n])
        and aggregates[n]["kl"] < aggregates["r23g"]["kl"]
    ]
    eligible.sort(key=lambda n: (aggregates[n]["kl"], aggregates[n]["nmse"], n))
    report_raw = get(F, prep + "report.json")
    need(sha(report_raw) == REPORT_SHA and done[0]["report_sha256"] == REPORT_SHA, "terminal report SHA")
    report = parse(report_raw, "factorial report")
    need(report["status"] == "R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
         and report["plan_sha256"] == PLAN_SHA and report["samples"] == 8, "report scope")
    need(report["metrics"] == aggregates, "aggregate metrics replay")
    need(report["eligible_for_primary_fit_only"] == eligible
         and report["selected"] == (eligible[0] if eligible else None), "eligibility decision replay")
    need(eligible == ["all8_revert15_40_42", "all8_revert15"]
         and report["selected"] == "all8_revert15_40_42", "expected candidate ordering")

    print(json.dumps({
        "status": "R23H_ALL8_FACTORIAL_PREFILL_INDEPENDENT_RECEIPT_REPLAY_PASS",
        "archive_sha256": ARCHIVE_SHA,
        "plan_sha256": PLAN_SHA,
        "report_sha256": REPORT_SHA,
        "samples": 8,
        "metrics": aggregates,
        "eligible_for_primary_fit_only": eligible,
        "selected": report["selected"],
        "checks": [
            "exact plan and four mask pins",
            "47 layers x 96 unique experts per mask",
            "Fit-only row inventory and Holdout exclusion",
            "8 capture receipts and terminal event",
            "original96/all8/all10 historical replay",
            "R23G historical replay",
            "independent metric aggregation and eligibility gate",
            "8-thread CPU execution and clean exit"
        ]
    }, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()
