#!/usr/bin/env python3
"""Generate three isolated L40/L42 96-expert candidates from R23H Fit evidence."""
import hashlib
import io
import json
import pathlib
import tarfile
import sys

ARCHIVE_SHA = "80f8dc3d8ba4e598652a286e68e8eecfc32e54046855a9a194c8b7e8869b95cd"
PRIMARY_REPORT_SHA = "eba41d87b5d1f418923584e778a303bdab0dbd1f78f54702ceb86fdea08fb4da"
PRIMARY_REPLAY_SHA = "7f1b8a0b116192837e016711abca20d020265bafcafd41ed7d8bc861c02e93e3"
LAYER_SUMMARY_SHA = "9dc475eecf9ee68e2b30c65a38f10a75c5797d86831b8b11412044c30ee7d7a6"
FACTORIAL_PLAN_SHA = "7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac"
FACTORIAL_REPORT_SHA = "3953ed75fb4275019b51df5d6e8e43b91a4e77c37747330be365c3cd67f8da05"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
RECOVERY_SHA = "ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
CANDIDATES = {
    "r23h_l40_l42_only": [40, 42],
    "r23h_l40_only": [40],
    "r23h_l42_only": [42],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def check_mask(mask, label):
    if set(mask) != {str(i) for i in range(1, 48)}:
        raise ValueError(label + " layer inventory")
    for layer, experts in mask.items():
        if type(experts) is not list or len(experts) != 96 or experts != sorted(set(experts)):
            raise ValueError(label + " expert count/order at layer " + layer)
        if any(type(expert) is not int or not 0 <= expert < 128 for expert in experts):
            raise ValueError(label + " expert range at layer " + layer)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: r23h-factorial2-prepare.py <factorial-primary.tgz> <output-dir>")
    archive_path = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    raw_archive = archive_path.read_bytes()
    if sha(raw_archive) != ARCHIVE_SHA:
        raise ValueError("primary evidence archive SHA mismatch")
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:gz") as archive:
        members = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    def get(name):
        if name not in members:
            raise ValueError("missing archive member " + name)
        return members[name]
    replay_raw = get("batch/report.json")
    if sha(replay_raw) != PRIMARY_REPORT_SHA:
        raise ValueError("primary report pin")
    screen_plan_raw = get("screen/plan.json")
    if sha(screen_plan_raw) != FACTORIAL_PLAN_SHA:
        raise ValueError("factorial screen plan pin")
    if sha(get("screen/report.json")) != FACTORIAL_REPORT_SHA:
        raise ValueError("factorial screen report pin")
    if sha(get("recovery/recovery-receipt.json")) != RECOVERY_SHA:
        raise ValueError("recovery receipt pin")
    replay = strict(get("batch/report.json"))
    if replay["status"] != "R23H_ALL8_FACTORIAL_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION" or replay["eligible_for_secondary_fit_only"]:
        raise ValueError("primary batch must remain failed/non-admitted")
    prefill_plan = strict(screen_plan_raw)
    if prefill_plan["teacher"] != TEACHER or prefill_plan["holdout"] != "FORBIDDEN":
        raise ValueError("prefill teacher/holdout contract")

    original_obj = strict(get("old-screen/original.json"))
    original = original_obj["candidates"]["96"]["kept"]
    old_plan = strict(get("old-screen/plan.json"))
    all8 = next(item["kept"] for item in old_plan["screen"] if item["name"] == "all8")
    check_mask(original, "original96")
    check_mask(all8, "all8")
    changed_base = [layer for layer in range(1, 48) if set(original[str(layer)]) != set(all8[str(layer)])]
    if changed_base != [10, 15, 40, 42]:
        raise ValueError("unexpected all8 source mask changes: " + str(changed_base))

    original_sha = sha(canon(original))
    all8_sha = sha(canon(all8))
    out.mkdir(parents=True, exist_ok=False)
    masks = {}
    receipts = {}
    for name, retained_layers in CANDIDATES.items():
        mask = {str(layer): list(original[str(layer)]) for layer in range(1, 48)}
        for layer in retained_layers:
            mask[str(layer)] = list(all8[str(layer)])
        check_mask(mask, name)
        changed = [layer for layer in range(1, 48) if set(mask[str(layer)]) != set(original[str(layer)])]
        if changed != retained_layers:
            raise ValueError(name + " changed layer binding")
        receipt = {
            "status": "R23H_FACTORIAL2_LAYER_ONLY_MASK_EXPLORATORY_NOT_ADMITTED",
            "method": name,
            "source_archive_sha256": ARCHIVE_SHA,
            "source_primary_report_sha256": PRIMARY_REPORT_SHA,
            "source_primary_independent_replay_sha256": PRIMARY_REPLAY_SHA,
            "source_layer_summary_sha256": LAYER_SUMMARY_SHA,
            "source_factorial_prefill_plan_sha256": FACTORIAL_PLAN_SHA,
            "source_factorial_prefill_report_sha256": FACTORIAL_REPORT_SHA,
            "source_original96_mask_sha256": original_sha,
            "source_all8_mask_sha256": all8_sha,
            "retained_all8_layers": retained_layers,
            "restored_original96_layers": [layer for layer in range(1, 48) if layer not in retained_layers],
            "changed_layers": changed,
            "mask_sha256": sha(canon(mask)),
            "mask": mask,
        }
        receipt_raw = canon(receipt) + b"\n"
        (out / (name + ".json")).write_bytes(receipt_raw)
        masks[name] = mask
        receipts[name] = {
            "receipt_sha256": sha(receipt_raw),
            "git_blob_sha": blob(receipt_raw),
            "mask_sha256": receipt["mask_sha256"],
            "changed_layers": changed,
            "retained_all8_layers": retained_layers,
        }

    plan = {
        "status": "R23H_FACTORIAL2_PREFILL_SCREEN_ONLY",
        "holdout": "FORBIDDEN",
        "teacher": TEACHER,
        "recovery_sha256": RECOVERY_SHA,
        "source_archive_sha256": ARCHIVE_SHA,
        "source_primary_report_sha256": PRIMARY_REPORT_SHA,
        "source_primary_independent_replay_sha256": PRIMARY_REPLAY_SHA,
        "source_layer_summary_sha256": LAYER_SUMMARY_SHA,
        "source_factorial_prefill_plan_sha256": FACTORIAL_PLAN_SHA,
        "source_factorial_prefill_report_sha256": FACTORIAL_REPORT_SHA,
        "source_original96_mask_sha256": original_sha,
        "source_all8_mask_sha256": all8_sha,
        "ids": prefill_plan["ids"],
        "controls": prefill_plan["controls"],
        "candidate_order": list(CANDIDATES),
        "candidates": receipts,
        "scope": "Three 96-expert masks retaining only Layer 40 and/or Layer 42 from the all8 mask; all other layers use original96. Eight zh-my calibration Fit Prefill rows only.",
    }
    plan_raw = canon(plan) + b"\n"
    (out / "plan.json").write_bytes(plan_raw)
    print(json.dumps({
        "status": plan["status"],
        "plan_sha256": sha(plan_raw),
        "plan_git_blob": blob(plan_raw),
        "source_original96_mask_sha256": original_sha,
        "source_all8_mask_sha256": all8_sha,
        "candidates": receipts,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
