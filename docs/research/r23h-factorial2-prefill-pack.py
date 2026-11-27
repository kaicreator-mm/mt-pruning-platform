#!/usr/bin/env python3
"""Pack bounded R23H factorial2 Fit evidence for independent 8845HS replay."""
import hashlib
import json
import pathlib
import tarfile

R = pathlib.Path.home() / ".cache/mt-pruning"
Q = R / "r23-results"
PREP = Q / "r23h-factorial2-prefill-v1"
RUN = R / "r23h-factorial2-prefill-run-v1"
D = R / "r23h-pareto-screen-v1"
USB = R / "r23g-recovery-ssd-38ed4798"
SD = Q / "r23h-pareto-prefill-v1"
OLD_FACTORIAL = Q / "r23h-all8-factorial-prefill-v2"
PRIMARY = Q / "r23h-all8-factorial-primary-v3"
PRIOR_R23G = Q / "r23g-l38-propagation-v1"
CORPUS = R / "run-d01b-r19/corpus/corpus.jsonl"

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
FACTORIAL_REPORT_SHA = "3953ed75fb4275019b51df5d6e8e43b91a4e77c37747330be365c3cd67f8da05"
MASKS = {
    "r23h_l40_l42_only": ("33cbec0828eef568abe6c77ffb586699ebe76cb54a2e399fcd5fff631252b5e1", "98b8b39d1b56db0e830c6fba9bb6667927ab92bc"),
    "r23h_l40_only": ("819b635fc56ae2d539987a5288bd2c16683242e06b75f8759e4c5296fed61e0a", "d51357b9f2c9e05e93876aebf03efd8ace44fb30"),
    "r23h_l42_only": ("6f89cf26d2de7a3f0ceba026c08ce160490ccfd15167ce194c66704436fb0e47", "47fb3ec41345a48d219dabd40a5fd0a63dfb53f4"),
}
PINS = {
    "driver": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "collector": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def add(files, prefix, entries):
    for name, path in entries:
        assert path.is_file() and not path.is_symlink(), str(path)
        limit = 32 * 1024 * 1024 if name in ("corpus/corpus.jsonl",) else 2 * 1024 * 1024
        assert path.stat().st_size <= limit, (str(path), path.stat().st_size)
        files[prefix + name] = path


def main():
    assert (RUN / "preflight-v3.exit-code").read_text().strip() == "0"
    assert (RUN / "preflight-v3.stderr").read_bytes() == b""
    assert (RUN / "execute-v1.exit-code").read_text().strip() == "0"
    assert (RUN / "execute-v1.stderr").read_bytes() == b""

    plan_raw = (PREP / "plan.json").read_bytes()
    assert sha(plan_raw) == PLAN_SHA and blob(plan_raw) == PLAN_BLOB
    plan = json.loads(plan_raw)
    assert plan["status"] == "R23H_FACTORIAL2_PREFILL_SCREEN_ONLY"
    assert plan["holdout"] == "FORBIDDEN"
    assert plan["source_primary_report_sha256"] == PRIMARY_REPORT_SHA
    assert plan["source_primary_independent_replay_sha256"] == PRIMARY_REPLAY_SHA
    assert plan["source_layer_summary_sha256"] == LAYER_SUMMARY_SHA

    report_raw = (PREP / "report.json").read_bytes()
    report = json.loads(report_raw)
    assert report["status"] == "R23H_FACTORIAL2_PREFILL_COMPLETE_NOT_FRESH_ADMISSION"
    assert report["plan_sha256"] == PLAN_SHA and report["samples"] == 8
    assert len([x for x in RUN.joinpath("execute-v1.log").read_text().splitlines() if '"status": "R23H_FACTORIAL2_PREFILL_CAPTURED"' in x]) == 8
    assert any('"status": "R23H_FACTORIAL2_TEACHER_READY"' in x for x in RUN.joinpath("execute-v1.log").read_text().splitlines())

    assert sha((PRIMARY / "report.json").read_bytes()) == PRIMARY_REPORT_SHA
    assert sha((RUN / "primary-replay.json").read_bytes()) == PRIMARY_REPLAY_SHA
    assert sha((RUN / "layer-summary.json").read_bytes()) == LAYER_SUMMARY_SHA
    assert blob((RUN / "factorial2.py").read_bytes()) == FACTORIAL2_BLOB
    assert sha((USB / "recovery-receipt.json").read_bytes()) == RECOVERY_SHA
    assert sha((USB / "expanded.json").read_bytes()) == EXPANDED_SHA
    assert sha((USB / "candidate.json").read_bytes()) == CANDIDATE_SHA
    assert sha(CORPUS.read_bytes()) == CORPUS_SHA
    assert blob((D / "original.json").read_bytes()) == ORIGINAL_BLOB
    assert sha((SD / "plan.json").read_bytes()) == PARETO_PLAN_SHA
    assert sha((SD / "report.json").read_bytes()) == PARETO_REPORT_SHA
    assert sha((OLD_FACTORIAL / "plan.json").read_bytes()) == FACTORIAL_PLAN_SHA
    assert sha((OLD_FACTORIAL / "report.json").read_bytes()) == FACTORIAL_REPORT_SHA
    assert blob((D / "driver.py").read_bytes()) == PINS["driver"]
    assert blob((D / "collector.py").read_bytes()) == PINS["collector"]
    recovery = json.loads((USB / "recovery-receipt.json").read_bytes())
    assert len(recovery["receipt_sha256"]) == 18
    for i, expected in enumerate(recovery["receipt_sha256"]):
        assert sha((PRIOR_R23G / f"{i:02d}.json").read_bytes()) == expected
    for name, (receipt_sha, git_blob) in MASKS.items():
        raw = (PREP / (name + ".json")).read_bytes()
        assert sha(raw) == receipt_sha and blob(raw) == git_blob
    for index in range(8):
        receipt = PREP / f"{index:02d}.json"
        assert receipt.is_file()
        data = json.loads(receipt.read_bytes())
        assert data["status"] == "R23H_FACTORIAL2_PREFILL_CAPTURED_NOT_ADMISSION"
        assert data["index"] == index and data["plan_sha256"] == PLAN_SHA

    files = {}
    add(files, "batch/", [(x, PREP / x) for x in ("plan.json", "report.json", *[f"{i:02d}.json" for i in range(8)])])
    add(files, "batch/masks/", [(name + ".json", PREP / (name + ".json")) for name in MASKS])
    add(files, "run/", [(x, RUN / x) for x in (
        "factorial2.py", "primary-replay.json", "layer-summary.json",
        "preflight-v3.log", "preflight-v3.stderr", "preflight-v3.exit-code",
        "execute-v1.log", "execute-v1.stderr", "execute-v1.exit-code",
    )])
    add(files, "primary/", [("report.json", PRIMARY / "report.json")])
    add(files, "prior-factorial/", [(x, OLD_FACTORIAL / x) for x in ("plan.json", "report.json")])
    add(files, "pareto/", [(x, SD / x) for x in ("plan.json", "report.json", *[f"{i:02d}.json" for i in range(8)])])
    add(files, "pareto-run/", [(x, D / x) for x in ("run.log", "run.stderr", "exit-code", "original.json", "driver.py", "collector.py")])
    add(files, "recovery/", [(x, USB / x) for x in ("recovery-receipt.json", "expanded.json", "candidate.json")])
    for i in range(18):
        add(files, "prior-r23g/", [(f"{i:02d}.json", PRIOR_R23G / f"{i:02d}.json")])
    add(files, "corpus/", [("corpus.jsonl", CORPUS)])

    dest = pathlib.Path("/tmp/r23h-factorial2-prefill-v1.tgz")
    with tarfile.open(dest, "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(dest, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(files)
        assert {item.name for item in members} == set(files)
    raw = dest.read_bytes()
    print("STATUS=R23H_FACTORIAL2_PREFILL_EVIDENCE_PACKED")
    print("ARCHIVE_SHA256=" + sha(raw))
    print("REPORT_SHA256=" + sha(report_raw))
    print("PLAN_SHA256=" + sha(plan_raw))
    print("FACTORIAL2_BLOB=" + blob((RUN / "factorial2.py").read_bytes()))
    print("FILES=" + str(len(files)))
    print("BYTES=" + str(len(raw)))
    print("ELIGIBLE=" + json.dumps(report["eligible_for_primary_fit_only"], separators=(",", ":")))
    print("SELECTED=" + str(report["selected"]))


if __name__ == "__main__":
    main()
