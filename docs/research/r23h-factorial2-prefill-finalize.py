#!/usr/bin/env python3
"""Finalize R23H factorial2 receipts after a post-capture aggregation bug.

This does not rerun the model. It validates all eight immutable receipts against
the frozen plan, source controls, prior R23G receipts, and the original capture
event log, then independently recomputes the aggregate prefill gate.
"""
import importlib.util
import json
import math
import pathlib
import sys

R = pathlib.Path.home() / ".cache/mt-pruning"
PREP = R / "r23-results/r23h-factorial2-prefill-v1"
RUN = R / "r23h-factorial2-prefill-run-v1"
RUNNER = RUN / "factorial2.py"


def need(value, message):
    if not value:
        raise ValueError(message)


def load_runner():
    spec = importlib.util.spec_from_file_location("factorial2_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    runner = load_runner()
    plan, rows, context, prior, old_screen, masks = runner.validate_sources()

    log_raw = (RUN / "execute.log").read_bytes()
    log_events = [runner.strict(line) for line in log_raw.splitlines() if line.startswith(b"{")]
    captures = [item for item in log_events if item.get("status") == "R23H_FACTORIAL2_PREFILL_CAPTURED"]
    need(len(captures) == 8, "all eight original capture events required")
    need([item["index"] for item in captures] == list(range(8)), "capture event index order")
    need([item["id"] for item in captures] == [row["id"] for row in rows], "capture event row order")
    need(any(item.get("status") == "R23H_FACTORIAL2_TEACHER_READY" for item in log_events), "teacher ready event")
    need(not any(item.get("status") == "R23H_FACTORIAL2_PREFILL_COMPLETE_NOT_FRESH_ADMISSION" for item in log_events), "unexpected original terminal event")

    capture_exit = (RUN / "exit-code").read_text().strip()
    need(capture_exit == "1", "capture run exit code must preserve the known post-capture failure")
    need(b"NameError: name 'candidates' is not defined" in log_raw, "expected post-capture aggregate NameError")
    retry_exit = (RUN / "execute-v1.exit-code").read_text().strip()
    need(retry_exit == "1", "duplicate retry failure evidence")
    retry_stderr = (RUN / "execute-v1.stderr").read_bytes()
    need(b"FileExistsError" in retry_stderr, "duplicate retry must be a no-overwrite collision")

    receipts = []
    names = list(masks)
    for index, row in enumerate(rows):
        raw = (PREP / f"{index:02d}.json").read_bytes()
        receipt_sha = runner.sha(raw)
        need(captures[index]["receipt_sha256"] == receipt_sha, "capture event receipt hash " + str(index))
        receipt = runner.strict(raw)
        need(
            receipt["status"] == "R23H_FACTORIAL2_PREFILL_CAPTURED_NOT_ADMISSION"
            and receipt["index"] == index
            and receipt["id"] == row["id"]
            and receipt["plan_sha256"] == runner.PLAN_SHA,
            "receipt identity " + str(index),
        )
        old = prior[row["id"]]
        need(receipt["teacher_hash"] == old["original96"]["teacher_hashes"][0]["sha256"], "teacher hash " + row["id"])
        need(set(receipt["metrics"]) == set(names), "metric inventory " + row["id"])
        for name, mask in masks.items():
            metric = receipt["metrics"][name]
            need(
                type(metric["kl"]) in (int, float)
                and math.isfinite(metric["kl"])
                and metric["kl"] >= 0
                and type(metric["nmse"]) in (int, float)
                and math.isfinite(metric["nmse"])
                and metric["nmse"] >= 0
                and type(metric["argmax_mismatch"]) is int
                and metric["argmax_mismatch"] in (0, 1)
                and len(metric["logits_sha256"]) == 64
                and metric["mask_sha256"] == runner.sha(runner.canon(mask)),
                "metric validity " + name + "/" + row["id"],
            )
        for name in ("original96", "all8", "all10"):
            need(receipt["metrics"][name] == old_screen[row["id"]]["metrics"][name], "historical control replay " + name + "/" + row["id"])
        old_metric = old["candidate96"]["logits"][0]
        old_hash = old["candidate96"]["masked_hashes"][0]["sha256"]
        actual = receipt["metrics"]["r23g"]
        need(
            actual["logits_sha256"] == old_hash
            and abs(actual["kl"] - old_metric["teacher_to_masked_kl"]) < 1e-12
            and abs(actual["nmse"] - old_metric["nmse"]) < 1e-12
            and actual["argmax_mismatch"] == int(old_metric["teacher_argmax"] != old_metric["masked_argmax"]),
            "R23G historical replay " + row["id"],
        )
        receipts.append(receipt)

    aggregates = {
        name: {
            "kl": math.fsum(item["metrics"][name]["kl"] for item in receipts) / 8,
            "nmse": math.fsum(item["metrics"][name]["nmse"] for item in receipts) / 8,
            "argmax_mismatch": sum(item["metrics"][name]["argmax_mismatch"] for item in receipts),
        }
        for name in names
    }
    candidate_names = list(runner.MASKS)
    eligible = [
        name for name in candidate_names
        if all(aggregates[name][key] <= aggregates["original96"][key] + 1e-12 for key in aggregates[name])
        and aggregates[name]["kl"] < aggregates["r23g"]["kl"]
    ]
    eligible.sort(key=lambda name: (aggregates[name]["kl"], aggregates[name]["nmse"], name))

    report = {
        "status": "R23H_FACTORIAL2_PREFILL_CAPTURE_COMPLETE_AGGREGATION_RECOVERED_NOT_ADMISSION",
        "plan_sha256": runner.PLAN_SHA,
        "teacher_sha256": runner.TEACHER,
        "samples": 8,
        "metrics": aggregates,
        "eligible_for_primary_fit_only": eligible,
        "selected": eligible[0] if eligible else None,
        "capture_execution": {
            "exit_code": 1,
            "capture_events": 8,
            "all_receipts_published": True,
            "log_sha256": runner.sha(log_raw),
            "failure": "NameError: candidates undefined after all eight immutable receipts and capture events were written",
        },
        "duplicate_retry": {
            "exit_code": 1,
            "failure": "FileExistsError while refusing to overwrite the existing receipt set",
            "evidence": "execute-v1.stderr",
        },
        "finalization": {
            "status": "R23H_FACTORIAL2_RECEIPTS_VALIDATED_AND_AGGREGATED",
            "method": "stdlib receipt reconciliation; no model rerun",
        },
        "limits": "Eight frozen zh-my calibration Fit Prefill receipts only. The original capture process exited nonzero after all receipts because of an aggregation NameError. This recovery finalizes those immutable receipts; it is not a clean execution PASS, Decode/primary propagation, Fresh, Holdout, free-running translation, product admission, or physical model build.",
    }
    report_sha = runner.publish(PREP / "report.json", report)
    runner.event(
        "R23H_FACTORIAL2_PREFILL_RECOVERY_FINALIZED",
        report_sha256=report_sha,
        capture_log_sha256=runner.sha(log_raw),
        report=report,
    )


if __name__ == "__main__":
    main()
