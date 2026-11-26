import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
pruned = Path(sys.argv[2])
smoke_path = Path(sys.argv[3])


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generation(records):
    records = Path(records)
    summary = load(str(records) + ".summary.json")
    timings = [json.loads(line) for line in Path(str(records) + ".timings.jsonl").read_text().splitlines() if line]
    prefill = sum(item["prefill_seconds"] for item in timings)
    decode = sum(item["decode_seconds"] for item in timings)
    return {
        "sentences": summary["sentences"],
        "per_split_direction": summary["per_split_direction"],
        "finish": summary["finish"],
        "new_tokens": summary["new_tokens"],
        "traced": summary["traced"],
        "seconds_this_invocation": summary["seconds_this_invocation"],
        "prefill_seconds_sum": prefill,
        "decode_seconds_sum": decode,
        "generated_tokens_per_decode_second": summary["new_tokens"] / decode if decode else None,
        "records_sha256": sha256(records),
        "journal_sha256": summary["journal"]["sha256"],
    }


def equivalence(path):
    data = load(path)
    summary = {key: data.get(key) for key in (
        "pass", "sentences", "identical_text", "all_argmax_agree", "all_route_ids_agree", "within_tol",
        "bitwise_identical_logits", "route_weights_bitwise_equal", "route_weights_within_tol", "length_match",
        "positions_compared", "logits_max_abs_diff", "route_id_mismatches", "route_weight_max_abs_diff")}
    summary["masked_witness_changed"] = data.get("binding", {}).get("masked_witness_changed")
    return summary


def prune(path):
    data = load(path)
    return {
        "verify_pass": data["verify"]["pass"],
        "tensor_bytes": data["manifest"]["tensor_bytes"],
        "tensors_checked": data["verify"]["tensors_checked"],
        "mismatched_tensors": data["verify"]["mismatched_tensors"],
        "identity_mapping": data["verify"]["identity_mapping"],
        "num_experts": data["manifest"]["num_experts"],
        "source_num_experts": data["manifest"]["source_num_experts"],
        "shards": len(data["manifest"]["files"]),
        "candidate_digest": data["candidate_digest"],
    }


result = {
    "baseline": generation(run / "baseline/records.jsonl"),
    "baseline_stats": load(run / "baseline/stats.json"),
    "selection_retained_mass": {keep: load(run / "selection.json")["candidates"][keep]["retained_mass_overall"]
                                for keep in ("96", "64")},
    "selection_retained_mass_min_layer": {keep: load(run / "selection.json")["candidates"][keep]["retained_mass_min_layer"]
                                          for keep in ("96", "64")},
    "k128": {
        "prune": prune(run / "k128/prune.json"),
        "mechanism": generation(run / "k128/mechanism/records.jsonl"),
        "equivalence": equivalence(run / "k128/equiv.json"),
    },
    "k96": {
        "prune": prune(run / "k96/prune.json"),
        "masked": generation(run / "k96/masked/records.jsonl"),
        "mechanism": generation(run / "k96/mechanism/records.jsonl"),
        "equivalence": equivalence(run / "k96/equiv.json"),
        "physical_checkpoint_bytes": sum(path.stat().st_size for path in (pruned / "k96").rglob("*") if path.is_file()),
        "full_validation": "NOT_RUN",
        "stock_check": "NOT_RUN",
    },
    "k64": {"status": "NOT_RUN"},
    "decision": "NOT_RUN",
    "canonical_report": "NOT_RUN",
    "final_test": "HARD_DISABLED_NOT_RUN",
    "smoke": load(smoke_path),
}
print(json.dumps(result, indent=2, sort_keys=True))
