import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mtpruning.engine import Checkpoint
from mtpruning.identity import sha256_file
from mtpruning.pipeline import (binding_path, committed_samples, frozen_dir, identity_of, journal_path,
                                read_json)
from mtpruning.reviewed import trace

records = Path(sys.argv[1])
corpus = Path(sys.argv[2])
model = Path(sys.argv[3])
expected_head = sys.argv[4]
receipt_path = Path(sys.argv[5])
binding = read_json(binding_path(records))
full = identity_of(binding)
_, contexts = frozen_dir(corpus.parent)
checkpoint = Checkpoint(model)
journal = trace.Journal(journal_path(records), binding)
try:
    samples = committed_samples(journal, contexts, full, checkpoint, 640, audit_required=False)
finally:
    journal.close()
assert len(samples) == 1, f"expected one committed smoke sample, found {len(samples)}"
sample = samples[0]
first = next(row for row in contexts.values() if row["split"] == "calibration")
assert sample["id"] == first["id"] == "calibration:zh-my:44"
assert sample["prompt_tokens"] == len(first["prompt_ids"]) == 74
assert sample["new_tokens"] == len(sample["tokens"]) > 0, "real decode produced no tokens"
assert sample["finish"] in ("eos", "length")
assert isinstance(sample["text"], str)
detail = binding["detail"]
protocol = detail["protocol"]
assert detail["code"]["git_head"] == expected_head
assert detail["code"]["files"]["mtpruning/engine.py"] == sha256_file("mtpruning/engine.py")
assert detail["mode"] == protocol["mode"] == "real"
assert detail["runtime"]["threads"] == 16
assert detail["max_new_tokens"] == 640
assert protocol["splits"] == ["calibration", "validation"]
assert protocol["limit"] is None
assert protocol["collects_routing"] is True
assert protocol["stats_splits"] == ["calibration"]
assert len(protocol["traced_ids"]) == 48
sparse_layers = set(checkpoint.sparse_layers)
assert len(sparse_layers) == 47
routing_summary = {}
for phase in ("prefill", "decode"):
    cells = sample["routing"][phase]
    expected_tokens = sample["prompt_tokens"] if phase == "prefill" else sample["new_tokens"]
    assert cells["tokens"] == expected_tokens
    assert set(cells["layer"]) == sparse_layers
    routing_summary[phase] = {
        "tokens": cells["tokens"],
        "sparse_layers_covered": len(set(cells["layer"])),
        "nonzero_expert_cells": len(cells["layer"]),
        "selection_count_sum": sum(cells["count"]),
        "expected_selection_count": expected_tokens * len(sparse_layers) * checkpoint.config["num_experts_per_tok"],
    }
    assert routing_summary[phase]["selection_count_sum"] == routing_summary[phase]["expected_selection_count"]
timings = [json.loads(line) for line in Path(str(records) + ".timings.jsonl").read_text().splitlines() if line]
assert len(timings) == 1 and timings[0]["id"] == sample["id"]
assert timings[0]["prefill_seconds"] > 0 and timings[0]["decode_seconds"] > 0
config = checkpoint.config
receipt = {
    "status": "PASS",
    "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "smoke_design": "full baseline identity; existing test crash hook stopped after the first committed sample",
    "expected_smoke_driver_exit": 1,
    "expected_stop": "SimulatedCrash: stopped after 1 commits, as the test asked",
    "source_head": expected_head,
    "identity": {name: binding[name] for name in ("teacher", "candidate", "corpus", "prompt", "runtime", "code",
                                                  "protocol", "decoding")},
    "protocol": protocol,
    "max_new_tokens": detail["max_new_tokens"],
    "model_dimensions": {
        "hidden_size": config["hidden_size"],
        "attention_width": config["num_attention_heads"] * config["head_dim"],
        "kv_width": config["num_key_value_heads"] * config["head_dim"],
        "num_hidden_layers": config["num_hidden_layers"],
        "sparse_layers": len(sparse_layers),
        "experts": checkpoint.num_experts,
        "top_k": checkpoint.config["num_experts_per_tok"],
    },
    "sample": {
        "id": sample["id"],
        "direction": sample["direction"],
        "prompt_tokens": sample["prompt_tokens"],
        "new_tokens": sample["new_tokens"],
        "finish": sample["finish"],
        "tokens": sample["tokens"],
        "text": sample["text"],
    },
    "routing": routing_summary,
    "timing": timings[0],
    "journal": {"path": str(journal_path(records)), "sha256": sha256_file(journal_path(records))},
    "timings_sha256": sha256_file(str(records) + ".timings.jsonl"),
    "real_reference_runtime_comparison": "NOT_RUN: the whole reference model was not loaded concurrently with the teacher engine",
}
receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"status": receipt["status"], "sample": receipt["sample"], "routing": routing_summary,
                  "timing": timings[0]}, ensure_ascii=False))
