"""D01b pipeline, one command per stage of a real experiment:

  corpus-check   the corpus on disk is the frozen one, byte for byte
  run            translate corpus sentences with one checkpoint into a journal
  select         expert sets from the calibration routing, coverage-checked
  prune          physical checkpoint, then a tensor-level check of the reload
  equiv          identity no-op and masked equivalence, position by position
  stock-check    the unmodified reference runtime on the pruned checkpoint
  decide         provisional candidate, from validation records only
  report

A ``run`` commits each sample (output, routing statistics, mask audit, trace
record) to a SQLite journal together with the run identity, and every aggregate
is rebuilt from the committed samples. So an interrupted run resumes without
losing anything and a finished run is never overwritten.

There is deliberately no final-test command in this build (R11-P1-05): the test
split is not exposed to any model before a separately authorized execution
design exists. ``decide`` keeps ``test_execution_authorized`` false.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .calibration import SampleAudit, SampleRouting, mask_audit_report, rebuild, save_stats, select, validate_payload
from .corpus import DIRECTIONS, SPLITS, build_corpus
from .engine import Checkpoint, Engine
from .evaluate import chrf_pp, compare_runs, save_trace, trace_complete
from .frozen import BlockedCorpus, load_frozen_corpus
from .identity import FIELDS, TOKENIZER_FILES, checkpoint_identity, digest, key, run_identity, sha256_file
from .prune import verify_pruned, write_pruned
from .reviewed import bridge, handoff, trace
from .spec import CORPUS_MANIFEST_SHA256, DECODING, MAX_CHRF_DROP, PLAN

# Test hook only: stop after this many commits of one invocation, to simulate a crash.
CRASH_AFTER = int(os.environ.get("MTPRUNING_TEST_CRASH_AFTER", "0"))
# Identity fields the baseline and every candidate must share; "decoding" covers decoding and
# max tokens; "mode" (R13-P1-02) keeps a rehearsal run from ever being mixed into a real one.
SHARED_FIELDS = ("teacher", "corpus", "prompt", "runtime", "code", "decoding", "mode")
# R11-P1-05: every final-test execution path is removed in this pre-real-run build. No
# review reference or decision hash can re-enable them; the exact external authorization
# (decision, candidate, corpus and protocol bound, atomic one-candidate ledger) is a
# separate later task, after a reviewed real validation decision exists.
FINAL_TEST_DISABLED = ("FINAL_TEST_DISABLED (R12): the test split is not executable in this pre-real-run "
                       "build. External, decision/candidate/corpus/protocol-bound authorization is deferred "
                       "to a later task and stays test_execution_authorized=false.")


class SimulatedCrash(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment() -> dict:
    import sacrebleu
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "sacrebleu": sacrebleu.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}


def write_text_atomic(path: Path, body: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body, encoding="utf-8", newline="\n")
    temporary.replace(path)


def dump(path: Path, data) -> None:
    write_text_atomic(Path(path), json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def corpus_paths(run_dir: Path) -> tuple[Path, Path]:
    """(manifest, corpus) of a run directory."""
    run_dir = Path(run_dir)
    return run_dir / "corpus" / "corpus_manifest.json", run_dir / "corpus" / "corpus.jsonl"


def frozen_dir(corpus_dir: Path, manifest_sha256=None) -> tuple[dict, dict]:
    """The frozen corpus of a corpus directory: (manifest, records by ID). Any violation stops the stage."""
    try:
        return load_frozen_corpus(corpus_dir, manifest_sha256 or CORPUS_MANIFEST_SHA256)
    except BlockedCorpus as error:
        raise SystemExit(f"{error.code}: {error}") from error


def frozen(run_dir: Path, manifest_sha256=None) -> tuple[dict, dict]:
    return frozen_dir(Path(run_dir) / "corpus", manifest_sha256)


def read_run(path: Path) -> tuple[dict | None, list[dict]]:
    """The run header and the sentence records of an exported records file."""
    if not Path(path).exists():
        return None, []
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or rows[0].get("type") != "run":
        raise ValueError(f"{path} does not start with a run header")
    return rows[0], [row for row in rows[1:] if row.get("type") == "sentence"]


def journal_path(records) -> Path:
    return Path(str(records) + ".sqlite")


def binding_path(records) -> Path:
    return Path(str(records) + ".identity.json")


def binding_of(identity: dict) -> dict:
    """The part of a run identity that is the same for every split."""
    return {name: value for name, value in identity.items() if name != "sample_keys"}


def identity_of(binding: dict) -> dict:
    return {**binding, "sample_keys": {split: key(binding, split) for split in SPLITS}}


def committed_samples(journal, contexts: dict, identity: dict, checkpoint: Checkpoint, max_new_tokens: int,
                      audit_required: bool | None = None) -> list[dict]:
    """Every committed sample of ``journal`` in ID order, each checked to be the corpus record it names."""
    top_k = int(checkpoint.config["num_experts_per_tok"])
    samples = []
    for sample_id, payload in journal.records():        # verifies every payload against its stored digest
        if sample_id not in contexts:
            raise ValueError(f"the journal holds {sample_id}, which is not in the frozen corpus")
        sample = {"id": sample_id, **payload}
        if payload.get("identity_key") != identity["sample_keys"][contexts[sample_id]["split"]]:
            raise ValueError(f"{sample_id} was committed under another run identity")
        validate_payload(sample, contexts[sample_id], checkpoint.sparse_layers, checkpoint.num_experts, top_k,
                         max_new_tokens, audit_required)
        samples.append(sample)
    return samples


def run_provenance(records_path: Path, contexts: dict, expected_traced: list[str] | None = None,
                   traces_dir: Path | None = None, checkpoint: Checkpoint | None = None,
                   audit_required: bool | None = None) -> dict:
    """Replay a run's journal against its exported records, and its traces against the journal.

    The records export must be exactly what the journal replays to - every row checked
    against its context (split, direction, line) and the journal payload it came from,
    no duplicates, every identity key matching - the journal file must be the one the
    summary recorded, the exported header (including its identity keys and mode) must be
    the journal's own, and - with ``expected_traced`` - exactly those sentences may carry
    traces, each recomputed digest matching the record committed with its sample and the
    position count matching prompt + generated tokens (R13-P1-01/P1-03). Every payload is
    checked against its stored digest and the run's sample keys even without
    ``checkpoint``; with it, every payload is also validated in full, and with
    ``audit_required`` the mask-audit presence rule is enforced and the accumulated
    changed-decision count of the journal (not of any editable summary) is returned. A
    checkpoint is only available while its bytes are on disk - the driver deletes a pruned
    checkpoint after its receipts, which is why later stages replay without one.
    """
    records_path = Path(records_path)
    header, rows = read_run(records_path)
    if header is None:
        raise SystemExit(f"{records_path} has no run header")
    journal_file = journal_path(records_path)
    if not journal_file.exists():
        raise SystemExit(f"{records_path}: the run journal is missing")
    summary = read_json(Path(str(records_path) + ".summary.json"))
    if summary.get("journal", {}).get("sha256") != sha256_file(journal_file):
        raise SystemExit(f"{records_path}: the journal changed after the records were exported")
    binding = read_json(binding_path(records_path))
    differ = [name for name in (*FIELDS, "decoding") if header["identity"].get(name) != binding.get(name)]
    if header["identity"].get("mode") != binding.get("detail", {}).get("mode"):
        differ.append("mode")
    if differ:
        raise SystemExit(f"{records_path}: the exported header does not match the journal's identity: "
                         f"{', '.join(sorted(set(differ)))}")
    full = identity_of(binding)
    if header.get("identity_keys") != full["sample_keys"]:
        raise SystemExit(f"{records_path}: the exported identity keys are not the journal's")
    journal = trace.Journal(journal_file, binding)
    samples, audit_changed = [], 0
    for sample_id, payload in journal.records():        # verifies every payload against its stored digest
        if sample_id not in contexts:
            raise SystemExit(f"the journal holds {sample_id}, which is not in the frozen corpus")
        if payload.get("identity_key") != full["sample_keys"][contexts[sample_id]["split"]]:
            raise SystemExit(f"{sample_id} was committed under another run identity")
        sample = {"id": sample_id, **payload}
        if checkpoint is not None:
            validate_payload(sample, contexts[sample_id], checkpoint.sparse_layers, checkpoint.num_experts,
                             int(checkpoint.config["num_experts_per_tok"]), header["max_new_tokens"],
                             audit_required)
        elif audit_required is not None and (sample.get("audit") is not None) is not audit_required:
            raise SystemExit(f"{sample_id}: mask audit presence does not match the run kind")
        if audit_required and sample.get("audit") is not None:
            audit_changed += sum(entry["changed"] for entry in sample["audit"].values())
        samples.append(sample)
    journal.close()
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"{records_path}: duplicate exported rows")
    if set(by_id) != {sample["id"] for sample in samples} or any(
            any(by_id[sample["id"]][name] != sample[name]
                for name in ("split", "direction", "line", "prompt_tokens", "new_tokens", "finish", "tokens",
                             "text", "identity_key"))
            for sample in samples):
        raise SystemExit(f"{records_path}: the exported records differ from what the journal holds")
    traced = sorted(row["id"] for row in rows if row.get("traced"))
    if expected_traced is not None and traced != sorted(expected_traced):
        raise SystemExit(f"{records_path}: traced sentences {len(traced)} do not cover exactly the expected "
                         f"{len(expected_traced)}")
    if traces_dir is not None:
        trace_digests = {}
        for sample in samples:
            if sample["id"] in (expected_traced or traced):
                expected_positions = sample["prompt_tokens"] + sample["new_tokens"]
                if not sample.get("trace") or not trace_complete(traces_dir, sample["id"], sample["trace"],
                                                                 expected_positions):
                    raise SystemExit(f"{sample['id']}: the trace on disk does not match the journal's record")
                trace_digests[sample["id"]] = sample["trace"]["sha256"]
    return {"records_sha256": sha256_file(records_path), "journal_sha256": sha256_file(journal_file),
            "teacher": header["identity"]["teacher"], "corpus": header["identity"]["corpus"],
            "code": header["identity"]["code"], "candidate": header["identity"]["candidate"],
            "mode": header["identity"].get("mode"), "traced": traced, "audit_changed": audit_changed,
            **({"trace_digests": trace_digests} if traces_dir is not None else {})}


# The identity projection of a run provenance that receipts bind: stable across replay
# options (witness audit counting, trace verification), comparable byte for byte.
RECEIPT_FIELDS = ("records_sha256", "journal_sha256", "teacher", "corpus", "code", "candidate", "mode", "traced")


def receipt_binding(provenance: dict) -> dict:
    return {name: provenance[name] for name in RECEIPT_FIELDS}


def first_per_direction(contexts: dict, split: str, count: int) -> list[str]:
    """The first ``count`` IDs of ``split`` per direction, in corpus order."""
    seen, ids = Counter(), []
    for sample_id, record in contexts.items():
        if record["split"] == split and seen[record["direction"]] < count:
            ids.append(sample_id)
            seen[record["direction"]] += 1
    if len(ids) != count * len(DIRECTIONS):
        raise SystemExit(f"the frozen corpus has fewer than {count} {split} sentences in some direction")
    return ids


@dataclass
class RunOptions:
    model: str
    corpus: str
    out: str
    splits: list
    max_new_tokens: int
    threads: int
    tokenizer: str | None = None
    mask: str | None = None          # selection JSON: deny the experts this candidate prunes
    mask_keep: int | None = None
    stats: str | None = None         # base path of the calibration routing statistics
    trace_dir: str | None = None
    trace_n: int = 0
    trace_split: str = "validation"    # held-out sentences: where pruning actually changes routing
    limit: int | None = None
    manifest_sha256: str | None = None
    cache_dir: str | None = None
    mode: str = "rehearsal"          # "real": pinned teacher only, no overrides (P2-02)


def execute_run(options: RunOptions) -> dict:
    """Translate, commit each sample, rebuild the aggregates, export the records. Resumable."""
    from transformers import AutoTokenizer
    splits = list(options.splits)
    if "test" in splits:
        raise SystemExit(FINAL_TEST_DISABLED)          # no path may expose the test split (R11-P1-05)
    if options.mode == "real":
        # P2-02: real-mode boundaries. A real run decodes with the checkpoint's own tokenizer,
        # covers every sentence of its splits (bounded witness runs are validation-only), and
        # runs the frozen corpus manifest only.
        if options.tokenizer is not None:
            raise SystemExit("real mode refuses a tokenizer override: the checkpoint's own tokenizer is used")
        if options.limit is not None and "calibration" in splits:
            raise SystemExit("real mode refuses --limit: a calibration run covers every calibration sentence")
        if options.manifest_sha256 not in (None, CORPUS_MANIFEST_SHA256):
            raise SystemExit("real mode runs the frozen corpus manifest only")
    manifest, contexts = frozen_dir(Path(options.corpus).parent, options.manifest_sha256)
    todo = [record for record in contexts.values() if record["split"] in splits]
    if options.limit is not None:
        seen, limited = Counter(), []
        for record in todo:
            if seen[record["split"], record["direction"]] < options.limit:
                limited.append(record)
            seen[record["split"], record["direction"]] += 1
        todo = limited
    checkpoint = Checkpoint(options.model)
    engine = Engine(checkpoint, threads=options.threads)
    mask, selection_sha256 = None, None
    if options.mask:
        mask, selection_sha256 = load_selection_mask(options.mask, options.mask_keep)
        engine.set_mask(mask)
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer or options.model)
    traced, seen = set(), Counter()
    if options.trace_dir and options.trace_n:
        for record in todo:
            if record["split"] == options.trace_split and seen[record["direction"]] < options.trace_n:
                traced.add(record["id"])
                seen[record["direction"]] += 1
    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not journal_path(out).exists()             # qualification hashes the actual bytes; a resume
    identity = run_identity(                           # may trust the content-keyed cache (R11-P1-02)
        model_dir=options.model, corpus_path=options.corpus,
        manifest_path=Path(options.corpus).parent / "corpus_manifest.json", cache_dir=options.cache_dir, mask=mask,
        selection_sha256=selection_sha256, decoding=DECODING, max_new_tokens=options.max_new_tokens,
        protocol={"mode": options.mode, "splits": splits, "limit": options.limit, "traced_ids": sorted(traced),
                  "collects_routing": bool(options.stats), "stats_splits": ["calibration"] if options.stats else []},
        threads=options.threads, tokenizer_dir=options.tokenizer, mode=options.mode, fresh=fresh)
    current = binding_of(identity)
    if fresh:                                          # nothing committed yet: this run defines the identity
        dump(binding_path(out), current)
    binding = read_json(binding_path(out))
    # A journal continues only under the identity it was started with: checkpoint content, corpus,
    # prompt and tokenizer, mask, runtime, code, decoding settings and protocol.
    differ = sorted(name for name in set(current) | set(binding) if current.get(name) != binding.get(name))
    if differ:
        raise ValueError(f"identity mismatch, the journal was started with another {', '.join(differ)}")
    journal = trace.Journal(journal_path(out), binding)   # and the stored identity must match the database
    full = identity_of(binding)
    top_k, depth = engine.top_k, len(checkpoint.sparse_layers)
    audit_required = mask is not None
    done = {sample["id"]: sample for sample in committed_samples(journal, contexts, full, checkpoint,
                                                                 options.max_new_tokens, audit_required)}
    started, commits = time.perf_counter(), 0
    with open(Path(str(out) + ".timings.jsonl"), "a", encoding="utf-8", newline="\n") as timing_log:
        for record in todo:
            committed = done.get(record["id"])
            if committed is not None and record["id"] in traced:
                # A committed witness sentence must still have exactly the trace its payload binds:
                # missing, corrupt or stale files are regenerated identically, not trusted (R11-P1-03).
                expected = committed.get("trace") or {"sha256": "", "positions": -1}
                if trace_complete(options.trace_dir, record["id"], expected):
                    continue
                committed = None                       # fall through: regenerate with the same outcome
            elif committed is not None:
                continue
            routing, audit = SampleRouting(engine), SampleAudit() if mask is not None else None
            engine.on_route, engine.audit = routing.hook, audit.hook if audit is not None else None
            generated = engine.generate(record["prompt_ids"], options.max_new_tokens, trace=record["id"] in traced)
            cells = routing.payload()
            trace_info = save_trace(options.trace_dir, record["id"], generated.trace) if record["id"] in traced else None
            # The trace covers exactly the prompt positions and one position per generated token;
            # any other count means the engine changed underneath the protocol (R13-P1-01).
            if trace_info is not None and trace_info["positions"] != generated.prompt_tokens + len(generated.tokens):
                raise RuntimeError("the trace position count does not match the generation record "
                                   "(prompt + generated tokens)")
            payload = {
                "identity_key": full["sample_keys"][record["split"]], "split": record["split"],
                "direction": record["direction"], "line": record["line"],
                "prompt_tokens": generated.prompt_tokens, "new_tokens": len(generated.tokens),
                "finish": generated.finish, "tokens": generated.tokens,
                "text": tokenizer.decode(generated.tokens, skip_special_tokens=True),
                "routing": cells, "audit": audit.payload(cells, depth) if audit is not None else None,
                "trace": trace_info,
            }
            journal.put(record["id"], payload)        # 'conflicting sample' if a committed copy differs
            timing_log.write(json.dumps({"id": record["id"], "at": utc_now(),
                                         "prefill_seconds": round(generated.prefill_seconds, 3),
                                         "decode_seconds": round(generated.decode_seconds, 3)}) + "\n")
            timing_log.flush()
            commits += 1
            if CRASH_AFTER and commits >= CRASH_AFTER:
                raise SimulatedCrash(f"stopped after {commits} commits, as the test asked")
    samples = committed_samples(journal, contexts, full, checkpoint, options.max_new_tokens, audit_required)
    summary = {
        "model": str(Path(options.model).resolve()), "num_experts": checkpoint.num_experts,
        "masked_keep": options.mask_keep if mask is not None else None, "splits": splits,
        "max_new_tokens": options.max_new_tokens, "decoding": DECODING, "exported": utc_now(),
        "identity": {**{name: binding[name] for name in (*FIELDS, "decoding")},
                     "mode": binding["detail"]["mode"]},
        "identity_keys": full["sample_keys"], "checkpoint_kind": binding["detail"]["checkpoint"]["kind"],
        "synthetic": binding["detail"]["checkpoint"]["kind"] == "synthetic", "mode": binding["detail"]["mode"],
        "environment": environment(), "commits_this_invocation": commits,
        "seconds_this_invocation": round(time.perf_counter() - started, 1), "sentences": len(samples),
        "per_split_direction": {f"{split}:{direction}": n for (split, direction), n in sorted(
            Counter((s["split"], s["direction"]) for s in samples).items())},
        "finish": dict(Counter(s["finish"] for s in samples)),
        "new_tokens": sum(s["new_tokens"] for s in samples),
        "traced": sum(bool(options.trace_dir)
                      and trace_complete(options.trace_dir, s["id"], s.get("trace")) for s in samples),
        "journal": {"path": str(journal_path(out)), "sha256": sha256_file(journal_path(out))},
        "routing": None, "mask_audit": None,
    }
    if samples:
        rebuilt = rebuild(samples, checkpoint.sparse_layers, checkpoint.num_experts, top_k,
                          options.max_new_tokens, contexts, audit_required=audit_required)
        if options.stats:
            calibration = [sample for sample in samples if sample["split"] == "calibration"]
            summary["routing"] = save_stats(options.stats, rebuild(calibration, checkpoint.sparse_layers,
                                                                   checkpoint.num_experts, top_k,
                                                                   options.max_new_tokens, contexts,
                                                                   audit_required=audit_required),
                                            checkpoint.sparse_layers)
        if mask is not None:
            summary["mask_audit"] = mask_audit_report(rebuilt)
    # No timestamp in the header: the export of an unchanged journal is byte for byte the same.
    header = {name: summary[name] for name in ("model", "num_experts", "masked_keep", "splits", "max_new_tokens",
                                                "decoding", "identity", "identity_keys", "checkpoint_kind",
                                                "synthetic", "mode")}
    header.update(type="run", corpus_sha256=manifest["corpus_sha256"])
    body = [json.dumps(header, ensure_ascii=False)]
    for sample in samples:
        body.append(json.dumps({
            "type": "sentence", "id": sample["id"], "split": sample["split"], "direction": sample["direction"],
            "line": sample["line"], "prompt_tokens": sample["prompt_tokens"], "new_tokens": sample["new_tokens"],
            "finish": sample["finish"], "tokens": sample["tokens"], "text": sample["text"],
            "identity_key": sample["identity_key"],
            "traced": bool(options.trace_dir) and trace_complete(options.trace_dir, sample["id"], sample.get("trace")),
        }, ensure_ascii=False))
    write_text_atomic(out, "\n".join(body) + "\n")
    dump(Path(str(out) + ".summary.json"), summary)
    return summary


def load_selection_mask(path, keep: int) -> tuple[dict, str]:
    """(original expert id per sparse layer, sha256 of the selection file) of one candidate."""
    candidate = read_json(path)["candidates"][str(keep)]
    return {int(layer): ids for layer, ids in candidate["kept"].items()}, sha256_file(path)


def masked_candidate_digest(mask: dict, selection_sha256: str) -> str:
    return digest({"kind": "masked", "kept": {str(k): v for k, v in sorted(mask.items())},
                   "selection_sha256": selection_sha256})


def pruned_candidate_digest(manifest: dict) -> str:
    files = {f["name"]: f["sha256"] for f in manifest["files"]}
    files["model.safetensors.index.json"], files["config.json"] = manifest["index_sha256"], manifest["config_sha256"]
    return digest({"kind": "pruned", "num_experts": manifest["num_experts"], "kept": manifest["kept"], "files": files})


def prune_candidate(source_dir: str, selection_path: str | None, keep: int, out_dir: Path, record_path: Path,
                    cache_dir=None) -> dict:
    """Write the pruned checkpoint, verify the reload tensor by tensor, record both."""
    source = Checkpoint(source_dir)
    if keep == source.num_experts:
        kept = {layer: list(range(source.num_experts)) for layer in source.sparse_layers}
    else:
        kept = {int(layer): ids for layer, ids in read_json(selection_path)["candidates"][str(keep)]["kept"].items()}
    # Stage-boundary recheck: the trusted cache avoids rehashing unchanged shards; any byte
    # that changed still gets a fresh hash and refuses against the pinned listing or manifest.
    manifest = write_pruned(source, kept, out_dir, source_identity=checkpoint_identity(source_dir, cache_dir,
                                                                                      fresh=False))
    record = {"manifest": manifest, "verify": verify_pruned(source, Checkpoint(out_dir)),
              "candidate_digest": pruned_candidate_digest(manifest)}
    dump(record_path, record)
    if not record["verify"]["pass"]:
        raise SystemExit(f"the reloaded pruned checkpoint differs from its source: {record['verify']['first_mismatches']}")
    return record


def cmd_corpus(args) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = build_corpus(args.tarball, tokenizer, args.out, args.n_calibration, args.n_validation,
                            args.n_test, args.seed)
    print(json.dumps({k: manifest[k] for k in ("records", "corpus_sha256", "seed", "sizes_per_direction")}))


def cmd_corpus_check(args) -> None:
    manifest, contexts = frozen_dir(args.corpus_dir, args.manifest_sha256)
    receipt = {
        "status": "FROZEN_CORPUS_CONTRACT_OK",
        "manifest_sha256": sha256_file(Path(args.corpus_dir) / "corpus_manifest.json"),
        "corpus_sha256": manifest["corpus_sha256"], "records": len(contexts),
        "sizes_per_direction": manifest["sizes_per_direction"], "directions": manifest["directions"],
        "language_codes": manifest["language_codes"], "tarball_sha256": manifest["tarball_sha256"],
        "checks": ["manifest bytes hash to the pinned value", "corpus bytes hash to the manifest value",
                   "r7/handoff.corpus: exact IDs, sizes and line numbers; R8 duplicate rules on normalized text"],
        "checked": utc_now(),
    }
    dump(args.out, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "records", "corpus_sha256")}))


def cmd_run(args) -> None:
    options = {name: getattr(args, name) for name in RunOptions.__dataclass_fields__ if hasattr(args, name)}
    summary = execute_run(RunOptions(**options))
    print(json.dumps({k: summary[k] for k in ("sentences", "finish", "new_tokens", "commits_this_invocation")}))


def cmd_select(args) -> None:
    """Expert sets from the calibration routing, after the reviewed coverage check of the aggregate."""
    run = Path(args.run_dir)
    manifest, contexts = frozen(run, args.manifest_sha256)
    out = run / "baseline" / "records.jsonl"
    header, rows = read_run(out)
    if header is None:
        raise SystemExit("no baseline run to select from")
    # The teacher on disk must still be the one the journal bound: content, not just the
    # path. A changed checkpoint refuses selection instead of selecting on stale bytes.
    current = checkpoint_identity(header["model"], args.cache_dir, fresh=False)
    binding = read_json(binding_path(out))
    if current["teacher"] != binding["teacher"]:
        raise SystemExit("the baseline checkpoint on disk is no longer the teacher the journal recorded")
    checkpoint = Checkpoint(header["model"])
    top_k = int(checkpoint.config["num_experts_per_tok"])
    max_new_tokens = binding["detail"]["max_new_tokens"]
    journal = trace.Journal(journal_path(out), binding)
    samples = committed_samples(journal, contexts, identity_of(binding), checkpoint, max_new_tokens,
                                audit_required=False)
    # The selection is built from calibration routing only.
    stats = rebuild([sample for sample in samples if sample["split"] == "calibration"], checkpoint.sparse_layers,
                    checkpoint.num_experts, top_k, max_new_tokens, contexts, audit_required=False)
    base = run / "baseline" / "stats"
    if not (base.with_name(base.name + ".npz").exists() and base.with_name(base.name + ".json").exists()):
        raise SystemExit("the baseline stage has not exported its statistics (interrupted?): rerun it")
    arrays = {f"{name}_{phase}": stats[name][phase].numpy()
              for name in ("count", "mass", "tokens") for phase in ("prefill", "decode")}
    arrays["sentences"] = stats["sentences"].numpy()
    # What is on disk must be what the journal holds; a stale or edited aggregate is refused, never reused.
    with np.load(str(base) + ".npz", allow_pickle=False) as on_disk:
        if set(on_disk.files) != set(arrays) or any(not np.array_equal(on_disk[name], arrays[name]) for name in arrays):
            raise SystemExit("the exported statistics differ from what the journal holds: rerun the baseline stage")
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {sample["id"] for sample in samples} or any(
            any(by_id[s["id"]][name] != s[name] for name in ("prompt_tokens", "new_tokens", "finish", "tokens", "text"))
            for s in samples):
        raise SystemExit("the exported records differ from what the journal holds: rerun the baseline stage")
    try:
        aggregate = bridge.adapt(corpus_paths(run)[1], out, base, corpus_sha256=manifest["corpus_sha256"],
                                 layer_ids=checkpoint.sparse_layers, n_experts=checkpoint.num_experts, top_k=top_k,
                                 expected_n=manifest["sizes_per_direction"]["calibration"])
    except ValueError as error:
        raise SystemExit(f"calibration coverage is incomplete, refusing to select: {error}") from error
    mass = stats["mass"]["prefill"] + stats["mass"]["decode"]
    candidates = {}
    for keep in [int(k) for k in args.keep.split(",")]:
        kept = select(mass, keep, top_k)
        kept_mass = torch.stack([mass[:, layer].index_select(1, torch.tensor(kept[layer])).sum(-1)
                                 for layer in range(len(kept))], 1)
        candidates[str(keep)] = {
            "kept": {str(layer): ids for layer, ids in zip(checkpoint.sparse_layers, kept)},
            "retained_mass_overall": {d: round(float(kept_mass[i].sum() / mass[i].sum()), 6)
                                      for i, d in enumerate(DIRECTIONS)},
            "retained_mass_min_layer": {d: round(float((kept_mass[i] / mass[i].sum(-1)).min()), 6)
                                        for i, d in enumerate(DIRECTIONS)},
        }
    selection = {
        "status": "SELECTION_FROM_COVERAGE_CHECKED_CALIBRATION", "num_experts": checkpoint.num_experts, "top_k": top_k,
        "rule": "per-direction routing-mass share per layer (prefill + decode), averaged over the six directions, "
                "top-k per layer",
        "identity": {"calibration_key": key(binding, "calibration"),
                     **{name: binding[name] for name in ("teacher", "corpus", "code")}},
        "calibration_sentences_per_direction": manifest["sizes_per_direction"]["calibration"],
        "aggregate_status": aggregate["status"], "aggregate_hashes": aggregate["hashes"], "candidates": candidates,
    }
    dump(args.out, selection)
    print(json.dumps({"status": selection["status"],
                      "retained_mass": {k: v["retained_mass_overall"] for k, v in candidates.items()}}))


def cmd_checkpoint_check(args) -> None:
    """Hash every shard of a checkpoint and compare with what the directory claims to be.

    This is the qualification entry point: the bytes are hashed fresh and, for a pinned
    snapshot, the exact 19-file inventory is enforced (R13-P1-02)."""
    identity = checkpoint_identity(args.model, args.cache_dir, exact_inventory=True)
    receipt = {"status": "CHECKPOINT_CONTENT_VERIFIED", **{name: identity[name] for name in
                                                            ("kind", "teacher", "candidate", "num_experts", "files")}}
    if "repo" in identity:
        receipt.update(repo=identity["repo"], revision=identity["revision"])
    receipt["checked"] = utc_now()
    dump(args.out, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "kind", "teacher", "num_experts")}))


def cmd_prune(args) -> None:
    record = prune_candidate(args.model, args.selection, int(args.keep), Path(args.out), Path(args.out_manifest),
                             args.cache_dir)
    manifest = record["manifest"]
    print(json.dumps({"keep": manifest["num_experts"], "tensor_bytes": manifest["tensor_bytes"],
                      "shards": len(manifest["files"]), "verify_pass": record["verify"]["pass"]}))


def cmd_verify(args) -> None:
    result = verify_pruned(Checkpoint(args.model), Checkpoint(args.pruned))
    dump(args.out, result)
    print(json.dumps(result))
    if not result["pass"]:
        raise SystemExit(1)


def cmd_equiv(args) -> None:
    """Compare two traced runs; the reference traces must cover exactly the expected sentences."""
    _, contexts = frozen(args.run_dir, args.manifest_sha256)
    expected = first_per_direction(contexts, args.trace_split, args.trace_n)
    reference_header, reference_rows = read_run(args.reference_records)
    candidate_header, candidate_rows = read_run(args.candidate_records)
    if reference_header is None or candidate_header is None:
        raise SystemExit("both runs are needed")
    differ = [name for name in SHARED_FIELDS
              if reference_header["identity"][name] != candidate_header["identity"][name]]
    if differ:
        raise SystemExit(f"the two runs were not made under the same {', '.join(differ)}")
    prune = read_json(args.prune_record)
    pruned = (prune["verify"]["pass"] and candidate_header["masked_keep"] is None
              and candidate_header["identity"]["candidate"] == pruned_candidate_digest(prune["manifest"])
              and prune["manifest"]["source"]["digest"] == reference_header["identity"]["teacher"])
    if args.kind == "noop":
        pair = pruned and reference_header["masked_keep"] is None and prune["verify"]["identity_mapping"] \
            and reference_header["num_experts"] == candidate_header["num_experts"] == args.keep
    else:
        mask, selection_sha256 = load_selection_mask(args.selection, args.keep)
        pair = pruned and reference_header["masked_keep"] == args.keep \
            and reference_header["identity"]["candidate"] == masked_candidate_digest(mask, selection_sha256) \
            and candidate_header["num_experts"] == args.keep \
            and {int(k): v for k, v in prune["manifest"]["kept"].items()} == mask
    if not pair:
        raise SystemExit("these are not the two runs this check is defined for")
    # Both runs must still be exactly the committed journals the records were exported from,
    # and every witness trace must match the digest committed with its sample (R11-P1-03).
    # Both checkpoints are on disk at this point, so the payload validation is full.
    provenance = {}
    for label, records_path, traces_dir in (("reference", args.reference_records, args.reference_traces),
                                            ("candidate", args.candidate_records, args.candidate_traces)):
        header, _ = read_run(records_path)
        if header is None:
            raise SystemExit(f"{records_path} has no run header")
        provenance[label] = receipt_binding(run_provenance(records_path, contexts, expected, Path(traces_dir),
                                                           checkpoint=Checkpoint(header["model"])))
    if args.kind == "masked":
        # A masked mechanism check without a real changed-route witness proves nothing:
        # zero changes are insufficient evidence, not an automatic pass (R11-P1-03). The
        # count comes from the masked run's committed journal payloads - not from the
        # editable summary - and the journal replay enforces the audit's presence.
        masked_reference = run_provenance(args.reference_records, contexts, expected, Path(args.reference_traces),
                                          checkpoint=Checkpoint(reference_header["model"]), audit_required=True)
        if masked_reference["audit_changed"] == 0:
            raise SystemExit("the mask changed no routing decision on this witness (counted from the journal): "
                             "zero changes are insufficient evidence, not an automatic pass")
        provenance["masked_witness_changed"] = masked_reference["audit_changed"]
    result = compare_runs(Path(args.reference_traces), Path(args.candidate_traces),
                          {row["id"]: row["text"] for row in reference_rows},
                          {row["id"]: row["text"] for row in candidate_rows}, expected)
    result["binding"] = provenance
    dump(args.out, result)
    print(json.dumps({k: result[k] for k in result if k != "per_sentence"}))
    if not result["pass"]:
        raise SystemExit(1)


def cmd_stock_check(args) -> None:
    """Re-translate the fixed per-direction witness sentences with the unmodified reference runtime.

    Before anything is loaded or generated (R13-P1-04/P1-05): the records must replay
    exactly from their journal; the witness sentences are the fixed plan's frozen
    validation IDs - picked by their *context*, never by a row's own split label, and a
    test-context ID is refused whatever a row claims; the actual model directory must be
    the candidate the records were produced from. The receipt binds the actually loaded
    checkpoint identity, tokenizer bytes, runtime and decoding.
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    wanted = set(args.splits.split(","))
    if wanted - {"calibration", "validation"}:
        raise SystemExit(FINAL_TEST_DISABLED + " only calibration/validation witness splits are allowed.")
    _, contexts = frozen_dir(Path(args.corpus).parent, args.manifest_sha256)
    header, rows = read_run(args.records)
    if header is None:
        raise SystemExit(f"{args.records} has no run header")
    # Journal replay first: the exported records must be exactly what the journal holds.
    run_provenance(args.records, contexts, checkpoint=Checkpoint(header["model"]))
    expected = first_per_direction(contexts, PLAN["stock_split"], PLAN["stock_per_direction"])
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"{args.records}: duplicate exported rows")
    picked = []
    for sentence_id in expected:           # the plan's witness subset, by ID, in every direction
        row = by_id.get(sentence_id)
        if row is None:
            raise SystemExit(f"the records lack the plan's witness sentence {sentence_id}: "
                             "the reference runtime cannot be checked")
        context = contexts[sentence_id]
        # The authoritative split is the frozen context's, not the row's own label.
        if context["split"] not in wanted:
            raise SystemExit(f"{sentence_id}: the plan's witness split is not among --splits")
        if any(row.get(name) != context[name] for name in ("split", "direction", "line")):
            raise SystemExit(f"{sentence_id}: the exported row does not match its frozen corpus context")
        if context["split"] == "test":
            raise SystemExit(f"{sentence_id}: a test-context sentence can never be a witness, "
                             "whatever the exported row claims")
        picked.append(row)
    # The actually loaded checkpoint must be the candidate the records came from.
    actual = checkpoint_identity(args.model, args.cache_dir, fresh=False)
    if actual["candidate"] != header["identity"]["candidate"] or actual["teacher"] != header["identity"]["teacher"]:
        raise SystemExit(f"{args.model} is not the checkpoint whose records this reference check is defined for")
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_files = {name: sha256_file(Path(args.model) / name) for name in TOKENIZER_FILES
                       if (Path(args.model) / name).exists()}
    results = []
    for row in picked:
        ids = torch.tensor([contexts[row["id"]]["prompt_ids"]])
        with torch.inference_mode():
            output = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True)
        results.append({"id": row["id"], "engine": row["text"], "reference_runtime": text,
                        "text_equal": text == row["text"]})
    same = sum(r["text_equal"] for r in results)
    # The receipt binds what was actually loaded and run, not what the records merely claim.
    binding = {"candidate": actual["candidate"], "teacher": header["identity"]["teacher"],
               "corpus": header["identity"]["corpus"], "mode": header["identity"].get("mode"),
               "records_sha256": sha256_file(args.records),
               "journal_sha256": sha256_file(journal_path(Path(args.records))),
               "picked_ids": [r["id"] for r in results], "model_path": str(Path(args.model).resolve()),
               "model_kind": actual["kind"], "tokenizer": tokenizer_files,
               "decoding": DECODING, "max_new_tokens": args.max_new_tokens, "threads": args.threads}
    dump(args.out, {"model": str(Path(args.model).resolve()), "sentences": len(results), "text_equal": same,
                    "transformers": transformers.__version__, "attention": "eager", "binding": binding,
                    "pass": same == len(results) and bool(results), "results": results})
    print(json.dumps({"sentences": len(results), "text_equal": same}))
    if same != len(results) or not results:
        raise SystemExit(1)


def shared_identity(headers: dict, where: str) -> dict:
    """The identity fields that must be equal across runs; refuses a mix."""
    first = next(iter(headers.values()))
    for name, header in headers.items():
        differ = [field for field in SHARED_FIELDS if header["identity"][field] != first["identity"][field]]
        if differ:
            raise SystemExit(f"{where}: run {name} differs in {', '.join(differ)}")
    return {field: first["identity"][field] for field in SHARED_FIELDS}


def verify_pilot_chain(run: Path, manifest_sha256: str, cache_dir=None) -> tuple[list[str], dict]:
    """The one evidence-chain verifier of the fixed pilot, shared by decide and report.

    ``decide`` fails closed on any problem; ``report`` turns problems into NOT_EVALUATED
    and a mechanism FAIL instead of ever qualifying a subset (R13-P1-03). Everything is
    derived from the journals and the actual files - never from copied PASS booleans:

      fixed plan        the candidates are exactly PLAN's, the no-op is PLAN's expert
                        count; a narrowed selection.json + CLI pair cannot shrink it;
      selection         recomputed with the deterministic selection rule from the current
                        baseline journal/aggregate and compared with the file, byte for
                        byte on the kept sets;
      teacher           the checkpoint on disk must still hash to the journal's teacher;
      mode              uniform real/rehearsal across every run in the chain; a real
                        decision requires a pinned-snapshot teacher, and candidates that
                        are pruned from exactly that teacher;
      noop              prune verify + identity mapping + recomputed digest + equivalence
                        receipt bound to the current baseline journal and records;
      per candidate     prune verify pass, recomputed candidate digest, source teacher and
                        kept sets, masked reference journal replay (witness audit counted
                        from the journal, nonzero), mechanism candidate journal replay,
                        validation candidate journal replay, exact witness trace bytes,
                        mechanism candidate identity == validation candidate identity,
                        equivalence receipt bound to the current runs, reference-runtime
                        receipt passing on exactly the plan's witness sentences;
      evidence          the inventory with every hash a decision must bind.
    """
    problems = []
    evidence: dict = {"plan": {name: (list(value) if isinstance(value, tuple) else value)
                               for name, value in PLAN.items()}}
    keeps = sorted(PLAN["candidate_keeps"])
    noop_keep = PLAN["experts"]
    baseline = run / "baseline" / "records.jsonl"
    baseline_header, _ = read_run(baseline)
    if baseline_header is None:
        return [f"{baseline} has no run header"], evidence
    try:
        manifest, contexts = frozen(run, manifest_sha256)
        # The witness count is the plan's, capped by what this corpus actually holds: the
        # frozen real corpus guarantees the plan's count, the tiny rehearsal fixture is
        # deliberately smaller. A real decision is refused unless the plan count is met.
        available = min(sum(1 for record in contexts.values()
                            if record["split"] == PLAN["trace_split"] and record["direction"] == direction)
                        for direction in DIRECTIONS)
        witness_count = min(PLAN["trace_witness_per_direction"], available)
        if witness_count < 1:
            raise SystemExit(f"the corpus holds no {PLAN['trace_split']} sentences")
        stock_ids = first_per_direction(contexts, PLAN["stock_split"], PLAN["stock_per_direction"])
        evidence["corpus_sha256"] = manifest["corpus_sha256"]
        evidence["witness_count"] = witness_count
    except SystemExit as error:                    # a blocked corpus blocks everything
        return [str(error)], evidence

    def read_receipt(path: Path, what: str, required: bool = True):
        if not path.exists():
            if required:
                problems.append(f"{what} is missing ({path.name})")
            return None
        try:
            return read_json(path)
        except (OSError, ValueError) as error:
            problems.append(f"{what} is unreadable: {error}")
            return None

    binding = read_json(binding_path(baseline))
    mode = binding.get("detail", {}).get("mode")
    evidence["mode"] = mode
    checkpoint_kind = binding.get("detail", {}).get("checkpoint", {}).get("kind")
    # The teacher on disk must still be the one the baseline journal bound.
    baseline_checkpoint = None
    try:
        current = checkpoint_identity(baseline_header["model"], cache_dir, fresh=False)
        if current["teacher"] != binding["teacher"]:
            problems.append("the baseline checkpoint on disk is no longer the teacher the journal recorded")
        baseline_checkpoint = Checkpoint(baseline_header["model"])
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        problems.append(f"the baseline checkpoint cannot be verified: {type(error).__name__}: {error}")
    if mode == "real":
        if checkpoint_kind != "pinned-snapshot":
            problems.append(f"a real decision requires a pinned-snapshot teacher, found {checkpoint_kind!r}")
        if baseline_header["num_experts"] != noop_keep:
            problems.append(f"a real decision requires the plan's {noop_keep}-expert teacher, "
                            f"found {baseline_header['num_experts']}")
        if evidence.get("witness_count") != PLAN["trace_witness_per_direction"]:
            problems.append(f"a real decision requires the plan's {PLAN['trace_witness_per_direction']} witnesses "
                            f"per direction, the corpus supports {evidence.get('witness_count')}")
    baseline_provenance = None
    try:
        baseline_provenance = run_provenance(baseline, contexts, checkpoint=baseline_checkpoint)
        evidence["baseline"] = {name: baseline_provenance[name]
                                for name in ("records_sha256", "journal_sha256", "teacher", "mode")}
    except SystemExit as error:
        problems.append(f"baseline replay: {error}")
    if baseline_checkpoint is not None:
        # The selection file must be the deterministic rule applied to the current journal.
        selection = read_receipt(run / "selection.json", "the selection")
        if selection is not None:
            evidence["selection_sha256"] = sha256_file(run / "selection.json")
            top_k = int(baseline_checkpoint.config["num_experts_per_tok"])
            max_new_tokens = binding["detail"]["max_new_tokens"]
            if sorted(int(k) for k in selection.get("candidates", {})) != keeps:
                problems.append(f"the selection's candidates {sorted(int(k) for k in selection.get('candidates', {}))} "
                                f"are not exactly the plan's {keeps}")
            if selection.get("identity", {}).get("teacher") != binding["teacher"] \
                    or selection.get("identity", {}).get("calibration_key") != baseline_header["identity_keys"]["calibration"]:
                problems.append("the selection was not made from this baseline run")
            stats_base = run / "baseline" / "stats"
            for name, path in (("corpus", corpus_paths(run)[1]), ("records", baseline),
                               ("stats_json", Path(str(stats_base) + ".json")),
                               ("stats_npz", Path(str(stats_base) + ".npz"))):
                if selection.get("aggregate_hashes", {}).get(name) != sha256_file(path):
                    problems.append(f"the selection's {name} hash does not match the run directory: rerun select")
            try:
                journal = trace.Journal(journal_path(baseline), binding)
                samples = committed_samples(journal, contexts, identity_of(binding), baseline_checkpoint,
                                            max_new_tokens, audit_required=False)
                journal.close()
                stats = rebuild([sample for sample in samples if sample["split"] == "calibration"],
                                baseline_checkpoint.sparse_layers, baseline_checkpoint.num_experts, top_k,
                                max_new_tokens, contexts, audit_required=False)
                mass = stats["mass"]["prefill"] + stats["mass"]["decode"]
                evidence["selection_kept"] = {}
                for keep in keeps:
                    recomputed = {str(layer): ids for layer, ids
                                  in zip(baseline_checkpoint.sparse_layers, select(mass, keep, top_k))}
                    evidence["selection_kept"][str(keep)] = recomputed
                    named = selection.get("candidates", {}).get(str(keep), {}).get("kept")
                    if named is None or {str(k): v for k, v in named.items()} != recomputed:
                        problems.append(f"the selection's kept set for {keep} does not match the deterministic "
                                        "selection rule on the current baseline journal")
            except (ValueError, KeyError, OSError) as error:
                problems.append(f"the selection cannot be recomputed from the baseline journal: "
                                f"{type(error).__name__}: {error}")

    # The identity no-op receipt of the unpruned checkpoint is part of the chain.
    noop_prune = read_receipt(run / f"k{noop_keep}" / "prune.json", "the identity no-op prune record")
    noop_equiv = read_receipt(run / f"k{noop_keep}" / "equiv.json", "the identity no-op equivalence receipt")
    if noop_prune is not None and noop_equiv is not None and baseline_provenance is not None:
        verify = noop_prune.get("verify", {})
        if not (verify.get("pass") and verify.get("identity_mapping")):
            problems.append("the identity no-op prune did not verify")
        expected_digest = pruned_candidate_digest(noop_prune["manifest"])
        if noop_equiv.get("pass") is not True:
            problems.append("the identity no-op equivalence did not pass")
        if noop_equiv.get("binding", {}).get("reference") != receipt_binding(baseline_provenance) \
                or noop_prune["manifest"]["source"]["digest"] != binding["teacher"]:
            problems.append("the identity no-op receipt does not belong to the current baseline")
        evidence["noop"] = {"prune_sha256": sha256_file(run / f"k{noop_keep}" / "prune.json"),
                            "equiv_sha256": sha256_file(run / f"k{noop_keep}" / "equiv.json"),
                            "candidate": expected_digest}
    elif noop_prune is not None:
        evidence["noop"] = {"prune_sha256": sha256_file(run / f"k{noop_keep}" / "prune.json")}

    # Per candidate: the fixed plan's folders, receipts, journals and witness bytes.
    evidence["candidates"] = {}
    for keep in keeps:
        folder = run / f"k{keep}"
        entry: dict = {}
        prune = read_receipt(folder / "prune.json", f"k{keep} prune record")
        equiv = read_receipt(folder / "equiv.json", f"k{keep} equivalence receipt")
        stock = read_receipt(folder / "stock_check.json", f"k{keep} reference-runtime receipt")
        if prune is not None:
            entry["prune_sha256"] = sha256_file(folder / "prune.json")
        if equiv is not None:
            entry["equiv_sha256"] = sha256_file(folder / "equiv.json")
        if stock is not None:
            entry["stock_sha256"] = sha256_file(folder / "stock_check.json")
        if prune is None or equiv is None or stock is None:
            evidence["candidates"][str(keep)] = entry
            continue
        verify, prune_manifest = prune.get("verify", {}), prune.get("manifest", {})
        if verify.get("pass") is not True:                     # the single-field counterexample
            problems.append(f"k{keep}'s prune reload did not pass (verify.pass is {verify.get('pass')!r})")
        if verify.get("identity_mapping") is not False:
            problems.append(f"k{keep} is not a real (non-identity) prune candidate")
        header, _ = read_run(folder / "records.jsonl")
        if header is None:
            problems.append(f"k{keep} has no validation run header")
            evidence["candidates"][str(keep)] = entry
            continue
        entry["candidate"] = header["identity"]["candidate"]
        entry["kept"] = {str(layer): ids for layer, ids in prune_manifest.get("kept", {}).items()}
        recomputed_digest = pruned_candidate_digest(prune_manifest)
        if header["identity"]["candidate"] != recomputed_digest:
            problems.append(f"k{keep}'s validation run is not the checkpoint its prune manifest describes")
        if prune_manifest.get("source", {}).get("digest") != binding["teacher"]:
            problems.append(f"k{keep} was not pruned from the baseline teacher")
        if mode == "real" and prune_manifest.get("source", {}).get("kind") != "pinned-snapshot":
            problems.append(f"a real k{keep} candidate must be pruned from the pinned teacher, "
                            f"found source kind {prune_manifest.get('source', {}).get('kind')!r}")
        named_kept = evidence.get("selection_kept", {}).get(str(keep))
        if named_kept is not None and {str(k): v for k, v in prune_manifest.get("kept", {}).items()} != named_kept:
            problems.append(f"k{keep} was not pruned the way the recomputed selection names")
        try:
            masked = run_provenance(folder / "masked" / "records.jsonl", contexts,
                                    traces_dir=folder / "masked" / "traces",
                                    checkpoint=baseline_checkpoint, audit_required=True)
            entry["masked"] = {"records_sha256": masked["records_sha256"], "journal_sha256": masked["journal_sha256"],
                               "witness_changed": masked["audit_changed"], "traced": masked["traced"],
                               "trace_digests": masked.get("trace_digests", {})}
            if masked["audit_changed"] == 0:
                problems.append(f"k{keep}'s masked witness changed no routing decision (counted from the journal)")
        except SystemExit as error:
            problems.append(f"k{keep} masked replay: {error}")
        try:
            mechanism = run_provenance(folder / "mechanism" / "records.jsonl", contexts,
                                       traces_dir=folder / "mechanism" / "traces")
            entry["mechanism"] = {"records_sha256": mechanism["records_sha256"],
                                  "journal_sha256": mechanism["journal_sha256"], "traced": mechanism["traced"],
                                  "trace_digests": mechanism.get("trace_digests", {}),
                                  "candidate": mechanism["candidate"]}
            if mechanism["candidate"] != header["identity"]["candidate"]:
                problems.append(f"k{keep}'s mechanism run is not the checkpoint its validation run evaluates")
        except SystemExit as error:
            problems.append(f"k{keep} mechanism replay: {error}")
        # The two witness runs of one candidate must cover exactly the same sentences, and a
        # real run must carry the plan's witness count - never a silently reduced one.
        if "masked" in entry and "mechanism" in entry                 and entry["masked"]["traced"] != entry["mechanism"]["traced"]:
            problems.append(f"k{keep}'s masked and mechanism witness sentences differ")
        traced_count = len(entry.get("masked", entry.get("mechanism", {})).get("traced", []))             if isinstance(entry.get("masked", entry.get("mechanism")), dict) else 0
        if mode == "real" and traced_count != len(DIRECTIONS) * PLAN["trace_witness_per_direction"]:
            problems.append(f"a real k{keep} must trace the plan's "
                            f"{len(DIRECTIONS) * PLAN['trace_witness_per_direction']} witness sentences, "
                            f"found {traced_count}")
        if mode != "real" and traced_count == 0:
            problems.append(f"k{keep} traced no witness sentences at all")
        try:
            validation = run_provenance(folder / "records.jsonl", contexts)
            entry["validation"] = {"records_sha256": validation["records_sha256"],
                                   "journal_sha256": validation["journal_sha256"]}
        except SystemExit as error:
            problems.append(f"k{keep} validation replay: {error}")
        if equiv.get("pass") is not True:
            problems.append(f"k{keep} did not pass the masked-equivalence check")
        if baseline_provenance is not None:
            if "masked" in entry and equiv.get("binding", {}).get("reference") != receipt_binding(masked):
                problems.append(f"k{keep}'s equivalence receipt does not belong to the current masked run")
            if "mechanism" in entry and equiv.get("binding", {}).get("candidate") != receipt_binding(mechanism):
                problems.append(f"k{keep}'s equivalence receipt does not belong to the current mechanism run")
        expected_stock = stock_ids
        stock_binding = stock.get("binding", {})
        if stock.get("pass") is not True:
            problems.append(f"k{keep} did not pass the reference-runtime check")
        if sorted(stock_binding.get("picked_ids", [])) != sorted(expected_stock):
            problems.append(f"k{keep}'s reference-runtime receipt does not cover exactly the plan's witness "
                            f"sentences {sorted(expected_stock)}")
        validation_records_sha = entry.get("validation", {}).get("records_sha256")
        if validation_records_sha is not None and stock_binding.get("records_sha256") != validation_records_sha:
            problems.append(f"k{keep}'s reference-runtime receipt does not match the current validation records")
        if stock_binding.get("candidate") != header["identity"]["candidate"] \
                or stock_binding.get("teacher") != binding["teacher"] \
                or stock_binding.get("mode") != mode:
            problems.append(f"k{keep}'s reference-runtime receipt is not bound to this run's actual identity")
        evidence["candidates"][str(keep)] = entry
    return problems, evidence


def cmd_decide(args) -> None:
    """Provisional candidate from validation records only. Never opens a test record.

    The shared verifier (verify_pilot_chain) must find the whole fixed-pilot evidence
    chain present, current and plan-compliant before anything qualifies; any problem
    blocks the decision instead of qualifying a subset (R13-P1-03).
    """
    import sacrebleu
    run = Path(args.run_dir)
    if sorted(int(k) for k in args.candidates.split(",")) != sorted(PLAN["candidate_keeps"]):
        raise SystemExit(f"the fixed pilot plan's candidates are {sorted(PLAN['candidate_keeps'])}, "
                         f"not {args.candidates}")
    if args.max_drop != PLAN["max_chrf_drop"]:
        raise SystemExit(f"the fixed pilot plan's gate is {PLAN['max_chrf_drop']}, not {args.max_drop}")
    manifest_sha256 = args.manifest_sha256 or CORPUS_MANIFEST_SHA256
    problems, evidence = verify_pilot_chain(run, manifest_sha256, args.cache_dir)
    if problems:
        raise SystemExit("the evidence chain is incomplete or stale:\n - " + "\n - ".join(problems))
    keeps = sorted(PLAN["candidate_keeps"])
    baseline = run / "baseline" / "records.jsonl"
    paths = {keep: run / f"k{keep}" / "records.jsonl" for keep in keeps}
    headers = {}
    for keep, path in ((None, baseline), *paths.items()):
        header, _ = read_run(path)
        headers[keep] = header
    if sacrebleu.__version__ != "2.6.0":
        raise SystemExit(f"expected the pinned sacrebleu 2.6.0, found {sacrebleu.__version__}")
    metric = chrf_pp()
    manifest_path, corpus_path = corpus_paths(run)
    try:
        result = handoff.evaluate(manifest_path, manifest_sha256, corpus_path, baseline, paths,
                                  lambda hypotheses, references: float(metric.corpus_score(hypotheses, [references]).score),
                                  args.max_drop)
    except ValueError as error:
        raise SystemExit(f"candidate decision refused: {error}") from error
    result["identity"] = shared_identity(headers, "candidate decision")
    result["runs"] = {
        ("baseline" if keep is None else f"k{keep}"): {
            "records_sha256": sha256_file(path), "validation_key": header["identity_keys"]["validation"],
            "candidate": header["identity"]["candidate"]}
        for (keep, path), header in zip(((None, baseline), *paths.items()), headers.values())}
    result["synthetic"] = headers[None]["synthetic"]
    result["max_new_tokens"] = headers[None]["max_new_tokens"]
    result["mode"] = evidence["mode"]
    # Every binding above was checked against current content: the provisional decision's
    # provenance is verified, unlike the reviewed helper's conservative default.
    result["checkpoint_provenance_verified"] = True
    result["evidence"] = evidence
    result["final_test"] = {"status": "HOLD", "reason": FINAL_TEST_DISABLED}
    if result["test_execution_authorized"] is not False:
        raise SystemExit("a provisional decision must keep test_execution_authorized false")
    result["note"] = ("provisional: validation sentences only; the test split has not been run for any candidate; "
                      "not a quality result until the selection identity and evidence are independently reviewed")
    dump(args.out, result)
    print(json.dumps({"status": result["status"], "chosen_keep": result["chosen_keep"],
                      "test_execution_authorized": result["test_execution_authorized"]}))


def cmd_report(args) -> None:
    from . import report
    built = report.build(Path(args.run_dir), args.manifest_sha256, args.max_drop, args.spot_n)
    print(json.dumps({"rows": len(built["chrf_pp"]), "mechanism_pass": built["verdict"]["mechanism_pass"],
                      "quality": built["verdict"]["quality"]}))


def add_command(commands, name: str, help: str, func, *options) -> None:
    """options: (flags, dest, kwargs) triples."""
    sub = commands.add_parser(name, help=help)
    for flags, dest, kwargs in options:
        sub.add_argument(*flags, dest=dest, **kwargs)
    sub.set_defaults(func=func)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mtpruning", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    required = {"required": True}
    pinned = {"help": "sha256 the corpus manifest must have (default: the committed frozen one)"}
    threads = (["--threads"], "threads", {"type": int, "default": 16})
    cache = (["--cache-dir"], "cache_dir", {"help": "where checkpoint content hashes are cached"})

    add_command(commands, "corpus", "build a new FLORES-200 sample (never to replace the frozen corpus)", cmd_corpus,
                (["--tarball"], "tarball", required), (["--tokenizer"], "tokenizer", required),
                (["--out"], "out", required),
                (["--n-calibration"], "n_calibration", {"type": int, "default": 24}),
                (["--n-validation"], "n_validation", {"type": int, "default": 20}),
                (["--n-test"], "n_test", {"type": int, "default": 20}),
                (["--seed"], "seed", {"type": int, "default": 42}))
    add_command(commands, "corpus-check", "check that the corpus on disk is the frozen one", cmd_corpus_check,
                (["--corpus-dir"], "corpus_dir", required), (["--out"], "out", required),
                (["--manifest-sha256"], "manifest_sha256", pinned))
    add_command(commands, "run", "translate corpus sentences into a journal (never the test split)", cmd_run,
                (["--model"], "model", required), (["--corpus"], "corpus", required), (["--out"], "out", required),
                (["--splits"], "splits", {"type": lambda s: s.split(","), "default": ["calibration", "validation"]}),
                (["--max-new-tokens"], "max_new_tokens", {"type": int, "default": 640}), threads,
                (["--mode"], "mode", {"choices": ["real", "rehearsal"], "default": "rehearsal",
                                      "help": "real: pinned teacher, the checkpoint's own tokenizer, "
                                              "the frozen manifest, no calibration limits"}),
                (["--tokenizer"], "tokenizer", {}),
                (["--mask"], "mask", {"help": "selection JSON: deny the experts pruned by candidate --mask-keep"}),
                (["--mask-keep"], "mask_keep", {"type": int}),
                (["--stats"], "stats", {"help": "base path for the calibration routing statistics"}),
                (["--trace-dir"], "trace_dir", {}), (["--trace-n"], "trace_n", {"type": int, "default": 0}),
                (["--trace-split"], "trace_split", {"default": "validation"}),
                (["--limit"], "limit", {"type": int, "help": "sentences per split and direction (dry runs)"}),
                (["--manifest-sha256"], "manifest_sha256", pinned), cache)
    add_command(commands, "select", "expert sets from the calibration routing, coverage-checked", cmd_select,
                (["--run-dir"], "run_dir", required), (["--keep"], "keep", required), (["--out"], "out", required),
                (["--manifest-sha256"], "manifest_sha256", pinned), cache)
    add_command(commands, "checkpoint-check", "hash every shard and check the checkpoint's provenance",
                cmd_checkpoint_check, (["--model"], "model", required), (["--out"], "out", required), cache)
    add_command(commands, "prune", "write a physically pruned checkpoint and verify the reload", cmd_prune,
                (["--model"], "model", required), (["--selection"], "selection", {}),
                (["--keep"], "keep", {**required, "type": int}), (["--out"], "out", required),
                (["--out-manifest"], "out_manifest", required), cache)
    add_command(commands, "verify", "compare every tensor of a pruned checkpoint with its source", cmd_verify,
                (["--model"], "model", required), (["--pruned"], "pruned", required), (["--out"], "out", required))
    add_command(commands, "equiv", "identity no-op or masked equivalence, on exactly the traced sentences", cmd_equiv,
                (["--kind"], "kind", {**required, "choices": ["noop", "masked"]}),
                (["--keep"], "keep", {**required, "type": int}),
                (["--run-dir"], "run_dir", required), (["--prune-record"], "prune_record", required),
                (["--reference-traces"], "reference_traces", required),
                (["--candidate-traces"], "candidate_traces", required),
                (["--reference-records"], "reference_records", required),
                (["--candidate-records"], "candidate_records", required),
                (["--selection"], "selection", {}),
                (["--trace-n"], "trace_n", {**required, "type": int}),
                (["--trace-split"], "trace_split", {"default": "validation"}),
                (["--out"], "out", required), (["--manifest-sha256"], "manifest_sha256", pinned))
    add_command(commands, "stock-check", "re-translate with the unmodified reference implementation",
                cmd_stock_check,
                (["--model"], "model", required), (["--corpus"], "corpus", required),
                (["--records"], "records", required), (["--out"], "out", required),
                (["--splits"], "splits", {"default": "validation"}),
                (["--limit"], "limit", {"type": int, "default": 1}),
                (["--max-new-tokens"], "max_new_tokens", {"type": int, "default": 640}), threads,
                (["--manifest-sha256"], "manifest_sha256", pinned), cache)
    add_command(commands, "decide", "provisional candidate from validation records only", cmd_decide,
                (["--run-dir"], "run_dir", required), (["--candidates"], "candidates", required),
                (["--out"], "out", required),
                (["--max-drop"], "max_drop", {"type": float, "default": MAX_CHRF_DROP}),
                (["--manifest-sha256"], "manifest_sha256", pinned), cache)
    add_command(commands, "report", "assemble report.json and report.md", cmd_report,
                (["--run-dir"], "run_dir", required),
                (["--max-drop"], "max_drop", {"type": float, "default": MAX_CHRF_DROP}),
                (["--spot-n"], "spot_n", {"type": int, "default": 3}),
                (["--manifest-sha256"], "manifest_sha256", pinned))

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
