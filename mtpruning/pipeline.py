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
from .identity import FIELDS, checkpoint_identity, digest, key, run_identity, sha256_file
from .prune import verify_pruned, write_pruned
from .reviewed import bridge, handoff, trace
from .spec import CORPUS_MANIFEST_SHA256, DECODING, MAX_CHRF_DROP

# Test hook only: stop after this many commits of one invocation, to simulate a crash.
CRASH_AFTER = int(os.environ.get("MTPRUNING_TEST_CRASH_AFTER", "0"))
# Identity fields the baseline and every candidate must share; "decoding" covers decoding and max tokens.
SHARED_FIELDS = ("teacher", "corpus", "prompt", "runtime", "code", "decoding")
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
                   traces_dir: Path | None = None, checkpoint: Checkpoint | None = None) -> dict:
    """Replay a run's journal against its exported records, and its traces against the journal.

    The records export must be exactly what the journal replays to, the journal file
    must be the one the summary recorded, the exported header must be the journal's own
    identity, and - with ``expected_traced`` - exactly those sentences may carry traces,
    each matching the digest committed with its sample (R11-P1-03/P1-04). Every payload
    is checked against its stored digest and the run's sample keys even without
    ``checkpoint``; with it, every payload is also validated in full. A checkpoint is
    only available while its bytes are on disk - the driver deletes a pruned checkpoint
    after its receipts, which is why later stages replay without one.
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
    if differ:
        raise SystemExit(f"{records_path}: the exported header does not match the journal's identity: "
                         f"{', '.join(differ)}")
    full = identity_of(binding)
    journal = trace.Journal(journal_file, binding)
    samples = []
    for sample_id, payload in journal.records():        # verifies every payload against its stored digest
        if sample_id not in contexts:
            raise SystemExit(f"the journal holds {sample_id}, which is not in the frozen corpus")
        if payload.get("identity_key") != full["sample_keys"][contexts[sample_id]["split"]]:
            raise SystemExit(f"{sample_id} was committed under another run identity")
        sample = {"id": sample_id, **payload}
        if checkpoint is not None:
            validate_payload(sample, contexts[sample_id], checkpoint.sparse_layers, checkpoint.num_experts,
                             int(checkpoint.config["num_experts_per_tok"]), header["max_new_tokens"])
        samples.append(sample)
    journal.close()
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {sample["id"] for sample in samples} or any(
            any(by_id[sample["id"]][name] != sample[name]
                for name in ("prompt_tokens", "new_tokens", "finish", "tokens", "text"))
            for sample in samples):
        raise SystemExit(f"{records_path}: the exported records differ from what the journal holds")
    traced = sorted(row["id"] for row in rows if row.get("traced"))
    if expected_traced is not None and traced != sorted(expected_traced):
        raise SystemExit(f"{records_path}: traced sentences {len(traced)} do not cover exactly the expected "
                         f"{len(expected_traced)}")
    if traces_dir is not None:
        for sample in samples:
            if sample["id"] in (expected_traced or traced):
                if not sample.get("trace") or not trace_complete(traces_dir, sample["id"], sample["trace"]):
                    raise SystemExit(f"{sample['id']}: the trace on disk does not match the journal's record")
    return {"records_sha256": sha256_file(records_path), "journal_sha256": sha256_file(journal_file),
            "teacher": header["identity"]["teacher"], "corpus": header["identity"]["corpus"],
            "code": header["identity"]["code"], "candidate": header["identity"]["candidate"], "traced": traced}


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
            payload = {
                "identity_key": full["sample_keys"][record["split"]], "split": record["split"],
                "direction": record["direction"], "line": record["line"],
                "prompt_tokens": generated.prompt_tokens, "new_tokens": len(generated.tokens),
                "finish": generated.finish, "tokens": generated.tokens,
                "text": tokenizer.decode(generated.tokens, skip_special_tokens=True),
                "routing": cells, "audit": audit.payload(cells, depth) if audit is not None else None,
                "trace": save_trace(options.trace_dir, record["id"], generated.trace)
                         if record["id"] in traced else None,
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
        "identity": {name: binding[name] for name in (*FIELDS, "decoding")},
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
                                                "decoding", "identity", "identity_keys", "checkpoint_kind", "synthetic")}
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
    """Hash every shard of a checkpoint and compare with what the directory claims to be."""
    identity = checkpoint_identity(args.model, args.cache_dir)
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
        provenance[label] = run_provenance(records_path, contexts, expected, Path(traces_dir),
                                           checkpoint=Checkpoint(header["model"]))
    if args.kind == "masked":
        # A masked mechanism check without a real changed-route witness proves nothing:
        # zero changes are insufficient evidence, not an automatic pass (R11-P1-03).
        audit = read_json(Path(str(args.reference_records) + ".summary.json")).get("mask_audit")
        if not audit:
            raise SystemExit("the masked run carries no mask audit: nothing records what the mask changed")
        changed = sum(phase["changed"] for phases in audit.values() for phase in phases.values())
        if changed == 0:
            raise SystemExit("the mask changed no routing decision on this witness: zero changes are "
                             "insufficient evidence, not an automatic pass")
    result = compare_runs(Path(args.reference_traces), Path(args.candidate_traces),
                          {row["id"]: row["text"] for row in reference_rows},
                          {row["id"]: row["text"] for row in candidate_rows}, expected)
    result["binding"] = provenance
    dump(args.out, result)
    print(json.dumps({k: result[k] for k in result if k != "per_sentence"}))
    if not result["pass"]:
        raise SystemExit(1)


def cmd_stock_check(args) -> None:
    """Re-translate a few sentences with the unmodified reference loader and implementation."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if "test" in set(args.splits.split(",")):
        raise SystemExit(FINAL_TEST_DISABLED)          # no path may expose the test split (R11-P1-05)
    _, contexts = frozen_dir(Path(args.corpus).parent, args.manifest_sha256)
    header, rows = read_run(args.records)
    if header is None:
        raise SystemExit(f"{args.records} has no run header")
    wanted, seen, picked = set(args.splits.split(",")), Counter(), []
    for row in rows:                       # records are exported in ID order
        if row["split"] in wanted and seen[row["direction"]] < args.limit:
            picked.append(row)
            seen[row["direction"]] += 1
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = []
    for row in picked:
        ids = torch.tensor([contexts[row["id"]]["prompt_ids"]])
        with torch.inference_mode():
            output = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True)
        results.append({"id": row["id"], "engine": row["text"], "reference_runtime": text,
                        "text_equal": text == row["text"]})
    same = sum(r["text_equal"] for r in results)
    # The receipt binds the run it checked: which candidate, which exact records bytes (R11-P1-04).
    binding = {"candidate": header["identity"]["candidate"], "teacher": header["identity"]["teacher"],
               "corpus": header["identity"]["corpus"], "records_sha256": sha256_file(args.records),
               "picked_ids": [r["id"] for r in results]}
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


def cmd_decide(args) -> None:
    """Provisional candidate from validation records only. Never opens a test record.

    Before anything qualifies, the whole evidence chain must be present and current
    (R11-P1-04): baseline journal -> aggregate -> selection -> prune/manifests -> identity
    no-op and per-candidate masked equivalence and reference-runtime receipts -> complete
    validation records. A missing, stale, cross-run, subset or mixed receipt blocks the
    decision instead of qualifying a subset.
    """
    import sacrebleu
    run = Path(args.run_dir)
    keeps = [int(k) for k in args.candidates.split(",")]
    manifest_sha256 = args.manifest_sha256 or CORPUS_MANIFEST_SHA256
    manifest, contexts = frozen(run, manifest_sha256)
    baseline = run / "baseline" / "records.jsonl"
    paths = {keep: run / f"k{keep}" / "records.jsonl" for keep in keeps}
    selection = read_json(run / "selection.json")
    planned = sorted(int(k) for k in selection["candidates"])
    if sorted(keeps) != planned:
        raise SystemExit(f"the candidates to decide on {sorted(keeps)} are not exactly the planned ones {planned}")
    baseline_header, _ = read_run(baseline)
    if baseline_header is None:
        raise SystemExit(f"{baseline} has no run header")
    binding = read_json(binding_path(baseline))
    # The teacher on disk must still be the one the baseline journal bound, and the selection
    # must still be the one made from this baseline's checked aggregate.
    current = checkpoint_identity(baseline_header["model"], args.cache_dir, fresh=False)
    if current["teacher"] != binding["teacher"]:
        raise SystemExit("the baseline checkpoint on disk is no longer the teacher the journal recorded")
    baseline_checkpoint = Checkpoint(baseline_header["model"])
    if selection["identity"]["teacher"] != binding["teacher"] \
            or selection["identity"]["calibration_key"] != baseline_header["identity_keys"]["calibration"]:
        raise SystemExit("the selection was not made from this baseline run")
    stats_base = run / "baseline" / "stats"
    for name, path in (("corpus", corpus_paths(run)[1]), ("records", baseline),
                       ("stats_json", Path(str(stats_base) + ".json")), ("stats_npz", Path(str(stats_base) + ".npz"))):
        if selection["aggregate_hashes"].get(name) != sha256_file(path):
            raise SystemExit(f"the selection's {name} hash does not match the run directory: rerun select")
    run_provenance(baseline, contexts, checkpoint=baseline_checkpoint)
    # The identity no-op receipt of the unpruned checkpoint is part of the chain.
    noop_keep = baseline_header["num_experts"]
    noop_prune_path = run / f"k{noop_keep}" / "prune.json"
    noop_equiv_path = run / f"k{noop_keep}" / "equiv.json"
    if not noop_prune_path.exists() or not noop_equiv_path.exists():
        raise SystemExit(f"no identity no-op evidence at {noop_prune_path.parent}")
    noop_prune = read_json(noop_prune_path)
    if not (noop_prune["verify"]["pass"] and noop_prune["verify"]["identity_mapping"]):
        raise SystemExit("the identity no-op prune did not verify")
    noop_equiv = read_json(noop_equiv_path)
    if not noop_equiv.get("pass") \
            or noop_equiv["binding"]["reference"]["journal_sha256"] != sha256_file(journal_path(baseline)) \
            or noop_equiv["binding"]["reference"]["teacher"] != binding["teacher"]:
        raise SystemExit("the identity no-op receipt does not belong to the current baseline")
    headers = {}
    for keep, path in ((None, baseline), *paths.items()):
        header, _ = read_run(path)
        if header is None:
            raise SystemExit(f"{path} has no run header")
        if keep is not None:
            # The candidate that was evaluated must be the one the selection named.
            prune = read_json(run / f"k{keep}" / "prune.json")
            named = {int(k): v for k, v in selection["candidates"][str(keep)]["kept"].items()}
            if ({int(k): v for k, v in prune["manifest"]["kept"].items()} != named
                    or header["identity"]["candidate"] != prune["candidate_digest"]):
                raise SystemExit(f"k{keep} was not pruned the way the selection names")
            # The masked-equivalence and reference-runtime receipts must exist, pass, and describe
            # exactly the files on disk now: a prior run's receipt or an altered record blocks.
            equiv_path, stock_path = run / f"k{keep}" / "equiv.json", run / f"k{keep}" / "stock_check.json"
            if not equiv_path.exists():
                raise SystemExit(f"k{keep} has no masked-equivalence receipt")
            equiv = read_json(equiv_path)
            if not equiv.get("pass"):
                raise SystemExit(f"k{keep} did not pass the masked-equivalence check")
            reference = run_provenance(run / f"k{keep}" / "masked" / "records.jsonl", contexts)
            candidate_provenance = run_provenance(run / f"k{keep}" / "mechanism" / "records.jsonl", contexts)
            if equiv["binding"]["reference"] != reference or equiv["binding"]["candidate"] != candidate_provenance \
                    or reference["teacher"] != binding["teacher"]:
                raise SystemExit(f"k{keep}'s equivalence receipt does not belong to the current runs")
            run_provenance(path, contexts)             # journal and exported records must agree
            if not stock_path.exists():
                raise SystemExit(f"k{keep} has no reference-runtime receipt")
            stock = read_json(stock_path)
            if not stock.get("pass"):
                raise SystemExit(f"k{keep} did not pass the reference-runtime check")
            if stock["binding"]["records_sha256"] != sha256_file(path) \
                    or stock["binding"]["candidate"] != header["identity"]["candidate"] \
                    or stock["binding"]["teacher"] != binding["teacher"]:
                raise SystemExit(f"k{keep}'s reference-runtime receipt does not match the current records")
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
    # Every binding above was checked against current content: the provisional decision's
    # provenance is verified, unlike the reviewed helper's conservative default.
    result["checkpoint_provenance_verified"] = True
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
                (["--manifest-sha256"], "manifest_sha256", pinned))
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
