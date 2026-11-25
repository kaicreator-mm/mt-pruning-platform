"""D01b pipeline: corpus → run → select → prune → verify → equiv → stock-check → report.

Every stage writes a file and nothing else; no stage depends on state left in
memory by another one. ``run`` is resumable per sentence, so an interrupted
experiment continues where its records stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .calibration import PHASES, MaskAudit, RoutingStats, select
from .corpus import DIRECTIONS, build_corpus, load_corpus
from .engine import Checkpoint, Engine
from .evaluate import compare_runs, save_trace
from .prune import verify_pruned, write_pruned


SPLITS_ORDER = [(split, d) for split in ("calibration", "validation", "test") for d in DIRECTIONS]

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment() -> dict:
    import sacrebleu
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "sacrebleu": sacrebleu.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def trace_path(directory: Path, sentence_id: str) -> Path:
    return Path(directory) / (sentence_id.replace(":", "__") + ".logits.npy")


def read_records(path: Path) -> tuple[dict | None, list[dict]]:
    """The run header (first line) and the sentence records of a records file."""
    header, rows = None, []
    if Path(path).exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("type") == "run":
                        header = row
                    elif row.get("type") == "sentence":
                        rows.append(row)
    return header, rows


def texts(rows: list[dict]) -> dict:
    return {row["id"]: row["text"] for row in rows}


def cmd_corpus(args) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = build_corpus(args.tarball, tokenizer, args.out, args.n_calibration, args.n_validation,
                            args.n_test, args.seed)
    print(json.dumps({k: manifest[k] for k in ("records", "corpus_sha256", "seed", "sizes_per_direction")}))


def cmd_run(args) -> None:
    from transformers import AutoTokenizer
    corpus = load_corpus(args.corpus)
    splits = args.splits.split(",")
    todo = [r for r in corpus if r["split"] in splits]
    if args.limit:
        seen, limited = Counter(), []
        for rec in todo:
            if seen[(rec["split"], rec["direction"])] < args.limit:
                limited.append(rec)
            seen[(rec["split"], rec["direction"])] += 1
        todo = limited
    checkpoint = Checkpoint(args.model)
    engine = Engine(checkpoint, threads=args.threads)
    mask = None
    if args.mask:
        selection = json.loads(Path(args.mask).read_text(encoding="utf-8"))
        if "candidates" in selection:
            selection = selection["candidates"][str(args.mask_keep)]
        mask = {int(k): v for k, v in selection["kept"].items()}
        engine.set_mask(mask)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    stats = RoutingStats(engine) if args.stats else None
    stats_splits = set((args.stats_splits or "").split(","))
    audit = MaskAudit() if mask else None
    decisions = {d: {phase: 0 for phase in PHASES} for d in DIRECTIONS}
    traced, seen = set(), Counter()
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        for rec in todo:
            if rec["split"] == args.trace_split and seen[rec["direction"]] < args.trace_n:
                traced.add(rec["id"])
                seen[rec["direction"]] += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header, rows = read_records(out)
    if header and header["environment"]["torch"] != torch.__version__:
        raise SystemExit("records were produced by another torch build; refusing to mix")
    done = {row["id"] for row in rows
            if not row["traced"] or trace_path(args.trace_dir, row["id"]).exists()}
    depth = len(checkpoint.sparse_layers)
    with open(out, "a", encoding="utf-8", newline="\n") as handle:
        if header is None:
            handle.write(json.dumps({
                "type": "run", "model": str(Path(args.model).resolve()), "num_experts": checkpoint.num_experts,
                "masked_keep": len(next(iter(mask.values()))) if mask else None, "splits": splits,
                "max_new_tokens": args.max_new_tokens, "decoding": "greedy (argmax of bf16 logits)",
                "corpus_sha256": sha256_file(args.corpus), "environment": environment(), "start": utc_now(),
            }, ensure_ascii=False) + "\n")
            handle.flush()
        finished = Counter()
        for rec in todo:
            if rec["id"] in done:
                continue
            direction = rec["direction"]
            collecting = stats is not None and rec["split"] in stats_splits
            if collecting:
                stats.begin(direction)
            engine.on_route = stats.hook if collecting else None
            engine.audit = None
            if audit is not None:
                audit.begin(direction)
                engine.audit = audit.hook
            gen = engine.generate(rec["prompt_ids"], args.max_new_tokens, trace=rec["id"] in traced)
            if audit is not None:
                decisions[direction]["prefill"] += gen.prompt_tokens * depth
                decisions[direction]["decode"] += len(gen.tokens) * depth
            if collecting:
                stats.sentences[DIRECTIONS.index(direction)] += 1
            row = {"type": "sentence", "id": rec["id"], "split": rec["split"], "direction": direction,
                   "line": rec["line"], "prompt_tokens": gen.prompt_tokens, "new_tokens": len(gen.tokens),
                   "finish": gen.finish, "tokens": gen.tokens,
                   "text": tokenizer.decode(gen.tokens, skip_special_tokens=True),
                   "prefill_seconds": round(gen.prefill_seconds, 3), "decode_seconds": round(gen.decode_seconds, 3),
                   "traced": rec["id"] in traced}
            if row["traced"]:
                save_trace(Path(args.trace_dir), rec["id"], gen.trace)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            finished[(row["split"], direction)] += 1
    _, rows = read_records(out)
    decode_seconds = sum(r["decode_seconds"] for r in rows)
    summary = {
        "model": str(Path(args.model).resolve()), "num_experts": checkpoint.num_experts,
        "masked_keep": len(next(iter(mask.values()))) if mask else None, "splits": splits,
        "max_new_tokens": args.max_new_tokens, "finished": utc_now(), "environment": environment(),
        "sentences": len(rows),
        "per_split_direction_total": {f"{s}:{d}": n for (s, d), n in sorted(Counter((r["split"], r["direction"]) for r in rows).items())},
        "finish": dict(Counter(r["finish"] for r in rows)),
        "new_tokens": sum(r["new_tokens"] for r in rows),
        "prefill_seconds": round(sum(r["prefill_seconds"] for r in rows), 1),
        "decode_seconds": round(decode_seconds, 1),
        "decode_tokens_per_second": round(sum(r["new_tokens"] for r in rows) / max(decode_seconds, 1e-9), 3),
        "traced": sum(bool(r["traced"]) for r in rows),
    }
    if stats is not None:
        summary["routing"] = stats.save(Path(args.stats))
    if audit is not None:
        summary["mask_audit"] = audit.report(decisions)
    dump(Path(str(out) + ".summary.json"), summary)
    print(json.dumps({k: summary[k] for k in ("sentences", "finish", "new_tokens", "decode_tokens_per_second")}))


def cmd_select(args) -> None:
    checkpoint = Checkpoint(args.model)
    stats = RoutingStats.load(args.stats)
    if stats.num_experts != checkpoint.num_experts:
        raise SystemExit("routing stats do not match the checkpoint")
    mass = stats.mass["prefill"] + stats.mass["decode"]
    top_k = int(checkpoint.config["num_experts_per_tok"])
    candidates = {}
    for keep in [int(k) for k in args.keep.split(",")]:
        kept = select(mass, keep, top_k)
        kept_mass = torch.stack([mass[:, layer].index_select(1, torch.tensor(kept[layer])).sum(-1)
                                 for layer in range(len(kept))], 1)
        candidates[str(keep)] = {
            "kept": {str(layer): ids for layer, ids in zip(stats.sparse_layers, kept)},
            "retained_mass_overall": {d: round(float(kept_mass[i].sum() / mass[i].sum()), 6)
                                      for i, d in enumerate(DIRECTIONS)},
            "retained_mass_min_layer": {d: round(float((kept_mass[i] / mass[i].sum(-1)).min()), 6)
                                        for i, d in enumerate(DIRECTIONS)},
        }
    dump(args.out, {"num_experts": checkpoint.num_experts, "top_k": top_k, "stats": str(args.stats),
                    "rule": "per-direction routing-mass share per layer, averaged over the six directions, "
                            "top-k per layer",
                    "candidates": candidates})
    print(json.dumps({k: v["retained_mass_overall"] for k, v in candidates.items()}))


def cmd_prune(args) -> None:
    source = Checkpoint(args.model)
    keep = int(args.keep)
    if keep == source.num_experts:
        kept = {layer: list(range(source.num_experts)) for layer in source.sparse_layers}
    else:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        kept = {int(k): v for k, v in selection["candidates"][str(keep)]["kept"].items()}
    manifest = write_pruned(source, kept, Path(args.out))
    result = verify_pruned(source, Checkpoint(args.out))
    dump(args.out_manifest, {"manifest": manifest, "verify": result})
    print(json.dumps({"keep": keep, "tensor_bytes": manifest["tensor_bytes"], "shards": len(manifest["files"]),
                      "verify_pass": result["pass"], "mismatched": result["mismatched_tensors"]}))


def cmd_verify(args) -> None:
    result = verify_pruned(Checkpoint(args.model), Checkpoint(args.pruned))
    dump(args.out, result)
    print(json.dumps(result))
    if not result["pass"]:
        raise SystemExit(1)


def cmd_equiv(args) -> None:
    _, reference_rows = read_records(args.reference_records)
    _, candidate_rows = read_records(args.candidate_records)
    result = compare_runs(Path(args.reference_traces), Path(args.candidate_traces),
                          texts(reference_rows), texts(candidate_rows))
    dump(args.out, result)
    print(json.dumps({k: result[k] for k in result if k != "per_sentence"}))
    if not result["pass"]:
        raise SystemExit(1)


def cmd_stock_check(args) -> None:
    """Re-translate a few sentences with the unmodified reference loader and implementation."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    corpus = {r["id"]: r for r in load_corpus(args.corpus)}
    _, rows = read_records(args.records)
    wanted, seen, picked = set(args.splits.split(",")), Counter(), []
    for row in rows:
        if row["split"] in wanted and seen[row["direction"]] < args.limit:
            picked.append(row)
            seen[row["direction"]] += 1
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = []
    for row in picked:
        ids = torch.tensor([corpus[row["id"]]["prompt_ids"]])
        with torch.inference_mode():
            output = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True)
        results.append({"id": row["id"], "engine": row["text"], "reference_runtime": text,
                        "text_equal": text == row["text"]})
    same = sum(r["text_equal"] for r in results)
    dump(args.out, {"model": str(Path(args.model).resolve()), "sentences": len(results), "text_equal": same,
                    "transformers": transformers.__version__, "attention": "eager",
                    "pass": same == len(results), "results": results})
    print(json.dumps({"sentences": len(results), "text_equal": same}))
    if same != len(results):
        raise SystemExit(1)


def cmd_report(args) -> None:
    # Which records folder holds which variant: a JSON file or inline, e.g. {"baseline": "baseline", "96": "k96"}.
    raw = args.variants.strip()
    variants = json.loads(Path(raw).read_text(encoding="utf-8") if Path(raw).is_file() else raw)
    if "baseline" not in variants:
        raise SystemExit("variants must name a baseline folder")
    from . import report
    built = report.build(Path(args.run_dir), variants, args.gate, args.spot_n)
    print(json.dumps({"rows": len(built["chrf_pp"]), "spot": len(built["spot_check_largest_test_drops"]),
                      "mechanism_pass": built["verdict"]["mechanism_pass"],
                      "quality": {k: v["test_pass"] for k, v in built["verdict"]["quality"].items()}}))


def add_command(commands, name: str, help: str, func, *options) -> None:
    """options: (flags, dest, kwargs) triples."""
    sub = commands.add_parser(name, help=help)
    for flags, dest, kwargs in options:
        sub.add_argument(*flags, dest=dest, **kwargs)
    sub.set_defaults(func=func)
    return sub


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mtpruning", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    required = {"required": True}

    add_command(commands, "corpus", "build the FLORES-200 sample", cmd_corpus,
                (["--tarball"], "tarball", required), (["--tokenizer"], "tokenizer", required),
                (["--out"], "out", required),
                (["--n-calibration"], "n_calibration", {"type": int, "default": 24}),
                (["--n-validation"], "n_validation", {"type": int, "default": 20}),
                (["--n-test"], "n_test", {"type": int, "default": 20}),
                (["--seed"], "seed", {"type": int, "default": 42}))
    add_command(commands, "run", "translate corpus sentences with one checkpoint", cmd_run,
                (["--model"], "model", required), (["--corpus"], "corpus", required),
                (["--out"], "out", required),
                (["--splits"], "splits", {"default": "calibration,validation,test"}),
                (["--max-new-tokens"], "max_new_tokens", {"type": int, "default": 640}),
                (["--threads"], "threads", {"type": int, "default": 16}),
                (["--tokenizer"], "tokenizer", {}),
                (["--mask"], "mask", {"help": "selection JSON: restrict routing to its kept experts"}),
                (["--mask-keep"], "mask_keep", {"type": int, "help": "candidate size inside a selection JSON"}),
                (["--stats"], "stats", {"help": "write routing statistics here (NPZ/JSON base name)"}),
                (["--stats-splits"], "stats_splits", {"default": "calibration"}),
                (["--trace-dir"], "trace_dir", {}),
                (["--trace-n"], "trace_n", {"type": int, "default": 0}),
                (["--trace-split"], "trace_split", {"default": "calibration"}),
                (["--limit"], "limit", {"type": int, "help": "sentences per split and direction (dry runs)"}))
    add_command(commands, "select", "direction-balanced expert selection from routing statistics", cmd_select,
                (["--model"], "model", required), (["--stats"], "stats", required),
                (["--keep"], "keep", required), (["--out"], "out", required))
    add_command(commands, "prune", "write a physically pruned checkpoint and verify the reload", cmd_prune,
                (["--model"], "model", required), (["--selection"], "selection", {}),
                (["--keep"], "keep", {**required, "type": int}),
                (["--out"], "out", required), (["--out-manifest"], "out_manifest", required))
    add_command(commands, "verify", "compare every tensor of a pruned checkpoint with its source", cmd_verify,
                (["--model"], "model", required), (["--pruned"], "pruned", required),
                (["--out"], "out", required))
    add_command(commands, "equiv", "compare two traced runs position by position", cmd_equiv,
                (["--reference-traces"], "reference_traces", required),
                (["--candidate-traces"], "candidate_traces", required),
                (["--reference-records"], "reference_records", required),
                (["--candidate-records"], "candidate_records", required), (["--out"], "out", required))
    add_command(commands, "stock-check", "re-translate with the unmodified reference implementation",
                cmd_stock_check,
                (["--model"], "model", required), (["--corpus"], "corpus", required),
                (["--records"], "records", required), (["--out"], "out", required),
                (["--splits"], "splits", {"default": "validation"}),
                (["--limit"], "limit", {"type": int, "default": 1}),
                (["--max-new-tokens"], "max_new_tokens", {"type": int, "default": 640}),
                (["--threads"], "threads", {"type": int, "default": 16}))
    add_command(commands, "report", "assemble report.json", cmd_report,
                (["--run-dir"], "run_dir", required),
                (["--variants"], "variants", {**required, "help": 'JSON object, e.g. {"baseline":"baseline","96":"k96"}'}),
                (["--gate"], "gate", {"type": float, "default": 1.0}),
                (["--spot-n"], "spot_n", {"type": int, "default": 3}))

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
