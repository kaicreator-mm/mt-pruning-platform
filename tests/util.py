"""Shared fixtures of the pipeline-level tests.

A dry run mirrors the real protocol at toy scale: a synthetic checkpoint with the
real expert counts (128 experts, top-8) and the released tokenizer, and a small
FLORES-200 sample with the same structure as the frozen corpus.

    MTPRUNING_TOKENIZER_DIR=<checkpoint dir with tokenizer.json> MTPRUNING_FLORES_TARBALL=<tarball>
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

TOKENIZER = os.environ.get("MTPRUNING_TOKENIZER_DIR")
TARBALL = os.environ.get("MTPRUNING_FLORES_TARBALL")
READY = bool(TOKENIZER and TARBALL)
REPO = Path(__file__).resolve().parents[1]
SIZES = {"n_calibration": 4, "n_validation": 3, "n_test": 3}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_world(root: Path, seed=11) -> dict:
    """A run directory holding a corpus, plus a synthetic 128-expert checkpoint with the released tokenizer."""
    from transformers import AutoTokenizer
    from mtpruning.corpus import build_corpus
    from mtpruning.testing import tiny_checkpoint
    root = Path(root)
    run = root / "run"
    (run / "corpus").mkdir(parents=True)
    model = tiny_checkpoint(root / "model", TOKENIZER, seed=seed, num_experts=128, top_k=8)
    build_corpus(TARBALL, AutoTokenizer.from_pretrained(TOKENIZER), run / "corpus", **SIZES)
    return {"root": root, "run": run, "model": model, "corpus": run / "corpus" / "corpus.jsonl",
            "manifest_sha256": sha256(run / "corpus" / "corpus_manifest.json"), "cache": root / "cache"}


def options(world: dict, **overrides):
    """Baseline-style run options: calibration + validation, statistics, two traced sentences per direction."""
    from mtpruning.pipeline import RunOptions
    base = dict(model=str(world["model"]), corpus=str(world["corpus"]),
                out=str(world["run"] / "baseline" / "records.jsonl"), splits=["calibration", "validation"],
                max_new_tokens=6, threads=2, stats=str(world["run"] / "baseline" / "stats"),
                trace_dir=str(world["run"] / "baseline" / "traces"), trace_n=2, trace_split="validation",
                manifest_sha256=world["manifest_sha256"], cache_dir=str(world["cache"]))
    base.update(overrides)
    return RunOptions(**base)


def cli(world: dict, *arguments) -> None:
    """Run one pipeline stage in this process, with the test's corpus pinned."""
    from mtpruning import pipeline
    pipeline.CRASH_AFTER = 0
    pipeline.main([*[str(a) for a in arguments], "--manifest-sha256", world["manifest_sha256"]])


def rehearse(world: dict, candidates=(96, 64), trace=2, stock_check=True) -> None:
    """Take a test world through every stage up to the provisional decision, like the driver."""
    from mtpruning import pipeline
    from mtpruning.pipeline import prune_candidate
    run, corpus, cache = world["run"], str(world["corpus"]), str(world["cache"])
    pipeline.execute_run(options(world))
    cli(world, "select", "--run-dir", run, "--keep", ",".join(map(str, candidates)), "--out", run / "selection.json")

    def mechanism(keep, model, kind, reference_folder, selection):
        folder = run / f"k{keep}"
        prune_candidate(str(world["model"]), str(run / "selection.json") if selection else None, keep,
                        world["root"] / "pruned" / f"k{keep}", folder / "prune.json", cache)
        pipeline.execute_run(options(world, model=str(model), out=folder / "mechanism" / "records.jsonl",
                                     splits=["validation"], limit=trace, stats=None,
                                     trace_dir=folder / "mechanism" / "traces", trace_n=trace))
        arguments = ["equiv", "--kind", kind, "--keep", keep, "--run-dir", run,
                     "--prune-record", folder / "prune.json", "--trace-n", trace,
                     "--candidate-traces", folder / "mechanism" / "traces",
                     "--candidate-records", folder / "mechanism" / "records.jsonl",
                     "--reference-traces", reference_folder / "traces",
                     "--reference-records", reference_folder / "records.jsonl",
                     "--out", folder / "equiv.json"]
        cli(world, *arguments, *(["--selection", run / "selection.json"] if selection else []))

    mechanism(128, world["root"] / "pruned" / "k128", "noop", run / "baseline", False)
    for keep in candidates:
        folder = run / f"k{keep}"
        pipeline.execute_run(options(world, out=folder / "masked" / "records.jsonl", splits=["validation"], limit=trace,
                                     stats=None, trace_dir=folder / "masked" / "traces", trace_n=trace,
                                     mask=run / "selection.json", mask_keep=keep))
        mechanism(keep, world["root"] / "pruned" / f"k{keep}", "masked", folder / "masked", True)
        pipeline.execute_run(options(world, model=world["root"] / "pruned" / f"k{keep}", out=folder / "records.jsonl",
                                     splits=["validation"], stats=None, trace_n=0))
        if stock_check:
            cli(world, "stock-check", "--model", world["root"] / "pruned" / f"k{keep}", "--corpus", corpus,
                "--records", folder / "records.jsonl", "--out", folder / "stock_check.json", "--splits", "validation",
                "--limit", 1, "--max-new-tokens", 6, "--threads", 2)
    cli(world, "decide", "--run-dir", run, "--candidates", ",".join(map(str, candidates)), "--out", run / "decision.json")


def driver(world: dict, environment: dict | None = None) -> subprocess.CompletedProcess:
    """Run scripts/run_d01b.sh on the test world in a subprocess."""
    env = {**os.environ, "PY": sys.executable, "ROOT": str(world["root"]), "RUN": str(world["run"]),
           "PRUNED": str(world["root"] / "pruned"), "CACHE_DIR": str(world["cache"]),
           "EVIDENCE_DIR": str(world["root"] / "evidence"), "MODEL_DIR": str(world["model"]),
           "TOKENIZER_DIR": TOKENIZER, "MANIFEST_SHA256": world["manifest_sha256"], "SKIP_WAIT": "1",
           "TRACE_LIMIT": "2", "MAX_NEW": "6", "THREADS": "2", "MODE": "rehearsal", **(environment or {})}
    return subprocess.run(["bash", str(REPO / "scripts" / "run_d01b.sh")], env=env, capture_output=True,
                          text=True, cwd=REPO, timeout=1800)
