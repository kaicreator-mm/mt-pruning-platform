"""Assemble report.json and report.md from a finished run directory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .corpus import DIRECTIONS, load_corpus
from .evaluate import chrf_pp, failure_flags, paired_bootstrap, score_from_stats, segment_stats
from .spec import MODEL_REVISION

SPLITS_ORDER = [(split, d) for split in ("calibration", "validation", "test") for d in DIRECTIONS]
# Failure flags a candidate must not add to, next to the chrF++ gate (issue #2).
DIAGNOSTIC_FLAGS = ("empty", "truncated", "digits_differ_from_reference")
NOT_RUN = [
    "80 experts per layer (the issue lists 96/80/64; this pilot ran 96 and 64)",
    "same-size random pruning baseline and the 7B dense model as quality anchors",
    "GGUF at the same quantization, and peak memory / speed on the 16 GiB GPU",
    "human review of severe errors (the spot-check list is agent-selected, not human-judged)",
    "sampling decoding (the model card suggests temperature 0.7); everything here is greedy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [row for row in (json.loads(line) for line in handle if line.strip())
                if row.get("type") == "sentence"]


def score_rows(metric, rows: list[dict], corpus_by_id: dict) -> dict:
    """chrF++ statistics and failure flags of one (variant, split, direction) group, in corpus order."""
    contexts = [corpus_by_id[row["id"]] for row in rows]
    group = {"ids": [row["id"] for row in rows], "texts": [row["text"] for row in rows],
             "flags": {}, "finish": {}}
    group["stats"] = segment_stats(metric, group["texts"], [con["reference"] for con in contexts])
    for row, con in zip(rows, contexts):
        for flag in failure_flags(con["source"], con["reference"], row["text"], con["tgt_lang"],
                                  row["finish"], row["tokens"]):
            group["flags"][flag] = group["flags"].get(flag, 0) + 1
        group["finish"][row["finish"]] = group["finish"].get(row["finish"], 0) + 1
    group["scores"] = [score_from_stats(metric, [stats]) for stats in group["stats"]]
    return group


def quality_table(run: Path, variants: dict, gate: float, spot_n: int) -> tuple[list, list]:
    metric, corpus_by_id = chrf_pp(), {r["id"]: r for r in load_corpus(run / "corpus" / "corpus.jsonl")}
    groups = {name: {} for name in variants}
    for name, folder in variants.items():
        rows = read_records(run / folder / "records.jsonl")
        for key in sorted({(r["split"], r["direction"]) for r in rows}, key=SPLITS_ORDER.index):
            groups[name][key] = score_rows(metric, [r for r in rows if (r["split"], r["direction"]) == key],
                                           corpus_by_id)
    table, spot = [], []
    for split in ("validation", "test"):
        for direction in DIRECTIONS:
            key, base = (split, direction), groups["baseline"].get((split, direction))
            if base is None:
                continue
            row = {"split": split, "direction": direction, "n": len(base["ids"]),
                   "baseline_chrf_pp": score_from_stats(metric, base["stats"]),
                   "baseline_flags": base["flags"], "baseline_finish": base["finish"], "candidates": {}}
            for name in variants:
                g = groups[name].get(key)
                if name == "baseline" or g is None:
                    continue
                score = score_from_stats(metric, g["stats"])
                boot = paired_bootstrap(metric, base["stats"], g["stats"])
                deltas = np.array(g["scores"]) - np.array(base["scores"])
                row["candidates"][name] = {
                    "chrf_pp": score, "delta": score - row["baseline_chrf_pp"],
                    "delta_ci95": [boot["ci95_low"], boot["ci95_high"]],
                    "identical_to_baseline": sum(a == b for a, b in zip(base["texts"], g["texts"])),
                    "sentences_better": int((deltas > 1e-9).sum()), "sentences_worse": int((deltas < -1e-9).sum()),
                    "worst_sentence_delta": float(deltas.min()), "flags": g["flags"], "finish": g["finish"],
                    "gate_pass": bool(score - row["baseline_chrf_pp"] <= gate),
                    "diagnostics_not_worse": all(g["flags"].get(f, 0) <= base["flags"].get(f, 0)
                                                 for f in DIAGNOSTIC_FLAGS),
                }
                if split == "test":
                    for i in np.argsort(deltas)[:spot_n]:
                        con = corpus_by_id[g["ids"][i]]
                        spot.append({"variant": name, "direction": direction, "id": g["ids"][i],
                                     "sentence_chrf_pp_delta": float(deltas[i]), "source": con["source"],
                                     "reference": con["reference"], "baseline": base["texts"][i],
                                     "candidate": g["texts"][i]})
            table.append(row)
    return table, spot


def mechanism_checks(run: Path) -> dict:
    """Per candidate folder: prune + reload verification and the trace comparison."""
    checks = {}
    for path in sorted(run.glob("k*/equiv.json")):
        folder = path.parent
        equiv = {k: v for k, v in load_json(path).items() if k != "per_sentence"}
        prune = load_json(folder / "prune.json") if (folder / "prune.json").exists() else {}
        manifest, verify = prune.get("manifest", {}), prune.get("verify", {})
        stock = folder / "stock_check.json"
        checks[folder.name] = {
            "keep": manifest.get("num_experts"), "identity_mapping": verify.get("identity_mapping"),
            "kind": "identity no-op" if verify.get("identity_mapping") else "masked equivalence",
            "tensors": manifest.get("tensors"), "tensor_bytes": manifest.get("tensor_bytes"),
            "shards": [{"name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in manifest.get("files", [])],
            "verify_pass": verify.get("pass"), "equiv": equiv,
            "stock_check": ({k: v for k, v in load_json(stock).items() if k != "results"} if stock.exists() else None),
        }
    return checks


def decide(checks: dict, table: list, gate: float) -> dict:
    noop = [c for c in checks.values() if c["identity_mapping"]]
    masked = {name: c for name, c in checks.items() if not c["identity_mapping"]}
    noop_pass = bool(noop) and all(c["verify_pass"] and c["equiv"]["pass"] for c in noop)
    masked_pass = {name: bool(c["verify_pass"] and c["equiv"]["pass"]) for name, c in masked.items()}
    names = sorted({name for row in table for name in row["candidates"]}, key=lambda s: -int(s))
    test_rows = [row for row in table if row["split"] == "test"]
    val_rows = [row for row in table if row["split"] == "validation"]
    quality = {}
    for name in names:
        quality[name] = {
            "test_pass": all(r["candidates"][name]["gate_pass"] for r in test_rows if name in r["candidates"]),
            "test_failing": [r["direction"] for r in test_rows if name in r["candidates"]
                             and not r["candidates"][name]["gate_pass"]],
            "test_diagnostics_failing": [r["direction"] for r in test_rows if name in r["candidates"]
                                         and not r["candidates"][name]["diagnostics_not_worse"]],
            "validation_pass": all(r["candidates"][name]["gate_pass"] for r in val_rows if name in r["candidates"]),
            "validation_failing": [r["direction"] for r in val_rows if name in r["candidates"]
                                   and not r["candidates"][name]["gate_pass"]],
            "test_delta": {r["direction"]: r["candidates"][name]["delta"] for r in test_rows if name in r["candidates"]},
            "test_delta_ci95": {r["direction"]: r["candidates"][name]["delta_ci95"]
                                for r in test_rows if name in r["candidates"]},
        }
    mechanism_pass = noop_pass and bool(masked_pass) and all(masked_pass.values())
    return {
        "gate": f"chrF++ drop <= {gate} absolute per direction (provisional; pilot sample, not production acceptance)",
        "noop_pass": noop_pass, "masked_pass": masked_pass, "mechanism_pass": mechanism_pass,
        "quality": quality,
        "diagnostics_rule": "no direction may have more empty, truncated or digit-mismatch flags than the baseline",
        "summary": ("mechanism PASS" if mechanism_pass else "mechanism FAIL")
                   + "; " + "; ".join(f"{name} experts: chrF++ gate {'PASS' if q['test_pass'] else 'FAIL'}, "
                                      f"failure diagnostics {'PASS' if not q['test_diagnostics_failing'] else 'FAIL'}"
                                      for name, q in quality.items()),
    }


def build(run_dir: Path, variants: dict, gate: float, spot_n: int) -> dict:
    run = Path(run_dir)
    table, spot = quality_table(run, variants, gate, spot_n)
    checks = mechanism_checks(run)
    baseline = load_json(run / "baseline" / "records.jsonl.summary.json")
    corpus_manifest = load_json(run / "corpus" / "corpus_manifest.json")
    report = {
        "generated": utc_now(),
        "scope": "D01b pilot: real Hy-MT3-30B-A3B weights, six zh/en/my directions, FLORES-200 "
                 "24/20/20 sentences per direction (calibration/validation/test)",
        "metric": "sacrebleu CHRF(word_order=2), default settings; deltas are candidate minus baseline",
        "environment": baseline.get("environment"), "decoding": "greedy, max_new_tokens=%s" % baseline.get("max_new_tokens"),
        "corpus": {k: corpus_manifest.get(k) for k in
                   ("corpus", "release", "license", "url", "tarball_sha256", "corpus_sha256", "seed",
                    "sizes_per_direction", "split_sizes", "language_codes", "duplicate_rule", "prompt_policy",
                    "records")},
        "verdict": decide(checks, table, gate),
        "mechanism_checks": checks,
        "calibration": baseline.get("routing"),
        "selection": load_json(run / "selection.json") if (run / "selection.json").exists() else None,
        "mask_audit": {p.parent.parent.name: {k: v for k, v in load_json(p).items() if k == "mask_audit"}
                       for p in sorted(run.glob("k*/masked/records.jsonl.summary.json"))},
        "run_timing": {p.parent.name: {k: v for k, v in load_json(p).items()
                        if k in ("sentences", "new_tokens", "prefill_seconds", "decode_seconds",
                                 "decode_tokens_per_second", "finish")}
                       for p in sorted(run.glob("*/records.jsonl.summary.json"))},
        "chrf_pp": table, "spot_check_largest_test_drops": spot, "not_run": NOT_RUN,
    }
    artifacts = {}
    for path in sorted(run.rglob("*.json")):
        if path.name == "report.json" or "traces" in path.parts:
            continue
        data = load_json(path)
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "per_sentence"}
        artifacts[str(path.relative_to(run)).replace("/", "__").removesuffix(".json")] = data
    report["artifacts"] = artifacts
    (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8", newline="\n")
    (run / "report.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def fmt(value, digits=2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def arrow(direction: str) -> str:
    src, tgt = direction.split("-")
    return f"{src}→{tgt}"


def render_markdown(report: dict) -> str:
    verdict, checks, table = report["verdict"], report["mechanism_checks"], report["chrf_pp"]
    names = list(verdict["quality"])
    env = report.get("environment") or {}
    out = [ "# D01b real-weight validation: expert pruning of Hy-MT3-30B-A3B", "",
            f"Generated {report['generated']}. {report['scope']}.", "",
            "## Verdict", "",
            f"- Mechanism (128 identity reload = no-op, and physical prune = masked original): "
            f"**{'PASS' if verdict['mechanism_pass'] else 'FAIL'}**",
            f"- Provisional quality gate, {verdict['gate']}:" ]
    for name, q in verdict["quality"].items():
        out.append(f"  - {name} experts: **{'PASS' if q['test_pass'] else 'FAIL'}** on the pilot test set"
                   f" (validation: {'PASS' if q['validation_pass'] else 'FAIL'})")
        failing = ", ".join(arrow(d) for d in q["test_diagnostics_failing"])
        out.append(f"  - {name} experts, no new empty / truncated / digit-mismatch flags in any direction: "
                   f"**{'PASS' if not failing else 'FAIL'}**" + (f" ({failing})" if failing else ""))
    out += [ "", "Small sample: these numbers cannot support a production-acceptance claim.", "",
             "## Inputs and environment", "",
             f"- Model: `tencent/Hy-MT3-30B-A3B` at revision `{MODEL_REVISION}`; every shard sha256-verified against the HF listing.",
             f"- Corpus: {report['corpus']['corpus']} ({report['corpus']['license']}), codes "
             f"{report['corpus']['language_codes']}. Duplicate rule: {report['corpus']['duplicate_rule']}. "
             f"Tarball sha256 "
             f"`{report['corpus']['tarball_sha256']}`, sampled corpus sha256 `{report['corpus']['corpus_sha256']}`; "
             f"{report['corpus']['sizes_per_direction']} sentences per direction (seed {report['corpus']['seed']}).",
             f"- Prompt policy: {report['corpus']['prompt_policy']}.",
             f"- Runtime: Python {env.get('python')}, torch {env.get('torch')}, transformers {env.get('transformers')}, "
             f"sacrebleu {env.get('sacrebleu')}; {env.get('torch_threads')} threads on {env.get('cpu_count')} logical CPUs "
             f"({env.get('platform')}).",
             f"- Decoding: {report.get('decoding')}. Metric: {report['metric']}.", "" ]

    out += ["## Mechanism checks", "",
            "| check | keep | sentences | identical text | argmax agree | route ids agree | within tol | bitwise | max abs Δlogit | max ULP | pass |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(checks, key=lambda s: -int(s[1:])):
        c, e = checks[name], checks[name]["equiv"]
        out.append(f"| {c['kind']} | {c['keep']} | {e['sentences']} | {e['identical_text']} | {e['all_argmax_agree']} | "
                   f"{e['all_route_ids_agree']} | {e['within_tol']} | {e['bitwise_identical_logits']} | "
                   f"{e['logits_max_abs_diff']:.4g} | {e['logits_max_ulp']} | {'PASS' if e['pass'] else 'FAIL'} |")
    out += [ "", f"\"Within tol\" = max |Δlogit| ≤ {next(iter(checks.values()))['equiv']['abs_tol'] if checks else 'n/a'} "
             "(two bf16 steps where the top logits sit); bitwise = every compared logit identical.", "" ]

    if report["mask_audit"]:
        out += ["## What the mask changes", "",
                "Routing decisions (token, layer) that differ from the unmasked model on the same hidden states, "
                "and the routing weight it loses.", "",
                "| keep | direction | phase | decisions | changed | changed % | lost weight |",
                "|---|---|---|---|---|---|---|"]
        for name, audit in report["mask_audit"].items():
            audit = audit.get("mask_audit") or {}
            for direction, phases in audit.items():
                for phase, values in phases.items():
                    out.append(f"| {name[1:]} | {arrow(direction)} | {phase} | {values['decisions']} | {values['changed']} | "
                               f"{100 * values['changed_fraction']:.3f}% | {values['lost_weight_mass']:.1f} |")
        out.append("")

    out += ["## chrF++ per direction", "", "| direction | split | n | baseline | " +
             " | ".join(f"{name} exp. Δ [95% CI] (identical/n; +better/−worse)" for name in names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for row in table:
        cells = []
        for name in names:
            c = row["candidates"].get(name)
            cells.append("n/a" if c is None else
                         f"{c['delta']:+.2f} [{c['delta_ci95'][0]:+.2f}, {c['delta_ci95'][1]:+.2f}] "
                         f"({c['identical_to_baseline']}/{row['n']}; +{c['sentences_better']}/−{c['sentences_worse']})")
        out.append(f"| {arrow(row['direction'])} | {row['split']} | {row['n']} | {row['baseline_chrf_pp']:.2f} | "
                   + " | ".join(cells) + " |")
    out += [ "", "Δ is candidate minus baseline in chrF++ points; CI from a paired bootstrap over sentences "
             "(2000 resamples, seed 42). \"identical\" counts outputs equal to the baseline text.", "" ]

    flags = sorted({f for row in table for g in [row["baseline_flags"]] + [c["flags"] for c in row["candidates"].values()]
                    for f in g})
    if flags:
        out += ["## Failure evidence (heuristic flags)", "",
                "| direction | split | variant | " + " | ".join(flags) + " |", "|---|---|---|" + "---|" * len(flags)]
        for row in table:
            groups = [("baseline", row["baseline_flags"])] + [(f"{n} exp.", c["flags"]) for n, c in row["candidates"].items()]
            for label, counts in groups:
                out.append(f"| {arrow(row['direction'])} | {row['split']} | {label} | "
                           + " | ".join(str(counts.get(f, 0)) for f in flags) + " |")
        out += [ "", "Flags point at sentences to look at; they are not error verdicts (a correct translation can "
                 "legitimately write a number differently from the single reference).", "" ]

    if report["spot_check_largest_test_drops"]:
        out += ["## Sentences to look at first (largest per-sentence chrF++ drops on the pilot test set)", "",
                "Source and reference sentences: FLORES-200 (Meta AI), CC-BY-SA 4.0.", ""]
        for item in report["spot_check_largest_test_drops"]:
            out += [f"**{item['variant']} experts, {arrow(item['direction'])}, {item['id']}, Δ {item['sentence_chrf_pp_delta']:+.2f}**",
                    f"- source: {item['source']}", f"- reference: {item['reference']}",
                    f"- baseline: {item['baseline']}", f"- candidate: {item['candidate']}", ""]
    out += ["## Not run in this task", ""] + [f"- {item}" for item in report["not_run"]] + [""]
    return "\n".join(out)
