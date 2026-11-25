"""Assemble report.json and report.md from a run directory.

Every quality statement goes through the reviewed r6 gate, which refuses a
missing or duplicated direction, an incomplete group, a non-finite score or an
inconsistent delta. A candidate whose records do not completely cover a split is
NOT_EVALUATED on that split; it is never PASS. The test split appears only after
the final-test stage ran for the reviewed candidate.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .corpus import DIRECTIONS
from .evaluate import chrf_pp, failure_flags, paired_bootstrap, score_from_stats, segment_stats
from .frozen import BlockedCorpus, load_frozen_corpus
from .pipeline import read_json, read_run, sha256_file, write_text_atomic
from .reviewed import bridge, handoff
from .spec import CORPUS_MANIFEST_SHA256, DECODING, MODEL_REVISION

# Failure flags a candidate must not add to, next to the chrF++ gate.
DIAGNOSTIC_FLAGS = ("empty", "truncated", "digits_differ_from_reference")
HELD = "the test split is held until the selection identity and evidence are independently reviewed"
NOT_RUN = [
    f"the test split for every candidate ({HELD})",
    "80 experts per layer (the issue lists 96/80/64; this pilot covers 96 and 64)",
    "same-size random pruning baseline and the 7B dense model as quality anchors",
    "GGUF at the same quantization, and peak memory / speed on the 16 GiB GPU",
    "human review of severe errors (the spot-check list is selected by a rule, not judged by a person)",
    "sampling decoding (the model card suggests temperature 0.7); everything here is greedy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def passes_gate(delta, gate) -> bool:
    """True when chrF++ dropped by at most ``gate`` points: candidate minus baseline >= -gate.

    A larger drop fails, an improvement of any size passes. So does anything that is
    not a finite number, and any gate that is negative or not finite.
    """
    return (type(delta) in (int, float) and math.isfinite(delta) and type(gate) in (int, float)
            and math.isfinite(gate) and gate >= 0 and delta >= -gate)


def no_new_failures(candidate: dict, baseline: dict) -> bool:
    """No more empty, truncated or digit-mismatch outputs than the baseline."""
    return all(candidate.get(flag, 0) <= baseline.get(flag, 0) for flag in DIAGNOSTIC_FLAGS)


def score_group(metric, records: dict, contexts: dict, ids: list[str]) -> dict:
    """chrF++ statistics of one (split, direction) group, aligned by ID."""
    group = [contexts[i] for i in ids]
    texts = [records[i]["text"] for i in ids]
    flags = Counter()
    for sample_id, context in zip(ids, group):
        for flag in failure_flags(context["source"], context["reference"], records[sample_id]["text"],
                                  context["tgt_lang"], records[sample_id]["finish"], records[sample_id]["tokens"]):
            flags[flag] += 1
    stats = segment_stats(metric, texts, [context["reference"] for context in group])
    return {"ids": ids, "texts": texts, "stats": stats, "flags": dict(flags),
            "scores": [score_from_stats(metric, [row]) for row in stats]}


def validation_records(manifest, contexts, path, keep, candidate) -> dict | None:
    """Validation records of a run, checked by the reviewed r7 loader; None when the run does not exist.

    Raises ValueError when validation coverage is incomplete, when the run is not
    the unmasked checkpoint it claims to be, or when a candidate saw the test split.
    """
    if read_run(path)[0] is None:
        return None
    return handoff.records(path, manifest, contexts, keep, candidate=candidate)[1]


def read_json_lines(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_records(manifest, contexts, path, keep) -> dict:
    """Test records of a run with exactly the expected coverage.

    The reviewed r7 loader is validation-only by design and never accepts a test
    record, so its checks are repeated here for the test split.
    """
    need, out, seen = handoff.need, {}, set()
    rows = read_json_lines(path)
    need(rows and rows[0].get("type") == "run", "missing run header")
    header = rows[0]
    need(header.get("corpus_sha256") == manifest["corpus_sha256"] and header.get("num_experts") == keep
         and "masked_keep" in header and header["masked_keep"] is None, "wrong run identity")
    need(header.get("splits") == ["test"], "not a test-only run")
    need(type(header.get("max_new_tokens")) is int and header["max_new_tokens"] > 0
         and header.get("decoding") == DECODING, "bad decoding contract")
    for row in rows[1:]:
        need(row.get("type") == "sentence" and row.get("id") in contexts and row["id"] not in seen,
             "invalid/duplicate run record")
        seen.add(row["id"])
        context = contexts[row["id"]]
        need(context["split"] == "test" and all(row.get(k) == context[k] for k in ("split", "direction", "line")),
             "run sample mismatch")
        need(row.get("prompt_tokens") == len(context["prompt_ids"]) and isinstance(row.get("tokens"), list)
             and all(type(t) is int and t >= 0 for t in row["tokens"]), "bad token record")
        need(type(row.get("new_tokens")) is int and len(row["tokens"]) == row["new_tokens"] <= header["max_new_tokens"]
             and row.get("finish") in ("eos", "length") and isinstance(row.get("text"), str), "bad generation record")
        need(row["finish"] != "length" or row["new_tokens"] == header["max_new_tokens"], "length finish inconsistent")
        out[row["id"]] = row
    need(set(out) == {i for i, c in contexts.items() if c["split"] == "test"}, "incomplete test records")
    return out


def gate(table: list, split: str, candidate: str, expected_n: int, max_drop: float) -> dict:
    """The reviewed gate on report rows. Whatever stops it from deciding is NOT_EVALUATED, never PASS."""
    try:
        return bridge.quality_gate(table, split, candidate, expected_n, max_drop)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return {"status": "NOT_EVALUATED", "reason": f"{type(error).__name__}: {error}"}


def outcome(verdict: dict) -> dict:
    """A gate result normalised for the report: status, failing directions, reason.

    NOT_EVALUATED (the rows do not cover the split) and NOT_RUN (the test split was
    held) have no directions, and they are never a pass.
    """
    if not str(verdict.get("status", "")).startswith("REPORT_GATE_"):
        return {"status": verdict.get("status", "NOT_EVALUATED"), "failing": [], "reason": verdict.get("reason")}
    directions = verdict["directions"]
    failed = {name: [d for d, v in directions.items() if not v[flag]]
              for name, flag in (("quality", "quality_pass"), ("diagnostics", "diagnostics_pass"))}
    failed["any"] = [d for d, v in directions.items() if not (v["quality_pass"] and v["diagnostics_pass"])]
    return {"status": verdict["status"], "failing": failed["any"], "failing_score": failed["quality"],
            "failing_diagnostics": failed["diagnostics"], "directions": directions}


def table_rows(metric, groups: dict, contexts: dict, split: str, max_drop: float, spot: list | None = None) -> list:
    """Report rows of one split in the layout bridge.quality_gate reads.

    ``groups`` maps variant name to records by ID and must cover the split exactly
    for every variant; rows are aligned by ID.
    """
    table = []
    for direction in DIRECTIONS:
        base = groups["baseline"]
        ids = sorted(i for i in base if contexts[i]["direction"] == direction)
        group = score_group(metric, base, contexts, ids)
        row = {"split": split, "direction": direction, "n": len(ids),
               "baseline_chrf_pp": score_from_stats(metric, group["stats"]), "baseline_flags": group["flags"],
               "candidates": {}}
        for name, records in groups.items():
            if name == "baseline":
                continue
            handoff.need(all(i in records for i in ids), f"{name} has no {split} output for every sentence of {direction}")
            other = score_group(metric, records, contexts, ids)
            score = score_from_stats(metric, other["stats"])
            boot = paired_bootstrap(metric, group["stats"], other["stats"])
            deltas = np.array(other["scores"]) - np.array(group["scores"])
            row["candidates"][name] = {
                "chrf_pp": score, "delta": score - row["baseline_chrf_pp"],
                "delta_ci95": [boot["ci95_low"], boot["ci95_high"]],
                "identical_to_baseline": sum(a == b for a, b in zip(group["texts"], other["texts"])),
                "sentences_better": int((deltas > 1e-9).sum()), "sentences_worse": int((deltas < -1e-9).sum()),
                "worst_sentence_delta": float(deltas.min()), "flags": other["flags"],
                "gate_pass": passes_gate(score - row["baseline_chrf_pp"], max_drop),
                "diagnostics_not_worse": no_new_failures(other["flags"], group["flags"]),
            }
            for i in (np.argsort(deltas)[:spot.capacity] if spot is not None else []):
                spot.append({"variant": name, "split": split, "direction": direction, "id": ids[i],
                             "sentence_chrf_pp_delta": float(deltas[i]), "source": contexts[ids[i]]["source"],
                             "reference": contexts[ids[i]]["reference"], "baseline": group["texts"][i],
                             "candidate": other["texts"][i]})
        table.append(row)
    return table


class SpotList(list):
    """Collects the largest per-sentence drops; ``capacity`` is the number kept per direction and variant."""

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity


def agree_with_reviewed_gate(table: list, verdict: dict, candidate: str) -> None:
    """The row flags must say what the reviewed gate says; a disagreement is a bug, not a result."""
    if not str(verdict.get("status", "")).startswith("REPORT_GATE_"):
        return
    rows = {row["direction"]: row for row in table}
    for direction, result in verdict["directions"].items():
        flags = rows[direction]["candidates"][candidate]
        if flags["gate_pass"] != result["quality_pass"] or flags["diagnostics_not_worse"] != result["diagnostics_pass"]:
            raise RuntimeError(f"report flags for {candidate} on {direction} disagree with the reviewed gate")


def test_used_outside_final_test(run: Path, baseline_keep: int) -> list[str]:
    """Runs other than the final-test stage whose records include the test split."""
    found = []
    for path in sorted(run.rglob("records.jsonl")):
        if "final_test" in path.relative_to(run).parts:
            continue
        header = read_run(path)[0]
        if header and "test" in header.get("splits", []):
            found.append(f"{path.relative_to(run)} (masked_keep={header.get('masked_keep')}, "
                         f"num_experts={header.get('num_experts')}, baseline has {baseline_keep})")
    return found


def mechanism_checks(run: Path) -> dict:
    """Per candidate folder: prune, verification of the reload, trace comparison, reference-runtime check."""
    checks = {}
    for path in sorted(run.glob("k*/equiv.json")):
        folder = path.parent
        prune = read_json(folder / "prune.json") if (folder / "prune.json").exists() else {}
        manifest, verify = prune.get("manifest", {}), prune.get("verify", {})
        stock = folder / "stock_check.json"
        checks[folder.name] = {
            "keep": manifest.get("num_experts"), "identity_mapping": verify.get("identity_mapping"),
            "kind": "identity no-op" if verify.get("identity_mapping") else "masked equivalence",
            "tensors": manifest.get("tensors"), "tensor_bytes": manifest.get("tensor_bytes"),
            "shards": [{"name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in manifest.get("files", [])],
            "verify_pass": verify.get("pass"),
            "equiv": {k: v for k, v in read_json(path).items() if k != "per_sentence"},
            "stock_check": ({k: v for k, v in read_json(stock).items() if k != "results"} if stock.exists() else None),
        }
    return checks


def decide(checks: dict, quality: dict, max_drop: float) -> dict:
    noop = [c for c in checks.values() if c["identity_mapping"]]
    masked = {name: c for name, c in checks.items() if not c["identity_mapping"]}
    noop_pass = bool(noop) and all(c["verify_pass"] and c["equiv"]["pass"] for c in noop)
    masked_pass = {name: bool(c["verify_pass"] and c["equiv"]["pass"]) for name, c in masked.items()}
    lines = [f"mechanism {'PASS' if noop_pass and masked_pass and all(masked_pass.values()) else 'FAIL'} "
             f"(identity reload {'PASS' if noop_pass else 'FAIL'}; masked equivalence "
             + (", ".join(f"{n[1:]}: {'PASS' if ok else 'FAIL'}" for n, ok in masked_pass.items()) or "NOT_RUN") + ")"]
    for name, q in quality.items():
        lines.append(f"{name} experts: validation {q['validation']['status']}, test {q['test']['status']}")
    return {
        "noop_pass": noop_pass, "masked_pass": masked_pass,
        "mechanism_pass": noop_pass and bool(masked_pass) and all(masked_pass.values()),
        "gate": f"per direction, chrF++ drop <= {max_drop} point and no more empty, truncated or digit-mismatch "
                "outputs than the baseline",
        "quality": quality, "quality_result": "HOLD: not a quality result until the final test is reviewed",
        "production_acceptance": False, "summary": "; ".join(lines),
    }


def build(run_dir: Path, manifest_sha256=None, max_drop: float = 1.0, spot_n: int = 3) -> dict:
    run = Path(run_dir)
    if not (type(max_drop) in (int, float) and math.isfinite(max_drop) and max_drop >= 0):
        raise SystemExit("max_drop must be a finite, non-negative number")
    try:
        manifest, contexts = load_frozen_corpus(run / "corpus", manifest_sha256 or CORPUS_MANIFEST_SHA256)
    except BlockedCorpus as error:
        raise SystemExit(f"{error.code}: {error}") from error
    metric, sizes = chrf_pp(), manifest["sizes_per_direction"]
    baseline_path = run / "baseline" / "records.jsonl"
    source, _ = read_run(baseline_path)
    if source is None:
        raise SystemExit("no baseline run to report on")
    baseline_keep = source["num_experts"]
    selection = read_json(run / "selection.json") if (run / "selection.json").exists() else None
    keeps = sorted({int(k) for k in selection["candidates"]} if selection else
                   {int(p.parent.name[1:]) for p in run.glob("k*/records.jsonl")}, reverse=True)

    # Validation: the baseline and every candidate, through the reviewed validation-only loader.
    problems, validation = {}, {}
    try:
        validation["baseline"] = validation_records(manifest, contexts, baseline_path, baseline_keep, candidate=False)
    except ValueError as error:
        validation["baseline"], problems["baseline"] = None, f"baseline run: {error}"
    for keep in keeps:
        try:
            validation[str(keep)] = validation_records(manifest, contexts, run / f"k{keep}" / "records.jsonl",
                                                       keep, candidate=True)
            if validation[str(keep)] is None:
                problems[str(keep)] = "no validation run"
        except ValueError as error:
            validation[str(keep)], problems[str(keep)] = None, str(error)
    spot = SpotList(spot_n)
    table_validation = []
    if validation["baseline"]:
        groups = {"baseline": validation["baseline"],
                  **{name: records for name, records in validation.items() if name != "baseline" and records}}
        # A candidate with an unusable run gets no row entry at all: the reviewed gate then
        # refuses the incomplete group, and the verdict names the reason.
        table_validation = table_rows(metric, groups, contexts, "validation", max_drop, spot)

    # Test: only what the final-test stage produced, and only for a completed entry of the exposure ledger.
    ledger = read_json(run / "test_exposure.json") if (run / "test_exposure.json").exists() else {"entries": []}
    table_test, tested, test_notes = [], set(), {}
    for entry in ledger["entries"]:
        name, keep = str(entry["keep"]), entry["keep"]
        if entry["status"] != "completed":
            test_notes[name] = "the final test did not complete"
            continue
        try:
            groups = {"baseline": test_records(manifest, contexts, run / "final_test" / "baseline" / "records.jsonl",
                                               baseline_keep),
                      name: test_records(manifest, contexts, run / "final_test" / f"k{keep}" / "records.jsonl", keep)}
            table_test = table_rows(metric, groups, contexts, "test", max_drop)
            tested.add(name)
        except ValueError as error:
            test_notes[name] = str(error)

    quality = {}
    for keep in keeps:
        name = str(keep)
        validation_gate = gate(table_validation, "validation", name, sizes["validation"], max_drop)
        test_gate = (gate(table_test, "test", name, sizes["test"], max_drop) if name in tested
                     else {"status": "NOT_RUN", "reason": HELD})
        agree_with_reviewed_gate(table_validation, validation_gate, name)
        agree_with_reviewed_gate(table_test, test_gate, name)
        quality[name] = {"max_drop": max_drop,
                         "validation": {**outcome(validation_gate),
                                        "problem": problems.get(str(keep)) or problems.get("baseline")},
                         "test": {**outcome(test_gate), "problem": test_notes.get(name)}}

    summary = read_json(run / "baseline" / "records.jsonl.summary.json")
    timing = {}
    for path in sorted(run.rglob("records.jsonl.summary.json")):
        data = read_json(path)
        timing[path.parent.relative_to(run).as_posix()] = {k: data.get(k) for k in (
            "sentences", "new_tokens", "finish", "masked_keep", "splits", "synthetic", "identity", "identity_keys")}
    outside = test_used_outside_final_test(run, baseline_keep)
    checks = mechanism_checks(run)
    report = {
        "generated": utc_now(),
        "scope": "D01b pilot: Hy-MT3-30B-A3B weights, six zh/en/my directions, frozen FLORES-200 corpus "
                 "(24/20/20 sentences per direction for calibration/validation/test)",
        "metric": "sacrebleu CHRF(word_order=2), default settings; deltas are candidate minus baseline",
        "synthetic": source.get("synthetic"),
        "checkpoint": {"kind": source.get("checkpoint_kind"), "teacher": source["identity"]["teacher"],
                       "model": source["model"]},
        "environment": summary.get("environment"),
        "decoding": f"{source['decoding']}, max_new_tokens={source['max_new_tokens']}",
        "corpus": {k: manifest.get(k) for k in (
            "corpus", "release", "license", "url", "tarball_sha256", "corpus_sha256", "seed", "sizes_per_direction",
            "split_sizes", "language_codes", "duplicate_rule", "prompt_policy", "records")},
        "corpus_receipt": read_json(run / "corpus_receipt.json") if (run / "corpus_receipt.json").exists() else None,
        "test_used_outside_final_test": outside, "exploratory": bool(outside),
        "decision": read_json(run / "decision.json") if (run / "decision.json").exists() else None,
        "decision_sha256": sha256_file(run / "decision.json") if (run / "decision.json").exists() else None,
        "verdict": decide(checks, quality, max_drop),
        "mechanism_checks": checks,
        "calibration": summary.get("routing"),
        "selection": selection,
        "mask_audit": {p.parent.parent.name: read_json(p).get("mask_audit")
                       for p in sorted(run.glob("k*/masked/records.jsonl.summary.json"))},
        "runs": timing,
        "chrf_pp": table_validation, "chrf_pp_test": table_test,
        "spot_check_largest_validation_drops": list(spot),
        "not_run": NOT_RUN,
    }
    artifacts = {}
    for path in sorted(run.rglob("*.json")):
        if path.name == "report.json" or "traces" in path.parts:
            continue
        data = read_json(path)
        artifacts[path.relative_to(run).as_posix().removesuffix(".json").replace("/", "__")] = (
            {k: v for k, v in data.items() if k != "per_sentence"} if isinstance(data, dict) else data)
    report["artifacts"] = artifacts
    write_text_atomic(run / "report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_text_atomic(run / "report.md", render_markdown(report))
    return report


def arrow(direction: str) -> str:
    src, tgt = direction.split("-")
    return f"{src}→{tgt}"


def render_markdown(report: dict) -> str:
    verdict, quality, checks = report["verdict"], report["verdict"]["quality"], report["mechanism_checks"]
    env, corpus, decision = report.get("environment") or {}, report["corpus"], report.get("decision")
    out = ["# D01b expert pruning of Hy-MT3-30B-A3B: pilot report", "",
           f"Generated {report['generated']}. {report['scope']}.", ""]
    if report.get("synthetic"):
        out += ["> **Synthetic random-weight checkpoint: nothing in this report is a real-model result.**", ""]
    out += ["## Verdict", "",
            f"- Mechanism: **{'PASS' if verdict['mechanism_pass'] else 'FAIL'}**. Identity reload of the unpruned "
            f"checkpoint equals the original: {'PASS' if verdict['noop_pass'] else 'FAIL'}. Physical prune equals the "
            "masked original: "
            + (", ".join(f"{n[1:]} experts {'PASS' if ok else 'FAIL'}" for n, ok in verdict["masked_pass"].items())
               or "NOT_RUN") + ".",
            f"- Quality gate: {verdict['gate']}."]
    for name, q in quality.items():
        for split in ("validation", "test"):
            result = q[split]
            line = f"- {name} experts, {split}: **{result['status']}**"
            if result.get("failing"):
                line += f" ({', '.join(arrow(d) for d in result['failing'])})"
            if result.get("problem") or result.get("reason"):
                line += f" [{result.get('problem') or result.get('reason')}]"
            out.append(line)
    out += ["", f"**{verdict['quality_result']}.**", ""]
    if report["exploratory"]:
        out += ["**The test split was used outside the final-test stage "
                f"({'; '.join(report['test_used_outside_final_test'])}): every test number below is exploratory.**", ""]
    out += ["## Inputs and environment", "",
            f"- Model: `tencent/Hy-MT3-30B-A3B` at revision `{MODEL_REVISION}`; checkpoint kind "
            f"`{report['checkpoint']['kind']}`, teacher digest `{report['checkpoint']['teacher']}`.",
            f"- Corpus: {corpus['corpus']} ({corpus['license']}), codes {corpus['language_codes']}. Duplicate rule: "
            f"{corpus['duplicate_rule']}. Tarball sha256 `{corpus['tarball_sha256']}`; sampled corpus sha256 "
            f"`{corpus['corpus_sha256']}`; {corpus['sizes_per_direction']} sentences per direction, seed {corpus['seed']}. "
            f"Prompt policy: {corpus['prompt_policy']}.",
            f"- Corpus receipt: {report['corpus_receipt']['status'] if report.get('corpus_receipt') else 'none'}.",
            f"- Runtime: Python {env.get('python')}, torch {env.get('torch')}, transformers {env.get('transformers')}, "
            f"sacrebleu {env.get('sacrebleu')}; {env.get('torch_threads')} threads on {env.get('cpu_count')} logical "
            f"CPUs ({env.get('platform')}).",
            f"- Decoding: {report['decoding']}. Metric: {report['metric']}.", ""]
    if decision:
        chosen = decision.get("chosen_keep")
        out += ["## Provisional decision (validation records only)", "",
                f"- Status `{decision['status']}`. "
                + (f"Smallest candidate passing every direction: {chosen} experts." if chosen is not None
                   else "No candidate passes every direction: FALLBACK_TO_TEACHER."),
                f"- decision.json sha256 `{report['decision_sha256']}`.",
                f"- `test_execution_authorized` is {decision['test_execution_authorized']}: {HELD}.", ""]
    out += ["## Mechanism checks", "",
            "| check | keep | sentences | identical text | argmax agree | route ids agree | within tol | "
            "full logits bitwise identical | max abs Δ of the 64 largest logits | pass |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(checks, key=lambda s: -int(s[1:])):
        c, e = checks[name], checks[name]["equiv"]
        out.append(f"| {c['kind']} | {c['keep']} | {e['sentences']} | {e['identical_text']} | {e['all_argmax_agree']} | "
                   f"{e['all_route_ids_agree']} | {e['within_tol']} | {e['bitwise_identical_logits']} | "
                   f"{e['logits_max_abs_diff']:.4g} | {'PASS' if e['pass'] else 'FAIL'} |")
    out += ["", "\"Within tol\" = the 64 largest logits of every compared position differ by at most 0.125 "
             "(two bf16 steps where the top logits sit); \"bitwise identical\" = the hash of the whole logit row "
             "is the same at every compared position.", ""]
    if report["mask_audit"]:
        out += ["## What the mask changes", "",
                "On the traced validation sentences: routing decisions (token, layer) that differ from the unmasked "
                "model on the same hidden states, and the routing weight it removes.", "",
                "| keep | direction | phase | decisions | changed | changed % | lost weight |",
                "|---|---|---|---|---|---|---|"]
        for name, audit in report["mask_audit"].items():
            for direction, phases in (audit or {}).items():
                for phase, values in phases.items():
                    share = "n/a" if values["changed_fraction"] is None else f"{100 * values['changed_fraction']:.3f}%"
                    out.append(f"| {name[1:]} | {arrow(direction)} | {phase} | {values['decisions']} | "
                               f"{values['changed']} | {share} | {values['lost_weight_mass']:.1f} |")
        out.append("")
    for split, key in (("validation", "chrf_pp"), ("test", "chrf_pp_test")):
        table = report[key]
        if not table:
            continue
        names = list(quality)
        out += [f"## chrF++ per direction, {split} split", "",
                "| direction | n | baseline | " + " | ".join(f"{n} experts: Δ [95% CI] (identical/n; +better/−worse)"
                                                            for n in names) + " |",
                "|---|---|---|" + "---|" * len(names)]
        for row in table:
            cells = []
            for n in names:
                c = row["candidates"].get(n)
                cells.append("no usable run" if not c or "chrf_pp" not in c else
                             f"{c['delta']:+.2f} [{c['delta_ci95'][0]:+.2f}, {c['delta_ci95'][1]:+.2f}] "
                             f"({c['identical_to_baseline']}/{row['n']}; +{c['sentences_better']}/−{c['sentences_worse']})")
            out.append(f"| {arrow(row['direction'])} | {row['n']} | {row['baseline_chrf_pp']:.2f} | "
                       + " | ".join(cells) + " |")
        out += ["", "Δ is candidate minus baseline in chrF++ points; CI from a paired bootstrap over sentences "
                 "(2000 resamples, seed 42). \"identical\" counts outputs equal to the baseline text.", ""]
    flags = sorted({f for row in report["chrf_pp"] for g in [row["baseline_flags"]]
                    + [c["flags"] for c in row["candidates"].values() if "flags" in c] for f in g})
    if flags:
        out += ["## Failure evidence on validation (heuristic flags)", "",
                "| direction | variant | " + " | ".join(flags) + " |", "|---|---|" + "---|" * len(flags)]
        for row in report["chrf_pp"]:
            for label, counts in [("baseline", row["baseline_flags"])] + [
                    (f"{n} experts", c["flags"]) for n, c in row["candidates"].items() if "flags" in c]:
                out.append(f"| {arrow(row['direction'])} | {label} | " + " | ".join(str(counts.get(f, 0)) for f in flags) + " |")
        out += ["", "Flags point at sentences to look at; they are not error verdicts (a correct translation can "
                 "legitimately write a number differently from the single reference).", ""]
    if report["spot_check_largest_validation_drops"]:
        out += ["## Sentences to look at first (largest per-sentence chrF++ drops on validation)", "",
                "Source and reference sentences: FLORES-200 (Meta AI), CC-BY-SA 4.0.", ""]
        for item in report["spot_check_largest_validation_drops"]:
            out += [f"**{item['variant']} experts, {arrow(item['direction'])}, {item['id']}, "
                    f"Δ {item['sentence_chrf_pp_delta']:+.2f}**",
                    f"- source: {item['source']}", f"- reference: {item['reference']}",
                    f"- baseline: {item['baseline']}", f"- candidate: {item['candidate']}", ""]
    out += ["## Not in this report", ""] + [f"- {item}" for item in report["not_run"]] + [""]
    return "\n".join(out)
