"""chrF++, failure evidence and trace comparison."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from sacrebleu.metrics import CHRF

from .corpus import DIRECTIONS

# Pre-registered equivalence bound on the bf16 logits: 2 ULPs where the top logits live (16–32).
EQUIV_ABS_TOL = 0.125
_DIGITS = str.maketrans("၀၁၂၃၄၅၆၇၈၉０１２３４５６７８９", "0123456789" * 2)


def digit_runs(text: str) -> list[str]:
    """Digit groups after folding Burmese and full-width digits to ASCII."""
    return sorted(re.findall(r"\d+", text.translate(_DIGITS)))


def script_share(text: str, lang: str) -> float:
    """Share of letters written in the target language's script (0 for text without letters)."""
    def in_script(code: int) -> bool:
        if lang == "zh":
            return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
        if lang == "my":
            return 0x1000 <= code <= 0x109F or 0xAA60 <= code <= 0xAA7F or 0xA9E0 <= code <= 0xA9FE
        return unicodedata.name(chr(code), "").startswith("LATIN")
    letters = [ord(c) for c in text if c.isalpha()]
    return sum(map(in_script, letters)) / len(letters) if letters else 0.0


def failure_flags(source: str, reference: str, hyp: str, tgt_lang: str, finish: str, tokens: list[int]) -> list[str]:
    """Heuristic evidence flags. They point at sentences to look at; they are not error verdicts."""
    flags = []
    norm = lambda s: " ".join(s.split())
    if not hyp.strip():
        flags.append("empty")
    if finish != "eos":
        flags.append("truncated")
    if norm(hyp) == norm(source) != norm(reference):
        flags.append("untranslated_copy")
    share, ref_share = script_share(hyp, tgt_lang), script_share(reference, tgt_lang)
    if ref_share >= 0.5 > share:
        flags.append("wrong_script")
    if digit_runs(hyp) != digit_runs(reference):
        flags.append("digits_differ_from_reference")
    if reference and (len(hyp) < 0.5 * len(reference) or len(hyp) > 2 * len(reference)):
        flags.append("length_ratio_outlier")
    if len(tokens) >= 16 and len(set(tokens)) < 0.5 * len(tokens):
        flags.append("degenerate_repetition")
    return flags


def chrf_pp() -> CHRF:
    return CHRF(word_order=2)


def chrf() -> CHRF:
    return CHRF(word_order=0)


def segment_stats(metric, hypotheses: list[str], references: list[str]) -> list[list[int]]:
    """Per-sentence chrF++ match statistics; summed, they give the corpus score."""
    return metric._extract_corpus_statistics(hypotheses, [references])


def score_from_stats(metric, rows) -> float:
    return metric._compute_score_from_stats(np.asarray(rows, dtype=np.int64).sum(axis=0).tolist()).score


def paired_bootstrap(metric, baseline_rows, other_rows, resamples=2000, seed=42) -> dict:
    """Corpus-level chrF++ delta (other − baseline) with a paired bootstrap over sentences."""
    baseline_rows, other_rows = np.asarray(baseline_rows), np.asarray(other_rows)
    if len(baseline_rows) != len(other_rows) or not len(baseline_rows):
        raise ValueError("paired bootstrap needs equally many segments")
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, len(baseline_rows), len(baseline_rows))
        deltas[i] = score_from_stats(metric, other_rows[idx]) - score_from_stats(metric, baseline_rows[idx])
    return {"delta": score_from_stats(metric, other_rows) - score_from_stats(metric, baseline_rows),
            "ci95_low": float(np.percentile(deltas, 2.5)), "ci95_high": float(np.percentile(deltas, 97.5)),
            "resamples": resamples, "seed": seed}


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance between bf16 values in representable steps."""
    def ordered(x):
        bits = x.view(torch.int16).to(torch.int32) & 0xFFFF
        return torch.where(bits >= 0x8000, 0x8000 - bits, bits)
    return (ordered(a) - ordered(b)).abs()


def save_trace(directory: Path, sentence_id: str, trace: dict) -> None:
    name = sentence_id.replace(":", "__")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{name}.logits.npy", trace["logits"].view(torch.uint16).numpy())
    np.savez_compressed(directory / f"{name}.route.npz", ids=trace["route_ids"].numpy().astype(np.int16),
                        w=trace["route_w"].numpy().astype(np.float32))


def load_trace(directory: Path, sentence_id: str) -> dict | None:
    name = sentence_id.replace(":", "__")
    if not (directory / f"{name}.logits.npy").exists():
        return None
    route = np.load(directory / f"{name}.route.npz")
    return {"logits": torch.from_numpy(np.load(directory / f"{name}.logits.npy")).view(torch.bfloat16),
            "route_ids": torch.from_numpy(route["ids"].astype(np.int64)), "route_w": torch.from_numpy(route["w"])}


def compare_trace(reference: dict, candidate: dict) -> dict:
    """Compare a candidate trace with the reference trace along the reference's token path.

    Both traces hold every position the model was run on: the prompt, then each
    generated token. They agree only while both models generated the same tokens,
    so the comparison stops at the first position where the greedy choice differs.
    """
    ref_logits, cand_logits = reference["logits"], candidate["logits"]
    n = min(ref_logits.shape[0], cand_logits.shape[0])
    ref_arg, cand_arg = ref_logits[:n].float().argmax(-1), cand_logits[:n].float().argmax(-1)
    mismatch = ref_arg != cand_arg
    first = int(mismatch.nonzero()[0]) if bool(mismatch.any()) else None
    stop = n if first is None else first + 1
    diff = (ref_logits[:stop].float() - cand_logits[:stop].float()).abs()
    ulp = bf16_ulp_distance(ref_logits[:stop], cand_logits[:stop])
    ids_match = (torch.sort(reference["route_ids"][:stop], dim=-1).values
                 == torch.sort(candidate["route_ids"][:stop], dim=-1).values).all(dim=-1)
    weight_diff = (reference["route_w"][:stop] - candidate["route_w"][:stop]).abs()
    return {
        "positions_compared": stop,
        "logits_max_abs_diff": float(diff.max()),
        "logits_mean_abs_diff": float(diff.mean()),
        "logits_max_ulp": int(ulp.max()),
        "logits_differing_elements": int((diff > 0).sum()),
        "logits_elements": int(diff.numel()),
        "argmax_mismatches": int(mismatch.sum()),
        "first_divergent_position": first,
        "route_id_mismatches": int((~ids_match).sum()),
        "route_positions": int(ids_match.numel()),
        "route_weight_max_abs_diff": float(weight_diff.max()),
    }


def compare_runs(reference_dir: Path, candidate_dir: Path, reference_text: dict, candidate_text: dict) -> dict:
    sentences = []
    for path in sorted(Path(reference_dir).glob("*.logits.npy")):
        sentence_id = path.name[:-len(".logits.npy")].replace("__", ":")
        reference, candidate = load_trace(reference_dir, sentence_id), load_trace(candidate_dir, sentence_id)
        if reference is None or candidate is None:
            raise ValueError(f"missing trace for {sentence_id}")
        result = compare_trace(reference, candidate)
        result.update(id=sentence_id, text_equal=reference_text[sentence_id] == candidate_text[sentence_id])
        sentences.append(result)
    if not sentences:
        raise ValueError("no traces to compare")
    same_text = [s["text_equal"] for s in sentences]
    argmax_ok = [s["argmax_mismatches"] == 0 for s in sentences]
    route_ok = [s["route_id_mismatches"] == 0 for s in sentences]
    tol_ok = [s["logits_max_abs_diff"] <= EQUIV_ABS_TOL for s in sentences]
    bitwise = [s["logits_differing_elements"] == 0 for s in sentences]
    return {
        "sentences": len(sentences), "abs_tol": EQUIV_ABS_TOL,
        "identical_text": sum(same_text),
        "all_argmax_agree": sum(argmax_ok),
        "all_route_ids_agree": sum(route_ok),
        "within_tol": sum(tol_ok),
        "bitwise_identical_logits": sum(bitwise),
        "positions_compared": sum(s["positions_compared"] for s in sentences),
        "logits_max_abs_diff": max(s["logits_max_abs_diff"] for s in sentences),
        "logits_max_ulp": max(s["logits_max_ulp"] for s in sentences),
        "route_id_mismatches": sum(s["route_id_mismatches"] for s in sentences),
        "pass": all(same_text) and all(argmax_ok) and all(route_ok) and all(tol_ok),
        "per_sentence": sentences,
    }
