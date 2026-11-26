"""chrF++, failure evidence and trace comparison."""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from sacrebleu.metrics import CHRF

from .identity import sha256_file

# Pre-registered equivalence bound on the bf16 logits: 2 ULPs where the top logits live (16–32).
# It is a reported diagnostic only: a pass is claimed on bitwise-identical full logit rows, never
# on a within-tolerance comparison of the kept logits alone.
EQUIV_ABS_TOL = 0.125
# Route weights are float32; the masked 128-expert router and the physically pruned router
# compute their scores with GEMMs of different shapes, so their sums can differ in the last
# place (~1e-7 here). A pass allows at most 16 float32 ULPs at these magnitudes; the logit
# rows themselves must still be bitwise identical.
ROUTE_WEIGHT_ABS_TOL = 1e-6
# The arrays a saved trace holds, in the fixed order the content digest runs over.
TRACE_ARRAYS = ("top_values", "top_index", "argmax", "row_hash", "ids", "w")
# Exact on-disk dtypes: a trace with any other dtype is not this engine's evidence.
TRACE_DTYPES = {"top_values": np.dtype("uint16"), "top_index": np.dtype("int32"),
                "argmax": np.dtype("int32"), "row_hash": np.dtype("uint8"),
                "ids": np.dtype("int32"), "w": np.dtype("float32")}
# The full-logit row fingerprint (engine.py trace_rows): 128-bit BLAKE2b over each
# position's raw logit row. Equality of these hashes is identity of the engine's rows;
# it is a fingerprint, not a retention of raw full logits and not a mathematical proof.
ROW_HASH_ALGORITHM = "BLAKE2b-128 per position logit row"
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


def trace_path(directory: Path, sentence_id: str) -> Path:
    return Path(directory) / f"{sentence_id.replace(':', '__')}.trace.npz"


def trace_digest(arrays: dict) -> str:
    """Content digest of a trace: every array's shape and bytes, in a fixed order."""
    digest = hashlib.sha256()
    for name in TRACE_ARRAYS:
        array = arrays[name]
        if torch.is_tensor(array):
            array = array.numpy()
        array = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_trace(directory: Path, sentence_id: str, expected: dict | None = None,
               expected_positions: int | None = None) -> dict | None:
    """The one trace reader every stage uses; returns the verified arrays or None.

    Loads the container, checks the exact array set and on-disk dtypes, checks the
    per-position shapes agree with the position count, recomputes the content digest
    from the actual bytes and compares it with the stored digest, and rejects
    non-finite route weights. With ``expected`` (the journal's {"sha256", "positions"})
    both must match; with ``expected_positions`` the position count must match too.
    A valid container with altered arrays, a digest-only stub, a wrong dtype or shape,
    or a truncated file is no evidence (R13-P1-01): the stored digest field alone is
    never trusted.
    """
    path = trace_path(directory, sentence_id)
    if not path.exists():
        return None
    arrays = {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(TRACE_ARRAYS) | {"digest"}:
                return None
            for name in TRACE_ARRAYS:
                array = data[name]
                if array.dtype != TRACE_DTYPES[name]:
                    return None
                arrays[name] = array
            stored = data["digest"]
            if stored.dtype.kind not in "US" or stored.shape != ():
                return None
            stored = str(stored)
            ranks = {"argmax": 1, "top_values": 2, "top_index": 2, "row_hash": 2, "ids": 3, "w": 3}
            if any(arrays[name].ndim != rank for name, rank in ranks.items()):
                return None
            positions = int(arrays["argmax"].shape[0])
            top = int(arrays["top_values"].shape[1])
            layers, k = int(arrays["ids"].shape[1]), int(arrays["ids"].shape[2])
            if (positions < 1 or top < 1 or layers < 1 or k < 1
                    or arrays["top_values"].shape != (positions, top)
                    or arrays["top_index"].shape != (positions, top)
                    or arrays["row_hash"].shape != (positions, 16)
                    or arrays["w"].shape != (positions, layers, k)
                    or not bool(np.isfinite(arrays["w"]).all())):
                return None
    except Exception:                              # truncated, malformed or otherwise unreadable
        return None
    if trace_digest(arrays) != stored:             # the stored digest counts only when the
        return None                                 # actual bytes recompute to it
    if expected is not None and (stored != expected.get("sha256") or positions != expected.get("positions")):
        return None
    if expected_positions is not None and positions != expected_positions:
        return None
    return {**arrays, "positions": positions, "digest": stored}


def trace_complete(directory: Path, sentence_id: str, expected: dict | None = None,
                   expected_positions: int | None = None) -> bool:
    """True when read_trace fully verifies the trace against the journal's record.

    An ``expected`` record is the {"sha256", "positions"} a committed sample carries;
    ``expected_positions`` is the generation record's prompt + generated token count.
    """
    return read_trace(directory, sentence_id, expected, expected_positions) is not None


def save_trace(directory: Path, sentence_id: str, trace: dict) -> dict:
    """Atomically publish one trace; returns the journal's {"sha256", "positions"} record of it.

    The file is written under a temporary name and renamed into place, so an
    interruption can never leave a partial file that a later stage mistakes for
    evidence. The digest binds the published arrays, not the container bytes.
    """
    path = trace_path(directory, sentence_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"top_values": trace["top_values"].view(torch.uint16).numpy(),
              "top_index": trace["top_index"].numpy(), "argmax": trace["argmax"].numpy(),
              "row_hash": trace["row_hash"].numpy(),
              "ids": trace["route_ids"].numpy().astype(np.int32),
              "w": trace["route_w"].numpy().astype(np.float32)}
    record = {"sha256": trace_digest(arrays), "positions": int(trace["argmax"].shape[0])}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, digest=np.array(record["sha256"]), **arrays)
    temporary.replace(path)
    return record


def load_trace(directory: Path, sentence_id: str) -> dict | None:
    """Load a trace for comparison - through the strict reader, never the raw container."""
    verified = read_trace(directory, sentence_id)
    if verified is None:
        return None
    return {"top_values": torch.from_numpy(verified["top_values"]).view(torch.bfloat16),
            "top_index": torch.from_numpy(verified["top_index"].astype(np.int64)),
            "argmax": torch.from_numpy(verified["argmax"].astype(np.int64)),
            "row_hash": torch.from_numpy(verified["row_hash"]),
            "route_ids": torch.from_numpy(verified["ids"].astype(np.int64)),
            "route_w": torch.from_numpy(verified["w"]),
            "digest": verified["digest"], "positions": verified["positions"]}


def compare_trace(reference: dict, candidate: dict) -> dict:
    """Compare a candidate trace with the reference trace along the reference's token path.

    Both traces hold every position the model was run on: the prompt, then each
    generated token. A shorter or longer candidate is a length mismatch, never a
    pass. They agree only while both models generated the same tokens, so logit,
    hash and routing comparisons stop at the first position where the greedy
    choice differs - a divergence fails the comparison in compare_runs anyway.
    Logits are compared on the union of the two top lists (an index missing from
    one list counts as outside it); whether the whole rows are identical is decided
    by their hashes. Route weights are aligned by sorted route id first, so a
    different slot order cannot hide a weight difference.
    """
    ref_len, cand_len = int(reference["argmax"].shape[0]), int(candidate["argmax"].shape[0])
    n = min(ref_len, cand_len)
    ref_arg, cand_arg = reference["argmax"][:n], candidate["argmax"][:n]
    mismatch = ref_arg != cand_arg
    first = int(mismatch.nonzero()[0]) if bool(mismatch.any()) else None
    stop = n if first is None else first + 1
    ref_index, cand_index = reference["top_index"][:stop], candidate["top_index"][:stop]
    union = torch.cat((ref_index, cand_index), dim=1)                    # [stop, top ref + top cand]
    minus_inf = torch.tensor(float("-inf"))

    def spread(index, values):
        # the value a run assigns to every index of the union; -inf outside its top list
        present = union[:, :, None] == index[:, None, :]
        picked = torch.where(present, values[:, None, :].expand(-1, union.shape[1], -1), minus_inf)
        return picked.amax(dim=-1)

    ref_values = spread(ref_index, reference["top_values"][:stop].float())
    cand_values = spread(cand_index, candidate["top_values"][:stop].float())
    both = torch.isfinite(ref_values) & torch.isfinite(cand_values)
    diff = torch.where(both, (ref_values - cand_values).abs(), torch.zeros_like(ref_values))
    same_rows = bool((reference["row_hash"][:stop] == candidate["row_hash"][:stop]).all())
    ref_ids, ref_order = torch.sort(reference["route_ids"][:stop], dim=-1)
    cand_ids, cand_order = torch.sort(candidate["route_ids"][:stop], dim=-1)
    ids_match = (ref_ids == cand_ids).all(dim=-1)
    weight_diff = (torch.gather(reference["route_w"][:stop], -1, ref_order)
                   - torch.gather(candidate["route_w"][:stop], -1, cand_order)).abs()
    layout_match = (reference["route_ids"].shape[1:] == candidate["route_ids"].shape[1:]
                    and reference["top_index"].shape[-1] == candidate["top_index"].shape[-1])
    return {
        "reference_positions": ref_len, "candidate_positions": cand_len, "length_match": ref_len == cand_len,
        "layout_match": bool(layout_match),
        "positions_compared": stop,
        "logits_max_abs_diff": float(diff.max()),
        "logits_mean_abs_diff": float(diff.sum() / max(int(both.sum()), 1)),
        "top_entries_outside_the_other_list": int((~both).sum()),
        "full_logits_bitwise_identical": same_rows,
        "argmax_mismatches": int(mismatch.sum()),
        "first_divergent_position": first,
        "route_id_mismatches": int((~ids_match).sum()),
        "route_positions": int(ids_match.numel()),
        "route_weight_max_abs_diff": float(weight_diff.max()),
        "route_weights_bitwise_equal": bool((weight_diff == 0).all()),
    }


def compare_runs(reference_dir: Path, candidate_dir: Path, reference_text: dict, candidate_text: dict,
                 expected_ids: list[str] | None = None) -> dict:
    """Compare every traced sentence. With ``expected_ids`` both runs must hold exactly those.

    A pass means: exact sentence coverage, equal trace lengths and layout, identical
    output text, every argmax and mapped route id in agreement, full logit rows bitwise
    identical at every compared position, and route weights equal within the declared
    float32 rounding - on the predeclared witness sentences only (R11-P1-03). A
    within-tolerance comparison of the kept logits alone is reported as a diagnostic
    and never labelled a full equivalence.
    """
    ids = sorted(path.name[:-len(".trace.npz")].replace("__", ":")
                 for path in Path(reference_dir).glob("*.trace.npz"))
    if expected_ids is not None and ids != sorted(expected_ids):
        raise ValueError(f"reference traces cover {len(ids)} of the {len(expected_ids)} expected sentences")
    if not ids:
        raise ValueError("no traces to compare")
    candidate_ids = sorted(path.name[:-len(".trace.npz")].replace("__", ":")
                           for path in Path(candidate_dir).glob("*.trace.npz"))
    if expected_ids is not None and candidate_ids != sorted(expected_ids):
        raise ValueError(f"candidate traces cover {len(candidate_ids)} of the {len(expected_ids)} expected sentences")
    sentences = []
    for sentence_id in ids:
        reference, candidate = load_trace(reference_dir, sentence_id), load_trace(candidate_dir, sentence_id)
        if reference is None or candidate is None:
            raise ValueError(f"missing trace for {sentence_id}")
        if sentence_id not in reference_text or sentence_id not in candidate_text:
            raise ValueError(f"missing output text for {sentence_id}")
        result = compare_trace(reference, candidate)
        result.update(id=sentence_id, text_equal=reference_text[sentence_id] == candidate_text[sentence_id],
                      reference_trace_sha256=sha256_file(trace_path(reference_dir, sentence_id)),
                      candidate_trace_sha256=sha256_file(trace_path(candidate_dir, sentence_id)),
                      trace_digest_equal=reference["digest"] == candidate["digest"])
        sentences.append(result)
    same_text = [s["text_equal"] for s in sentences]
    argmax_ok = [s["argmax_mismatches"] == 0 for s in sentences]
    route_ok = [s["route_id_mismatches"] == 0 for s in sentences]
    tol_ok = [s["logits_max_abs_diff"] <= EQUIV_ABS_TOL for s in sentences]
    bitwise = [s["full_logits_bitwise_identical"] and s["length_match"] for s in sentences]
    weights_ok = [s["route_weight_max_abs_diff"] <= ROUTE_WEIGHT_ABS_TOL for s in sentences]
    coverage_ok = [s["length_match"] and s["layout_match"] for s in sentences]
    return {
        "sentences": len(sentences), "abs_tol": EQUIV_ABS_TOL, "compared_logits": "64 largest per position",
        "route_weight_abs_tol": ROUTE_WEIGHT_ABS_TOL, "row_hash_algorithm": ROW_HASH_ALGORITHM,
        "claim": "engine logit-row fingerprints (BLAKE2b-128) identical at every compared position, route ids "
                 "identical, route weights within float32 rounding, on the traced witness sentences only; "
                 "fingerprint equality of the engine's rows: not raw full-logit retention, not a mathematical "
                 "proof, and not a whole-run claim",
        "identical_text": sum(same_text),
        "all_argmax_agree": sum(argmax_ok),
        "all_route_ids_agree": sum(route_ok),
        "within_tol": sum(tol_ok),
        "bitwise_identical_logits": sum(bitwise),
        "route_weights_bitwise_equal": sum(s["route_weights_bitwise_equal"] for s in sentences),
        "route_weights_within_tol": sum(weights_ok),
        "length_match": sum(s["length_match"] for s in sentences),
        "positions_compared": sum(s["positions_compared"] for s in sentences),
        "logits_max_abs_diff": max(s["logits_max_abs_diff"] for s in sentences),
        "route_id_mismatches": sum(s["route_id_mismatches"] for s in sentences),
        "route_weight_max_abs_diff": max(s["route_weight_max_abs_diff"] for s in sentences),
        "pass": (all(same_text) and all(argmax_ok) and all(route_ok) and all(bitwise) and all(weights_ok)
                 and all(coverage_ok)),
        "per_sentence": sentences,
    }
