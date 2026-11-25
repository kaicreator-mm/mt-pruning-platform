import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mtpruning.corpus import FLORES_LANG, build_prompt, prompt_lang, sample_indices
from mtpruning.evaluate import (EQUIV_ABS_TOL, chrf_pp, compare_runs, compare_trace,
                                failure_flags, load_trace, paired_bootstrap, read_trace, save_trace,
                                score_from_stats, segment_stats, trace_complete, trace_path)


def flags(source, reference, hyp, tgt="en", finish="eos", tokens=(1, 2, 3)):
    return failure_flags(source, reference, hyp, tgt, finish, list(tokens))


def dir_with(trace, sentence_id: str, directory):
    save_trace(directory, sentence_id, trace)
    return directory


class FailureFlagTests(unittest.TestCase):
    def test_flags(self):
        self.assertIn("empty", flags("Hello", "你好", "  ", "zh"))
        self.assertIn("truncated", flags("Hello", "你好", "你好", "zh", finish="length"))
        self.assertIn("untranslated_copy", flags("Hello world", "你好世界", "Hello  world", "zh"))
        self.assertNotIn("untranslated_copy", flags("Hello", "Hello", "Hello", "en"))   # a real namesake
        self.assertIn("wrong_script", flags("你好", "Hello world", "你好", "en"))
        self.assertIn("wrong_script", flags("Hello", "ကမ္ဘာကြီး", "Hello world", "my"))
        self.assertNotIn("wrong_script", flags("你好", "Hello world", "Hello there", "en"))
        self.assertIn("digits_differ_from_reference", flags("9:30", "9:30", "nine thirty", "en"))
        self.assertNotIn("digits_differ_from_reference", flags("၉:၃၀", "9:30", "at 9:30", "en"))
        self.assertNotIn("digits_differ_from_reference", flags("9:30", "၉:၃၀", "၉:၃၀ နာရီ", "my"))   # digits folded
        self.assertIn("length_ratio_outlier", flags("x" * 400, "y" * 400, "short", "en"))
        self.assertIn("degenerate_repetition", flags("a", "b", "x x x x", "en", tokens=[7] * 20))

    def test_clean_sentence_has_no_flags(self):
        self.assertEqual(flags("The cat sat.", "Le chat s'est assis.", "Le chat s'est assis.", "en", tokens=range(10)), [])


class MetricTests(unittest.TestCase):
    hyps = ["The cat sat on the mat.", "A second, longer sentence about a dog and its owner in the park today.",
            " ", "Numbers 9:30 and ၇၅ are kept."]
    refs = ["The cat sat on the mat.", "Another longer sentence about a cat and its owner in the garden right now.",
            "Something is missing here.", "Numbers 9:30 and 75 are kept."]

    def test_segment_statistics_reproduce_corpus_and_sentence_scores(self):
        metric = chrf_pp()
        rows = segment_stats(metric, self.hyps, self.refs)
        self.assertAlmostEqual(score_from_stats(metric, rows), metric.corpus_score(self.hyps, [self.refs]).score)
        for row, hyp, ref in zip(rows, self.hyps, self.refs):
            self.assertAlmostEqual(score_from_stats(metric, [row]), metric.sentence_score(hyp, [ref]).score)

    def test_paired_bootstrap(self):
        metric = chrf_pp()
        rows = segment_stats(metric, self.hyps, self.refs)
        same = paired_bootstrap(metric, rows, rows, resamples=100)
        self.assertEqual((same["delta"], same["ci95_low"], same["ci95_high"]), (0.0, 0.0, 0.0))
        worse = paired_bootstrap(metric, segment_stats(metric, self.refs, self.refs), rows, resamples=200)
        self.assertLess(worse["delta"], 0)
        self.assertLess(worse["ci95_high"], 0)
        with self.assertRaises(ValueError):
            paired_bootstrap(metric, [], [], resamples=10)


class TraceTests(unittest.TestCase):
    @staticmethod
    def trace(steps, vocab=2048, seed=0):
        from mtpruning.engine import trace_rows
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(steps, vocab, generator=generator).bfloat16()
        ids = torch.stack([torch.randperm(8, generator=generator)[:2] for _ in range(steps * 2)]).reshape(steps, 2, 2)
        return {**trace_rows(logits), "logits": logits, "route_ids": ids,
                "route_w": torch.rand(steps, 2, 2, generator=generator)}

    @staticmethod
    def refreshed(trace):
        from mtpruning.engine import trace_rows
        return {**trace, **trace_rows(trace["logits"])}

    def test_a_traced_position_keeps_the_top_logits_and_binds_the_whole_row(self):
        trace = self.trace(9)
        self.assertEqual((tuple(trace["top_values"].shape), tuple(trace["row_hash"].shape)), ((9, 64), (9, 16)))
        # bf16 has 8 mantissa bits, so the largest logit can be tied: the first index of the top
        # list need not be argmax's choice, but the values must agree.
        self.assertTrue(torch.equal(trace["top_values"][:, 0], trace["logits"].float().gather(1, trace["argmax"][:, None])[:, 0]))
        self.assertEqual(int(trace["top_values"].shape[1]), 64)
        self.assertEqual(self.refreshed(trace)["row_hash"].tolist(), trace["row_hash"].tolist())
        self.assertLess(float(trace["top_values"][:, -1].max()), float(trace["logits"].max()))

    def test_identical_traces(self):
        a = self.trace(7)
        result = compare_trace(a, {k: v.clone() for k, v in a.items()})
        self.assertEqual((result["positions_compared"], result["argmax_mismatches"],
                          result["route_id_mismatches"], result["logits_max_abs_diff"]), (7, 0, 0, 0.0))
        self.assertTrue(result["full_logits_bitwise_identical"])

    def test_comparison_stops_at_first_divergent_choice(self):
        a = self.trace(9)
        b = {k: v.clone() for k, v in a.items()}
        b["logits"][4, 1] = a["logits"][4].max() + 8            # position 4 now prefers token 1
        result = compare_trace(a, self.refreshed(b))
        self.assertEqual((result["first_divergent_position"], result["positions_compared"],
                          result["argmax_mismatches"]), (4, 5, 1))

    def test_a_changed_row_is_detected_even_below_the_top_logits(self):
        a = self.trace(6)
        b = {k: v.clone() for k, v in a.items()}
        b["logits"][2, -1] = torch.nextafter(b["logits"][2, -1], torch.tensor(1.0, dtype=torch.bfloat16))
        result = compare_trace(a, self.refreshed(b))
        self.assertFalse(result["full_logits_bitwise_identical"])
        self.assertEqual(result["logits_max_abs_diff"], 0.0)    # the change is below the kept list
        b["logits"][2, 0] += 0.25                                 # inside the kept list
        result = compare_trace(a, self.refreshed(b))
        self.assertGreaterEqual(result["logits_max_abs_diff"], 0.2)

    def test_an_index_missing_from_the_other_top_list_is_reported(self):
        a = self.trace(5)
        b = {k: v.clone() for k, v in a.items()}
        b["top_index"][3, -1] = b["top_index"][3, -2]            # one index of a's list is absent from b's
        self.assertEqual(compare_trace(a, b)["top_entries_outside_the_other_list"], 1)

    def test_save_load_round_trip_and_compare_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            first = self.trace(6, seed=3)
            record = save_trace(a, "test:zh-my:12", first)
            self.assertEqual(sorted(record), ["positions", "sha256"])
            self.assertEqual(record["positions"], 6)
            self.assertTrue(trace_complete(a, "test:zh-my:12"))
            self.assertTrue(trace_complete(a, "test:zh-my:12", record))
            loaded = load_trace(a, "test:zh-my:12")
            self.assertEqual(loaded["digest"], record["sha256"])
            for name in ("top_values", "top_index", "argmax", "row_hash", "route_ids", "route_w"):
                self.assertTrue(torch.equal(loaded[name], first[name]), name)
            save_trace(b, "test:zh-my:12", {k: v.clone() for k, v in first.items()})
            text = {"test:zh-my:12": "same"}
            passing = compare_runs(a, b, text, text, ["test:zh-my:12"])
            self.assertTrue(passing["pass"])
            self.assertEqual((passing["bitwise_identical_logits"], passing["positions_compared"],
                              passing["route_weights_bitwise_equal"]), (1, 6, 1))
            sentence = passing["per_sentence"][0]
            self.assertEqual(sentence["reference_trace_sha256"], sentence["candidate_trace_sha256"])
            with self.assertRaisesRegex(ValueError, "expected sentences"):
                compare_runs(a, b, text, text, ["test:zh-my:12", "test:zh-my:13"])
            save_trace(b, "test:zh-my:12", self.trace(6, seed=4))
            failing = compare_runs(a, b, text, {"test:zh-my:12": "different"})
            self.assertFalse(failing["pass"])
            self.assertNotEqual(failing["per_sentence"][0]["reference_trace_sha256"],
                                failing["per_sentence"][0]["candidate_trace_sha256"])
        self.assertEqual(EQUIV_ABS_TOL, 0.125)

    def test_trace_records_are_checked_against_the_journal_and_published_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = self.trace(6, seed=3)
            record = save_trace(directory, "test:zh-my:12", first)
            self.assertEqual([p.name for p in directory.iterdir()], ["test__zh-my__12.trace.npz"])  # no .tmp left
            self.assertFalse(trace_complete(directory, "test:zh-my:12",
                                            {**record, "sha256": "0" * 64}))   # a foreign digest is not complete
            self.assertFalse(trace_complete(directory, "test:zh-my:12", {**record, "positions": 5}))
            path = trace_path(directory, "test:zh-my:12")
            path.write_bytes(path.read_bytes()[:64])     # an interrupted write: parses badly or not at all
            self.assertFalse(trace_complete(directory, "test:zh-my:12", record))
            self.assertIsNone(load_trace(directory, "test:zh-my:12"))

    def test_a_rewritten_array_with_an_unchanged_stored_digest_is_not_evidence(self):
        """R13-P1-01: the reader recomputes the digest from the actual arrays - a valid
        container whose row hashes were altered keeps the stored digest and still fails."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = save_trace(directory, "s", self.trace(6, seed=3))
            path = trace_path(directory, "s")
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            arrays["row_hash"][3, 0] ^= 0xFF                     # the full-logit fingerprints change ...
            np.savez_compressed(path, digest=arrays.pop("digest"), **arrays)   # ... the stored one does not
            self.assertIsNone(read_trace(directory, "s"))
            self.assertFalse(trace_complete(directory, "s", record))
            self.assertIsNone(load_trace(directory, "s"))
            # a stub that only carries the digest and the argmax is not a trace either
            np.savez_compressed(path, digest=np.array(record["sha256"]), argmax=arrays["argmax"])
            self.assertIsNone(read_trace(directory, "s"))
            # a wrong dtype for any required array is rejected before the digest
            save_trace(directory, "s", self.trace(6, seed=3))
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            arrays["w"] = arrays["w"].astype(np.float64)
            digest = arrays.pop("digest")
            np.savez_compressed(path, digest=digest, **arrays)
            self.assertIsNone(read_trace(directory, "s"))
            # an extra array is rejected: the container layout itself is pinned
            save_trace(directory, "s", self.trace(6, seed=3))
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            np.savez_compressed(path, spare=np.zeros(3, dtype=np.int32), **arrays)
            self.assertIsNone(read_trace(directory, "s"))
            # and the position count must match the generation record when given
            save_trace(directory, "s", self.trace(6, seed=3))
            self.assertTrue(trace_complete(directory, "s", record, expected_positions=6))
            self.assertFalse(trace_complete(directory, "s", record, expected_positions=7))

    def test_a_length_mismatch_is_never_a_pass(self):
        a, b = self.trace(7, seed=1), self.trace(5, seed=1)      # the candidate stopped early
        result = compare_trace(a, b)
        self.assertFalse(result["length_match"])
        self.assertEqual(result["positions_compared"], 5)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "c"
            save_trace(directory, "s", b)
            text = {"s": "same"}
            failing = compare_runs(dir_with(self.trace(7, seed=1), "s", Path(tmp) / "r"), directory, text, text, ["s"])
            self.assertFalse(failing["pass"])
            self.assertEqual((failing["length_match"], failing["bitwise_identical_logits"]), (0, 0))

    def test_route_weights_are_aligned_by_sorted_id_and_must_match(self):
        a = self.trace(5, seed=2)
        b = {k: v.clone() for k, v in a.items()}
        permuted = torch.tensor([1, 0])                           # same pairs, swapped slot order
        b["route_ids"] = a["route_ids"][:, :, permuted]
        b["route_w"] = a["route_w"][:, :, permuted]
        result = compare_trace(a, b)
        self.assertEqual((result["route_id_mismatches"], result["route_weight_max_abs_diff"]), (0, 0.0))
        self.assertTrue(result["route_weights_bitwise_equal"])
        b["route_w"][:, :, 0] += 1e-3                              # a real weight difference
        result = compare_trace(a, b)
        self.assertGreater(result["route_weight_max_abs_diff"], 0.0)
        self.assertFalse(result["route_weights_bitwise_equal"])
        text = {"s": "same"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failing = compare_runs(dir_with(a, "s", root / "r"), dir_with(b, "s", root / "c"), text, text, ["s"])
            self.assertFalse(failing["pass"])                      # weights are part of the verdict
            self.assertEqual((failing["all_route_ids_agree"], failing["route_weights_bitwise_equal"]), (1, 0))

    def test_the_candidate_side_must_cover_exactly_the_expected_sentences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = {"s": "same", "t": "same"}
            reference = dir_with(self.trace(3, seed=5), "s", root / "r")
            save_trace(reference, "t", self.trace(3, seed=6))
            candidate = dir_with(self.trace(3, seed=5), "s", root / "c")
            with self.assertRaisesRegex(ValueError, "candidate traces cover"):
                compare_runs(reference, candidate, text, text, ["s", "t"])
            save_trace(candidate, "t", self.trace(3, seed=6))
            self.assertTrue(compare_runs(reference, candidate, text, text, ["s", "t"])["pass"])


class CorpusLogicTests(unittest.TestCase):
    @staticmethod
    def texts():
        return {(split, code): [f"{split} {code} line {i}" for i in range(size)]
                for split, size in (("dev", 997), ("devtest", 1012)) for code in FLORES_LANG.values()}

    def test_sample_is_disjoint_and_reproducible(self):
        texts = self.texts()
        first, set_aside = sample_indices(texts, 24, 20, 20, seed=42)
        self.assertEqual(first, sample_indices(texts, 24, 20, 20, seed=42)[0])
        self.assertNotEqual(first, sample_indices(texts, 24, 20, 20, seed=43)[0])
        self.assertEqual(set_aside, {lang: {"dev": 0, "devtest": 0} for lang in ("zh", "en", "my")})
        for lang, picks in first.items():
            self.assertEqual([len(picks[s]) for s in ("calibration", "validation", "test")], [24, 20, 20])
            self.assertFalse(set(picks["validation"]) & set(picks["test"]))
            self.assertTrue(all(0 <= i < 997 for i in picks["calibration"]))
            self.assertTrue(all(0 <= i < 1012 for i in picks["test"]))
        with self.assertRaises(ValueError):
            sample_indices(texts, 1000, 20, 20, seed=42)

    def test_duplicates_across_splits_and_inside_a_sample_are_avoided(self):
        texts = self.texts()
        texts[("dev", "zho_Hans")][5] = texts[("devtest", "zho_Hans")][3]      # same source sentence, other split
        texts[("dev", "eng_Latn")][9] = texts[("devtest", "mya_Mymr")][7]      # leaks through a target language
        texts[("devtest", "eng_Latn")][12] = texts[("devtest", "eng_Latn")][11]
        picks, set_aside = sample_indices(texts, 24, 20, 20, seed=42)
        self.assertEqual(set_aside["zh"], {"dev": 2, "devtest": 2})
        for groups in picks.values():
            self.assertTrue({5, 9}.isdisjoint(groups["calibration"]))
            held_out = groups["validation"] + groups["test"]
            self.assertTrue({3, 7}.isdisjoint(held_out))
            self.assertFalse({11, 12} <= set(held_out))
            self.assertEqual(len(held_out), 40)

    def test_prompt_policy(self):
        text, lang = build_prompt("zh", "my", "你好")
        self.assertEqual((lang, "缅甸语" in text, text.startswith("将以下文本翻译为")), ("zh", True, True))
        self.assertTrue(text.endswith("：\n\n`你好`"))
        text, lang = build_prompt("en", "zh", "hi")
        self.assertEqual(lang, "en")
        self.assertTrue(text.startswith("Translate the following text into `Chinese`. Note that you should "
                                        "**only output the translated result without any additional explanation**:\n\n`hi`"))
        self.assertEqual(prompt_lang("my"), "en")


if __name__ == "__main__":
    unittest.main()
