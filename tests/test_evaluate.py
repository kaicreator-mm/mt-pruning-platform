import tempfile
import unittest
from pathlib import Path

import torch

from mtpruning.corpus import FLORES_LANG, build_prompt, prompt_lang, sample_indices
from mtpruning.evaluate import (EQUIV_ABS_TOL, bf16_ulp_distance, chrf_pp, compare_runs, compare_trace,
                                failure_flags, load_trace, paired_bootstrap, save_trace,
                                score_from_stats, segment_stats)


def flags(source, reference, hyp, tgt="en", finish="eos", tokens=(1, 2, 3)):
    return failure_flags(source, reference, hyp, tgt, finish, list(tokens))


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
    def trace(steps, vocab=64, seed=0):
        generator = torch.Generator().manual_seed(seed)
        ids = torch.stack([torch.randperm(8, generator=generator)[:2] for _ in range(steps * 2)]).reshape(steps, 2, 2)
        return {"logits": torch.randn(steps, vocab, generator=generator).bfloat16(), "route_ids": ids,
                "route_w": torch.rand(steps, 2, 2, generator=generator)}

    def test_ulp_distance(self):
        one = torch.tensor([1.0], dtype=torch.bfloat16)
        self.assertEqual(bf16_ulp_distance(one, torch.nextafter(one, torch.full_like(one, 2.0))).item(), 1)
        self.assertEqual(bf16_ulp_distance(torch.tensor([0.0], dtype=torch.bfloat16),
                                           torch.tensor([-0.0], dtype=torch.bfloat16)).item(), 0)
        self.assertEqual(bf16_ulp_distance(one, -one).item(), 2 * 0x3F80)   # both sides of zero

    def test_identical_traces(self):
        a = self.trace(7)
        result = compare_trace(a, {k: v.clone() for k, v in a.items()})
        self.assertEqual((result["positions_compared"], result["argmax_mismatches"],
                          result["route_id_mismatches"], result["logits_max_abs_diff"]), (7, 0, 0, 0.0))

    def test_comparison_stops_at_first_divergent_choice(self):
        a = self.trace(9)
        b = {k: v.clone() for k, v in a.items()}
        other = (int(a["logits"][4].argmax()) + 1) % a["logits"].shape[1]
        b["logits"][4, other] = a["logits"][4].max() + 8        # position 4 now prefers another token
        result = compare_trace(a, b)
        self.assertEqual((result["first_divergent_position"], result["positions_compared"],
                          result["argmax_mismatches"]), (4, 5, 1))

    def test_compare_runs_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            first = self.trace(6, seed=3)
            save_trace(a, "test:zh-my:12", first)
            save_trace(b, "test:zh-my:12", {k: v.clone() for k, v in first.items()})
            loaded = load_trace(a, "test:zh-my:12")
            self.assertTrue(torch.equal(loaded["logits"], first["logits"]))
            self.assertTrue(torch.equal(loaded["route_ids"], first["route_ids"]))
            self.assertTrue(torch.equal(loaded["route_w"], first["route_w"]))
            text = {"test:zh-my:12": "same"}
            passing = compare_runs(a, b, text, text)
            self.assertTrue(passing["pass"])
            self.assertEqual((passing["bitwise_identical_logits"], passing["positions_compared"]), (1, 6))
            diverged = self.trace(6, seed=4)
            save_trace(b, "test:zh-my:12", diverged)
            failing = compare_runs(a, b, text, {"test:zh-my:12": "different"})
            self.assertFalse(failing["pass"])
        self.assertEqual(EQUIV_ABS_TOL, 0.125)


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
