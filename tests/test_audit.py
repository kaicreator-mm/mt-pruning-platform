"""R11-P1-01 regression: mask audit aggregation.

The audit rows accumulate across samples; they used to be overwritten, so two
witness samples of one direction and phase kept only the last one's counts.
Reproducer from the review: [8,3,.5] + [12,4,.75] = [20,7,1.25], never [12,4,.75].
"""
import math
import unittest

from mtpruning.calibration import AUDIT_FIELDS, PHASES, rebuild, validate_payload

SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW = [1], 8, 2, 16


def entry(tokens: int, changed: int, lost: float) -> dict:
    """One phase's audit: decisions are exactly the phase's tokens times the sparse layer count."""
    return {"decisions": tokens * len(SPARSE_LAYERS), "changed": changed, "lost_mass": lost}


def sample(sample_id: str, line: int, prompt_tokens: int, decode_tokens: int, audit: dict) -> dict:
    """One well-formed committed sample whose routing and audit agree by construction."""
    routing = {}
    for phase, tokens in (("prefill", prompt_tokens), ("decode", decode_tokens)):
        # one cell row per (layer, expert) pair: two experts of the single sparse layer
        routing[phase] = ({"tokens": tokens, "layer": [1, 1], "expert": [0, 1],
                           "count": [tokens, tokens], "mass": [0.25 * tokens, 0.75 * tokens]}
                          if tokens else {"tokens": 0, "layer": [], "expert": [], "count": [], "mass": []})
    return {"identity_key": "a" * 64, "split": "calibration", "direction": "zh-my", "line": line,
            "prompt_tokens": prompt_tokens, "new_tokens": decode_tokens, "finish": "eos",
            "tokens": [7] * decode_tokens, "text": "out", "routing": routing, "audit": audit, "trace": None}


class MaskAuditAccumulationTests(unittest.TestCase):
    @staticmethod
    def contexts(samples):
        return {s["id"]: {"split": "calibration", "direction": "zh-my", "line": s["line"],
                          "prompt_ids": [1] * s["prompt_tokens"]} for s in samples}

    def two_witness_samples(self):
        # The review's exact numbers, prefill: [8, 3, .5] then [12, 4, .75] on one direction.
        first = sample("cal:zh-my:0", 0, 8, 2, {"prefill": entry(8, 3, 0.5), "decode": entry(2, 1, 0.25)})
        second = sample("cal:zh-my:1", 1, 12, 3, {"prefill": entry(12, 4, 0.75), "decode": entry(3, 2, 0.5)})
        return [{"id": sample_id, **payload}
                for sample_id, payload in zip((first["identity_key"] + "0", first["identity_key"] + "1"),
                                              (first, second))]

    def test_audit_accumulates_across_samples(self):
        samples = self.two_witness_samples()
        stats = rebuild(samples, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, self.contexts(samples),
                        audit_required=True)
        # prefill: [8,3,.5] + [12,4,.75] = [20,7,1.25]; decode accumulates the same way.
        prefill, decode = stats["audit"][0, 0], stats["audit"][0, 1]
        self.assertEqual(prefill[0].item(), 8 + 12)
        self.assertEqual(prefill[1].item(), 3 + 4)
        self.assertEqual(prefill[2].item(), 0.5 + 0.75)
        self.assertEqual((decode[0].item(), decode[1].item(), decode[2].item()), (2 + 3, 1 + 2, 0.25 + 0.5))

    def test_totals_match_an_independent_sum(self):
        samples = self.two_witness_samples()
        stats = rebuild(samples, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, self.contexts(samples),
                        audit_required=True)
        for phase_index, phase in enumerate(PHASES):
            for field_index, field in enumerate(AUDIT_FIELDS):
                independent = math.fsum(s["audit"][phase][field] for s in samples)
                self.assertAlmostEqual(float(stats["audit"][0, phase_index, field_index]), independent, places=12)

    def test_run_kind_requires_audit_presence(self):
        samples = self.two_witness_samples()
        contexts = self.contexts(samples)
        rebuild(samples, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, contexts, audit_required=True)
        stripped = [{**s, "audit": None} for s in samples]
        with self.assertRaisesRegex(ValueError, "does not match the run kind"):
            rebuild(stripped, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, contexts, audit_required=True)
        rebuild(stripped, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, contexts, audit_required=False)
        with self.assertRaisesRegex(ValueError, "does not match the run kind"):
            rebuild(samples, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, contexts, audit_required=False)
        mixed = [samples[0], {**samples[1], "audit": None}]
        with self.assertRaisesRegex(ValueError, "mixed masked and unmasked"):
            rebuild(mixed, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, contexts)

    def test_decisions_must_equal_tokens_times_sparse_layer_count(self):
        context = {"split": "calibration", "direction": "zh-my", "line": 0, "prompt_ids": [1] * 8}
        payload = sample("cal:zh-my:0", 0, 8, 2, {"prefill": entry(8, 3, 0.5), "decode": entry(2, 1, 0.25)})
        validate_payload(payload, context, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, audit_required=True)
        for phase in PHASES:                       # one decision fewer than the phase saw
            payload["audit"][phase]["decisions"] -= 1
        with self.assertRaisesRegex(ValueError, "decisions disagree"):
            validate_payload(payload, context, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, audit_required=True)
        payload = sample("cal:zh-my:0", 0, 8, 2, {"prefill": entry(8, 3, 0.5), "decode": entry(2, 1, 0.25)})
        for phase in PHASES:                       # an audit that counted more changes than decisions
            payload["audit"][phase]["changed"] += 100
        with self.assertRaisesRegex(ValueError, "changed more decisions than it saw"):
            validate_payload(payload, context, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, audit_required=True)
        payload = sample("cal:zh-my:0", 0, 8, 2, {"prefill": entry(8, 3, 0.5), "decode": entry(2, 1, 0.25)})
        for phase in PHASES:
            payload["audit"][phase]["lost_mass"] = payload["audit"][phase]["changed"] + 0.5
        with self.assertRaisesRegex(ValueError, "lost more weight"):
            validate_payload(payload, context, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, audit_required=True)
        payload = sample("cal:zh-my:0", 0, 8, 2, {"prefill": entry(8, 3, 0.5), "decode": entry(2, 1, 0.25)})
        payload["audit"]["prefill"]["lost_mass"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid mask audit"):
            validate_payload(payload, context, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, audit_required=True)

    def test_aggregate_decisions_are_rechecked_against_routing(self):
        samples = self.two_witness_samples()
        contexts = self.contexts(samples)
        for phase in PHASES:                       # an audit whose totals do not match the routing
            samples[0]["audit"][phase]["decisions"] += 1
        with self.assertRaisesRegex(ValueError, "mask audit decisions disagree"):
            rebuild(samples, SPARSE_LAYERS, NUM_EXPERTS, TOP_K, MAX_NEW, contexts, audit_required=True)


if __name__ == "__main__":
    unittest.main()
