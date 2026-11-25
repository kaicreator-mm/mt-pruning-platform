"""The engine against the reference implementation shipped in transformers.

Two reference configurations of the pinned release matter:
  default        eager attention + the default ``grouped_mm`` experts path, i.e.
                 what the model card's snippet runs. The engine is tested to be
                 bit-identical to it.
  eager experts  the per-expert loop in ``modeling_hy_v3`` (bf16 accumulation, an
                 equally official path). The engine must stay within LOGIT_TOL.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from mtpruning.engine import Checkpoint, Engine, expert_combine
from mtpruning.prune import verify_pruned, write_pruned
from mtpruning.testing import tiny_checkpoint

SPARSE = [1, 2]
KEPT = {1: [0, 2, 4, 6], 2: [1, 2, 5, 7]}
# bf16 logits: two representable steps where the top logits sit (16–32).
LOGIT_TOL = 0.125


def reference(path, **options):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **options).eval()


class EngineVsReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = Checkpoint(tiny_checkpoint(cls.root / "tiny", cls.root / "none", seed=7))
        torch.manual_seed(0)
        cls.prompts = [torch.randint(0, 120000, (n,)).tolist() for n in (1, 2, 9, 17)]
        cls.model = reference(cls.source.path, attn_implementation="eager")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pruned(self, kept) -> Engine:
        """Write ``kept`` into a fresh directory and reload it."""
        target = Path(tempfile.mkdtemp(dir=self.root)) / "model"
        write_pruned(self.source, kept, target)
        return Engine(Checkpoint(target))

    def assertIdentical(self, first: Engine, second: Engine, max_new_tokens=12):
        for ids in self.prompts:
            a = first.generate(ids, max_new_tokens, trace=True)
            b = second.generate(ids, max_new_tokens, trace=True)
            self.assertEqual((a.tokens, a.finish), (b.tokens, b.finish))
            self.assertTrue(torch.equal(a.trace["route_ids"], b.trace["route_ids"]), "routing ids differ")
            self.assertTrue(torch.equal(a.trace["route_w"], b.trace["route_w"]), "routing weights differ")
            self.assertTrue(torch.equal(a.trace["logits"], b.trace["logits"]), "logits differ")

    def test_prefill_logits_are_identical_to_the_reference(self):
        engine = Engine(self.source)
        for ids in self.prompts:
            with torch.inference_mode():
                want = self.model(torch.tensor([ids])).logits[0]
            got = engine.forward(ids, engine.new_cache(len(ids) + 4), "prefill")
            self.assertTrue(torch.equal(got, want), f"max abs diff {(got.float() - want.float()).abs().max()}")

    def test_decode_logits_are_identical_step_by_step(self):
        from transformers import DynamicCache
        engine = Engine(self.source)
        ids = self.prompts[2]
        cache, reference_cache = engine.new_cache(len(ids) + 8), DynamicCache(config=self.model.config)
        with torch.inference_mode():
            want = self.model(torch.tensor([ids]), cache=reference_cache).logits[0, -1]
        got = engine.forward(ids, cache, "prefill")[-1]
        for _ in range(6):
            self.assertTrue(torch.equal(got, want), f"max abs diff {(got.float() - want.float()).abs().max()}")
            token = int(want.float().argmax())
            with torch.inference_mode():
                want = self.model(torch.tensor([[token]]), cache=reference_cache).logits[0, -1]
            got = engine.forward([token], cache, "decode")[-1]

    def test_greedy_tokens_match_the_reference(self):
        engine = Engine(self.source)
        for ids in self.prompts:
            with torch.inference_mode():
                output = self.model.generate(input_ids=torch.tensor([ids]), max_new_tokens=12, do_sample=False)
            self.assertEqual(engine.generate(ids, 12).tokens, output[0, len(ids):].tolist())

    def test_expert_combine_is_the_reference_experts_path(self):
        experts = self.model.model.layers[1].mlp.experts
        torch.manual_seed(1)
        count, inter, top_k = experts.num_experts, experts.intermediate_dim, self.model.config.num_experts_per_tok
        x = torch.randn(37, experts.hidden_dim).bfloat16()
        top = torch.stack([torch.randperm(count)[:top_k] for _ in range(37)])
        weights = torch.rand(37, top_k)
        weights = weights / weights.sum(-1, keepdim=True) * self.model.config.router_scaling_factor
        with torch.inference_mode():
            want = experts(x, top, weights)
            got = expert_combine(x, top, weights, [w[:inter] for w in experts.gate_up_proj],
                                 [w[inter:] for w in experts.gate_up_proj], list(experts.down_proj))
        self.assertTrue(torch.equal(got, want))

    def test_split_linear_equals_fused_grouped_mm_at_model_shapes(self):
        # The one assumption behind zero-copy expert weights, at the real dimensions.
        torch.manual_seed(2)
        hidden, inter = 2048, 768
        gate_up = (torch.randn(2 * inter, hidden) * 0.02).bfloat16()
        down = (torch.randn(hidden, inter) * 0.02).bfloat16()
        for rows in (1, 3, 13, 97):
            x, offsets = torch.randn(rows, hidden).bfloat16(), torch.tensor([rows], dtype=torch.int32)
            a = torch.randn(rows, inter).bfloat16()
            with torch.inference_mode():
                fused = torch._grouped_mm(x, gate_up.T[None], offs=offsets)
                split = torch.cat((torch.nn.functional.linear(x, gate_up[:inter]),
                                   torch.nn.functional.linear(x, gate_up[inter:])), dim=1)
                down_stock = torch._grouped_mm(a, down.T[None], offs=offsets)
            self.assertTrue(torch.equal(split, fused))
            self.assertTrue(torch.equal(torch.nn.functional.linear(a, down), down_stock))

    def test_stays_within_tolerance_of_the_eager_experts_path(self):
        eager = reference(self.source.path, attn_implementation="eager", experts_implementation="eager")
        engine = Engine(self.source)
        for ids in self.prompts:
            with torch.inference_mode():
                want = eager(torch.tensor([ids])).logits[0]
            got = engine.forward(ids, engine.new_cache(len(ids) + 4), "prefill")
            self.assertLessEqual((got.float() - want.float()).abs().max().item(), LOGIT_TOL)

    def test_masked_model_equals_physical_prune(self):
        pruned = self.pruned(KEPT)
        self.assertEqual(pruned.ckpt.num_experts, 4)
        self.assertTrue(verify_pruned(self.source, pruned.ckpt)["pass"])
        masked = Engine(self.source)
        masked.set_mask(KEPT)
        self.assertIdentical(masked, pruned)

    def test_mask_removes_experts(self):
        # The mask has to remove experts, otherwise the equivalence tests would be vacuous.
        free = Engine(self.source).generate(self.prompts[3], 4, trace=True)
        masked_engine = Engine(self.source)
        masked_engine.set_mask({1: [0, 1], 2: [0, 1]})
        seen = []
        masked_engine.audit = lambda phase, changed, lost: seen.append((phase, changed, lost))
        masked = masked_engine.generate(self.prompts[3], 4, trace=True)
        self.assertEqual(set(masked.trace["route_ids"].flatten().tolist()), {0, 1})
        self.assertGreater(len(set(free.trace["route_ids"].flatten().tolist())), 2)
        self.assertTrue(any(changed > 0 for _, changed, _ in seen))
        self.assertTrue(all(lost >= 0 for _, _, lost in seen))

    def test_identity_prune_is_a_noop(self):
        identity = {layer: list(range(8)) for layer in SPARSE}
        pruned = self.pruned(identity)
        self.assertTrue(verify_pruned(self.source, pruned.ckpt)["identity_mapping"])
        self.assertIdentical(Engine(self.source), pruned)

    def test_reference_loader_opens_the_pruned_checkpoint(self):
        from transformers import AutoModelForCausalLM
        pruned = self.pruned(KEPT)
        model, reports = AutoModelForCausalLM.from_pretrained(pruned.ckpt.path, dtype=torch.bfloat16,
                                                              attn_implementation="eager", output_loading_info=True)
        for kind in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertEqual(sorted(reports[kind]), [], kind)
        self.assertEqual(model.config.num_experts, 4)
        for ids in self.prompts:
            with torch.inference_mode():
                output = model.eval().generate(input_ids=torch.tensor([ids]), max_new_tokens=8, do_sample=False)
            self.assertEqual(pruned.generate(ids, 8).tokens, output[0, len(ids):].tolist())

    def test_forward_never_writes_into_weights(self):
        names = ["model.embed_tokens.weight", "model.layers.1.mlp.router.gate.weight",
                 "model.layers.1.mlp.expert_bias", "model.layers.2.mlp.experts.3.gate_proj.weight",
                 "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]
        before = {name: self.source.tensor(name).clone() for name in names}
        masked = Engine(self.source)
        masked.set_mask(KEPT)
        masked.generate(self.prompts[2], 6, trace=True)
        # A fresh Checkpoint maps the files again, so an in-process copy-on-write would show up here.
        fresh = Checkpoint(self.source.path)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(fresh.tensor(name), tensor), name)

    def test_bad_masks_are_rejected(self):
        engine = Engine(self.source)
        for bad in ({1: [0, 1, 2, 3]}, {1: [0], 2: [0, 1, 2, 3]}, {1: [0, 0, 1, 2], 2: [0, 1, 2, 3]},
                    {1: [0, 1, 2, 9], 2: [0, 1, 2, 3]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                engine.set_mask(bad)
            engine.set_mask(None)

    def test_context_limit(self):
        engine = Engine(self.source)
        with self.assertRaises(ValueError):
            engine.generate([1] * (engine.max_context + 1), 1)


if __name__ == "__main__":
    unittest.main()
