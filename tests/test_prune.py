import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from mtpruning.calibration import retained_mass, select
from mtpruning.engine import Checkpoint, expected_keys
from mtpruning.prune import save_shard, sha256_file, validate_kept, verify_pruned, write_pruned
from mtpruning.testing import tiny_checkpoint

SPARSE = [1, 2]
KEPT = {1: [0, 2, 4, 6], 2: [1, 2, 5, 7]}


class SelectTests(unittest.TestCase):
    def test_directions_are_normalised_before_averaging(self):
        # One direction carries 100x the routing mass but the same profile: the ranking must not move.
        torch.manual_seed(0)
        base = torch.rand(1, 3, 8).double()
        mass = torch.cat([base * 100, base.repeat(5, 1, 1)])
        expected = [sorted(sorted(range(8), key=lambda i: (-base[0, layer, i], i))[:4]) for layer in range(3)]
        self.assertEqual(select(mass, 4, 2), expected)

    def test_outvoted_favourite_is_dropped(self):
        mass = torch.ones(6, 1, 8).double()
        mass[0, 0, 7] = 3                        # only the first direction favours expert 7 ...
        mass[1:, 0, 0] = 3                       # ... the other five favour expert 0
        self.assertEqual(select(mass, 1, 1), [[0]])
        retained = retained_mass(mass, [[0]])
        self.assertAlmostEqual(retained[0, 0].item(), 0.1)
        self.assertTrue(torch.all(retained[1:] == 0.3))

    def test_ties_break_on_lower_index(self):
        self.assertEqual(select(torch.ones(6, 2, 8).double(), 3, 2), [[0, 1, 2]] * 2)

    def test_negatives(self):
        mass = torch.ones(6, 2, 8).double()
        bad = [mass[:5], mass.repeat(2, 1, 1), mass * float("nan"), -mass, mass[:, :, :1] * 0]
        for profile in bad:
            with self.assertRaises(ValueError):
                select(profile, 4, 2)
        for keep in (1, 9):
            with self.assertRaises(ValueError):
                select(mass, keep, 2)


class ShardTests(unittest.TestCase):
    def test_hand_written_shard_reads_like_a_library_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = torch.randn(300, 64).bfloat16()
            named = [("a", big), ("b", big[10:20]), ("c", torch.arange(6, dtype=torch.float32))]
            save_shard(Path(tmp) / "hand.safetensors", named, {"format": "pt"})
            # safetensors.torch.save_file refuses views into one storage, hence the hand-written writer.
            with self.assertRaises(RuntimeError):
                save_file(dict(named), Path(tmp) / "lib.safetensors", metadata={"format": "pt"})
            loaded = load_file(Path(tmp) / "hand.safetensors")
            for name, tensor in named:
                self.assertTrue(torch.equal(loaded[name], tensor))
            self.assertEqual(loaded["c"].dtype, torch.float32)
            with safe_open(Path(tmp) / "hand.safetensors", framework="pt") as handle:
                self.assertEqual(handle.metadata(), {"format": "pt"})
                self.assertEqual(sorted(handle.keys()), ["a", "b", "c"])


class PruneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = Checkpoint(tiny_checkpoint(cls.root / "source", cls.root / "no-tokenizer", seed=3))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def prune(self, kept) -> Path:
        """Write ``kept`` into a fresh directory; returns it."""
        target = Path(tempfile.mkdtemp(dir=self.root))
        write_pruned(self.source, kept, target)
        return target

    def test_physical_prune_reloads_with_the_expected_mapping(self):
        folder = self.prune(KEPT)
        pruned = Checkpoint(folder)
        result = verify_pruned(self.source, pruned)
        self.assertTrue(result["pass"], result["first_mismatches"])
        self.assertFalse(result["identity_mapping"])
        self.assertEqual(result["tensors_checked"], len(pruned.weight_map))
        self.assertEqual(json.loads((folder / "config.json").read_text())["num_experts"], 4)
        self.assertEqual(pruned.num_experts, 4)
        self.assertEqual(pruned.kept, KEPT)
        self.assertEqual(set(pruned.weight_map), expected_keys(pruned.config, pruned.sparse))
        # router rows and biases follow their experts; non-routed tensors are untouched
        for layer, ids in KEPT.items():
            prefix = f"model.layers.{layer}.mlp."
            self.assertTrue(torch.equal(pruned.tensor(prefix + "router.gate.weight"),
                                        self.source.tensor(prefix + "router.gate.weight")[ids]))
            self.assertTrue(torch.equal(pruned.tensor(prefix + "expert_bias"),
                                        self.source.tensor(prefix + "expert_bias")[ids]))
            for local, origin in enumerate(ids):
                for kind in ("gate", "up", "down"):
                    self.assertTrue(torch.equal(pruned.tensor(f"{prefix}experts.{local}.{kind}_proj.weight"),
                                                self.source.tensor(f"{prefix}experts.{origin}.{kind}_proj.weight")))
        self.assertTrue(torch.equal(pruned.tensor("model.embed_tokens.weight"),
                                    self.source.tensor("model.embed_tokens.weight")))

    def test_shards_match_their_manifest(self):
        folder = self.prune(KEPT)
        manifest = json.loads((folder / "prune_manifest.json").read_text())
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        self.assertEqual(index["metadata"]["total_size"], manifest["tensor_bytes"])
        self.assertGreaterEqual(sum(f["bytes"] for f in manifest["files"]), manifest["tensor_bytes"])
        for shard in manifest["files"]:
            path = folder / shard["name"]
            self.assertEqual(path.stat().st_size, shard["bytes"])
            self.assertEqual(sha256_file(path), shard["sha256"])
            with safe_open(path, framework="pt") as handle:
                stored = sorted(handle.keys())
            self.assertEqual(stored, sorted(n for n, s in index["weight_map"].items() if s == shard["name"]))

    def test_identity_prune_is_a_noop_copy(self):
        identity = {layer: list(range(8)) for layer in SPARSE}
        result = verify_pruned(self.source, Checkpoint(self.prune(identity)))
        self.assertTrue(result["pass"] and result["identity_mapping"])

    def test_bad_mappings_are_rejected(self):
        bad = [
            {1: [0, 1, 2, 3]},                                        # a sparse layer is missing
            {1: [0, 1, 2, 2], 2: [0, 1, 2, 3]},                       # duplicate
            {1: [0, 1, 2, 8], 2: [0, 1, 2, 3]},                       # out of range
            {1: [0, 1, 3, 2], 2: [0, 1, 2, 3]},                       # not sorted: would reorder experts
            {1: [0, 1], 2: [0, 1, 2, 3]},                             # fewer than top_k
            {1: [0, 1, 2, 3], 2: [0, 1, 2]},                          # different widths
            {1: [0, 1, 2, True], 2: [0, 1, 2, 3]},                    # bool is not an expert id
            {1: [0, 1, 2, 3.0], 2: [0, 1, 2, 3]},                      # float
            {3: [0, 1, 2, 3], 1: [0, 1, 2, 3]},                       # dense layer named
        ]
        for kept in bad:
            with self.assertRaises(ValueError, msg=str(kept)):
                validate_kept(kept, self.source.sparse_layers, self.source.num_experts, 2)
            with self.assertRaises(ValueError):
                self.prune(kept)

    def test_refuses_to_overwrite(self):
        folder = self.prune(KEPT)
        with self.assertRaises(ValueError):
            write_pruned(self.source, KEPT, folder)
        with self.assertRaises(ValueError):
            write_pruned(self.source, {1: [0, 1]}, folder)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        tiny_checkpoint(self.path / "model", self.path / "none", seed=1)
        self.index = self.path / "model" / "model.safetensors.index.json"
        self.config = self.path / "model" / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def rewrite(self, path, change):
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def test_rejects_unknown_layout_and_unsupported_config(self):
        self.assertEqual(Checkpoint(self.path / "model").sparse, [False, True, True])
        cases = [
            (self.index, lambda d: d["weight_map"].update({"model.layers.1.mlp.unknown": d["weight_map"].pop("model.layers.1.mlp.expert_bias")})),
            (self.index, lambda d: d["weight_map"].pop("lm_head.weight")),
            (self.config, lambda d: d.update(moe_router_use_sigmoid=False)),
            (self.config, lambda d: d.update(num_shared_experts=2)),
            (self.config, lambda d: d.update(tie_word_embeddings=True)),
            (self.config, lambda d: d.update(rope_scaling={"rope_type": "yarn"})),
            (self.config, lambda d: d.update(mlp_layer_types=["dense", "dense", "sparse"])),
        ]
        for path, change in cases:
            original = path.read_text()
            self.rewrite(path, change)
            with self.assertRaises(ValueError, msg=str(path)):
                Checkpoint(self.path / "model")
            path.write_text(original)


if __name__ == "__main__":
    unittest.main()
