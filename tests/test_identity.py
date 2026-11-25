"""Run identity and checkpoint provenance: what a recorded sample is bound to."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mtpruning import identity
from mtpruning.testing import tiny_checkpoint
from tests import util
from tests.util import READY


class CheckpointProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = tiny_checkpoint(self.root / "model", self.root / "none", seed=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_declared_checkpoint_is_bound_by_content(self):
        first = identity.checkpoint_identity(self.model)
        self.assertEqual(first["kind"], "synthetic")
        self.assertEqual(first["num_experts"], 8)
        self.assertEqual(identity.checkpoint_identity(self.model), first)
        shard = next(self.model.glob("model-0000[12]*.safetensors"))
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "differs from its declared provenance"):
            identity.checkpoint_identity(self.model)

    def test_a_directory_of_unknown_provenance_is_refused(self):
        (self.model / "checkpoint_provenance.json").unlink()
        with self.assertRaisesRegex(ValueError, "provenance unknown"):
            identity.checkpoint_identity(self.model)
        (self.model / "checkpoint_provenance.json").write_text('{"kind": "whatever"}')
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint provenance"):
            identity.checkpoint_identity(self.model)

    def test_pinned_listing(self):
        (self.model / "checkpoint_provenance.json").unlink()
        names = ["config.json", "model.safetensors.index.json", *sorted({shard.name for shard in self.model.glob("*.safetensors")})]
        pinned = {name: identity.sha256_file(self.model / name) for name in names}
        with mock.patch.object(identity, "pinned_manifest", return_value=pinned):
            bound = identity.checkpoint_identity(self.model)
        self.assertEqual(bound["kind"], "pinned-snapshot")
        self.assertEqual((bound["repo"], bound["revision"]),
                         ("tencent/Hy-MT3-30B-A3B", "08ae9acea91abe265c6236e757e49d8851e1ce06"))
        with mock.patch.object(identity, "pinned_manifest", return_value={**pinned, names[-1]: "0" * 64}):
            with self.assertRaisesRegex(ValueError, "differs from the pinned revision"):
                identity.checkpoint_identity(self.model)
        with mock.patch.object(identity, "pinned_manifest", return_value={name: pinned[name] for name in names[:-1]}):
            with self.assertRaisesRegex(ValueError, "provenance unknown"):
                identity.checkpoint_identity(self.model)

    def test_hashes_are_reused_while_size_and_mtime_are_unchanged(self):
        cache = self.root / "cache"
        names = ["config.json", *sorted(shard.name for shard in self.model.glob("*.safetensors"))]
        real = identity.sha256_file
        with mock.patch.object(identity, "sha256_file", side_effect=real) as hashing:
            first = identity.content_hashes(self.model, names, cache)
            self.assertEqual(hashing.call_count, len(names))
            self.assertEqual(identity.content_hashes(self.model, names, cache), first)
            self.assertEqual(hashing.call_count, len(names))          # nothing rehashed
            shard = self.model / names[-1]
            status = shard.stat()
            os.utime(shard, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000))
            self.assertEqual(identity.content_hashes(self.model, names, cache), first)
            self.assertEqual(hashing.call_count, len(names) + 1)      # only the touched file
        self.assertEqual(identity.content_hashes(self.model, names), first)   # no cache: every file again


@unittest.skipUnless(READY, util.__doc__)
class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.world = util.make_world(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def identity(self, **overrides):
        from mtpruning.pipeline import corpus_paths
        arguments = dict(model_dir=self.world["model"], corpus_path=self.world["corpus"],
                         manifest_path=corpus_paths(self.world["run"])[0], cache_dir=self.world["cache"],
                         decoding="greedy (argmax of bf16 logits)", max_new_tokens=6,
                         protocol={"splits": ["calibration", "validation"]}, threads=2)
        arguments.update(overrides)
        return identity.run_identity(**arguments)

    def test_fields_follow_the_reviewed_contract(self):
        from mtpruning.reviewed import cache_plan
        mine = self.identity()
        self.assertTrue(set(identity.FIELDS) <= set(mine))
        binding = {name: mine[name] for name in identity.FIELDS}
        for split in ("calibration", "validation", "test"):
            self.assertEqual(mine["sample_keys"][split], cache_plan.key({**binding, "split": split}))
        self.assertNotEqual(mine["sample_keys"]["validation"], mine["sample_keys"]["test"])
        self.assertEqual(mine["detail"]["checkpoint"]["kind"], "synthetic")

    def test_every_dependency_moves_the_key(self):
        reference = self.identity()["sample_keys"]["calibration"]
        selection = self.world["run"] / "selection.json"
        both = [0, 1, 2, 3, 4, 5, 6, 7]
        selection.write_text(json.dumps({"candidates": {"96": {"kept": {"1": both, "2": both}}}}))
        variants = {
            "decoding settings": dict(max_new_tokens=5),
            "decoding rule": dict(decoding="sampled"),
            "protocol": dict(protocol={"splits": ["calibration"]}),
            "thread count": dict(threads=3),
            "mask": dict(mask={1: both, 2: both}, selection_sha256=util.sha256(selection)),
            "mask of another candidate": dict(mask={1: both, 2: both[:-1] + [8]}, selection_sha256=util.sha256(selection)),
            "selection file": dict(mask={1: both, 2: both}, selection_sha256="0" * 64),
        }
        for what, change in variants.items():
            with self.subTest(what):
                self.assertNotEqual(self.identity(**change)["sample_keys"]["calibration"], reference)
        with mock.patch.object(identity, "code_identity", return_value={"files": {"x.py": "1" * 64}}):
            self.assertNotEqual(self.identity()["sample_keys"]["calibration"], reference)
        self.assertEqual(self.identity()["sample_keys"]["calibration"], reference)

    def test_corpus_bytes_must_match_the_manifest(self):
        corpus = self.world["corpus"]
        original = corpus.read_bytes()
        corpus.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "corpus bytes differ from the manifest"):
            self.identity()
        corpus.write_bytes(original)

    def test_a_checkpoint_without_a_tokenizer_is_refused(self):
        for name in os.listdir(self.world["model"]):
            if name.startswith("tokenizer") or name == "chat_template.jinja":
                (self.world["model"] / name).unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            self.identity()
