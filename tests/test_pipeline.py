"""Stage-level checks of the pipeline on a dry-run world (see tests/util.py)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mtpruning import pipeline
from mtpruning.pipeline import read_json
from tests import util
from tests.util import READY


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@unittest.skipUnless(READY, util.__doc__)
class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.world = util.make_world(Path(cls.tmp.name))
        util.rehearse(cls.world, stock_check=False)
        pipeline.CRASH_AFTER = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, change=None):
        run = Path(tempfile.mkdtemp(dir=self.world["root"])) / "run"
        shutil.copytree(self.world["run"], run)
        if change:
            change(run)
        return run

    def test_corpus_check(self):
        run, receipt = self.world["run"], self.world["run"] / "corpus_receipt.json"
        util.cli(self.world, "corpus-check", "--corpus-dir", run / "corpus", "--out", receipt)
        data = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "FROZEN_CORPUS_CONTRACT_OK")
        self.assertEqual(data["records"], 6 * sum(util.SIZES.values()))
        self.assertEqual(data["manifest_sha256"], self.world["manifest_sha256"])
        bad = self.copy(lambda run: (run / "corpus" / "corpus_manifest.json").write_text(
            (run / "corpus" / "corpus_manifest.json").read_text(encoding="utf-8") + " ", encoding="utf-8"))
        with self.assertRaisesRegex(SystemExit, "BLOCKED_CORPUS_CONTRACT"):
            util.cli(self.world, "corpus-check", "--corpus-dir", bad / "corpus", "--out", bad / "receipt.json")
        # The committed frozen manifest governs when no other hash is pinned.
        with self.assertRaisesRegex(SystemExit, "BLOCKED_CORPUS_CONTRACT"):
            pipeline.main(["corpus-check", "--corpus-dir", str(run / "corpus"), "--out", str(run / "never.json")])
        self.assertFalse((run / "never.json").exists())

    def test_checkpoint_check(self):
        out = self.world["root"] / "checkpoint.json"
        pipeline.main(["checkpoint-check", "--model", str(self.world["model"]), "--cache-dir", str(self.world["cache"]),
                       "--out", str(out)])
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((data["status"], data["kind"], data["num_experts"]), ("CHECKPOINT_CONTENT_VERIFIED", "synthetic", 128))
        self.assertEqual(len(data["files"]), 4)          # config, index and two shards
        self.assertEqual(data["teacher"], read_json(self.world["run"] / "baseline" / "records.jsonl.identity.json")["teacher"])

    def test_run_refuses_the_test_split_outside_the_final_test_stage(self):
        with self.assertRaisesRegex(SystemExit, "belongs to the final-test stage"):
            pipeline.execute_run(util.options(self.world, splits=["test"]))
        with self.assertRaisesRegex(SystemExit, "belongs to the final-test stage"):
            pipeline.execute_run(util.options(self.world, splits=["validation", "test"]))

    def test_equiv_refuses_pairs_it_is_not_defined_for(self):
        run = self.world["run"]
        common = ["equiv", "--run-dir", run, "--trace-n", 2,
                  "--reference-records", run / "baseline" / "records.jsonl",
                  "--reference-traces", run / "baseline" / "traces"]

        def candidate(keep):
            folder = run / f"k{keep}"
            return ["--candidate-records", folder / "mechanism" / "records.jsonl",
                    "--candidate-traces", folder / "mechanism" / "traces", "--prune-record", folder / "prune.json",
                    "--out", run / "never.json"]
        with self.assertRaisesRegex(SystemExit, "not the two runs"):
            util.cli(self.world, *common, "--kind", "noop", "--keep", 96, *candidate(96))     # a real prune is no no-op
        with self.assertRaisesRegex(SystemExit, "not the two runs"):
            util.cli(self.world, *common, "--kind", "noop", "--keep", 64, *candidate(128))    # wrong expert count
        masked = run / "k96" / "masked"
        with self.assertRaisesRegex(SystemExit, "not the two runs"):
            util.cli(self.world, "equiv", "--run-dir", run, "--kind", "masked", "--keep", 64, "--trace-n", 2,
                     "--reference-records", masked / "records.jsonl", "--reference-traces", masked / "traces",
                     "--selection", run / "selection.json", *candidate(96))
        incomplete = self.copy(lambda run: next((run / "baseline" / "traces").glob("*.trace.npz")).unlink())
        with self.assertRaisesRegex(ValueError, "reference traces cover"):
            util.cli(self.world, "equiv", "--run-dir", incomplete, "--kind", "noop", "--keep", 128, "--trace-n", 2,
                     "--reference-records", incomplete / "baseline" / "records.jsonl",
                     "--reference-traces", incomplete / "baseline" / "traces",
                     "--candidate-records", incomplete / "k128" / "mechanism" / "records.jsonl",
                     "--candidate-traces", incomplete / "k128" / "mechanism" / "traces",
                     "--prune-record", incomplete / "k128" / "prune.json", "--out", incomplete / "never.json")
        self.assertFalse((incomplete / "never.json").exists())

    def test_decide_refuses_mixed_identity_and_unnamed_candidates(self):
        def mixed(run):
            path = run / "k96" / "records.jsonl"
            rows = lines(path)
            rows[0]["identity"]["teacher"] = "0" * 64
            write(path, rows)

        def renamed(run):
            path = run / "k96" / "prune.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["manifest"]["kept"]["1"] = list(reversed(data["manifest"]["kept"]["1"]))
            path.write_text(json.dumps(data), encoding="utf-8")

        def exposed(run):
            path = run / "k64" / "records.jsonl"
            rows = lines(path)
            rows[0]["splits"] = ["validation", "test"]
            write(path, rows)
        for change, message in ((mixed, "differs in teacher"), (renamed, "not pruned the way the selection names"),
                                (exposed, "candidate test already exposed")):
            with self.subTest(message):
                run = self.copy(change)
                with self.assertRaisesRegex(SystemExit, message):
                    util.cli(self.world, "decide", "--run-dir", run, "--candidates", "96,64", "--out", run / "never.json")

    def test_selection_and_decision_are_bound_to_the_run_identity(self):
        run = self.world["run"]
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        baseline = json.loads((run / "baseline" / "records.jsonl.identity.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["identity"]["teacher"], baseline["teacher"])
        self.assertEqual(selection["identity"]["calibration_key"],
                         json.loads((run / "baseline" / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
                         ["identity_keys"]["calibration"])
        self.assertEqual(sorted(selection["aggregate_hashes"]), ["corpus", "records", "stats_json", "stats_npz"])
        decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["identity"]["corpus"], selection["identity"]["corpus"])
        self.assertNotIn("test", decision["runs"]["k64"]["validation_key"])     # keys are hashes
        self.assertEqual(decision["runs"]["k64"]["records_sha256"], util.sha256(run / "k64" / "records.jsonl"))

    def test_the_dry_run_really_routes_around_pruned_experts(self):
        """The masked equivalence would prove nothing if the mask never changed a routing decision.

        On the random checkpoint the pruned experts are chosen on held-out validation sentences,
        so the comparison covers the case the mechanism exists for.
        """
        run = self.world["run"]
        for keep, least_prefill, least_decode in ((96, 1, 0), (64, 20, 5)):
            with self.subTest(keep=keep):
                audit = json.loads((run / f"k{keep}" / "masked" / "records.jsonl.summary.json")
                                   .read_text(encoding="utf-8"))["mask_audit"]
                prefill = sum(phases["prefill"]["changed"] for phases in audit.values())
                decode = sum(phases["decode"]["changed"] for phases in audit.values())
                seen = sum(phases["decode"]["decisions"] for phases in audit.values())
                self.assertGreaterEqual(prefill, least_prefill)
                self.assertGreaterEqual(decode, least_decode)
                self.assertLessEqual(decode, seen)
                self.assertGreater(seen, 0)
                result = json.loads((run / f"k{keep}" / "equiv.json").read_text(encoding="utf-8"))
                self.assertTrue(result["pass"], "the physical prune must still equal the masked original")
                self.assertEqual(result["bitwise_identical_logits"], result["sentences"])
                self.assertEqual(result["logits_max_abs_diff"], 0.0)
