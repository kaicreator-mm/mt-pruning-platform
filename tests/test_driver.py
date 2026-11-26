"""The driver script on a dry-run world (see tests/util.py): it stops before the test
split, the final test needs the review, and a changed corpus blocks everything."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests import util
from tests.util import READY


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@unittest.skipUnless(READY, util.__doc__)
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.world = util.make_world(Path(cls.tmp.name))
        cls.result = util.driver(cls.world)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def output(self, result):
        return (result.stdout + result.stderr)[-6000:]

    def test_01_stops_before_the_test_split(self):
        result, run = self.result, self.world["run"]
        self.assertEqual(result.returncode, 0, self.output(result))
        for stage in ("frozen corpus contract", "checkpoint content", "selection from the calibration routing",
                      "identity reload of the unpruned checkpoint", "k96: masked original", "k64: validation outputs",
                      "provisional decision from validation records only", "final test: DISABLED"):
            self.assertIn(stage, result.stdout)
        self.assertNotIn("final test for the reviewed candidate", result.stdout)
        for path in run.rglob("records.jsonl"):
            self.assertNotIn("test", lines(path)[0]["splits"], str(path))
        self.assertFalse((run / "final_test").exists())
        self.assertFalse((run / "test_exposure.json").exists())
        decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        self.assertIs(decision["test_execution_authorized"], False)
        self.assertEqual(decision["final_test"]["status"], "HOLD")
        self.assertEqual(sorted(decision["runs"]), ["baseline", "k64", "k96"])
        self.assertEqual(decision["identity"]["teacher"],
                         json.loads((run / "baseline" / "records.jsonl.identity.json").read_text(encoding="utf-8"))["teacher"])
        self.assertEqual(sorted(p.name for p in (self.world["root"] / "pruned").iterdir()), [])   # rebuilt from the selection
        evidence = self.world["root"] / "evidence"
        for name in ("corpus_receipt.json", "decision.json", "report.md", "selection.json",
                     "k128/equiv.json", "k96/equiv.json", "k64/equiv.json", "k96/prune.json", "k64/prune.json",
                     "k96/masked/records.jsonl.summary.json", "inputs/checkpoint_content.json", "inputs/host.txt",
                     "inputs/disk_preflight.json", "inputs/retained_run_location.txt",
                     "inputs/retained_manifest.check.txt", "retained_manifest.sha256"):
            self.assertTrue((evidence / name).exists(), name)
        preflight = json.loads((evidence / "inputs" / "disk_preflight.json").read_text(encoding="utf-8"))
        self.assertTrue(preflight["sufficient"])
        self.assertTrue(preflight["assumptions"]["witness_count_never_reduced"])
        self.assertEqual(preflight["assumptions"]["largest_copy_experts"], 128)
        self.assertTrue(all(item["sufficient"] for item in preflight["filesystems"]))
        self.assertEqual({path["role"] for item in preflight["filesystems"] for path in item["paths"]},
                         {"RUN", "PRUNED", "EVIDENCE_DIR"})
        for stage in ("baseline", "k128", "k96", "k64", "evidence"):
            self.assertTrue((evidence / "inputs" / f"disk_preflight_{stage}.json").exists(), stage)
        self.assertEqual((evidence / "inputs" / "retained_run_location.txt").read_text().strip(), str(run.resolve()))
        # the retained manifest covers the run directory's evidence, witness traces included
        retained = (evidence / "retained_manifest.sha256").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any("baseline/traces/" in line and line.endswith(".trace.npz") for line in retained), retained)
        self.assertFalse(any(line.endswith("retained_manifest.sha256") for line in retained))
        checked = subprocess.run(["sha256sum", "-c", "retained_manifest.sha256"], cwd=run,
                                 capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
        self.assertNotIn("FAILED", (evidence / "inputs" / "retained_manifest.check.txt").read_text())
        # The journals and their identity bindings are evidence too: an independent reviewer
        # can reconstruct every aggregate and every trace digest from them (R11-P1-04).
        journals = sorted(p.relative_to(evidence).as_posix() for p in evidence.rglob("*.sqlite"))
        self.assertIn("baseline/records.jsonl.sqlite", journals)
        self.assertTrue((evidence / "baseline" / "records.jsonl.identity.json").exists())
        self.assertTrue(json.loads((evidence / "k128" / "equiv.json").read_text(encoding="utf-8"))["pass"])

    def test_03_final_test_is_disabled_before_any_stage_runs(self):
        decision = util.sha256(self.world["run"] / "decision.json")
        # Even with a decision hash and a review reference, nothing runs: this build has no
        # final-test execution path, and no review-reference string re-enables one (R11-P1-05).
        refused = util.driver(self.world, {"FINAL_TEST": "1", "REVIEWED_DECISION_SHA256": decision,
                                           "REVIEW_REF": "driver test review"})
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("FINAL_TEST is disabled", refused.stdout + refused.stderr)
        self.assertNotIn("baseline: calibration", refused.stdout)       # refused before any stage ran
        self.assertFalse((self.world["run"] / "final_test").exists())

    def test_04_final_test_execution_paths_are_removed(self):
        from mtpruning import pipeline
        with self.assertRaises(SystemExit) as stopped:                  # the CLI command is gone
            pipeline.main(["final-test", "--run-dir", str(self.world["run"]), "--decision-sha256", "0" * 64,
                           "--review-ref", "unit test review"])
        self.assertEqual(stopped.exception.code, 2)                     # argparse: no such command
        with self.assertRaisesRegex(SystemExit, "FINAL_TEST_DISABLED"):
            pipeline.execute_run(util.options(self.world, splits=["test"]))
        with self.assertRaisesRegex(SystemExit, "FINAL_TEST_DISABLED"):
            util.cli(self.world, "stock-check", "--model", self.world["model"], "--corpus", self.world["corpus"],
                     "--records", self.world["run"] / "k64" / "records.jsonl",
                     "--out", self.world["run"] / "never.json", "--splits", "validation,test",
                     "--limit", 1, "--max-new-tokens", 6, "--threads", 2)
        self.assertFalse((self.world["run"] / "never.json").exists())

    def test_02_changed_corpus_blocks_the_driver_before_any_run(self):
        world = dict(self.world)
        world["root"] = self.world["root"] / "blocked"
        world["run"] = world["root"] / "run"
        world["cache"] = world["root"] / "cache"
        shutil.copytree(self.world["run"] / "corpus", world["run"] / "corpus")   # a fresh run directory
        corpus = world["run"] / "corpus" / "corpus.jsonl"
        tampered = corpus.read_bytes() + b"\n"                           # same sentences, other bytes
        corpus.write_bytes(tampered)
        result = util.driver(world)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BLOCKED_CORPUS_CONTRACT", result.stdout + result.stderr)
        self.assertNotIn("baseline: calibration", result.stdout)
        self.assertEqual([p.name for p in (world["run"]).iterdir()], ["corpus"])  # no run, no selection ...
        self.assertEqual(corpus.read_bytes(), tampered)                   # ... and nothing silently resampled
