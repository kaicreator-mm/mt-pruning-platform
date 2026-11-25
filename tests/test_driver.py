"""The driver script on a dry-run world (see tests/util.py): it stops before the test
split, the final test needs the review, and a changed corpus blocks everything."""
import json
import shutil
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
                      "provisional decision from validation records only", "final test: HOLD"):
            self.assertIn(stage, result.stdout)
        self.assertNotIn("final test for the reviewed candidate", result.stdout)
        for path in run.rglob("records.jsonl"):
            self.assertNotIn("test", lines(path)[0]["splits"], str(path))
        self.assertFalse((run / "final_test").exists())
        self.assertFalse((run / "test_exposure.json").exists())
        decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        self.assertIs(decision["test_execution_authorized"], False)
        self.assertEqual(sorted(decision["runs"]), ["baseline", "k64", "k96"])
        self.assertEqual(decision["identity"]["teacher"],
                         json.loads((run / "baseline" / "records.jsonl.identity.json").read_text(encoding="utf-8"))["teacher"])
        self.assertEqual(sorted(p.name for p in (self.world["root"] / "pruned").iterdir()), [])   # rebuilt from the selection
        evidence = self.world["root"] / "evidence"
        for name in ("corpus_receipt.json", "decision.json", "report.md", "selection.json",
                     "k128/equiv.json", "k96/equiv.json", "k64/equiv.json", "k96/prune.json", "k64/prune.json",
                     "k96/masked/records.jsonl.summary.json", "inputs/checkpoint_content.json", "inputs/host.txt"):
            self.assertTrue((evidence / name).exists(), name)
        self.assertEqual(list(evidence.rglob("*.sqlite")), [])       # journals stay in the run directory
        self.assertTrue(json.loads((evidence / "k128" / "equiv.json").read_text(encoding="utf-8"))["pass"])

    def test_03_final_test_refuses_to_run_without_the_review(self):
        decision = util.sha256(self.world["run"] / "decision.json")
        missing = util.driver(self.world, {"FINAL_TEST": "1", "REVIEWED_DECISION_SHA256": decision})
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("needs REVIEWED_DECISION_SHA256 and REVIEW_REF", missing.stdout + missing.stderr)
        self.assertNotIn("baseline: calibration", missing.stdout)       # refused before any stage ran
        self.assertFalse((self.world["run"] / "final_test").exists())

    def test_04_final_test_runs_once_with_the_review(self):
        run, decision_path = self.world["run"], self.world["run"] / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision["chosen_keep"] is None:
            self.skipTest("the random checkpoint produced FALLBACK_TO_TEACHER on validation")
        reviewed = util.sha256(decision_path)
        environment = {"FINAL_TEST": "1", "REVIEWED_DECISION_SHA256": reviewed, "REVIEW_REF": "driver test review"}
        first = util.driver(self.world, environment)
        self.assertEqual(first.returncode, 0, self.output(first))
        self.assertIn("final test for the reviewed candidate", first.stdout)
        ledger = json.loads((run / "test_exposure.json").read_text(encoding="utf-8"))
        self.assertEqual([(e["keep"], e["status"], e["review_ref"]) for e in ledger["entries"]],
                         [(decision["chosen_keep"], "completed", "driver test review")])
        tested = sorted(p.relative_to(run).as_posix() for p in run.rglob("records.jsonl")
                        if "test" in lines(p)[0]["splits"])
        self.assertEqual(tested, ["final_test/baseline/records.jsonl", f"final_test/k{decision['chosen_keep']}/records.jsonl"])
        self.assertIn(f"{decision['chosen_keep']} experts, test: **REPORT_GATE_", (run / "report.md").read_text(encoding="utf-8"))
        before = {p.relative_to(run).as_posix(): util.sha256(p) for p in (run / "final_test").rglob("records.jsonl")}
        again = util.driver(self.world, environment)                      # a rerun resumes and changes nothing
        self.assertEqual(again.returncode, 0, self.output(again))
        self.assertEqual(len(json.loads((run / "test_exposure.json").read_text(encoding="utf-8"))["entries"]), 1)
        self.assertEqual(before, {p.relative_to(run).as_posix(): util.sha256(p)
                                  for p in (run / "final_test").rglob("records.jsonl")})

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
