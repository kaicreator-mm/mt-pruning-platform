"""The report: a quality statement only through the reviewed gate, and only on complete coverage."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mtpruning import report
from tests import util
from tests.util import READY

DIRECTIONS = ("zh-my", "my-zh", "zh-en", "en-zh", "en-my", "my-en")
EVERY = {direction: 0.0 for direction in DIRECTIONS}


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def table(deltas, candidate_flags=None, baseline_flags=None, n=3):
    """Report rows in the layout bridge.quality_gate reads, with the row flags the report would compute."""
    candidate_flags, baseline_flags = candidate_flags or {}, baseline_flags or {}

    def candidate(direction, delta):
        mine, theirs = candidate_flags.get(direction, {}), baseline_flags.get(direction, {})
        counts = all(type(value) is int for flag in (mine, theirs) for value in flag.values())
        return {"chrf_pp": 50.0 + delta, "delta": delta, "flags": mine,
                "gate_pass": report.passes_gate(delta, 1.0),
                "diagnostics_not_worse": report.no_new_failures(mine, theirs) if counts else False}
    return [{"split": "validation", "direction": d, "n": n, "baseline_chrf_pp": 50.0,
             "baseline_flags": baseline_flags.get(d, {}), "candidates": {"96": candidate(d, delta)}}
            for d, delta in deltas.items()]


def gate_of(rows, max_drop=1.0):
    return report.gate(rows, "validation", "96", 3, max_drop)


class GateSignTests(unittest.TestCase):
    """P1-01: the gate is "drops by no more than max_drop", not "differs from the baseline by at most +1"."""

    def test_boundary_drops(self):
        deltas = {**EVERY, "zh-my": -1.01, "my-zh": -1.0, "zh-en": 2.0, "en-zh": 0.0, "en-my": -0.5, "my-en": -7.5}
        verdict = gate_of(table(deltas))
        self.assertEqual({d: v["quality_pass"] for d, v in verdict["directions"].items()},
                         {"zh-my": False, "my-zh": True, "zh-en": True, "en-zh": True, "en-my": True, "my-en": False})
        self.assertEqual(verdict["status"], "REPORT_GATE_FAIL")
        self.assertEqual(sorted(report.outcome(verdict)["failing_score"]), ["my-en", "zh-my"])
        for drop, expected in ((-1.0, "REPORT_GATE_PASS"), (-1.01, "REPORT_GATE_FAIL"), (0.0, "REPORT_GATE_PASS"),
                               (2.0, "REPORT_GATE_PASS"), (-5.0, "REPORT_GATE_FAIL")):
            with self.subTest(drop=drop):
                self.assertEqual(gate_of(table({**EVERY, "zh-my": drop}))["status"], expected)
        self.assertEqual(gate_of(table({**EVERY, "zh-my": -1.01}), max_drop=2.0)["status"], "REPORT_GATE_PASS")
        self.assertEqual(gate_of(table({**EVERY, "zh-my": -1.0}), max_drop=0.5)["status"], "REPORT_GATE_FAIL")

    def test_own_flag_matches_the_reviewed_gate(self):
        for delta, expected in ((-1.01, False), (-1.0, True), (-0.99, True), (0.0, True), (2.0, True), (-100.0, False)):
            self.assertIs(report.passes_gate(delta, 1.0), expected, delta)
        for delta in (float("nan"), float("inf"), float("-inf"), "-1", None, True):
            self.assertFalse(report.passes_gate(delta, 1.0), delta)
        for max_drop in (float("nan"), float("inf"), -1.0, "1", None, True):
            self.assertFalse(report.passes_gate(-0.5, max_drop), max_drop)
        self.assertTrue(report.passes_gate(0.0, 0.0))
        self.assertFalse(report.passes_gate(-1e-9, 0.0))
        with self.assertRaises(SystemExit):
            report.build(Path("nowhere"), max_drop=float("nan"))     # checked before anything is read

    def test_diagnostics_gate_too(self):
        verdict = gate_of(table({**EVERY, "zh-my": 0.5}, candidate_flags={"zh-my": {"empty": 1}}))
        self.assertEqual(verdict["status"], "REPORT_GATE_FAIL")       # a good score does not excuse new empty outputs
        self.assertTrue(verdict["directions"]["zh-my"]["quality_pass"])
        self.assertEqual(report.outcome(verdict)["failing"], ["zh-my"])
        self.assertEqual(report.outcome(verdict)["failing_diagnostics"], ["zh-my"])
        same = {"truncated": 1}
        tolerated = gate_of(table(EVERY, candidate_flags={"zh-my": same}, baseline_flags={"zh-my": same}))
        self.assertEqual(tolerated["status"], "REPORT_GATE_PASS")     # no increase
        worse = gate_of(table(EVERY, candidate_flags={"my-en": {"digits_differ_from_reference": 2}},
                              baseline_flags={"my-en": {"digits_differ_from_reference": 1}}))
        self.assertEqual(worse["status"], "REPORT_GATE_FAIL")
        self.assertEqual(report.outcome(worse)["failing_diagnostics"], ["my-en"])
        self.assertEqual(report.outcome(worse)["failing_score"], [])

    def test_incomplete_or_inconsistent_rows_are_never_a_pass(self):
        complete = table(EVERY)
        cases = {
            "five directions": complete[:5],
            "no rows": [],
            "duplicated direction": complete + [complete[0]],
            "small group": table(EVERY, n=2),
            "candidate missing": [{**row, "candidates": {}} for row in complete],
            "score not finite": table({**EVERY, "zh-my": float("nan")}),
            "score out of range": table({**EVERY, "zh-my": 60.0}),
            "flag counts not integers": table(EVERY, candidate_flags={"zh-my": {"empty": "one"}}),
            "unknown flag": table(EVERY, candidate_flags={"zh-my": {"made_up": 1}}),
        }
        for what, rows in cases.items():
            with self.subTest(what):
                verdict = gate_of(rows)
                self.assertEqual(verdict["status"], "NOT_EVALUATED")
                self.assertTrue(verdict["reason"])
                self.assertEqual(report.outcome(verdict)["failing"], [])

    def test_row_flags_are_cross_checked_against_the_reviewed_gate(self):
        verdict = gate_of(table({**EVERY, "zh-my": -3.0}))
        rows = table({**EVERY, "zh-my": -3.0})
        report.agree_with_reviewed_gate(rows, verdict, "96")
        rows[0]["candidates"]["96"]["gate_pass"] = True
        with self.assertRaisesRegex(RuntimeError, "disagree with the reviewed gate"):
            report.agree_with_reviewed_gate(rows, verdict, "96")
        rows[0]["candidates"]["96"]["gate_pass"] = False
        rows[0]["candidates"]["96"]["diagnostics_not_worse"] = False
        with self.assertRaisesRegex(RuntimeError, "disagree with the reviewed gate"):
            report.agree_with_reviewed_gate(rows, verdict, "96")
        # nothing to disagree with when the rows do not cover the split
        report.agree_with_reviewed_gate([], report.gate([], "validation", "96", 3, 1.0), "96")


@unittest.skipUnless(READY, util.__doc__)
class ReportCoverageTests(unittest.TestCase):
    """P1-03/P1-04: strict coverage and bound receipts on a rehearsed run directory."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.world = util.make_world(Path(cls.tmp.name))
        util.rehearse(cls.world)          # with the reference-runtime receipts the chain requires

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, change=None, keep_decision=True):
        run = Path(tempfile.mkdtemp(dir=self.world["root"])) / "run"
        shutil.copytree(self.world["run"], run)
        if change:
            change(run)
        if not keep_decision:
            # The damaged-record scenarios below exercise the coverage gate, not the decision
            # binding: a decision bound to the intact bytes would rightly refuse to build.
            (run / "decision.json").unlink()
        return run

    def build(self, change=None, keep_decision=True):
        run = self.copy(change, keep_decision)
        return run, report.build(run, self.world["manifest_sha256"])

    def quality(self, built, candidate, split):
        return built["verdict"]["quality"][candidate][split]

    def test_complete_rehearsal_holds_the_test_split(self):
        _, built = self.build()
        verdict = built["verdict"]
        self.assertTrue(verdict["mechanism_pass"])
        self.assertTrue(verdict["noop_pass"])
        self.assertEqual(verdict["masked_pass"], {"k96": True, "k64": True})
        for name in ("96", "64"):
            self.assertIn(self.quality(built, name, "validation")["status"], ("REPORT_GATE_PASS", "REPORT_GATE_FAIL"))
            self.assertIsNone(self.quality(built, name, "validation")["problem"])
            held = self.quality(built, name, "test")
            self.assertEqual((held["status"], held["reason"]), ("NOT_RUN", report.HELD))
        self.assertIn("HOLD", verdict["quality_result"])
        self.assertIs(verdict["production_acceptance"], False)
        self.assertFalse(built["exploratory"])
        self.assertEqual(built["decision"]["status"], "PROVISIONAL_VALIDATION_SELECTION_ONLY")
        self.assertIs(built["decision"]["test_execution_authorized"], False)
        self.assertIsNone(built["corpus_receipt"])        # the in-process rehearsal writes no receipt
        self.assertTrue(built["synthetic"])
        self.assertEqual(built["chrf_pp_test"], [])
        self.assertEqual([(row["direction"], row["n"]) for row in built["chrf_pp"]], [(d, 3) for d in DIRECTIONS])
        for path in self.world["run"].rglob("records.jsonl"):
            self.assertNotIn("test", lines(path)[0]["splits"], str(path))
        for name in ("96", "64"):       # the reviewed validation-only selection agrees with the validation gate
            self.assertIs(built["decision"]["candidates"][name]["pass"],
                          self.quality(built, name, "validation")["status"] == "REPORT_GATE_PASS")

    def test_missing_direction_is_not_evaluated_never_pass(self):
        def drop(run):
            path = run / "k96" / "records.jsonl"
            write(path, [row for row in lines(path) if row.get("direction") != "zh-my"])
        _, built = self.build(drop, keep_decision=False)
        self.assertEqual(self.quality(built, "96", "validation")["status"], "NOT_EVALUATED")
        self.assertIn("incomplete validation records", self.quality(built, "96", "validation")["problem"])
        self.assertIn(self.quality(built, "64", "validation")["status"], ("REPORT_GATE_PASS", "REPORT_GATE_FAIL"))
        self.assertNotIn("96", built["chrf_pp"][0]["candidates"])     # no partial row the gate could read
        self.assertNotIn("96 experts: validation REPORT_GATE", built["verdict"]["summary"])

    def test_duplicate_extra_and_misplaced_records_are_refused(self):
        def duplicate(run):
            path = run / "k64" / "records.jsonl"
            write(path, lines(path) + [lines(path)[1]])
        def extra(run):
            path = run / "k64" / "records.jsonl"
            write(path, lines(path) + [{**lines(path)[1], "id": "validation:zh-my:9999"}])
        def misplaced(run):
            path = run / "k64" / "records.jsonl"
            rows = lines(path)
            rows[1]["split"] = "test"
            write(path, rows)
        def reshaped(run):
            path = run / "k64" / "records.jsonl"
            rows = lines(path)
            rows[1]["tokens"] = rows[1]["tokens"][:-1]
            write(path, rows)
        def header_only(run):
            path = run / "k64" / "records.jsonl"
            write(path, lines(path)[:1])
        for change, reason in ((duplicate, "invalid/duplicate run record"), (extra, "invalid/duplicate run record"),
                               (misplaced, "run sample mismatch"), (reshaped, "bad generation record"),
                               (header_only, "incomplete validation records")):
            with self.subTest(reason=reason, change=change.__name__):
                _, built = self.build(change, keep_decision=False)
                self.assertEqual(self.quality(built, "64", "validation")["status"], "NOT_EVALUATED")
                self.assertIn(reason, self.quality(built, "64", "validation")["problem"])

    def test_candidate_that_saw_the_test_split_makes_any_test_exploratory(self):
        def expose(run):
            path = run / "k64" / "records.jsonl"
            rows = lines(path)
            rows[0]["splits"] = ["validation", "test"]
            write(path, rows)
        _, built = self.build(expose, keep_decision=False)
        self.assertEqual(self.quality(built, "64", "validation")["status"], "NOT_EVALUATED")
        self.assertIn("candidate test already exposed", self.quality(built, "64", "validation")["problem"])
        self.assertTrue(built["exploratory"])
        self.assertEqual(len(built["test_used_outside_final_test"]), 1)

    def test_incomplete_baseline_leaves_no_candidate_evaluated(self):
        def drop(run):
            path = run / "baseline" / "records.jsonl"
            rows = lines(path)
            missing = next(row["id"] for row in rows[1:] if row["split"] == "validation")
            write(path, [row for row in rows if row.get("id") != missing])
        _, built = self.build(drop, keep_decision=False)
        self.assertEqual(built["chrf_pp"], [])
        for name in ("96", "64"):
            self.assertEqual(self.quality(built, name, "validation")["status"], "NOT_EVALUATED")
            self.assertIn("baseline", self.quality(built, name, "validation")["problem"])
        self.assertIn("validation NOT_EVALUATED", built["verdict"]["summary"])

    def test_changed_corpus_bytes_block_the_report(self):
        def tamper(run):
            path = run / "corpus" / "corpus.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as stopped:
            self.build(tamper)
        self.assertIn("BLOCKED_CORPUS_CONTRACT", str(stopped.exception))
        run = self.copy(tamper)
        self.assertFalse((run / "report.json").exists())

    def test_the_decision_is_bound_to_the_records_it_was_made_from(self):
        run = self.copy()
        decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(decision["runs"]), ["baseline", "k64", "k96"])
        built = report.build(run, self.world["manifest_sha256"])          # the binding holds as rehearsed
        self.assertIs(built["decision"]["test_execution_authorized"], False)
        self.assertEqual(built["decision"]["final_test"]["status"], "HOLD")
        victim = 64 if decision["chosen_keep"] != 64 else 96

        def altered(run):    # the records the decision bound were edited afterwards
            path = run / f"k{victim}" / "records.jsonl"
            rows = lines(path)
            rows[1]["text"] = rows[1]["text"] + " later"
            write(path, rows)
        with self.assertRaisesRegex(SystemExit, "no longer match the run directory"):
            self.build(altered)
        run = self.copy()
        forged = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        forged["identity"]["teacher"] = "0" * 64                          # a decision from another run
        (run / "decision.json").write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "made from another baseline run"):
            report.build(run, self.world["manifest_sha256"])

    def test_the_mechanism_needs_every_planned_receipt(self):
        def drop(run):       # one candidate's reference-runtime receipt is missing
            (run / "k64" / "stock_check.json").unlink()
        run, built = self.build(drop)
        self.assertFalse(built["verdict"]["mechanism_pass"])
        self.assertEqual(built["verdict"]["masked_pass"]["k64"], False)
        self.assertIn("64: FAIL", built["verdict"]["summary"])
        def gone(run):       # a whole candidate folder is missing
            shutil.rmtree(run / "k64")
        run, built = self.build(gone, keep_decision=False)
        self.assertFalse(built["verdict"]["mechanism_pass"])
        self.assertIn("missing evidence for k64", built["verdict"]["summary"])
