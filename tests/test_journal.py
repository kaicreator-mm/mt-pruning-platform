"""Per-sample persistence: an interrupted run resumes to exactly the result of an
uninterrupted one, a finished run is never overwritten, and a journal cannot be
continued under another identity or with a sample that does not match."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import util
from tests.util import READY


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


MASK = {"1": [0, 1, 2, 3, 4, 5, 6, 7], "2": [0, 1, 2, 3, 4, 5, 6, 7]}


@unittest.skipUnless(READY, util.__doc__)
class JournalResumeTests(unittest.TestCase):
    def setUp(self):
        from mtpruning import pipeline
        self.tmp = tempfile.TemporaryDirectory()
        self.world = util.make_world(Path(self.tmp.name))
        self.pipeline = pipeline

    def tearDown(self):
        self.pipeline.CRASH_AFTER = 0
        self.tmp.cleanup()

    def arrays(self, base):
        with np.load(str(base) + ".npz", allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}

    def assertSameAggregate(self, first, second):
        a, b = self.arrays(first), self.arrays(second)
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertTrue(np.array_equal(a[name], b[name]), name)

    def assertSameOutputs(self, first, second):
        from mtpruning.pipeline import read_run
        self.assertEqual(read_run(first), read_run(second))

    def test_interrupted_then_resumed_equals_uninterrupted(self):
        p = self.pipeline
        p.execute_run(util.options(self.world))                    # baseline: calibration + validation, one go
        whole = self.world["run"] / "baseline"
        other = self.world["run"] / "again"
        resumed = util.options(self.world, out=other / "records.jsonl", stats=other / "stats",
                               trace_dir=other / "traces")
        seen = []
        for crash_at in (3, 7):                                     # interrupted twice, at different points
            p.CRASH_AFTER = crash_at
            with self.assertRaises(p.SimulatedCrash):
                p.execute_run(resumed)
            self.assertFalse((other / "stats.npz").exists())       # an interrupted run exports no aggregate
            seen.append(util.sha256(Path(str(other / "records.jsonl") + ".sqlite")))
        self.assertNotEqual(seen[0], seen[1])
        p.CRASH_AFTER = 0
        summary = p.execute_run(resumed)
        self.assertSameOutputs(whole / "records.jsonl", other / "records.jsonl")
        self.assertSameAggregate(whole / "stats", other / "stats")
        self.assertEqual(read_json(whole / "records.jsonl.summary.json")["mask_audit"],
                         read_json(other / "records.jsonl.summary.json")["mask_audit"])
        self.assertEqual(summary["sentences"], 6 * (util.SIZES["n_calibration"] + util.SIZES["n_validation"]))

    def test_rerunning_a_finished_run_changes_nothing(self):
        p, options = self.pipeline, util.options(self.world)
        p.execute_run(options)
        whole = self.world["run"] / "baseline"
        before = util.sha256(whole / "records.jsonl")
        arrays, audit = self.arrays(whole / "stats"), read_json(whole / "records.jsonl.summary.json")["routing"]
        summary = p.execute_run(options)
        self.assertEqual(summary["commits_this_invocation"], 0)
        self.assertEqual(util.sha256(whole / "records.jsonl"), before)
        after = self.arrays(whole / "stats")                           # never overwritten with an empty aggregate
        self.assertEqual(list(after), list(arrays))
        self.assertTrue(all(np.array_equal(after[name], arrays[name]) for name in arrays))
        self.assertEqual(read_json(whole / "records.jsonl.summary.json")["routing"], audit)

    def test_conflicting_duplicate_and_corrupted_payload_are_refused(self):
        from mtpruning.reviewed import trace
        p = self.pipeline
        p.execute_run(util.options(self.world))
        records = self.world["run"] / "baseline" / "records.jsonl"
        journal_file = Path(str(records) + ".sqlite")
        binding = read_json(Path(str(records) + ".identity.json"))
        journal = trace.Journal(journal_file, binding)             # no context manager: close by hand
        sample_id, payload = next(iter(journal.records()))
        journal.close()
        self.assertGreater(payload["routing"]["prefill"]["tokens"], 0)
        self.assertTrue(all(value > 0 for value in payload["routing"]["prefill"]["mass"]))
        journal = trace.Journal(journal_file, binding)
        journal.put(sample_id, payload)                              # the same sample again is idempotent
        with self.assertRaisesRegex(ValueError, "conflicting sample"):
            journal.put(sample_id, {**payload, "text": payload["text"] + " "})
        journal.close()
        database = sqlite3.connect(journal_file)
        database.execute("UPDATE samples SET payload = replace(payload, '\"routing\"', '\"detouring\"') WHERE id = ?",
                         (sample_id,))
        database.commit()
        database.close()
        with self.assertRaisesRegex(ValueError, "corrupt sample"):
            p.execute_run(util.options(self.world))

    def test_a_sample_committed_under_another_identity_is_refused(self):
        from mtpruning.reviewed import trace
        p = self.pipeline
        p.execute_run(util.options(self.world))
        records = self.world["run"] / "baseline" / "records.jsonl"
        journal_file = Path(str(records) + ".sqlite")
        binding = read_json(Path(str(records) + ".identity.json"))
        journal = trace.Journal(journal_file, binding)
        sample_id, payload = next(iter(journal.records()))
        journal.close()
        forged = {**payload, "identity_key": "0" * 64}
        journal = trace.Journal(journal_file, binding)
        with self.assertRaisesRegex(ValueError, "conflicting sample"):
            journal.put(sample_id, forged)                               # the journal itself refuses
        journal.close()
        database = sqlite3.connect(journal_file)                         # written behind the journal's back
        database.execute("UPDATE samples SET payload = ?, sha = ? WHERE id = ?",
                         (trace.canon(forged), trace.digest(forged), sample_id))
        database.commit()
        database.close()
        with self.assertRaisesRegex(ValueError, "committed under another run identity"):
            p.execute_run(util.options(self.world))

    def test_a_journal_refuses_to_continue_under_another_identity(self):
        p = self.pipeline
        p.execute_run(util.options(self.world))
        records = self.world["run"] / "baseline" / "records.jsonl"
        before = util.sha256(records)
        selection = self.world["run"] / "selection.json"
        selection.write_text(json.dumps({"candidates": {"96": {"kept": MASK}}}))
        changes = [("decoding settings", dict(max_new_tokens=5)), ("sample limit", dict(limit=1)),
                   ("traced sentences", dict(trace_n=1)), ("statistics collection", dict(stats=None)),
                   ("split scope", dict(splits=["calibration", "validation", "test"], test_authorized=True)),
                   ("mask", dict(mask=str(selection), mask_keep=96))]
        for what, change in changes:
            with self.subTest(what), self.assertRaisesRegex(ValueError, "identity mismatch"):
                p.execute_run(util.options(self.world, **change))
        altered = {"files": {"mtpruning/elsewhere.py": "0" * 64}}
        with mock.patch("mtpruning.identity.code_identity", return_value=altered):
            with self.assertRaisesRegex(ValueError, "identity mismatch.*code"):
                p.execute_run(util.options(self.world))
        with mock.patch("mtpruning.identity.runtime_identity", return_value={"elsewhere": True}):
            with self.assertRaisesRegex(ValueError, "identity mismatch.*runtime"):
                p.execute_run(util.options(self.world))
        self.assertEqual(util.sha256(records), before)       # refused runs leave the evidence alone

    def test_changed_corpus_bytes_block_the_run(self):
        p, corpus = self.pipeline, self.world["corpus"]
        p.execute_run(util.options(self.world))
        original = corpus.read_bytes()
        at = original.index(b'"source": "') + len(b'"source": "')
        corpus.write_bytes(original[:at] + bytes([original[at] ^ 1]) + original[at + 1:])
        with self.assertRaises(SystemExit) as stopped:
            p.execute_run(util.options(self.world))
        self.assertIn("BLOCKED_CORPUS_CONTRACT", str(stopped.exception))
        corpus.write_bytes(original)
        with self.assertRaises(SystemExit) as stopped:            # the pinned manifest hash governs too
            p.execute_run(util.options(self.world, manifest_sha256="0" * 64))
        self.assertIn("BLOCKED_CORPUS_CONTRACT", str(stopped.exception))
        p.execute_run(util.options(self.world))                     # restored: resumes normally

    def test_selection_refuses_an_interrupted_calibration(self):
        p, run = self.pipeline, self.world["run"]
        p.CRASH_AFTER = 5
        with self.assertRaises(p.SimulatedCrash):
            p.execute_run(util.options(self.world))
        p.CRASH_AFTER = 0
        self.assertFalse((run / "baseline" / "stats.npz").exists())
        with self.assertRaises(SystemExit) as stopped:                 # nothing was exported: nothing to select from
            util.cli(self.world, "select", "--run-dir", run, "--keep", "96,64", "--out", run / "selection.json")
        self.assertIn("baseline", str(stopped.exception))
        self.assertFalse((run / "selection.json").exists())
        p.execute_run(util.options(self.world))                         # finished: the aggregate exists
        (run / "baseline" / "stats.npz").unlink()                       # but is lost or not yet written
        with self.assertRaises(SystemExit) as stopped:
            util.cli(self.world, "select", "--run-dir", run, "--keep", "96,64", "--out", run / "selection.json")
        self.assertIn("has not exported", str(stopped.exception))
        self.assertFalse((run / "selection.json").exists())

    def test_selection_refuses_stale_or_edited_aggregates(self):
        p, run = self.pipeline, self.world["run"]
        p.execute_run(util.options(self.world))
        base = run / "baseline" / "stats"
        reference = util.sha256(run / "baseline" / "records.jsonl")
        saved = {name: value for name, value in self.arrays(base).items()}
        np.savez_compressed(str(base) + ".npz", **{**saved, "mass_prefill": saved["mass_prefill"] * 0})
        with self.assertRaises(SystemExit) as stopped:
            util.cli(self.world, "select", "--run-dir", run, "--keep", "96,64", "--out", run / "selection.json")
        self.assertIn("exported statistics differ", str(stopped.exception))
        np.savez_compressed(str(base) + ".npz", **saved)
        exported = read_json_lines(run / "baseline" / "records.jsonl")
        victim = min(row["id"] for row in exported[1:] if row["split"] == "calibration")
        database = sqlite3.connect(str(run / "baseline" / "records.jsonl") + ".sqlite")
        database.execute("DELETE FROM samples WHERE id = ?", (victim,))
        database.commit()
        database.close()
        with self.assertRaises(SystemExit) as stopped:
            util.cli(self.world, "select", "--run-dir", run, "--keep", "96,64", "--out", run / "selection.json")
        self.assertRegex(str(stopped.exception), "exported (records|statistics) differ")   # never reused
        self.assertFalse((run / "selection.json").exists())
        summary = p.execute_run(util.options(self.world))      # a rerun recomputes the missing sample
        self.assertEqual(summary["commits_this_invocation"], 1)
        self.assertEqual(util.sha256(run / "baseline" / "records.jsonl"), reference)
        util.cli(self.world, "select", "--run-dir", run, "--keep", "96,64", "--out", run / "selection.json")
        selection = read_json(run / "selection.json")
        self.assertEqual(selection["status"], "SELECTION_FROM_COVERAGE_CHECKED_CALIBRATION")
        self.assertEqual(selection["aggregate_status"], "AGGREGATE_COVERAGE_CHECKED_NOT_PROVENANCE")
        self.assertEqual(sorted(selection["candidates"]), ["64", "96"])

    def test_mask_audit_of_an_interrupted_masked_run_matches_an_uninterrupted_one(self):
        p, run = self.pipeline, self.world["run"]
        p.execute_run(util.options(self.world))
        util.cli(self.world, "select", "--run-dir", run, "--keep", "96,64", "--out", run / "selection.json")
        masked = dict(splits=["validation"], limit=2, stats=None, trace_n=2, mask=str(run / "selection.json"),
                      mask_keep=64)
        whole = run / "k96" / "masked"
        p.execute_run(util.options(self.world, out=whole / "records.jsonl", trace_dir=whole / "traces", **masked))
        other = run / "elsewhere"
        resumed = util.options(self.world, out=other / "records.jsonl", trace_dir=other / "traces", **masked)
        p.CRASH_AFTER = 4
        with self.assertRaises(p.SimulatedCrash):
            p.execute_run(resumed)
        p.CRASH_AFTER = 0
        p.execute_run(resumed)
        self.assertSameOutputs(whole / "records.jsonl", other / "records.jsonl")
        audit = read_json(whole / "records.jsonl.summary.json")["mask_audit"]
        self.assertEqual(audit, read_json(other / "records.jsonl.summary.json")["mask_audit"])
        changed = sum(phases[phase]["changed"] for phases in audit.values() for phase in ("prefill", "decode"))
        self.assertGreater(changed, 0)                # the mask really removed experts that held-out tokens use
        for phases in audit.values():
            self.assertGreater(phases["prefill"]["decisions"], 0)
            for phase in ("prefill", "decode"):
                self.assertLessEqual(phases[phase]["changed"], phases[phase]["decisions"])


@unittest.skipUnless(READY, util.__doc__)
class JournalTests(unittest.TestCase):
    """The reviewed journal itself, as used here: exactly once per sample, conflict and corruption detected."""

    def test_exactly_once_with_conflict_and_corruption_detection(self):
        from mtpruning.reviewed import trace
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.sqlite"
            journal = trace.Journal(path, {"teacher": "a"})
            journal.put("s1", {"tokens": [1, 2]})
            journal.put("s1", {"tokens": [1, 2]})                   # the same sample again: no-op
            with self.assertRaisesRegex(ValueError, "conflicting sample"):
                journal.put("s1", {"tokens": [1, 3]})
            journal.put("s10", {"tokens": [3]})
            journal.put("s2", {"tokens": [2]})
            self.assertEqual([i for i, _ in journal.records()], ["s1", "s10", "s2"])    # lexicographic ID order
            journal.close()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                trace.Journal(path, {"teacher": "b"})
            journal = trace.Journal(path, {"teacher": "a"})
            self.assertEqual(len(list(journal.records())), 3)
            journal.close()
