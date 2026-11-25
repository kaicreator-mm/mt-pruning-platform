"""Stage-level checks of the pipeline on a dry-run world (see tests/util.py)."""
import json
import shutil
import sqlite3
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
        util.rehearse(cls.world)                    # with the reference-runtime receipts
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
        with self.assertRaisesRegex(SystemExit, "FINAL_TEST_DISABLED"):
            pipeline.execute_run(util.options(self.world, splits=["test"]))
        with self.assertRaisesRegex(SystemExit, "FINAL_TEST_DISABLED"):
            pipeline.execute_run(util.options(self.world, splits=["validation", "test"]))
        with self.assertRaises(SystemExit) as stopped:   # the final-test command itself is gone
            pipeline.main(["final-test", "--run-dir", str(self.world["run"]), "--decision-sha256", "0" * 64,
                           "--review-ref", "any string"])
        self.assertEqual(stopped.exception.code, 2)      # argparse: no such command
        with self.assertRaisesRegex(SystemExit, "FINAL_TEST_DISABLED"):
            util.cli(self.world, "stock-check", "--model", self.world["model"], "--corpus", self.world["corpus"],
                     "--records", self.world["run"] / "k64" / "records.jsonl", "--out", self.world["run"] / "never.json",
                     "--splits", "test", "--limit", 1, "--max-new-tokens", 6, "--threads", 2)
        self.assertFalse((self.world["run"] / "never.json").exists())

    def test_real_mode_boundaries(self):
        with self.assertRaisesRegex(SystemExit, "frozen corpus manifest only"):
            pipeline.execute_run(util.options(self.world, mode="real"))
        selection = self.world["run"] / "selection.json"
        with self.assertRaisesRegex(SystemExit, "tokenizer override"):
            pipeline.execute_run(util.options(self.world, mode="real", tokenizer=str(selection)))
        with self.assertRaisesRegex(SystemExit, "refuses --limit"):
            pipeline.execute_run(util.options(self.world, mode="real", limit=1))
        from mtpruning.identity import run_identity
        arguments = dict(model_dir=self.world["model"], corpus_path=self.world["corpus"],
                         manifest_path=self.world["run"] / "corpus" / "corpus_manifest.json",
                         decoding="greedy (argmax of bf16 logits)", max_new_tokens=6,
                         protocol={"splits": ["validation"]}, threads=2, mode="real")
        with self.assertRaisesRegex(ValueError, "synthetic checkpoint is only a rehearsal teacher"):
            run_identity(**arguments)

    def test_decide_requires_the_full_receipt_chain(self):
        def drop_equiv(run):
            (run / "k64" / "equiv.json").unlink()

        def drop_stock(run):
            (run / "k64" / "stock_check.json").unlink()

        def drop_noop(run):
            (run / "k128" / "equiv.json").unlink()

        def swapped(run):    # a prior run's receipt in place of this one's
            shutil.copy(run / "k96" / "equiv.json", run / "k64" / "equiv.json")

        def altered(run):    # records edited after the receipts were written
            path = run / "k64" / "records.jsonl"
            rows = lines(path)
            rows[1]["text"] = rows[1]["text"] + " later"
            write(path, rows)

        def interrupted_journal(run):    # the journal moved on after the records were exported
            database = sqlite3.connect(str(run / "k64" / "records.jsonl") + ".sqlite")
            victim = lines(run / "k64" / "records.jsonl")[1]["id"]
            database.execute("DELETE FROM samples WHERE id = ?", (victim,))
            database.commit()
            database.close()

        def narrowed(run):   # narrowing BOTH the selection file and the CLI list (R13-P1-03)
            data = json.loads((run / "selection.json").read_text(encoding="utf-8"))
            data["candidates"].pop("64")
            (run / "selection.json").write_text(json.dumps(data), encoding="utf-8")

        def verify_flag(run):    # the single-field counterexample: prune.verify.pass -> false
            path = run / "k64" / "prune.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["verify"]["pass"] = False
            path.write_text(json.dumps(data), encoding="utf-8")

        def kept_ids(run):       # kept sets edited inside the selection file
            data = json.loads((run / "selection.json").read_text(encoding="utf-8"))
            kept = data["candidates"]["64"]["kept"]
            kept["1"] = list(reversed(kept["1"]))
            (run / "selection.json").write_text(json.dumps(data), encoding="utf-8")

        def stale_noop(run):     # a prior run's no-op receipt in place of this one's
            shutil.copy(run / "k96" / "equiv.json", run / "k128" / "equiv.json")

        def witness_deleted(run):    # a retained witness trace damaged after the equivalence
            path = next((run / "k64" / "masked" / "traces").glob("*.trace.npz"))
            path.write_bytes(path.read_bytes()[:64])
        for change, message in ((drop_equiv, "k64 equivalence receipt is missing"),
                                (drop_stock, "k64 reference-runtime receipt is missing"),
                                (drop_noop, "the identity no-op equivalence receipt is missing"),
                                (swapped, "k64's equivalence receipt does not belong"),
                                (stale_noop, "the identity no-op receipt does not belong"),
                                (altered, "k64 validation replay"),
                                (interrupted_journal, "k64 validation replay"),
                                (narrowed, "are not exactly the plan's"),
                                (verify_flag, "k64's prune reload did not pass"),
                                (kept_ids, "kept set for 64 does not match the deterministic selection rule"),
                                (witness_deleted, "k64 masked replay.*does not match the journal's record")):
            with self.subTest(message):
                run = self.copy(change)
                with self.assertRaisesRegex(SystemExit, message):
                    util.cli(self.world, "decide", "--run-dir", run, "--candidates", "96,64", "--out", run / "never.json")
                self.assertFalse((run / "never.json").exists())

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
        with self.assertRaisesRegex(SystemExit, "does not match the journal's record"):
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
        for change, message in ((mixed, "does not match the journal's identity: teacher"),
                                (renamed, "not pruned the way the recomputed selection names"),
                                (exposed, "reference-runtime receipt does not match")):
            with self.subTest(message):
                run = self.copy(change)
                with self.assertRaisesRegex(SystemExit, message):
                    util.cli(self.world, "decide", "--run-dir", run, "--candidates", "96,64", "--out", run / "never.json")

    def test_decide_counts_the_witness_from_the_journal_not_the_summary(self):
        """R13-P1-03: editing the summary's audit (zero->positive or positive->zero) changes
        nothing - decide rebuilds the changed-route count from the committed journal."""
        def zeroed(run):
            path = run / "k64" / "masked" / "records.jsonl.summary.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["mask_audit"] = {direction: {phase: {**values, "changed": 0} for phase, values in phases.items()}
                                  for direction, phases in data["mask_audit"].items()}
            path.write_text(json.dumps(data), encoding="utf-8")

        def inflated(run):
            path = run / "k64" / "masked" / "records.jsonl.summary.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["mask_audit"] = {direction: {phase: {**values, "changed": values["changed"] + 10_000}
                                              for phase, values in phases.items()}
                                  for direction, phases in data["mask_audit"].items()}
            path.write_text(json.dumps(data), encoding="utf-8")
        for change in (zeroed, inflated):
            with self.subTest(change.__name__):
                run = self.copy(change)
                util.cli(self.world, "decide", "--run-dir", run, "--candidates", "96,64", "--out", run / "decision.json")
                self.assertTrue((run / "decision.json").exists())

    def test_stock_refuses_mislabeled_test_context_and_foreign_model_without_loading(self):
        """R13-P1-04/P1-05: a frozen test-context sentence smuggled in under a validation row
        label is refused by the journal replay (rows must equal journal payloads must equal
        the frozen contexts) - and a stock-check whose model is not the records' candidate
        is refused before anything is loaded. Both refusals happen with zero model loads."""
        from unittest import mock

        def forge(run):
            from mtpruning.frozen import load_frozen_corpus
            from mtpruning.pipeline import identity_of, journal_path
            from mtpruning.reviewed import trace as reviewed_trace
            records = run / "k64" / "records.jsonl"
            binding = read_json(Path(str(records) + ".identity.json"))
            full = identity_of(binding)
            _, contexts = load_frozen_corpus(run / "corpus", self.world["manifest_sha256"])
            test_id = next(i for i, c in contexts.items() if c["split"] == "test")
            context = contexts[test_id]
            tokens = len(context["prompt_ids"])
            cells = {"layer": [1] * 8 + [2] * 8, "expert": list(range(8)) * 2,
                     "count": [tokens] * 8 + [tokens] * 8, "mass": [1.0] * 16}
            routing = {"prefill": {"tokens": tokens, **cells},
                       "decode": {"tokens": 1, "layer": [1] * 8 + [2] * 8, "expert": list(range(8)) * 2,
                                  "count": [1] * 16, "mass": [1.0] * 16}}
            payload = {"identity_key": full["sample_keys"]["test"], "split": "test", "direction": context["direction"],
                       "line": context["line"], "prompt_tokens": tokens, "new_tokens": 1,
                       "finish": "eos", "tokens": [5], "text": "x", "routing": routing, "audit": None, "trace": None}
            journal = reviewed_trace.Journal(journal_path(records), binding)
            journal.put(test_id, payload)
            journal.close()
            rows = lines(records)
            rows.append({"type": "sentence", "id": test_id, "split": "validation", "direction": context["direction"],
                         "line": context["line"], "prompt_tokens": tokens, "new_tokens": 1,
                         "finish": "eos", "tokens": [5], "text": "x", "identity_key": payload["identity_key"],
                         "traced": False})
            write(records, rows)
            summary_path = Path(str(records) + ".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["journal"]["sha256"] = util.sha256(journal_path(records))
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

        def no_load(model_cls, path, **kwargs):
            raise AssertionError("the reference runtime must not be loaded for a refused stock check")
        for change, arguments, message in (
                (forge, ["--model", self.world["root"] / "pruned" / "k64"],
                 "differ from what the journal holds"),
                (lambda run: None, ["--model", self.world["model"]],
                 "is not the checkpoint whose records this reference check is defined for")):
            with self.subTest(message):
                run = self.copy(change)
                with mock.patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=no_load), \
                        mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=no_load):
                    with self.assertRaisesRegex(SystemExit, message):
                        util.cli(self.world, "stock-check", *arguments, "--corpus", self.world["corpus"],
                                 "--records", run / "k64" / "records.jsonl", "--out", run / "never.json",
                                 "--splits", "validation", "--limit", 1, "--max-new-tokens", 6, "--threads", 2,
                                 "--cache-dir", self.world["cache"])
                self.assertFalse((run / "never.json").exists())

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

    def test_the_editable_summary_never_drives_the_witness_gate(self):
        """R13-P1-03: equiv and decide count the changed-route witness from the committed
        journal, so editing the summary (zero->positive or positive->zero) changes nothing:
        a real witness still passes with its summary zeroed, and a zero journal witness is
        refused however positive the summary claims to be."""
        from mtpruning import pipeline as pipeline_module

        def edit_summary(run, value):
            path = run / "k96" / "masked" / "records.jsonl.summary.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["mask_audit"] = {direction: {phase: {**values, "changed": value} for phase, values in phases.items()}
                                  for direction, phases in data["mask_audit"].items()}
            path.write_text(json.dumps(data), encoding="utf-8")

        def equiv(run):
            folder = run / "k96"
            util.cli(self.world, "equiv", "--kind", "masked", "--keep", 96, "--run-dir", run,
                     "--prune-record", folder / "prune.json", "--trace-n", 2,
                     "--candidate-traces", folder / "mechanism" / "traces",
                     "--candidate-records", folder / "mechanism" / "records.jsonl",
                     "--reference-traces", folder / "masked" / "traces",
                     "--reference-records", folder / "masked" / "records.jsonl",
                     "--selection", run / "selection.json", "--out", folder / "equiv.json")
            return json.loads((folder / "equiv.json").read_text(encoding="utf-8"))["pass"]
        run = self.copy(lambda run: edit_summary(run, 0))
        self.assertTrue(equiv(run))                       # the journal's witness stands on its own
        util.cli(self.world, "decide", "--run-dir", run, "--candidates", "96,64", "--out", run / "decision.json")
        self.assertTrue((run / "decision.json").exists())
        # a genuinely zero witness (identity mask) is refused however positive the summary is:
        # the masked run's journal carries changed=0, and its physical no-op counterpart passes
        util.cli(self.world, "select", "--run-dir", run, "--keep", "128", "--out", run / "sel128.json",
                 "--cache-dir", self.world["cache"])
        identity_mask = dict(splits=["validation"], limit=2, stats=None, trace_n=2,
                             mask=str(run / "sel128.json"), mask_keep=128)
        zero_folder = run / "zero"
        pipeline_module.execute_run(util.options(self.world, out=zero_folder / "masked" / "records.jsonl",
                                                 trace_dir=zero_folder / "masked" / "traces", **identity_mask))
        summary_path = zero_folder / "masked" / "records.jsonl.summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        forged = {**summary, "mask_audit": {direction: {phase: {**values, "changed": 99} for phase, values in
                                                       phases.items()} for direction, phases in
                                          (summary.get("mask_audit") or {}).items()}}
        summary_path.write_text(json.dumps(forged), encoding="utf-8")      # claims 99 changes
        pipeline_module.execute_run(util.options(self.world, model=self.world["root"] / "pruned" / "k128",
                                                 out=zero_folder / "mechanism" / "records.jsonl",
                                                 trace_dir=zero_folder / "mechanism" / "traces",
                                                 splits=["validation"], limit=2, stats=None, trace_n=2))
        with self.assertRaisesRegex(SystemExit, "insufficient evidence"):
            util.cli(self.world, "equiv", "--kind", "masked", "--keep", 128, "--run-dir", run,
                     "--prune-record", run / "k128" / "prune.json", "--trace-n", 2,
                     "--candidate-traces", zero_folder / "mechanism" / "traces",
                     "--candidate-records", zero_folder / "mechanism" / "records.jsonl",
                     "--reference-traces", zero_folder / "masked" / "traces",
                     "--reference-records", zero_folder / "masked" / "records.jsonl",
                     "--selection", run / "sel128.json", "--out", run / "never2.json")
        self.assertFalse((run / "never2.json").exists())

    def test_a_corrupt_or_missing_trace_is_regenerated_from_the_journal(self):
        run, p = self.world["run"], pipeline
        rows = [json.loads(line) for line in (run / "baseline" / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        traced_id = next(row["id"] for row in rows[1:] if row.get("traced"))
        from mtpruning.evaluate import trace_path
        target = trace_path(run / "baseline" / "traces", traced_id)
        before, journal_before = util.sha256(target), util.sha256(run / "baseline" / "records.jsonl")
        target.write_bytes(b"corrupted")                    # an interrupted or damaged witness file
        summary = p.execute_run(util.options(self.world))
        self.assertEqual(util.sha256(target), before)       # regenerated bit for bit
        self.assertEqual(util.sha256(run / "baseline" / "records.jsonl"), journal_before)
        self.assertEqual(summary["commits_this_invocation"], 1)   # exactly the damaged sample reran
        target.unlink()                                     # and a fully missing one comes back too
        summary = p.execute_run(util.options(self.world))
        self.assertEqual(util.sha256(target), before)
        self.assertEqual(summary["commits_this_invocation"], 1)

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
