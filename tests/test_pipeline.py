"""End-to-end dry run of every stage on a random tiny checkpoint.

Needs the released tokenizer files and the FLORES-200 tarball:
    MTPRUNING_TOKENIZER_DIR=<checkpoint dir with tokenizer.json> MTPRUNING_FLORES_TARBALL=<tarball>
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from mtpruning import pipeline
from mtpruning.testing import tiny_checkpoint

TOKENIZER = os.environ.get("MTPRUNING_TOKENIZER_DIR")
TARBALL = os.environ.get("MTPRUNING_FLORES_TARBALL")


@unittest.skipUnless(TOKENIZER and TARBALL, "set MTPRUNING_TOKENIZER_DIR and MTPRUNING_FLORES_TARBALL")
class PipelineDryRun(unittest.TestCase):
    def test_every_stage_on_a_tiny_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = Path(tiny_checkpoint(root / "model", TOKENIZER, seed=11))
            corpus = root / "corpus" / "corpus.jsonl"
            pipeline.main(["corpus", "--tarball", TARBALL, "--tokenizer", TOKENIZER,
                           "--out", str(corpus.parent)])
            common = ["--corpus", str(corpus), "--max-new-tokens", "6", "--threads", "2"]
            run = lambda out: ["run", "--out", out, *common]
            baseline = ["--model", str(model), "--stats", str(root / "baseline/stats"),
                        "--trace-dir", str(root / "baseline/traces"), "--trace-n", "1", "--limit", "2"]
            pipeline.main([*run(str(root / "baseline/records.jsonl")), *baseline])
            pipeline.main(["select", "--model", str(model), "--stats", str(root / "baseline/stats"),
                           "--keep", "4", "--out", str(root / "selection.json")])
            self.assertEqual(len(json.loads((root / "selection.json").read_text())["candidates"]["4"]["kept"]), 2)

            for keep, extra in ((4, ["--selection", str(root / "selection.json")]), (8, [])):
                pipeline.main(["prune", "--model", str(model), *extra, "--keep", str(keep),
                               "--out", str(root / f"pruned{keep}"),
                               "--out-manifest", str(root / f"k{keep}/prune.json")])
                self.assertTrue(json.loads((root / f"k{keep}/prune.json").read_text())["verify"]["pass"])

            # First calibration sentence per direction is traced in every run.
            small = ["--splits", "calibration", "--limit", "1", "--trace-n", "1"]
            pipeline.main([*run(str(root / "k4/masked/records.jsonl")), "--model", str(model),
                           "--mask", str(root / "selection.json"), "--mask-keep", "4",
                           "--trace-dir", str(root / "k4/masked/traces"), *small])
            pipeline.main([*run(str(root / "k4/records.jsonl")), "--model", str(root / "pruned4"),
                           "--trace-dir", str(root / "k4/traces"), "--limit", "2", "--trace-n", "1"])
            pipeline.main([*run(str(root / "k8/records.jsonl")), "--model", str(root / "pruned8"),
                           "--trace-dir", str(root / "k8/traces"), *small])

            def equiv(name, reference, candidate):
                pipeline.main(["equiv", "--reference-traces", str(root / reference / "traces"),
                               "--candidate-traces", str(root / candidate / "traces"),
                               "--reference-records", str(root / reference / "records.jsonl"),
                               "--candidate-records", str(root / candidate / "records.jsonl"),
                               "--out", str(root / name / "equiv.json")])
                return json.loads((root / name / "equiv.json").read_text())

            noop = equiv("k8", "baseline", "k8")
            self.assertTrue(noop["pass"])
            self.assertEqual(noop["bitwise_identical_logits"], noop["sentences"])
            masked = equiv("k4", "k4/masked", "k4")
            self.assertTrue(masked["pass"])
            self.assertEqual(masked["bitwise_identical_logits"], masked["sentences"])

            pipeline.main(["stock-check", "--model", str(root / "pruned4"),
                           "--records", str(root / "k4/records.jsonl"), "--out", str(root / "k4/stock_check.json"),
                           "--splits", "calibration", "--limit", "1", "--max-new-tokens", "6", "--threads", "2",
                           *common[:2]])
            self.assertTrue(json.loads((root / "k4/stock_check.json").read_text())["pass"])

            # A rerun must resume, not duplicate.
            pipeline.main([*run(str(root / "baseline/records.jsonl")), *baseline])
            rows = [json.loads(line) for line in
                    (root / "baseline/records.jsonl").read_text(encoding="utf-8").splitlines()]
            ids = [row["id"] for row in rows if row["type"] == "sentence"]
            self.assertEqual(len(ids), len(set(ids)))

            pipeline.main(["report", "--run-dir", str(root),
                           "--variants", json.dumps({"baseline": "baseline", "4": "k4", "8": "k8"})])
            report = json.loads((root / "report.json").read_text())
            self.assertEqual([r["n"] for r in report["chrf_pp"]], [2] * 12)
            # Only k4 was run on validation/test; k8 (the identity prune) ran on calibration alone.
            self.assertTrue(all(set(r["candidates"]) == {"4"} for r in report["chrf_pp"]))
            self.assertEqual(len(report["spot_check_largest_test_drops"]), 6 * 2)   # two test sentences per direction


if __name__ == "__main__":
    unittest.main()
