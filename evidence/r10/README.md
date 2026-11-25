# R10 — correction and pre-real-run gate (receipt)

Task: issue #1 comment 7171469761 (`MT_PRUNING_D01B_R10_THREE_P1_CORRECTION`), addressing the three P1 findings of
PR #4 comment 7170552638. PR #4 stays a draft.

**Status: corrections implemented and tested on CC-LAN200. No real calibration has been started, there is no real
quality result, the test split has not been run for any candidate, and the final test is on HOLD.**

## 1. Local driver, inspected first

- The pre-R10 driver was found in its download wait loop; no stage had started. Its whole log is
  `uncorrected-driver.log` (one line). It was stopped (SIGTERM) so that uncorrected code cannot launch a real
  calibration, and no `run_d01b.sh` or pipeline process has run since.
- The download was left alone and is still running. No verified shard was deleted or downloaded again.
- The run directory holds only the frozen corpus and its receipt: there is no baseline journal, no candidate
  output, no selection. A GitHub comment stops nothing by itself; this was done by hand on the machine.

## 2. History

- Review base `93a6637ff2c231140d8d2abcbf18822da9f70498`; main `b2b486fcfd3db1820e14aaba389f3ae002688d10`
  (reviewed offline demos R1–R8) was merged with a merge commit, `79de5631d44f0b566fbc9ca3e2b15ed87a701ef9`.
  No reviewed history was rewritten. `demos/` is byte for byte what main has.
- The R10 commit sits on top of `79de563`; its hash is reported in issue #1 and PR #4 (a commit cannot contain
  its own hash).

## 3. P1-01 — chrF++ gate had the wrong sign

- Was: `candidate - baseline <= gate`, which passed any drop and failed improvements larger than the gate.
- Now: `report.passes_gate(delta, gate)` is `delta >= -gate`, with a finite non-negative gate and a finite delta;
  anything else fails. The three diagnostics (empty, truncated, digits differ from the reference) gate as well:
  a candidate passes a direction only if the score does not drop by more than the gate and no diagnostic count
  grows.
- The verdict itself comes from the reviewed `r6/bridge.py:quality_gate`, which never trusts the report's own
  flags; `agree_with_reviewed_gate` raises if the two disagree.
- Tests (`tests/test_report.py:GateSignTests`): −1.01 fails, −1.0 passes, +2.0 passes, −0.5 passes, −7.5 fails;
  NaN/±inf/string/bool/None deltas and NaN/inf/negative/string/bool/None gates all fail; zero gate; a good score
  with one new empty output fails; five of six directions, an empty table, a duplicated direction, a short group,
  a missing candidate, a non-finite or out-of-range score and non-integer or unknown flags are NOT_EVALUATED.

## 4. P1-02 — resume lost calibration data; identity too weak to notice

- Every sample is committed once, atomically, with its output, its routing cells (count and mass per layer and
  expert, both phases) and its mask audit, to a SQLite journal: the reviewed `r4/trace.py:Journal`, which refuses
  a conflicting sample, a corrupted payload and a foreign identity.
- Aggregates are never carried in memory across restarts and never written from a partial run: they are rebuilt
  from the committed samples, in ID order, whenever needed. A finished run re-exported is byte for byte the same.
- Before selecting, `select` rebuilds the statistics from the journal, requires the exported files to equal that
  rebuild, and then runs the reviewed coverage check `r6/bridge.py:adapt` (exact calibration IDs per direction,
  prompt and decode token counts, per-layer counts and masses for both phases).
- A journal opens only under the identity it was started with. The identity follows the reviewed
  `r5/cache_plan.py` fields (teacher, candidate, corpus, prompt, runtime, protocol, code, split) and binds:
  - **checkpoint content**: every shard is hashed and compared with what the directory claims to be (the pinned
    HF listing for the released snapshot, the prune manifest for a pruned checkpoint, or a declared synthetic
    provenance). A directory with unknown provenance is refused;
  - tokenizer files, the prompt templates and the chat template;
  - the frozen corpus bytes and manifest; the mask and the hash of the selection file;
  - runtime (Python, torch, transformers, safetensors, numpy, sacrebleu, platform, host, threads);
  - code: hashes of every executed file, including the reused helpers;
  - decoding settings and max new tokens, which are part of the protocol hash and so of every sample key.
- Tests (`tests/test_journal.py`, `tests/test_identity.py`): a run interrupted twice and resumed equals an
  uninterrupted run (outputs, aggregate arrays, audit); rerunning a finished run commits nothing and changes
  nothing; a conflicting duplicate and a corrupted payload are refused; a sample committed under another
  identity is refused; a journal cannot continue under another decoding setting, limit, trace set, statistics
  mode, split scope, mask, code or runtime; changed corpus bytes give `BLOCKED_CORPUS_CONTRACT`; an interrupted
  calibration has no aggregate and cannot be selected; an edited or stale aggregate is refused; the masked
  audit of an interrupted masked run equals the uninterrupted one.

## 5. P1-03 — vacuous PASS on missing coverage; test exposure

- The frozen corpus is checked at every stage: manifest bytes against the pinned hash, corpus bytes against the
  manifest, then the reviewed `r7/handoff.py:corpus` (exact IDs, sizes, line numbers, alignment and the R8
  duplicate rules). Any failure is `BLOCKED_CORPUS_CONTRACT`; nothing is resampled.
- Validation records are loaded through the reviewed `r7/handoff.py:records`, which requires exactly the
  validation IDs of all six directions, rejects duplicates, extra IDs and wrong split/direction/line, and
  refuses a candidate run that contains test records.
- Report rows are aligned by ID. A candidate with incomplete or invalid records gets no row entry; the reviewed
  gate then reports NOT_EVALUATED for that split and the verdict names the reason. There is no `all()` over
  whatever rows happen to exist.
- The provisional decision is `r7/handoff.py:evaluate` (validation only, `test_execution_authorized: false`),
  after checking that the baseline and every candidate share teacher, corpus, prompt, runtime, code and decoding,
  and that each candidate was pruned exactly as the selection names.
- The driver no longer touches the test split. It runs: corpus check → checkpoint content → baseline
  (calibration + validation) → selection → identity reload (no-op) → per candidate: physical prune, masked
  equivalence, reference runtime, complete validation → provisional decision → report. `run` refuses the test
  split outright; only `final-test` can run it, it needs the hash of the reviewed `decision.json` and a
  reference to that review, and it records the use in `test_exposure.json`. A second candidate is refused.
- The mechanism checks now run on held-out validation sentences. On the calibration sentences the selection was
  built from, the pruned experts are the ones hardly ever chosen, so the mask would not have changed any routing
  decision and the check would have been vacuous. The first dry run on the random checkpoint was exactly that:
  0 changed decisions. `tests/test_pipeline.py:test_the_dry_run_really_routes_around_pruned_experts` now fails if
  it happens again. It requires at least 20 changed prefill and 5 changed decode decisions with 64 experts kept
  (observed: 10–118 of 132–702 prefill and 1–6 of 12 decode decisions per direction) while the physical prune
  still equals the masked original bit for bit.
- Tests (`tests/test_report.py:ReportCoverageTests`, `tests/test_pipeline.py`, `tests/test_driver.py`):
  a missing direction, a duplicated record, an extra ID, a wrong split, a reshaped token list and a header-only
  run are NOT_EVALUATED with the reason; an incomplete baseline leaves no candidate evaluated; a candidate that
  saw the test split marks the report exploratory; changed corpus bytes stop the report, the `run` stage and the
  whole driver before anything runs, leaving the corpus as found; `decide` refuses mixed identities, a candidate
  not pruned as named, and an exposed candidate; the final test needs the reviewed hash and a review reference,
  runs once, resumes without duplicating, and refuses another candidate.

## 6. Checks on this tree

```
cd ~/mt-pruning-platform
python -m compileall -q mtpruning tests
python -m pyflakes mtpruning tests                      # no findings
TMPDIR=<on disk> MTPRUNING_TOKENIZER_DIR=<snapshot> MTPRUNING_FLORES_TARBALL=<tarball> \
  python -m unittest discover -s tests -t . -v
```

`python -m unittest discover -s tests -t . -v` → **Ran 78 tests in 353.2 s, OK**, none skipped (`unittest-full.log`):
`test_driver` 4, `test_evaluate` 13, `test_engine` 13, `test_identity` 8, `test_journal` 10, `test_pipeline` 7,
`test_prune` 11, `test_report` 12.
`test_engine` still holds the engine to bit-identical logits against stock transformers 5.24.1.

Tests write temporary files under `~/.cache/mt-pruning/test-tmp`, not `/tmp`: `/tmp` on CC-LAN200 is a 31 GB
tmpfs with a per-user quota. The pinned environment is unchanged (`evidence/env/pip-freeze.txt`).

## 7. Reviewed helpers reused (loaded from their files, hashed into every run's code identity)

| file | sha256 |
|---|---|
| `demos/followup/r4/trace.py` | `5efb59d5ff98fe20bcb4b4a59c1507a3a42628b4984a83899c156383708f64d0` |
| `demos/followup/r5/cache_plan.py` | `dcb41f692cadcd5b064b764ab936e9392d19fd15035de60172382e62e7724a5b` |
| `demos/followup/r6/bridge.py` | `11b60c9c7989da454586c193208b9c7707d33d37d7a0af70a34266d9e8a28d8c` |
| `demos/followup/r7/handoff.py` | `cfb98954182758250e5c1b457cec2f5bb576a22ce1284f3406d81cdd3748c658` |

They check aggregates, records, the corpus and gates. They do not prove checkpoint provenance; `mtpruning/identity.py`
does that, and it has not been run against the real checkpoint yet (download incomplete).

## 8. Frozen corpus receipt

`evidence/d01b/corpus/corpus_receipt.json`, produced on the real corpus by
`python -m mtpruning corpus-check --corpus-dir ~/.cache/mt-pruning/run-d01b/corpus …` with the committed manifest
hash as the pin:

- `FROZEN_CORPUS_CONTRACT_OK`, 384 records, 24 / 20 / 20 sentences per direction;
- manifest sha256 `e516cc92a805eb3273c9376741ba5b4f24229d03d7189f0275d2829de848b901`;
- corpus sha256 `9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4`;
- FLORES-200 tarball sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`;
- R8 duplicate rules of `r7/handoff.corpus` passed on the actual bytes;
- rebuilding the corpus from the tarball into a scratch directory gave the same bytes for both files.
  The frozen corpus was not touched.

## 9. Checkpoint and runtime status (CC-LAN200, 2026-11-25 20:28 UTC)

- Download straight from huggingface.co, resumable, per-file verification: 32.77 GB of 60.13 GB received (54.5 %).
  12 of 19 files verified so far: the six small files and shards 02, 03, 04, 09, 12, 13. Shards 01, 05, 06, 07, 08,
  10 and 11 are in progress. The link is Wi-Fi at roughly 2–3 MB/s, so these numbers move; the snapshot above is
  the moment of writing.
- `checkpoint-check` (hash every shard against the pinned listing): **NOT_RUN**, the download is incomplete.
- Real calibration, selection, mechanism checks, candidate validation: **NOT_RUN**.
- Machine otherwise idle for this task: no driver, no pipeline process.

## 10. Was the old run or the test exposed?

No. The pre-R10 driver never left its wait loop, so there is no baseline output, no candidate output and no
journal. The test split has not been run for any candidate, and neither has validation. No fresh held-out set
is needed.

## 11. Not claimed

Everything under "tests" is synthetic: a random-weight 128-expert checkpoint and a small FLORES sample at toy
scale. It shows the protocol and the checks work; it says nothing about translation quality. No real quality PASS
exists yet.

## 12. Next

Independent scoped review of the exact head. After acceptance, on CC-LAN200 with the retained weights:

```
bash scripts/run_d01b.sh          # stops at the provisional decision; never runs the test split
```

and, once that decision is reviewed, `FINAL_TEST=1 REVIEWED_DECISION_SHA256=… REVIEW_REF=… bash scripts/run_d01b.sh`.
