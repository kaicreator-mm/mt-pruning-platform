# R19 — HY-V3 attention correction passed; real pilot blocked by k96 mechanism equivalence

OUTPUT_LABEL=MT_PRUNING_D01B_R19_ATTENTION_WIDTH_FIX_AND_REAL_PILOT_RESTART

## Verdict

**CORRECTION_AND_REGRESSION=PASS. REAL_SMOKE=PASS. BASELINE_AND_K128=PASS. K96_MECHANISM_EQUIVALENCE=FAIL. OVERALL=BLOCKED.**

The R18 attention-width defect is corrected and verified on the real model. A new source identity,
RUN, pruned-output directory, and evidence directory were used; the R18 journal and failure evidence
were not reused or modified. The complete 144-row calibration, 120-row baseline validation, and 48
baseline witnesses succeeded. Deterministic selection and k128 no-op equivalence also succeeded.

The k96 physical checkpoint and both 48-row mechanism runs completed, but masked-original versus
physical-k96 equivalence returned `pass=false`. The driver therefore stopped before k96 full
validation, stock checks, k64, decision, and report. No tolerance or quality gate was changed, and
no further source correction was attempted under R19.

## Exact source and experiment identity

- Review base / R18 evidence HEAD: `919786dc69f166dfbc928de9bf65e56b3ecfb545`
- Previous accepted source HEAD: `36b10d3bbd9941ad069d6b23284a6feb4a5c0f4b`
- Corrected and tested code HEAD: `085a6ff67d8b869ef0507348507c29f8e5bc7991`
- Corrected source tree: `592e074b66e1e1e7f521aa31cdd8e6809eb9c712`
- Main HEAD at dispatch: `b2b486fcfd3db1820e14aaba389f3ae002688d10`
- Model: `tencent/Hy-MT3-30B-A3B@08ae9acea91abe265c6236e757e49d8851e1ce06`
- Teacher: `0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94`
- Frozen corpus: `9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4`
- New RUN: `/home/cc/.cache/mt-pruning/run-d01b-r19`
- New pruned root: `/home/cc/.cache/mt-pruning/pruned-r19`
- Fixed parameters: `MODE=real CANDIDATES="96 64" TRACE_LIMIT=8 MAX_NEW=640 THREADS=16 FINAL_TEST=0`;
  no LIMIT, tokenizer, manifest, or corpus override.

Only `mtpruning/engine.py`, `mtpruning/testing.py`, and `tests/test_engine.py` changed in the tested
source commit. Engine now distinguishes residual hidden width from `attention_width=n_heads*head_dim`
and `kv_width=n_kv*head_dim`, uses explicit Q/K/V views and attention-width reshape, validates every
layer's Q/K/V/O shapes at load, and rejects nonpositive attention/config dimensions. Pruning and
protocol code were not changed.

## Correction and admission evidence

| Check | Actual result |
|---|---|
| Real all-layer metadata probe | **PASS: 48/48 layers, 192/192 Q/K/V/O tensors, 0 mismatches**; attention=4096, KV=512, hidden=2048. Metadata only, no payload mutation. |
| Focused Engine suite | **30/30 PASS** in 5.064 s; equal-width and hidden=32/attention=64/grouped-KV configurations. |
| Reference prefill, cached decode, greedy | PASS for both configurations; exact required comparisons preserved. |
| Identity no-op and masked-vs-physical prune tests | PASS for both configurations; reference loader and changed-route tests retained. |
| Shape/config negative tests | PASS for Q/K/V/O mismatch names with expected/actual shapes, nonpositive dimensions and invalid head grouping. |
| Full pinned unittest | **124 tests in 265.106 s, OK**, exit 0; no skips reported. |
| compileall / pyflakes / bash syntax / pip check | All exit 0; pyflakes no findings; pip reports no broken requirements. |
| Real first calibration smoke | **PASS**: full baseline identity, 74 prompt tokens, 326 generated tokens, EOS; prefill 12.257 s, decode 113.638 s. |
| Smoke routing | Prefill 27,824 and decode 122,576 selections, exactly `tokens×47×8`; 47/47 sparse layers covered. |
| Real reference model loaded concurrently | NOT_RUN, to avoid exceeding RAM; tiny pinned reference parity is mandatory and passed. |

The real smoke used the existing post-commit crash hook once, then an independent validator checked
the committed sample and full protocol identity. Diagnostic invocation/schema mistakes and the
metadata probe's initial context-manager error are retained in raw logs; none was counted as a
product pass. The final validator exited 0 before the full driver resumed.

## Actual experiment stages

| Stage | Status | Evidence |
|---|---|---|
| Frozen corpus/checkpoint/live resource admission | PASS | Fresh 19/19 checkpoint and corpus receipts; all new output paths. |
| Calibration | **PASS, 144/144** | 24 rows per six directions; 47 sparse layers. |
| Baseline validation | **PASS, 120/120** | 20 rows per direction, all EOS. |
| Baseline witnesses | **PASS, 48/48 traces** | Exact fixed held-out IDs. |
| Deterministic selection | PASS | Coverage-checked 96/64 selections and retained-mass receipt. |
| k128 physical identity | PASS | 13 shards, 60,129,463,808 tensor bytes; tensor reload verify PASS. |
| k128 mechanism equivalence | **PASS** | 48/48 text/argmax/route/fingerprint equality; 10,147 positions; max logit and route-weight differences 0. |
| k96 physical prune/reload | **PASS** | 10 shards, 45,929,772,672 tensor bytes; 14,161 tensors, 0 mismatches. |
| k96 masked-original mechanism run | PASS, 48/48 | 4,307 generated tokens, 48 traces. |
| k96 physical mechanism run | PASS, 48/48 | 4,345 generated tokens, 48 traces. |
| **k96 masked-vs-physical equivalence** | **FAIL** | `equiv.json pass=false`; details below. Dependent stages stopped. |
| k96 full 120 validation and six stock witnesses | NOT_RUN | Not admitted after mechanism failure. |
| k64 prune/mechanism/validation/stock | NOT_RUN | Not started. |
| Provisional decision and canonical quality report | NOT_RUN | Required mechanism chain incomplete; no fallback chosen. |
| Final test | HARD_DISABLED_NOT_RUN | No test-split inference. |

### k96 blocking receipt

`k96/equiv.json` records:

| Metric | Actual value |
|---|---:|
| `pass` | **false** |
| Identical text | 32/48 |
| `all_argmax_agree` | 0/48 |
| `all_route_ids_agree` | 0/48 |
| `within_tol` | 0/48 |
| Length match | 37/48 |
| Positions compared | 1,509 |
| Maximum absolute logit difference | 10.6875 |
| Route ID mismatches | 9,203 |
| Maximum absolute route-weight difference | 1.820320725440979 |
| Masked changed-route witness | 218,003 (nonzero; masking was not vacuous) |

This is a substantive mechanism blocker, not a resource failure and not evidence that the attention
correction failed. Baseline and k128 independently validate the corrected Engine on the real model.
R19 does not authorize an additional pruning/numeric correction or a relaxed equivalence claim;
the cause of k96 non-equivalence requires separate bounded diagnosis and review.

## Selection, size, quality, and speed results

Overall calibration retained mass (the selection stage completed for both planned candidates):

| Direction | k96 | k64 |
|---|---:|---:|
| zh-my | 0.946051 | 0.837986 |
| my-zh | 0.942819 | 0.818607 |
| zh-en | 0.931132 | 0.768510 |
| en-zh | 0.939623 | 0.792610 |
| en-my | 0.947656 | 0.833937 |
| my-en | 0.934896 | 0.789016 |

| Model | Expert retention | Physical bytes | Mechanism result | chrF++ / quality gate |
|---|---:|---:|---|---|
| Original 128 | 128/128 | Snapshot 60,143,159,279; weights 60,129,463,808 | Baseline complete | NOT_RUN by canonical decision/report |
| k128 no-op | 128/128 | Weights 60,129,463,808 | PASS | NOT_RUN |
| k96 | 96/128 = 75% | Checkpoint 45,942,544,507; weights 45,929,772,672 | **FAIL** | NOT_RUN |
| k64 | Planned 64/128 only | NOT_RUN | NOT_RUN | NOT_RUN |

All six direction chrF++ scores, per-direction deltas, and empty/truncated/digit-mismatch comparisons
are **NOT_RUN**. No candidate or 128 fallback was selected. Baseline translations and routing data
are retained and independently usable, but they are not presented as a completed quality decision.

Generation throughput below is measured from committed timings as `new_tokens / sum(decode_seconds)`;
it excludes model loading and prefill and is not a production benchmark.

| Run | Rows | New tokens | Prefill sum (s) | Decode sum (s) | Tokens/decode-second |
|---|---:|---:|---:|---:|---:|
| 128 baseline calibration+validation | 264 | 22,420 | 2,770.715 | 7,639.714 | 2.9347 |
| k128 mechanism | 48 | 4,235 | 541.888 | 1,466.514 | 2.8878 |
| k96 masked-original mechanism | 48 | 4,307 | 501.344 | 1,499.115 | 2.8730 |
| k96 physical mechanism | 48 | 4,345 | 513.340 | 1,489.538 | 2.9170 |

## Interruption and resource record

The first attached baseline run completed and exported all 264 rows and 48 traces, then its external
`tee` received SIGPIPE after an SSH reset (exit 141). This was an operational output-pipe interruption,
not a Python product exception. The same journal was independently checked and resumed with zero
sample recomputation (`commits_this_invocation=0`). All subsequent logging was detached from SSH.

The detached resume ran 1:50:12 before the real k96 equivalence failure. GNU time reported user CPU
95,503.41 s, system CPU 960.77 s, maximum RSS 61,853,228 KiB, and `Swaps: 0`. Monitored swap peaked
at about 1.28 GB and had headroom; disk preflights passed before every admitted stage. The failed k96
checkpoint remains retained rather than being deleted by the completed-driver cleanup path.

## Evidence and retention

- Original R18 evidence remains intact: 15/15 manifest checks passed again.
- R19 retained manifest: **286/286 files SHA256 verified**, including RUN journals and traces, the
  physical k96 checkpoint, correction tests, smoke, interruption/resume logs, and diagnostic scripts.
- Retained manifest SHA256: `1f5be17ef29c73487e26ad74b8fbbb168c23c07212e8e1c10818fd936b59752e`.
- Retained root: `/home/cc/.cache/mt-pruning`; reverify with
  `sha256sum -c r19-real/retained_manifest.sha256` from that root.
- `evidence/d01b-r19/` carries compact run receipts, records, selection, stats, prune and equivalence
  JSON/JSONL/NPZ (about 2 MB); bulky SQLite journals and raw traces stay in the original RUN.
- `evidence/r19/` carries the correction/test/smoke/report logs, this report, machine receipt, and
  compact/retained manifests. Raw host-network details and model weights are not published.
- `execution_receipt.json` is the machine-readable stage and metrics record. Public compact files
  are independently bound by `compact_manifest.sha256` and `compact_manifest.check.txt`.

DOWNLOAD=UNTOUCHED; MODEL_CONFIG=UNTOUCHED; FROZEN_CORPUS=UNTOUCHED; QUALITY_GATE=UNCHANGED;
NEW_AGENT=NOT_RUN; FINAL_TEST=NOT_RUN; TRAINING=NOT_RUN; GGUF=NOT_RUN; GPU_DEPLOYMENT=NOT_RUN;
GITHUB_ACTIONS=NOT_RUN; MERGE=NOT_RUN. No final-quality, production, or final-test claim is made.
