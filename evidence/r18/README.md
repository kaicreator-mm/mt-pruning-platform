# R18 — real-model execution blocked at the first calibration prefill

OUTPUT_LABEL=MT_PRUNING_D01B_R18_REAL_MODEL_CALIBRATION_PRUNING_VALIDATION

## Verdict

**OVERALL=BLOCKED. Baseline execution FAIL; every dependent pruning and quality stage NOT_RUN.**

The reviewed driver was invoked once with the required real parameters at reviewed HEAD
`36b10d3bbd9941ad069d6b23284a6feb4a5c0f4b`. Checkpoint and frozen-corpus verification passed.
The first calibration sample then failed in attention prefill with a real tensor-dimension
mismatch. The SQLite journal contains zero completed samples. No routing aggregate, baseline
translation, pruned checkpoint, quality score, or candidate selection was produced.

This is a material source blocker under the R18 dispatch. No product source, experiment protocol,
corpus, sample count, witness count, precision, or quality gate was changed, and no patch was
silently applied.

## Exact identity and fixed invocation

- Reviewed source HEAD: `36b10d3bbd9941ad069d6b23284a6feb4a5c0f4b`
- Reviewed source tree: `7a35004b49535c1cad84ede4621e29ad1889fe0c`
- Tested code HEAD: `591f154365bcfe0b49b2d463767f5b3c60f00106`
- Main HEAD at dispatch: `b2b486fcfd3db1820e14aaba389f3ae002688d10`
- Driver blob: `4a66f2ad1ccbfc0e49a249d76476d340c0b108d0`
- Model: `tencent/Hy-MT3-30B-A3B@08ae9acea91abe265c6236e757e49d8851e1ce06`
- Teacher digest: `0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94`
- Frozen corpus SHA256: `9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4`
- Parameters: `MODE=real`, `CANDIDATES="96 64"`, `TRACE_LIMIT=8`, `MAX_NEW=640`,
  `THREADS=16`, `FINAL_TEST=0`; no LIMIT, manifest, tokenizer, or corpus override.
- Start/end UTC: `2026-11-26T10:08:34Z` / `2026-11-26T10:10:26Z`; driver exit code 1.

`command.txt`, `source_identity.txt`, and `source_hashes.sha256` bind the invocation and live
source. All reviewed source hashes still match the R16 receipt.

## Stage status

| Stage | Actual status | Real result |
|---|---|---|
| Live source/environment/resource preflight | PASS | Clean reviewed tree; pinned Python imports and `pip check` passed; no competing experiment found. |
| Checkpoint content | PASS | Fresh `CHECKPOINT_CONTENT_VERIFIED`, pinned snapshot, 19/19 files, 128 experts. |
| Frozen corpus contract | PASS | 384 records total; calibration 144 and validation 120 available under the frozen contract; test split not executed. |
| Baseline disk preflight | PASS | 284,328,960,000 bytes free versus 68,018,291,691 conservative requirement. |
| Calibration, 24 × 6 = 144 | **FAIL** | First sample prefill failed; **0/144 completed**, no routing statistics. |
| Baseline validation, 20 × 6 = 120 | NOT_RUN | 0/120; no translations or 48 witness traces. |
| Deterministic 96/64 selection | NOT_RUN | No calibration journal coverage exists. |
| k128 physical identity and witness equivalence | NOT_RUN | Dependent stage fail-stopped. |
| k96 physical prune/reload/equivalence/validation/stock | NOT_RUN | No candidate generated. |
| k64 physical prune/reload/equivalence/validation/stock | NOT_RUN | No candidate generated. |
| Provisional decision and canonical report | NOT_RUN | Required receipt chain is absent; no fallback selection was made. |
| Final test | HARD_DISABLED_NOT_RUN | `FINAL_TEST=0`; no test inference. |

## Blocking failure

The complete unmodified traceback is in `driver.log`. Its terminal exception is:

```text
File "/home/cc/mt-pruning-platform/mtpruning/engine.py", line 408, in forward
    h = residual + F.linear(attn.transpose(0, 1).reshape(n, self.hidden), layer.wo)
RuntimeError: shape '[74, 2048]' is invalid for input of size 303104
```

A metadata-only safetensors probe (`failure_probe.json`, reproducible with `failure_probe.py`)
confirmed the dimensions without loading tensor payloads or performing inference:

| Quantity | Real checkpoint value |
|---|---:|
| `hidden_size` | 2048 |
| `num_attention_heads` | 32 |
| `head_dim` | 128 |
| Attention output width (`32 × 128`) | 4096 |
| Layer-0 `q_proj.weight` shape | `[4096, 2048]` |
| Layer-0 `o_proj.weight` shape | `[2048, 4096]` |
| Failed prefill tokens | 74 |
| Failed input elements | 303104 = `74 × 4096` |
| Requested reshape | `[74, 2048]` |

Thus the reviewed reshape uses `hidden_size=2048` although this checkpoint's attention projection
and `o_proj` input width are 4096. This diagnosis does not authorize or validate a fix. A bounded
source correction, regression coverage using this decoupled attention-width configuration, and a
new reviewed real execution are required before dependent R18 stages can proceed.

## Model, quality, and candidate results

The retained original snapshot measures 60,143,159,279 bytes; its safetensors index declares
60,129,463,808 weight bytes. These are original-input measurements, not a pruned-model result.
The pruning output directory is 0 bytes.

| Candidate | Physical model | Expert retention | chrF++ / diagnostics | Tokens/s | Candidate peak RSS |
|---|---|---:|---|---|---|
| Original 128 baseline | Input verified; baseline run failed | 128/128 input | NOT_RUN | NOT_RUN | NOT_RUN |
| k128 no-op | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| k96 | NOT_RUN | Planned 96/128 only; not realized | NOT_RUN | NOT_RUN | NOT_RUN |
| k64 | NOT_RUN | Planned 64/128 only; not realized | NOT_RUN | NOT_RUN | NOT_RUN |

| Direction | Baseline chrF++ | k96 chrF++ | k64 chrF++ | Empty/truncated/digit comparison |
|---|---|---|---|---|
| zh-my | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| my-zh | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| zh-en | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| en-zh | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| en-my | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| my-en | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

**Candidate selection=NOT_RUN.** The 128-expert fallback was not invoked because the required
chain never reached `decide`; presenting it as a quality-selected result would be false.

## Actual resource receipts for the failed attempt

These values describe only checkpoint/corpus checks, loading, and the failed first prefill. They
are not translation-throughput or steady-state inference measurements.

| Measurement | Actual value |
|---|---:|
| Wall time | 111.72 s (`1:51.72`) |
| User / system CPU | 91.03 s / 30.82 s |
| CPU utilization reported by GNU time | 109% |
| Driver maximum RSS | 6,647,940 KiB (6,807,490,560 bytes) |
| Swap at start and final observation | 105,123,840 bytes used; GNU time reported `Swaps: 0` |
| Available memory at 10:09:34Z sample | 62,900,801,536 bytes |
| Final available disk | 284,328,730,624 bytes |
| Pruned output | 0 bytes |
| Generated tokens / tokens per second | 0 completed samples; NOT_MEASURED |

## Retained raw evidence and public compact evidence

- Original retained root: `/home/cc/.cache/mt-pruning` on CC-LAN200.
- `retained_manifest.sha256` covers 15 files under `run-d01b/` and `r18-real/`, including the
  SQLite journal, identity, zero-byte timings file, corpus, driver/resource logs, and failure probe.
- Independent `sha256sum -c` completed with **15/15 OK**; output is in
  `retained_manifest.check.txt`.
- Retained manifest SHA256: `866a2fd808d0ad619051c2df017523c6a3d768ffc534134615c6a165f501b738`.
- Reverify from the retained root with `sha256sum -c r18-real/retained_manifest.sha256`.
- The public PR carries compact logs, receipts, identity, probe, and manifests. Model weights,
  corpus-derived bulky outputs, and any future raw traces remain outside GitHub.
- The raw host file remains only in the retained R18 directory; the committed host receipt was
  restored to its prior public form to avoid adding fresh host-network details.

`compact_manifest.sha256` and `compact_manifest.check.txt` bind the public R18 evidence files and
the fresh driver-generated input receipts. `execution_receipt.json` supplies the machine-readable
stage and result status.

## Scope holds

DOWNLOAD=UNTOUCHED; SOURCE_MODIFICATION=NOT_RUN; NEW_AGENT=NOT_RUN; FINAL_TEST=NOT_RUN;
TRAINING=NOT_RUN; GGUF=NOT_RUN; GPU_DEPLOYMENT=NOT_RUN; GITHUB_ACTIONS=NOT_RUN; MERGE=NOT_RUN.
No completion, quality, production, or final-test claim is made.
