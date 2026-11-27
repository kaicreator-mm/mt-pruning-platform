# R23H — all8 layer-reversion ablation status

## Current gate

**Prefill screen and independent receipt replay: PASS. Primary fit propagation: prepared, not executed.**

This stage investigates whether a small number of layer-local expert-set reversions can repair the bidirectional primary-fit regressions observed in the earlier 96-expert candidate masks. Every candidate retains exactly 96 experts in each of the 47 MoE layers. These are fixed-mask diagnostics, not free-running translation-quality results.

## Why this branch exists

The prior all10 candidate completed 18 fixed fit trajectories, but independent replay rejected it: the zh-my / decode aggregate KL was 0.0113571688 versus 0.0112749403 for original96, and argmax mismatches were 2 versus 1, despite lower NMSE (0.04096836 versus 0.04440272).

The four remaining Pareto candidates were then tested over 10 primary fit trajectories. None qualified for secondary fit:
- all12 and all14: my-zh decode KL and NMSE regressions.
- primary4_10: my-zh decode KL regression.
- primary4_14: my-zh prefill and decode KL regressions.

The original all8 mask was the only earlier candidate close enough to justify layer-local refinement. Its changed layers relative to original96 were 10, 15, 40 and 42.

## Three one-factor reversion masks

| Candidate | Reverted layers from all8 | Mask SHA-256 | Git blob |
|---|---|---|---|
| all8_revert40 | 40 | 02de6ad74290659e3bd9b301ef606c45c129e032abcbbaaf867fe61bb26af84e | 255c95382a5888f9749d7626395182ba514c3c8d |
| all8_revert42 | 42 | 054614572884a518ce2a6538a0f39c7884667d1551312216df767ee4720f1f0b | f738ea900952ab19dbf6fd4cbca235dc40e56d05 |
| all8_revert40_42 | 40 and 42 | 96d26e3699ee87f2560fbbf15a53b6016b7d96aadc744f5aade4b076a90dab2b | 97809cb5092ee6b903a4def8ee0139902d24f17d |

## Actual V100-side prefill screen

Eight zh-my calibration fit prompts were evaluated. All candidates use the same frozen teacher, same prompts, and original96 / R23G / all8 / all10 controls.

| Mask | KL | NMSE | Argmax mismatches | Prefill gate |
|---|---:|---:|---:|---|
| original96 | 0.0798451807 | 0.0472553381 | 0 | Baseline |
| R23G | 0.1552965070 | 0.0447482708 | 0 | Comparison |
| all8 | 0.0607164133 | 0.0459805188 | 0 | Control |
| all10 | 0.0666124565 | 0.0409663549 | 0 | Control |
| all8_revert40 | 0.0621050783 | 0.0473691103 | 0 | Rejected |
| all8_revert42 | 0.0600118587 | 0.0461767721 | 0 | Eligible |
| all8_revert40_42 | 0.0615093730 | 0.0463108474 | 0 | Eligible |

The predeclared screen requires KL, NMSE and argmax mismatch not to exceed original96, plus KL lower than R23G. all8_revert40 fails the NMSE comparison. The other two qualify for the next primary-fit-only batch.

Screen plan SHA-256: 29e553c04e2cc61cbe999aac0604e9440f248480cadb6c8631f4758d87712acc

Screen report SHA-256: f35caa64d979ff6c781239a6cab25061af04235df77a282d5c6585a1704de521

Evidence archive SHA-256: ec3164d575309de811c75759539ae661e1d58c85daf00189a7a26bf3e1557b1b

Independent 8845HS receipt/aggregation replay output SHA-256: 6c405ee0be368b8db584af201e0f7bf71b018cb4c690de4f0d12888736ea64cc

Independent replay status: R23H_ALL8_ABLATION_PREFILL_INDEPENDENT_RECEIPT_REPLAY_PASS.

## Next bounded work

1. Run both eligible masks over the 10 primary fit trajectories: 8 zh-my and 2 my-zh. Capture prefill plus eight actual propagated decode steps under the same fixed teacher-forced prefixes.
2. Pack the complete evidence, including the three-mask screen and prior all10 receipts; independently replay every receipt and aggregate before proceeding.
3. If either candidate qualifies, evaluate every eligible candidate over the remaining 8 secondary fit rows (zh-en, en-zh, en-my and my-en; two each). This avoids choosing between two masks from only one side of the primary gate.
4. Independently replay the complete 18-row fixed-fit evidence. Only then consider a separately gated free-running translation-quality evaluation.

No Fresh, holdout, native free-running translation, or physical model materialization is authorized by the current prefill result.

## Execution state

The primary batch runner, setup script, evidence packer and independent replay script are committed on parallel-r19-offline-analysis. The runner acquires the exclusive experiment lock before creating output, verifies the old all10 failure and exact screen receipts, and checks resource and teacher identity before writing its admission record.

The pinned execution engine is intentionally CPU-only: mtpruning/engine.py is the lean CPU mmap engine for the approximately 60 GB HF-layout checkpoint, and requirements.txt pins torch==2.10.0+cpu. CUDA_VISIBLE_DEVICES empty and zero V100 utilization are expected for this stage; the 16 GB V100 is not used by this engine. Host RAM preflight is mandatory.

The first v1 run produced one of ten fit rows for both candidates and was then manually interrupted after a mistaken GPU requirement. Its admission, run log and two candidate receipts are preserved in r23h-all8-ablation-primary-cpu-abort.json as a valid partial CPU diagnostic, not a completed batch. The isolated v2 run recomputes all ten rows under the same pinned CPU environment so the final aggregate comes from one coherent run.

The v2 runner is pinned to the CPU engine and writes to a separate output directory. It retains the same strict receipt validation, fit-only scope, and no-Fresh/no-holdout limits.
