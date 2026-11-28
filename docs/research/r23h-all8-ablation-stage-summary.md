# R23H — all8 layer-reversion ablation status

## Current gate

**Primary fit propagation: 10/10 complete. Independent receipt replay: PASS. No candidate qualifies for secondary fit.**

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

## CPU mmap prefill screen

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

## Primary fit propagation — v3

Both prefill-eligible masks were evaluated over the ten frozen primary fit rows (8 zh-my and 2 my-zh), with prefill plus eight propagated decode steps per prompt. The run used the pinned CPU mmap engine; no Fresh, holdout, or free-running generation was used.

| Candidate | Direction / phase | KL | NMSE | Argmax mismatches |
|---|---|---:|---:|---:|
| all8_revert42 | zh-my / prefill | 0.0600118587 | 0.0461767721 | 0 |
| all8_revert42 | zh-my / decode | 0.0110190720 | 0.0451547260 | 1 |
| all8_revert42 | my-zh / prefill | 0.3340542402 | 0.0806651258 | 0 |
| all8_revert42 | my-zh / decode | 0.0460633108 | 0.0396257974 | 1 |
| all8_revert40_42 | zh-my / prefill | 0.0615093730 | 0.0463108474 | 0 |
| all8_revert40_42 | zh-my / decode | 0.0114029998 | 0.0452438280 | 1 |
| all8_revert40_42 | my-zh / prefill | 0.3383674613 | 0.0809816723 | 0 |
| all8_revert40_42 | my-zh / decode | 0.0472938921 | 0.0398306613 | 1 |
| original96 | zh-my / prefill | 0.0798451807 | 0.0472553381 | 0 |
| original96 | zh-my / decode | 0.0112749403 | 0.0444027229 | 1 |
| original96 | my-zh / prefill | 0.3809461720 | 0.1440526690 | 0 |
| original96 | my-zh / decode | 0.0394477213 | 0.0392434584 | 1 |

**Primary-fit decision: neither candidate is eligible for secondary fit.**

- all8_revert42 regressions: zh-my/decode NMSE; my-zh/decode KL and NMSE.
- all8_revert40_42 regressions: zh-my/decode KL and NMSE; my-zh/decode KL and NMSE.

Batch admission SHA-256: d3793042e04e4b7ec806eff56e72346d8359212f13b59e04568a0bede695c9fb

Batch report SHA-256: 8ab5c2fc607ad49a589636df39ab46f801327d7c39cbbf70ce29b0fb2b23d2af

99-file evidence archive SHA-256: 5154ce5979b93ec849c25a688bf4a99180643f40a95f78f3dfe4fdd7f729a752

Independent Windows replay status: R23H_ALL8_ABLATION_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS

Independent replay output SHA-256: 89ab4d14f73563ee6e1a05cdeeb2143cf67c277a43c78629c2468b310514c552

The independent replay confirms every receipt, mask, teacher identity, fixed-prefix trajectory, group aggregate and terminal report. The replay JSON is preserved in r23h-all8-ablation-primary-replay.json.

### Layer attribution — exploratory only

- all8_revert42 differs from original96 at layers 10, 15 and 40.
- all8_revert40_42 differs from original96 only at layers 10 and 15.
- Reverting layers 40 and 42 does not remove the bidirectional decode regressions. The per-layer output-NMSE deltas show propagated differences in many later layers (roughly 20–44), even where those layers' masks are unchanged. This is a downstream-state effect, not evidence that those later layers' expert masks are themselves wrong.
- The layer attribution is correlational and must not be treated as a causal ranking or as product-quality evidence.

## Next bounded work

The next hypothesis is that the remaining layer-10 and layer-15 mask changes are driving downstream decode drift. Test a small factorial of layer-local reversions before another full primary batch:
1. all8_revert10
2. all8_revert15
3. all8_revert10_40_42 (only layer 15 remains changed)
4. all8_revert15_40_42 (only layer 10 remains changed)

First run a prefill-only fit screen against the same eight zh-my rows and original96/R23G controls. Only masks passing that bounded screen may enter a fresh ten-row primary fit batch. Keep all decisions fit-only; no Fresh or holdout consumption.

## Execution state

The primary batch runner, setup script, evidence packer and independent replay script are committed on parallel-r19-offline-analysis. The runner acquires the exclusive experiment lock before creating output, verifies the old all10 failure and exact screen receipts, and checks resource and teacher identity before writing its admission record.

The pinned execution engine is intentionally CPU-only: mtpruning/engine.py is the lean CPU mmap engine for the approximately 60 GB HF-layout checkpoint, and requirements.txt pins torch==2.10.0+cpu. CUDA_VISIBLE_DEVICES empty and zero V100 utilization are expected for this stage; the 16 GB V100 is not used by this engine. Host RAM preflight is mandatory.

The first v1 run produced one of ten fit rows for both candidates and was then manually interrupted after a mistaken GPU requirement. Its admission, run log and two candidate receipts are preserved in r23h-all8-ablation-primary-cpu-abort.json as a valid partial CPU diagnostic, not a completed batch. The isolated v3 run recomputes all ten rows under the same pinned CPU environment so the final aggregate comes from one coherent run.

The v3 runner is pinned to the CPU engine and writes to a separate output directory. It retains the same strict receipt validation, fit-only scope, and no-Fresh/no-holdout limits.
