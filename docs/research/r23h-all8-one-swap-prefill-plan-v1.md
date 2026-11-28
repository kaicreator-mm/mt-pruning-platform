# R23H — all8 one-expert local swap screen v1

## Objective

The frozen 96-expert-per-layer `all8` mask passed the bidirectional 16-row Prefill screen, but its 18-row primary Fit propagation was rejected solely on `zh-my/decode/nmse` (0.045155772986588086 versus 0.04440272289099043 for `original96`). The purpose of this bounded screen is to test whether one local expert replacement in one of the four changed layers can preserve the useful Prefill gains while improving the parent mask.

This is a diagnostic search, not a translation-quality or product-admission run.

## Frozen source masks

- `original96`: mask SHA-256 `23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7`.
- `all8`: mask SHA-256 `1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b`.
- `R23G`: mask SHA-256 `77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa`.

The `all8` mask is reconstructed from the independently pinned `all8_revert10.json` and `all8_revert15.json` sources, then checked against its exact mask hash. It differs from `original96` only at layers 10, 15, 40 and 42.

| Layer | Removed from original96 | Added by all8 |
|---:|---|---|
| 10 | 42, 95 | 78, 100 |
| 15 | 6, 64 | 4, 71 |
| 40 | 13, 84 | 58, 62 |
| 42 | 47, 120 | 63, 126 |

## Candidate family

For each changed layer, generate the four one-for-one swaps that remove one of the two `all8` additions and restore one of the two original experts. This produces 16 candidates total. Every candidate must retain exactly 96 experts in every one of the 47 MoE layers and differ from `all8` in exactly one layer and one expert replacement.

No other layer or tensor is changed. No top-k, router, tokenizer, shared expert, training, or model-weight materialization is involved.

## Actual Prefill screen

- 16 frozen Fit prompts: 8 `zh-my` and 8 `my-zh`; Holdout is forbidden.
- Remeasure `original96`, `all8`, and `R23G` in the same run.
- Reuse the previous factorial run only for exact teacher-logit identity and historical `original96`/`R23G` baseline replay.
- Pin the exact teacher identity and source code; acquire the exclusive batch lock; fail closed on resource blockers.
- CPU mmap engine, 8 threads. CUDA/GPU use is intentionally zero for this oracle stage.

A candidate is eligible for a separately gated primary Fit propagation only if, in both directions, its Prefill KL, NMSE and argmax-mismatch count are no worse than both `original96` and `all8`, and at least one metric strictly improves over `all8`. If no candidate passes, stop this candidate family without Decode or any further admission.

## Explicit exclusions

No Decode, Fresh, Holdout, free-running translation, chrF++ claim, product/production admission, training, physical pruning, model publication, or deletion of existing evidence. An eligible Prefill candidate is only a candidate for the next fixed-Fit propagation gate.

## Evidence path

- Runner: `r23h-all8-one-swap-prefill-v1.py`
- Runtime output: `~/.cache/mt-pruning/r23-results/r23h-all8-one-swap-prefill-v1/`
- Independent replay host: Windows 11 Pro / Ryzen 7 8845HS via Gateway port 22024.
- The independent replay must verify the archive, exact source and teacher pins, all 16 prompt identities, all 16 receipt hashes, baseline replays, reconstructed 47×96 masks, candidate swaps, aggregates and the bidirectional gate before any primary Fit propagation.
