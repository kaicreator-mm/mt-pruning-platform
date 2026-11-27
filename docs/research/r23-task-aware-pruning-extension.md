# R23 task-aware expert pruning extension

Status: PLANNED_CONDITIONAL, adopted as coordinator research plan on owner continuation. Existing R21–R27 scope, six-direction per-direction quality thresholds, and fixed validation/test isolation remain unchanged. Current R23A stopped fail-closed after 10/24 receipts; resolve the historical greedy-prefix mismatch before selection. No new candidate, training, or validation-based fitting is authorized by this plan alone.

## Task DAG

| Task | Dependencies | Bounded deliverable / decision |
|---|---|---|
| R23A | R22 fail | Finish frozen24 fit/holdout capture; verify all receipts; choose at most one candidate from existing five-mask layer inventory, or reject. |
| R23B | R23A protocol repair; may prepare independently | Capture standard REAP, frequency, routing weight, activation norm and weighted-output contributions under exact same teacher/token/phase identities; compare with existing local reconstruction proxy on calibration only. |
| R23C | R23B capture | Task-oriented functional profiles and pairwise substitutability/interaction: bounded shortlists from fit; single/paired deletion and top-k replacement error, held-out calibration rejection. |
| R23D | R23B/C | Compute layer loss-budget curves for K64/80/96/112/128, compare uniform96 against fixed total budget4512 and minimum K>=8. Nonuniform mask-only study first; no physical runtime changes until clear benefit. |
| R23E | R23C or D candidate | Measure actual masked-model propagation, teacher-forced common prefixes and separate free-running quality on calibration; compare local error with downstream hidden-state/logit drift and routing changes. |
| R23F | R23A–E evidence as applicable | Select at most one new96-equivalent candidate on fit; independent held-out calibration admission; actual GGUF materialization only for supported uniform budget or separately qualified representation. Frozen120 native quality validation, unchanged six direction gates. |
| R24/R25 | Qualified candidate | Matched Q4 deployment and bounded smaller budget follow existing plan. |
| R26/R27 | Conditional | Recovery or final product acceptance only under existing plan. |

Avoid exhaustive sweeps: R23A may qualify a uniform96 candidate directly; additional branches only run when quality fails or enable justified further compression. Stop redundant mask trials. Never optimize against validation scores. Repeated use of existing validation set is development evaluation, not an untouched final test.

## R23B metric contract

Use exact per-token routed expert outputs f_e(x), effective normalized route weights g_e(x), teacher top-k membership. Record count, sum of g, sum of L2 norm, sum of g*L2 norm, and sum of squared weighted norm separately. Define standard REAP score as conditional mean weighted activation norm over tokens routed to the expert; distinguish from global mean, count-weighted total, and current squared-energy measure. Zero-count experts are unobserved, not proven unimportant. Bind source revision and actual implementation before claiming published-method reproduction. Existing R23 energy vector alone cannot recover the per-token REAP mean (norm of a sum or sqrt of summed squared norms is not the same statistic). A new observation hook is required. Test exact synthetic metric oracles and real observer noninterference, including token-wise weights, top-k, shared experts, and tie handling.

Profile dimensions: layer/expert ID, source-target direction, prefill/decode, token counts, route stats, output metrics, local reconstruction impact, uncertainty/support, data/protocol identity. Functional labels include numeral/date/unit preservation, named entity/transliteration, negation, word order/grammar, ambiguity/context, register, and repetition/termination; labels are hypotheses supported by targeted tests, not a claim of exclusive semantic storage. Use calibration examples and separately frozen diagnostic contrast sets; track coverage and cross-language overlap. Do not infer expert specialization from frequency alone.

## R23C interaction and replacement

Same hidden states and actual route normalization, calculate loss L(S) for retained set S; use interaction residual L(S without a,b)-L(S without a)-L(S without b)+L(S). Compare same-budget swaps, expert output similarity on common input, and actual replacement cost. Output cosine alone is not substitutability. Select bounded pairs by fit-only route/energy/functional profiles, record support and uncertainty, do not run all 8128 pairs per layer. Reuse R23 sparse-union optimization where candidate inventory fixed, with independent exhaustive witness. A new undeclared mask requires new observations or sufficient retained outputs; never silently use uncomputed experts.

## R23D/E limits

Per-layer budget study uses actual estimated bytes including fixed/shared tensors, not merely average K. Current GGUF header and engine uniform expert inventory cannot physically realize heterogeneous K_l. A mask-only variable budget is not a storage saving. First demonstrate benefit, then consider separately scoped model format/runtime work. Sequential propagation uses a common frozen token trajectory to isolate state differences and reports free-running translation separately; do not conflate forced prefixes with greedy reproductions. Preserve same engine and numerical configuration across candidates. Distinguish local error proxy from actual end-to-end effects.

## Optional LLM use

Owner allows configured dh-deepseek-api variable on aliyun-sg and flash model. Only use when task-relevant (e.g. proposed calibration diagnostic labels, blinded translation semantic adequacy review); first verify configured endpoint and advertised exact flash model without printing credentials. Use public/synthetic or approved calibration text, no secrets, no private data, no held-out test fitting. Record exact model identifier, prompt schema, input/output digests, model settings, failure and costs if available. Validate structured response and manually/deterministically crosscheck a bounded subset. LLM labels are advisory evidence, never the deterministic numerical gate, teacher-output calculation, or substitute for bilingual human evaluation. No API call needed to repair current numerical prefix issue. No credential access in this preparation.

## Current failure

R23 batch stopped at frozen calibration:my-en:163 before step7 due to argmax disagreement, 10/24 immutable records retained. Original reference16 CPU threads, R23 current8. Diagnostic job job_01M8E3ZB2G00HAW2QFBG6FSQ23 checks same frozen prefix and teacher with 8/16/8 threads. Do not declare root cause before actual outputs. No skip, replacement, silent weakening of oracle or holdout fitting.
