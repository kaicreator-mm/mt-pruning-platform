# P08 — Pinned SacreBLEU replay and post-R20 decision readiness
Exact R19 source 3212303e1e02994a5124db8c85195b91325c7a7c, prior parallel head bc635abe40f2a8c9c1863505558bdcb30273afee. R20-R1 unchanged. Existing validation outputs only; no inference, weight changes, training, GPU or final-test scoring.

## Executed evidence
Existing separate server aliyun-sg: dependency probe job job_01M8CK58JM001S1GPKDA6Q0JF8 exit0; isolated temporary pip --target directory installed sacrebleu==2.6.0 and numpy==2.5.5, no global installation, cleaned via trap; no model downloads. Actual replay job job_01M8CK6MNV00VCVYNSE19ATA00 exit0, py_compile PASS, Python3.12.3 (not identical to R19 Python3.13.18). Top-level metric and NumPy versions match original pinned versions; transitive deps not hash-pinned. All six exact source Git blob checks, actual corpus SHA256, exact120 validation IDs, 48 paired witness IDs equal baseline traced IDs, row/header keys and teacher/corpus/prompt/runtime/code/decoding/mode identity checked. Stage-specific protocol and candidate keys intentionally differ. No journal/trace payload replay or checkpoint-byte revalidation. Pinned evaluate.py blob ac122a7f46f0381cba5a2b019c42e391c670d968 AST extraction of eight named metric/flag functions and _DIGITS constant, actual function bodies unchanged; no torch import or full original evaluator run. Three focused sanity checks, corpus sum statistics consistency, 2000 paired resamples seed42 per comparison. Package signature nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0. Exact script and JSON retained; JSON blob ef2a06e7e5aef2704f4a87cd98f29098e50c46b0.

## Actual metric replay
All 30 P01 rounded chrF++ values matched pinned package within 0.0005 points (max 0.00047632055411384044); P01 score uncertainty removed, P01/P02 subset qualification limits remain. P08 intervals use project's NumPy RNG seed42, not P02 LCG; minor numerical CI difference expected. Same 8 paired witnesses per direction, not independent population estimates. No full candidate validation, no final decision.
|Direction|Baseline20 chrF++|Baseline20 chrF|Paired baseline8 chrF++|Physical96 8 chrF++|Delta chrF++|Delta chrF|95% paired percentile interval|
|---|---:|---:|---:|---:|---:|---:|---|
|zh-my|39.760|50.692|40.088|38.353|-1.735|-1.492|[-5.274, 0.805]|
|my-zh|21.122|24.642|18.910|17.626|-1.283|-1.497|[-2.609, 0.708]|
|zh-en|58.625|61.600|59.839|58.204|-1.635|-1.436|[-3.339, 0.240]|
|en-zh|32.122|37.476|29.542|31.422|1.880|2.193|[0.292, 3.797]|
|en-my|43.065|53.587|44.596|43.484|-1.113|-0.628|[-3.451, 0.608]|
|my-en|52.860|55.349|53.876|53.017|-0.859|-1.442|[-6.209, 3.776]|
chrF uses character ngrams only; chrF++ includes whitespace-based word ngrams. Myanmar output score differs appreciably; both report supplementary character metric, retain preregistered chrF++ gate. Five of six directions have negative point delta under both metrics, but 8-sample uncertainty means no candidate acceptance/rejection.

## Heuristic flag replay
|Direction|Baseline20 empty|Truncated|Digit mismatch|Degenerate repetition|Paired digit mismatch baseline->physical|
|---|---:|---:|---:|---:|---:|
|zh-my|0|0|3|20|0 -> 0|
|my-zh|0|0|1|0|0 -> 0|
|zh-en|0|0|3|0|0 -> 0|
|en-zh|0|0|1|0|1 -> 1|
|en-my|0|0|0|20|0 -> 1|
|my-en|0|0|1|0|1 -> 0|
Baseline120 empty0/truncated0/digit mismatch9. All40 Myanmar-target baseline outputs flagged degenerate_repetition, and all16 paired Myanmar-target outputs flagged in baseline, physical and masked. Rule tests >=16 tokens and unique token count < half length. Likely language/tokenizer sensitivity; not proof each is a false positive or actual repetition. Existing rule is diagnostic only, not one of three gates. Review examples/contiguous repeated ngrams and tokenizer fragmentation before using as a quality signal; do not change gate during R20-R1. Keep chrF and flags separate, no new human judgment claim. En-my paired digit mismatch0->1 in both physical and masked; preserve exact flagged IDs for later review.

## Finite post-R20 decision table (proposal; no dispatch)
First read exact R20-R1 receipt, actual candidate identities, full runtime-matched120 rows each, build/quant hashes, actual GPU residency and prefill/decode/end-to-end timings. Missing stage stays NOT_RUN. Keep BF16 mechanism diagnosis separate from GPU exploration and no inference qualification if physical correctness fails. Compare all six directions against same runtime/precision128 reference, not Q4 vs BF16 as pure pruning loss. Existing gate chrF++ delta>=-1 all directions and no increases empty/truncated/digit mismatch. Preserve old test HOLD.
- If k64 actual full paired quality meets gate and deployment measured fits: prioritize confirming k64 canonical mechanism and final-test eligibility; do not automatically run all alternative masks or smaller candidates.
- If k64 capacity fits but quality misses, k96 meets gate: try ONE uniform k64 robust or phase-balanced alternative on same runtime/precision, retain original and baseline for fair comparison. Only use P04 exact mask and original teacher for missing experts; no generic cache.
- If k64 still misses and k96 capacity fails, consider k80 or measured precision/offload tradeoff; if both k96 and k64 fail, diagnose runtime/quantization vs pruning loss before deeper compression. k112 is a possible fallback if needed; no claim monotonic quality.
- If GPU/convert/runtime fails, record exact blocker and available partial results, resolve existing execution path; no automatic full CPU rerun or speculative unrelated work.
- If quantization dominates loss, try a single higher precision or fixed tensor-type policy only if actual capacity supports; if pruning dominates, compare selection or restore experts before any training. Router-only recovery remains unproven (prior synthetic negative); no training on validation/test.
- Final test not authorized here; requires canonical candidate qualification and separate approved single final evaluation. No merges/production claims on proxy or subset results.
All prior P01-P07 research remains evidence, not product implementation. This closes the specific pinned scoring gap; further model experiments need R20-R1 results.
