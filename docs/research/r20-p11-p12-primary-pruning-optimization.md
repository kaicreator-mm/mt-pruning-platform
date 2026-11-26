# P11/P12 — Primary Chinese-Burmese 96-expert guarded pruning
Owner direction: keep Chinese/English/Burmese only; prioritize Chinese↔Burmese, aim for 96 or fewer experts without large quality loss. Existing six-direction validation gates unchanged. Research base 6ade0526c5b7d1eb48b9dbebbcac3f904ac8b253. R20-R1 active, no modifications to PR #4 or its working directory; no training, new inference or final-test evaluation.

## Actual P11 strict guard (negative result)
Executed on separate aliyun-sg, verified original R19 aggregate and P04 mask hashes, replayed original96 exactly; job job_01M8CQXDPP00GYK1765P5PJXQH exit0, compile PASS. Per-layer primary four cells not allowed to decrease, secondary per-cell loss at most0.5pp, per-layer overall mean not allowed to decline. Bounded exhaustive single-exchange neighborhood, max12 accepted exchanges/layer. Only5 expert slots changed, four layers improved primary minimum; primary minimum group 93.7841%→93.8450%, overall mean93.6195%→93.6232%. Too small to prioritize physical candidate. Retain source/result, do not dispatch P11 generation.

## Actual P12 globally guarded96
Separate aliyun-sg temporary target dependency install NumPy2.5.5/SciPy1.16.3; standard library verification of frozen R19 stats/corpus/records, exact P04 mask file Git blob and each canonical mask SHA, original96 replay. Job job_01M8CR0K18008CAQM0ZN7NCHAQ exit0, compile PASS; solver HiGHS status Optimal within configured relative MIP gap, actual gap 0.00004938952544659422. 258 feasible distinct layer options from original/P04 balanced/robust/myanmar2 and primary-weighted1.5/2/3 options. One 96-expert set chosen per layer; fixed objective sum of layer-local weakest primary retention plus0.25 primary mean. All original global primary phase means and both token-weighted primary direction masses protected; secondary phase means allowed to decline at most0.5pp; global all-group mean and weakest primary/all-group cells not below original. Independent post-solve recomputation checks passed. This is a finite option inventory optimum, not all possible masks; the 0.5pp proxy slack and objective are research choices, not new quality gates. No validation outcomes used to choose individual experts. 47 layers ×96 sorted original IDs; 59 replaced positions out of4512.

|Calibration proxy|Original96|Global guarded96|Change(pp)|
|---|---:|---:|---:|
|All groups mean|93.6195%|93.6281%|0.0086|
|Primary four groups mean|94.3291%|94.5099%|0.1808|
|Weakest primary group mean|93.7841%|94.3349%|0.5508|
|Weakest primary layer/group cell|87.7428%|90.0257%|2.2829|
|Mean per-layer primary minimum|92.9962%|93.4191%|0.4228|
|All groups weakest cell|84.6988%|84.6988%|0.0000|

|Direction token-weighted mass|Original96|Global guarded96|Change(pp)|
|---|---:|---:|---:|
|zh-my|94.6051%|94.6161%|0.0110|
|my-zh|94.2819%|94.3473%|0.0654|
|zh-en|93.1132%|93.0768%|-0.0364|
|en-zh|93.9623%|93.8365%|-0.1257|
|en-my|94.7656%|94.6235%|-0.1421|
|my-en|93.4896%|93.3053%|-0.1843|

Candidate mask blob 150616a10fbf503f815fb2f7fc281a81dcc28fea; canonical mask SHA256 4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c; results blob 8d6114bd050154c2015f9072e4430a5c5b1d05b2. Retain exact candidate and script. These modest proxy gains do not establish actual chrF++ recovery; global mean nearly unchanged, weakest all-group cell unchanged, secondary directions may lose some mass. Do not claim model quality PASS. P12 is the proposed single next 96-expert real witness candidate, not a production selection. Compare to original96 and runtime-matched128; if insufficient, investigate output-sensitive expert saliency, local output reconstruction/teacher-guided expert substitution using disjoint calibration data before lower budgets. No training by default. Do not select experts by fitting the frozen validation translations.

## Resource and execution boundary
Live read-only check: V100 is actively running full-f16-k128-v2 (R20 evidence repair), GPU PID397709, ~12982MiB occupied, free disk 53744791552 bytes. Previous three FP16 output sets120/120 each, enriched receipts present, v2 baseline still active. Existing 96 HF intermediate + another FP16 GGUF would need roughly92GB before overhead, exceeds actual free bytes. Do not generate either new large artifact, delete retained R20 files, or use GPU while writer is active. Once R20 completed and disk/GPU checked, use original teacher and exact P12 mask with existing physical pruner and pinned conversion path, serial materialization; reserve disk, retain verified evidence, explicitly resolve stale artifact retention or direct one-pass GGUF slicing if needed. Direct GGUF slice is not implemented/qualified. Existing artifact read-only inventory confirms router and bias plus three expert FFN tensors per sparse layer require same ordered original IDs. No new agent or auto-run queued. Use Gateway reverse tunnel on completion. First real test should be one bounded paired 96 candidate, prioritize both primary directions; full six-direction validation before gate, final test remains HOLD. 
