# P07 — Direction-profile reuse and transfer stress
Exact R19 source 3212303e1e02994a5124db8c85195b91325c7a7c; P06 base 6d62573ae01570a0a724fafb5445a7e9dfc96810. No modification of active R20-R1, GPU or model inference. Offline calibration-proxy research, NOT validation quality.

## Executed evidence
Reused P03 byte checks (four input SHA256 and two Git blob hashes), original-teacher header, exact144 calibration IDs and token/count/mass coverage; replayed original96/64 masks exactly. Existing separate server aliyun-sg, temporary files only, no package install, no weights. First job job_01M8CJTFKX005TMACBWX7SBFB2 exited124 (90s timeout), caused by repeated normalization of the same vector in inner loops; no result accepted. Replaced repeated deterministic normalization with precomputed vectors, otherwise identical objective. Final job job_01M8CJY6KV002RZ1S79PY2H1BA exit0 and py_compile passed. Source and result JSON saved; result blob efe51ec6d7408413300895469edf9cd935fe10c9. 4 budgets (96,80,64,48) x9 omission scenarios (3 pair-only,6 single-direction omitted)=36 comparisons. Each mask generated using ONLY seen groups, equal normalized phase groups; omitted routing aggregates used only for descriptive proxy comparison. No new language, independent test sample or actual student inference; no per-sample journal replay. Original six-direction validation/test contract unchanged. These analyses use original-teacher marginal routing, not causal expert ablation.

## 1. Directly reuse a bilingual mask? No evidence of safety
Select experts only from zh-en and en-zh, compare on the other four Myanmar-related directions' original-teacher calibration mass against full six-direction phase-balanced selector, same uniform count.
|Experts|All-six selection: Myanmar mean|Zh/en-only selection: Myanmar mean|Difference (pp)|All-six weakest omitted cell|Zh/en-only weakest omitted cell|
|---:|---:|---:|---:|---:|---:|
|96|93.86%|83.99%|-9.87|88.81%|25.89%|
|80|87.82%|76.35%|-11.48|79.68%|19.18%|
|64|79.90%|66.40%|-13.49|67.69%|13.36%|
|48|70.09%|56.36%|-13.73|47.65%|9.56%|
Even at96, omitted Myanmar floor25.89% vs88.81%. At64, omitted mean66.40% vs79.90%, floor13.36% vs67.69%. Strong evidence against assuming existing pruned bilingual mask is suitable for new third language. It is a calibration proxy, NOT measured chrF++ or proof about arbitrary unseen languages. Similar comparisons for other two-language pairs are retained; all 12 pair-only cases reduce omitted mean vs full selector.

## 2. Leaving out one translation direction
|Experts|Largest omitted mean loss (pp)|Smallest omitted mean loss (pp)|Worst omitted direction|
|---:|---:|---:|---|
|96|-1.06|-0.63|omit_zh-en|
|80|-1.62|-0.98|omit_my-en|
|64|-2.79|-1.15|omit_my-en|
|48|-3.07|-1.68|omit_my-en|
All24 single-direction omission cases lose held-out mean vs full selector; do not infer from reverse direction alone. This is not a holdout-quality experiment. Need fresh original-teacher profiles for all four new directions in a new triple, not merely source/target language marginals.

## 3. Work reuse amount from actual R19
|Scope|Samples|Prefill tokens|Decode tokens|
|---|---:|---:|---:|
|All six|144|16412|12475|
|Reusable zh-en/en-zh (conditional)|48|3062|1276|
|Four existing Myanmar directions|96|13350|11199|
For same24/direction protocol, 48/144=33.33% sample-count reuse; in this observed triple the reusable portion is 18.66% prefill, 10.23% decode, 15.02% combined token counts. Not a wall-time measurement or prediction for a different language. New third language may have very different token lengths. Need 96 new calibration samples, 120 validation rows on complete new candidate; if original zh/en reference generation truly identical can reuse reference outcomes by exact IDs, never candidate results across masks. Final test stays disabled; newly chosen validation set must remain disjoint from all reused calibration and final-test material.

## 4. Reuse contract for future implementation
Existing demos/followup/r2/reuse.py blob b0015c2859a0c816b60af9e0681c6fb09d93b84f already composes direction profiles, requires same model/tokenizer/prompt/router/collector/protocol identity, original-teacher role, six directions, same layers/counts. Do not rewrite it now. R19 actual baseline header has teacher 0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94, prompt 47183c48e4c44e7cc1aad7c55b279ed93cb46afe6232224edbfe5457fe683c92, runtime 0cf5ce22c729e8e428cf912fc3756e779752394ea162fe2683fcf561ce400776, protocol 6c149fd82a952bf659a6d8d36e298b005cb4084843d1aab1d32211a75a501235, code 58f5d9de8e1599bbb38c3d5d955d6f730040f9ff41dde00e45ddf1476319ad73, decoding a23d9cce20c4a43cf6dc2c44f64ada724471f19365ebb019a0016acfd894efeb. Source header blob 8b2b9a3394c52973201115913374a8a8e1e29749. identity.py blob 0a2375314d599af492e6bffc8e24287150262f58 hashes whole PROMPT and TARGET_NAME dictionaries and tokenizer bytes: adding a language can change global prompt hash despite identical zh/en prompts. Do not overwrite existing run header or pretend exact global identities match. Future derived per-direction receipt must bind original run/aggregate SHA, exact calibration IDs and prompt-token sequence hashes, effective tokenizer files, original weight/router and relevant collector/runtime/protocol identity, phase/count/mass axes. Explicitly prove same effective prompt behavior before reuse, otherwise re-collect. Distinct new corpus/selection/candidate identity for new triple; preserve original teacher statistics, not pruned-model statistics. Existing P07 did not implement or test real cross-language cache composition. Need actual new language and independent corpus validation before claiming feature.

## 5. Close scope
No additional speculative selection sweeps until R20-R1 real results. Prior R5 router-only recovery synthetic negative evidence means training should not be assumed beneficial; if real quality gate fails, first compare uniform phase-balanced/robust at same budget, then consider bounded router recovery only after a separate training/validation split and resource evidence. Do not train on frozen validation/test. No current implementation/training, no new local task. R20-R1 unchanged, no merge/qualification.
