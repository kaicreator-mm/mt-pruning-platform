# P02 — R19 paired quality uncertainty and calibration design
Source exact 3212303e1e02994a5124db8c85195b91325c7a7c; P01 report remains unchanged. Actual existing validation text only; no model inference, GPU, test-set scoring or gate change. R20-R1 remains running and untouched.

## Paired uncertainty on 48 physical k96 witnesses
Recomputed P01 single-reference chrF++ algorithm in JS based on pinned sacrebleu v2.6.0 source (char1–6 word1–2, beta2, effective order). Same exact inputs as P01:
- evidence/d01b-r19/corpus/corpus.jsonl (5fc8e3586a486e89ca6e0e6644d6c79ad1e4de72)
- evidence/d01b-r19/corpus/corpus_manifest.json (05322b3a96e06d2a7676dcc39b6efccd15093bfb)
- evidence/d01b-r19/baseline/records.jsonl (8b2b9a3394c52973201115913374a8a8e1e29749)
- evidence/d01b-r19/k96/mechanism/records.jsonl (7593b21a0395650fca1094d2b2900a8f6a4b94c4)
- evidence/d01b-r19/k96/masked/records.jsonl (1423d039695659dea7d4ea074907bb258d7a825c)
Verified exact 120 validation IDs, matched 48 physical/masked IDs, run headers and context bindings. Three sanity checks (identity/disjoint metric and identical paired bootstrap). No installed pinned Python sacrebleu; scores require package cross-check, not official report. For each direction, resample 8 paired sentences with replacement 2000 times using deterministic LCG initial seed42 (state=1664525*state+1013904223 modulo 2^32); recompute corpus score from summed ngram counts per resample, linear percentile 2.5/97.5. Leave-one-out (LOO) recomputes corpus delta on 7 pairs. Fractions below -1 are empirical resample fractions, NOT posterior probabilities. Intervals conditional on small selected witness subset, not full validation or population CIs; no multiple-comparison inference. No cross-direction pooling due aligned source-line dependence.

| Direction | Physical96 minus baseline | 95% paired percentile interval | Resample fraction delta<-1 | LOO delta range | Physical minus masked | Paired interval |
|---|---:|---|---:|---|---:|---|
|zh-my|-1.735|[-5.350, 0.780]|67.5%|[-2.527, -0.749]|1.170|[0.041, 2.826]|
|my-zh|-1.283|[-2.548, 0.686]|63.6%|[-1.905, -0.639]|0.020|[-0.022, 0.097]|
|zh-en|-1.635|[-3.382, 0.178]|67.3%|[-1.980, -0.483]|0.014|[0.000, 0.040]|
|en-zh|1.880|[0.154, 3.724]|0.0%|[1.251, 2.448]|0.292|[-0.116, 0.771]|
|en-my|-1.113|[-3.482, 0.635]|52.2%|[-1.708, -0.185]|-0.263|[-1.897, 0.852]|
|my-en|-0.859|[-6.149, 3.809]|47.0%|[-2.403, 0.696]|1.107|[-0.172, 3.395]|

Original masking equivalence FAIL remains; quality similarity cannot establish numerical equivalence. Inspect whether threshold is sensitive to one sample, rather than making an acceptance decision on eight. Full120 per model and pinned metric needed.

## Calibration sampling structure
From frozen corpus IDs only (calibration/validation): {"calibration":{"rows":144,"uniqueParallelLines":69,"rowsPerLine":{"2":66,"4":3}},"validation":{"rows":120,"uniqueParallelLines":55,"rowsPerLine":{"2":50,"4":5}}}. Shared aligned source lines mean directional rows are not independent across directions. Per-direction scores stay separate; later aggregate/bootstrap must cluster by FLORES parallel line, not independently resample all directional rows. Do not expose/use test references for selection.

## P03 ready-to-run selection comparison, no new algorithm framework
Existing main helpers:
- `demos/followup/r6/bridge.py` (f3329d7659a49820201522f9cad16024a86bf4f0) validates real aggregate coverage/counts; `prepare_candidates.py` (14d507a39e31d5064b809776bf2299a9b9f3c36d) uses `demos/followup/r1/prep.py` plan balanced/worst_first/random. R19 original selection normalizes sum(prefill,decode) within each direction; alternative balanced normalizes each of twelve direction/phase groups. Worst-first greedy is heuristic, not global optimum.
- Input exact R19 baseline/stats.npz SHA256 8190126a4e69943cd7bceb5f136b8ca9b9201c06bda005b2d24ffd9b229a4169, corpus SHA256 9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4, records 37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8, stats.json aa1092914aaa4310397a9e87e9dd3d4897bba9e2cda7001481487fd7d5912c72. Verify hashes first. UTF8 GitHub connector fetch, fetch_blob and contents endpoint all rejected binary; direct raw curl DNS failed. Real P03 mass reranking NOT_RUN. No need to interrupt ongoing R20-R1 or request new agent.
- Once bytes accessible, run existing helper on exact retained input files, 47 layer IDs1..47, experts128, top_k8, per-direction24, keeps96,80,64,48,32, methods balanced/worst_first/random(seed42). Also replay original mass selector for all keeps and require96/64 ID match to R19. Compare normalized mean/min per direction-phase/layer, aggregate route mass, retained count fraction, mask overlap and fixed tensor size; only use calibration to generate choices. Include random anchors and phase-vs-direction imbalance. Sensitivity based on actual quality requires separate validation; coactivation and expert output norms not present in aggregate (cannot infer from marginal count/mass). Predeclare one best alternative per size based on calibration proxy before full validation, not tune from selected witness results. No decision or model mutation until later validation.
- Example invocation from repo root (read-only input paths; output outside current run):
```bash
python demos/followup/r6/prepare_candidates.py \
  --corpus "$CORPUS" --records "$RECORDS" --stats "$STATS_BASE" \
  --corpus-sha256 9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4 \
  --layer-ids "$(seq -s, 1 47)" --experts 128 --top-k 8 \
  --per-direction 24 --keeps 96,80,64,48,32 --method balanced \
  --out "$NEW_OUTPUT/phase-balanced.json"
```
Repeat worst_first and random to distinct outputs. This command is a prepared future invocation, not an executed real result. Do not run on the active local Agent while R20-R1 owns resources.

## Status
P02 actual resampling complete; P03 real aggregate input blocked. No code change, no canonical acceptance, no merge, no new agent.
