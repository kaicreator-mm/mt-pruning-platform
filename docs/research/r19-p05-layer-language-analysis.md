# P05 — Contextual language routing and hypothetical nonuniform expert budgets
Exact R19 source 3212303e1e02994a5124db8c85195b91325c7a7c; prior P04 a9a5ceb862b0e25f4cdb8e5a94666127fdf04cb3. R20-R1 remains unchanged. Offline study only; no model inference, GPU, new agent, weight modification, test scoring or runtime implementation.

## Execution and evidence
Existing separate server aliyun-sg, six small public input files in temporary directory, SHA256 and Git blob pins checked with same P03 stdlib reader, exact144 calibration IDs, token/count coverage and finite routing mass checks. Python compile PASS, bounded actual job job_01M8CJCMAB00BKFH3D5ZP853GA exit0, no dependencies installed; temporary files removed. Script r19-p05-layer-language-analysis.py, result JSON blob 366b905d048d6e487407bed2d9ad040900da14a3. No individual journal re-aggregation or out-of-sample verification. The source text and target references were not used for routing decisions. Read input corpus only to match calibration IDs and prompt lengths. No test-set scoring.

## 1. Marginal routing is more language-dependent during generation
For each layer and phase, equal-weight normalized distributions for two relevant translation directions: source language in prefill, target language in decode. Each direction mixes prompt instructions, content and task context; decode also reflects generated sequence and lengths. This is contextual routing affinity, NOT evidence of experts with isolated linguistic functions. Three language pair comparisons per layer:
|Pair|Prefill mean top64 overlap|Decode mean top64 overlap|Prefill mean total variation distance|Decode mean total variation distance|
|---|---:|---:|---:|---:|
|zh-en|73.07%|66.76%|23.86%|48.24%|
|zh-my|70.88%|63.26%|36.31%|57.92%|
|en-my|68.88%|55.52%|36.61%|59.72%|
Top64 sets of all three languages intersect in mean36.55 experts/layer in prefill and28.45 in decode; unions mean92.34 and101.70. Not a proven universally shared core; top64 is a descriptive threshold. A deliberately descriptive high affinity criterion: dominant language >=2/3 of summed equally normalized language mass for the expert and >=1/128 within that language. Per 6016 layer/expert slots: prefill zh119/en138/my236 (493 total); decode zh404/en709/my472 (1585 total). Phase-specific counts may overlap across phases; no coactivation or causal saliency data. Decode TV higher for all three pairs, supporting phase-aware selection as a proxy hypothesis.

## 2. Redundancy and layer differences
6014/6016 (99.97%) distinct layer/expert slots selected at least once across all12 calibration groups; per-layer min126 max128. Rare use is NOT dead weight; a simple zero-frequency purge cannot substantially compress. Effective expert count (inverse Simpson concentration on equal group mix) spans31.79–88.11, not active top-k or number safely removable. Uniform balanced64 mean layer retention74.46–85.89%; weakest group cell per layer66.47–83.22%. Full 47 layer statistics in JSON. Lowest effective count layers: 17:31.79, 21:38.30, 27:39.29, 47:41.28, 16:41.54; highest: 3:88.11, 24:80.70, 8:79.11, 4:77.77, 15:75.97. Weakest balanced64 cells: 4:66.47%, 36:67.69%, 15:67.75%, 8:67.78%, 40:67.90%. These are calibration proxies, not measured layer ablation sensitivity.

## 3. Nonuniform allocation at identical physical slot count (hypothetical)
Keep 32..128 experts/layer in steps8, fixed per-layer phase-balanced nested ranking; total slots exactly47*k. max_mean greedily assigns chunks by diminishing marginal mean gain (optimal for this fixed prefix discrete average objective). max_min_then_mean finds highest feasible minimum cell threshold over prefix breakpoints, then spends residual budget by marginal mean gain (optimal floor within fixed-prefix choices, not over arbitrary masks). Uniform robust from P04 is an actual uniform candidate mask with different ranking, included as reference.
Each cell below shows mean / global worst-cell mass:
|Mean budget|Uniform balanced|Nonuniform max mean|Nonuniform max min then mean|Uniform robust P04|
|---:|---|---|---|---|
|96|93.91% / 88.67%|94.03% / 86.30%|93.96% / 89.95%|93.57% / 90.04%|
|80|87.87% / 79.51%|87.98% / 70.06%|87.90% / 81.83%|87.43% / 81.14%|
|64|79.79% / 66.47%|79.90% / 63.84%|79.78% / 69.96%|79.34% / 70.00%|
|48|69.49% / 47.26%|69.60% / 44.10%|69.31% / 53.86%|69.02% / 52.98%|
Nonuniform max-mean increases mean only about0.11pp at every budget, sometimes hurts minimum substantially (80:79.51%->70.06%). Nonuniform max-min improves uniform balanced floor, but k64 uniform robust70.00% versus heterogeneous69.96%; at k48 heterogeneous53.86% vs uniform robust52.98%, both proxy-only. At k64 max-min plan counts48x2 layers,56x10,64x21,72x14; exactly3008 slots, same estimated tensor bytes31730081536 if format truly stores different counts. Existing prune config and pinned llama.cpp HY-V3 require scalar physical expert count; padding to max width would defeat equal-slot budget. No actual heterogeneous checkpoint or runtime tested. No parameter count, latency or quality guarantee.

## 4. Decision recommendation
Do not start heterogeneous runtime work before uniform GPU comparison. Existing uniform robust and phase-balanced masks have similar or better floor proxy with much lower deployment complexity. Preserve P05 allocation counts for later research; prioritize matched uniform64 or96 alternatives contingent on R20-R1 actual quality and VRAM. Language-affinity statistics are not sufficient to remove experts for an unsupported language, transfer experts across layers, or implement dynamic per-language swapping. Need token-level coactivation and actual ablation/quality for causal conclusions. Avoid more speculative selector sweeps without real quality feedback. No new local task, merge or acceptance.
