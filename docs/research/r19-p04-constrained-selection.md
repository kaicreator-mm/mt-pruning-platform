# P04 — bounded calibration-only expert selection
R20-R1 remains unchanged. Exact source 3212303e1e02994a5124db8c85195b91325c7a7c; prior P03 45a711643a66151acc200a1bdc35a3e558d084a8. This is an offline research branch, no inference, new agent, GPU, weight conversion, final-test scoring or selection from validation outcomes.

## Method and executed evidence
Reused exact P03 input byte verification (4 SHA256, 2 Git blob pins), coverage checks and phase-balanced selector. In a temporary directory on existing separate server aliyun-sg, stdlib Python compile and execution passed, job job_01M8CJ330E009EPD3M5NGDXXER exit0; no dependency installation or model downloads, only public aggregate and supporting records, temp directory removed. Original96/64 reproduced. The first tool response delivered only 65536 of 146651 output characters, so initial JSON decoding failed on the coordinator side; recovered complete output via get_job max_bytes262144, parsed successfully, no experiment rerun. Four budgets (96,80,64,48), three methods, 12 plans. Each plan contains exact sorted unique expert IDs for all47 layers, with SHA256 of canonical JSON map. Saved executable script and output. No independent sample-level journal replay or actual model quality result.

- balanced: P03 equal normalized 12 direction/phase vectors.
- robust: initialize from balanced. At every layer constrain mean of 12 normalized retained shares to no worse than balanced minus 0.005 (0.5 percentage point); improve the minimum of 12 shares monotonically. At most12 exchanges, deterministic shortlist16 incoming/16 outgoing based on four current weakest groups, maximize min then mean, stable ID tie break. Checks floor and min invariant per layer. Local heuristic, not global optimum; no statistical generalization claim.
- myanmar2: sensitivity scenario only, multiply weights of all four Myanmar-involving directions (both source and target) by2, English/Chinese-only by1, each phase equally within direction. This is an illustrative assumption, not an owner-selected product priority. Do not alter fixed six-direction quality gate.

## Actual real calibration proxy
Average = mean across12 groups and47 layers. Worst group = min across12 group layer averages. Worst cell = min over564 cells. All are route mass, NOT chrF++.
|Experts|Balanced avg|Robust avg|Balanced weakest group|Robust weakest group|Balanced weakest cell|Robust weakest cell|Exchanges|
|---:|---:|---:|---:|---:|---:|---:|---:|
|96|93.91%|93.57%|92.53%|93.00%|88.67%|90.04%|158|
|80|87.87%|87.43%|84.94%|86.00%|79.51%|81.14%|152|
|64|79.79%|79.34%|74.08%|76.37%|66.47%|70.00%|152|
|48|69.49%|69.02%|58.46%|62.37%|47.26%|52.98%|144|

## Myanmar-weight sensitivity
Minimum of the four token-weighted Myanmar directions vs minimum of two non-Myanmar directions. Illustrative weights can improve Myanmar min while worsening overall weakest cell and some direction/phase retention. Does not establish improved translation.
|Experts|Balanced Myanmar min|Myanmar2 Myanmar min|Balanced non-Myanmar min|Myanmar2 non-Myanmar min|Myanmar2 weakest cell|
|---:|---:|---:|---:|---:|---:|
|96|92.96%|93.19%|93.89%|93.51%|87.61%|
|80|86.26%|86.70%|87.61%|87.01%|76.95%|
|64|77.49%|78.08%|78.96%|78.26%|65.38%|
|48|66.53%|67.25%|67.86%|67.05%|45.61%|

## Handoff
Recommend retaining original as reference, phase-balanced as high-mean alternative, robust as risk-protection alternative. Prioritize one matched-budget experiment after R20-R1 based on actual quality and VRAM: if64 viable capacity but misses quality, compare64 robust vs original with unchanged validation/precision/runtime; if96 needed, compare96 robust/phase-balanced;80 intermediate as needed. Not automatically run all12 candidates. The phase-balanced and robust maps generally differ by only a few percent of slots but require fresh source-derived physical prune because some IDs are outside original masks. Do not assume nesting across robust budgets; no padding or runtime change. No final-test use or canonical decision, no merge or extra agent. Actual model quality unknown; R19 earlier mechanism failure still open.
