# R19 parallel offline evidence analysis (P01)
Status: EXPLORATORY_ONLY. Source head 3212303e1e02994a5124db8c85195b91325c7a7c. Does not modify PR #4, R20-R1 or runtime. No new inference, GPU, model download, or final-test scoring. All quality calculations below use validation rows only. Test references were not extracted or evaluated. Canonical mechanism/decision HOLD.

## 1. Real-output quality analysis
Read exact Git blobs:
- evidence/d01b-r19/corpus/corpus.jsonl: 5fc8e3586a486e89ca6e0e6644d6c79ad1e4de72
- evidence/d01b-r19/corpus/corpus_manifest.json: 05322b3a96e06d2a7676dcc39b6efccd15093bfb
- evidence/d01b-r19/baseline/records.jsonl: 8b2b9a3394c52973201115913374a8a8e1e29749
- evidence/d01b-r19/k96/mechanism/records.jsonl: 7593b21a0395650fca1094d2b2900a8f6a4b94c4
- evidence/d01b-r19/k96/masked/records.jsonl: 1423d039695659dea7d4ea074907bb258d7a825c
- evidence/d01b-r19/selection.json: d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed
- evidence/d01b-r19/baseline/stats.json: e364a572e0773a1b435d30873f3cbfbd39e1bf8c
Source corpus identity 9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4; exact validation manifest IDs, record uniqueness, context/direction/line/prompt length, real BF16 greedy header, generation length checked. 120 baseline validation rows (20 per direction); both physical and masked k96 sets have same 48 validation IDs (8 per direction). No inference on held-out test split. Actual scores below computed in JavaScript in connector execution from exact retrieved texts, independently implementing single-reference corpus chrF++ (char orders 1–6, word orders 1–2, beta2, mixed case, no character whitespace, effective order), based on sacrebleu v2.6.0 chrf.py blob f7d4f6858d0c6005f97ad8011a0b17bd97c2bcea and helpers.py blob 72ec14461658249fcd63a139623f3ead9a4aa057. Six focused checks passed (identical/disjoint/partial, punctuation precedence, Unicode, digit folding). This is not an execution of pinned Python sacrebleu and has not been cross-checked against that package. R20-R1 retains authority for pinned scoring and final quality. No claim of validated translation quality from identity or subset. Flags = empty/truncated/digit-run mismatch, heuristic not semantic error. Use paired baseline8, not baseline20, to compare subset. No bootstrap or human review here.

|Direction|Baseline20 chrF++|Baseline20 flags|Paired baseline8|Physical96 (8)|Delta|Masked96 (8)|Paired flags baseline→physical|Physical vs baseline exact text|
|---|---:|---|---:|---:|---:|---:|---|---:|
|zh-my|39.760|0/0/3|40.088|38.353|-1.735|37.183|0/0/0 → 0/0/0|0/8|
|my-zh|21.122|0/0/1|18.910|17.626|-1.283|17.606|0/0/0 → 0/0/0|2/8|
|zh-en|58.625|0/0/3|59.839|58.204|-1.635|58.190|0/0/0 → 0/0/0|2/8|
|en-zh|32.122|0/0/1|29.542|31.422|1.880|31.130|0/0/1 → 0/0/1|3/8|
|en-my|43.065|0/0/0|44.596|43.484|-1.113|43.747|0/0/0 → 0/0/1|4/8|
|my-en|52.860|0/0/1|53.876|53.017|-0.859|51.910|0/0/1 → 0/0/0|0/8|

The small subset is descriptive only, not candidate acceptance/rejection, and failed masked equivalence stays FAIL. Do not select a model or alter the <=1-point gate from these data. Paired 8-per-direction samples are selected witnesses, not independent test observations. Baseline full evaluation and paired subset should be cross-checked by R20-R1 pinned sacrebleu, no new generation needed.

## 2. Selection structure and route coverage
All 47 layers have unique, valid 96/64 selected IDs; k64 is a strict subset of k96 in every layer (47/47). 1504 slots removed each step. Each layer has its own expert identities; equal local expert IDs in different layers are not shared parameters.
|Direction|Prefill tokens|Decode tokens|Decode share|96 worst layer retained mass|64 worst layer retained mass|
|---|---:|---:|---:|---:|---:|
|zh-my|1471|5145|77.8%|91.88%|76.65%|
|my-zh|5108|543|9.6%|91.64%|77.37%|
|zh-en|1447|665|31.5%|89.68%|70.81%|
|en-zh|1615|611|27.4%|88.71%|70.78%|
|en-my|1663|4902|74.7%|92.01%|76.33%|
|my-en|5108|609|10.7%|89.90%|70.27%|
The equal-direction sum-of-prefill/decode objective still weights phases by token mass inside each direction; Burmese generation differs sharply from Burmese input. Future offline compare separately normalized prefill/decode, minimum-direction constraint, and existing average rule on the SAME calibration aggregate. Do not promote retained mass to accuracy. The binary NPZ (blob 8b29293b98e8f1cf21cb7f907b3a529d2b0ab418, 509133 bytes, recorded SHA256 8190126a4e69943cd7bceb5f136b8ca9b9201c06bda005b2d24ffd9b229a4169) could not be retrieved as bytes by the available UTF8 GitHub connector (decode failure), direct container fetch DNS failed. Thus no measured 80/48/32 mass, layer sensitivity, phase re-ranking, or new masks here. Use existing demos/followup/r3/search.py and reviewed calibration selector later; don't create another framework.

## 3. Size forecast anchored in measured bytes
R19 measured 128 tensor bytes 60129463808, 96 tensor bytes 45929772672. Removed 32*47 slots; inferred 9441284 bytes per layer/expert, consistent with 3*2048*768*2 BF16 FFN + 2048*2 BF16 router row +4 FP32 bias. Formula B(k)=3330699264+443740348*k (bytes; excludes checkpoint metadata/tokenizer and runtime buffers; counts fixed/shared/attention weights). Actual mixed dtype; not exact parameter count.
|Experts/layer|Total expert slots|Tensor bytes|Decimal GB|Evidence|
|---:|---:|---:|---:|---|
|128|6016|60129463808|60.129|measured tensor bytes|
|96|4512|45929772672|45.930|measured tensor bytes|
|80|3760|38829927104|38.830|estimate only|
|64|3008|31730081536|31.730|estimate only|
|48|2256|24630235968|24.630|estimate only|
|32|1504|17530390400|17.530|estimate only|
Q4/Q6 sizes, peak VRAM and throughput require actual GGUF; linear scaling of total bytes is only rough, not a capacity proof. Conditional follow-up: if 96 passes and 64 fails, 80 is a useful intermediate (do not assume monotonic quality); if 64 passes, try48 then32 with actual scores; if 96 fails, first distinguish numerical/runtime/quant effects and try112 or alternative selection. Only choose using complete paired validation and later held-out test with separately authorized execution.

## 4. Nonuniform per-layer counts — source constraint
Inspected existing prune.py at 3212303e1e02994a5124db8c85195b91325c7a7c (770f55b8fc20f0e78cd8b32673829da6e1a9c461): validates same keep count per layer, single config.num_experts. Pinned llama.cpp 9c003a97ab34a69cc49243f9b02014038205ffae conversion/hunyuan.py (2088953c1aa1a2b215a4dae3663797604f1d4538) reads scalar num_experts; src/models/hy-v3.cpp (f53d94827ee753e8965304941440b50995fe0b41) uses global n_expert for router/bias/expert tensors and graph, src/llama-model.cpp (72ad1b0e371832f3c5cfaeeca989a61904cc98eb) reads scalar expert count (expert USED count array is a different feature). No direct support for variable physical expert counts established. Avoid padding to largest count and claiming compression; real heterogeneous formats/loading/kernel changes would be needed unless a compatible runtime is demonstrated. Offline allocation simulation only after actual NPZ access; defer implementation until uniform results and tradeoffs known.

## 5. Handoff
R20-R1 proceeds unchanged. This parallel report is data analysis, not source correction, model qualification or new local Agent task. Keep CPU/GPU/runtime tracks distinct; future results verify the exact same validation set. No merge or final acceptance.
