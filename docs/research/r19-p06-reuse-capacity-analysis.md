# P06 — Expert payload reuse and deployment capacity plan
R20-R1 stays unchanged. Exact R19 3212303e1e02994a5124db8c85195b91325c7a7c; P05 base f43818397601eb4a6fb99bf9a0e15b8d951f2adb. Offline calculation/source inspection only; no GPU, model conversion, quantization, new agent, test inference or runtime code changes.

## 1. Actual mask overlap and reuse
Fetched exact14 existing masks (12 P04 and two original R19); verified all47 layers, sorted unique IDs in0..127 and expected sizes. Source blobs in JSON plan. Missing IDs and layer maps are real set differences, not estimates. A source-local ID must be resolved via the parent's retained original-ID list, never mistaken for original expert ID. No tensors copied.
|Target|Expert slots available in original k96|Must fetch from original128|Layers with missing IDs|
|---|---:|---:|---:|
|64/original|3008|0|0|
|96/balanced|4354|158|45|
|96/robust|4282|230|47|
|80/balanced|3739|21|15|
|80/robust|3717|43|28|
|64/balanced|3005|3|3|
|64/robust|2998|10|10|
|64/myanmar2|3005|3|3|
|48/balanced|2256|0|0|
|48/robust|2252|4|4|
Original64 is nested, but optimized64 is NOT necessarily nested in original96. For robust64, 2998/3008 reusable, 10 missing in10 layers. For same-width robust96, 230 missing; direct subset slicing would silently generate wrong model. Source-derived model remains required for nonnested candidates.

## 2. Potential quantized expert pool, not a shipping format
Across all14 masks, total independently stored routed FFN slots48128, union4778 (98..104/layer). Each slot contains three FFN matrices, 4,718,592 elements. Independent BF16 FFN payload 454.193 GB vs unique pool 45.091 GB; pure Q4_K expert FFN payload 127.742 GB vs 12.682 GB. This is a hypothetical content reuse across every existing candidate, **not** an expected saving for the actual next one-candidate experiment. No proof all quant policies share bytes. Pool omits router/fixed tensors, metadata and assembly copies. A standard standalone GGUF cannot reference external pool by default; assembled file still requires full selected bytes. Do not build a generic weight store for this pilot.

## 3. Exact-block capacity scenarios
Measured R19 BF16/mixed tensor payload B(k)=3330699264+47*k*9441284; reproduces measured128 and96 exactly. The per-expert FFN payload is 9437184 bytes; remainder4100 per slot fits router row and bias. Pinned ggml-common.h blob 1dbbe326d0fc6a5ce8a86d237cb94901340443af: QK_K=256, Q4_K=144 bytes/block (4.5 effective bits), Q6_K=210 (6.5625 bits). Widths2048 and768 divisible by256. Conditional scenarios below quantize *all routed FFN matrices* to same type, keep fixed and router/bias byte totals equal to original. This is neither actual Q4_K_M/Q6_K nor conservative upper/lower bound on actual GGUF; actual mixed type policy, source casts, metadata, alignment and buffers may differ.
|Experts/layer|BF16 tensor GB|All routed Q4_K scenario GB|All routed Q6_K scenario GB|Q4 remaining GiB after 2048 FP16 KV|Q6 remaining GiB after 2048 FP16 KV|
|---:|---:|---:|---:|---:|---:|
|128|60.129|19.323|26.642|-2.184|-8.999|
|96|45.930|15.325|20.814|1.540|-3.572|
|80|38.830|13.326|17.900|3.402|-0.858|
|64|31.730|11.327|14.986|5.264|1.856|
|48|24.630|9.328|12.072|7.125|4.569|
|32|17.530|7.329|9.158|8.987|7.283|
Remaining GiB uses 16GiB nominal device capacity minus scenario tensor bytes and theoretical KV only. Negative means scenario cannot fully reside; positive is NOT actual fit. V100 available memory, backend graph/workspace, memory alignment, CUDA context and offload must be measured. Theoretical KV 48 attention layers x 4 KV heads x128 head dim x2 (K,V) x2 bytes FP16 =98304 bytes/token; 512=48MiB, 2048=192MiB, 8192=768MiB, 16384=1536MiB. Batch1 one sequence, no MTP, no padding. Actual llama.cpp context allocation measured separately. Focus k64 Q4_K_M first and Q6_K if actual capacity/quality warrants; k96 Q4_K_M may need offload. No exact speed estimate.

## 4. Quantize once, slice later — source-supported candidate, not implementation
Pinned llama.cpp 9c003a97ab34a69cc49243f9b02014038205ffae:
- conversion/hunyuan.py blob 2088953c1aa1a2b215a4dae3663797604f1d4538 lines439–456 stacks experts in increasing local IDs along axis0; source has dynamic scalar count.
- src/llama-quant.cpp blob c1fab892fa1a92e5f537295a29027033470eb4ff lines820–885 quantization chunks do not cross expert boundaries and use per-expert imatrix slice; lines520–545,694–702 select tensor types depending on model/layer, other policies may be more global. Lines291–307 skip expert gating quantization and nonmatrix tensors; verify actual router/bias dtype and axes.
- gguf writer blob 28b481e6526d97b4487b2b04249125f37ffb201a lines338–409 supports raw quant dtype/shape, but not an external shared cache guarantee; reader blob d8b07ef018706f6b1cd3f9f767e0ef319d066009.
- quantize CLI blob d58695196301b9d0b656e6316d177be7700895db supports --tensor-type and --tensor-type-file; --prune-layers is layer removal, NOT routed expert pruning. No existing safe direct expert-slice CLI established.
Thus a nested candidate might reuse raw quant blocks from full-parent GGUF without requantization **if** exact per-tensor type, row/block layout, source cast, quantizer/build/settings and imatrix slice identity match. Do not assume Q4_K_M types remain equal across counts; explicitly compare actual tensor type map. For imatrix-free fixed policy first, compare one real smaller candidate made by normal HF->FP16->quant and alternative slice from verified parent. Need equality of all retained raw expert slabs and other tensors, router rows/bias mapped by original IDs, scalar expert count and tensor shape metadata, correct tokenizer/template/EOS and offsets/alignment. Verify parent SHA256, identity manifest, each output tensor payload, load by pinned llama.cpp, same greedy tokens/quality on fixed validation witnesses. Reject unknown tensor roles, shape/type mismatch, missing original IDs, unsupported model/split/MTP/metadata. If bytes differ, quantify reason; do not call the two paths identical. No direct quantized surgery in active R20-R1; use current ordinary conversion as reference. First evaluate whether conversion is actually a bottleneck using its real timing. Only implement a small bounded experiment if worthwhile; avoid cache framework or custom runtime.

## 5. Next actions after active task
Read actual R20-R1 capacity/timing/results, select one comparable uniform alternative from P04. Retain full128 source for missing experts. Reuse actual verified existing quant artifacts only by content/provenance. If conversion cost material, prioritize a single k96->nested64 slice witness with original source and unchanged policy; if not, ordinary streaming prune and existing llama.cpp conversion suffice. Distinct experimental lineage for FP16/BF16 and Q4/Q6; canonical mechanism remains unresolved, no quality gate change, final test disabled. No merge.
