# R23D CPU-only packed gated FFN and route-driven cache replay

Status: PACKED_GATED_FFN_SCALAR_CPU_PASS; K_BLOCK_DIRECT_LINEAR_SYNTHETIC_PASS; GLOBAL_CACHE_INVALIDATION_FIXED; ORDERED_ROUTE_CACHE_REPLAY_SYNTHETIC_PASS; FIT44_WITNESSED_NOT_DISK_VERIFIED. No real teacher computation, new qualified model, native GGML inference, CUDA, speed or deployment qualification.

All jobs below actually ran on aliyun-sg; no V100 connection or model deletion, DeepSeek, credentials or merge. Dedicated CPU environment and original engine dependencies exact-pinned as in previous report. No fresh holdout or real calibration data consumed.

## Packed FFN and K block scalar arithmetic

Pinned llama.cpp Python quant source inspection job job_01M8EEJVHJ00VCW0X339XDVJ5T exit0 (quants.py exact SHA f896726209ba8c91afd083dd87031c2d39dc4ef6a78e8a6c12763eeb8ed6ff84). New r23d-packed-k-linear.cpp commit236b547035a7fd98ed1fc0961b03d65321ef7634 SHA256 63b58c2a2376bd30e012228b05a3411422dec4567294030fd7d086d91062af4f; r23d-packed-gated-ffn-test.py commitd17e13baf7132166639966a7164b66b97f8b30d3 SHA256 c33681d3db0132843c1c704cebf875f817c490d67f20034b944cba44355ab840. Actual job job_01M8EEM4VR00VNBBC8FM220S9C exit0. Base simple scalar kernel source SHA84597fdce1da1b975e902b0e3ecdf31a67cb3960116b4f823ceb072da67b7efb. C++17 -O2 -ffp-contract=off -Wall -Wextra -Werror.

Direct K-block scale/min/nibble decoding for Q4_K and signed scale, high/low bits for Q6_K; no full dequantized matrix in C++ kernel. Explicit shape, input/output size and finite scale/input guards; nonzero return invalidates entire output. Caller must supply valid nonoverlapping buffers. Python test composes packed gate and up, SiLU and product, packed down, and weighted two-expert aggregation. FP32 activations and stage rounding, FP64 scalar accumulation, independent FP64 oracle using pinned GGML dequantizer. Fixed tolerance for linear output (sum absolute terms times 1e-6 plus 1e-7) and FFN/mixture relative L2 <=2e-6. Shapes hidden/intermediate256/256 and actual2048/768, batch1/3. Q4_0/Q8_0 actual quantization of synthetic weights, Q4_K/Q6_K valid synthetic packed blocks with finite small scales, NOT actual K weight quantization. 72 linear checks,24 gated FFN comparisons,12 mixture comparisons and8 K negative checks passed.

| Type | max linear abs error | max FFN relative L2 | max mixture relative L2 |
|---|---:|---:|---:|
| Q4_0 | 5.939359715156911e-8 | 7.364833660305833e-8 | 7.022942896321841e-8 |
| Q8_0 | 5.955493787013211e-8 | 6.656693052584621e-8 | 6.554128292706484e-8 |
| Q4_K | 5.856459051045704e-8 | 4.1774582950304714e-8 | 3.0076126492379794e-8 |
| Q6_K | 5.800825597646053e-8 | 7.750211852087528e-8 | 7.45666473337717e-8 |

Different numerical contract from exact BF16 on-demand engine. Not a fused FFN kernel or integrated full autoregressive packed runtime. No real model quality, native SIMD/GGML or performance test. Do not silently replace BF16 reference.

## Global cache fail-closed repair

r23d-ragged-dequant-cache-v2.py commitdf41027f8e0d2bccfafe8891cd86f7c14093dcfa blobbb8bcc4dee528c6e6a98cf84b21bfe5256b31bcb SHA256 39977df42e6e1c52b9cb9e746bc08f5028dc3e44628498d1d84a6d6daa0a91fb. Actual job job_01M8EENZHV000P7PV07ZJ064FY exit0. Identity stat missing/unavailable and failed load now invalidate and clear all entries; existing ordinary mutation guard unchanged. Seven unit negatives, original eight end-to-end comparisons and cache counters reproduced exactly; no new latency or memory claims. Actual missing-file mutation on populated live adapter tested in following job. Still single-threaded, immutable-pack assumption; no concurrent attacker or allocator/RSS qualification.

## Ordered actual synthetic engine route replay

r23d-route-trace-cache-replay.py commit3860078ba1ce01aac511a78361c1baa4cafd8f7c blobc44c77ca6197597e9fc21f0b4e89b8f51d137e76 SHA256 e8a724417add3235e9ae353ac847d697570539a1b501baefba9bdee34ba3ce0d. Actual job job_01M8EEQ49X00J5S60HRMPMYJ9S exit0. Synthetic 128-wide Top8 HY-V3, five sparse layers64/80/96/112/128,480 retained experts, Q4_0 actual synthetic quantization. Three free greedy prompts (first repeated),8 max new tokens each. 135 actual ordered route hooks, 3846 matrix requests (prefill966/decode2880), canonical ordered trace SHA256 231c241bfde64a089752df60620ea7839aba157eba7263c746b91cbcb28ff04e. Derive actual grouped engine sorted original expert ID union per hook, convert to compact local index, gate/up/down access order; match every request against independent actual uncached matrix-access instrumentation. Preserve phase, original mask, exact traversal order. Invalid duplicate route rejected. Global LRU replay exactly matched actual live engine hit/miss/eviction/bypass/retained/peak counters for all five capacities, generated tokens/finish and full traces exact. Equal static layer partition replay is simulated here; prior small-fixture live layer cache equivalence remains separate. No real translation trace. Actual fixture matrix size4096 bytes; capacities expressed in matrix units (not actual30B model measured memory). Trace retained by reproducible fixture and exact digest, not raw trace publication.

| Matrix cap | Global misses | Layer misses (simulated) | Global prefill misses | Global decode misses |
|---:|---:|---:|---:|---:|
| 0 | 3846 | 3846 | 966 | 2880 |
| 24 | 3846 | 3846 | 966 | 2880 |
| 120 | 3051 | 3012 | 966 | 2085 |
| 240 | 2046 | 2043 | 897 | 1149 |
| 480 | 1083 | 1125 | 669 | 414 |

Layer policy marginally better at120/240, worse at480; no universal winner. Replay is based on ordered hook-level grouped accesses, not aggregate frequency/importance statistics. Actual real teacher routing traces remain unavailable while V100 offline; do not reconstruct fictional order from profiles. Real cache policy should use frozen actual calibration traces, no holdout selection. Model-level latency/memory and cache admission require measured host resources. Global missing payload invalidation confirmed against populated actual cache.

## Next and blockers

Actual R23F fit44/48 witnessed unchanged, disk not verified. Uniform96 quality-gated path unchanged. Remaining CPU work: integrate full packed FFN execution into reference graph with separate FP32 numeric contract and routing comparisons; validate native GGML low-bit kernel and nonuniform graph interface; real-scale immutable source writer and exact resource admission; actual K quantizer fixtures. No need to block offline work on power restoration. No production candidate or final quality claim.
