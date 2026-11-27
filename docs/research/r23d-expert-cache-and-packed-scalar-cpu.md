# R23D CPU-only expert cache and direct packed-weight arithmetic

Status: SYNTHETIC_BOUNDED_CACHE_PASS; SYNTHETIC_LAYER_CACHE_COMPARISON_PASS; PACKED_SCALAR_LINEAR_CPU_PASS; ANALYTIC_RESOURCE_PLAN_PASS; FIT44_WITNESSED_NOT_DISK_VERIFIED. No actual 30B model, translation quality, production runtime, GPU, native GGML kernel or deployment qualification.

All four actual jobs below ran on aliyun-sg in previous isolated CPU environment (torch2.10.0+cpu, numpy2.4.4, safetensors0.7.0 where relevant), with original engine and dependencies pinned as in prior reports. No connection to V100, model/data deletion, DeepSeek, credentials, system package change or merge.

## 1. Bounded matrix cache

Source r23d-ragged-dequant-cache.py commit4586d1be8cefcd5611189b21528c645b421c5562 blob9f0d3737d2b2b260cd31fa8c291af9e945101256 SHA256 9b9338d8c2fc4cfeb3e61660296f48280551cf926915b36dcc6e41f876427dc9. Actual Gateway job job_01M8EE09DR00SHG6SD3SJABMMQ exit0.

LRU accounts retained contiguous CPU BF16 matrix payload bytes; exact accounting on insertion/eviction, zero and oversize bypass, immutable quantized payload identity guard, ordinary in-place mutation invalidates and clears cache. Five unit negatives plus inference comparisons against original uncached quantized adapter, 8 total combinations. All 9 fixed logits,27 route hooks, KV and greedy trace exact per combination. Same synthetic model/sequence, original quantization library and matrix dimensions as prior on-demand fixture (not real checkpoint or translation). No eager routed expert dequantization.

| Type | capacity bytes | hits | misses/dequant requests | evictions | peak retained bytes |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 0 | 0 | 357 | 0 | 0 |
| Q4_0 | 24576 | 0 | 357 | 351 | 24576 |
| Q4_0 | 73728 | 135 | 222 | 204 | 73728 |
| Q4_0 | 196608 | 312 | 45 | 0 | 184320 |
| Q8_0 | 0 | 0 | 357 | 0 | 0 |
| Q8_0 | 24576 | 0 | 357 | 351 | 24576 |
| Q8_0 | 73728 | 126 | 231 | 213 | 73728 |
| Q8_0 | 196608 | 312 | 45 | 0 | 184320 |

Full capacity reduced dequant requests by 312/357, about87.4%, not measured time improvement. Cache assumes single-threaded inference-only immutable file and immutable returned tensors. stat dev/inode/size/mtime/ctime checks are ordinary mutation guards, not defense against adversarial concurrent mutation. Global cache missing-file stat errors fail the access but are not explicitly converted to clear-all invalidation by this version; partition adapter below covers missing-file invalidation. Byte cap covers retained BF16 tensor payload only, not temporary dequantization arrays, live references to evicted weights, GEMM workspace, RSS, OS page cache, or VRAM.

## 2. Equal-budget partition by layer

Source r23d-ragged-layer-cache.py commitd619815f87857e425f567a5765d99f379b50c8f6 blob685bee37799b6469d5619358c33520ad9f72dbc8 SHA256 6a8c85cbd8b7a657a5e51bc9096469ea71aeca0b7c19a70b56dda1e58e86f3f9. Actual job job_01M8EE2580002JHEQ19TFTBKXS exit0.

Equal static layer byte quotas sum exactly to cap; aggregate sum of individual layer peaks is a conservative retained-byte bound. Missing quantized payload invalidates all partitions and clears cached data. Both policies use same frozen synthetic model and inputs, 12 inference comparisons exact (9 fixed logits,27 route hooks, KV and greedy each). All request ledgers reconcile.

| Type | capacity bytes | global misses | partition misses |
|---|---:|---:|---:|
| Q4_0 | 24576 | 357 | 357 |
| Q4_0 | 73728 | 222 | 219 |
| Q4_0 | 122880 | 123 | 123 |
| Q8_0 | 24576 | 357 | 357 |
| Q8_0 | 73728 | 231 | 228 |
| Q8_0 | 122880 | 132 | 132 |

No strong preference established by this fixture. Do not choose production policy or capacity on toy results. Need actual 47-layer route-access traces with prefill/decode distinction and same total memory budget, realistic cache state and measured overhead. No fresh holdout consumption.

## 3. Direct packed scalar CPU matrix product

Sources r23d-packed-scalar-linear.cpp commit92ae29c18dcb532b333803a10b04405a0adb2fb9 SHA256 84597fdce1da1b975e902b0e3ecdf31a67cb3960116b4f823ceb072da67b7efb; r23d-packed-scalar-linear-test.py commit4b7d0a44cf875a15058e705f6126875a9c81a0ff SHA256 499d12d14a872347c723e0fa66bdebe021f13f1ffb51f9685635f934d87daf5c. Actual job job_01M8EE3VRQ00SKXBVQVEH8YSFC exit0. GCC13.3.0 C++17 -O2 -Wall -Wextra -Werror -fPIC -shared. Python independently uses actual pinned llama.cpp quantizer and dequantizer (commit9c003a97ab34a69cc49243f9b02014038205ffae, exact library file SHA pins in test).

Packed Q4_0/Q8_0 weight matrix times FP32 activation, direct block reads and scalar FP64 accumulation then FP32 rounding; no full dequantized weight matrix allocated in C++ kernel. Q4_0 nibbles and Q8_0 signed bytes interpreted explicitly with little-endian binary16 scale. Shapes include13x32,64x128,768x2048,2048x768 and batch1/3. Per-format8 actual random matrix checks and11572 outputs, plus explicit signed/subnormal scale block oracle and6 negative checks (unaligned shape, short buffers, nonfinite input/scale). Independent numerical bound uses sum of absolute terms *1e-6 +1e-7 against FP64 matmul of pinned dequantized FP32 values.

| Format | max absolute error | max error/bound |
|---|---:|---:|
| Q4_0 | 6.838954334398295e-6 | 0.023623584771100336 |
| Q8_0 | 7.625719263160136e-6 | 0.02090386977115044 |

All passed. Custom scalar arithmetic only, not native ggml CPU SIMD, CUDA, Q4_K/Q6_K, quantized activation dot kernel, or end-to-end packed FFN runtime. Numerical tolerance, not bitwise identity. Oracle uses FP32 dequantization and FP64 reference, not BF16-rounded on-demand engine; do not claim actual reference generation equivalence for this path. Return code nonzero invalidates whole output (overflow during computation can leave partial writes); caller owns buffers and must supply valid nonoverlapping storage. No benchmarking or speed claim. Next integrate separate packed FFN evaluation with controlled numerical contract, then native kernel design; no forced substitution into previously exact BF16 path.

## 4. Explicit resource planner

Source r23d-expert-cache-resource-plan.py commitd4860eb88bef27309f334861710d511d04187a96 SHA256 e3655539923f19866da55692cfc5cf30f2746eca2aab41a420bde05b335978fb. Actual job job_01M8EE57Q300A9E2CC0X6BWV8F exit0, geometry/budget/deficit/cap checks and3 negatives.

Actual geometry known from pinned research:47 sparse layers,2048 hidden,768 expert width,top8,4512 retained expert instances in diagnostic nonuniform allocation. BF16 matrix3145728 bytes (3MiB), three matrices/expert9437184 bytes (9MiB), one top8 set per layer72MiB, across47 layers3384MiB (3548381184 bytes). Full routed expert BF16 payload42580574208 bytes. Explicit total minus noncache, transient and safety reservations yields capped cache budget or accounting deficit. Illustrative (not measured) 16GiB total,10GiB noncache,2GiB transient,2GiB safety gives2GiB cache and4 whole-expert equivalents per layer at equal quota,227 equivalents globally. Actual LRU is matrix-level; equivalents are not guaranteed complete expert sets. Top8 calculation is one-token set accounting, not prefill peak or measured working set. No claim 16GiB deployment fits. Actual resource admission pending host restoration and measured memory/disk.

## Next

Actual fit44/48 unchanged, disk verification and resume pending power restoration. Offline work can continue on actual route trace cache simulator, robust global cache invalidation, direct packed gated FFN numerical contract, K-block kernels, native GGML runtime interface and real-scale source/publisher contract. Keep research variants separate from uniform96 quality-gated product path; no final model admitted. All source/report changes on PR5 only.
