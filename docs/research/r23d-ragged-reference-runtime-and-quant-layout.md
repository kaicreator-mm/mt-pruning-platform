# R23D CPU-only reference runtime and quantized slab evidence

Status: RAGGED_REFERENCE_RUNTIME_SYNTHETIC_PASS; QUANT_BLOCK_LAYOUT_SYNTHETIC_PASS; FIT44_GATEWAY_WITNESSED_NOT_DISK_VERIFIED; REAL_TEACHER_AND_NATIVE_RUNTIME_NOT_RUN. No production quality or deployment acceptance.

## Scope and environment

User requested more work not requiring V100. No V100 connection attempted. No large checkpoint copied, no original model/evidence deletion, no DeepSeek requests or credentials. All actual tests in this batch ran on aliyun-sg. The host did not originally have torch, numpy or safetensors installed. A dedicated temporary Python venv /tmp/r23-ragged-cpu-env.0ZeMon/venv was created, no system Python package changes. Initial install job job_01M8ECHC1M000VPS5M1C1Z9NEG reported Gateway control internal_error after successful torch install; environment independently rechecked and completed in job job_01M8ECKYN400GMZ75WDDGFDWC7 exit0. First adapter test job job_01M8ECMSS500B38HF5V32Q8ZRH failed on missing packaging; job job_01M8ECNFT000WAMSMHQ8GHG3G4 added packaging but failed on missing corpus dependency; job job_01M8ECPSX300N3J3S9C54R1XVQ failed on missing reviewed demo dependency. These are setup failures, not inference results. Final retry fetched the missing pinned sources without changing adapter or model arithmetic. Preserve Gateway job records.

Actual environment validation job job_01M8ECVMCP00CAM0S55FJDT8ET exit0: pip check No broken requirements found, 945M dedicated env, 25G root free, MemAvailable2239352kB, SwapFree1048572kB. Keep this disposable environment for subsequent CPU work; no model stored in it. Exact installed versions:

```
filelock==4.2.4
fsspec==2026.9.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.7
numpy==2.4.4
packaging==26.0
safetensors==0.7.0
setuptools==78.1.0
sympy==1.14.0
torch==2.10.0+cpu
typing_extensions==4.16.0
```

## Actual compact HF research pack and reference runtime

Source r23d-ragged-reference-engine.py commit b3f8cd22d662e7e5e1a1d95d406d7f69f52fb56b, blob56ceecad1fd2db0d33399840e2231c718c7e6ab0 SHA256 dce3cdfcfd2461e3a8ffef95831937b03b29870245939b76aeb7ed5da7e97013.

Actual CPU test job job_01M8ECRACS00B3CJRDVRSA3MYS exit0, PASS_CPU_RAGGED_REFERENCE_ENGINE:101 physical tensor comparisons; three sparse layers count3/5/8;54 fixed logit checks;162 route hook comparisons;6 free greedy trace checks,3 deliberately tied prompts;6 negative tests. Fixed sequences compare logits, original expert IDs and order, routing weights, and all populated KV cache bytes. Greedy compares generated tokens, finish and every trace tensor. The source is a random-weight tiny HY-V3 checkpoint, not actual translation weights. No native llama.cpp tests.

Architecture:
- Independent format r23-ragged-hf-full-router-v1, original config retained, canonical manifest pinned, exact payload hash and physical key inventory. Removed experts physically absent, compact local indices derived from sorted original-ID masks.
- Full original router and FP32 bias retained. Inherited original Engine._route computes same original-axis choice scores, masks before top-k, normalized unbiased routing weights. Fixed original-ID to compact index lookup before existing Engine.expert_combine (same grouped-by-expert GEMMs and FP32 combine). Route callbacks and inherited greedy traces preserve original expert IDs. Original attention, shared FFN, dense FFN, KV cache, residual and generation reused.
- No silent mask removal or changes; invalid/extra payload rejected. Synthetic builder intentionally limited to <=64MiB source and source provenance kind synthetic. Single safetensors shard, not HF standard layout or a GGUF. No production loader qualification or real checkpoint writer. Actual source full tensor verification is independent against fixture source; loader alone validates pinned synthetic manifest/payload and shape/inventory, not trust of an arbitrary third-party manifest.

Original engine code pinned to PR4 HEAD3212303e1e02994a5124db8c85195b91325c7a7c, engine blob2af43ca1710b029fdde0d1135ea9879fe8defe56, testing e8979ac3cd42b739dd93448f3384c188c08568c4, prune770f55b8fc20f0e78cd8b32673829da6e1a9c461, identity0a2375314d599af492e6bffc8e24287150262f58, corpus221676be5353b7d1dd71ec2e846818c3cb253f14, reviewed5d44b6cd61aa1d4a4a78b141306c43c8c8818ce9, specf01448188370c10065ee3672f3443b2aee6c4a15. Reviewed demo dependencies r6/bridge.py f3329d7659a49820201522f9cad16024a86bf4f0, r7/handoff.py e20612f23a189de41b4f573b4e262ce99d5b5d69, r4/trace.py a6fd6ad00b130bd5b827d4110cb40a3618bce3ee, r5/cache_plan.py c27319d0369f1e0e292992b20c12f4b93b0ac8a9. Exact source bytes checked by git hash-object before tests, temporary test files cleaned, original project checkout untouched.

## Actual 128-wide / Top-8 supplementary test

Source r23d-ragged-128-top8-integration.py commit724bb8fb11f20e80ade9a026e7b06dcf4adf3a4c blob9488ba1eef6ed0e8a2c08bfbebb0b2a80ed69512 SHA256 4992a316a656d7a3b9171b6fd6badc56055d5824403bb7199c258db6c1d306be. Actual job job_01M8ECTV7N006XBTW3F0J2TDNE exit0, PASS_CPU_RAGGED_128_TOP8. 5 sparse layers with retained counts64/80/96/112/128; top8, source640 experts to compact480, removed160 expert instances and480 actual FFN tensors. All1519 retained tensors compared to exact source mapping;36 fixed-logit and180 route-hook comparisons,4 greedy trace comparisons. Includes ordinary and deliberately all-equal routing scores. Rehashed malformed payloads with truncated router dimension, wrong FFN dimension, or extra tensor rejected (3 negatives). Original mask vs compact equivalence on synthetic models at actual routing width and top-k, not actual hidden dimensions, model weights, architecture runtime precision/placement, or translation quality.

## Quantized expert slab layout

Pinned llama.cpp commit9c003a97ab34a69cc49243f9b02014038205ffae. Header inspection job job_01M8ECNXPD002J5DV4XYF7PMPN exit0, ggml-common.h SHA2560061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c. Actual blocks: Q4_0(32,18),Q8_0(32,34),Q4_K(256,144),Q6_K(256,210), all in elements/bytes. Source inspection job job_01M8ECQ7YA004MGK9RFE2R6K3B exit0: gguf constants.py SHA3e637e339284da81df558c83b9aa4efb9ca3106cf983da15da67766a5d9118db, lazy.py SHA dbcf8b7376284ab2d920103657cc52c1c62e93fec3856a08a1ae638394e6a7c1, quants.py SHA f896726209ba8c91afd083dd87031c2d39dc4ef6a78e8a6c12763eeb8ed6ff84.

Source r23d-ragged-quant-block-layout.py commit444c8554f417429f9cd39dfd1eec59e1db4ca5f0 blob a1b29370a7d7bddb2f3679f4783a7b784066fd34 SHA25655611730ef27ace061c55de19936230887734efc3ce4c52ede04b129b04ec028. Actual job job_01M8ECSMQT00VGK0JT3045FXY3 exit0, PASS_CPU_QUANT_BLOCK_AXIS_ORACLE,8 synthetic tensor checks,6 negative tests. Q4_0/Q8_0 inputs actually quantized from synthetic weights by pinned Python quantizer. Q4_K/Q6_K valid synthetic packed blocks (finite scales), not actual K quantizer output. Compare whole selected expert slab bytes independently and dequantized selected arrays exactly against full source dequantization. Reject unaligned inner row, invalid ID, unsupported type, short payload. Blocks are within rows; do not cut inside blocks or change their scale data. Full router/bias remain FP32 in proposed GGUF-derived format, no quantized routing claim.

For actual average96 across47 layers (=4512 experts), each expert three routed FFN matrices with 2048x768 elements. Analytic routed FFN payload only:
| Type | Bytes |
| --- | ---: |
| F16 | 42580574208 |
| Q4_K | 11975786496 |
| Q6_K | 17464688640 |
| Q8_0 | 22620930048 |

This is NOT Q4_K_M mixed-tensor quantization, complete model size, quality, or VRAM estimate. Shared/dense/attention/embedding/output/router, headers, alignment, buffers, KV, runtime residency and mixed precision policy not included. No full model quantized, no native inference run. R24 unchanged, conditional on quality-approved candidate.

## Next CPU-only work and actual blockers

Reference runtime proof now covers actual routing axis and top-k at small hidden/sequence dimensions. Next non-V100 tasks: implement bounded multi-shard compact physical writer/loader contract with source provenance, streaming byte verification, shape/dtype coverage for all tensor families, atomic publication and simulated crash recovery; validate quantized physical pack loader and production grouped computation without assuming K block layout implies native kernel qualification. Native llama.cpp graph/loader changes need separate contract and tests; do not mislabel research pack as standard GGUF. Keep reference runtime and uniform96 product path separate. Actual 44/48 fit prefix still requires restored V100 disk verification; do not start new teacher experiment on aliyun-sg (insufficient RAM/disk for original full checkpoint). No actual quality-qualified new candidate. All changes on PR5 research branch, no merge.
