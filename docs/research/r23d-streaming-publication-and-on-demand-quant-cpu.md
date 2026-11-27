# R23D multi-shard streaming, atomic publication and on-demand quantized CPU runtime

Status: SYNTHETIC_STREAMING_SHARDED_PACK_PASS; SYNTHETIC_SOURCE_BOUND_FAULTS_PASS; SYNTHETIC_Q4_0_Q8_0_ON_DEMAND_PASS; FIT44_WITNESSED_NOT_DISK_VERIFIED. No actual new teacher computation, real model construction, translation-quality acceptance or native GGUF qualification.

## Scope

User requested continuing without V100. All three tests below actually ran on aliyun-sg using the previous dedicated CPU-only environment /tmp/r23-ragged-cpu-env.0ZeMon/venv (torch2.10.0+cpu, safetensors0.7.0, numpy2.4.4). Original engine and all dependent source files exact-pinned as in prior report r23d-ragged-reference-runtime-and-quant-layout.md. No V100 connection, system Python modification, external LLM/DeepSeek request, secret access, original model/data deletion, or merge. Temporary synthetic test directories cleaned by their owners. Research branch PR5 only.

## 1. Streaming multi-shard compact HF research format

Source r23d-ragged-streaming-shards.py commit19a85b49ee976c0ad78b75671f6ea1256ea8a281 blob2a53ccf64f8dccdc31da096b58d59e91cb8820f8 SHA256 be6679be3320c4b9ec2e682b227225948b90ec54479380440054ceafeee61143. Actual job job_01M8EDCGAF00RJSS3EW3Q88TR4 succeeded exit0.

PASS_CPU_STREAMING_RAGGED_SHARDS: 25 output safetensors shards, 1519 independent source tensor checks, 65536-byte copy chunks, 262144-byte shard payload cap (header bytes excluded), 9 fixed sequence logit comparisons,45 route hook comparisons, exact KV and greedy trace,7 negative cases, simulated prepublication failure visibility, original fixture hashes unchanged. Fixture has original128 experts, top8, five sparse layers with budgets64/80/96/112/128, source640 to compact480 expert instances.

Format r23-ragged-hf-sharded-full-router-v1:
- strict bounded safetensors header parser, BF16/F32 only, rejects duplicate JSON keys, unsupported dtype, gap/overlap, bad extents, trailing bytes, unsafe shard names and inconsistent index.
- validate exact source provenance, original configuration and all tensor family shapes/dtypes (attention, normalization, embedding, output, dense, shared, router, bias and expert FFN). Full router/bias unchanged. Compact only routed FFN names and raw tensor slices.
- deterministic sorted tensor partition within configurable shard payload limit; no full weight dictionary loaded by writer, copy and hash raw data in 64KiB chunks. Metadata/header memory is separate, not bounded by the data copy size. The source is intentionally restricted to synthetic provenance and <=64MiB total. No real checkpoint allowed by this entry.
- canonical manifest with mask, source provenance digest, config digest, shard and per-tensor hashes, weight map. Staging directory is separate, each file fsync, final stage directory fsync. Before publish, replay the original source plan and compare every retained tensor byte and hash. Exclusive Linux renameat2(RENAME_NOREPLACE), then parent directory fsync. No overwrite, incomplete stage retained; verified complete stage can be published after explicit re-verification. Simulated failure only, not actual power-loss durability across filesystems.
- sharded loader verifies structural layout and pinned shard hashes, opens safetensors handles lazily; reference runtime uses mmap-backed tensors and original-axis routing. Loader initialization still scans payload hashes, and Engine initialization maps the required tensors; not a measured bound on total RAM, file handles or page cache. Only trusted immutable local files assumed; adversarial concurrent file mutation/TOCTOU not qualified. No production writer, tokenizer export, product CLI, GGUF or native loader support.

## 2. Multi-chunk and fault injection supplement

Source r23d-ragged-streaming-fault-tests.py commitb75d6283e6f2168549e5e089c91d3911f36016be blobf2f41c4f4248e3a9a21aef6b711a9faf4196a690 SHA25636d202e5d2b106971ff04218f9468b40f8a9586d64f7388a1ebd22d2c1949620. Actual job job_01M8EDE6NJ00G7557FQ2G2HY9Y succeeded exit0.

PASS_CPU_STREAMING_MULTICHUNK_FAULTS: fixture vocabulary4096, 115 observed copy read requests, maximum65536 bytes, largest tensor8 chunks,101 tensor comparisons. Deliberately changed embedding byte, recomputed all relevant output hashes and canonical manifest digest: structural inspect accepts internally consistent altered pack, source-bound verification and publication reject. Deliberately interrupted fifth copy: no final directory or completion manifest, partial stage preserved, publish rejects. Valid original stage still publishes and independent tensor comparison succeeds. This differentiates artifact self-consistency from source-bound verification. Real power cut NOT_TESTED.

## 3. On-demand quantized routed FFN CPU prototype

Source r23d-ragged-quant-on-demand-cpu.py commit41941d381d45a1b7e9fc5774e426a805c58f2852 blob b9f938cc0ff430c127017e2b30d4ae04392f82b6 SHA256cf56d5a26a796b8145e4265028f1540eee4ca24de071ad78a3161791557409b4. Actual job job_01M8EDG0C800YSAZ9QEG4VQ3WY succeeded exit0.

PASS_CPU_QUANT_RAGGED_ON_DEMAND: actual Q4_0 and Q8_0 quantization of synthetic BF16 expert weights,48 packed routed FFN matrices per format. Other tensors remain unquantized in plain.safetensors, original router axis preserved. Separate synthetic format r23-quant-ragged-ffn-fixture-v1, not standard GGUF. Pinned llama.cpp Python quantizer/dequantizer from commit9c003a97ab34a69cc49243f9b02014038205ffae, exact pins from earlier layout module. Loader validates inventory, shape/dtype, contiguous quantized ranges, payload hashes and per-tensor digest on access. Expert weight sequence accesses are lazy; original expert_combine indexes only routed local experts. Selected matrix is read, dequantized, rounded BF16 and fed into existing grouped CPU FFN computation; no persistent dequantized expert cache. No native quantized GEMM.

Per format results:
- Q4_0:0 eager expert dequant requests,357 selected matrix dequant requests, max1152 quantized bytes per request,9 fixed logit comparisons,27 route hooks, exact KV and greedy trace, corruption rejected on next affected matrix read.
- Q8_0:0 eager expert dequant requests,357 selected matrix dequant requests, max2176 quantized bytes per request,9 fixed logit comparisons,27 route hooks, exact KV and greedy trace, corruption rejected.

Oracle independently quantizes/dequantizes kept original tensors in original full masked engine using same pinned quantization library (not the compact storage reader), so this tests mapping, data access, dequantization and engine integration; does NOT compare quantized weights with original precision for quality. Repeated dequantization costs and runtime RSS not measured; per-tensor reads are bounded by fixture matrix size, not 64KiB streaming-copy contract. Fixture builder retains plain tensor dictionary and is not a production streaming quantizer. Only Q4_0/Q8_0 actually quantized here; Q4_K/Q6_K prior slab-layout witness not end-to-end quantization or inference. No native llama.cpp, real model, speed or VRAM qualification.

## Remaining work

Actual R23F fit44/48 Gateway-witnessed prefix still requires V100 restoration/disk verification. Uniform96 path unchanged. Nonuniform reference research now has synthetic streaming multi-shard physical storage, source-bound publication and on-demand quantized expert access. Remaining CPU-only candidates: robust immutable-source publication contract for real scale, artifact resource plan and multi-shard/native loader integration, bounded dequantization cache policy, native GGML graph design and actual CPU kernel equivalence. Real source admission, physical model construction, actual masked-vs-compact teacher comparison, end-to-end translation quality and deployment remain blocked on actual host/data and preceding quality gates. No automatic real-build admission or user intervention required for further offline tasks.
