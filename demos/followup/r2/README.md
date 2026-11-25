# Download-window demos (D01b preflight / D04 / D05 design)

Separate from D01b. Do not modify its active branch, dataset or checkpoint. Python standard library only. No model weights downloaded or executed here.

## D01b checkpoint preflight

Run `python preflight.py /path/to/hf/snapshot inventory.json` (fresh output path). It reads config, checkpoint index and 8-byte length + JSON header of each available shard; no weight payload is read. During downloading, only completed/visible snapshot files may be present; missing/partial files produce exit 2 and BLOCKED. Re-run with new output after download. For stable snapshot only: not a live downloader monitor. Header inventory is not proof of payload integrity; verify actual pinned revision, full file SHA256 (compare publisher hashes where available), checkpoint architecture, real reload and routing separately. Header inventory includes names, shapes, dtype, byte counts, and shard identity, without guessing tensor roles. Unsupported dtypes fail closed. Only indexed sharded Safetensors layouts are supported. D01b owns real adapter and config interpretation.

## D04 profile reuse

`python reuse.py records.json plan.json --third my --keep 64` consumes six **original teacher** direction records. Each record has `identity` (model_revision: full 40-char commit; tokenizer_sha256, prompt_sha256, router_sha256, collector_sha256, protocol_sha256: full SHA256), `role: original_teacher_calibration`, `direction: zh-en`, `corpus_sha256`, `layers: {"1":{"prefill":[...],"decode":[...]}}`. Collector protocol must bind actual selected routing weights, top-k, dtype/backend and corpus collection procedure; verify actual inputs before issuing identities. Corpus digest is per direction. Require exact matching identity and layer set. For future zh/en/th, reuse zh-en/en-zh only if protocol/identity unchanged, collect zh-th/th-zh/en-th/th-en afresh. This is plan-only, no inference. Reuse does not prove a smaller model works. Compare to fresh collection on the same sample IDs and quality evaluation on separate held-out data; if protocol changes invalidate cache. The existing D01b stats may require a reviewed mapping to this format.

## D05 quantization plan

Use pinned llama.cpp revision, first verify unpruned and pruned BF16 GGUF reload and matched prompts. Then compare Q4_K_M/Q6_K and optional in-domain imatrix on the SAME selected pruning candidate. `--tensor-type` is tensor-level (regex), not individual expert-level when the checkpoint/GGUF stacks experts into a single tensor. Expert-specific precision is an experimental extension, not supported merely by a regex. Check actual GGUF tensor names/shape before deciding overrides. Keep Router/bias and shared tensors at baseline precision for the first controlled comparison. No quantization of already quantized sources. Use validation split only for search, separate final test once, do not reuse D01b's final test as a tuning set. If final test was already consumed, allocate new held-out samples. `--dry-run` only estimates quantized size, not VRAM; measure peak GPU allocation/residency and latency during actual translation. Refer to https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md and pin observed runtime version before execution. No GGUF converter/runtime executed here.

## Verification

`python -m py_compile preflight.py reuse.py test_demo.py` and `python -m unittest -v`: 12 synthetic tests pass. No real headers/profile or quantization tested. No formal platform or training introduced.
