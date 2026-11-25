# D01 follow-up preparation (standalone, not platform)

D01b is the authority for actual checkpoint layout, router semantics, prompts, and model execution. This independent directory supplies data and CPU-side experiment helpers. Integrate after D01b publishes exact HEAD; do not replace its adapter or already-frozen data split. All output here is preparation, never translation quality evidence.

## D01c: Data and evaluation

Use existing frozen D01b data if available, otherwise download only after accepting FLORES+ access conditions (CC BY-SA 4.0, provider requests no public redistribution). Dataset: https://huggingface.co/datasets/openlanguagedata/flores_plus . The current card identifies Mandarin as cmn_Hans, English eng_Latn, Burmese mya_Mymr. Verify actual configs and pinned revision on local download. Old FLORES zho_Hans can be adapted by passing aligned rows to `align`; do not silently mix versions. `align` requires matching IDs and split, not row position. `split_data` selects groups before expanding to six directions, excludes exact normalized text duplicates, not near duplicates or article overlaps. Dev is calibration; devtest validation/test are disjoint by group. This is a pilot, not a statistically sufficient production test. Source dataset is public benchmark and pretraining contamination is unknown.

```sh
python -m pip install -r requirements.txt
python prep.py fetch FULL_DATASET_COMMIT_SHA local-corpus
python prep.py evaluate local-corpus/validation.json teacher-predictions.json candidate-predictions.json quality.json
```

Predictions are JSON mapping sample ID to `{ "text": "...", "truncated": false }`. Same deterministic model settings and official translation prompt must be enforced by D01b, including context and max generation length. Keep raw predictions private if derived from gated data; public PR may contain hashes, aggregate metrics and code. `stress.json` includes 8 authored trilingual draft scenarios, 48 potential directions; not calibrated, native reviewed or an official reference. Use only for manual checks, invariant review, not for chrF++ gate. Verify terminology, negation, markup, numbers and code/ID preservation. Numeric check is a heuristic and can flag correct number words or formatting, so review failures.

## D02: Candidate plans

```sh
python prep.py plan profiles.json plan64.json --keep 64 --method balanced
python prep.py plan profiles.json random64.json --keep 64 --method random
python prep.py plan profiles.json worst64.json --keep 64 --method worst_first
```

Input profile mapping: layer -> direction (`zh-en`, etc.) -> phase (`prefill`, `decode`) -> 128 nonnegative accumulated actual selected routing weights. This is a mass ranking, NOT REAP saliency. Each group normalized independently; balanced ranks average mass; worst_first is greedy worst-group coverage; random seeded per layer. Plans contain original expert IDs, not rewritten IDs. Apply only through D01b's checked adapter. No-op/actual block mask equivalence and checkpoint reload required before quality tests. Profile data must be tagged with revision/corpus/adapter hash externally; this demo does not validate those identities. Follow matrix.json, choose candidate on validation only; report all failures. Test once; if test fails, do not tune on it. Baseline comparison is per direction with no mean-score override. Add 7B and random controls before claiming pruning-specific benefit. No training yet.

## D01d: Export and deployment

Official GGUF: https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF ; upstream CLI: https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench . Pinned converter must support actual pruned HY-V3 shape and metadata; first convert unpruned baseline locally with same toolchain, then candidate; do not silently fall back to original GGUF. Use exact HF candidate revision and SHA manifest, clean llama.cpp source and matching compiled binaries. `deploy.py` dry-run prints commands; `--execute` performs sequential conversion/quantization/bench with logs and hashes; requires existing output parent and conservative disk budget. Actual GPU not available here, no converter/CLI execution was performed. `-ngl 99` is a request, not proof of full residency. Use runtime offload logs (including expert/attention/KV placement), actual device ID and sampled VRAM, no other GPU workloads, inspect CPU fallback; run realistic deterministic six-direction translations at 512/2048/8192 contexts with exact token counts, p50/p95 latency and prompt/decode t/s. `llama-bench` synthetic throughput alone is not translation latency or 8192 context validation. Report measured VRAM and whether fallback occurred, and run quality evaluation on each quantized candidate. If converter/quant/runtime fails, mark BLOCKED/FAIL, not PASS.

```sh
python deploy.py --llama /path/to/llama.cpp --bin /path/to/build/bin --revision FULL_LLAMA_COMMIT_SHA --model /path/to/candidate --out /path/to/new-output --quant Q4_K_M
# Append --execute only after verifying source, matching binary build provenance, resources and candidate receipt.
```

## Local tests

`python -m py_compile prep.py deploy.py test_prep.py` then `python -m unittest -v`. Tests use synthetic profile/data and a fake metric injected only to validate gate wiring. They do not exercise dataset access, actual sacrebleu, converter, or GPU. Dependency pins are proposed, installation untested here. No weights or gated dataset included. No PRD/UI/backend yet.
