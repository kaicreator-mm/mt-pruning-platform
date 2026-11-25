# Download-window R3 — D06–D12 bounded CPU demos

Parent: kaicreator-mm/mt-pruning-platform #1/#2. This is independent preparation, not production code or the D01b model adapter.

## Run

Python 3.13.5, torch 2.10.0+cpu, safetensors 0.7.0 tested. `stream_prune.py`, `search.py`, `stress.py` themselves use only stdlib. The tests require torch and safetensors. From this directory:

```sh
python -m py_compile stream_prune.py search.py stress.py test_r3.py test_stress.py
python -m unittest -v
python stress.py
```

## Components and contracts

- D06 `search.profile(events,n,k)` consumes one actual or synthetic trace event per token/layer with selected expert IDs, actual routing weights and **unweighted expert output L2 norms**; fields `layer`, `direction`, `phase`, `ids`, `weights`, `output_norms`. Frequency, routing mass, weighted-output-norm saliency are different statistics. The saliency is a REAP-like proxy, not a verified implementation or accuracy result. All six directions and both phases required per layer. Equal group normalization protects against token count imbalance. `select` uses fixed k; supported methods frequency/mass/saliency/coactivation/random. Coactivation adds a weighted pair-count bonus to a greedy saliency heuristic; no optimality or causal cooperation claim. Real trace collection and provenance must come from D01b; this module does not hook the model or authenticate trace content.
- D07 coactivation: group-normalized pair frequencies, seeded comparison, coverage diagnostic. Test only verifies designed artificial cases.
- D08 `sensitivity(rows,budget,k)` accepts `{layer: {retained_count: estimated_validation_loss}}` as Python mappings, finds a discrete additive proxy minimum within an expert-count budget. Assumes comparable cost per expert. Each option must be measured independently on validation in actual experiments; joint nonuniform candidate must be reevaluated. Nonuniform counts may not be supported by the runtime and are not exported by D11.
- D09 `choose(reports,max_drop=1)` selects smallest eligible bytes, otherwise returns FALLBACK_TO_TEACHER. Each report: `id`, `bytes`, `split=validation`, `validation_identity` (SHA256 binding to frozen data/teacher/prompt/metric protocol), `directions` (all six, each with `n`, `baseline`, `candidate`, `baseline_errors`, `candidate_errors`; each error map has empty/truncated/number_mismatch). Reject mixed identities/baselines, missing directions, test split, invalid values. The identity is compared, not independently authenticated; actual final-test isolation and provenance enforcement belong to experiment runner. No production acceptance.
- D10 `stress_r3.json` contains 12 authored trilingual draft cases; `stress.py` expands 72 source prompts (no reference field), with exact literal-count diagnostics. These are not human-reviewed references, may include awkward Burmese; no semantic quality score. Do not calibrate or select on these if reserved for diagnostic holdout. Existing R1 draft 8 cases separate.
- D11 `stream_prune.py SOURCE OUTPUT PLAN.json` streams byte ranges from local, complete, unchanging teacher snapshot. Output must be a new sibling/outside teacher path. Input plan: `original_count`, `keep_count`, `top_k`, `count_field`, `top_k_field`, `layers` (layer -> sorted unique original expert IDs), `required_roles` (expert/router and bias if present), `tensors` (exhaustive inventory mapping name -> `{role:copy}` or `{role:expert|router|bias,layer:...,axis:0}`), `source_sha256` (config/index/all indexed shards). Use a reviewed architecture adapter to classify every tensor; this demo cannot detect a falsely classified tensor. Supports fused contiguous axis-0 expert layout and row-wise router/bias only, no split-per-expert layout, alternate axis, FP8, packed quantized weights or arbitrary architecture. It preserves byte ranges/dtype and shard metadata, updates one reviewed expert count field and index total size; no tokenizer or sidecar copying, no complete model export or forward pass. Copies payload in <=1 MiB chunks; headers/index/plan resident, so not a measured total RSS claim. Hashes inputs before and after, hashes outputs, stages output before rename. Run single writer, immutable input snapshot; no concurrency/crash durability guarantee. It does not overwrite teacher or existing output. Use actual adapter to check all config aliases, router semantics, expected layers and tensor classification.
- D12 deterministic output hashes and receipt; test generates two candidates from identical source/plan and compares exact receipts and output digests. Real clean reload / masked equivalence / generation remains D01b.

## Actual CPU result

28/28 unit tests, five Python files syntax-check, actual Safetensors read/write against two generated shards (not pretrained model): F32 expert/bias, BF16 router/shared, I64 attention. Retain 8/16 experts; payload 272960 -> 140832 bytes; all retained and unchanged tensors exactly equal to PyTorch indexing/copy; no-op payload equal; repeated output hashes identical. Negative cases include missing roles, unclassified inventory, wrong axis/shape, duplicate IDs, changed teacher, existing output, invalid trace, mixed validation identity/baseline, test split, missing directions and literal contracts. The payload reduction is fixture-specific. No real Hy-MT3, FLORES download, chrF++, GGUF or GPU results. No full-model quality claim.

## Integration

Keep D01b as actual checkpoint adapter authority. Integrate in separate branch after main exists, review actual D01b interface and reconcile contracts; do not substitute this streaming demo for D01b or interfere with download. Run real fixture slice/forward/quality only after verified source revision, inventory, axis and router mapping. Do not use test set for model selection. D08/D05 nonuniform or expert-specific quantization remains research until engine compatibility is demonstrated. No UI/training. GitHub issue tracks evidence until source can be committed.
