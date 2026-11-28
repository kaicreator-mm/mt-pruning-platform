# R21/R23 native prefix and sparse counterfactual witness

Status: SCOPED_ACTUAL_PREFIX_REPLAY_AND_SYNTHETIC_DIFFERENTIAL_COMPLETE. Research base 7037a70b09c3ba220b70be90c1a38920f947d3eb. Active native full128/96 serial job job_01M8DVGG550050NBY708JNDKXZ unchanged; no competing GPU or real model workload. Old-protocol results exploratory only, canonical mechanism and final-test HOLD.

## R21 independent native prefix audit

Actual Gateway job job_01M8DWV29G00S0HB3PBQ9TEQ24 exit0, Python compile PASS. Read only completed journal prefix, verified exact scorer/protocol/driver blobs, frozen prompt IDs and corpus, original model identities and size, admission hashes/provenance, runtime plan, raw request/response/log SHA and effective settings, actual generated tokens, EOS and per-request task sequence. Added check that every captured request log is present byte-for-byte in increasing offsets in actual master server log. 128/96 pilot8+8 and128-full91 rows,107/107 raw capture/master slices match. Active128-full prefix91/120 all EOS,0 empty,0 length; full96 not started. Five pilot/full matching sample IDs have exact raw tokens AND text5/5, remaining pilot witness not yet in full prefix. Full source hashes verified by existing execution driver; not independently re-read by this auditor. No final qualification.

|Direction|Rows|R19 BF16 chrF++|Native128 FP16|Exact text/token matches|Digits BF16→native|
|---|---:|---:|---:|---:|---:|
|zh-my|20|39.759619|39.064691|10|3→3|
|my-zh|20|21.121995|21.430077|11|1→1|
|zh-en|20|58.625453|58.625453|20|3→3|
|en-zh|20|32.122041|33.206154|13|1→1|
|en-my|11 (partial)|44.421705|44.552760|3|0→0|

BF16 bridge descriptive only, different precision/runtime. Existing Burmese token-diversity repetition heuristic unchanged, not a native human adjudication. Full12891 prefix wall1333.442s, prefill9177 tokens/403314.531ms, decode7764 incl EOS/925046.987ms. Actual complete pair only after both120, then run pinned scorer and independent pilot/full check. No validation fitting.

## R23 sparse counterfactual witness

Earlier exhaustive output witness computes all128 expert functions on every selected row, even though the fixed five96-mask inventory only compares selected top8 routes. New r23-sparse-counterfactual-witness.py computes the union of expert IDs selected by full teacher plus all declared candidate masks, using exact same scores, bias, normalized weights and batch shape. Each required expert evaluated on all selected rows (preserves GEMM shape), uncomputed tensor slots NaN, undeclared mask rejected. Preserves original teacher combine/shared computation and observer drift/energy/loss semantics, rejects boundary ties. Original collector adapter does not alter Engine._moe teacher result. Separately records actual counterfactual token-expert evaluation count, not misleading nominal count.

Actual V100 CPU torch2.10.0+cpu synthetic differential job job_01M8DWYT5800R2C0KTD2WJZ12T exit0, compile PASS, CUDA disabled/one thread. Twelve seeded synthetic128-expert BF16 layers, five random96 masks each, one- and two-row cases:120 candidate error comparisons exact, computed expert bytes, full reconstruction drift and teacher energy identical to exhaustive,4 negative checks. Extra candidate expert evaluations4608→631 (86.306% reduction); full tiny Engine differential nine logits checks/six observations exact, extra64→38. Real model reduction/runtime not measured. Fixed candidate inventory only: cannot score arbitrary new mask without capture of required experts. Counter excludes teacher combine, shared expert, other layers and IO; no claimed end-to-end speedup. Full-sized temporary value tensor remains allocated with NaNs; memory savings not claimed. Real driver integration/admission and actual one-sample oracle on original30B pending after active full pair; use existing exhaustive path as real finite crosscheck if needed, no actual real capture now.

## Resource and dispatch boundary

Actual read-only probe job job_01M8DWSNK700D6EP0FHBR5PPRM exit0 at14:59+08:00 saw128-full88/120 GPU PID517196/13030MiB, disk53494407168 bytes. Follow-up job job_01M8DWW4T0008DZW2YJ0PGE1FK exit0 at15:00+08:00 saw92/120, free53484437504. Existing native model validation needs no new large file. Guarded96 output45953803936 bytes plus provisional10GiB reserve still exceeds free by3206784672 bytes at this probe. aliyun-sg free27578036224 bytes, not enough for whole prior64 HF intermediate (~31.74GB) plus reserve. V100 conda package and HF caches inventoried/dry-run only; conda cache hardlinks and unrelated HF models cannot be assumed safely reclaimable. No cache/model deletion, no partition changes, no reserve bypass, no transfer. Re-evaluate actual need after native paired result. Current finite job serially executes96 after128, then scorer; do not launch duplicate. R22/R23 conditional after pair, no training/final test/merge.
