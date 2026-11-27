# R22 Guarded96 prefix audit and R24 quantization dry-run preparation

Status: ACTIVE_FULL_VALIDATION_PREFIX_ONLY; R24_SIZE_DRY_RUN_ONLY. Source HEAD 485164f78ad3ac6ee1d7c655f60516b0ad0eef14. Active R22 full job job_01M8DZXJSX00VRT8NSR3D86469 remains untouched. No new inference, candidate tuning, quantized model generation, cleanup or training.

## Independent active-prefix replay

Actual Gateway job job_01M8E0F3GD0077AEBHD10RDTEB succeeded exit 0. Reused exact existing audit blob f20d52bb37005f8573375bbe0166d17b05618daf with the sole scoped adaptation of admission comment IDs for guarded96: pilot 7212864536, full 7212917572; pinned scorer ee010f8e92e649dfca9b930c8d19c414d99e26e2 and native protocol/driver as before. The executable adapter and command are retained in the Gateway job receipt. Read-only CPU with one thread, no weight rehash or new generation. All 8 pilot and 22 active full journal rows independently verified against exact requests, effective response settings, generated raw tokens and corresponding sequential master log bytes. All 30 EOS, zero empty/length, pilot/full overlap 1/1 exact raw tokens/text. Replayed both existing full128/original96 baseline journals with pinned scorer and journal hashes.

Full prefix journal SHA256 66d53616ade795ae8fedf962abb06bee1642157681686eb6cba30518c7253fd6; admission f93cb112c2e9cd5e1eb3d39819f8f094aca8a588ed315f158c268e6140d808ee; run plan d7c0906f2b1ba26cb207ee7d6b4fd5fbbc22099b2d74be7fcf3d1018084fac46; master log snapshot 5eefad34eb37a1e0c6c180eeb2c63b129f156ac051977cce7841737beb7decd3; 22 matched slices. Active prefix not complete; source weight full hash is bound to run plan, not recomputed by this audit.

| Direction | n | 128 | Original96 | Guarded96 | Guarded96 delta vs128 |
|---|---:|---:|---:|---:|---:|
| zh-my |20/20|39.064691|38.186786|38.816713|-0.247978|
| my-zh |2/20|18.175405|16.632071|15.595549|-2.579856|

zh-my meets numerical and digit/empty/truncation directional limits on the completed 20 matched IDs; all three models have 3 digit flags and 20 repetition flags on these IDs under the pinned evaluator. The Guarded96 recovery of original96 chrF++ loss is (38.816713-38.186786)/(39.064691-38.186786) = 71.75% on that one direction only. my-zh is partial and not a gate; do not extrapolate. No end-to-end qualification or validation-based mask fitting. Existing repetition metric remains a limitation, not newly resolved. Prefix totals 3954 generated raw tokens, 1835 prompt tokens, prefill79062.068 ms, decode446559.077 ms, request wall526.740176 s; no performance claim from different generated lengths.

## R24 read-only preparation

Actual quantizer help and capacity job job_01M8E0FW0600HRNPF8D9CEHW74 exit 0; exact llama-quantize SHA256 a063c8dea6bf2a670f4436d68e39c040de2c504b01e022ee6ea27a852915ecbd. Live disk free39164948480 bytes, original96 Q4 reference size13933995552 bytes, 10GiB reserve, remaining headroom14493534688 bytes at reference size. Actual source inspection job job_01M8E0GFS70084VFN16ZHYBJQB exit 0 confirmed dry-run skips output stream creation and quantization computations, estimates row sizes, and warns if importance matrix required.

Actual dry-run job job_01M8E0H17Q00NGVW3WEKM1P6B1 exit 0, exact quantizer binary pin, guarded96 F16 published path, CUDA_VISIBLE_DEVICES empty, one thread, nice15, bounded timeout, isolated temporary directory. Source file size unchanged and no new model written. The reported sizes are tensor payload estimates, not exact future GGUF file sizes or runtime memory.

| Type | Estimated tensor MiB | BPW | Interpretation |
|---|---:|---:|---|
| F16 input |43820.07|16.0066|Existing 45953803936-byte GGUF |
| Q4_K_M |13283.60|4.8522|Approx 13.93 GB payload |
| Q6_K |17994.35|6.5730|Approx 18.87 GB payload |

Q4 log SHA256 cfdf490839f9d67a9e2c6f7cf897e1932d45b8138ae873179579b28fd2ce0c9e; Q6 log SHA256 dd703131f7e45b0b2cc73d6f77cb9dad9831ebd052cf7c10dec168d3f60a04f1. Logs were temporary, exact extracted output retained in durable Gateway receipt; no claim that raw temporary log files remain. No imatrix-required warning in extracted output. Q4 may fit V100 with overhead subject to actual runtime load and KV context; Q6 tensor payload exceeds V100 physical VRAM before overhead and would require hybrid placement or another GPU. Do not infer deployability from dry-run.

## Conditional R24 execution contract preparation

After R22 full quality passes (or R23 replacement passes), pin accepted candidate identity, complete source hash, quantizer binary and dependencies, same mask/expert/tensor descriptor identities. Preflight exclusive resources and disk capacity including metadata/alignment and 10GiB reserve; publish new candidate atomically after actual output hash, metadata/tensor shape reader validation, and quantization receipt. Never requantize Q4 into Q6; both from qualified F16. No quantization launched in this batch.

Use native /completion frozen greedy protocol and exact held-out validation IDs. First six-direction plus repeat pilot; then full120 if admitted. Separate paired comparisons: (1) 128 F16 vs accepted96 F16 (pruning), (2) accepted96 F16 vs same-mask Q4 with same hybrid placement (quantization), (3) same-mask Q4 hybrid vs full GPU (placement), (4) final full GPU Q4 vs 128 F16 end-to-end gate. Reuse existing baseline where exact config matches, never substitute original96 Q4 artifact. Actual memory/load/context/timing measurements and token reproducibility must be captured. Same per-direction numerical/diagnostic gates remain, not a promise Q4 will pass. Q6 fallback only if justified and feasible after Q4 evidence. No further GPU/large IO task while R22 active, no validation fitting, no final-test or merge. R23 remains conditional on R22 results; canonical mechanism HOLD.
