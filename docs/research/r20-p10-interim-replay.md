# P10 — Read-only interim R20 evidence replay

Status: READ_ONLY_INTERIM_REPLAY_NOT_QUALIFICATION. R20-R1 active; no writer changes, no new GPU inference, no final-test execution, no claim of final candidate acceptance. Exact R19 source 3212303e1e02994a5124db8c85195b91325c7a7c; research base e621f77659a13f6939ab57f2f12f81fd93fb0ca4. Through authorized aliyun-sg reverse SSH loopback tunnel, read only completed records and associated output files on CC-LAN200. Scoring executed separately on aliyun-sg, temporary pip --target with SacreBLEU2.6.0/NumPy2.5.5; actual pinned evaluator functions extracted without changing them. First replay job job_01M8CQGT2A00T108PK2P6W66VR exit28 (corpus curl timeout); bounded retry with increased download timeouts job job_01M8CQJ1S400V5GKCQXG2M0TPS exit0, py_compile PASS. Exact successful command retained, result JSON blob 77582393167fd8d0ff196a1dc7104838f80a0ceb. Temporary snapshot removed after replay; source output records and their prefix digests remain the underlying evidence, not an immutable uploaded snapshot.

## Observed live stage
At snapshot 2026-11-26T20:08:18.572718+00:00, 128/96 each 120 outputs, 64 113 outputs. All353 records have unique valid IDs within their run, zero command exit, actual output file SHA256 and raw text match, parsed assistant text matches journal; exact frozen120 validation IDs and recorded prompt token IDs match original corpus, all120 prompt files match manifest hashes, selected command options match expected FP16 KV/greedy/context2048/n-cpu-moe40. Verified frozen corpus blob/hash and evaluate.py blob. Manifest records pinned llama.cpp source; we have not independently rebuilt binary or verified actual processed token sequences for every request. Model file sizes stat-ed; author GGUF verification receipts captured and their declared byte sizes match. No full weights read or SHA recomputation (to avoid competing with active run).

## Observed translation results
All model scores use actual stored outputs; 128 and96 are complete validation runs. 64 has five full directions (20 each) and 13/20 my-en. For that partial direction its delta uses exactly the 13 matching 128 outputs (baseline 51.510), not the full baseline shown in the second column.
|Direction|128 FP16 full|96 FP16|96 delta|64 FP16 observed|64 matched delta|64 n|
|---|---:|---:|---:|---:|---:|---:|
|zh-my|38.596|38.416|-0.180|33.180|-5.416|20|
|my-zh|21.104|19.808|-1.296|14.292|-6.811|20|
|zh-en|58.550|56.999|-1.551|53.043|-5.507|20|
|en-zh|33.124|31.258|-1.866|24.217|-8.908|20|
|en-my|43.524|42.518|-1.006|31.814|-11.711|20|
|my-en|50.786|52.174|1.388|47.528|-3.982|13|
96 crosses the -1 point threshold in four directions (my-zh, zh-en, en-zh, en-my) and increases digit mismatch flags in three. 64 already crosses -1 in all five completed directions, including primary zh-my and my-zh. These are descriptive results; no new gate or test authorization. Focus primary pair separately.

|Direction|128 full digit flags|96 flags|128 matched to64 flags|64 flags|
|---|---:|---:|---:|---:|
|zh-my|2|3|2|3|
|my-zh|1|2|1|7|
|zh-en|3|3|3|5|
|en-zh|1|1|1|4|
|en-my|0|2|0|3|
|my-en|1|1|1|5|
No empty outputs among observed rows. These are reference digit comparison heuristics, not adjudicated semantic errors.

## Evidence gaps requiring R20 author attention
Follow-up read-only job job_01M8CQMGAE00RP5G65MW0HH7SB exit0 at 2026-11-26T20:09:37.884931+00:00: 128/96 enriched rows120 each; 64 raw117 rows still active. All raw and enriched rows inspected lack finish and generated token ID arrays, stop_type null. Pinned report.py blob 7e524c898dd49e1439e089e644bb4d03f83367a1 score_group directly indexes records[sample_id]["finish"] and ["tokens"]; observed full_quality.py sha256 3a216865b3f7bf89a0c83008a62ef2fdd982d8ef8376ed58415a7b29486f50a9 directly passes these rows. Unless adapted, that path cannot compute the complete existing flags and may fail with KeyError. Do not manufacture EOS or token arrays; actual stop evidence or explicit UNKNOWN/NOT_ASSESSED with fail-closed qualification is required. Re-tokenizing decoded text is not original generated token evidence. Server logs contain 118/120/117 release records respectively, with truncated=0 for first two server tracks and truncated=1 twice for the running64; this server flag must be distinguished from generation limit/stop reason. It is not by itself EOS evidence. Request author to reconcile IDs and stop evidence during current closeout, not to launch another full generation run without necessity.

128 execution modes 2 direct CLI +118 server, 96 120 server, 64 all observed server. Therefore shared precision and most command parameters do not establish identical serving/cache behavior on all120 rows; full aggregate timing should not be claimed as a controlled speedup. Client RSS is not model server RSS. Observed wall sums 128 1644.930s, 96 1581.052s, 64 1913.517s (incomplete); sampled device VRAM peaks 13341/10937/8471 MiB, respectively. Do not equate device-wide peaks with model-only allocations.

## Author diagnostic and artifacts (read, not independent mechanism rerun)
Author receipt diagnosis_summary.json: 141 selected same-input full/compact comparisons, no expert set/pre-sigmoid/sigmoid/choice mismatch or exact boundary tie, 138 slot-order and45 normalized-weight-by-ID mismatches, first selected hidden mismatch position21/model layer6, canonical sorting did not remove all combination discrepancies. Actual physical mapping sound per retained R19 checks; preserve canonical mechanism HOLD. The four GGUF verification receipts report PASS, 760 shape checks and zero mismatches; actual F16 sizes 60159655456, 45953803808, 31747952288 bytes. Q4_K_M96 author receipt 13933995552 bytes. Six-witness author comparison hybrid FP16 vs Q4 GPU aggregate wall speedup3.155x with two of six texts identical; precision and placement differ, not full quality or pure hardware speedup.

## Next boundary
Do not commit to k64 based on capacity; its observed primary-pair losses are already large. Do not treat original k96 as passing. Preserve active R20, finish the current evidence normalization/assessment, then choose a bounded same-budget alternative (existing P04 phase-balanced or robust96) or higher keep count if needed, with separate comparison to matched128. Owner scope only zh/en/my, prioritize zh-my/my-zh; no new language expansion, no additional speculative sweeps, no test/merge/training. No modification to active scripts, model files, running process or PR4.
