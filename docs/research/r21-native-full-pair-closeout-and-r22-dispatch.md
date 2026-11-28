# R21 native full pair closeout and R22 materialization dispatch

Status: ORIGINAL96_FULL_QUALITY_GATE_FAIL; GUARDED96_REAL_MATERIALIZATION_DISPATCHED. This is not a candidate publication or runtime qualification. Owner continued execution and authorized cleanup of unused artifacts if required. No additional cleanup needed or performed.

## Actual completed deterministic pair
Original finite job job_01M8DVGG550050NBY708JNDKXZ reported control API 500 despite exit0, so do not claim that Gateway job succeeded. Actual read-only server probe job job_01M8DYXQJB005TFYNJX9B30FRQ exit0 at 2026-11-27T15:36:15+08:00 confirmed both completed120-row journals, complete receipts and GPU idle. Independent audit job job_01M8DYYR7R0091322YWNM0JFXM exit0 verified all256 pilot/full rows, master-log slices and actual effective parameters, all EOS/zero empty/zero length, both pilot/full six unique sample IDs exactly reproduced (12/12). Full128 journal SHA256 a4eb8d2adc9d446139449d11571359ea28d2a7a959a1dc644e5e6cab16da7ea3; full96 6ce9ad6e2afb9bc8edf34409864fd4372b6b7057d9c941420af09b03783a1a92. Exact pinned full scorer job job_01M8DZ117400MW1KCB808CTJ7E exit0; SacreBLEU2.6.0 nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0. Existing per-direction maximum1-point loss and no increased empty/truncation/digit flags.

|Direction|128|Original96|Delta|Digit flags128→96|Gate|
|---|---:|---:|---:|---|---|
|zh-my|39.064691|38.186786|-0.877905|3→3|PASS|
|my-zh|21.430077|19.406907|-2.023170|1→2|FAIL|
|zh-en|58.625453|57.620952|-1.004501|3→3|FAIL|
|en-zh|33.206154|31.724528|-1.481626|1→1|FAIL|
|en-my|43.189511|42.623970|-0.565541|0→2|FAIL|
|my-en|52.978995|52.537493|-0.441502|1→2|FAIL|

Only one of six full gates passes. My-zh paired bootstrap CI [-3.269260,-0.608312] (2000 resamples seed42); small samples and digit diagnostics remain limitations. No validation-based mask fitting. Full128 request wall1731.618s, prompt14428 tokens/544796.113ms, decode10018/1180150.380ms. Full96 wall1602.171s, prompt14428/448947.313ms, decode10112/1146790.202ms. Request wall reduction7.48%; prefill and decode work differ, no isolated GPU speed claim. Quality fail takes priority. Old protocol results exploratory. Canonical mechanism and final-test HOLD.

## R22 actual resource admission and dispatch
Research evidence HEAD db1fedc1e27c33558e42e2ed754819fd14d7e57c; issue #1 comment7212684171. P12 exact canonical mask 4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c, original128 F16 source SHA c0f31f9243f2be7c67971b3195606bbb5ed4fa9ceb035add2520480b67f38e5b. Use previously tested standalone driver blob331638736bf7a4efb02c2003dd6c6b2f0cbc84d3, source and dependency pins. Actual finite job job_01M8DZ3TCB00ESYBATRTR03S4A started; preflight no blockers, free85132255232 bytes, expected45953803936 output, reserve10737418240, shortfall0. Exclusive admission retained at ~/.cache/mt-pruning/r22-admissions/guarded96-f16.json. Actual server probe job job_01M8DZ5VCA008QB54MTX60FZ5J exit0 at15:40:52+08:00 verified private unpublished staging directory and Python driver PID527785, no GPU process. Source full SHA check, output write, payload hashes, reader validation, safe publication pending; do not treat staging directory or dispatch as finished model. No extra HF intermediate, no new deletion, no other GPU or heavy CPU tasks. Earlier combined continuation submission was blocked; this uses existing individually admitted materialization driver only. Once actual verified publication receipt available, verify runtime and finite native pilot against frozen baseline, then decide full candidate validation or R23 actual calibration capture. No automatic full inference, lower budget, training, final test or merge. R23 sparse pilot integration synthetic evidence retained at prior HEAD95d9d74c42d0080610decab6c553c5834e8a53e8; not real capture.

Machine-readable complete pair: r21-native-full-pair-results.json. No user intervention currently required.
