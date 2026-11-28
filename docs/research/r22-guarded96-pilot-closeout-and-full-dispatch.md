# R22 Guarded96 physical publication and native pilot closeout

Status: REAL_BYTES_PUBLISHED; NATIVE_PILOT_COMPLETE; FULL_VALIDATION_DISPATCHED_NOT_COMPLETE. No final quality acceptance. Research source HEAD 26fb845739659faf74a086f206421f52fc542280; original R20 HEAD 3212303e1e02994a5124db8c85195b91325c7a7c.

## Published artifact and evidence

Materialization job job_01M8DZ3TCB00ESYBATRTR03S4A succeeded exit 0; verified actual source bytes, 766 tensor payload hashes, and GGUF reader. Model 45953803936 bytes, SHA256 4b9df0b0382e22c7cc613fd06b50911b3aae22996073184affcbf4128516f41f. Frozen P12 mask SHA256 4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c. Live probe job job_01M8DZPSDA00GFVR0GJFM7BX2G exit 0 confirmed bundle receipt SHA256 982988a75c517bf2d4a2e31685fea5f344d8c738144ecd2af586ad776cbb9c9d. Admission #1 comment 7212864536.

## Actual native pilot

Job job_01M8DZQJMG004H0R8ZJNJF9PCK succeeded exit 0, eight EOS requests, six unique IDs, two exact primary repetitions. Pinned native driver blob 0cdca8d43df86722775aad3d391001c78872926f, protocol 85b3bbabeafa4b6bf318a2f8fb25e44900d5626f, scorer ee010f8e92e649dfca9b930c8d19c414d99e26e2. Same runtime and frozen native settings as retained 128 pilot, not the old v2 protocol.

| Direction | Native128 pilot | Guarded96 pilot | Delta |
|---|---:|---:|---:|
| zh-my |31.031863|34.045354|+3.013491|
| my-zh |8.083672|11.248499|+3.164828|
| zh-en |45.813462|45.813462|0|
| en-zh |37.198220|37.198220|0|
| en-my |50.811083|50.811083|0|
| my-en |48.005352|47.345034|-0.660317|

Metric signature nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0. One unique sample per direction, no bootstrap or gate. Existing degenerate repetition flags for zh-my and en-my unchanged; no new digit/empty/truncation flags. No general quality improvement claim. Candidate journal SHA256 3650e8fbf1fdf4869c44deacd43b9c7b19ec955bd3a4f008401666812ef88f99, plan SHA256 fea860e330460944273c9658097dc9367b2d556faa723a4db02223cf3fda4fae. Baseline journal SHA256 7cdad9f3efac24261b85b520979189f37beb366041afc62a765123bb973179f2. Independent read-only job job_01M8DZWFHZ00GEJNX549690JJN exit 0 confirmed all 8 log slices against master log, exact repeated raw tokens/text, nonempty EOS, and complete receipt SHA256 d6f7c11735f698fbfa3ae12256a337e685f0b0728f831d30b5ee69cec50eb682.

## Full validation admission and dispatch

Owner continued and allowed unused cleanup. Admission issue #1 comment 7212917572. Actual job job_01M8DZXJSX00VRT8NSR3D86469 running, resource preflight no blockers, free 39176216576 bytes, 120 frozen validation IDs, manifest SHA256 b4fd031ae37e5fff2ac50d7cd7b098e5c2a7c4a11f23a7420c983a3eb80098d9. Before load exact model full hash required; existing 128-full baseline journal a4eb8d2adc9d446139449d11571359ea28d2a7a959a1dc644e5e6cab16da7ea3 and pilot journal pinned. Six pilot/full exact token/text comparisons before full scoring. No automatic further candidate launch. Existing per-direction maximum one-point chrF++ loss and no increase in digit/empty/truncation counts unchanged. Original96 full failed; this new candidate's full results are pending.

Actual lightweight inventory job job_01M8DZYQW800EDQ38ZS0NNV0E1 exit 0 showed 39175335936 bytes free and active GPU PID 529558 consuming 10546 MiB (10549 MiB total used). Preserved all existing models: source128 F16 60159655456 bytes, original96 F16 45953803808 bytes, original64 F16 31747952288 bytes, original96 Q4_K_M 13933995552 bytes, Guarded96 F16 45953803936 bytes. No deletion needed or performed. The existing Q4_K_M artifact is original96, not Guarded96; similar expected new Q4 storage is only an estimate, not an actual Guarded96 quantization or deployment result.

## Conditional next stages

Use existing R21–R27 plan rather than introduce duplicate task IDs. After full validation and independent raw log audit: if Guarded96 passes the full matched quality gate, advance to R24 matched quantization and deployment checks (R23 optional diagnostic work, if needed); otherwise run the previously prepared R23 bounded calibration-only sparse real pilot with exhaustive first-layer oracle, fit/holdout separation and no validation-based mask fitting. Preserve all evidence and source models. No simultaneous heavy workloads, training, final-test or merge. R25 lower budgets remain conditional on qualified96; canonical mechanism and final-test remain HOLD.
