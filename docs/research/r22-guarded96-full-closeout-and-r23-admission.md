# R22 full Guarded96 validation closeout and R23 admission

Status: FULL_QUALITY_GATE_FAIL; R23_FIRST_REAL_SPARSE_CAPTURE_ADMITTED_NOT_COMPLETE. Research parent 1a7a597ae6f849c9a770ec1175fabaa3062d30ca; original engine HEAD 3212303e1e02994a5124db8c85195b91325c7a7c.

Full native job job_01M8DZXJSX00VRT8NSR3D86469 succeeded exit0. Candidate 120/120, six pilot/full overlapping unique samples reproduced exact raw tokens and text. Pinned scorer returned FULL_QUALITY_GATE_FAIL. No change to model SHA256 4b9df0b0382e22c7cc613fd06b50911b3aae22996073184affcbf4128516f41f or frozen validation protocol. Existing per-direction maximum loss1 point, no increased digit/empty/truncated counts.

|Direction|128|Original96|Guarded96|Guarded delta|Digit flags 128→Guarded|Gate|
|---|---:|---:|---:|---:|---|---|
|zh-my|39.064691|38.186786|38.816713|-0.247978|3→3|PASS|
|my-zh|21.430077|19.406907|19.752771|-1.677306|1→1|FAIL|
|zh-en|58.625453|57.620952|56.890407|-1.735046|3→3|FAIL|
|en-zh|33.206154|31.724528|30.404332|-2.801822|1→1|FAIL|
|en-my|43.189511|42.623970|42.651650|-0.537861|0→1|FAIL|
|my-en|52.978995|52.537493|51.748854|-1.230140|1→0|FAIL|

Paired bootstrap 2000 resamples seed42, 95% intervals zh-my[-2.094618,1.199028], my-zh[-2.909280,-0.293866], zh-en[-4.079725,0.787588], en-zh[-5.393010,-0.728580], en-my[-1.544734,0.462666], my-en[-3.354759,1.172634]. Small samples limit conclusions. Repetition flags existing20/20 for target Burmese directions, unchanged; not resolved. Guarded96 improves both primary directions relative to original96 but worsens several secondary directions. Neither96 is quality-qualified.

Independent actual job job_01M8E2XCXF00WK6FPFGYB13N8P exit0: 8 pilot +120 full raw/effective settings/master log slices verified, all EOS, zero length/empty, 6/6 exact overlap. Both original128 and original96 full journals replayed. Full candidate journal SHA256 7ef433052cb2dd25d3d0949b7067c948b6433a269a8ccea491421908e615191e; master log 3f44bf9da2e8703f0285dbb8fef55b522fa7181af45af63722ea38a7eb8130dc; run plan d7c0906f2b1ba26cb207ee7d6b4fd5fbbc22099b2d74be7fcf3d1018084fac46; admission f93cb112c2e9cd5e1eb3d39819f8f094aca8a588ed315f158c268e6140d808ee. Total wall1665.845377s, prefill444953.241ms for14428 tokens, decode1214133.909ms for10100 raw tokens; no isolated speed claim.

Actual resource check job job_01M8E2XM1Y003S9J19JXWHNA7Y exit0, GPU idle, MemAvailable62774340kB, SwapFree3601592kB, disk free39102103552 bytes, exact engine HEAD and clean restricted paths. No deletion. R23 admission #1 comment7213914526, one previously frozen calibration-only sample zh-my:348 and five fixed masks, actual sparse driver blob19c8c851b1647cc0f495d6a7f1122219c42a0cdb, 47 sparse layers × four positions, first-layer same-state exhaustive oracle. Fresh full teacher identity before capture, live exclusive resources checked. This is an admission, not an assertion of capture completion. No validation-based fitting or new candidate, training, final-test or merge. R24/R25 conditional only; canonical mechanism HOLD.
