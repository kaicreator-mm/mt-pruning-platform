# R21 actual native deterministic 128/96 pilot acceptance

Status: REAL_PILOT_PROTOCOL_PASS; NOT_FULL_QUALITY_QUALIFICATION. Owner completed prior writer handoff and authorized direct V100 execution. Verified handoff comments #1 7204023935/7211531504 and actual zero GPU/model processes job job_01M8DV3BJ100P0E516CBWNAZER exit0. Prior PR #4 HEAD 3212303e1e02994a5124db8c85195b91325c7a7c unchanged; prior old-protocol291 records and unjournaled raw evidence remain exploratory. Admission #1 comment7211579974. Exact research execution source 89fdb5aedc37bf52489b8f8f7cbdaa8c33a3711a.

Actual Gateway job job_01M8DV4DH300C1K89885YVNCXQ exit0: serial original128 and original96 FP16 GGUF full source rehash before load,9 pinned binary/library hashes, matched native /completion, frozen prompt IDs, temperature0/top_k1/top_p1/min_p0/seed0/max640/cache_prompt=false, single server slot, existing hybrid n-cpu-moe40. Eight requests each: six original validation witness IDs, two repeated Chinese↔Burmese samples. All16 actual responses pass effective-settings/token counts/terminal EOS/raw sampler+token log checks. Both repeats per model match raw tokens and text exactly. Actual pinned scorer exit0 rechecked raw request/response/log capture digests, journal and frozen inputs. Pilot one sample per direction (not release gate):

|Direction|128|96|Delta|
|---|---:|---:|---:|
|zh-my|31.031863|34.045354|+3.013491|
|my-zh|8.083672|9.195011|+1.111339|
|zh-en|45.813462|45.813462|0|
|en-zh|37.198220|37.198220|0|
|en-my|50.811083|50.811083|0|
|my-en|48.005352|54.272968|+6.267616|

Existing degenerate_repetition heuristic flags both Burmese-target samples in both models; no new digit/empty/truncated flags. No statistical/general quality inference from six examples. All pilot gates deliberately null. Do not use validation to fit expert identities. Actual timing across8 requests each (includes repeats, excludes model load):128 wall89.071638s, prefill799 tokens/38331.968ms, decode446/50354.884ms;96 wall82.136247s, prefill799/31130.524ms, decode453/50647.304ms. Both8 EOS,0 empty. Small pilot does not prove deployment throughput or memory bound.

Evidence inventory read-only job job_01M8DVEH2300TMK9JVM1FB66GK exit0:29 files each with individual SHA256; retained at ~/.cache/mt-pruning/native-greedy-runs/{128,96}-pilot; separate exact admission files under native-greedy-admissions. Both journal and admission SHA verified; zero GPU compute processes after completion, free53529661440 bytes. Complete machine-readable results and file manifest r21-native-greedy-real-pilot-results.json. No new large model files, deletion, training, final test, CI or merge.

Decision: admit the same native protocol serial full120 validation each for128 and96 using existing models. Full comparison alone assesses the unchanged per-direction chrF++ <=1 absolute loss and no increase in empty/truncated/digit flags; keep canonical mechanism/final-test HOLD. Do not automatically run64 or new masks until actual full comparison; R22/R23 priority determined by the result. Full run uses all120 frozen IDs in order, including six pilot IDs; repeat their exact outputs as a separate reproducibility check when comparing evidence. No new user action required for this finite experiment. No further task auto-dispatch beyond full pair.
