# P09 — Tokenization-sensitive repetition diagnostic study
Source R19 3212303e1e02994a5124db8c85195b91325c7a7c, prior parallel head a1dd0c88388f349501e7aa51b92cb22ffef3d072. Active R20-R1 unchanged. No new model inference, GPU, final-test scoring or changes to existing evaluator/gate.

## Actual replay
Separate existing server aliyun-sg, standard-library Python, six exact public input Git blobs verified, frozen corpus SHA256 checked, exact120 baseline validation IDs and48 paired physical/masked witness IDs, shared run identity and row context checks. Temporary directory removed after execution. Pinned evaluate.py failure_flags/digit_runs/script_share extracted unchanged via AST; not full evaluator or journal replay. Job job_01M8CKJSTC00NAC43GJBHJRV6Z exit0, compile PASS, 11 synthetic checks PASS (including Myanmar repeated phrase, internal and trailing repetitions, short repeat, empty, unique sequence and exact span checks). Initial tool output page limited to65536 characters; retrieved complete output via get_job, no rerun. Script and all216 output-record diagnostics retained in JSON blob 5e1679a7c4e97853e2570ecec67cf5725971d4fc. These include repeated witness IDs across runs, not216 independent samples.

## Original heuristic vs exact tandem repeats
Existing flag: >=16 generated tokens and distinct tokens <50% of token count. Describes token diversity, not necessarily degeneration. Investigational repeat scanner checks exact consecutive repeated blocks, >=3 copies, periods1..32 tokens or1..64 NFC codepoints (whitespace collapsed), flag if repeated span >=24 units and >=half sequence or ends at sequence end. The 24-unit threshold and period limits are study defaults, not calibrated production thresholds. No semantic paraphrase, noncontiguous loops, longer-period repeats or grapheme-aware parsing. No human labeling. Synthetic controls demonstrate selected detectable cases only.
|Direction|Baseline median distinct-token ratio|Median tokens per normalized codepoint|Existing flag|Token tandem flag|Text tandem flag|
|---|---:|---:|---:|---:|---:|
|zh-my|26.62%|1.155|20/20|0/20|0/20|
|my-zh|90.96%|0.611|0/20|0/20|0/20|
|zh-en|91.81%|0.222|0/20|0/20|0/20|
|en-zh|94.39%|0.584|0/20|0/20|0/20|
|en-my|26.88%|1.154|20/20|0/20|0/20|
|my-en|88.32%|0.206|0/20|0/20|0/20|
Across baseline120 old40 vs token0/text0; physical96 witness48 old16 vs0/0; masked96 witness48 old16 vs0/0. All72 old flags are Myanmar-target records; no exact triple repeat within scanned periods found in any of216 records or corresponding reference texts. Counts do not establish that every translation is correct or that all flags are false positives. Character metric is codepoint-based, not Myanmar grapheme/word count. Observed tokens/codepoint and low diversity are compatible with tokenizer fragmentation, but no tokenizer-internal causal proof. Do not report a precision/recall estimate from unlabeled data. A natural repeated phrase may trigger proposed detector, and real degeneration may evade it.

## Recommendation and next boundary
Keep preregistered chrF++ and empty/truncated/digit-mismatch gates unchanged. Current repetition flag is already diagnostic only; present it as low token diversity, not a translation error verdict. Consider a supplemental contiguous-repeat diagnostic in later product evaluation after labeled native-language review and independently fixed stress samples; do not silently replace existing evaluator or reinterpret historical flags. No runtime/evaluator changes in P09. More synthetic tests or heuristic sweeps are not a substitute for actual translation quality evidence. P01-P09 parallel studies complete; next meaningful model decision depends on R20-R1 real measurements. No new local task, training, merge or qualification.
