# R23C/D completion; R23F expanded-fit and conservative stable96 chain

Status: CD_COMPLETE_REPLAYED_NOT_QUALITY; NONUNIFORM_FIT_DIAGNOSTIC_MASK_FROZEN; EXPANDED_FIT_RUNNING; STABLE96_AND_PARTIAL_PROXY_CONDITIONAL_DISPATCHED. No quality-qualified new model.

## Actual CD completion and replay repair

Actual CD job job_01M8E7139400NYNQNQTJT00TPG succeeded exit0, all12 fit receipts, report SHA256 8e35997647591872f4c83f9dc7bfcae63052c604383bc1b2cb15bc8650eee63a, log871c07a0254964d159c7adf8b7d82357e421e101318a05706d497ef7f0ea3ad8, stderr empty. Initial expanded chain job job_01M8E7WA5800F5Y9TKJY7G1JP3 failed exit1 before new fit/DP publication, actual read-only probe job job_01M8E87T5500D6E1XWFSW7EJ90 exit0: replay assertion; stderr SHA3b9efa8d7c314843c85c0d91266a0e4188a7d2137f06e4135c99e81c04c61194, old stdout empty. Source emits groups tuple and JSON loads list. Canonical JSON equality preserves exact numeric values, order and content rather than container types. Repaired gate commit4eb0e72eabdf7272195c84417b0d5b331c6ce608 blob3bc231f40d9793454a40876dda81398747ffa6ef SHA256 ea950886c7cd13115e5594ecccec7be59a1f4731c6c1f8c6df5184d381c40857, actual six-negative roundtrip tests job job_01M8E89F7Y00MHT6524EX406AF exit0. Actual v2 job job_01M8E8AFXR00DC3E9W4Z31BCDX passed full canonical report replay, all12 pinned receipt hashes, exact predecessor and resource checks. Confirmed false negative, not numerical tolerance or changed evidence. Preserve original logs.

## Actual CD/DP summary

Actual read-only job job_01M8E8BPRT006HMVYW3K61J4T5 exit0, allocation SHA256 04c2238ac5bb47609bc0ed452f80690b4037c531c9f3dc33b93394c89bf6c6bc. Exact discrete DP for one additive fit objective; fit group gates subsequently pass; 4512 total experts across47 layers. Layer counts 64:7,80:12,96:3,112:24,128:1. Fit primary mean local error relative gain60.0591869867%, all-group mean relative gain51.1752553227%, no group violations, overall guard true. Actual original-centered budget means (primary/all):
- K64 .0463710990398726/.06805506580952798
- K80 .02898703655796258/.038791256053885
- K96 .01794313066065817/.022371999011367116
- K112 .0022330318132178363/.002884952668956446
- K128 1.7747579775887192e-8/1.763325195930204e-8 (numerical observer floor).
- swap_a .0122099249805461/.019219332424159897
- swap_b .016389462977252366/.019553123182985438
- swap_ab .01069486103479858/.016431478511044963

564 layer×direction-phase interaction cells, residual<-1e-8:11, >1e-8:23, near-zero:530;155 joint better than original,22 joint better than both singles. These are descriptive cells, not 564 independent samples or causal proof of semantic expert functions. The allocation is fit-only, not generalization or actual quality. Source commit e85994429acd845681ff54857bb4d67ee6d4b13f, actual mask freeze job job_01M8E8G3E900ZH02DB22Z451DM exit0, 2 negatives and actual frozen nonuniform96avg mask digest2339adfdbb8a438ecfa046384cced8f652fd1a523e722a66bf4480002a0500c9, receipt931e44f76ae47ba4c567b94335d7ffbb36f0cdf330bb20589ed55dd058983f17, file ~/.cache/mt-pruning/r23-results/nonuniform96avg-mask-v1.json. No physical GGUF heterogeneous representation support proven. Mask-only propagation diagnostic pending.

## Expanded fit and conservative uniform96

Expanded v2 admission SHA256 b629b515a65d0342e275c77e1bf2c2f8c43913bab92c24ce76ec3b5b953c6f87, issue admission7215761953. Fresh teacher identity exact; latest checked19/48 fit profiles (12 old +7 new), no new holdout profiles. Actual durable job job_01M8E8AFXR00DC3E9W4Z31BCDX running. Grouped expanded plan unchanged:48 fit/30 aligned lines,24 fresh/14 disjoint lines, original R19 calibration only.

Frozen stable96 method source02e6c69d08cfd5ca9611a2dc31fb8206c12de666 SHA256 355babd5551f604977edfad9048875a57f2b1192ed9abb5c6aaf7d22a4e68b6c, actual syntax+4 synthetic checks job job_01M8E8D7HD00H6N1QC164C1WX6 exit0. Equal per-example normalized route-mass and weighted-output-norm share mixture, group-balanced direction/phase primary x2, >=90% leave-one-aligned-line-out positive margin, both experts observed in >=2 lines, at most2 swaps/layer against original96, cumulative contribution share primary no loss, secondary<=2%, overall no loss. Not local reconstruction or translation acceptance. Only one predetermined method; no fresh outcomes used. Actual conditional lightweight job job_01M8E8FBEQ00VYN9ANJ2751VKT waits bounded3000s for full48 complete exact receipt inventory, then outputs expanded-stable96-v1.json, no model load/holdout.

Fixed-mask proxy driver source f616a79ffd5711f2d2fc01dbab136738dc59201e SHA256 ced20ef343a4c40f5db2b14faa09cb206901d17e0d2e061dffaa9e0e8916efe7, actual syntax+3 synthetic decision tests job job_01M8E8JAQ3001QPRS1WR0GT4T0 exit0. Actual conditional first24 fit job job_01M8E8M2MZ004EC74D4JMBCMDN dispatched under issue comment7215865589, dependency wait2100s overall3600s. Exact stable candidate byte replay required; skip if no swaps; live exclusive lock and fresh teacher identity before actual capture. First24 no final decision; subsequent24 only after pinned first receipts. Fit48 and fresh24 separate, fresh only after complete fit gate (>=5% primary improvement, primary nonregression, secondary<=2%, overall guard). Grouped aligned-line descriptive diagnostics; no independent CI claim. No physical build unless accepted and independently verified. All heavy tasks serial. If interrupted preserve partial immutable receipts and diagnose; never overwrite evidence or silently refit.

No DeepSeek/external LLM calls or secret access. No deletion, training, merge, final-test use or quality acceptance. Gateway capacity checked and current bounded tasks dispatched within 4 concurrent slots and 3600s maximum per job. No owner intervention needed.
