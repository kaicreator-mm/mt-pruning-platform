# Current demo language scope — owner direction

Owner instruction: 不要考虑其它语言，目前只考虑中英缅为主，特别是中缅。

## Scope and priorities
- Current demo is limited to Chinese (zh), English (en), Burmese (my). Chinese↔Burmese is the primary translation pair, in both directions. English↔Burmese and Chinese↔English remain supported secondary pairs. No fourth language, language-extension experiments, general multilingual platform, or new direction-profile cache implementation in current scope.
- For future quality analysis and draft stress cases, prioritize Chinese↔Burmese: everyday conversation, trade/logistics, technical terms, quantities/units, names, dates, negation/conditions, long messages, and protected placeholders. These are proposed initial coverage areas, not new product requirements or acceptance thresholds. Native Burmese review is needed before draft cases become reference benchmarks.
- Existing six-direction equal-weight experiments, frozen corpus, quality thresholds, source identities, and R20-R1 task remain unchanged. Do not retroactively reweight results, relax secondary-direction gates, change the active writer, or use final-test data. Future changes to selector weights or acceptance gates need explicit scoped experimental decisions. Prior P07 research is retained as historical evidence; fourth-language follow-up is deferred.
- Reuse P04 existing uniform masks and existing stress utilities rather than expanding the algorithm/platform surface. Select next real candidate after R20-R1 actual quality/capacity. In particular, examine zh-my and my-zh separately; no aggregate average may conceal their regression.

Status: owner scope recorded; no new inference, no task dispatch, no implementation or acceptance decision.
