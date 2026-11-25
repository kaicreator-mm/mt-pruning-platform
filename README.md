# Hy-MT3 专家剪枝技术可行性验证（Demo）

目标（见 [issue #1](https://github.com/kaicreator-mm/mt-pruning-platform/issues/1)）：先验证技术，不建设正式平台。
首个语言组合 zh / en / my，六方向分别报告，zh→my 优先；不训练、不修改 tokenizer、共享专家或 top-k。
GitHub 是事实源；采用有限实验、精确提交证据、失败闭锁。

- **D01a（仅证明机制）**：CPU 合成张量上的方向均衡专家选择与物理裁剪。代码与原始输出在 [`d01a/`](d01a/)，
  结论是 `PASS_SYNTHETIC_MECHANISM_ONLY`，不含任何真实翻译。
- **D01b（真实权重验证）**：本仓库其余部分，见下文。

**D01b 目前没有真实模型结果。** 权重还在下载，真实校准尚未开始；测试集也从未对任何候选运行过。

## D01b 的流程

| 步骤 | 内容 |
|---|---|
| 模型 | `tencent/Hy-MT3-30B-A3B`，固定 revision `08ae9acea91abe265c6236e757e49d8851e1ce06`；每个分片重新计算 sha256 并与固定清单比对 |
| 语料 | FLORES-200（CC-BY-SA 4.0）。校准取自 dev（24 句/方向）；验证和测试是 devtest 中两个不相交的子集（各 20 句/方向）。公开的 FLORES-200 没有 test，所以两者都来自 devtest |
| 语料合同 | 每个阶段先核对语料字节与已提交的清单，再交给已评审的 `r7/handoff.py` 检查（精确 ID、行号、R8 重复规则）。不一致即 `BLOCKED_CORPUS_CONTRACT`，绝不重新抽样 |
| prompt | 模型卡上的 Default Translation 模板；源语言是中文用中文 prompt，否则用英文 prompt；官方 chat template |
| 记录 | 每个样本的输出、路由统计和 mask 审计一起提交到 SQLite 日志（已评审的 `r4/trace.py`）。所有聚合都从日志重建，中断后恢复不丢数据，完成的运行不会被覆盖 |
| 运行身份 | 日志绑定模型内容、分词器、语料、prompt、mask、运行时、代码和解码参数；身份不同的运行不能续写同一份日志 |
| 专家选择 | 仅用校准集的路由：每层每个方向的路由权重占比取六方向平均，保留 top-K。选择前先用 `r6/bridge.py` 核对聚合覆盖 |
| 物理裁剪 | 写出全新的 checkpoint：router 行、expert bias、专家矩阵按新索引落盘，其余张量原样复制；逐张量与源对比后再重载 |
| 机制检查 | 128 专家恒等重载 = 原模型（no-op）；物理裁剪 = 在原模型上屏蔽被删专家（masked）。在验证集句子上逐位置比较，这些句子不参与选择 |
| 候选评估 | 候选只在验证集上运行。临时决策由 `r7/handoff.py` 给出（通过所有方向的最小候选，否则回退到原模型） |
| 最终测试 | **暂缓（HOLD）**：只有经过独立评审的决策才能运行 test，而且只运行一次 |

质量门槛：每个方向 chrF++ 下降不超过 1 分，且空输出、截断、数字不匹配的数量都不多于基线。判定统一走
`r6/bridge.py` 的独立门槛；记录不完整（缺方向、重复、多余、组不满）的候选记为 NOT_EVALUATED，永远不会被判为通过。

贪心解码。模型卡建议的采样参数（temperature 0.7）不用于本实验，否则剪枝前后的差异会和采样噪声混在一起。

### 冻结语料

`evidence/d01b/corpus/corpus_manifest.json` 记录每个源语言在各分割的行号、文件哈希和抽样规则；仓库不提交语料正文。
用同一个 tarball 和种子重建，结果逐字节相同（corpus sha256 `9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4`）。
抽样时排除了跨分割或在样本内重复的句子（三种语言任一、NFKC 归一化后比较）；FLORES-200 里这类重复为 0。
语言代码是 FLORES-200 的 `zho_Hans` / `eng_Latn` / `mya_Mymr`，不要和 FLORES+（`cmn_Hans`）混用。
后续任务直接复用这份冻结语料，不要回溯修改。

### 运行时

`mtpruning/engine.py` 是一个 CPU bf16 推理引擎，逐算子镜像 transformers 5.24.1 对该模型的默认执行路径
（eager attention、sigmoid + expert-bias 路由、默认的 `grouped_mm` 专家路径）。唯一的不同是专家权重保持为
mmap 只读视图，而不是在堆上拼成融合的 3 维张量——60 GB 的 checkpoint 因此可以靠页缓存运行在 64 GB 内存的机器上。
`tests/test_engine.py` 在随机权重的 checkpoint 上要求本引擎与官方实现的 logits **逐位相同**。

注意：`modeling_hy_v3.py` 里可见的逐专家循环（bf16 逐个累加）与默认的 `grouped_mm` 路径（fp32 求和后只舍入一次）
在末位上不同，两者都是官方路径。本引擎镜像默认路径，并用容差测试覆盖另一条。

## 运行

```bash
# 1. 权重：直连 huggingface.co，断点续传，逐文件校验（约 60 GB）
bash scripts/download_hf.sh manifests/hf_manifest.tsv <snapshot_dir> 3

# 2. 环境（Python 3.13，依赖版本见 requirements.txt）
bash scripts/setup_env.sh

# 3. 测试（端到端部分需要已下载的 tokenizer 文件和 FLORES tarball；临时文件请放到磁盘上，不要用 tmpfs 的 /tmp）
MTPRUNING_TOKENIZER_DIR=<snapshot_dir> MTPRUNING_FLORES_TARBALL=<tarball> TMPDIR=<disk dir> \
  python -m unittest discover -s tests -t . -v

# 4. 全流程到临时决策为止，不运行 test；中断后重跑会从已有输出处继续
bash scripts/run_d01b.sh

# 5. 最终测试：仅在决策经过独立评审之后
FINAL_TEST=1 REVIEWED_DECISION_SHA256=<decision.json 的 sha256> REVIEW_REF=<评审引用> bash scripts/run_d01b.sh
```

各阶段也可以单独调用：
`python -m mtpruning corpus-check | checkpoint-check | run | select | prune | equiv | stock-check | decide | final-test | report`。
`run` 拒绝 test 分割，它只能由 `final-test` 运行；`final-test` 需要已评审决策的哈希和评审引用，并把使用记录写进
`test_exposure.json`。如果 test 已经对某个候选运行过，再对另一个候选运行会被拒绝。

## 目录

```
d01a/            D01a 合成验证（原样保留，demo.py 的 sha256 与 issue 记录一致）
demos/followup/  已评审并合并的离线辅助工具（R1–R8），本分支只读取其中的 r4–r7
mtpruning/
  engine.py      CPU bf16 推理引擎、mask 屏蔽、路由 trace
  identity.py    checkpoint 来源与内容绑定、运行身份
  frozen.py      冻结语料合同
  calibration.py 逐样本路由统计、mask 审计、聚合重建、方向均衡选择
  prune.py       物理裁剪写出与逐张量校验
  evaluate.py    chrF++、失败证据、trace 比较
  pipeline.py    各阶段命令行
  report.py      汇总 report.json / report.md
  reviewed.py    按文件加载 demos/followup 中已评审的辅助模块
tests/           单元与负例测试、对照官方实现的测试、小模型上的完整协议演练、驱动脚本测试
scripts/         下载、建环境、驱动
manifests/       固定 revision 的文件清单与哈希
evidence/        运行环境、语料清单与回执、各阶段产物和报告
```

## 不包含

最终测试（暂缓，等待评审）、80 专家/层、同规模随机剪枝对照、7B 稠密模型对照、GGUF 同量化、16 GiB GPU 上的峰值与速度、
人工严重错误抽查，以及生产验收级别的样本量。小样本（每方向 20 句）的 chrF++ 只能说明趋势，不能宣称生产可用。
