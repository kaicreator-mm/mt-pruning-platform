# Hy-MT3 专家剪枝技术可行性验证（Demo）

目标（见 [issue #1](https://github.com/kaicreator-mm/mt-pruning-platform/issues/1)）：先验证技术，不建设正式平台。
首个语言组合 zh / en / my，六方向分别报告，zh→my 优先；不训练、不修改 tokenizer、共享专家或 top-k。
GitHub 是事实源；采用有限实验、精确提交证据、失败闭锁。

- **D01a（仅证明机制）**：CPU 合成张量上的方向均衡专家选择与物理裁剪。代码与原始输出在 [`d01a/`](d01a/)，
  结论是 `PASS_SYNTHETIC_MECHANISM_ONLY`，不含任何真实翻译。
- **D01b（真实权重验证）**：本仓库其余部分，见下文。结果在 [`evidence/d01b/report.md`](evidence/d01b/report.md)。

## D01b 做了什么

| 步骤 | 内容 |
|---|---|
| 模型 | `tencent/Hy-MT3-30B-A3B`，固定 revision `08ae9acea91abe265c6236e757e49d8851e1ce06`，13 个分片逐一 sha256 校验 |
| 语料 | FLORES-200（CC-BY-SA 4.0）dev 做校准（24 句/方向），devtest 抽两个不相交子集做验证和测试（各 20 句/方向）；公开的 FLORES-200 没有 test，所以验证/测试都来自 devtest |
| prompt | 模型卡上的 Default Translation 模板；源语言是中文用中文 prompt，否则用英文 prompt；官方 chat template |
| 路由采集 | 原始模型逐方向跑 prefill 和 decode，记录实际选择的专家 ID 与权重 |
| 专家选择 | 每层每个方向的路由权重占比取六方向平均，保留 top-K（与 D01a 同一规则） |
| 物理裁剪 | 写出全新的 checkpoint：router 行、expert bias、专家矩阵按新索引落盘，其余张量原样复制；逐张量与源对比后再重载 |
| 等价性 | 128 专家恒等重载 = 原模型（no-op）；物理裁剪 = 在原模型上屏蔽被删专家（masked）；逐位置比较 logits、贪心选择和路由 ID |
| 质量 | 验证集和测试集上的 chrF++（sacrebleu，默认设置），每方向报告差值及配对 bootstrap 区间，附空输出、截断、数字、文字体系等失败证据 |

贪心解码。模型卡建议的采样参数（temperature 0.7）不用于本实验，否则剪枝前后的差异会和采样噪声混在一起。

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

# 3. 测试（需要已下载的 tokenizer 文件和 FLORES tarball 才会跑端到端那一项）
MTPRUNING_TOKENIZER_DIR=<snapshot_dir> MTPRUNING_FLORES_TARBALL=<tarball> \
  python -m unittest discover -s tests -t .

# 4. 全流程；中断后重跑会从已有输出处继续
bash scripts/run_d01b.sh
```

各阶段也可以单独调用：`python -m mtpruning corpus | run | select | prune | verify | equiv | stock-check | report`。

## 目录

```
d01a/            D01a 合成验证（原样保留，demo.py 的 sha256 与 issue 记录一致）
mtpruning/
  engine.py      CPU bf16 推理引擎、mask 屏蔽、路由 trace
  corpus.py      FLORES-200 抽样与 prompt
  calibration.py 路由统计、方向均衡选择
  prune.py       物理裁剪写出与逐张量校验
  evaluate.py    chrF++、失败证据、trace 比较
  pipeline.py    各阶段命令行
  report.py      汇总 report.json / report.md
tests/           单元与负例测试、对照官方实现的测试、小模型端到端演练
scripts/         下载、建环境、全流程驱动
manifests/       固定 revision 的文件清单与哈希
evidence/        运行环境、语料清单、各阶段产物和报告
```

## 不包含

80 专家/层、同规模随机剪枝对照、7B 稠密模型对照、GGUF 同量化、16 GiB GPU 上的峰值与速度、人工严重错误抽查，
以及生产验收级别的样本量。小样本（每方向 20 句测试）的 chrF++ 只能说明趋势，不能宣称生产可用。
