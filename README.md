# Hy-MT3 专家剪枝技术可行性验证（Demo）

目标（见 [issue #1](https://github.com/kaicreator-mm/mt-pruning-platform/issues/1)）：先验证技术，不建设正式平台。
首个语言组合 zh / en / my，六方向分别报告，zh→my 优先；不训练、不修改 tokenizer、共享专家或 top-k。
GitHub 是事实源；采用有限实验、精确提交证据、失败闭锁。

## D01a：合成张量上的机制验证

[`d01a/`](d01a/) 是 D01a 的源码、说明和原始输出，原样保留。结论是 `PASS_SYNTHETIC_MECHANISM_ONLY`：
在 CPU 合成张量上验证了方向均衡的专家选择、物理裁剪、保存重载和负例，不涉及任何真实模型或真实翻译。
`demo.py` 的 sha256 为 `3d348279581ca15207a850e997aa82049a6b18ec775785bf08cb244fd8397bf6`，
与 issue 中记录的源码哈希一致。

D01b（真实权重验证）在分支 `d01b-real-hy-mt3-demo` 上进行。
