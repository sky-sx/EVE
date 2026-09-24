# 当前实现与 Local Plasticity Stage 0 结果

日期：2026-09-22。最高优先级依据是[当前架构原文](canonical_architecture.txt)。起点 main 为 `a7c72597bdbac10819a88c802b3cb25b1b9a6c83`。旧 e-prop Stage 0 的程序与两份 NOT SUPPORTED 报告保留在[历史归档](../archive/stage_zero_legacy/README.md)；下面的数字属于此前 strict exact-one-hot Goodness 的 Stage 0 正式运行，不能作为当前 `1/n` Goodness 的结果。

> **适用性边界：** 本报告中的全部 Stage Zero 长训练数字使用
> pre-correction Block temporal implementation。它们不能外推为 corrected
> synapse mixing + CTM per-neuron private NLM + real-time age Block 的学习结论。

| 部分 | 当前代码位置 | 本轮核验 |
| --- | --- | --- |
| Block/Core | `acnt/block.py`、`acnt/core.py` | 当前代码为 synapse mixing + A/At + per-neuron real-time NLM；表中长训练结果仍来自修正前实现 |
| Eye/Hand | `acnt/adapters.py`、`acnt/control.py` | 只构造 Eye 与 27 位 Hand；红/蓝 A–Z 和全绿 1080p RGB；独立 Logistic 噪声与硬阈值 |
| Local Plasticity | `acnt/plasticity.py` | 原样使用 `CorrelationRule` 数学；Stage 0 的 `goodness_id=None` 让 10 组都接收外部 `g*`；无本地事件时 `e` 不变 |
| 新 Stage 0 | [活动代码](../experiments/stage_zero/README.md)、[锁定配置](../reports/stage_zero_local_plasticity_lock.json) | 此前 strict-reward 五 seed × (270 初始 + 1080 训练 + 270 冻结) 全部完成；8100 行 raw 独立审计通过。当前 `1/n` 外部 Goodness 尚未正式运行 |

[此前 strict-reward 正式报告](../reports/stage_zero_local_plasticity_report.md)包含每 seed、每类、颜色、延迟、训练窗口、局部状态、参数变化及全部 source SHA256。CPU/CUDA 极短 smoke 和性能 smoke 都通过，正式运行选择 CPU。活动测试已从旧 e-prop API 迁移，最终全量测试结果见报告及本轮提交记录。

此前 strict-reward 行为结果：初始/训练/冻结 exact success 分别为 `0/1350`、`0/5400`、`0/1350`，所有 `g*=1` 计数为零。目标平均 `p` 从 0.505055 到 0.496885，非目标平均 `p` 从 0.501505 到 0.496812。250/500/1000 ms 三档冻结 exact success 均为 0/450，目标平均 `p` 分别为 0.496042、0.497009、0.497604。没有观察到延迟对应的明显成功率差异。

训练时全部组形成有限、非零的 `e`，所有 10 个 Block 及 Eye/Hand Adapter 参数都变化，NaN/Inf 为零；送达时三档 delay 的全部训练 episode 均保有非零 `e`。这些属于机制运行证据，不能替代行为学习证据。此前运行的结论为 **NOT SUPPORTED**。当前 Stage 0 仅将外部 `g*` 改为目标位激活时 `1/n`（`n` 为激活 Hand 位数），其他机制与预算不变；本轮尚未运行正式实验，因此没有新的行为结论。四 Block e-prop oracle 仍是未来独立对照。
