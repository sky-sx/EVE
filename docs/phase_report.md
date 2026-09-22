# 当前实现状态记录

日期：2026-09-22。架构依据为[用户当前 TXT 的逐字副本](canonical_architecture.txt)。此前 Phase 1–15 和 e-prop 验收数字属于旧版本，不是当前 Local Plasticity 架构的完成证明；旧 Stage 0 已[归档](../archive/stage_zero_legacy/README.md)。

| 部分 | 当前代码位置 | 已核对的边界 | 尚未证明 |
|---|---|---|---|
| Block/Core | `acnt/block.py`、`acnt/core.py` | FP32 状态、active 来源、一次更新、ReadIn 强制触发；原文 `gamma=sigma(delta_t*ticktime)` | 长时序稳定性与学习效果 |
| 六器官 | `acnt/adapters.py`、`acnt/runtime.py` | 六个不同 Block；readin/readout 分离；机械执行状态不回流 Core | 真实采集、DirectInput、声音合成 |
| 离散终端 | `acnt/control.py` | 各坐标独立 Logistic 噪声、硬阈值；不使用 Softmax | 行为是否获得有效长期定向 |
| Goodness | `acnt/runtime.py` | `g∈[0,1]`、`g*` 覆盖、B_g/A_g 的 `c_g` 局部校准 | 教师拟合质量 |
| Local Plasticity | `acnt/plasticity.py` | 参数元素局部状态，可替换 `F_e/F_w`；正式路径无 e-prop 梯度状态 | 哪种状态和规则能完成长期 credit |
| 新 Stage 0 | [协议草案](stage_zero_plan.md) | 原文列出 10×100、只留 eye/hand、外部 `g*` 与独立探索 | 尚未实现或运行 |

最近做过非 Stage 0 最小检查：1080p eye 与 ear 输入、六器官运行、局部状态有限、外部 `g*` 覆盖、goodness 开关和独立噪声路径。旧测试中仍有按 `EligibilityBank`、`feedback_seed` 或 `g_bar` 写的断言；不能把旧测试总数当作当前架构通过数。后续结果应以实际重新运行的日期、命令和产物记录，且与历史版本分开。

历史 Stage 0 的两次正式报告均为 **NOT SUPPORTED**。该结论说明当时协议未观察到映射学习，不是对当前局部塑性架构的实验结果。归档中的源码、测试、Teacher 表和报告保留作追溯，不应在当前 Runtime 下直接复用。
