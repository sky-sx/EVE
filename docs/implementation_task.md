# 当前 ACNT 实现任务与验收边界

依据：[Local Plasticity 版架构原文](canonical_architecture.txt)。本文是工程任务清单，不改变原文，也不把历史 Stage 0 的 e-prop 代码接回正式运行路径。

## 已有运行骨架

- `Block` 有 `active/neuron_size/ticktime/hold_tick/z/a/o/A/At/W_ij/b/r/W_c/b_c`，并执行原文的一次更新公式。
- `Core` 对 active 来源求和，支持自连接、历史与按时调度；ReadIn 新样本可强制更新一次。
- 六个不同 Block 绑定六种器官 Adapter；eye 接收 1080p 张量，ear 接收原始音频窗口，hand/route 生成独立离散控制，speak/goodness 生成连续输出。
- 控制终端实现逐坐标 Logistic 噪声与硬阈值；执行开关关闭时只写机械日志。
- `Plasticity` 为参数元素维护局部状态，收集各连接真实 pre/post 活动；有 `g*` 时 B_g/A_g 用 `c_g` 校准，其他连接只收 `g_eff`。默认 `CorrelationRule` 是可替换实验候选。

以上说明代码路径，不宣称已学会任何任务，也不宣称 `F_e/F_w` 已找到最佳形式。

## 本轮完成的核验

- 已将归档外活动测试迁移到当前 Local Plasticity API，覆盖数值 F_e、事件计数 retention、延迟 F_w、参数注册、无 autograd 学习路径、goodness_id 可选和完整 Runtime 回归。测试总数以本轮实测为准。
- 已建立[当前 Stage 0](../experiments/stage_zero/README.md)，此前按[锁定配置](../reports/stage_zero_local_plasticity_lock.json)完成 strict-reward 五 seed 实验；其[报告](../reports/stage_zero_local_plasticity_report.md)仅适用于旧 `g*`。当前 `1/n` 定义尚未正式运行。

## 尚待实现或核验

1. 接入真实截图/音频采集、键鼠 DirectInput 执行和约 30 维发声器官参数到声学几何的映射时，保持相同感知与机械边界。
2. 准备架构原文列出的长音频、音画资料、题单、Teacher VLM、虚拟机与发声器官工具；题单/Teacher 文本若进入 ACNT，必须先转成图像或语音。
3. 未来另行建立隔离的四 Block e-prop 科学对照系统。对照结果只用于研究、诊断和候选选择，不能进入正式参数更新。

## 验收原则

核心验收必须分别检查运行可执行性、局部规则是否遵守信息边界、机械日志与执行开关、以及在单条时间轨迹上的学习表现。参数发生变化、一次 mock smoke 或旧 Stage 0 的测试通过数，都不能替代学习证据。旧 Stage 0 已[归档](../archive/stage_zero_legacy/README.md)，其结论只针对旧实现。
