# 新 Local Plasticity Stage 0 正式协议

最高优先级定义见[当前架构原文](canonical_architecture.txt)。本协议只检验当前 `acnt/plasticity.py` 的第一版 `CorrelationRule` 候选；旧 e-prop Stage 0 位于[归档](../archive/stage_zero_legacy/README.md)，其学习代码及结论不属于本实验。

- 10 个始终 active 的 Block，`neuron_size=100`、`hold_tick=4`、`ticktime=0.25 s`。Block 0 是 Eye ReadIn，Block 1 是 Hand ReadOut，2–9 是普通 Core。实验仅构造 EyeAdapter 和 27 离散量、零连续量的 HandAdapter。
- 1080p RGB FP32 黑底 5×7 内置字形：A–Z 各有红、蓝版本，对应同一字母键；全绿屏幕对应鼠标左键。字形规格、颜色、SHA256 和独立 RNG 种子写入配置及日志。目标标签只存在于环境侧。
- 每 episode 在逻辑时间 0/250/500 ms 进行三次新 Eye 输入及全部 Block 的 snapshot 更新；第三轮后采样一次 Hand。每一控制量独立使用 `q+Logistic(0,.25)>0`，允许多键与零键。
- 环境读取实际 27 位动作；仅当目标位是唯一激活位时 `g*=1`，否则 `g*=0`。外部 `g*` 是唯一 `g_eff`。没有 Goodness ReadOut、教师表、分级奖励或 baseline。
- 每 episode 独立抽取 250/500/1000 ms 延迟，阶段内均衡。动作到送达间没有 frame、动作、Block forward 或 local plastic event；`e_c` 原值保持。送达时仅调用当前 `F_w`。这三档只是实验控制变量，不改变 ACNT 时间定义。
- 训练固定当前 `CorrelationRule`：`learning_rate=.001`、`retention=.95`、`parameter_clip=None`；`e_c` 仅在真实局部事件调用 `F_e`。初始与冻结评价都不更新 `e_c` 或参数。
- 五个 seed 11/22/33/44/55，每个为初始 270、训练 1080、冻结 270 episode。类、颜色和延迟均衡；target/color/Hand/delay 的 RNG 独立。阶段边界清空动态状态和局部塑性状态，保留参数。

正式配置、source SHA256 与稀疏 reward 预检见[锁定文件](../reports/stage_zero_local_plasticity_lock.json)及[预检报告](../reports/stage_zero_local_plasticity_preflight.md)。活动实现与完整命令见[实验 README](../experiments/stage_zero/README.md)。raw 与 checkpoint 存在被 Git 忽略的 `runs/stage_zero_local/`；独立审计和紧凑报告在 `reports/`。

主结论只依据 exact success、target/non-target 概率、目标位命中、非目标误触及冻结表现。参数变化或非零 `e` 不构成学会映射的证据。规范允许的四 Block e-prop benchmark 是未来独立科学对照，本轮不执行。
