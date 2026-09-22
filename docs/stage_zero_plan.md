# 新架构 Stage 0 条件（尚未实施）

本页从[当前 Local Plasticity 架构原文](canonical_architecture.txt)提取 Stage 0 条件，只用于准备新的实验。旧 e-prop Stage 0 已[归档](../archive/stage_zero_legacy/README.md)；其程序、指标和 NOT SUPPORTED 结论不能自动转用到本阶段。

- 总共 10 个 Block，每个强制 100 个神经元；所有 Block 始终 active。
- 强制关闭 route、goodness、ear、speak，只保留 eye、hand。
- 输入为黑底模拟图像，包含 26 个字母、红色与蓝色，以及全绿屏幕；eye 仍按 1080p 图像接口输入。
- hand 只允许 26 个字母键与鼠标左键 click 的最终离散输出；最终信号只写机械日志，不真实控制设备。
- 外部读取机械日志：输出与图片对应字母时，或全绿屏幕 click 时，提供 `g*=1`；其他情况为 `g*=0`。goodness readout 保持关闭。
- 27 个最终离散控制量分别输出连续倾向 `q`、采样独立 `Logistic(0,τ)` 噪声，再用硬阈值形成 0/1。26 字母之间不使用 Softmax，允许同一时刻多个键成立。
- 正式训练只使用各连接的局部塑性状态及延迟到达的唯一 `g_eff=g*`；不使用 e-prop、`∂q/∂θ`、硬阈值或 world 梯度，不要求 world reset，也不在同一状态尝试正反动作。

应记录连续倾向、噪声与最终动作、机械日志、局部状态保留及参数变化，并检验正确字母和全绿 click 的发生倾向是否逐渐提高。这些是未来实验的验收问题，本页不宣称结果。原文还允许用很小的四 Block e-prop oracle/benchmark 衡量局部规则的信息损失；该对照必须隔离，不能向正式训练提供梯度或额外 credit。
