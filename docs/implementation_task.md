# 当前 ACNT 工程缺口

唯一规范见 [canonical_architecture.txt](canonical_architecture.txt)。普通 Block / Adapter 参数已统一交给 Persistent ΔW Plasticity；Goodness designated group 排除。Block 仅负责神经动力学；学习在 forward 前进行稀疏真实参数运动，在 Goodness 后决定未来方向。

1. 在固定 source hash、独立种子和短预算预检基础上，正式验证 [ABC Stage Zero](stage_zero_plan.md)。单元测试和参数变化不证明行为学习。
2. 研究连续 Goodness 差值在输入变化、随机动作和世界变化下的混杂因素，以及长期稳定性与固定步幅适用范围。
3. delayed Goodness 当前仅允许一个未评价 trial；overlapping trials 尚未定义，不能绕过 unresolved 检查。
4. 接入真实截图、音频与键鼠执行；机械结果不得返回 Core。Speak 到声学几何的映射仍待接入。
5. 通用 checkpoint 恢复仍需独立设计；ABC checkpoint 保存方向、学习 RNG、参数和诊断用于检查，不声称能完整恢复世界轨迹。

不升级版本号，不增加其他 production learning mode。历史实验代码和数字仅供历史审计。
