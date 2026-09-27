# Persistent ΔW Plasticity：ABC Stage Zero 协议

当前唯一活动协议为 [experiments/stage_zero_abc](../experiments/stage_zero_abc/README.md)，遵循[架构规范](canonical_architecture.txt)。

- 两个始终 active 的 Block，各 10 neurons、hold_tick=4、ticktime=250 ms，保留 snapshot synapse mixing 与 private real-time NLM。
- A/B/C 确定性 one-hot 输入直接进入 Block 0 的 o；Block 1 连接 3 离散量的 HandAdapter。每 episode 在 0/250/500 ms 三次正常更新，随后采样一次 Hand。独立 Logistic noise scale=0.25，threshold=0。
- 环境提供唯一 Goodness：目标位未激活为 0，激活时为 1/全部激活位数。exact success 是纯评估，不参与学习。没有 teacher、autograd 或额外监督路径。
- 训练使用生产 Plasticity：delta_magnitude=.001，subset_fraction=.001，plasticity_seed=seed+300000，parameter_clip=None。学习 RNG 与 target/Hand RNG 独立。
- 第一轮训练只建立真实 Goodness baseline，随后每 episode 第一个 forward 前随机稀疏移动一次；末尾 Goodness 消费此 subset。坏结果反转 ΔW、不回滚 W。
- 本 ABC 协议 delay_ms=0。生产算法允许单 trial idle delay，但禁止 overlapping pending movement。阶段切换只清 runtime credit，保留 W、ΔW、学习 RNG 和神经历史；冻结不抽样，恢复时重新建基准，不使用初始评价均值作训练基准。
- 默认 seeds=11/22/33/44/55，每 seed 初始 300、训练 3000、冻结 600。正式长训练需单独安排，本次仅做短 smoke。
- 记录 source SHA256、配置、每轮唯一 Goodness、goodness_delta、selected_parameter_count、delta_flip_count、ΔW 统计、参数变化和 NaN/Inf。日志与 checkpoint 留在 runs，研究报告另行整理。

历史的 10-Block 视觉 Stage Zero 与旧学习机制已归档，已有 NOT SUPPORTED 结果不覆盖、不改写。当前候选尚无正式 ACNT 行为学习结论。
