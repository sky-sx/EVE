# 当前 ABC Stage Zero

使用同一个 production Persistent ΔW Plasticity。协议见 [stage_zero_plan.md](../../docs/stage_zero_plan.md)。只做短 smoke：

```powershell
python -m experiments.stage_zero_abc.runner --device cpu --seeds 11 --initial 3 --training 6 --frozen 3 --output runs/delta_w_abc_smoke
```

默认 delta_magnitude=.001、subset_fraction=.001、plasticity_seed=seed+300000；与 Hand 随机源独立。首个训练 Goodness 只建 baseline，后续 forward 前移动，下降时仅反转选中方向。最多一个 pending trial，当前 ABC 不加 delay。冻结保留参数和方向，不采样；各阶段沿连续神经历史运行。

JSONL 记录 delta_w 统计、selected_parameter_count、delta_flip_count、goodness_delta、动作和有限性；config.json 锁定 source hash。结果路径应使用新目录，避免覆盖历史实验。

现阶段 IMPLEMENTED / NOT YET FORMALLY VALIDATED，尚无正式 ACNT 行为学习结论。短 smoke 和单元测试不能作为 SUPPORTED。
