# Stage Zero: deterministic fractional environment Goodness

**结论：NOT SUPPORTED。** 在固定的五 seed、每 seed 270 initial + 1080 training + 270 frozen 预算下，fractional Goodness 确实进入原有 delayed e-prop 路径并引发参数更新，但 8,100 次动作中没有一次 exact one-hot success。该结论只针对本协议和预算；分数为正或参数变化均不代表学会视觉到动作映射。

## 本轮唯一变量

从 DeepSeek 冻结 Teacher 表均值切换为环境根据实际采样动作确定性计算的标量：

```python
correct_pressed = int(actions[target])
pressed_count = int(actions.sum())
goodness = correct_pressed / pressed_count if pressed_count > 0 else 0.0
```

27 个 Hand control 仍为独立 Bernoulli sampling。多按错误键的影响仅来自 `pressed_count` 分母。训练链是 RGB 图像 → ACNT → 27-bit Hand 采样 → 环境 Fractional Goodness → 250 ms 逻辑延迟 → 现有 `Plasticity.apply_goodness()`。没有 Teacher lookup、Teacher 表文件、API 请求、classifier loss、per-output reward 或额外 shaping。Exact one-hot success 继续作为外部行为验收。

本轮起点：`d9d7aa78596feb52d09259ebd2c1519a221ae837`。运行时记录的源码 SHA256 与审计时文件一致；`acnt/*.py` 无改动。固定参数包括 learning_rate=0.001、rho=0.9、ema_alpha=0.1、initial_g_bar=0.5、ticktime=0.25 秒。架构、三帧时序、seed/RNG、阶段 reset 和实验预算未调整。

## 正式运行

- 设备：NVIDIA GeForce RTX 5080，CUDA FP32，确定性算法启用；五个 seed：11、22、33、44、55。
- 每 seed：270 initial、1080 training、270 frozen；window=270。总计 8,100 episode，其中 5,400 个训练 episode。
- 正式运行耗时 1322.58 秒；没有筛选 seed、提前停止或根据结果调参。
- 本地原始数据：`runs/stage_zero/fractional-formal-20260921/`。CSV、q/p/action bits、per-class、学习曲线、参数摘要、checkpoint、metadata 和 audit.json 保留在本地；`runs/` 按仓库规则不上传。

### 五 seed 结果

| Seed | 阶段 | Exact success | Target p | Non-target p | Mean fractional Goodness |
|---|---|---:|---:|---:|---:|
| 11 | initial | 0/270 | 0.507993 | 0.513001 | 0.038988 |
| 11 | training | 0/1080 | 0.515519 | 0.512859 | 0.036507 |
| 11 | frozen | 0/270 | 0.514909 | 0.514344 | 0.038957 |
| 22 | initial | 0/270 | 0.517241 | 0.515013 | 0.037339 |
| 22 | training | 0/1080 | 0.513594 | 0.513249 | 0.037739 |
| 22 | frozen | 0/270 | 0.514896 | 0.513712 | 0.033516 |
| 33 | initial | 0/270 | 0.509285 | 0.510855 | 0.037010 |
| 33 | training | 0/1080 | 0.509779 | 0.509936 | 0.035665 |
| 33 | frozen | 0/270 | 0.512263 | 0.510715 | 0.038305 |
| 44 | initial | 0/270 | 0.480433 | 0.477394 | 0.036747 |
| 44 | training | 0/1080 | 0.477278 | 0.476367 | 0.039098 |
| 44 | frozen | 0/270 | 0.477981 | 0.476710 | 0.037007 |
| 55 | initial | 0/270 | 0.483909 | 0.493327 | 0.035878 |
| 55 | training | 0/1080 | 0.493932 | 0.493423 | 0.039425 |
| 55 | frozen | 0/270 | 0.493716 | 0.493574 | 0.033412 |

### 合并指标

| 阶段 | Episodes | Exact success | Target p | Non-target p | Mean Goodness | Target-bit hit | Non-target false rate | Mean active actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 1350 | 0 | 0.499772 | 0.501918 | 0.037192 | 0.503704 | 0.501624 | 13.545926 |
| training | 5400 | 0 | 0.502020 | 0.501167 | 0.037687 | 0.510926 | 0.502514 | 13.576296 |
| frozen | 1350 | 0 | 0.502753 | 0.501811 | 0.036239 | 0.483704 | 0.500427 | 13.494815 |

**4,092/8,100（50.52%）** episode 的 Goodness 大于零；exact success 仍为 **0/8,100**。三阶段 mean Goodness 仅是训练标量的诊断统计。Target p 的微小上移与 non-target p 的微小下移没有形成任何 exact one-hot 成功，不能单独判定学习成立。

### 参数与 baseline

| Seed | 正分数 episodes / 1620 | 训练参数更新 / 1080 | 参数 delta norm | 最终 g_bar |
|---|---:|---:|---:|---:|
| 11 | 841 | 1080 | 0.236038 | 0.029056 |
| 22 | 822 | 1080 | 0.243779 | 0.041279 |
| 33 | 818 | 1080 | 0.236258 | 0.030681 |
| 44 | 793 | 1080 | 0.258715 | 0.037117 |
| 55 | 818 | 1080 | 0.250851 | 0.035009 |

五个 seed 均从 `g_bar=0.5` 开始；每个 seed 的 200/200 参数张量有变化。初始及 frozen 评估均未更新参数。

## 与上一轮 Teacher 实验的对照

上一轮 [冻结 Teacher 报告](stage_zero_teacher_report.md)使用相同五 seed、预算和 ACNT/e-prop 设置，结论同为 **NOT SUPPORTED**，exact success 也是 **0/8,100**。逐行比较两个本地正式 run，1,350 个 initial episode 的 target、q/p、采样 action bits 和逻辑时间完全一致，确认了学习前的同一起点。两轮 reward 定义不同，mean Goodness 数值不宜直接比较。当前实验能说明：仅把 Teacher Goodness 换成指定的 Fractional Goodness，在这次预算内没有获得 exact one-hot 行为成功；不能据此归因于 ACNT 的所有可能配置。

## 验证与审计

- 完整 pytest：**253 passed in 12.43s**。
- `python -m experiments.stage_zero.audit runs/stage_zero/fractional-formal-20260921`：**8,100/8,100 行通过**，结论 `NOT SUPPORTED`，所有运行源码 SHA256 匹配。
- Audit 从 raw action bits 独立重算 Fractional Goodness、correct/exact、动作数、`delta=goodness-g_bar_before`、baseline、250 ms 延迟及 eligibility decay；NaN/Inf 均为 0。
- 初末参数哈希与 checkpoint 匹配，固定反馈与 seed 初始化一致；initial/frozen 参数不变。所有训练 episode 均有参数更新，但这不构成行为成功。
- 上一轮 Teacher 校准文件和报告保持原样。本轮没有新增正式长实验后的调参或第二变量修改。

原始数据目录受 `.gitignore` 排除；GitHub 仅发布代码、测试、协议文档和本报告。
