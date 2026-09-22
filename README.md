# ACNT：Local Plasticity 架构与运行骨架

当前架构的唯一规范是用户提供的 [Local Plasticity 版架构原文](docs/canonical_architecture.txt)。本 README 和 docs/ 中其他说明只帮助阅读与核对；如有冲突，以该原文为准。旧 e-prop 实验、测试结果和报告已移入[历史 Stage 0 归档](archive/stage_zero_legacy/README.md)，不代表当前架构的学习效果。

## 当前状态

`acnt/` 实现了 Block、六种器官 Adapter、ReadIn/ReadOut 连接、独立 Logistic 噪声加硬阈值的离散控制、机械日志，以及连接本地的塑性状态。生产运行路径不使用 e-prop、参数 Jacobian、BPTT、策略梯度或固定随机反馈。现用的 `CorrelationRule` 是可替换的实验候选，不是架构规定的唯一 `F_e/F_w`。

已做过不涉及 Stage 0 的最小运行检查：1080p eye 张量与原始音频窗口可经过 Adapter；Block 和四种 ReadOut 完成一步；goodness 外部 `g*` 覆盖 `g`，B_g/A_g 只用 `c_g` 校准；机械日志生成，参数及局部状态保持有限。此前 strict exact-one-hot `g*` 的五 seed Stage 0 已完成并独立审计 8100 行，结果为 **NOT SUPPORTED**，见[历史正式报告](reports/stage_zero_local_plasticity_report.md)。当前 Stage 0 只把外部 `g*` 改为“目标位激活时 1/激活位数，否则 0”，exact one-hot 仍仅用于评价；新定义尚未正式运行。真实截图/音频采集、DirectInput 执行、仿生发声器官与 e-prop 小系统对照尚未接入。

## 架构速览

| 类别 | 器官 | 规范中的边界 |
|---|---|---|
| ReadIn | eye、ear | 1080p 截图和原始音频经各自神经网络 Adapter 输入不同 Block；新输入强制对应 Block 更新一次 |
| ReadOut | hand、speak | 各自由独立 Block 输出；hand 产生键鼠控制，speak 产生约 30 维连续发声器官参数 |
| ReadOut | goodness | 独立 B_g 与 A_g，输出唯一 `g∈[0,1]` |
| ReadOut | route | 独立且永久 active 的 Block，输出总 Block 上限维的独立激活/休眠控制 |
| Core | Block | 每个 Block 保存 `z/a/A/At/W/b/W_c/b_c` 等状态，按规范中的一次更新公式计算 |

每个器官恰好绑定一个不同 Block。hand 和 route 的每个最终离散量先由 Adapter 输出连续倾向 `q`，再分别加入独立 `Logistic(0, τ)` 噪声，以 `1[q+ε>θ]` 形成真实 0/1 控制；多个键或 Block 可以同时成立。鼠标位移、speak 和 goodness 是连续量。人控执行开关在最终信号生成后生效：关闭时只写机械日志；执行结果和失败不返回 Core。

`g_eff` 是除 goodness 子系统外唯一全局塑性调制标量：同刻有准确外部 `g*` 时取 `g*`，否则取 `g`。B_g/A_g 不用自己的 `g_eff` 自我强化；同刻有 `g*` 时只用 `c_g=g*-g` 做局部校准。每个可塑连接维护自身 `e_c`，只利用真实局部 pre/post 活动形成状态，并在唯一标量到达时更新参数。具体 `F_e/F_w` 尚属研究问题。

## 文档与代码

- [规范原文](docs/canonical_architecture.txt)：当前架构的完整定义，包括 Block 公式、新 Stage 0 条件和 e-prop 对照边界。
- [实现任务与缺口](docs/implementation_task.md)、[架构边界核对](docs/clarifications.md)、[Adapter 实现记录](docs/adapter_choices.md)、[当前状态记录](docs/phase_report.md)。
- [新架构 Stage 0 协议](docs/stage_zero_plan.md)与[活动实验](experiments/stage_zero/README.md)：冻结当前 CorrelationRule，训练 `g*` 采用目标位激活时的 `1/n` 势函数；[此前 strict-reward 预检](reports/stage_zero_local_plasticity_preflight.md)与[报告](reports/stage_zero_local_plasticity_report.md)保留作历史记录。
- `acnt/block.py`、`core.py`：Block 状态与调度；`adapters.py`：六种 Adapter；`runtime.py`：器官连接与机械边界；`control.py`：独立噪声和硬阈值；`plasticity.py`：局部状态和可替换规则。
- [历史 Stage 0 归档](archive/stage_zero_legacy/README.md)：旧 e-prop 实验源码、专属测试和报告，仅作历史记录。

## 最小运行

Python 3.11+，依赖见 `pyproject.toml`。以下是运行骨架的 mock 输入检查，不是 Stage 0：

```powershell
python -m pip install -e ".[test]"
python -m acnt --steps 2 --log-file runs/demo/mechanical.jsonl --summary-file runs/demo/summary.json
```

活动测试已迁移至 Local Plasticity API。运行 `python -m pytest -q` 获取当前结果；旧 e-prop 测试与历史验收数字只留在归档。

## 报告与数据边界

旧 Stage 0 的两轮五 seed 报告结论均为 **NOT SUPPORTED**，只适用于当时的 e-prop 实现与实验协议。原始 `runs/stage_zero/` 由 `.gitignore` 排除，保留为本地历史数据，新实验使用独立的被忽略目录 `runs/stage_zero_local/`，不会覆盖历史数据。模型权重、大数据和批量日志仍按仓库忽略规则处理；通用报告模板见 `reports/`。GitHub README 模板见 [docs/README.github.template.md](docs/README.github.template.md)。
