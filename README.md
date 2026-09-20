# ACNT Runtime v0

## 报告索引

**如果你需要了解某类大文件、大数据或批量日志，请先看对应报告。**

本仓库只收录源码、配置、脚本、文档、必要的小型数据、经审查的少量关键日志和整理后的报告。模型权重、checkpoint、大数据、原始数据、原始日志、批量及历史日志、缓存和临时运行产物不直接上传。

| 内容 | 对应报告 | 当前报告状态 |
|---|---|---|
| ACNT Stage Zero 实验验证 | [正式实验报告](reports/stage_zero_report.md) | 五 seeds 已完成；NOT SUPPORTED |
| 模型权重与 checkpoint | [模型产物报告](reports/model_artifacts_report.md) | 模板，待填写核验结果 |
| 大数据集与原始数据 | [数据集摘要报告](reports/dataset_summary_report.md) | 模板，待填写核验结果 |
| 批量、历史与原始日志 | [日志摘要报告](reports/logs_summary_report.md) | 模板，待填写核验结果 |
| 其他大文件或数量过多的文件 | [大文件报告](reports/large_files_report.md) | 模板，待填写核验结果 |

`runs/latest.log` 为可选的关键日志路径，本次未新增或选取日志；忽略规则中的例外不代表已收录。下文历史验收目录是本地证据路径，不承诺随仓库发布。报告模板不代表已完成产物审计。

### 信息闭环规则

1. 先看 README，再按索引查看对应报告。
2. 报告足够判断当前问题时闭环，停止追查原始数据。
3. 报告不足时列出“我还需要哪些信息”，包含问题、缺少的统计或字段、时间范围和完成标准。
4. Codex 按清单在授权范围内收集，整理为补充报告，更新 README 索引，不直接上传原始内容。
5. 再判断是否足够；足够即结束，不足则继续补充需求。

### GitHub 整理与发布规则

可复用的完整 README 模板见 [README.github.template.md](docs/README.github.template.md)，使用时复制到目标仓库根目录并填写占位符。四份报告在 `reports/` 下，填写后作为正式报告发布。

`.gitignore` 默认排除模型产物、数据目录、日志、运行产物和常见缓存；仅允许经审查的 `runs/latest.log` 作为运行日志例外。必要的小样本建议放在 `samples/`，对于被忽略的格式，只按精确文件路径增加例外。

建议普通文件不超过 1 MiB、关键日志不超过 256 KiB、小样本合计不超过 5 MiB；这些是项目审查阈值，不是 GitHub 平台限制。超出时优先整理报告；低于阈值也须检查用途、数量、隐私和授权。日志节选必须标注来源、筛选条件和省略范围。

提交前检查暂存区的文件内容、大小、总量以及已有 Git 历史。`.gitignore` 不会限制文件大小，也不会移除已跟踪文件。不要通过强制添加绕过原始数据排除规则；报告、样本和配置同样需要检查凭据与敏感信息。

## 当前实现状态

Phase 1–15 已完成。当时全套 **146 项测试通过**；独立命令行 mock 验收连续运行 **64 步**，生成 256 条机械日志，16 次同刻 mock teacher，产生 31 种 active 集合。参数保持有限值，88 个参数张量发生变化。这里只验证执行链，不证明学习有效。

本次四项最小修复后，完整 `pytest -q` 为 **153 passed**；新增验证内部 VJP、快照传播、ReadIn 脉冲及 sigmoid 校准。

**Stage Zero 已完成正式五 seed 实验**：每 seed 1,080 次训练、270 次初始化测评、270 次冻结测评。严格 exact-match 奖励在全部阶段为零，结论 **NOT SUPPORTED**；这表示当前实验未观察到映射学习。最新完整测试 **208 passed**，无跳过或失败。未接真实键鼠执行或声音合成，正式 Runtime 数学和版本保持不变。详见 [完整报告](reports/stage_zero_report.md)、[预检记录](reports/stage_zero_preflight.md) 和 [实验协议](experiments/stage_zero/DESIGN.md)。

架构依据为 docs/canonical_architecture.txt、任务 docs/implementation_task.md，以及优先级更高的后续用户澄清 docs/clarifications.md。逐阶段文件、公式和测试记录见 docs/phase_report.md。

## 运行

常规 Python 3.11+ 环境：

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m acnt --steps 12 --log-file runs/demo/mechanical.jsonl --summary-file runs/demo/summary.json
```

Phase 1–15 的历史验收环境为 Python 3.12.14、PyTorch 2.13.0+cpu、pytest 9.1.1。Stage Zero 正式运行使用 Python 3.11.9、PyTorch 2.10.0+cu130、RTX 5080 / CUDA 13.0。以下保留历史命令。在 D:\EVE\EVE0.6 执行：

```powershell
$env:PYTHONPATH = 'D:\EVE\EVE_0_5\.testdeps'
$acntPython = 'C:\Users\12633\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $acntPython -m pytest -q
& $acntPython -m acnt --steps 12 --log-file runs/demo/mechanical.jsonl --summary-file runs/demo/summary.json
```

仅复用现有第三方库，不导入旧 EVE 实现；项目源代码不依赖机器专属路径。默认 CLI 运行八个 Block（六器官和两个普通 Block）；加 --core-only 可运行原有三 Block smoke。显式逻辑时间每步前进 250ms，命令不等待真实时钟。机械 JSONL 以 append 模式写入，独立验收应使用新路径。

最近独立验收目录：runs/runtime-v0-20260918-223626/。

- summary.json：步数、参数有限性、变化参数和 active 集合统计。
- mechanical.jsonl：四类 ReadOut 完整机械日志。
- ticks.jsonl：逐步状态和训练标量诊断。

## Stage Zero 复现

```powershell
python -m experiments.stage_zero --output runs/stage_zero/reproduction --device cuda --seeds 11 22 33 44 55 --train-episodes 1080 --evaluation-episodes 270 --window 270
python -m experiments.stage_zero.audit runs/stage_zero/reproduction
```

输出目录必须不存在。无 CUDA 时可用 `--device cpu`；短 smoke 使用一个 seed、27 次训练和 27 次测评。
正式原始 CSV/JSON/checkpoint 保存在本地 `runs/stage_zero/formal-20260920/`，不会作为 bulk 数据上传；报告保留全部 seed、27 类结果及失败分析。

## 代码

| 文件 | 职责 |
|---|---|
| acnt/block.py | Block 状态、history、r/a/h/z 和局部计算图 |
| acnt/core.py | 通信、active 和 ticktime 调度 |
| acnt/adapters.py | 用户选定的六种 Adapter |
| acnt/runtime.py | 器官挂接、ReadIn、ReadOut、校准和整机 step |
| acnt/control.py | 独立 Logistic noise、概率 p、严格硬阈值 |
| acnt/plasticity.py | 逐参数资格迹、延迟 goodness、EMA 和参数更新 |
| acnt/hand.py | 明确的 82 键 mock 布局、鼠标名称、训练器日志筛选 |
| acnt/mechanical.py | 内存及 JSONL 日志 |
| acnt/__main__.py | mock runtime 与命令行入口 |
| experiments/stage_zero/ | 独立 Stage Zero driver、环境、测评、原始日志审计；不调用 Runtime.step |

所有计算使用 FP32；早期验收为 CPU，Stage Zero 已完成 CPU/CUDA smoke 与 CUDA 正式实验。Block 无 batch 维；Adapter 单体支持单样本或一个 batch 维，接 Core 时只接受单样本。LN 使用非仿射 population variance，epsilon=1e-5。

A/At 从旧到新存储并共同淘汰，递推从新到旧，W_c[idx] 使用当前队列索引。每次 h 从零开始，最新项 delta_t=0、gamma=0.5。按用户修订，gamma=sigmoid((delta_t_ms/1000)/ticktime)，ticktime 为正数秒。

Core 只更新到期且 active 的 Block。每轮所有 due/forced Block（含 self-edge）读取同一轮前 committed z 快照，新 z 下一传播轮才可见。inactive 保留状态和历史，不参加当前输入求和。新 ReadIn 可无视 route 和未到期调度，强制对应 Block 在当前轮激活并更新一次，轮末恢复原 active。ReadIn o 是一次性脉冲，更新后归零；全零样本仍是新事件。

## Adapter 与机械边界

eye 是两层步幅 CNN + Linear，完整 1080p 直接进卷积，无平均池化；ear 是固定原始音频窗口 + Linear；hand 是两层 MLP；speak、goodness、route 是 Linear。

hand 输出 **82 个键盘坐标 + 3 个鼠标按键 + 连续 dx/dy**，修饰键独立，支持组合。全部控制都写日志，包括方括号和其他按键；read_training_controls(record) 只在读取端筛选 A–Z 与左键 click。82 键名称是明确声明的 mock 布局，不声称适配所有物理 82 键键盘。

所有 ReadOut 先形成最终信号，再看人控执行开关。hand/speak 默认关闭，只记日志；route/goodness 默认开启。mock executor 的返回码被丢弃，异常只记机械日志，不反馈 Core。关闭物理执行仍计算相同采样事件和资格项。

goodness 关闭时仍计算并记录 g，但不校准 A_g、不把自身 g 用于一般训练；独立外部 teacher 仍可提供 g_eff。route 永远 active，goodness 的 active 由人控决定；最终 active 覆盖不反写原始随机采样 a。没有 DirectInput、VocalTractLab 或 world reset。

## 塑性

Runtime.step 顺序执行 Block、route、hand、speak、goodness 和参数更新。ReadOut 每轮可读取保持的 z，Block 更新仍受 active/ticktime 控制。

仅同刻 g/g* 校准 A_g：L=1/2*(g-g*)²，使用当前局部 dg/dparameter。旧 teacher 不缓存、不复用，不校准其他时刻预测或 B_g。其他参数的 g_eff 取同刻 teacher（若有），否则取当前 g；delta=g_eff-更新前g_bar。

每个实际参数张量独立保存资格项。旧 A、来源 z（含 self）是 detached 局部输入；当前 a/h/z/readout 保留一轮局部图，形成资格项后丢弃，不跨世界或旧历史反向传播。Block 使用固定随机反馈 L=sum_r B_i^(r)*(a-p)/tau 的 VJP；ReadIn adapter 随当前局部图获得资格项。连续 ReadOut 保留均值导数，hand 连续项只含 dx/dy。离散 adapter 取 sum(stopgrad((a-p)/tau)*q) 的局部导数，直接离散梯度不重复进入 Block，不对 p、threshold 或 world 求导。

e=exp(-delta_seconds/tau)*e_previous+local_derivative。noise tau 与 decay tau 均等于 Block.ticktime。延迟 goodness 到达时再次按实际延迟衰减，合并同一参数的内部、连续、离散资格项，更新参数后乘 rho，随后更新 g_bar 的 EMA。A_g 完全排除一般参数更新。

默认 learning_rate=0.001、rho=0.9、EMA alpha=0.1、初始 g_bar=0.5；可显式配置。mock CLI 使用 learning_rate=0.0001、参数裁剪 [-10,10]、ticktime=0.25 秒。

非仿射 LN 使 mean(z) 理论上恒为零，因此内部学习已改为 neuron-specific VJP，并回归验证普通 Block / ReadIn tag 明确非零。固定 feedback_seed 默认为 0，反馈保存为 buffer，不进入 forward 或 Goodness 参数更新。Goodness 输出使用 sigmoid，同刻校准规则不变；参数 clipping 保留。详见 docs/clarifications.md。这些测试不代表 Stage 0 已经收敛。

Runtime.enable_plasticity(...) 可显式配置超参数，Runtime.step 默认启用塑性。可分开调用 update_blocks、generate_*，然后 learn_goodness(signal, delivered_ms=...) 验证延迟；teacher 按生成时刻配对，资格迹按送达时刻衰减。
