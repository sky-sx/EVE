# ACNT / EVE Core Runtime

这是 EVE 的 ACNT v0.x 本地研究运行时。当前生产路径实现六器官 Adapter、snapshot Core scheduling、CTM-style synapse mixing、per-neuron private NLM、真实时间历史输入和Persistent ΔW Plasticity；不宣称已经完成行为学习。

当前架构的唯一规范是 [docs/canonical_architecture.txt](docs/canonical_architecture.txt)。

## 架构速览

- Eye、Ear 是 ReadIn；Hand、Speak、Goodness、Route 是 ReadOut，六者绑定不同 designated Block。
- 每次 Core event 的所有 Block 读取同一个 old committed z snapshot。
- Block 先用 W_ij 完成跨 Block/跨 neuron 的 synapse mixing，再经 GLU 与 LayerNorm 得到 pre-activation a。
- A/At 保存最近 hold_tick 个 a 与真实逻辑时间戳；每个 neuron 的 private NLM 独立读取自己的 activation history、real-time age 和 validity，产生 activation / Block output state z。
- ticktime 同时是 scheduler 间隔尺度与 age 的 ms 归一化尺度。
- Hand/Route 使用逐坐标 q + Logistic noise + threshold；没有 Softmax 动作竞争。
- 唯一 Goodness 的连续差值决定已选 ΔW 保持或反转；参数不回滚，学习路径不使用 autograd。

## 代码与文档入口

- `acnt/block.py`：synapse mixing、A/At、grouped private NLM。
- `acnt/core.py`：active set、ticktime scheduler、old-state snapshot。
- `acnt/runtime.py`：六器官绑定、ReadIn/ReadOut 与机械边界。
- `acnt/plasticity.py`：持久 ΔW、统一稀疏元素抽样、单 pending trial 与唯一 Goodness 路径。
- [实现缺口](docs/implementation_task.md)、[当前阶段报告](docs/phase_report.md)、[Stage Zero 协议](docs/stage_zero_plan.md)、[Adapter 选择](docs/adapter_choices.md)。
- [当前 ABC Stage Zero](experiments/stage_zero_abc/README.md) 与历史归档 [archive/stage_zero_legacy](archive/stage_zero_legacy/README.md)。

## 当前实验状态

Persistent ΔW Plasticity：**IMPLEMENTED / NOT YET FORMALLY VALIDATED**。默认 delta_magnitude=0.001、subset_fraction=0.001、plasticity_seed=0；首次 Goodness 只建基准，下一轮 forward 前移动参数，坏结果仅反转方向。冻结恢复重新建基准，延迟反馈最多允许一个未评价 trial。

OR/XOR 是共用生产算法的极小机制 sanity；CPU 200,000 step、seed 11/22/33 下 OR 与 XOR 各 3/3 成功，最终准确率均为 100%；完整结果和 1,000 step 对比见[阶段报告](docs/phase_report.md)，不能外推为 ACNT 行为学习。当前唯一 Stage Zero 协议为 `experiments/stage_zero_abc`。旧 `experiments/stage_zero` 指向历史归档；此前 Local Plasticity / CorrelationRule 的 **NOT SUPPORTED** 报告与原始数字保持不变，不是新 ΔW 机制的验证结果。尚无正式 ACNT 行为学习结论。

真实截图/音频采集、DirectInput 执行与仿生发声器官映射仍未接入。

## 最小运行

需要 Python 3.11+，依赖见 `pyproject.toml`。

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m experiments.delta_w_sanity --steps 1000 --seeds 11 22 33
python -m acnt --core-only --steps 12
python -m acnt --steps 2 --log-file runs/demo/mechanical.jsonl --summary-file runs/demo/summary.json
```

## Train original ACNT end to end

The original Adapter -> Core -> Adapter now has an explicit training path:
`OriginalTrainingRuntime` + `GoodnessTrainer`, using bounded BPTT and scalar-Goodness
REINFORCE. It preserves original GLU/LN, private NLMs, A/At and snapshot communication;
it uses neither G2 event-flow dynamics nor the old delta-W update.

See [algorithm, APIs and reproduction](docs/training/ORIGINAL_ACNT_TRAINING.md)
and [measured learning results](reports/original_acnt_training_2026-10-01.md).

```powershell
python -m experiments.original_acnt_train --seeds 11 22 33 --updates 200 --batch-size 8
```

This is a validated short-window research training baseline. Default `Runtime.step`
keeps its original behavior; learned deployment uses `learn=False` after commit.
