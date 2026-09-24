# ACNT / EVE Core Runtime

这是 EVE 的 ACNT v0.x 本地研究运行时。当前生产路径实现六器官 Adapter、snapshot Core scheduling、CTM-style synapse mixing、per-neuron private NLM、真实时间历史输入和连接局部塑性；不宣称已经完成行为学习。

当前架构的唯一规范是 [docs/canonical_architecture.txt](docs/canonical_architecture.txt)。

## 架构速览

- Eye、Ear 是 ReadIn；Hand、Speak、Goodness、Route 是 ReadOut，六者绑定不同 designated Block。
- 每次 Core event 的所有 Block 读取同一个 old committed z snapshot。
- Block 先用 W_ij 完成跨 Block/跨 neuron 的 synapse mixing，再经 GLU 与 LayerNorm 得到 pre-activation a。
- A/At 保存最近 hold_tick 个 a 与真实逻辑时间戳；每个 neuron 的 private NLM 独立读取自己的 activation history、real-time age 和 validity，产生 activation / Block output state z。
- ticktime 同时是 scheduler 间隔尺度与 age 的 ms 归一化尺度。
- Hand/Route 使用逐坐标 q + Logistic noise + threshold；没有 Softmax 动作竞争。
- 唯一 Goodness 标量沿现有 Local Plasticity Dynamics 调制普通参数；学习路径不使用 autograd。

## 代码与文档入口

- `acnt/block.py`：synapse mixing、A/At、grouped private NLM。
- `acnt/core.py`：active set、ticktime scheduler、old-state snapshot。
- `acnt/runtime.py`：六器官绑定、ReadIn/ReadOut 与机械边界。
- `acnt/plasticity.py`：连接局部事件、eligibility 与唯一 Goodness 路径。
- [实现缺口](docs/implementation_task.md)、[当前阶段报告](docs/phase_report.md)、[Stage Zero 协议](docs/stage_zero_plan.md)、[Adapter 选择](docs/adapter_choices.md)。
- [Stage Zero harness](experiments/stage_zero/README.md) 与历史归档 [archive/stage_zero_legacy](archive/stage_zero_legacy/README.md)。

## 当前实验状态

现存 Stage Zero 长训练结果来自修正前的 Block temporal implementation，不能外推到当前 synapse mixing + private real-time NLM。原始报告与数据保留作历史证据；新的正式结论必须在 corrected source hash 上重新运行。本次架构修正只运行极短 smoke，不启动长训练。

真实截图/音频采集、DirectInput 执行与仿生发声器官映射仍未接入。

## 最小运行

需要 Python 3.11+，依赖见 `pyproject.toml`。

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m acnt --core-only --steps 12
python -m acnt --steps 2 --log-file runs/demo/mechanical.jsonl --summary-file runs/demo/summary.json
```
