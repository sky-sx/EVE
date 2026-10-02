# ACNT / EVE Core Runtime

> **当前研究主线（2026-10-02）：Online Self-Write ACNT。**
>
> 目标不是继续增加一套外部优化器，而是让 ACNT 在连续运行中，根据唯一标量 **Goodness** 形成在线信用、产生写入方向，并实际修改包括自身写入器在内的神经参数。

当前候选保留原版 Adapter → Core → Adapter 前向结构，在其上加入 **独立向量 Write Adapter + 有界在线信用 + 自校准写入 + 单一有界总 Goodness**。实验生命内部时间单调推进，不通过神经状态 reset、参数 rollback 或 experience replay 来完成学习。

目前已经有小模型上的行为学习、自身参数写入、规则再次适应和 Goodness 预测证据；但三次十分钟连续实验都还没有满足末段平台判据，真实屏幕/鼠标闭环也尚未接入，因此**不宣称已经验证一般终生学习**。

- [Self-Write 总报告](reports/acnt_self_write_session_2026-10-02.md)
- [训练研究入口](docs/training/README.md)
- [当前写入与信用算法](docs/training/CALIBRATED_WRITE.md)
- [单一有界总 Goodness](docs/training/BOUNDED_TOTAL_GOODNESS.md)
- [公开精简证据与源码版本](reports/acnt_self_write_session_2026-10-02/manifest.json)

---

## 当前主线：Online Self-Write

当前研究候选可以概括为：

~~~text
ReadIn
  ↓
Original ACNT Core
  ↓
ReadOut / Action
  ↓
Environment
  ↓
single Goodness
  ↓
online credit / trace
  ↓
F / Write
  ↓
ACNT parameters, Write parameters, Goodness parameters
  ↓
next continuous trajectory
~~~

关键约束：

- **只有一个最终 Goodness 标量**进入当前学习主线。
- 主体不读取外部经历日志；旁观日志、诊断标签和未执行动作的 counterfactual 不返送主体。
- 同一实验生命内部不重置神经状态、不回滚参数、不重放同一经历。
- Write 不是额外的“学习 Block 类型”，而是作用于普通 ACNT 参数空间的写入接口。
- 全覆盖候选要求所有实际有效参数在结构上可写，包括 Block 间连接、Write 自身以及 Goodness 相关参数。
- 在线训练还保留固定规模的数值信用状态（如 J、trace、moment、normalization statistics）；因此不能声称全部长期信息只存在于 Core hidden state 中。
- 当前算法包含裁剪、停止部分生命周期导数和校准规则，属于**研究候选**，不是“已经发现了无偏终生梯度”。

### 当前小模型

最新有界总 Goodness 实验使用：

~~~text
7 ordinary Blocks
28 Core neurons
3,839 effective parameters
740 Write parameters
58 Goodness parameters
148 independent F
~~~

在一次 600 秒连续生命（seed 66）中：

| 指标 | 结果 |
|---|---:|
| 决策次数 | 4,673 |
| 最后 1,000 次平均总 Goodness | **0.845592** |
| 最后 1,000 次动作正确率 | **81.1%** |
| 预测最终 Goodness 的 MSE | **0.016041** |
| 实际写过的参数坐标 | **3,727 / 3,839** |
| Goodness 参数写入覆盖 | **58 / 58** |
| Write 参数写入覆盖 | **740 / 740** |
| 末段平台判据 | **未满足** |

这条结果说明当前候选已经能够在连续轨迹中产生有效行为和自身写入，但预算截止时仍在变化，不能把 81.1% 当作稳定后的最终能力。

在较短的固定条件任务上，自校准独立向量 Write 的三个 seed 末 100 次期望正确率为：

~~~text
0.999850
0.999916
0.999884
~~~

实际动作正确率均为 1.00。完整阶段结果、ablation、规则切换和十分钟曲线见 [总报告](reports/acnt_self_write_session_2026-10-02.md)。

---

## 原始 ACNT 前向

Self-Write 主线没有把 ACNT Core 替换成另一套网络。当前实验仍从原版前向出发：

- Eye、Ear 是 ReadIn；Hand、Speak、Goodness、Route 是 ReadOut。
- 六个器官绑定不同 designated Block。
- Core event 使用 old committed state snapshot。
- Block 进行跨 Block / 跨 neuron synapse mixing。
- GLU + LayerNorm 产生 pre-activation。
- A/At 保留有限历史及真实逻辑时间戳。
- 每个 neuron 有 private NLM，读取自身 history、real-time age 和 validity。
- Hand / Route 使用逐坐标输出，不使用 Softmax 动作竞争。

生产 Runtime 的结构规范仍见 [docs/canonical_architecture.txt](docs/canonical_architecture.txt)。

**注意：该规范描述现有 Runtime；Self-Write 是当前研究主线，但尚未替换默认 Runtime.step。**

### 核心代码

- **acnt/block.py**：原始 Block 前向、synapse mixing、历史与 private NLM。
- **acnt/core.py**：active set、scheduler、snapshot communication。
- **acnt/adapters.py**：器官 Adapter。
- **acnt/calibrated_write.py**：当前自校准向量 Write 主体。
- **acnt/bounded_goodness.py**：单一有界总 Goodness。
- **acnt/goodness_prediction.py**：Goodness prediction 研究实现。
- **acnt/self_write.py、address_write.py、grouped_write*.py**：Self-Write 演化过程中的前序研究实现。
- **experiments/calibrated_write_duration.py**：当前连续 Self-Write 实验入口。

---

## 证据层级

当前仓库中几条训练路线需要明确区分：

| 路线 | 当前定位 |
|---|---|
| **Online Self-Write** | **当前主线**。连续在线全参数自修改；最新版本使用单一有界总 Goodness。 |
| Original ACNT + bounded BPTT / REINFORCE | 已验证的**短窗训练 baseline**，用于证明原始 ACNT Core 可以端到端学会小型时序任务。 |
| Persistent Delta-W | 现有 Runtime 中保留的旧学习路径和机制 baseline，**不是当前研究主线**。 |
| G0-G3 event-flow | 独立参考线 / 历史交接，用于研究 event-time semantics 与在线信用，不代表当前 Self-Write candidate。 |
| Local Plasticity / CorrelationRule | superseded 历史路线，正式结果保持 **NOT SUPPORTED**。 |

### Original ACNT end-to-end baseline

原版六器官 ACNT 在一个 Ear cue → recurrent Core → Hand action 的四事件时序任务上，使用 scalar-Goodness REINFORCE + bounded BPTT：

| Seed | 初始 expected G | 训练后 expected G | Accuracy |
|---|---:|---:|---:|
| 11 | 0.499711 | **0.991494** | **100%** |
| 22 | 0.500442 | **0.986264** | **100%** |
| 33 | 0.501040 | **0.995481** | **100%** |

Freeze-Core 对照分别为 50.00%、50.00%、66.67%。这证明小型任务上原始 ACNT Core 本身能够参与有用的端到端行为学习，但该方法有有限 BPTT 窗口，不是当前终生在线 Self-Write 方案。

详见：

- [算法与 API](docs/training/ORIGINAL_ACNT_TRAINING.md)
- [实测结果](reports/original_acnt_training_2026-10-01.md)

---

## 当前边界

目前**还没有**证明：

- 一般终生学习或无限时间稳定性；
- 百万参数规模的 Self-Write 可训练性；
- 未知长延迟 Goodness 的可靠信用分配；
- 长期规则切换后的能力保留而非重新学习；
- 真实 1080p 像素到真实鼠标动作的身体闭环学习；
- 真实音频、DirectInput 与仿生发声器官的完整接入。

因此下一阶段最重要的不是继续增加更多平行训练路线，而是把当前 Self-Write 候选冻结成可比较快照，逐步进入：

~~~text
real pixels
    ↓
Eye
    ↓
ACNT + Self-Write
    ↓
Hand
    ↓
real mouse
    ↓
environment outcome
    ↓
single Goodness
~~~

并在固定 source hash、多 seed、长轨迹下观察稳定末段行为。

---

## 复现当前 Self-Write 证据

需要 Python 3.11+。

先核对公开证据与完整回归：

~~~powershell
python -m pip install -e ".[test,reports]"
python -m experiments.summarize_session --verify-published
python -m pytest -q
~~~

启动一条新的 600 秒 bounded-total-G 生命：

~~~powershell
python -m experiments.calibrated_write_duration --seconds 600 --seed 66 --width 4 --threads 1 --bounded-total --output runs/new_bounded_life
~~~

输出目录必须是新目录；复现会产生一条新的独立生命，不会“续跑”历史报告中的 seed 66 状态。

完整原始逐事件日志和大体积 checkpoint 按仓库策略保留本地；公开 manifest 保存结果记录、配置、源码版本与 SHA256 映射。

---

## 其他运行入口

原始 Runtime smoke：

~~~powershell
python -m acnt --core-only --steps 12
python -m acnt --steps 2 --log-file runs/demo/mechanical.jsonl --summary-file runs/demo/summary.json
~~~

短窗 end-to-end baseline：

~~~powershell
python -m experiments.original_acnt_train --seeds 11 22 33 --updates 200 --batch-size 8
~~~

旧 Persistent Delta-W 的 OR/XOR sanity：

~~~powershell
python -m experiments.delta_w_sanity --steps 1000 --seeds 11 22 33
~~~

G0-G3 event-flow 参考线：

~~~powershell
python -m experiments.training.run_all
python -m experiments.training.run_all --include-g3
~~~

---

## 项目状态一句话

> **EVE / ACNT 当前已经从“核心能否被训练”进入“能否在单一 Goodness 下持续在线修改自身，并最终形成真实身体闭环能力”的阶段；当前研究主线是 Online Self-Write。**
