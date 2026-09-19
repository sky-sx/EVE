# ACNT 第一版可运行实现任务

上方提供的 **ACNT 当前 canonical architecture** 是本次实现的唯一架构依据。

本任务不得引用、恢复或混入此前任何旧版 ACNT / CTM / TNN / N+1 / Replay / Top-k / Memory Block 等已经废弃的设计。

如果旧代码、旧文档、历史实现与当前 canonical architecture 冲突：

**一律以当前 canonical architecture 为准。**

本轮不要重新设计 ACNT，不要替我补充新的认知模块，不要为了“工程完整性”增加当前规范中不存在的结构。

---

# 一、本轮唯一目标

本轮最重要的事情只有一个：

> **先让当前 ACNT 架构真实地跑起来。**

这里的“跑起来”指：

- 能正确创建 Block；
- Block 能执行一次完整更新；
- 多个 Block 能相互传递状态；
- `active` 能真正决定 Block 是否参加当前计算；
- ReadIn / ReadOut Block 的角色能够正确挂接；
- Adapter 能真正和对应 Block 接起来；
- route / goodness / hand / speak / eye / ear 的基本执行链可以运行；
- e-prop 所需的数据结构和资格迹能够建立并执行更新流程；
- 离散 ReadOut 的 `q + noise + threshold` 机制能够执行；
- 各部分具有对应的基础 test；
- 整个程序可以在最小 mock 输入条件下启动并完成若干更新步骤，而不是启动即报错。

本轮**不要求证明任何学习效果**。

因此本轮暂时不关心：

- goodness 是否提高；
- 字母任务是否学会；
- route 是否学会合理稀疏；
- hand 是否学会正确动作；
- e-prop 是否最终收敛；
- ACNT 是否产生智能行为；
- 任何 benchmark 数值。

这些属于后面的实验阶段。

本轮验收标准首先是：

> **架构实现正确，并且程序能够稳定执行。**

---

# 二、本轮明确不做的内容

本轮不要准备：

- 中英文长音频训练集；
- 动画；
- 纪录片；
- 大模型蒸馏题单；
- DeepSeek V4 teacher 数据；
- 虚拟机训练环境；
- VocalTractLab；
- 真实模拟发声器官；
- 正式训练数据；
- Stage 0 字母/绿屏实验。

尤其注意：

> **本轮不运行 Stage 0。**

Stage 0 是当前 ACNT 本体能够稳定运行以后才进行的第一个学习实验。

这一次只做 ACNT Runtime / Plasticity Skeleton 本身。

---

# 三、实现顺序必须分阶段进行

不要一口气把所有代码全部写完后再统一测试。

必须按照下面的顺序推进。

---

# Phase 1 — Block 本体

第一步只实现 Block。

此时不要实现 Adapter，不要实现 eye / ear / hand / speak / goodness / route 的完整器官逻辑。

按照 canonical architecture，实现 `Block B_i` 的全部必要状态和一次更新过程。

至少包括：

- `active`
- `neuron_size`
- `ticktime`
- `hold_tick`
- `z`
- `a`
- `o`
- `A`
- `At`
- `W_ij`
- `b`
- `r`
- `W_c`
- `b_c`
- `sigma`
- `LN`

以及 canonical architecture 中规定的完整 Block update。

实现时不要擅自改公式。

如果代码实现与数学规范之间存在尺寸不一致、变量含义不明确或无法直接编码的地方：

> **停止该部分实现，明确指出具体冲突，并向我询问。**

不要自行修公式。

---

# Phase 2 — Block-to-Block 通讯

Block 本体能够单独运行以后，再实现 Block 之间的通信。

需要验证：

```text
B_j.z
→ W_ij
→ B_i.r
→ B_i.a
→ history
→ B_i.z
```

确认：

- `i=j` 的自连接能够存在；
- 不同 Block 之间能够通信；
- 多个 active Block 的输入能够累加；
- inactive Block 不参与 `sum_{j active} W_ij z_j`；
- 不因为某个 Block inactive 而破坏其他 Block 的运行。

这一阶段先只使用人工构造的小规模 Block。

不要加入 Adapter。

---

# Phase 3 — active 机制

随后专门验证 `active`。

需要明确区分：

```text
Block 存在
```

和：

```text
Block 本次参与更新
```

测试至少覆盖：

1. 所有 Block active；
2. 部分 Block inactive；
3. inactive Block 不参与其他 Block 当前输入求和；
4. inactive Block 自身当前不执行普通更新；
5. 再次 active 后能够继续运行；
6. route Block 永久 active 的机制之后能够建立在这个基础上；
7. ReadIn 强制单次 active 的机制之后能够建立在这个基础上。

---

# Phase 4 — 第一轮 Block 系列测试

到这里必须先停下来完成一轮完整测试。

这一轮 test 只验证 Block/Core，不验证 Adapter。

至少测试：

## 单 Block

- 初始化；
- tensor shape；
- 一次 update；
- 多次连续 update；
- history push / pop；
- `hold_tick` 上限；
- `At` 与 `A` 一一对应；
- `LN`；
- `sigma`；
- `z/a/r/h` 中不能出现无解释 NaN/Inf。

## 多 Block

- 2 Block 通信；
- 3+ Block 通信；
- self-edge；
- active / inactive；
- active 集合变化；
- 多次异步或顺序更新不会直接崩溃。

## 时间历史

- `delta_t` 正确形成；
- `ticktime` 真正进入 `gamma`；
- history 从最新向最旧遍历；
- history 未填满时也能运行。

只有 Block 系列 test 通过以后，才进入 Adapter。

---

# Phase 5 — Adapter 设计门

**这是强制人工确认点。**

在开始写任何正式 Adapter 之前：

> **必须停止编码并向我汇报 Adapter 的设计候选。**

不得自行决定。

需要分别告诉我：

- eye adapter 推荐哪些实现；
- ear adapter 推荐哪些实现；
- hand adapter 推荐哪些实现；
- speak adapter 推荐哪些实现；
- goodness adapter 推荐哪些实现；
- route adapter 推荐哪些实现。

原则是：

> 没有必要采用复杂结构时，优先简单常见网络。

候选一般可以来自类似：

- Linear；
- MLP；
- CNN；
- 必要时其他非常常规的神经结构。

但不要因为“视觉一般用 CNN”就直接替我决定。

你需要向我说明每个候选：

- 输入；
- 输出；
- 大致层数；
- 为什么可能适合；
- 最简单可运行方案是什么；
- 是否存在明显计算成本差异。

然后明确向我询问：

> **这一版分别选择哪一种 Adapter？**

在得到我的回答前：

**不得继续实现正式 Adapter。**

---

# Phase 6 — Adapter 单体实现与测试

得到我的选型以后，再写 Adapter。

首先不要接 Block。

每个 Adapter 独立 test。

需要验证：

```text
input
→ adapter
→ expected-shape output
```

重点不是输出语义正确，而是：

- shape 正确；
- dtype 正确；
- forward 能运行；
- parameter 能枚举；
- parameter 可建立 eligibility；
- batch / non-batch 规则明确；
- 不出现 NaN/Inf。

---

# Phase 7 — Adapter + Block 联调

Adapter 单体 test 通过以后，再把 Adapter 接入对应 Block。

需要分别验证：

```text
external input
→ ReadIn Adapter
→ o
→ Block
→ z
```

以及：

```text
Block
→ z
→ ReadOut Adapter
→ output
```

注意 canonical architecture 的约束：

> 每一个 ReadIn / ReadOut 都对应一个且仅对应一个不同 Block。

所以必须验证：

- eye Block 独立；
- ear Block 独立；
- hand Block 独立；
- speak Block 独立；
- goodness Block 独立；
- route Block 独立；

任何两个器官都不得共用一个 Block。

---

# Phase 8 — ReadIn 测试

然后实现 ReadIn 的运行行为。

## eye

验证：

```text
1080p mock screenshot
→ eye adapter
→ eye Block.o
→ eye Block update
```

并验证：

> 新 eye 输入到达时，即使 route 当前没有选择 eye Block，也必须强制让 eye Block active 一次。

本轮可以使用随机生成或纯色 mock 1080p tensor。

不需要真实桌面训练数据。

## ear

同理：

```text
mock raw audio
→ ear adapter
→ ear Block
```

验证新音频输入能够强制触发一次 ear Block。

本轮不需要准备真实长音频语料。

---

# Phase 9 — Route 测试

实现 route ReadOut。

验证：

```text
route Block.z
→ route adapter
→ q_i
→ independent noise ε_i
→ threshold
→ active_i
```

需要验证：

- 输出维度等于配置的总 Block 上限；
- 每个 Block 有独立 q；
- 每个 Block 有独立 noise；
- 不使用 Softmax；
- 多个 Block 可以同时 active；
- 多个 Block 可以同时 inactive；
- route Block 自身永远不能被 route 输出关闭；
- route 控制可以真实改变下一轮参与计算的 active Block 集合。

这里只验证机制。

不验证 route 是否“学得好”。

---

# Phase 10 — Goodness 测试

随后实现 goodness ReadOut。

验证：

```text
B_g
→ z_g
→ A_g
→ g
→ clamp [0,1]
```

同时验证：

```text
存在 g*
→ g_eff = g*

不存在 g*
→ g_eff = g
```

还必须检查：

- goodness 只有一个标量；
- 不产生第二 reward；
- 不产生 confidence；
- 不产生 route goodness；
- 不产生 exploration goodness；
- A_g 不通过最大化 g 训练；
- A_g 只通过 teacher `g*` 校准。

本轮只验证校准链能够 forward/update。

不验证其预测质量。

---

# Phase 11 — 其他 ReadOut

随后依次实现：

## hand

先验证连续 adapter 输出。

然后验证离散动作：

```text
q
→ Logistic noise
→ hard threshold
→ a ∈ {0,1}
```

每个离散键独立。

不要 Softmax。

连续鼠标运动输出不经过该机制。

本轮默认进入 mechanical log，不要求真实 DirectInput。

---

## speak

验证：

```text
speak Block
→ adapter
→ approximately 30-dimensional continuous output
```

这里只验证约 30 维控制参数能够形成。

本轮不接 VocalTractLab。

不生成真实声音。

---

# Phase 12 — ReadOut 人控开关

所有 ReadOut 都需要测试人控执行开关。

执行顺序必须是：

```text
Block
→ Adapter
→ （如果是离散动作则 noise）
→ final control signal
→ human execution switch
```

不是先看权限再决定要不要计算。

当执行开关关闭：

```text
final signal
→ mechanical log
```

但：

```text
不真实执行
不返回 failed
不返回 blocked
不返回 success
```

Core 不应收到这些机械状态。

---

# Phase 13 — e-prop 基础实现

随后把当前 canonical architecture 规定的 e-prop 资格迹接入。

这一轮仍然不要求学习成功。

目标只是验证：

```text
forward
→ local eligibility
→ delayed g_eff
→ delta
→ parameter update
```

能够执行。

至少覆盖：

- `W_ij`
- `b`
- `W_c`
- `b_c`
- Adapter parameters

每个具体 parameter tensor 单独维护 eligibility。

验证：

```text
e(t)
=
lambda * e(previous)
+
local derivative
```

以及：

```text
lambda = exp(-Δt / tau)
```

能够运行。

---

# Phase 14 — 离散控制 eligibility

对 route / hand 等具有离散控制的输出，不能对 hard threshold 求导。

必须实现 canonical architecture 的：

```text
p = sigmoid((q - threshold) / tau)

d_logpi = (a - p) / tau
```

以及：

```text
e_ctrl
=
lambda * e_ctrl(previous)
+
d_logpi * ∂q/∂parameter
```

这里只验证：

- 数值能够形成；
- eligibility 能保存；
- 延迟 goodness 到达后能触发更新；
- 不要求 world 可微；
- 不要求 threshold 可微；
- 不要求 counterfactual；
- 不要求 reset world。

---

# Phase 15 — 最小整机 Smoke Test

最后才做第一版 ACNT 的整体 smoke test。

这里不要运行 Stage 0。

只建立一个非常小的 mock runtime，例如：

```text
create several Blocks
↓
create six organ Blocks
↓
feed mock eye / ear input
↓
run Blocks
↓
generate route
↓
modify active set
↓
generate hand / speak / goodness outputs
↓
generate mechanical logs
↓
inject optional mock g*
↓
execute one e-prop update
↓
continue several ticks
```

验收标准只有：

> **整个流程能够连续跑若干步而不报错，并且关键状态变化符合架构规定。**

---

# 四、测试原则

测试必须跟实现同步进行。

不要：

```text
先写完整 ACNT
→ 最后一次性 test
```

应采用：

```text
Block
→ test

Block communication
→ test

active
→ test

Block system
→ test

Adapter
→ test

Adapter + Block
→ test

ReadIn
→ test

Route
→ test

Goodness
→ test

ReadOut
→ test

e-prop
→ test

whole-system smoke test
```

这样如果出现错误，可以知道错误属于哪一层。

---

# 五、不要为了“跑起来”而偷换架构

“本轮只要求跑起来”不意味着允许使用假实现绕过核心公式。

允许 mock：

- 外部图像；
- 外部音频；
- world；
- DirectInput；
- VocalTractLab；
- g* teacher；
- mechanical executor。

但不能 mock 掉：

- Block update；
- Block communication；
- active；
- Adapter forward；
- route mask；
- goodness；
- eligibility；
- `q + noise + threshold`；
- `d_logpi`；
- 参数 update。

这些就是这一轮真正需要实现的 ACNT 本体。

---

# 六、工程风格

第一版以：

```text
正确
简单
可运行
可测试
可读
```

为优先级。

不要提前进行：

- 大规模性能优化；
- CUDA kernel；
- 分布式训练；
- 复杂配置系统；
- Registry 套娃；
- Factory 套娃；
- Plugin 系统；
- 为未来可能需求建立大量抽象层；
- 复杂 GUI；
- 数据集基础设施。

代码文件数量保持克制。

必要数学公式旁边写清楚注释，使代码能够和 canonical architecture 一一核对。

---

# 七、每个 Phase 完成后的输出要求

每完成一个 Phase，给出：

1. 修改了哪些文件；
2. 实现了什么；
3. 对应 canonical architecture 哪一部分；
4. 执行了哪些 test；
5. test 结果；
6. 当前还没有实现什么；
7. 是否发现规范中存在无法无歧义实现的问题。

如果测试失败：

> 先修复当前 Phase。

不要带着已知失败继续向后实现。

---

# 八、最重要的停止条件

出现以下情况必须停止并问我：

1. canonical architecture 的公式存在无法唯一解释的 tensor shape；
2. 两条 canonical 规则互相冲突；
3. 必须增加当前规范不存在的新结构才能继续；
4. Adapter 即将开始设计；
5. 需要在多个合理 Adapter 架构之间做选择；
6. 任何修改会改变 ACNT 当前数学定义。

尤其再次强调：

> **Adapter 选型一定要先问我。不要自行决定。**

---

# 九、本轮最终完成状态

完成本轮以后，预期得到的不是“已经训练成功的 ACNT”。

而是：

```text
ACNT Runtime v0
```

它应该满足：

```text
Block 能运行
Block 能通信
active 能控制计算
ReadIn 能刺激 Block
ReadOut 能读出 Block
route 能改变 active set
goodness 能形成唯一标量
连续和离散 ReadOut 都能形成输出
e-prop eligibility 能运行
延迟 goodness 能触发参数更新
整个最小系统能连续运行
```

到这里立即停止。

**不要自动开始 Stage 0。**

Stage 0 将作为下一轮独立任务进行。