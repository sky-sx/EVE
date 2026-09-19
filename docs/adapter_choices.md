# Phase 5 — 原始 Adapter 候选（现已完成选型）

用户已选择 eye=CNN+Linear（无平均池化），ear=固定窗口Linear，hand=两层MLP，
speak/goodness/route=Linear。当前生效决定与待确认事项见 `clarifications.md`。
下方保留选型前的候选记录，不代表当前实现采用池化方案。

根据用户任务的强制设计门：在得到选择前不编写正式 Adapter。
本文件只是可供确认的候选，不是 canonical 的新增部分，也不表示已经选定。
所有候选都保持六个器官各占一个不同 Block。

记 n 为各自 Block 的 neuron_size，M 为总 Block 上限，D 为离散键/鼠标按键数，
C 为连续鼠标控制数，T 为单次原始音频的采样点数。D/C/T 是接口配置，不限制为 Stage 0 的按键集合。
层数按有参数的 Linear/Conv 层计数；池化、Flatten、ReLU 不增加可训练层。

| 器官 | 输入 → 输出 | 最简单可运行候选与理由 | 其他候选、层数与成本差异 |
|---|---|---|---|
| eye | 完整 RGB 1080p `[3,1080,1920]` → `[2n]` | **建议：adapter 内平均池化到 `[3,18,32]`，Flatten，再 1 层 Linear**。完整截图到达 adapter；计算规模小，保留粗空间位置，足以验证输入链，但会丢失小字等细节。 | **池化 + 2 层 MLP**：`1728→128→2n`，中间 ReLU，加入非线性，参数和计算通常更高。**小 CNN**：2 层步幅 Conv2d + 池化 + Linear，共 3 个可训练层，可学习局部空间模式，但在 1080p 上卷积需要更多激活内存和计算。**全图 Flatten + Linear**：只有 1 层、结构最直接，但矩阵极大，不建议作为轻量 smoke 的首选。 |
| ear | 原始 PCM 波形 `[channels,T]` → `[2n]` | **建议：固定长度窗口 Flatten + 1 层 Linear**。不引入频谱或语义前端，最容易核对输入与资格迹参数清单。T 必须固定配置。 | **2 层 MLP**：`channels*T→128→2n`，中间 ReLU，增加非线性但仍需固定 T。**小 Conv1d**：2 层卷积 + 固定输出大小池化 + Linear，共 3 个可训练层，能学习局部时序模式并接收变长窗口；权重共享，计算成本取决于窗口长度与通道数，不能断言总比 Linear 贵或便宜。 |
| hand | `[n]` → `[D+C]`，前 D 项是连续倾向 q，其余是连续鼠标控制 | **建议：1 层 Linear**。直接产生各控制坐标，容易枚举参数、核对独立 q 与连续输出，参数量小。 | **2 层 MLP**：`n→64→D+C`，中间 ReLU；可表达非线性组合，通常增加参数和乘加。这里不输出硬 01，noise/threshold 属于随后执行链。 |
| speak | `[n]` → `[30]` 连续控制参数 | **建议：1 层 Linear**。即可形成所需 30 维控制向量，成本低，足够本轮验证。 | **2 层 MLP**：`n→64→30`，中间 ReLU；允许非线性控制映射，通常增加成本。当前不接发声器官或声音生成。 |
| goodness | `[n]` → `[1]`，随后统一 clamp 到 `[0,1]` | **建议：1 层 Linear**。单标量预测和 teacher 校准关系最容易核对，成本低。 | **2 层 MLP**：`n→16→1`，中间 ReLU；可拟合非线性关系，通常增加成本。两者都只形成一个标量，校准公式和参数范围须先澄清。 |
| route | `[n]` → `[M]` 连续倾向 q | **建议：1 层 Linear**。每个候选 Block 各有一个输出坐标，参数和计算小。 | **2 层 MLP**：`n→64→M`，中间 ReLU；可表达非线性组合，通常增加成本。两者都不加 Softmax，随后逐坐标 noise/threshold，route 自身由运行规则保持 active。 |

建议的低成本首版组合是：eye 采用 adapter 内池化 + Linear，其他五类采用 Linear。
这仅针对“跑通并验证本体”的当前目标，不声称拥有合适的长期感知能力或学习效果。
如果希望第一版就保留更多可学习的视觉/音频局部结构，可分别选择小 CNN / Conv1d。

成本可按形状直接核对：Linear 从 I 维到 O 维需要 `I*O+O` 个参数，
2 层 MLP 隐层 H 需要 `I*H+H+H*O+O` 个参数。
它们的准确相对成本取决于 n、输出数和隐层大小，不能无条件认为 MLP 更贵。
以上是由候选形状作出的估算，未运行任何 Adapter 性能 benchmark。

例如仅用于容量说明，若 eye 的 n=16，则全图 Linear 权重为
`3*1080*1920*32=199,065,600` 个 FP32 数，约 796 MB（十进制，不含资格迹）。
池化到 18×32 后则为 `3*18*32*32=55,296` 个，约 221 KB。
池化仍须读取完整输入，约 24.9 MB；不能把矩阵变小理解为跳过 1080p 输入。
此例不是 Block 配置选择，也不是 Stage 0。

这些候选只使用普通算子；已核对官方算子定义：
[Linear 的输入输出和权重尺寸](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)、
[Conv1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html)、
[AdaptiveAvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool2d.html)。
它们仅用于确认库接口，ACNT 架构依据仍然只有用户 canonical 原文。

后续 Adapter 单体拟支持 FP32、单样本及带前置 batch 维的测试；
接到 Core 时一次提供一个样本，不把 batch 维混进 z/a/o。
eye 的 RGB、音频 channels/T、CNN 通道数和 MLP 隐层宽度均为当前提议，尚未固化为实现。

**请决定：这一版 eye、ear、hand、speak、goodness、route 分别选择哪一种 Adapter？**

此外，`phase_report.md` 中四项数学问题必须在实现相应部分前由用户澄清：
校准损失、资格迹输出轴收缩、ticktime 到 tau 的映射、goodness 校准参数范围。
