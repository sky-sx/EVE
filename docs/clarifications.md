# 当前用户澄清（优先于 canonical 原文中的冲突描述）

本文件记录当前用户澄清，包含 Phase 5 后的最小修复要求；与旧阶段记录冲突时以本文件为准。

## 已确认

- eye：CNN + Linear，不能先平均池化到 18×32。
- ear：固定长度原始音频窗口 + 一层 Linear，目前只需 mock 音频。
- hand：两层 MLP，完整 82 键键盘和鼠标控制。26 个字母键加左键 click 仅是训练器读取日志的筛选条件，不裁剪 Adapter 的输出或完整机械日志。
- speak：一层 Linear，形成约 30 维连续输出。
- goodness：一层 Linear，输出唯一标量。
- route（口述 root）：一层 Linear。
- Block 局部资格项采用 neuron-specific learning signal 的 VJP：
  `T_i,theta = sum_k L_i,k * [partial z_i,k / partial theta_i]_local`，不再对 `mean(z)` 求导。
- A_g 校准采用 `1/2*(g-g*)²`，只更新 A_g。
- B_g 参数继续走一般 goodness 参数更新，不走 A_g 的校准损失。
- g 的出现频率高于稀疏 teacher g*，没有同刻 teacher 时 g_eff 可以采用 g。
- ticktime 表示一个 tick 的正数秒数，允许浮点；Logistic noise 的 tau 与塑性 tau 都直接等于 ticktime。
- 用户后续确认：`gamma=sigmoid((delta_t_ms/1000)/ticktime)`；Core 按 ticktime 调度，
  ReadIn 新输入依照原强制规则可提前触发一次更新。时间戳继续以整数毫秒保存。

## 最新确认的 teacher 与 hand 规则

1. 只校准和 g* 同一时间产生的当前 g，不回溯此前预测，也不复用该标签校准以后预测。
   同刻有 teacher 时，A_g 用 `1/2*(g-g*)²` 校准，其他参数采用 `g_eff=g*`。
   同刻没有 teacher 时，A_g 不更新，其他参数采用 `g_eff=g`。
2. 其他参数的唯一更新信号保持 `delta=g_eff-g_bar`，g_bar 为训练用滑动均值。
3. hand 使用完整 82 键接口，修饰键各自独立，可同时形成组合键；鼠标离散按键与
   连续 dx/dy 均保留。所有最终控制均写机械日志。A–Z 和左键 click 的筛选只属于
   训练器读取端，本轮不会自动进行 Stage 0。

这些问题均已明确，允许继续 Phase 10–15。

## 已选网络的具体大小

这些是小规模首版实现参数，不新增器官或认知模块。

- eye：完整 `[3,1080,1920]` → Conv2d(3,8,k11,stride10,pad5) + ReLU
  → `[8,108,192]` → Conv2d(8,16,k7,stride6,pad3) + ReLU
  → `[16,18,32]` → Flatten → Linear(9216,2*n)。没有平均池化。
- ear：默认单通道固定 1600 个采样点，可显式配置窗口长度与通道数；
  Flatten → Linear(channels*T,2*n)，不附加音频语义前端。
- hand：Linear(n,64) → ReLU → Linear(64,D+C)。默认 D=82 个键盘键 + 3 个鼠标按键，
  C=2 个连续鼠标方向。键位名称在实现中显式列出，不隐式生成组合键类别。
- speak/goodness/route：分别 Linear(n,30)、Linear(n,1)、Linear(n,M)。

Adapter 保留局部求导能力，Phase 13–14 已接入实际资格迹衰减和参数更新。
当前 A_g 的局部校准资格项只服务于同刻 g/g*；不把旧预测或旧 teacher 混入当前校准。

## 最小修复：random e-prop、快照传播、输入脉冲与 sigmoid Goodness

非仿射 LN 使 mean(z) 在精确算术下为零，旧 `grad(mean(z))` 无法提供有效内部学习。
主动力学及 LN 均保持原样；改用 `observe_vector(z, L, now_ms=...)` 的 PyTorch VJP，
即 `autograd.grad(z, parameters, grad_outputs=L.detach(), allow_unused=True, retain_graph=True)`。
不构建完整 neuron × parameter Jacobian，不使用 BPTT；来源 z 和旧 A 均保持 detached。

每个 Block 对 hand、route 各有固定 `B_i^(r)`，形状 `[n_i,D_r]`。
`Runtime.enable_plasticity(feedback_seed=0)` 用独立 CPU generator 生成零均值正态分布、
标准差 `1/sqrt(D_r)` 的矩阵，并作为非训练 buffer 保存；不消耗动作随机数，不进入 forward，
不属于 theta，也不接受 Goodness 更新。恢复 state_dict 前需以相同结构启用塑性。

`DERIVED: random e-prop feedback is a learning-layer implementation choice.`

离散实际采样仍为 `q + Logistic noise + threshold`，`d_logpi=(a-p)/tau`。
每个本轮更新的 Block 使用 `L_i=sum_r B_i^(r) d_logpi_r`；逐 ReadOut 累加 VJP 与对 L 求和
在数学上等价。离散 adapter 仅累积 `sum d_logpi * dq/dtheta_adapter`，不再直接向 Block
累计离散控制梯度，防止与反馈重复。ReadIn adapter 经当前 o 的局部图获取同一 VJP。
未更新 Block 没有新局部项，保留历史 tag 的时间衰减；没有离散采样时不凭空产生内部项。
speak 和 hand dx/dy 保留现有连续输出与均值导数路径。

资格迹为 `E(t)=exp(-delta_seconds/tau)*E(previous)+T(t)`，tau 为 Block.ticktime。
延迟 Goodness 仍使用唯一 `delta=g_eff-g_bar`，合并内部、连续与离散资格迹更新参数，
再乘 rho、更新 g_bar；没有新增 reward、confidence、critic 或 route-goodness。

Core.step / Runtime.update_blocks 在每个调度事件开始建立一次 committed z 快照。
所有 due/forced Block 都读取该快照，包括自连接。Python visitation order 不影响本轮结果；
每次 update_block 独立调用则视为单独事件。ReadIn 强制激活时只把其旧 z 加入本轮来源，
新 z 要到下一传播轮且该来源参与 active 集合时才可被看到，轮末恢复原 active。

ReadIn 新样本只形成一次 o 脉冲。Core 完成 Block 更新后用全零、无图的新 buffer 替换 o，
不会原位破坏本轮返回的局部图；该图仍支持随后 hand/route 的 VJP，提取后释放。
下一轮无输入既不调用 adapter，也不会重放旧 o 或重复生成 ReadIn adapter 资格项。
全零图像/音频是有效新事件：仍调用带 bias 的 adapter，强制更新一次并清零 o。

Goodness 输出为 `g=sigmoid(A_g(z_g))`；数学上 `0<g<1`。FP32 在极端 logit 下仍可能
舍入到 0/1，普通有限 logit 不再受到旧 clamp 区间外零梯度的限制。
同刻 teacher 校准仍为 `L_g=1/2*(g-g*)²`，只更新 A_g，teacher/g_eff/g_bar 时序不变。
Plasticity 中保护参数上下界的 `value.clamp(...)` 保留。

回归覆盖显式 Jacobian 与 VJP 等价、多 ReadOut 求和、固定反馈与非零普通 Block/ReadIn tag、
延迟更新、无离散重复梯度、顺序/ID 重标号不变性、输入脉冲与全零事件，以及 sigmoid 校准。
这些测试验证局部学习链，不代表已经验证 Stage 0 任务收敛。
