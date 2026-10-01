# 自修改启动训练：从原版前向得到一个可测的候选

2026-10-01。实现：[self_write.py](../../acnt/self_write.py)。
实验：[self_write_online.py](../../experiments/self_write_online.py)。
这是一种候选训练规则，不是终生学习已经解决的声明。

用户进一步要求 Write 精确覆盖包括自身在内的全部参数。**本局部原型不
满足该要求**：新增 718 个参数中 526 个由外层规则更新，Write 自身不在
写入范围。全参数接口要求与参考见 [FULL_PARAMETER_WRITE.md](FULL_PARAMETER_WRITE.md)。

## 两种训练机制要区分

旧 original_online 实现直接用动作信用更新原版行为参数，没有学得参数写入
控制器。这里加入第七个普通原版 Block，并将 Write Adapter 接到其输出，
它们生成实际生效的参数修改；启动训练只更新这套控制器的 φ。

被写行为参数 θ 是运行状态的一部分。既不向控制器提供参数梯度/正确修改，
也不由启动训练直接更新 θ。θ 改变只来自 Write Adapter 的输出。
控制器的实际影响必须经过写入：其普通出边进入六个行为 Block 的权重
固定为零，避免控制器绕过写入通过普通连接直接改变行为。

## 保留了什么，限制了什么

七个 Block 都调用原版 Block.transition：旧来源快照、GLU、非仿射 LN、
有界预激活 History、时间/有效位输入、私有 NLM 完全沿用原算子。
新增 Block 属于 Core；其 z/History 就是经历压缩的内部状态。

当前只测试全激活、同步事件；保留六个器官位置，但仅使用原版 Ear/Hand
Adapter。未接完整 Runtime 的 Eye/Speak/Goodness/Route 执行接口，不验证异步
路由、图像、语音或连续动作。它是原版算子的实验核，不是 Runtime 替代品。

第七个 Block 与其它 Block 使用相同类型和前向。它参与参数写入，是由
Write Adapter 的连接决定的角色，没有独立的“学习 Block”架构或算子。

LearningReadIn 只输入已发生的 [G, G有效位, 自己的动作, 动作有效位]。
当前 Core 输出通过原版连接进入接有 Write 的普通 Block。不输入目标、私有动作
关联、未来反馈、梯度方向或原始历史账本。没有 Goodness 值预测器。

当前写入目标仅为 Hand Block <- Ear Block 的一块连接矩阵，以及 Hand
第一个离散输出的偏置，共 33 个实际生效参数。所有其它行为参数固定。
控制器输出 u、v、偏置目标、写入门，采用：

\[
T_W=2\tanh(u)\tanh(v)^T,\quad T_b=4\tanh(c),\quad
\rho=0.02\sigma(g),\quad \theta'=\theta+\rho(T-\theta).
\]

这是受限的低秩目标写入，不是任意参数更新。凸组合使这组行为参数保持
在预先界定的范围内，但不保证整个网络或实际世界稳定。
主体通过普通 Core Block 的有限状态形成写入判断，不积累经历记录。

## 沿时间训练写入控制器

把所有 z/有界 History 和实际行为参数 θ 合成 X，共 145 个标量：

\[
X_{t+1}=F_\phi(X_t,o_t,G_t,a_t),\quad
M_{t+1}=\partial_XF_\phi M_t+\partial_\phi F_\phi.
\]

一事件图求完局部 Jacobian 就释放，只携带数值 M。M 的 θ 行明确记录
过去写入怎样受 φ 影响；后来的神经前向读取这些 θ，所以训练信用穿过
实际参数写入，不是旧算法的 stop-actor-update 切线。

对 Hand 的实际 Bernoulli 动作，得到控制器的条件策略信用：

\[
e_t^\phi=(\partial_X\log\pi(a_t|X_t))M_t.
\]

第一轮即时反馈的启动方向为 (G-0.5)e；0.5 是固定的预动作参照，不是
从反馈分析得到的平均值。原始 G 同时进入接有 Write 的普通 Block，其分析能力
由训练形成。控制器方向经过范数限制，再用小步 Adam 式上升更新 φ，
单次外层位移不超过 0.02。当前实际 X、History、时间戳和 M 均不重置。

事件顺序：用已提交状态产生动作 → 世界给即时 G → 用旧 φ 更新 Core
状态并提交行为参数写入 → 用该动作的已保存数值信用更新 φ。
该信用只在当前即时反馈事件内使用，随后释放；无跨事件动作关联表。
当前写入只影响后续动作，不能获得先前动作的伪造信用。

固定 φ、条件于已实现的观察/动作/G 时，未截断 M 是准确的神经轨迹导数，
测试与整体短历史自动微分和有限差分比较。它不对世界的离散动作后果
硬求导；更新 φ 后使用 stop-outer-update 近似。即时奖励乘当前动作条件
信用也没有纳入所有过去随机动作对未来反馈分布的影响，不是完整终生
回报的无偏梯度。数值验证通过不意味着这种启动方向必定行为有效。

## 资源和验证

φ 含接有 Write 的普通 Block、反馈输入投影、Write Adapter 的 526 个参数。
永久数值学习资源为 X、145×526 的 M、两组外层更新矩，共 **309,868 字节**，
不随运行时长增长。这不含网络参数、局部计算图、临时 Jacobian 和评估日志。
主体的经历状态与训练器的敏感度分别报告；稠密 M 仅适合小模型。
敏感度范数上限 1000，发生截断时导数变为近似。

测试检查与 canonical Core 前向一致、权重写入后实际参与前向、
跨多次写入的整体导数/有限差分、切断写入后信用消失、外层更新不改变
已发生状态，以及永久资源和 History 槽数固定。

实验使用单条连续模拟轨迹，零重置、零重放。即时单标量反馈用于隔离
“启动训练能否教会写入器”；不使用旧固定延迟账本。
恒定正确动作任务仅证明能学出有用写入；条件提示任务才检验上下文相关
行为，交替规律任务仍不是长期保留所有旧能力的实验。

```powershell
python -m pytest tests/test_self_write.py -q -p no:cacheprovider --basetemp=.test-tmp/self-write
python -m experiments.self_write_online --seeds 11 22 33 --modes full --decisions 1000 --task constant
python -m experiments.self_write_online --seeds 11 --modes frozen cut_write constant_controller --decisions 500 --task constant
python -m experiments.self_write_online --seeds 11 --modes full --decisions 1000 --task cue
```

frozen 保留写入但不训练 φ；cut_write 保留写入数值、切断它的元信用；
no_write 禁止写入；constant_controller 让 Write 只接收零输入，仍训练其
偏置，检查恒定任务是否只是学会静态写入。对照必须按相同决策数比较。
评估日志只供旁观分析，主体不能读回它们。

已完成的正、负结果见 [实验报告](../../reports/self_write_bootstrap_2026-10-01.md)。

后续 [输入诊断](../../reports/self_write_input_diagnostic_2026-10-01.md) 发现原版
无写入状态保有可读提示，旧写入会损伤下游表示。现已增加 `write_mode=delta`
作为零修改不改变参数的对照，使用 clip(θ+ρD)；边界处采用截断的局部导数。
target 模式仍是早期实验的明确基线。增量模式保留更多可读信息，但当前
自修改训练仍未学会条件任务，不能将这一局部改进写成训练成功。
