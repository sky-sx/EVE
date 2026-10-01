# 全参数地址自写：首轮在线启动实验

2026-10-01。用户选择优先验证按地址写入；分组方向 F 保留为后续候选。
此前 `FullParameterWriteSurface` 只验证了执行器。本轮接入原版 Block
前向和真实权重转移，仍是小规模机制实验，并未接入完整 ACNT Runtime。

## 前向与权重的唯一生效值

七个普通原版 Block，width=4、hold=3、NLM hidden=2。保留六个器官位置，
第七个普通 Block 接共享 Write Adapter。原版 Block.transition 的 GLU、
非仿射 LN、带年龄/有效性 History 和私有 NLM 均不替换。采用 all-active
同步时序；当前任务连接 Ear/Hand 及 G/自身动作输入，不测试完整器官环境。

所有参数都以同一公式得到实际生效值：

\[
\theta_t=\theta_{origin}+d_t.
\]

origin 是当前参数坐标原点，d 是实际写入累计的偏移；两者之和是唯一
生效权重。不是两个独立生效的网络，也不是一个不可修改的控制网络。
前向中所有注册 Parameter 都由该值替换，包括 SharedWriteAdapter 本身。
Write 可通过改变 d 修改任意坐标的实际权重。

共同生成器接收第七个 Block 的 z、当前参数值及三个固定地址特征：
参数张量编号、张量内坐标、相对张量大小。对每个地址输出一个标量：

\[
u_{t,p}=g_{\theta_t}(z_t^{(6)},\theta_{t,p},address_p),
\quad |u_{t,p}|\le s.
\]

所有命令使用同一旧权重快照生成，再一次提交：

\[
\theta_{t+1}=clip(\theta_t+u_t,-4,4).
\]

主体状态与 History 不回算、不清空。clip 会使边界处控制受限，本轮
只验证全地址覆盖和有界幅度下的学习，不宣称边界处依然满秩可控。
初始 Write 最后一层为零，实现初始零写入，避免随机写入先破坏前向。
按地址生成不证明这个 81 参数的共享函数能表达任意 3,896 维更新。

## 实际参数覆盖

|部分|实际参数数|Write 可写|
|---|---:|---|
|七个普通 Block|2,912|全部|
|Ear Adapter|40|全部|
|Hand Adapter|823|全部|
|G/自身动作输入投影|40|全部|
|共享 Write Adapter|81|全部，包含输出层自身|
|总计|3,896|3,896 个地址|

与只有六个 Block、Ear/Hand 的同配置核相比，新增 729 个参数，包括
普通 Block、新增连接、反馈投影和 Write，全部计入覆盖。
参数的 requires_grad 仅决定外部启动选择，不决定 Write 的修改权限。

## 外部启动训练仍然存在

只对同一批实际 Write 坐标的 origin 做外部启动更新；不能直接更新
行为参数。Write 本身同时可被其输出自修改，没有另一个新增可训练
控制器网络。外部 Adam 风格启动规则不等于系统已经自主学会训练规则。

联合状态 X=(普通 Block z/History, d)，维数 4,008。phi 是 81 个 Write
origin 坐标。单事件流式敏感度：

\[
T_{t+1}=\partial_X f_t\,T_t+\partial_\phi f_t.
\]

实现用单事件 double-backward JVP 直接计算此乘积，不构造完整 4008²
Jacobian。单事件计算图用后丢弃；无训练轨迹磁带。固定 origin、不裁剪时
与展开导数一致；外部 origin 更新被停止求导，敏感度裁剪后也是近似。
真实 G 和动作在条件导数中视为已观察常数，没有世界导数。

实际动作 score 为 grad_X log(pi) * T。Hand 实际参数来自 d；被外部
启动的 Write 坐标不直接进入当次 Hand logits。收到 G 后，外部规则用
(G-.5)*score 更新 Write origin，固定 .5 是控制变量，并不替代 Core 对 G
的分析。这仍是即时奖励的局部在线启动试验，不是完整终身策略梯度。

## 单向运行协议与资源

先随机输入四样本声音，经过三事件间隔采样二元动作，再接收动作正确
与否的一个 G。主体收到 G、G有效位、自身动作和动作有效位，不收到
私有标签或期望奖励。每轮任务 1,000 次决策、4,000 个事件，严格递增
时间，无神经状态重置、反事实试走、参数回滚或经历重放。

主体联合状态/敏感度/两个优化矩共 1,315,272 bytes，随运行时间不增长。
固定地址特征另占 46,752 bytes；模型 origin 权重另占 15,584 bytes。
此外有固定索引和原版 Block 兼容缓冲区、单事件计算图及临时工作空间。
这里的数值不是进程峰值内存。敏感度和优化矩属于额外训练状态，不
冒充普通 Core H；没有新增经历账本。实验 JSON 行是观察者日志，主体
不读取它。参数位移只保存当前累计状态，没有逐事件修改历史。

## 针对性验证

`tests/test_address_write.py` 验证：全部坐标实际写入，包括 writer 自身；
全参数替换后关闭写入仍与原版 Core 一致；四事件含自写的流式导数与
完整展开一致；禁用写入或切断其信用使外部 Write 行为信用为零；启动
更新不清空联合状态和 History、固定资源、拒绝时间倒退。

完整展开仅用于独立单元测试，不是在线主体的学习方式。
本轮增加六项测试后全仓 229 项通过。

结果见 [首轮报告](../../reports/address_write_online_2026-10-01.md)。

```powershell
python -m experiments.address_write_online --task cue --decisions 1000 --seeds 11 22 33 --output runs/address_write_online/cue_v1
```
