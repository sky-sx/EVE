# 当前架构边界与实现核对

本文件按[用户当前 Local Plasticity 架构原文](canonical_architecture.txt)整理边界，不具有高于原文的优先级。旧文档中的 e-prop、固定反馈、`g_bar`、`(a-p)/τ` 学习因子、sigmoid goodness 和 `γ=sigmoid(Δt/ticktime)` 均不是当前正式架构。

## Block 与器官

Block 的 `r`、`a`、历史 `A/At`、CfC 递推及 `z` 依照原文。原文写的是 `γ=sigma(delta_t*ticktime)`，时间戳 `At` 为毫秒。当前 Core 在一个调度事件里读取统一的旧 `z` 快照，避免遍历顺序影响；这是当前调度实现，不作为原文以外的新认知模块。ReadIn 有新输入时无视 route 强制更新一次，随后 `o` 归零。route Block 永久 active；goodness Block 的 active 由人控开关决定。

## 唯一评价流

`g=clamp(A_g(z_g),0,1)`，只有一个标量输出。同刻存在准确 `g*` 时，其他 Block/Adapter 的 `g_eff=g*`；否则取 `g`。B_g 和 A_g 不以自身 `g_eff` 做自我奖励。同刻有 `g*` 时，二者可用 `c_g=g*-g` 与各自局部活动及局部塑性状态校准；无 `g*` 时不做这种校准。`c_g` 不送到其他连接。goodness 人控开关关闭时，输出只进机械日志；外部 `g*` 仍可作为其他连接的 `g_eff`。

## 局部塑性

正式 ACNT 只用各局部连接自己的 `w_c`、真实 pre/post 活动和 `e_c`；`g_eff` 只在参数更新时到达。`F_e` 与 `F_w` 的内部状态维度和具体公式尚未由规范固定。当前 `acnt/plasticity.py` 用每个参数元素一份状态和可替换的 `CorrelationRule` 候选实现运行骨架；候选的保持系数、相关活动统计与权重步长是实现参数，不等同于架构的定式。延迟好度只调制当前仍保留的局部状态，不回放计算图。

正式路径不维护 e-prop eligibility、Jacobian、BPTT、反向敏感度、策略梯度、固定随机反馈或第二条全局 reward。e-prop 仅允许在将来独立的小型科学对照系统中作 oracle/benchmark，不能为正式模型提供更新信号。

## 离散控制与机械边界

hand 的键、鼠标离散按键以及 route 的每个目标 Block 都先得到独立连续 `q`，分别采样独立 `ε~Logistic(0,τ)`，再取 `a=1[q+ε>θ]`。相应概率 `p=sigma((q-θ)/τ)` 只是概率解释。route 自身强制 active 的硬规则不参与学习。终端真实出现的 `q/ε/a` 可进入自身局部状态，不能当作额外 world 反馈返回 Core。多个离散量不使用 Softmax，也不要求重置 world 或尝试反事实动作。

所有 ReadOut 先生成最终信号，再检查人控执行开关。关闭时只记录机械日志；执行成功或失败也不主动把状态送回 Core。speak 与 goodness 的连续输出不经过 0/1 阈值。
