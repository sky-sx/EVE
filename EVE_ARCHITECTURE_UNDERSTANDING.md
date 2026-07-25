# EVE Architecture Understanding

## 1. EVE 是什么？

EVE 是运行在电脑上的可成长数字生命体研究装置。
- 持续接收屏幕、声音、光标、键盘等输入
- 通过本地 LLM + 多个 TNN 协作完成思考与动作
- Memory 保存一切产出及关联
- Dock 将慢速/重复计算蒸馏为 TNN
- 能力从运行中生长，而非预先硬编码

## 2. EVE 不是什么？

- 不是传统 Router/Planner/Tool-Agent 框架
- 不是商业产品或通用平台
- 不是预先规定所有任务类别的固定流程系统
- 不通过多 Agent 角色分工

## 3. 六大模块真实职责

| 模块 | 职责 | 不负责 |
|------|------|--------|
| main | 启动、停止、GUI、配置加载、快照恢复 | 运行时决策 |
| input | 捕获屏幕/音频/光标/键盘，形成 state | 理解、判断、保存 |
| output | 执行通过 Safegate 的动作（mouse/keyboard/speak） | 决策 |
| memory | 保存多模态结果及关联，多图索引检索 | 运行时调度 |
| core | 运行循环、TNNGraph、Safegate、激素、节律 | 训练 TNN |
| dock | 接收训练订单 → 获取数据 → 训练 → 保存 TNN | 运行时加载决策 |

## 4. world / myself / blackboard

- **world**：EVE 对外部环境的认识（场景、对象、检测结果、不确定性）
- **myself**：EVE 对自身状态的认识（权限、资源、激素、TNN、倾向）
- **blackboard**：不同频率节点交换临时结果的共享区（非长期记忆）

三者只在内存运行；Markdown 快照仅用于启动恢复和停机保存。

## 5. MemoryUnit / Catalog / Event / 索引边

- **MemoryUnit** = MemoryID + Payload（文本/图像/音频/轨迹/输出/摘要等）
- **Catalog** = MemoryID → 存储路径/类型/hash/持久状态 的映射
- **Event** = 一组 MemoryID + 文本描述，组织为一次经历
- **索引边** = 时间边、内容相似边、因果候选边等，构成多图索引

MemoryUnit 本身不携带 reward/importance/confidence 等解释字段。

## 6. STM / MTM / LTM

- **STM**：近期新输入和临时数据的 MemoryID 集合
- **MTM**：当前任务主动使用的工作集 MemoryID
- **LTM**：真实 Payload 持久存储位置

三者共享同一 MemoryID 空间，不复制 Payload。STM/MTM 通过 Catalog → LTM 读取实际数据。

## 7. TNN Store 与 Memory 的边界

- TNN 使用独立 TNN ID 空间
- 权重保存在 TNN Store（`eve/memory/TNNweights/`）
- Memory 可保存关于 TNN 的描述、报告和关联，但权重本体不在 Memory
- 不要把 TNN 权重当成普通 MemoryUnit Payload

## 8. Dock / Core / LLM 的职责区分

| 角色 | 负责 | 不负责 |
|------|------|--------|
| Dock | 执行训练订单 → 获取数据 → teacher → 训练 → 评估 → 保存 TNN | 决定训练什么、加载哪个 TNN |
| Core | 维护运行状态、TNNGraph、加载/卸载 TNN、Safegate、激素 | 训练 TNN |
| LLM | 理解、策略、active_tnn 选择、训练候选发现 | 低延迟反应路径 |

训练什么由 LLM/用户/复盘决定，Dock 只执行。

## 9. TNNGraph 为什么是动态图？

因为运行时需要：
- 节点加载/卸载
- 边增加/删除
- 节点替换
- 不同运行频率共存
- 多上游多下游
- 输出缓存与过期
- 活动子图随任务变化

不能是固定 Pipeline。活动图 = 完整能力关系中当前加载运行的子集。

## 10. 活动子图 vs 完整能力关系 vs 运行轨迹

- **完整能力关系**：所有可用 TNN 之间的逻辑承接（通过 SourceRef 表达）
- **活动子图**：当前已加载 TNN 形成的实际运行图
- **运行轨迹**：某时刻的活动节点/边/频率/异常/图变更记录

## 11. TNN 与规则/YOLO/VLM/LLM 的关系

- **TNN** = Tiny Neural Network，可训练的 PyTorch 网络，有参数、forward、训练路径
- **LLM/VLM/YOLO** = 已有大模型，可作为教师、基线或通用节点
- **规则** = 可用作教师，但不能冒充正式 TNN
- TNN 的能力应被训练出来，不应作为固定规则写死

## 12. Safegate 为什么必须是硬边界？

- 控制真实电脑输出的最后防线
- 不可被任何学习能力绕过
- Esc 急停最高优先级
- 人类接管立刻冻结输出
- 所有输出路径必须经过 Safegate

## 13. 激素与循环节律

六大激素（dopamine/serotonin/norepinephrine/oxytocin/cortisol/acetylcholine）：
- 根据成功/失败/表扬/批评/风险/新奇度逐轮更新
- 影响 LLM 上下文、倾向和 10-20 秒大循环间隔
- 不能直接产生动作或绕过 Safegate
- 当前阶段用简单线性更新，不建复杂生物模型

## 14. 本轮必须完成的代码

- main 启动停止、GUI、配置
- input 捕获+Buffer、人类活动检测
- output disabled/mock/real 三模式
- Safegate 完整实现
- world/myself/blackboard 运行时+快照
- MemoryUnit/Catalog/索引/Event/STM/MTM/LTM/Retrieval
- TNN 基类/Store/保存加载/描述
- TNNGraph 动态图/调度/缓存/轨迹
- LLM Prompt/解析/错误反馈
- 激素更新/节律
- Dock 训练订单/教师/训练/评估
- 睡眠复盘基础流程
- 朴素 GUI（连接真实状态）
- 工程测试覆盖关键职责

## 15. 留到后续实验的问题

- 红圆/蓝三角/文本条件三阶段（代码完成后的验证实验）
- 具体 TNN 的长期训练参数
- 激素映射函数精确参数
- 复杂记忆合并/遗忘策略
- 云端 LLM 调用时机
- TNN 自动拆分/合并
