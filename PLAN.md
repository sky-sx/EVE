# EVE个人研究实验计划

## 使用方法

每个Phase只验证一个核心问题。

每轮开始前：

- 读取本Phase；
- 看清“本轮不做什么”；
- 必要时复制工程目录作为备份；
- 不因为看到后续计划而提前实现后续模块。

每轮结束后只记录：

- 实际修改；
- 实际运行结果；
- 失败位置；
- 下一轮是否需要调整。

---

# Phase 1：最小Mock运行闭环

## 研究问题

EVE能否在不读取真实屏幕、不加载模型、不操作电脑的情况下，完成：

```text
假state
→ ActionCandidate
→ Safegate
→ MockOutput
→ OutputResult
→ 日志
```

## 主要假设

如果最小闭环稳定，后续真实输入、TNN、LLM和Memory都可以接入统一动作与反馈通道。

## 最小实现

建议文件：

```text
eve/main.py
eve/state.py
eve/core/__init__.py
eve/core/loop.py
eve/core/safegate.py
eve/output/__init__.py
eve/output/mouse.py
eve/output/keyboard.py
eve/output/speak.py
tests/test_safegate.py
tests/test_runtime_loop.py
```

### state.py

只保存本轮需要的状态：

- cold_started；
- emergency_stopped；
- output_mode；
- mouse/keyboard/speak权限；
- blocked_until_ns；
- 当前假state；
- pending ActionCandidate；
- 最近OutputResult。

不要提前加入world、myself、blackboard、TNN或激素。

### ActionCandidate

第一版只保留：

- action_id；
- kind；
- payload；
- created_at_ns；
- valid_until_ns；
- origin。

### Safegate

实现：

- 默认disabled；
- 权限判断；
- Esc急停；
- 人类接管冻结5秒；
- 重复人类活动刷新冻结时间；
- 动作过期阻断；
- EVE-origin事件不触发自我冻结。

### Output

仅实现：

- disabled：返回阻断结果；
- mock：记录动作但不调用系统API。

### loop.py

先提供同步、可测试的`run_once()`。

流程：

```text
检查cold start
→ 读取候选动作
→ Safegate
→ 对应MockOutput
→ 返回OutputResult
→ 写JSONL日志
```

可以在`run_once()`稳定后再加最简单循环。

### main.py

默认启动后：

- 不捕获屏幕；
- 不加载YOLO；
- 不加载LLM；
- 不操作键鼠；
- 使用mock或disabled；
- 支持cold start、一次假动作、Esc急停和退出。

旧YOLO演示不得继续作为正式main入口。

## 观察结果

- 默认启动是否绝对不会真实操作电脑；
- 同一个动作在不同权限状态下是否得到可解释结果；
- Esc后pending和新动作是否都被拒绝；
- 日志能否完整看到输入候选、安全判断和输出结果；
- 循环停止后是否还有后台线程。

## 测试

至少覆盖：

- disabled阻断；
- 无权限阻断；
- mock允许；
- emergency stop最高优先级；
- 五秒冻结；
- 重复接管刷新；
- 过期动作拒绝；
- EVE-origin不自我冻结；
- 未cold start不执行；
- 循环停止；
- 无真实输出函数调用。

## 本轮不做

- 真实屏幕输入；
- YOLO；
- 本地LLM；
- Memory；
- blackboard；
- TNN；
- Dock；
- 激素；
- 完整UI；
- 真实鼠标、键盘或语音输出。

## Phase 1结束条件

看到一条可重复的Mock日志链，并且测试证明真实输出没有发生。

---

# Phase 2：真实屏幕与光标buffer

## 研究问题

现有屏幕和光标捕获能否稳定形成统一时间轴下的最近1秒state？

## 最小实现

优先复用：

```text
eve/input/screen_capture.py
eve/input/cursor_capture.py
```

新增或重写：

```text
eve/input/buffer.py
eve/input/capture.py
tests/test_input_buffer.py
tests/test_capture_timing.py
```

先只接：

- 屏幕；
- 光标；
- 单调时间戳。

提供：

- latest；
- 最近1秒；
- 指定时间范围；
- 安全停止；
- 丢帧和实际频率统计。

## 观察结果

- 1080p捕获的实际FPS；
- P50/P95捕获间隔；
- 1秒buffer内帧数；
- 屏幕和光标时间对齐误差；
- 停止后线程是否退出；
- 长时间运行是否内存增长。

## 本轮不做

- 音频；
- 键盘状态；
- 活动窗口；
- YOLO；
- Memory写入；
- 真实输出。

## 结束条件

连续运行受控时间后，能够读取latest和最近1秒数据，并打印真实频率与内存占用。

---

# Phase 3：最小world、myself与blackboard

## 研究问题

不同频率节点能否通过最小运行时状态交换结果，而不形成固定调用链？

## 最小实现

```text
eve/runtime_state.py
eve/core/blackboard.py
tests/test_blackboard.py
```

### world

只加入Phase 3真实需要的外部状态。

### myself

只加入：

- 权限；
-资源；
- 当前运行模式；
- loaded/available模型摘要。

### blackboard

每项结果暂定包含：

- entry_id；
- kind；
- producer；
- reference_time_ns；
- produced_at_ns；
- valid_until_ns；
- payload。

支持：

- 写入；
- 按kind/producer读取；
- latest；
- TTL过期；
- 清理。

## 实验

使用两个不同频率的假节点：

```text
节点A产生目标位置
节点B读取目标位置并产生ActionCandidate
```

验证B不直接调用A，而是读取blackboard。

## 本轮不做

- 持久化Memory；
- TNN权重；
- LLM；
- 完整world schema；
- 复杂冲突解决。

---

# Phase 4：单个TNN运行

## 研究问题

一个真实的小型神经网络能否通过descriptor加载、按频率读取输入并把命名输出写入blackboard？

## 最小实现

```text
eve/core/tnn_store.py
eve/core/tnn_runtime.py
eve/memory/TNNweights/<tnn_id>/
tests/test_tnn_runtime.py
```

TNN artifact至少包含：

```text
descriptor.json
tn_structure.json
weights.pt
```

descriptor只实现当前用例需要的字段。

第一个TNN可以使用合成输入或非常简单的图像分类/坐标回归任务，但必须是真实PyTorch网络，而不是硬编码函数。

## 观察结果

- 加载时间；
- 单次推理P50/P95；
- 按频率运行是否稳定；
- 输出是否正确进入blackboard；
- 卸载后是否释放；
- 输入缺失时是否暂停或返回明确状态。

## 本轮不做

- 多TNN图对象；
- active_tnn LLM选择；
- Dock训练；
- 动态binding；
-真实动作。

---

# Phase 5：最小Memory

## 研究问题

一次输入、模型输出、动作候选和结果能否分别保存为MemoryUnit，并通过MemoryID重新读取真实对象？

## 最小实现

```text
eve/memory/memorizer.py
eve/memory/catalog.py
eve/memory/indexes.py
eve/memory/LTM/
tests/test_memory_basic.py
```

实现：

- 分配MemoryID；
- 图片、文本和JSON payload写入LTM；
- Catalog映射；
- STM近期ID集合；
- MTM当前任务ID集合；
- 简单时间索引；
- 最小Event；
- 按ID读取。

## 核心验证

```text
MemoryID
→ Catalog
→ LTM真实对象
```

STM和MTM不得保存第二份权威payload。

## 本轮不做

- merge；
- lazy redirect；
-图压缩；
- 遗忘；
- 复杂语义索引；
- 睡眠整理。

---

# Phase 6：首个受控小游戏与数据记录

## 研究问题

EVE能否在一个自包含、可重复的桌面小游戏中记录完整经历，为训练TNN提供数据？

## 推荐任务

优先使用红球或气球点击小游戏：

- 场景可控；
- 成功标准明确；
- 输入和动作时间容易记录；
- 可设置固定seed；
- 可比较速度和命中率；
- 失控风险较低。

## 需要建立

```text
game/
experiment runner
baseline policy或人类示范
输入截图/轨迹
动作开始时间
动作结果
成功/失败
MemoryID关联
```

第一轮可以：

- 只回放，不真实点击；
- 或在独立小游戏窗口中人工授权真实点击。

## 时间语义

必须区分：

- 教师或LLM产生标签所需时间；
- 标签参考的观测MemoryID；
- 目标动作应作用的时间范围；
- action horizon。

不要把慢教师的延迟蒸馏给目标TNN。

## 结束条件

形成一批可回放、可检查、时间对齐明确的数据。

---

# Phase 7：最小Dock训练

## 研究问题

Dock能否从Memory数据和教师标签训练一个新TNN，并生成可被Phase 4 runtime加载的artifact？

## 最小实现

```text
eve/dock/trainer.py
eve/dock/TinyNN.py
eve/dock/workspace/
tests/test_dock_training.py
```

训练订单只包含本任务需要的字段：

- 新建或补训；
- 目标角色；
- 输入MemoryID；
- 标签MemoryID或teacher；
- 输入输出定义；
- action horizon；
- 延迟预算；
- 结构约束；
- 评估方式。

## 教师

第一次可以使用：

- 人类示范；
- 规则程序；
- 现有YOLO或基础模型；
- 离线LLM标签。

不要求先完成本地LLM大循环。

## 输出

- descriptor；
- tn_structure；
- weights；
- 训练日志；
- 离线指标；
- 延迟指标；
- 数据和Prompt引用。

Dock完成后只注册为available，不自动真实部署。

---

# Phase 8：第一次真实成长证明

## 研究问题

EVE是否真的从经历中获得了一项更快或更稳定的新能力？

## 流程

```text
目标TNN不存在或关闭
→ 运行受控小游戏并记录基线
→ 收集经历和教师标签
→ Dock训练TNN
→ Runtime加载新TNN
→ 使用同一任务、seed范围和指标复测
→ 比较
→ 禁用新TNN确认可回退
```

## 主要指标

至少选择：

- 成功率；
- P50/P95反应时间；
- 错误点击；
- unsafe/被Safegate阻断次数；
- 资源消耗；
- 慢教师调用次数。

## 成长成立条件

- 至少一个预先定义的核心指标明确改善；
- 成功不是由于改变任务难度或测试条件；
- 安全指标不恶化；
- 新TNN可以关闭并回到基线；
- 日志和Memory能解释训练数据与行为变化。

Phase 8是EVE第一个真正的核心里程碑。

---

# Phase 9：本地LLM大循环与active_tnn

## 研究问题

本地LLM能否在不阻塞快速路径的情况下更新理解、长期策略和希望在场的TNN名单？

## 步骤

1. 分别验证DeepSeek和Qwen能否加载；
2. 选择更适合当前机器的一种；
3. 先用FakeLLM固定JSON验证接口；
4. 再替换为真实本地LLM；
5. 大循环不重叠；
6. 输出`active_tnn` ID集合；
7. Core比较`active_tnn`与`loaded_tnn`并加载/卸载。

## 输入

- world摘要；
- myself摘要；
- blackboard重点；
- 当前任务；
- available/loaded TNN list；
-必要Memory检索。

## 本轮不做

- LLM每帧路由；
- 通用Graph修改；
- 自动训练；
- 激素复杂模型。

---

# Phase 10：六激素与内在节律

## 研究问题

连续变化的内部状态是否能让LLM思考频率和倾向产生可观察但不过度僵硬的变化？

## 最小实现

从简单线性更新开始：

```text
new = old + event_delta + long_drift + recovery
```

先记录六个值和变化来源。

把LLM间隔限制在10—20秒，根据：

- 突发变化；
- 风险；
- 新奇；
- 稳定；
- 资源压力；
- 用户反馈

进行简单映射。

## 观察结果

- 平稳时是否逐渐放慢；
- 突发时是否加快；
- 是否出现振荡；
- 是否影响active_tnn和长期策略；
- 是否错误地直接控制动作。

---

# Phase 11：LLM发现训练机会

## 研究问题

本地LLM能否根据Memory、失败、重复慢路径和资源状态，提出有价值的训练建议？

第一版只允许LLM输出：

```text
不训练
继续收集数据
建议新TNN
建议补训TNN
请求云端分析
```

训练建议必须转换成可检查订单后，才交给Dock。

早期可以要求人工确认，不追求完全自动。

---

# Phase 12：睡眠复盘与长期反馈

## 研究问题

更长时间范围的结果能否重新关联早期策略和动作，并改善后续训练或Prompt？

逐步加入：

- Event整理；
- 跨时间关联；
- revision；
- merge和lazy redirect；
- 索引折叠；
- 视觉预兆保护；
- 稠密关联压缩；
- 遗忘；
- 云端LLM辅助复盘。

只在Phase 8成长闭环稳定后实施，避免Memory复杂度掩盖基础问题。

---

# 后续可选方向

- 音频和语音；
- 键盘状态与真实键盘输出；
- 活动窗口识别；
- 完整控制面板；
- 长期策略TNN；
- 多种小游戏；
- 更复杂的Memory索引；
- 云端LLM；
- TNN联合训练；
- 机会式后台训练。

这些按个人兴趣和实验需要选择，不形成强制路线。
