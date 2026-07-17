# EVE当前架构

## 1. 顶层结构

正式代码根为：

```text
eve/
```

除`main.py`外，主要部分为：

```text
input/
output/
memory/
core/
dock/
```

`src/eve/`中的旧stub仅供参考，不作为正式实现基础。

## 2. 总体数据流

```text
input持续捕获
→ buffer形成最近一段时间的state
→ 基础模型、TNN和LLM读取所需信息
→ 结果写入blackboard、world或myself
→ 动作节点形成ActionCandidate
→ Safegate判断权限、急停和人类接管
→ output执行或模拟执行
→ OutputResult和环境变化被记录
→ 有价值的结果作为MemoryUnit写入LTM
→ Dock从Memory取得数据训练新TNN
```

快速路径不能等待慢速LLM、Memory整理或Dock训练。

## 3. main.py

只负责：

- 初始化；
- 冷启动；
- 启停循环；
- Esc急停；
- 控制权限；
- 状态显示；
- 正常停机和必要快照。

早期不要求完整GUI。CLI、日志窗口或简单OpenCV/Qt面板均可。

## 4. Input

### capture

捕获屏幕、音频、光标、键盘、活动窗口和时间戳。

捕获只提供原始输入，不进行长期保存和高级判断。

### buffer

维护最近约1秒的多模态输入，提供：

- `latest`
- 指定时间范围读取
- 统一单调时间戳

尚未决定保存的原始输入只存在于buffer，不属于Memory。

早期实验可以先只实现屏幕和光标，其他输入按真实需求添加。

## 5. Output与Safegate

`output/mouse.py`、`keyboard.py`和`speak.py`只执行已经通过Safegate的动作。

输出模式至少包括：

```text
disabled
mock
real
```

默认必须是`disabled`或`mock`。

Safegate只处理：

- 是否得到对应权限；
- Esc是否触发；
- 人类是否接管；
- 当前动作是否过期；
- 输出模式是否允许；
- 必须阻断的最低安全条件。

Safegate不重新理解任务，也不替代TNN或LLM决策。

## 6. Runtime State

### state

Input对其他模块提供的近期多模态状态。

### world

EVE对外部世界当前状态的理解。

### myself

EVE对自身权限、资源、任务、模型、TNN、激素和倾向的理解。

### blackboard

不同节点之间交换当前结果的共享区域。

Blackboard不是长期记忆。结果可以被覆盖、过期和清理。每项结果只保留实际需要的生产者和时间语义。

## 7. Memory

### 7.1 基本结构

```text
MemoryUnit = MemoryID + payload
Catalog = MemoryID → LTM中的真实对象
```

`memory/LTM/`是实际图片、文本、音频、轨迹、state、模型输出和动作结果的权威存储位置。

### 7.2 STM与MTM

- STM：最近产生或最近使用的MemoryID集合，可带可丢弃缓存；
- MTM：当前任务主动使用的MemoryID工作集，可带可丢弃缓存；
- LTM：真实payload所在位置。

STM和MTM查找都通过：

```text
MemoryID → Catalog → LTM对象
```

不建立三套payload。

### 7.3 Event和索引

Event组织一组MemoryID为一次经历。

索引只负责从时间、内容、任务或其他角度找到MemoryID/EventID。第一版只做真实实验需要的索引。

Revision、Merge、索引折叠、遗忘和稠密图压缩在基础成长闭环跑通后再逐步加入。

## 8. TNN

每个TNN拥有自己的descriptor，至少说明：

- ID和版本；
- 输入SourceRef；
- 命名输出；
- 运行频率或触发方式；
- 观察历史范围；
- 输出负责的时间范围；
- 动作型TNN的action horizon；
- 延迟预算；
- 权重位置和结构描述。

潜在上下游关系写在下游TNN的输入SourceRef中。

例如：

```text
tnn:ball_detector.target_position
```

已经表示下游读取`ball_detector`的`target_position`输出。

## 9. TNN Graph

TNN Graph不是独立文件或独立数据结构。

它只是：

> 当前已加载TNN按照各自descriptor，从state、blackboard和其他TNN命名输出读取数据时自然形成的承接关系。

不建立：

- Capability Graph；
- Runtime Graph；
- Runtime Graph Trace；
- 全局边表；
- Graph Manager；
- `runtime_graph.json`。

实际发生过的生产、读取和动作由普通日志与Memory记录，需要时可事后重建。

## 10. active_tnn与loaded_tnn

- `active_tnn`：本地LLM希望当前在场的TNN ID集合；
- `loaded_tnn`：实际成功加载的TNN集合。

第一版`active_tnn`只包含ID。

只有真实实验出现“同一TNN必须动态切换上游”的需求时，才增加最小binding。

## 11. 本地LLM和云端LLM

主要使用一个本地LLM进行较慢的通用思考。

它可以：

- 更新world和myself；
- 形成长期策略；
- 判断接下来哪些TNN应当在场；
- 发现可能值得训练的能力；
- 必要时提交困难问题给云端LLM。

本地LLM不每帧调度TNN，也不成为固定Router。

云端LLM是更强但更慢的异步教师或辅助节点，其迟到结果只能影响当前仍有效的问题或未来行为。

## 12. 激素

六种激素保留为连续状态：

- dopamine
- serotonin
- norepinephrine
- oxytocin
- cortisol
- acetylcholine

它们逐轮变化，影响LLM上下文、倾向和10—20秒的大循环间隔，但不能直接产生动作或绕过Safegate。

激素具体函数必须通过运行观察逐步调整，不预先复杂化。

## 13. Dock

Dock接收具体训练订单，负责：

- 取得Memory数据；
- 取得教师标签；
- 创建或补训TNN；
- 训练；
- 离线评估；
- 延迟评估；
- 生成descriptor、结构和权重；
- 注册为available TNN。

Dock不决定当前加载哪些TNN。

成长是否成立，最终由同一真实任务训练前后对比证明。
