# EVE 代码收缩矩阵

基线：2026-07-25，`main` 分支，工作区开始时无未提交修改。统计排除
`eve/core/deepseek-7b/`、`eve/core/qwen/`、`eve/core/yolo26/` 内的本地模型和第三方代码。

| 对象 | 决定 | 当前真实调用者 / 生命周期判断 | 删除后承接者 | 对纵向闭环影响与迁移步骤 |
|---|---|---|---|---|
| `eve/main.py` | REWRITE_MINIMALLY | 进程入口；当前只启动 GUI，未启动 `EVELoops` | 原文件 | 改为可限时运行的安全 CLI，真实启动/停止 capture 与一个核心循环 |
| `eve/config.py` | REWRITE_MINIMALLY | 仅入口使用；大量未来模型/GUI配置无消费者 | 原文件 | 只保留路径、循环周期、捕获频率与安全输出模式 |
| `eve/state.py` | KEEP + MERGE | 运行期唯一共享内存对象成立 | 合并 `runtime_state.py`、Blackboard 队列语义、六激素值 | 保留安全状态；动作使用独立 consume-once 队列，普通状态使用 latest-value |
| `eve/core/runtime_state.py` | MERGE | 没有独立线程、资源或故障模式，仅二次包装 state | `eve/state.py` | 快照与最小状态更新并入权威 RuntimeState |
| `eve/input/buffer.py` | KEEP + MERGE | 独立的近期窗口和并发边界成立 | 合并 `schemas.py` 的 `TimedSample` | 保留单调时间戳、latest/range/约一秒快照 |
| `eve/input/capture.py` | REWRITE_MINIMALLY | 捕获线程生命周期成立 | 合并实际屏幕/光标读取 | 单线程按频率写 Buffer；异常结构化暴露；stop 验证线程退出 |
| `eve/input/schemas.py` | MERGE | 小 dataclass，无独立生命周期 | `buffer.py` | 删除旧 `CursorState` |
| `eve/input/screen_capture.py` | DELETE | 与 `capture.py` 重复的屏幕实现，正式入口未使用 | `capture.py` | 保留一种可选真实捕获路径 |
| `eve/input/cursor_capture.py` | DELETE | 与 `capture.py` 重复且依赖旧 Schema | `capture.py` | 保留一种可选真实捕获路径 |
| `eve/core/graph.py` | DELETE | 独立 Node/Edge/Trace/Cache 与当前架构冲突 | `eve/core/tnn.py` 中的 loaded TNN 字典与命名输出 | SourceRef 直接读取 state/blackboard/`tnn:<id>.<field>` |
| `eve/core/tnn_base.py` | MERGE | descriptor/执行/保存职责被 Store 与 Graph 拆散 | `eve/core/tnn.py` | 保留最小 descriptor、节点协议、按频率运行 |
| `eve/core/tnn_store.py` | MERGE | Store 没有独立线程；重复 descriptor/structure 元数据 | `eve/core/tnn.py` | 当前闭环只加载明确节点；descriptor 为唯一结构事实源 |
| `eve/core/loop.py` | REWRITE_MINIMALLY | 核心循环生命周期成立，但八线程架构和静默异常不成立 | 原文件 | 单核心线程顺序执行 TNN→动作→Safegate→Output→Memory |
| `eve/core/safegate.py` | KEEP | 独立安全边界与失败模式成立 | 原文件 | 保留 cold start、急停、权限、模式、过期、人类接管 |
| `eve/output/*.py` | KEEP | 三类真实输出各有独立系统后端与失败模式 | 原文件 | 正式 CLI 仅允许 disabled/mock；所有调用仍先经过 Safegate |
| `eve/core/hormones.py` | MERGE | 固定事件映射、趋势与行为解释未经实验验证 | `MyselfState.hormones` | 只保留六个连续值与最小逐轮回归，不直接生成动作 |
| `eve/core/sleep.py` | PAUSE_FROM_RUNTIME + DELETE | 提前实现 Memory merge、技能发现、训练订单等完整系统 | `RuntimeState.sleep_requested` 与显式快照 | 不进入正式运行路径 |
| `eve/core/model_adapter.py` | PAUSE_FROM_RUNTIME + DELETE | 未在真实模型验证，且当前闭环没有调用需求 | 无 | 模型接入待具体实验再实现 |
| `eve/core/prompts.py` | PAUSE_FROM_RUNTIME + DELETE | 8 套预设 Prompt 无当前真实调用链 | 无 | 专用 Prompt 留待具体实验订单 |
| `eve/memory/catalog.py` | MERGE | 只是 Memorizer 的一个字典包装 | `memorizer.py` | 单一 Catalog 映射仍保留 |
| `eve/memory/event.py` | MERGE | 小字典包装，无独立生命周期 | `memorizer.py` | Event 仅组织 MemoryID |
| `eve/memory/indexes.py` | DELETE | 稠密图折叠、redirect/merge 尚无真实压力 | `memorizer.py` 的时间/类型/关键词最小检索 | 删除多图索引事实副本 |
| `eve/memory/retrieval.py` | MERGE | 仅遍历 Catalog 的薄包装 | `memorizer.py` | 保留 ID、时间、类型、关键词检索 |
| `eve/memory/memorizer.py` | REWRITE_MINIMALLY | Memory 持久化生命周期成立 | 原文件 | 不可变 payload、Catalog、STM/MTM ID、Event 与最小检索集中实现 |
| `eve/dock/order.py`、`tiny_nn.py`、`trainer.py` | PAUSE_FROM_RUNTIME + DELETE | 随机/规则 fallback、固定 64 维向量和同集 MSE 不诚实 | 无 | 等真实数据契约、教师标签和独立验证集出现后重建具体训练实现 |
| `eve/gui/control_panel.py` | PAUSE_FROM_RUNTIME + DELETE | 781 行五页 GUI 早于闭环，且当前入口被其阻塞 | CLI 状态输出 | 不影响纵向闭环；以后只按真实观测需求添加 |
| `eve/main_legacy_demo.py` | DELETE | 旧 YOLO 演示，不是正式入口 | 无 | 防止第二正式路径 |
| 旧结构测试 | DELETE / REWRITE | 大量验证 Graph、Manager、GUI、伪 Dock 名词 | 纵向行为测试 | 新测试覆盖入口启停、SourceRef、consume-once、Safegate、Memory 和异常 |

## 基线证据

- 第一方正式 Python：38 文件，7,533 行。
- 活动测试：14 个 `test_*.py`，2,985 行。
- 架构型类命中 12 个（Manager / Graph / Trace / Adapter）。
- 已确认静默异常：`core/loop.py` 的 graph、action、memory 三条运行分支。
- 已确认入口断链：`main.py` 没有创建或启动 `EVELoops`。
- 已确认 SourceRef 缺陷：`tnn:<id>.<field>` 被错误按两个冒号字段拆分。
- 已确认动作重复风险：Blackboard 的 `read()` 是 latest/history 读取，不是消费队列。

本文件是执行记录，不是未来架构承诺。生成后继续实施，不等待确认。
