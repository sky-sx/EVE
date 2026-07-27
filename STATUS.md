# EVE 当前实现状态

更新时间：2026-07-27 CST。

本文件只描述当前工作区代码事实。它不把 `ARCHITECTURE.md` 中的目标能力写成已经实现，也不把第三方模型仓库“存在”写成模型已经接入。

## 1. 当前可运行纵向切片

正式 CLI 当前提供 `smoke` 和 `observe` 两个安全 Profile：

```text
合成输入（smoke）或真实桌面输入（observe）
→ 独立 screen/cursor Capture 线程
→ InputBuffer
→ 规则 SmokeActionNode
→ Blackboard / consume-once ActionCandidate
→ Safegate
→ Mock/阻断 OutputResult
→ JSONL 日志
→ 有界异步 Memory Writer / 增量 Catalog
→ 状态快照、flush 与统一线程停止
```

`observe` 已在 Windows 实机连续读取真实屏幕和光标 30 分钟，正式入口始终禁止真实 Output。该结果证明半真实输入闭环，不证明真实动作或成长能力。

## 2. 模块事实

| 区域 | 当前真实状态 | 主要缺口 |
|---|---|---|
| `eve/main.py` | 支持 `smoke`、`observe`；明确拒绝 `control`；使用全局 Windows Esc 状态；统一启动、错误传播、关闭和运行摘要 | 没有监视控制 GUI；尚未开放且本阶段禁止真实控制 |
| `eve/input` | 屏幕与光标独立采集，目标 30 FPS/60 Hz；帧带 ID/单调时间，光标带速度；Buffer 是有容量的一秒窗口并支持专用 latest/state 接口 | 没有音频、键盘状态和活动窗口采集 |
| `eve/output` | mouse/keyboard 有 disabled、mock、real 后端；speak 有 disabled/mock | 正式 CLI 禁止 real；TTS real 未实现；真实输出未做本轮人工验证 |
| `eve/core/safegate.py` | 检查急停、冷启动、模式、权限、5 秒人类接管冻结和动作过期 | 没有完整键盘/鼠标系统级接管 hook；当前 5 秒只是 v1 值 |
| `eve/state.py` | 内存中有 world、myself、Blackboard 五类最新结果、动作队列、独立 active/loaded TNN 语义、状态和统计 | world/myself 仍很薄；快照未覆盖完整恢复状态 |
| `eve/core/tnn.py` | 支持 SourceRef、频率调度、TTL 命名输出、规则节点以及已训练 TNN artifact 的加载/卸载 | 没有持久化完整 TNN Graph；没有本地 LLM 声明 `active_tnn` 的生命周期 |
| `eve/core/loop.py` | 执行到期节点、Blackboard、安全门和 Mock/阻断反馈；Memory 只做非阻塞入队；关键错误可停止主生命周期 | 尚无本地/云端 LLM、完整资源策略、睡眠复盘和长期状态循环 |
| `eve/memory/memorizer.py` | 有界异步 Writer、优先级溢出策略、增量 `catalog.jsonl`、JSON/NPY 可恢复 Payload、flush 统计、TNN artifact 保存与解析 | 尚不是完整多模态 LTM；Event 未持久化；没有关联边、跨时间复盘或图压缩 |
| `eve/dock/tinynn.py` | 定义 PyTorch `TinyNN` 统一接口、推理、训练步骤和权重持久化契约 | 接口本身不代表具体能力已训练成功 |
| `eve/dock/trainer.py` | 支持 JSON 结构或 Python 模型、训练队列、训练/评估、artifact 生成与写入 Memory | teacher 字段尚未形成真实教师调用；没有延迟门槛、影子评估、完整 Graph 登记和 Core 空闲协作 |
| `eve/gui` | 只有空包 | 目标监视控制窗口未实现 |
| 本地/云端模型 | 目录或第三方代码存在 | 没有本地 LLM、云端 LLM、VLM、YOLO 或语音模型的 EVE 运行接入证据 |

## 3. TNN artifact 训练切片

当前工作区新增了一条合成训练与重载路径：

```text
Memory 中的合成训练样本
→ TrainingOrder
→ JSON 结构生成 model.py，或复用 Python TinyNN
→ training_step / evaluation_step
→ weights + structure + description + training metadata
→ Memory/TNNweights
→ Core 重新加载
→ 推理结果与训练后模型一致
→ 卸载并清理运行状态
```

该切片已经覆盖卷积、池化、MLP、残差/拼接、多输出、分类 target、自定义训练步骤和 artifact 完整性。它是合成模型与合成数据证据，不是 EVE 已经从真实经历长出可用能力的证据。

## 4. 测试证据

2026-07-27 在本机 Python 3.11 环境实际执行：

```text
python -m pytest -q
.......................
23 passed in 2.79s
```

测试文件与主要行为：

| 文件 | 覆盖 |
|---|---|
| `tests/test_input_minimal.py` | Buffer 时间单调、窗口读取、Capture 异常与停止 |
| `tests/test_memory_minimal.py` | Memory CRUD、Event、STM/MTM 与最小检索 |
| `tests/test_safegate.py` | disabled/mock/real 结果语义、权限、急停、过期与接管 |
| `tests/test_minimal_runtime.py` | SourceRef、TNN 上下游、动作单次消费、异常可见、入口启停 |
| `tests/test_tnn_artifact_pipeline.py` | 训练、artifact、Core 重载、结构校验与自定义训练 |
| `tests/test_runtime_integration.py` | Profile、Buffer 并发/关闭、异步 Memory、NPY、溢出、Blackboard、全局 Esc、错误传播和线程清理 |
| `tests/test_forbidden_architecture.py` | 无静默核心异常及历史禁止项 |

`test_forbidden_architecture.py` 仍把 `TNNGraph` / `RuntimeGraph` 名称列为禁止项，这是旧文档“图不持久化”决定的遗留保护，与当前完整架构要求存在冲突。测试目前虽然通过，但后续实现完整 TNN Graph 前必须把它改成“禁止重复、固定或过度设计的图”，而不是禁止持久化能力图本身。

## 5. 第一轮运行时实机证据

2026-07-27 在 Windows 桌面实际运行：

```text
python -m eve.main --profile smoke --duration 3
→ exit 0，30.00 screen FPS，60.00 cursor Hz
→ Safegate block 1，真实 Output 0
→ Memory written 6 / dropped 0 / failed 0

python -m eve.main --profile observe --duration 1800
→ exit 0，持续 1800.032 秒
→ 28.82 screen FPS，59.69 cursor Hz
→ 平均屏幕采集延迟 19.03 ms
→ Core 32.34 Hz，TNN 调用 28,206 次
→ Safegate allow 0 / block 1，真实 Output 0
→ Memory written 1,795 / dropped 0 / failed 0
→ runtime error 0，退出后项目线程和进程均已清理
```

长跑暖机后的工作集采样稳定在约 744—755 MB，未随时间线性增长。之后规则节点路径改为延迟导入 PyTorch，以降低不加载训练 TNN 时的基础资源占用；该调整已通过完整自动化回归和真实 observe 短测。

## 6. 证据等级

- 静态存在：文件、类或接口存在；不能证明可运行。
- 单元测试：局部契约可重复验证；不能证明真实环境能力。
- Mock 纵向切片：模块可以协作，但输入、节点或输出可能是假的。
- 合成模型切片：真实训练代码作用于合成数据；不能证明任务泛化。
- 半真实切片：至少真实输入或真实模型进入链路，动作仍可 Mock。
- 真实闭环：真实输入、受控真实动作与实际环境反馈形成闭环。
- 真实成长：同一任务训练前后以预先定义指标证明可信改善。

当前最高证据为“真实桌面输入的半真实闭环 + 合成模型训练切片”。仓库尚无真实控制或真实成长证据。

## 7. 当前架构偏差

1. 缺少持久化完整 TNN Graph，且旧测试仍禁止相关名称。
2. `active_tnn` 与实际加载集合已有独立状态，但尚无 LLM 驱动的差分生命周期。
3. Dock 训练先于 Memory 图、本地 LLM 和生命周期完成，尚未接入完整成长闭环。
4. `observe` 已接入真实屏幕和光标；音频、键盘状态与活动窗口仍未接入。
5. 六大激素只有中性回落占位，没有事件更新和 10—20 秒节律。
6. 本地/云端 LLM、长期策略、基础感知和睡眠复盘均未接入。
7. Memory 支持 JSON 与独立 NPY 数组，但尚未形成完整多模态关联体系。
8. GUI、资源监控和完整恢复快照未实现。

下一步按 `ROADMAP.md` 处理，不再从已删除的历史 Prompt、审计或阶段报告恢复实现承诺。
