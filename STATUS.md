# EVE 当前实现状态

更新时间：2026-07-27 CST。

本文件只描述当前工作区代码事实。它不把 `ARCHITECTURE.md` 中的目标能力写成已经实现，也不把第三方模型仓库“存在”写成模型已经接入。

## 1. 当前可运行纵向切片

正式 CLI 当前运行的是安全、限时的最小闭环：

```text
合成 screen/cursor reader
→ Capture 线程
→ InputBuffer
→ 规则 SmokeActionNode
→ 命名 TNN 输出
→ consume-once ActionCandidate
→ Safegate
→ disabled/mock OutputResult
→ JSONL 日志
→ input_snapshot / tnn_output / output_result MemoryUnit
→ 状态快照与线程停止
```

这证明线程生命周期、SourceRef、动作单次消费、安全门、Mock 输出、日志和最小 Memory 写入能够协作。它没有证明真实桌面输入、真实动作或成长能力。

## 2. 模块事实

| 区域 | 当前真实状态 | 主要缺口 |
|---|---|---|
| `eve/main.py` | 可启动、限时运行并停止 Capture/Core；检查控制台 Esc；保存 JSON 快照 | 没有目标架构中的监视控制 GUI；正式 CLI 总是注入合成 screen/cursor reader |
| `eve/input` | 线程安全 Buffer 支持 latest、range、约 1 秒 snapshot；Capture 可使用真实 mss/pyautogui reader | CLI 未走真实 reader；没有音频、键盘状态和活动窗口采集；默认屏幕频率为 10fps，不是目标 30fps |
| `eve/output` | mouse/keyboard 有 disabled、mock、real 后端；speak 有 disabled/mock | 正式 CLI 禁止 real；TTS real 未实现；真实输出未做本轮人工验证 |
| `eve/core/safegate.py` | 检查急停、冷启动、模式、权限、5 秒人类接管冻结和动作过期 | 没有完整键盘/鼠标系统级接管 hook；当前 5 秒只是 v1 值 |
| `eve/state.py` | 内存中有 world、myself、Blackboard、动作队列、loaded TNN、命名输出和部分快照 | world/myself 仍很薄；快照未覆盖完整恢复状态；`active_tnn` 当前在加载时同步写入，尚未与 `loaded_tnn` 分离 |
| `eve/core/tnn.py` | 支持 SourceRef、频率调度、TTL 命名输出、规则节点以及已训练 TNN artifact 的加载/卸载 | 没有持久化完整 TNN Graph；没有本地 LLM 声明 `active_tnn` 的生命周期 |
| `eve/core/loop.py` | 单线程顺序执行到期 TNN、动作、安全门、Output、Memory，并显式记录异常 | 尚无基础感知、本地 LLM、云端 LLM、资源监控、完整激素节律、睡眠复盘和状态持久化循环 |
| `eve/memory/memorizer.py` | JSON payload、Catalog、STM/MTM ID、最小 Event、类型/时间/关键词检索、TNN artifact 保存与解析 | 尚不是完整多模态 LTM；Event 未持久化；没有关联边、跨时间复盘、图压缩和预兆保护 |
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
................
16 passed in 2.94s
```

测试文件与主要行为：

| 文件 | 覆盖 |
|---|---|
| `tests/test_input_minimal.py` | Buffer 时间单调、窗口读取、Capture 异常与停止 |
| `tests/test_memory_minimal.py` | Memory CRUD、Event、STM/MTM 与最小检索 |
| `tests/test_safegate.py` | disabled/mock/real 结果语义、权限、急停、过期与接管 |
| `tests/test_minimal_runtime.py` | SourceRef、TNN 上下游、动作单次消费、异常可见、入口启停 |
| `tests/test_tnn_artifact_pipeline.py` | 训练、artifact、Core 重载、结构校验与自定义训练 |
| `tests/test_forbidden_architecture.py` | 无静默核心异常及历史禁止项 |

`test_forbidden_architecture.py` 仍把 `TNNGraph` / `RuntimeGraph` 名称列为禁止项，这是旧文档“图不持久化”决定的遗留保护，与当前完整架构要求存在冲突。测试目前虽然通过，但后续实现完整 TNN Graph 前必须把它改成“禁止重复、固定或过度设计的图”，而不是禁止持久化能力图本身。

## 5. 证据等级

- 静态存在：文件、类或接口存在；不能证明可运行。
- 单元测试：局部契约可重复验证；不能证明真实环境能力。
- Mock 纵向切片：模块可以协作，但输入、节点或输出可能是假的。
- 合成模型切片：真实训练代码作用于合成数据；不能证明任务泛化。
- 半真实切片：至少真实输入或真实模型进入链路，动作仍可 Mock。
- 真实闭环：真实输入、受控真实动作与实际环境反馈形成闭环。
- 真实成长：同一任务训练前后以预先定义指标证明可信改善。

当前最高证据为“Mock 纵向切片 + 合成模型训练切片”。仓库尚无真实成长证据。

## 6. 当前架构偏差

1. 缺少持久化完整 TNN Graph，且旧测试仍禁止相关名称。
2. `active_tnn` 与实际加载集合尚未按目标语义分离。
3. Dock 训练先于 Memory 图、本地 LLM 和生命周期完成，尚未接入完整成长闭环。
4. 运行入口使用合成输入，Input 种类和频率未达到完整目标。
5. 六大激素只有中性回落占位，没有事件更新和 10—20 秒节律。
6. 本地/云端 LLM、长期策略、基础感知和睡眠复盘均未接入。
7. Memory 目前以 JSON 为主，未形成完整多模态与关联体系。
8. GUI、资源监控和完整恢复快照未实现。

下一步按 `ROADMAP.md` 处理，不再从已删除的历史 Prompt、审计或阶段报告恢复实现承诺。
