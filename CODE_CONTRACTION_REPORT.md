# EVE 当前代码收缩与真实闭环修正报告

日期：2026-07-25。基线提交：`59f6b0b`，分支：`main`。开始时工作区干净；
本任务未 commit、未 push、未移动或复制模型文件。

## 1. 执行摘要

主要偏离是：入口只启动 GUI 而未启动核心循环；`graph.py` 建立了独立节点、
边、缓存和 Trace；State/Memory/Hormone/Sleep/Dock 被拆成大量薄包装或提前实现；
Blackboard 把动作当可重复读取状态；三条核心异常被静默吞掉；TNN SourceRef 解析错误。

本次将同一生命周期合回少量大模块，删除独立 Graph/Trace、二次 State 包装、
复杂 Sleep、假通用 Dock、预设 Prompt/Adapter 和五页 GUI。当前已经恢复一条可直接
运行和测试的最小闭环，但其输入与动作节点是合成/规则占位，输出是 Mock，不冒充真实
成长或真实桌面动作。

## 2. 修改前后

统计排除本地模型目录 `deepseek-7b/`、`qwen/` 与 vendored `yolo26/`。

| 指标 | 收缩前 | 收缩后 |
|---|---:|---:|
| 第一方 Python 文件 | 38 | 19 |
| 第一方 Python 行数 | 7,533 | 1,643 |
| Manager/Registry/Graph/Trace/Adapter 架构类 | 12 | 0 |
| 活动运行模块（不含 `__init__`） | 30+ | 12 |
| 测试文件 | 14 | 5 |
| 测试数（仅记录） | 173 的历史声明 | 11 个行为测试 |

## 3. 文件处置

| 处置 | 文件 | 原因 / 承接 |
|---|---|---|
| 重写 | `main.py` | GUI-first 改为安全、限时、真实启停资源的 CLI |
| 重写 | `state.py` | Runtime State、world/myself、latest Blackboard、动作队列与 TNN 输出唯一权威 |
| 重写 | `core/loop.py` | 八循环改为一条顺序因果链 |
| 新建 | `core/tnn.py` | 合并 descriptor、SourceRef、加载集合、频率与命名输出 |
| 保留 | `core/safegate.py` | 独立安全边界成立 |
| 重写 | `input/buffer.py` | 合并 TimedSample，使用 deque 和单调时间 |
| 重写 | `input/capture.py` | 合并屏幕/光标读取，一个线程、显式错误、验证 stop |
| 保留 | `output/mouse.py`、`keyboard.py`、`speak.py` | 独立系统后端；正式 CLI 不开启 real |
| 重写 | `memory/memorizer.py` | 合并 Catalog、Event、STM/MTM ID 与最小检索 |
| 暂停 | `dock/`、`gui/` | 仅保留包说明，不进入运行路径 |
| 删除 | `core/graph.py` | descriptor SourceRef 自然关系取代独立图/边/Trace |
| 删除 | `core/runtime_state.py` | 并入 `state.py` |
| 删除 | `core/hormones.py` | 六值并入 `MyselfState`，移除固定事件/倾向系统 |
| 删除 | `core/sleep.py` | 基础成长闭环前不运行完整复盘/训练发现 |
| 删除 | `core/model_adapter.py`、`prompts.py` | 当前没有真实模型调用者 |
| 删除 | `core/tnn_base.py`、`tnn_store.py` | 当前最小运行职责并入 `core/tnn.py`；不保留重复 artifact 元数据 |
| 删除 | `input/schemas.py`、`screen_capture.py`、`cursor_capture.py` | 合并为一套 Input 实现 |
| 删除 | `memory/catalog.py`、`event.py`、`indexes.py`、`retrieval.py` | 合并薄包装，移除复杂图/merge |
| 删除 | `dock/order.py`、`tiny_nn.py`、`trainer.py` | 移除随机/规则 fallback、固定 64 维与同集 MSE 假通用训练 |
| 删除 | `gui/control_panel.py`、`main_legacy_demo.py` | 移出正式运行路径并消除第二入口 |
| 重写 | `tests/` | 从名词测试改为五组纵向行为测试 |

## 4. 权威事实源

| 事实 | 唯一权威位置 |
|---|---|
| Runtime State | `RuntimeState` |
| TNN descriptor | 已加载节点的 `TNNDescriptor` |
| loaded / active TNN | `RuntimeState.loaded_tnn` / `myself.active_tnn` |
| TNN 命名输出 | `RuntimeState.tnn_outputs` |
| Memory Catalog | `Memorizer.catalog`，持久化为 `catalog.json` |
| 动作队列 | `RuntimeState.action_queue` |
| 已消费动作 | `RuntimeState.consumed_action_ids` |
| OutputResult | `RuntimeState.latest_output`；历史结果在 Memory payload |

## 5. 运行链证据

本轮实际 smoke：

```text
Capture(synthetic readers)
→ InputBuffer(cursor/screen)
→ SmokeActionNode [rule_placeholder_not_trained]
→ action_candidate named output
→ ActionCandidate("smoke-action-1")
→ consume once
→ Safegate(mode=mock, permission=true)
→ Mock mouse OutputResult
→ runs/smoke/eve.jsonl
→ input_snapshot + tnn_output + output_result MemoryUnit
→ core/capture verified stop
```

日志：`runs/smoke/eve.jsonl`。Catalog：`runs/smoke/memory/catalog.json`。

## 6. 命令与原始结果

系统 `python`/`py -3.11` 不可用，因此使用 Codex 工作区的 Python 3.12。
`pytest` 临时安装到忽略目录 `runs/.testdeps`，未加入正式依赖。

```text
python -m compileall eve tests -q
```

通过，无标准输出。为避免仓库既有只读 `__pycache__`，设置
`PYTHONPYCACHEPREFIX=runs/pycache`。该命令也编译了未改动的 vendored YOLO Python。

```text
python -m pytest -q -p no:cacheprovider
...........                                                              [100%]
11 passed in 0.34s
```

```text
python -m eve.main --mode mock --smoke-seconds 0.5 --run-dir runs/smoke
{"output_mode": "mock", "executed": false, "simulated": true,
 "memory_units": 3, "error": null, "threads_stopped": true,
 "log": "runs\\smoke\\eve.jsonl"}
```

## 7. 未完成事项

- 尚未以真实屏幕/光标输入运行本闭环。
- 尚未加载、运行或评估训练所得 TNN artifact；当前 smoke 节点是规则占位。
- 尚未运行真实鼠标、键盘或语音输出，且正式 CLI 有意禁止 real 模式。
- 尚未接入本地 LLM/VLM/YOLO；模型文件存在不等于模型已验证。
- Dock、Sleep、GUI 当前暂停，不是已完成能力。
- 键盘人类活动 hook 未实现。

## 8. 回滚说明

开始基线是提交 `59f6b0b` 且无用户未提交修改。回滚前先保存当前差异，例如生成
只读 patch；随后可从 `59f6b0b` 恢复本报告“文件处置”列出的跟踪文件，并删除本次
新增的 `CODE_CONTRACTION_PLAN.md`、`CODE_CONTRACTION_REPORT.md`、
`eve/core/tnn.py` 与三个新测试文件。不要使用 `git reset --hard`，也不要在未保存
当前差异时执行整仓恢复。本任务没有自动执行回滚。
