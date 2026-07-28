# EVE 文件结构与数据关系收拢报告

## 2026-07-28 GUI 与正式交互运行时增量

本节是对 2026-07-27 结构收拢报告的增量，不改变既定顶层文件结构。

- `main.py`：移除 control 占位阻断，加入八页 PySide6 GUI、冷启动/暂停/恢复/正常停机/急停/显式解除、权限变更、设置、快照恢复和状态刷新。
- `input/capture.py`、`input/buffer.py`：增加键盘活动与活动窗口小数据；仍由 Buffer 独占 Capture 进程和共享内存，并区分 EVE 输出与用户接管。
- `core/loop.py`：加入有界 LLM/VLM/cloud/TNN 队列、原子 JSON 更新、帧绑定、CUDA 实测、六个最小激素、节点/资源状态、五 TNN 上限和快照。
- `core/safegate.py`：改为鼠标与逐键原子授权，组合键必须全部满足，Unicode/粘贴同时检查 `send_text`、CTRL 和 V。
- `output/*.py`：完成真实鼠标/键盘边界、按键释放、Unicode 粘贴和可停止的异步 TTS。
- `memory/memorizer.py`：恢复最小 Event，补充真实计数、检索、TNN artifact 列表和强制整理进度/ETA。
- `pyproject.toml`：声明 GUI、模型、量化、资源、真实输出和 TTS 所需依赖。
- `tests/test_control_runtime.py`：覆盖八页 GUI、冷启动前静止、生命周期、LLM 原子更新、VLM 迟到隔离、权限和 TNN 五槽上限。

RTX 5080 Tensor 同步测试通过，现有螺旋三分类 TNN 的参数和输出均为 `cuda:0`。本地 LLM 默认路径已配置为 `eve/core/deepseek-7b`，VLM 默认路径已配置为 `eve/core/qwen`；代码强制 4-bit NF4，未伪造模型结果。YOLO 位于 `eve/core/yolo26`，本阶段尚未接入教师链路。真实键鼠/TTS 和物理 Esc 仍等待用户人工授权验收。

本阶段另完成 `runs/control_stage_observe_10m_20260728` 的真实输入十分钟长跑：601.797 秒，屏幕 28.7913 FPS，光标 59.7749 Hz，Core 32.1579 Hz，TNN 调用 9,460 次，Memory 写入/丢弃/失败为 601/0/0，runtime error 为 0，退出后 Core、Memory、Capture 和项目线程均已停止。observe 未开放真实 Output。

日期：2026-07-27 CST

## 1. 收拢结果

本轮只调整结构、控制关系和数据流，没有新增 EVE 能力。

删除：

- `eve/state.py`
- `eve/core/tnn.py`
- `eve/config.py`（未使用且不属于最终正式结构）
- Memory 中未使用的 Event 与 `related_ids` 预留接口

职责合并：

- `eve/state.py` 的必要运行状态并入 `eve/core/loop.py`；
- `eve/core/tnn.py` 的 artifact 加载、卸载、调度、输入解析、推理、输出、动作候选与异常处理并入 `eve/core/loop.py`；
- 未复制 SourceRef、Descriptor、Protocol、Smoke 正式类和重复 load/attach 层；
- Main 原有的 Capture 生命周期和 TNN 装配职责分别下沉到 Buffer 与 Core。

## 2. 最终数据流

```text
main.py
  ↓
InputBuffer
  ↔ capture.py 独立进程
  ↓
core/loop.py
  ↔ memorizer.py
  ↓
safegate.py
  ↓
output
  ↓ feedback
core/loop.py
```

不存在 Main/Core/Safegate/TNN/Output 越过 Buffer 读取或控制 Capture 的路径，Main 也不装配 TNN 内部运行节点。

## 3. Capture 进程结构

`InputBuffer.start_capture()` 使用 `multiprocessing.get_context("spawn")` 创建 `eve-capture` 子进程。

子进程内部使用屏幕和光标两个线程。屏幕帧写入共享内存 Ring Buffer，Pipe 只发送 frame ID、时间戳、slot、shape、dtype；光标、health、error 和 stop 控制也走 Pipe。任一采集线程异常都会发送结构化 error 并停止子进程。

父进程 Buffer 附加共享内存、维护最近一秒窗口、形成 human activity/takeover 状态并监控进程。停止时 join Capture、关闭 Pipe、释放并 unlink 共享内存。

Main 的关闭顺序为 Core Loop → Memory Writer → InputBuffer/Capture。

## 4. Core、TNN 与状态

`core/loop.py` 用简单结构保存：

```text
world / myself / blackboard
active_tnn / loaded_tnn / tnn_status
loop_status / permissions / resource_status
emergency_stop / latest_error
```

`active_tnn` 与 `loaded_tnn` 没有合并。具体 TNN 仍继承 `dock/tinynn.py` 的 `TinyNN`；Core 从 Memorizer 解析 artifact，导入具体 `model.py`、加载权重、调用 `infer()` 并负责卸载。

Safegate 仅做权限、急停、冷启动、接管截止时间、有效期、动作类型与最低范围检查。人类活动判断在 Buffer 完成。

Memory 仍异步写入并增量追加 Catalog；数组使用 NPY 而非 repr；支持 ID、类型、关键词和时间条件的最小检索；TNN artifact 仍在同一文件保存与读取。

训练与推理设备均采用 CUDA 优先策略：RTX 5080 可用时 Trainer 默认选择 `cuda`，Core artifact 加载也默认选择 `cuda`；没有 CUDA 时回退 CPU，并保留显式 CPU 覆盖。实际加载现有螺旋三分类 artifact 得到：

```text
trainer_default: cuda
core_node_device: cuda
parameter_device: cuda:0
gpu: NVIDIA GeForce RTX 5080
```

## 5. 测试与运行证据

收拢前基线：

```text
23 passed
```

收拢后最终回归：

```text
python -m pytest -q
..........................
27 passed in 4.40s
```

验收覆盖文件集合和导入边界、Capture 独立 PID、Buffer 生命周期/健康/窗口、Capture 异常到 Main 非零退出、active/loaded 分离、TinyNN artifact 重载、Mock-only observe、Esc、Core/Memory 错误和资源清理。

Smoke：

| 指标 | 结果 |
|---|---:|
| 退出码 / 时长 | 0 / 3.187 s |
| 屏幕 / 光标 | 29.58 FPS / 60.13 Hz |
| Core | 32.17 Hz |
| Safegate allow / block | 0 / 1 |
| 真实 Output | 0 |
| Memory written / dropped / failed | 5 / 0 / 0 |
| Capture/线程残留 | 无 |

十分钟真实 observe：

| 指标 | 结果 |
|---|---:|
| 退出码 / 时长 | 0 / 600.313 s |
| 屏幕平均 FPS | 28.7817 |
| 光标平均 Hz | 59.7612 |
| 屏幕平均采集延迟 | 20.0439 ms |
| 光标平均采集延迟 | 0.0017 ms |
| Core | 32.2633 Hz |
| TNN 调用 | 9,446 |
| Safegate allow / block | 0 / 1 |
| Mock / 真实 Output | 0 / 0 |
| Memory written / dropped / failed | 600 / 0 / 0 |
| runtime error / shutdown | 0 / 1 |
| EVE/Capture/resource tracker 残留 | 无 |

产物位于 `runs/structure_observe_10m_final/`。

此前 30 分钟数据来自收拢前的线程式 Capture 版本，不作为本轮独立进程架构的验收证据。

## 6. 未实现内容

- real Output 与真实受控闭环；
- 音频、键盘状态和活动窗口采集；
- 本地/云端 LLM；
- 六大激素；
- 复杂 Memory Graph、睡眠整理、图压缩；
- 自动训练、自动替换和影子部署；
- GUI 与长期资源管理。

物理 Esc 没有由 Codex 代替用户按下；Windows 全局 Esc 检测与清理路径由自动化测试覆盖。
