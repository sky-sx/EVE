# EVE 当前实现状态

更新时间：2026-07-29 CST

## 2026-07-29 红蓝圆三角两阶段实现

两阶段成长闭环已进入正式代码：

- 阶段一具备结构化 VLM 教师标签、完整 Experience、按条件检索、
  TrainingOrder、样本不足等待、Dock 自动训练、独立留出评价、TNN
  artifact 保存、自动加载和环境反馈。
- 阶段二具备红圆/蓝三角独立 TNN、多个 TNN 同时调度、SourceRef
  上下游、独立回归数据集和候选拒绝门槛。旧模型在候选未通过时不会被
  替换。
- QNN 已实现为真实可训练 TinyNN critic：从屏幕状态、鼠标动作和环境
  reward 学习 `q_value`，使用独立留出集评价并保存完整 artifact。
- Core 会在 Safegate 前用 QNN 比较同轮鼠标候选，只保留最高且过阈值的
  动作；选择、拒绝、预测奖励和真实反馈均写入 Blackboard/Memory。
- Dock 支持 `fitness_data`、`minimum_qnn_fitness` 和
  `minimum_qnn_margin`，可用已加载 QNN 比较候选 TNN 与旧版本。
- QNN 有独立加载状态，不占五个动作 TNN 槽位；GUI 可观察模型、版本、
  设备、阈值、评价次数及最近决定，并可随停机快照恢复。
- LLM-based self update loop 可输出受限的结构化 TrainingOrder；模型
  结构只能选择系统提供的安全模板，不能让 LLM 注入任意 Python。
- Memory 已验证真实 `STM -> MTM -> LTM` 晋升并可列出 LTM ID。
- 正常停机保存可读的 `world.md` 和 `self.md`，TNN 的输入、动作模板及
  active 状态可在下次启动恢复。
- 正式实验环境位于 `experiments/red_blue_shapes.py`，支持
  `red_only`、`red_and_blue`、`instruction_driven` 三种模式。
- Input 感知只包含屏幕、光标、键盘活动和用户文本，不读取活动窗口
  标题或进程名。

自动回归：

```text
python -m pytest -q
49 passed in 7.65s
```

尚未由自动测试替代的证据：物理 Esc、真实鼠标授权、真实桌面 VLM
识别质量、两阶段实机命中率、QNN 实机奖励校准及一次人工观察的
停机/恢复过程。

## 2026-07-28 GUI 与正式交互运行时状态（当前权威摘要）

以下摘要取代本文后半部分的旧阶段“尚未实现”列表；旧记录仅保留为历史证据。

- `python -m eve.main --profile control` 已启用 PySide6 GUI，不再返回占位退出码 2。
- GUI 共八页：实时视觉、文本与认知、资源与节点、冷启动与急停、Memory/Blackboard、权限与倾向、设置/激素/反馈、TNN。
- 程序启动只打开 GUI；用户点击冷启动后才启动 Capture、Core、Memory writer、LLM、YOLO、VLM 教师 worker 和 TNN 调度。
- Capture 仍由 Buffer 独占管理并运行在独立进程；进程内现有屏幕、光标和键盘活动采集线程。没有活动窗口标题/进程名、麦克风、录音或语音识别输入。
- Core 现有普通 LLM 对话、内部结构化 LLM 决策、YOLO/TNN 运行时视觉、按需 VLM 教师复核、OpenAI-compatible 云端请求和 TNN 生命周期队列。模型推理与 TNN 加载不在 GUI 主线程执行。
- 本地 LLM 默认使用 `eve/core/deepseek-7b` 并强制 CUDA 4-bit NF4；VLM 教师默认使用 `eve/core/qwen`，只在显式教师复核请求时按需加载；YOLO 默认使用 `eve/core/yolo26/weights/yolo26n.pt`。
- 本地 LLM 已真实完成普通文本对话，保持 `cuda:0`、`4bit-nf4`、`ready`，测试回复耗时约 1.485 秒。YOLO 已在 RTX 5080 上真实加载、预热和推理，合成帧推理约 8.27 ms。
- Core 的结构化 LLM 结果采用整批预校验和原子更新；无效结果不会部分覆盖 world、myself、Blackboard 或 active_tnn。
- YOLO 持续把检测框、类别、置信度和帧时间写入 `current_visual_result`；视觉 TNN 若消费 `state:screen` 并输出 `detections`，可成为当前运行时视觉结果。两者都绑定参考帧、模型、请求/完成时间，迟到结果只进入 `last_runtime_visual_result`，不会冒充当前结果。
- GUI 的显式 YOLO 分析请求使用有界队列冻结目标帧，并把图像、请求 ID 和结果关联写入 Memory。VLM 教师会携带同一帧的 YOLO/TNN 候选进行复核；教师结果写入独立的 `latest_teacher_review`，迟到结果记为 `stale`，不会覆盖运行时视觉。
- 真实鼠标、键盘、Unicode 文本和异步 TTS 执行路径已接入，但每次启动所有原子权限均重新为 `false`。本轮未代替用户授权并执行真实键鼠/TTS。
- Safegate 分别检查鼠标移动/点击/双击/滚轮/拖拽、逐键权限、组合键全部按键、`send_text` 和 `speak`；急停会清空待执行动作并停止输出。
- 六个最小激素值、自然恢复、表扬/批评反馈和行为倾向已接入；它们不能绕过权限或 Safegate。
- Memory 已支持 Event、STM/MTM/LTM 真实计数、按 ID/关键词/时间检索、强制整理的真实进度与动态 ETA。
- `MAX_LOADED_TNN = 5`；第六个加载请求被明确拒绝且原五个保持不变。TNN 加载前检查文件、设备、RAM 和 VRAM。TNN 页现显示模型路径、最近运行、输入/输出摘要、输出时间、真实 SourceRef 上下游、参数与 buffer 总内存及加载/卸载结果。
- GUI 不再直接调用 Memorizer；Memory 页读取和强制整理均通过 `EVEApplication` 的窄接口。
- 正常停机保存 world、内部兼容键 `myself`、必要 Blackboard、active_tnn、loaded_tnn 描述和模型设置，并生成用户可读的 `world.md` 与 `self.md`；权限和 API key 不进入恢复快照。

当前完整自动回归：

```text
python -m pytest -q
37 passed in 5.52s
```

本阶段真实 `observe` 十分钟长跑：

```text
run: runs/control_stage_observe_10m_20260728
duration: 601.797 s
screen: 28.7913 FPS / 19.5298 ms average capture latency
cursor: 59.7749 Hz / 0.00125 ms average capture latency
core: 32.1579 Hz
TNN invocations: 9,460
Memory written/dropped/failed: 601/0/0
runtime errors: 0
shutdown records: 1
threads stopped: true
Capture process stopped: true
real Output calls: 0
```

真实 CUDA 验证：

```text
torch: 2.10.0+cu130
torch CUDA: 13.0
GPU: NVIDIA GeForce RTX 5080
Compute Capability: 12.0
CUDA Tensor + synchronize: passed
spiral_three_class parameter device: cuda:0
spiral_three_class output device: cuda:0
```

真实 Qwen VLM 教师验证：

```text
run: runs/vlm_teacher_json_check_20260728_163031
device: cuda:0
quantization: 4bit-nf4
is_loaded_in_4bit: true
reference_frame_id: 5081
reviewed candidate: synthetic-candidate
result: 可解析的紧凑 JSON，确认红色矩形候选且 corrections 为空
Memory: vlm_teacher_result 1 / screen_image 1
cleanup: Core、Memory writer 和全部 eve-* 线程已停止
```

冻结帧在约 4.281 秒生成期间超过 Buffer 一秒实时窗口，因此结果按设计标为 `stale`，只作为绑定帧的教师记录保存，未覆盖当前实时视觉。

当前仍未完成或无证据：

- 本地 LLM 普通对话、YOLO 实时推理和 Qwen VLM 合成帧教师复核均已验证；真实桌面内容上的 VLM 教师质量仍属于完整人工验收的一部分；
- OpenAI-compatible 云端仅完成接口和错误状态，未配置 API key 做真实调用；
- 物理 Esc、真实鼠标/键盘/TTS 与完整 24 步人工交互验收仍需用户在桌面会话中完成；
- TNN 自动拆分/合并和通用影子部署仍未实现；QNN 与红蓝圆三角成长闭环已实现，但尚待实机验收；
- 复杂 Memory Graph、图压缩和自动睡眠策略仍未实现。

本文只记录当前工作区已经实现并实际验证的事实。

## 正式运行文件

除 `__init__.py`、测试、模型/权重及数据外，EVE 顶层正式运行代码为：

```text
eve/
├── main.py
├── input/
│   ├── capture.py
│   └── buffer.py
├── output/
│   ├── keyboard.py
│   ├── mouse.py
│   └── speak.py
├── memory/
│   └── memorizer.py
├── core/
│   ├── loop.py
│   ├── qnn.py
│   └── safegate.py
└── dock/
    ├── trainer.py
    └── tinynn.py
```

已删除 `eve/state.py`、`eve/core/tnn.py` 和未使用的 `eve/config.py`。

## 当前控制关系

```text
main.py
  ↓ 只创建、启动、停止和检查 InputBuffer
input/buffer.py
  ↔ Capture 独立进程（input/capture.py）
  ↓ 最近一秒 state / latest / range
core/loop.py
  ↔ memory/memorizer.py
  ↓ 动作候选
core/safegate.py
  ↓ 允许或阻断
output
  ↓ 执行反馈
core/loop.py
```

TNN 关系：

```text
dock/tinynn.py 定义 TinyNN 最小接口
  ↓
具体 TNN 的 model.py
  ↓
core/loop.py 从 Memory artifact 加载、调度、推理和卸载
```

`active_tnn` 是希望在场的集合，`loaded_tnn` 是实际加载成功的映射，两者保持独立。

## Input 与 IPC

- Capture 由 `InputBuffer` 用 `multiprocessing` 的 `spawn` 上下文创建；
- Capture 子进程内部使用屏幕线程和光标线程；
- 屏幕帧写入共享内存 Ring Buffer；
- Pipe 只传 frame ID、时间戳、slot、shape、dtype、光标小数据、健康状态、错误和控制命令；
- `InputBuffer` 负责进程、Pipe、共享内存、健康状态和最近一秒窗口；
- 人类光标活动及五秒接管截止时间在 Buffer 中形成状态；
- Main、Core、Safegate、TNN 和 Output 都不直接导入或读取 Capture。

## Core、Safegate 与 Memory

运行时状态已经合并进 `core/loop.py`，使用明确的 dict/set/deque，不再保留状态 dataclass、枚举、激素占位或独立 TNN Runtime 抽象。

Safegate 只检查权限、急停、冷启动状态、人类接管、动作有效期、动作类型和最低范围，不做目标判断、规划或光标轨迹分析。

`memory/memorizer.py` 仍负责异步有界写入、JSON/NPY 可恢复存储、增量 Catalog、按 ID 和最小条件检索及 TNN artifact。未使用的 Event 和 `related_ids` 已删除。

## 默认计算设备

- Trainer 在 CUDA 可用时默认使用 `cuda`，否则回退 CPU；
- Core 加载 TNN artifact 时同样默认使用 `cuda`，否则回退 CPU；
- 调用方仍可显式传入 `"cpu"`；
- 当前 PyTorch 识别到 `NVIDIA GeForce RTX 5080`；
- 已用现有螺旋三分类 artifact 验证：Trainer 默认 `cuda`，Core 节点为 `cuda`，模型参数实际位于 `cuda:0`。

## 当前验证结果

最终回归：

```text
python -m pytest -q
..........................
27 passed in 4.40s
```

最终 smoke：

- 退出码 0，运行 3.187 秒；
- 屏幕 29.58 FPS，光标 60.13 Hz，Core 32.17 Hz；
- Safegate 阻断 1，真实 Output 0；
- Memory 写入 5、丢弃 0、失败 0；
- Capture 子进程和项目线程均停止。

最终真实 observe 十分钟长跑：

- 退出码 0，运行 600.313 秒；
- 屏幕 28.7817 FPS，光标 59.7612 Hz；
- 平均屏幕采集延迟 20.0439 ms；
- Core 32.2633 Hz，TNN 调用 9,446 次；
- Safegate allow/block 为 0/1，真实 Output 0；
- Memory 写入 600、丢弃 0、失败 0；
- runtime error 0，存在正常 shutdown 记录；
- 退出后没有 EVE、Capture spawn、resource tracker 或 pytest 进程残留。

长跑产物位于 `runs/structure_observe_10m_final/`。

## 尚未实现

- 真实鼠标、键盘或语音控制；
- 音频、键盘状态和活动窗口采集；
- 本地或云端 LLM 大循环；
- 六大激素逻辑；
- Memory Graph、睡眠整理和图压缩；
- 自动 TNN 替换、影子部署或自动训练；
- GUI 和长期资源策略。

自动化测试验证了全局 Esc 路径；本轮没有由 Codex 代替用户实际按下物理 Esc。
