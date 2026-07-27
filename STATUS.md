# EVE 当前实现状态

更新时间：2026-07-27 CST

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

## 当前验证结果

最终回归：

```text
python -m pytest -q
..........................
26 passed in 4.12s
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
