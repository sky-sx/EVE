# EVE当前状态

更新时间：2026-07-17 23:00 CST  
Phase 1 + 2 已完成。

## Phase 1 完成情况

最小 Mock 运行闭环。程序默认不产生真实系统输出；ActionCandidate 经过 Safegate 得到 Mock 或阻断结果，并记录嵌套 JSONL 日志。

### 核心文件

| 文件 | 职责 |
|---|---|
| eve/state.py | RuntimeState, ActionCandidate, SafegateResult, OutputResult |
| eve/core/safegate.py | 9条Safegate规则 |
| eve/core/loop.py | run_once() + JSONL嵌套日志 + try/finally |
| eve/output/mouse.py, keyboard.py, speak.py | mock/disabled 三输出 |
| eve/main.py | 安全实验入口 |

## Phase 2 完成情况

真实屏幕 + 光标的一秒统一 buffer。在单调时间轴下稳定捕获屏幕帧和光标坐标，并提供 latest/range/snapshot 接口。

### 新增文件

| 文件 | 职责 |
|---|---|
| eve/input/schemas.py | TimedSample, CursorState |
| eve/input/__init__.py | Input 包 |
| eve/input/buffer.py | InputBuffer — 线程安全, latest/range/snapshot, 自动淘汰 |
| eve/input/capture.py | CaptureManager — mss+pyautogui线程, CaptureTiming统计 |
| tests/test_input_buffer.py | 15个buffer单元测试 |
| tests/test_capture_timing.py | 10个捕获timing测试 |

### 测试结果 (59/59 通过)

```
python -m pytest tests/test_safegate.py tests/test_runtime_loop.py tests/test_input_buffer.py tests/test_capture_timing.py tests/test_forbidden_architecture.py -v
============================= 59 passed in 11.25s ==============================
```

| 测试组 | 数量 | 覆盖 |
|---|---|---|
| Safegate | 18 | 9条规则全覆盖 |
| Runtime Loop | 14 | run_once, 模拟/阻断/异常/日志嵌套/线程 |
| InputBuffer | 15 | store/latest/range/snapshot/淘汰/线程安全 |
| CaptureTiming | 10 | FPS/P50/P95/光标Hz/内存/线程清理/单调性 |
| Architecture | 2 | 禁止名称扫描 |

### 已验证的 Phase 2 能力

- mss 屏幕捕获 ≥ 30fps，帧间隔 P50/P95
- pyautogui 光标采样频率
- 屏幕与光标时间戳在 monotonic_ns 下对齐
- buffer.range() 按时间范围精确过滤
- buffer.snapshot() 按 duration 返回所有 kind 快照
- 自动淘汰超出保留期的旧样本
- 多线程 store 不丢数据
- 停止后捕获线程正确退出
- 内存增长可追踪

### 未实现

- 真实键盘/鼠标/语音输出
- world/myself/blackboard 运行时
- Memory (LTM/Catalog/STM/MTM)
- TNN 加载/运行/卸载
- 本地 LLM 集成
- 激素与节律
- Dock 训练
- 控制窗口 UI
- 音频、键盘、窗口捕获

### 当前下一步

执行 Phase 3：world、myself、blackboard 运行时状态与共享结果场。

## 环境状态

- Python 3.11.7
- PyTorch 2.10.0+cu130, CUDA 13.1, TensorRT 10.16.1.11
- GPU: RTX 5080 16GB
- 无有效 Git 仓库
- 所有模型仅文件存在，未验证加载
