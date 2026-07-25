# Architecture to Code Map

| 生命周期 / 事实 | 当前权威代码 | 说明 |
|---|---|---|
| 进程启停 | `eve/main.py` | 安全 CLI、cold start、Esc/中断急停、顺序停机 |
| Runtime State | `eve/state.py::RuntimeState` | world、myself、blackboard、动作队列、loaded TNN、命名输出唯一内存事实源 |
| Input 捕获 | `eve/input/capture.py::Capture` | 一个线程生命周期；真实 reader 可选，异常显式上报 |
| 最近输入窗口 | `eve/input/buffer.py::InputBuffer` | 单调时间戳、latest/range/最近约一秒 |
| TNN descriptor 与运行 | `eve/core/tnn.py` | SourceRef、按频率运行、命名输出、ActionCandidate 转换；无独立图 |
| 主运行链 | `eve/core/loop.py::CoreLoop` | TNN→动作→Safegate→Output→Memory，单线程顺序可读 |
| 安全边界 | `eve/core/safegate.py` | cold start、急停、权限、模式、过期、人类接管 |
| 系统输出 | `eve/output/{mouse,keyboard,speak}.py` | 所有调用由 CoreLoop 在 Safegate 后分派 |
| Memory | `eve/memory/memorizer.py` | Catalog、不可变 payload、STM/MTM ID、Event、最小检索 |
| 六激素 | `eve/state.py::MyselfState` | 仅六连续值与向中性基线慢速回归 |
| Dock / GUI | 包 `__init__.py` 声明暂停 | 不在正式运行路径，无假完成实现 |

不再存在独立 Runtime Graph、Graph Trace、全局边表、RuntimeStateManager、
HormoneManager、SleepManager、IndexManager 或 EventManager。
