# Architecture Conformance

| 当前架构要求 | 证据 | 状态 |
|---|---|---|
| 正式代码根是 `eve/` | `pyproject.toml` 与 imports | 符合 |
| 默认无真实输出 | CLI 只接受 disabled/mock；smoke `executed=false` | 符合 |
| 所有动作经 Safegate | `CoreLoop.step()` 只调用 `run_once()` | 符合 |
| State 单一权威 | `RuntimeState` 直接持有 world/myself/blackboard | 符合 |
| TNN Graph 非独立实体 | SourceRef 直接读取 `state.tnn_outputs` | 符合 |
| 动作 consume-once | `action_queue` + `consumed_action_ids` | 符合 |
| OutputResult 进入 Memory | `_remember_chain()` 创建 `output_result` 单元 | 符合 |
| 核心异常不静默 | 结构化 `RuntimeErrorRecord` + JSONL | 符合 |
| 六激素不过度解释 | 仅六值与 neutral settle | 符合 |
| Sleep/Dock/GUI 不提前进入运行 | 旧实现已移除 | 符合 |
| Memory 无复杂图/merge | 单 Catalog 与线性最小检索 | 符合 |
| Dock 不制造假训练 | 当前暂停，无随机标签/64维哈希/MSE假评估 | 符合 |

## 诚实边界

- `SmokeActionNode` 是规则 smoke 占位，不是训练所得神经网络。
- 本轮 smoke 使用合成输入与 Mock Output，属于模拟验证。
- real output 模块没有在本轮运行；不声称真实桌面动作。
- 本地模型目录与 vendored YOLO 未移动、复制或验证。
