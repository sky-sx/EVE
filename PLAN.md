# EVE 下一步实验计划

当前只规划一个紧邻闭环的实验：

## 真实输入、Mock 输出的受控 smoke

研究问题：`Capture` 的真实屏幕与光标 reader 能否在不启用真实输出的前提下，
连续运行 30 秒并稳定形成单调的一秒窗口？

固定条件：

- `output_mode=mock`；
- 不加载本地 LLM、YOLO、VLM；
- 只使用明确标记的 smoke 规则节点；
- 最大运行 30 秒；
- Esc / KeyboardInterrupt 立即急停；
- 记录捕获异常、实际样本数、线程退出状态和 MemoryID。

成功条件：

- 屏幕和光标均产生单调样本；
- SourceRef 能读取最新真实输入；
- 动作仍只模拟一次；
- 无真实系统输出；
- capture/core 两线程均正常退出；
- 日志明确标记真实输入 + Mock 输出。

本实验完成前不恢复 Dock、Sleep、复杂 Memory Graph、GUI 或通用模型适配层。
