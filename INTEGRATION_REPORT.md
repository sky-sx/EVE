# EVE 架构收拢报告

更新时间：2026-07-30。

## 本轮收拢

- 移除实验程序、实验数据和正式运行时中的实验专用入口。
- 移除 Core 运行期 QNN 及其恢复、评分、排序、反馈和状态展示路径。
- 将权限与急停检查收回 Core 动作出队边界，不再保留独立安全模块文件。
- Dock 仅保留通用 JSON TNN 与显式 Python TNN 两种来源。
- 将 TNN 串行同步调用改为共享执行池上的独立到期调度。
- 将真实 Output 从 Core 主循环移到有界 worker 队列。
- 将环境反馈改为 `candidate_id`、`action_id`、执行时间和环境事件四项精确
  绑定。
- 修复 STM 截断导致 Catalog 中对象失去层级的问题，并校正晋升命名。

## 当前数据流

```text
Capture -> Buffer -> Core
                   |- 到期 TNN -> 原子发布输出缓存
                   |- 动作候选 -> 格式/时效检查 -> 权限与急停检查
                   `- 有界 Output 队列 -> 执行前复检 -> Output

TrainingOrder -> Dock workspace -> 验收
                              |- 通过 -> Memory TNN artifact -> Core 加载
                              `- 拒绝 -> training report，不注册正式 TNN
```

## 验收口径

实际测试、smoke、GUI 生命周期、词项检索和频率记录以本轮最终命令输出为准；
本文件不预先声称尚未运行的结果。
