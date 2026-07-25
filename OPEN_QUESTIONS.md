# EVE当前未决问题

这些问题不要求现在全部回答。只在对应实验即将使用时决定。

## Phase 1附近

1. Phase 1日志放在`runs/phase1/`还是更通用的`runs/<experiment>/`？
2. Esc在无GUI的早期版本中使用keyboard hook、控制台输入还是测试注入？
3. 人类鼠标和键盘活动的统一检测在Phase 1只做模拟，还是提前接入只读hook？

推荐：Phase 1先模拟接管事件，Phase 2或真实输出前再接真实hook。

## Phase 2附近

1. 屏幕捕获是否能在当前显示器配置下达到稳定30fps？
2. buffer保存原始帧还是压缩帧引用？
3. 音频、键盘和活动窗口何时加入？

推荐：先屏幕和光标，出现具体需求后再扩展。

## Blackboard

1. `reference_time_ns`、`produced_at_ns`和`valid_until_ns`是否足够？
2. 同kind结果覆盖还是并存？
3. 冲突结果由下游选择还是由LLM复盘？

由Phase 3双频节点实验决定。

## TNN

1. 第一只真实TNN选择何种最小任务？
2. descriptor最少字段是什么？
3. 同一TNN动态切换上游的真实用例何时出现？
4. 需要binding前，是否只使用固定SourceRef？

推荐：在真实用例出现前保持固定SourceRef和ID-only active_tnn。

## Memory

1. LTM目录按类型、日期还是run组织？
2. 大图片/音频是否使用内容哈希去重？
3. 第一版Event字段是什么？
4. 第一版只做时间索引是否足够？

由Phase 5实际读写便利性决定。

## 首次成长任务

1. 红球点击还是气球点击？
2. 教师使用人类示范、规则程序、YOLO还是LLM？
3. 第一个目标TNN是目标检测、坐标转换还是动作TNN？
4. 真实点击何时开启？

推荐：先做自包含小游戏；先Mock动作，再人工授权真实点击。

## 本地LLM

1. DeepSeek与Qwen哪一个在RTX 5080 16GB上更适合？
2. 量化、CPU offload或服务化是否必要？
3. 大循环Prompt的第一版结构是什么？

到Phase 9再实测，不提前决定。

## 激素

1. 基线浓度、恢复速度和事件delta；
2. 10—20秒间隔映射；
3. 如何避免振荡；
4. 哪些激素变化真正有行为意义。

到Phase 10根据已有运行日志确定。

## 长期Memory

1. merge与revision的实际触发条件；
2. 稠密二部关系压缩是否真的出现；
3. 视觉预兆保护如何测量；
4. 遗忘需要什么条件。

在Phase 8以后、有真实Memory规模时再讨论。
