# 当前 ACNT 工程缺口

当前实现边界以 [canonical_architecture.txt](canonical_architecture.txt) 为唯一规范。Block 已实现 snapshot synapse mixing、A/At、real-time age/validity 与 grouped per-neuron private NLM；Local Plasticity 已观察全部 NLM grouped 参数，且不使用 autograd。

仍待实现或核验：

1. 接入真实截图与音频采集，同时保持 EVE 的感知边界。
2. 接入键鼠 DirectInput 机械执行；执行结果与失败不得返回 Core。
3. 将 Speak 的连续输出接入约 30 维发声器官到声学几何的映射。
4. 在 corrected source hash 上重新执行正式 Stage Zero；修正前的长训练结果不能用于判断当前 Block。
5. 为当前 CorrelationRule 继续建立可复现实验与独立审计。参数发生变化、一次 smoke 或非零 eligibility 都不等于学会任务。

旧实验和候选机制只留在历史归档，不接回生产运行路径。
