# Adapter 实现记录

[架构原文](canonical_architecture.txt)只规定六种神经网络 Adapter 的角色、输入输出边界和局部塑性要求，不固定 CNN/Linear 的层数。下表记录 `acnt/adapters.py` 当前使用的网络形状；它们是实现选择，不是新增架构规则。

| 器官 | 当前网络 | 规范输出 |
|---|---|---|
| eye | 完整 RGB `[3,1080,1920]` → 两层步幅 Conv2d + ReLU → Flatten → Linear | `[2n]`，输入对应 Block 的 `o` |
| ear | 原始音频固定窗口 `[channels,T]` → Flatten → Linear，默认单声道、1600 采样点 | `[2n]`，输入对应 Block 的 `o` |
| hand | `n→64→D+C`，中间 ReLU；当前 `D=85`、`C=2` | 每个键/鼠标按键的连续 `q`；鼠标位移保持连续 |
| speak | Linear(`n,30`) | 约 30 维连续发声器官参数 |
| goodness | Linear(`n,1`) | Runtime 将唯一标量 clamp 到 `[0,1]` |
| route | Linear(`n,M`) | 每个候选 Block 的连续倾向 `q_i`，`M` 为总 Block 上限 |

`n` 是各器官对应 Block 的神经元数。六个器官分别占用不同 Block。当前 eye Adapter 接受完整 1080p 张量；ear Adapter 接受原始音频窗口，不把文字、题单或 Teacher 文本直接送入 ReadIn。实际截图/录音采集、DirectInput 设备执行和声学几何计算尚未接入。

Adapter 中每个可塑连接按与 Block 相同的局部 `e_c/F_e/F_w` 接口更新。当前 `CorrelationRule` 只是可替换候选。Adapter 输出后，hand/route 的离散坐标各自加独立 Logistic 噪声并经过硬阈值；不使用 Softmax、Bernoulli sampler、梯度或额外评价流。连续坐标不经过这一步。具体规则见[架构边界核对](clarifications.md)。
