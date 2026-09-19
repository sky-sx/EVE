# 当前用户澄清（优先于 canonical 原文中的冲突描述）

本文件记录 Phase 5 回复中的决定，不覆盖用户原始架构快照。

## 已确认

- eye：CNN + Linear，不能先平均池化到 18×32。
- ear：固定长度原始音频窗口 + 一层 Linear，目前只需 mock 音频。
- hand：两层 MLP，完整 82 键键盘和鼠标控制。26 个字母键加左键 click 仅是训练器读取日志的筛选条件，不裁剪 Adapter 的输出或完整机械日志。
- speak：一层 Linear，形成约 30 维连续输出。
- goodness：一层 Linear，输出唯一标量。
- route（口述 root）：一层 Linear。
- Block 局部资格项采用输出均值导数：
  `e_W_local = (1/n) * sum_k(partial z_i,k / partial W)`。
- A_g 校准采用 `1/2*(g-g*)²`，只更新 A_g。
- B_g 参数继续走一般 goodness 参数更新，不走 A_g 的校准损失。
- g 的出现频率高于稀疏 teacher g*，没有同刻 teacher 时 g_eff 可以采用 g。
- ticktime 表示一个 tick 的正数秒数，允许浮点；Logistic noise 的 tau 与塑性 tau 都直接等于 ticktime。
- 用户后续确认：`gamma=sigmoid((delta_t_ms/1000)/ticktime)`；Core 按 ticktime 调度，
  ReadIn 新输入依照原强制规则可提前触发一次更新。时间戳继续以整数毫秒保存。

## 最新确认的 teacher 与 hand 规则

1. 只校准和 g* 同一时间产生的当前 g，不回溯此前预测，也不复用该标签校准以后预测。
   同刻有 teacher 时，A_g 用 `1/2*(g-g*)²` 校准，其他参数采用 `g_eff=g*`。
   同刻没有 teacher 时，A_g 不更新，其他参数采用 `g_eff=g`。
2. 其他参数的唯一更新信号保持 `delta=g_eff-g_bar`，g_bar 为训练用滑动均值。
3. hand 使用完整 82 键接口，修饰键各自独立，可同时形成组合键；鼠标离散按键与
   连续 dx/dy 均保留。所有最终控制均写机械日志。A–Z 和左键 click 的筛选只属于
   训练器读取端，本轮不会自动进行 Stage 0。

这些问题均已明确，允许继续 Phase 10–15。

## 已选网络的具体大小

这些是小规模首版实现参数，不新增器官或认知模块。

- eye：完整 `[3,1080,1920]` → Conv2d(3,8,k11,stride10,pad5) + ReLU
  → `[8,108,192]` → Conv2d(8,16,k7,stride6,pad3) + ReLU
  → `[16,18,32]` → Flatten → Linear(9216,2*n)。没有平均池化。
- ear：默认单通道固定 1600 个采样点，可显式配置窗口长度与通道数；
  Flatten → Linear(channels*T,2*n)，不附加音频语义前端。
- hand：Linear(n,64) → ReLU → Linear(64,D+C)。默认 D=82 个键盘键 + 3 个鼠标按键，
  C=2 个连续鼠标方向。键位名称在实现中显式列出，不隐式生成组合键类别。
- speak/goodness/route：分别 Linear(n,30)、Linear(n,1)、Linear(n,M)。

Adapter 保留局部求导能力，Phase 13–14 已接入实际资格迹衰减和参数更新。
当前 A_g 的局部校准资格项只服务于同刻 g/g*；不把旧预测或旧 teacher 混入当前校准。

## 平均资格项的数学结果

当前 LN 不含可训练 affine 参数，因此其输出在精确算术下均值为零，即便包含 epsilon。
每次 h 从零开始，以标量 gamma 混合两个零均值向量，最终 mean(z) 也为零。
所以用户指定的 `grad(mean(z))` 对 Block 参数、经 o 进入 Block 的 ReadIn 参数理论上为零，
FP32 可能产生舍入残差。该结果已经向用户说明；没有擅自改为平方、绝对值、投影或其他资格项。
后续将按用户规则验证执行，不将零值或舍入残差当作学习成功证据。
