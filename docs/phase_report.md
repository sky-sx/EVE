# ACNT 阶段记录

当前进度：Phase 1–15 完成，**146 项测试通过**。独立 mock CLI 连续运行 64 步，未运行 Stage 0。
后续用户澄清优先级高于原始架构中被修订的公式，详见 `clarifications.md`。
Phase 1–4 下方保留的是首次通过时的历史记录；后续时间公式修订及回归结果另记。

## Phase 1 — Block

文件：`acnt/block.py`、`acnt/__init__.py`、`tests/test_block.py`、`pyproject.toml`、`.gitignore`。

实现：完整必要状态，FP32 参数，ReadIn 专有 o/非 ReadIn 零输入，r/a 计算，
A/At 配对 push/pop，反向历史递推，h/z，LN 和 sigma。
对应 canonical 的 Block 状态定义和一次更新伪代码；隔离测试使用空源集合。

测试：`python -m pytest tests/test_block.py -q`，**11 passed**。
覆盖初始化、尺寸与参数清单、手算单次更新、ReadIn o、LN、sigmoid、连续更新、
历史容量/配对、时间倒退与配置验证。

尚未实现：多 Block 调度、器官、Adapter、塑性。
规范问题：本阶段未发现无法实现的尺寸/公式歧义。

## Phase 2 — 通信

文件：新增 `acnt/core.py`、`tests/test_communication.py`，更新包入口。

实现：Core 根据实际 active 集合提供源 z，支持不同神经元数量、自连接、多来源累加，
以及顺序/指定子集更新。对应 `sum_{j active} W_ij z_j` 和后续完整更新链。

测试：`python -m pytest tests/test_block.py tests/test_communication.py -q`，**19 passed**。
新增 8 项，覆盖 2/3/4 Block、异构尺寸、自连接、非零 inactive 来源排除、
顺序读取语义及重复异步更新。

尚未实现：active 专项接口/完整测试、器官、Adapter、塑性。
规范问题：无。规范未限定全局同步方式；本实现明确采用顺序更新。

## Phase 3 — active

文件：更新 `acnt/core.py`，新增 `tests/test_active.py`。

实现：单项 active 设置、全量布尔 mask、active_ids；Block 存在与参加计算分离。
对应 canonical 的 `active` 状态和只对 active 来源求和的规定。

测试：`python -m pytest tests/test_block.py tests/test_communication.py tests/test_active.py -q`，
**26 passed**。新增 7 项，覆盖全激活、部分/全部休眠、状态历史保留、恢复、
掩码变化对输入求和的真实影响、一次激活的基础能力和非法 mask 原子拒绝。

尚未实现：route 永久激活和 ReadIn 新输入触发的正式器官规则；Adapter、塑性。
这里仅验证这些规则所需的基础 active 能力，没有宣称完成后续器官测试。
规范问题：无。

## Phase 4 — Core 完整测试

文件：新增 `acnt/__main__.py`、`tests/test_history.py`、`tests/test_core_system.py`、
`README.md`、本报告和两份用户原文快照；调整 Block 自动时钟采样位置以逐句匹配原文。

对应 canonical：history 时间、队列索引、ticktime/gamma、h 递推、
active 与连续 Block 调度。入口只运行 Core smoke，不是整机 smoke。

测试：`python -m pytest -q`，**41 passed in 2.96s**。
其中 Phase 4 新增 9 项独立 history 数值测试和 6 项 Core 系统测试。
包含 500 步混合 active/异步更新、非有限值拒绝、输入校验和真实子进程启动。
另执行 `python -m acnt --steps 12`，正常退出并产生 12 条状态记录，
active 与 updated 集合逐步一致，history 长度始终不超过 hold_tick。

运行环境：Python 3.12.14、PyTorch 2.13.0+cpu、pytest 9.1.1。
阶段顺序实际为 11 → 19 → 26 → 41 项测试，前一阶段通过后再进入下一阶段。

尚未实现：Phase 5 之后的 Adapter/器官/plasticity/整机 smoke。
Core 公式无未决冲突；后续规范的待确认问题列在下方。

## 首次审查时的数学问题（部分已由后续回复解决）

1. **校准损失冲突**：canonical 第 75 行是 `L_g=1/2(g-g*)²`，第 94 行是
   `L_g=1/2(g_eff-g*)²`。第 74 行又规定存在 teacher 时 `g_eff=g*`，
   因而后一个平方损失恒为零，无法表达前述拟合 teacher 的目标。需明确采用哪一式。
2. **资格迹输出维度**：若参数 θ 的形状为 S，`∂z_i/∂θ` 的形状为 `(n_i, *S)`，
   连续 readout 的 `∂y_i/∂θ` 同样多一个输出维度；`θ += η δ e_θ` 的右侧无法直接
   加回形状 S 的 θ。需明确输出轴的收缩/局部学习信号，不能自行 sum、mean 或增加投影。
3. **塑性时间常数**：只规定 `tau_i` 由 `ticktime` 导出，未给出映射及单位、零/负
   ticktime 的处理。离散 Logistic 噪声也使用 tau 记号，需明确与资格迹衰减 tau 是否独立。
4. **校准参数范围**：第 74 行指除 A_g 外其他 Block 参数走 g_eff；第 95 行却使用前文
   表示整个 Block 参数组的 θ_g。需明确 teacher 校准仅更新 A_g，还是连同 B_g，
   以及 B_g 是否仍执行一般好度更新，避免重复或替代更新。

用户后续已确定：A_g 用 g-g* 校准且只更新 A_g，B_g 保持一般更新；
局部导数对输出维取平均；时间公式改为秒/除法且 tau 同 ticktime；Adapter 已选定。
当前未决问题仅以 `clarifications.md` 为准，对应损失、eligibility 和更新尚未编码。

## Phase 5 — 人工选型点已通过

仅撰写 `docs/adapter_choices.md`，列出六类 Adapter 的候选、输入输出、层数、
适用理由和计算成本；没有任何 Adapter 实现文件或正式选型。
首次停在此处后，用户已经完成六类选型。eye 改选 CNN + Linear 且无平均池化，
hand 改选两层 MLP，其余选择 Linear。决定记录于 `clarifications.md`，未运行 Stage 0。

## 时间定义修订及 Core 回归

用户明确确认 `gamma=sigmoid((delta_t_ms/1000)/ticktime)`，ticktime 为正数秒，
tau_decay=tau_noise=ticktime，Core 按该时间间隔调度。
更新 `acnt/block.py`、`acnt/core.py`、Core smoke 的示例间隔，以及对应旧测试。
新增 `tests/test_timing.py`，历史闭式预期改为新定义而非保留旧乘法。

Core 原有四组测试 **34 passed**；历史和新增调度测试 **18 passed**。
覆盖 0/249/250/500ms 的不同 Block 间隔、提前调用跳过、inactive 和倒退时间拒绝。
当前阶段无时间公式未决问题；尚未实现资格迹衰减，但其 tau 已明确。

## Phase 6 — Adapter 单体

文件：新增 `acnt/adapters.py`、`tests/test_adapters.py`；记录用户澄清。
实现已选的六类普通神经网络，支持单样本和一个前置 batch 维。
对应 canonical 的神经 Adapter、ReadIn 输出 2n、ReadOut 输出对应控制维数。

测试：`python -m pytest tests/test_adapters.py -q`，**23 passed**。
覆盖 FP32/shape/finite、参数枚举及逐参数局部求导、batch 一致性、
eye 学习型空间缩小且无平均池化、原始 q 不经 Softmax 或阈值。
尚未实现：当时未接 Block，没有资格迹递推或参数学习。
规范问题：Adapter 选型已明确；hand 最终是否带连续 dx/dy 待答，类支持 C=0/C=2。

## Phase 7 — Adapter + Block

文件：新增 `acnt/runtime.py`、`tests/conftest.py`、`tests/test_bindings.py`，更新包入口。
实现六器官与六个不同 Block 一一挂接，输入到 o、Block 更新、各自 z 到 readout。
拒绝 Block 复用、维度不符和将 batch 直接作为单 Block 状态。
对应 canonical 的“一器官一个且不同 Block”和 ReadIn/ReadOut 计算链。

全套测试：**84 passed**（本阶段新增 9 项）。当时尚未实现 ReadIn 强制和 route 输出。
本阶段没有新增数学歧义；已知塑性问题仍等用户明确。

## Phase 8 — 新输入强制激活

文件：更新 `acnt/runtime.py`，新增 `tests/test_readin.py`。
实现完整 1080p mock/原始音频经过 Adapter 到对应 Block；
新输入在当前轮临时激活来源并强制更新一次，无视 route 休眠及尚未到期的调度，之后恢复。
对应 canonical 的 ReadIn 单次强制 active 规则。

测试：`python -m pytest tests/test_readin.py -q`，**4 passed**。
覆盖 eye/ear 提前更新、下一轮不重复触发、其他 Block 在同轮读到输入状态、原 active 恢复。
当时尚未实现 route 与 goodness/其他 readout 执行链，没有新增规范冲突。

## Phase 9 — route

文件：新增 `acnt/control.py`、`tests/test_route.py`，更新 Runtime 和 ReadIn 回归测试。
实现连续 q → 各坐标独立 Logistic noise → 严格 > threshold → bool 提议 → 下一轮 active 集合。
输出维度等于配置总 Block 容量；不用 Softmax/Bernoulli sampler；route 永久 active；
goodness active 由人控设定，route 不能覆盖。noise tau 使用用户确认的 Block.ticktime。

测试：`python -m pytest tests/test_route.py tests/test_readin.py -q`，**12 passed**，
其中 route 新增 8 项。包含固定种子逆 CDF 数值对照、多开/多关、下一轮实际计算变化、
永久 route、人控 goodness 和噪声 tau。

尚未实现：Phase 10 goodness 校准、Phase 11 hand/speak 最终控制、Phase 12 执行开关/机械日志、
Phase 13–14 e-prop、Phase 15 整机 smoke。现有局部 autograd 不是已实现的学习机制。
暂停原因：需要用户明确稀疏 teacher 的时间配对及其他参数的 delta 基准。

最终独立审查补充了一项配置边界校验：普通 Block 不能持有未绑定 Adapter 的 ReadIn o。
修复 Runtime 全量绑定校验并新增拒绝测试后，全套结果 **97 passed**。

## Phase 10 — goodness

用户明确：只校准同一时刻产生的 g/g*；无同刻 teacher 时 A_g 不更新；
其他参数保留 `delta=g_eff-g_bar`，g_eff 可为当前 g* 或 g。

文件：更新 `acnt/runtime.py`、`docs/clarifications.md`，新增 `tests/test_goodness.py`。
实现 `clamp(A_g(z_g),0,1)` 唯一标量、同刻 teacher 覆盖、当前 g 的平方校准损失、
逐参数当前局部校准资格项及仅 A_g 的参数更新。B_g 被明确隔离。
对应 canonical goodness 条款及用户最新校准规则；不缓存或复用 teacher。

测试：`python -m pytest tests/test_goodness.py -q`，**19 passed**。
覆盖 scalar/clamp、g*=0、当前 g 而非 g_eff 的损失、只改 A_g、稀疏 teacher 不回溯不复用、
饱和区零导数、非法标签或错时戳拒绝。
当时尚未实现 hand/speak 最终输出、执行开关和一般资格迹；本阶段没有剩余规范歧义。

## Phase 11 — hand / speak

文件：新增 `acnt/hand.py`、`acnt/mechanical.py`、`tests/test_readouts.py`；
更新 `acnt/adapters.py`、`acnt/runtime.py` 和原 shape 测试。
按用户最新要求，hand 由完整 82 键、鼠标左/右/中键、连续 dx/dy 组成，共 87 维。
离散坐标独立 noise/threshold，组合键不额外建类别；连续坐标不受阈值处理。
speak 形成 30 维连续参数。全部控制正常记录，A–Z/左键筛选只在日志读取端。
对应 canonical hand/speak 与用户完整键盘/日志澄清。

测试：readout 新增 **5 项通过**；连同 Adapter 与挂接回归 **38 passed**。
验证全部键唯一、组合键、方括号留在完整日志、27 项读取筛选、连续鼠标和 speak。
当时尚未完成人控开关及一般塑性；本阶段无未决问题。

## Phase 12 — 人控执行开关

文件：更新 `acnt/runtime.py` 和 readout 日志期望，新增 `tests/test_execution.py`。
所有 readout 先形成最终信号再检查开关；机械 executor 返回值被丢弃，异常只在日志中保存。
关闭开关也运行 Adapter/noise 并正常记录，不向 Core 返回 success/failed/blocked。
关闭 route 不应用新 mask，但 route Block 仍 active；goodness 开关控制 B_g active 和自身训练输出。
独立外部 teacher 仍可供一般训练使用。对应 canonical 的人控和机械边界规则。

测试：执行开关新增 **8 项通过**；连同 goodness/readout/route 回归 **40 passed**。
包括四个 ReadOut 关闭时实际执行了 Adapter、未调用 executor、结果不污染 Core、
route 开关仅影响应用、完整 JSONL 持久化。
一次旧 speak 日志断言因新增 execution_enabled 字段失败，已修正断言后全部通过，才进入 Phase 13。
当时尚未实现一般 e-prop；无规范阻塞。

## Phase 13 — 局部资格迹与延迟 goodness

文件：新增 `acnt/plasticity.py`、`tests/test_plasticity.py`；更新 Block/Core/Runtime。
每个实际参数张量独立维护资格项；包含所有 W_ij、b、W_c、b_c、对应 Adapter 参数。
A_g 使用独立的同刻局部校准路径，不进入一般更新。
实现用户指定输出均值导数、秒制 exp 衰减、延迟送达、delta 对旧 g_bar、参数更新、rho 和 EMA。
Block 的持久状态/旧历史/邻居 z 均 detach，只保留当前更新的局部图。
连续 readout 局部项可进入所属 Block 和 Adapter 参数；无新输入不重复生成 ReadIn 局部导数。
对应 canonical e-prop 与用户输出均值、时间和校准范围澄清。

测试：新增 **7 项通过**；连同 Block/history 回归 **33 passed**。
使用非零解析导数检查平均和衰减，验证延迟参数变化、rho、EMA、所有参数 shape、
没有跨历史/邻居梯度、连续 readout 贡献及 A_g 排除。
当时尚未实现离散控制资格迹和整机 step；LN 均值零这一已说明的数学结果保持不变。

## Phase 14 — 离散控制资格迹

文件：更新 `acnt/plasticity.py`、`acnt/runtime.py`，新增 `tests/test_control_eligibility.py`。
实现 `d_logpi=(a-p)/tau` 和 `sum(d_logpi.detach()*q)` 局部求导，
多个独立动作贡献求和而非平均；不对硬阈值、p 或 world 求导。
内部、连续和离散资格项共同更新同一实际参数；route/hand 使用原始实际采样，
route 最终 active 角色覆盖不伪造原始采样事件。执行开关不改变采样或资格项。
对应 canonical 最终离散控制与 e-prop 条款。

测试：新增 **7 项通过**；连同一般塑性 **14 passed**。
包含手算 score/Jacobian、共享参数多动作求和、跨时间递推、延迟参数更新、
连续鼠标坐标排除、开关不影响资格项、route 角色覆盖不替换采样。
当时只剩整机入口与整体验收，未发现新增规范冲突。

## Phase 15 — 最小整机 smoke

文件：更新 `acnt/runtime.py`、`acnt/__main__.py`、README 和原 CLI 测试；
新增 `tests/test_runtime_smoke.py`，保存独立验收日志。
创建八个小 Block（六器官加两个普通 Block），完整 mock 1080p/原始音频进入 ReadIn，
依次执行 Block、route、hand/speak/goodness、机械日志、稀疏同刻 teacher、资格项和参数更新。
默认 CLI 为整机 smoke；`--core-only` 保留此前 Core 入口。
对应用户 Phase 15 的 Runtime/Plasticity Skeleton 验收。

测试：整机新增 **3 项通过**，覆盖 20 步连续运行、稀疏 teacher 与完整日志、
延迟 goodness 后继续下一轮、真实 subprocess 命令行和输出文件。
最终 `python -m pytest -q`：**146 passed in 4.64s**。

独立命令行验收：**64 步、8 Block、256 条机械日志、16 次 teacher、31 种 active 集合**。
全部参数为有限值，88 个参数张量变化；这不是学习效果或 benchmark 声明。
文件位于 `runs/runtime-v0-20260918-223626/`：`summary.json`、`ticks.jsonl`、`mechanical.jsonl`。

当前没有剩余 Runtime v0 实现阶段或未决数学选型。真实 DirectInput、VocalTractLab、
正式数据集、学习效果验证和 Stage 0 均属于用户明确排除的本轮范围；**Stage 0 未运行**。
