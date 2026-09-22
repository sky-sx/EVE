# ACNT Stage Zero 正式实验报告

**结论：NOT SUPPORTED。** 在本次固定协议和预算下，没有观察到可重复的视觉→动作映射学习。
五个 seed 的初始化、训练、冻结测评均为零 exact-match 成功；参数确实更新过，但正奖励从未出现。
此结论只评价本次 Stage Zero 条件下的学习证据，不否定整个 ACNT 架构的可能性。

## A. 实现与版本

实际基线 / 测试 commit：`be9c1cf604b5e3e66d57774cd1ab1704df68fdd5`。
开始时工作树干净，位于 main；本地最近两条提交为 `be9c1cf`、`80d6153`。
Git HTTPS 直连超时后，通过 GitHub commit API 对 main 的实时解析核对了相同 SHA。
新增实验代码位于该 commit 之上的工作树，未假称这些新文件已包含在基线 commit 中。
训练时保存了 21 个核心源码、实验源码、测试及协议文件的 SHA256；事后全部匹配。
训练完成后新增的 audit 模块与审计测试不参与训练，单独属于结果核验工具。

新增文件与职责：

| 路径 | 职责 |
|---|---|
| `experiments/__init__.py`、`experiments/stage_zero/__init__.py`、`__main__.py` | 包与命令行入口 |
| `experiments/stage_zero/environment.py` | 固定 5×7 字体渲染、A–Z/绿屏映射、均衡顺序、严格 scalar Goodness |
| `experiments/stage_zero/harness.py` | 10×100 组合、真实 Eye 事件、snapshot 调度、27/0 Hand、延迟送达 |
| `experiments/stage_zero/runner.py` | 初始/训练/冻结阶段、CSV/JSON、设备、参数和源码指纹 |
| `experiments/stage_zero/evaluation.py` | 逐类统计、窗口曲线、预先声明的证据判据 |
| `experiments/stage_zero/audit.py` | 从完整 CSV 独立重算奖励、延迟、衰减、统计与源码指纹 |
| `experiments/stage_zero/DESIGN.md`、`README.md` | 实现前协议和复现说明 |
| `tests/test_stage_zero_environment.py`、`test_stage_zero_harness.py` | 环境和真实 ACNT 路径测试 |
| `tests/test_stage_zero_evaluation.py`、`test_stage_zero_runner.py`、`test_stage_zero_audit.py` | 统计、落盘、冻结及损坏日志检测 |
| `reports/stage_zero_preflight.md`、本报告 | 基线/运行预算和正式证据 |

旧文件仅修改根 `README.md`，加入实验入口与结论，纠正“未运行 Stage 0”的旧状态。
**`acnt/`、`pyproject.toml` 和旧测试均未修改，无核心 bug 修复，无架构或版本升级。**

## B. 系统边界与架构完整性

10 个 Block 均为 100 neurons，始终 active：Block 0=Eye ReadIn，Block 1=Hand ReadOut，2–9=普通 Core。
只实例化原有 EyeAdapter 与 HandAdapter(27 discrete, 0 continuous)。Eye 输入保持 FP32 RGB
`[3,1080,1920]`，黑底白字固定居中、字体 cell=5×7、像素放大倍数=64；绿屏=(0,1,0)，
对应原始 RGB=(0,255,0)。无缩小输入、one-hot 输入、文字编号输入或图像增强。

独立 harness 通过方法引用直接复用 Runtime.encode_readin、decode_readout、_observe_discrete，
并调用原有 Runtime.enable_plasticity；没有复制 Block 或 e-prop 数学。
该初始化方法会额外创建未使用的 route feedback buffers，已明确记录：没有 Route Block/Adapter、
没有 route forward、sample 或 eligibility。固定 feedback 全部仍为不可训练 buffers，checkpoint 审计确认未改变。

没有调用 Runtime.step / update_blocks / generate_hand / generate_route / generate_speak /
generate_goodness。正式 Hand Runtime 的 85+2 接口保持原状。Stage Zero 用原有 HandAdapter
生成 27 个 q，再调用原有 sample_discrete，保留独立 Logistic sampling，threshold=0。
未构造 Ear、Speak、Goodness、Route adapter，未运行 replay/sleep/LLM/VLM，无 executor 或物理键鼠注入。

target 只在环境渲染、严格奖励和记录指标时使用；`StageZero.act(image, ...)` 没有 target 参数。
没有 CE、softmax/categorical loss、argmax、teacher label 梯度、optimizer/backward 或新增分类器。
Eye 内部卷积和 Hand 内部 MLP 都是原有 Adapter，并非新增旁路。

## C. 一次 episode 的实际路径

相对时间为三次独立呈现 `0/250/500 ms`，动作产生于 `500 ms`，唯一 scalar reward 于 `750 ms` 送达；
下一 trial 从 `1000 ms` 开始。每帧都是相同刺激的**新事件**，ReadIn pulse 每次都被消费；不是 o 常驻。
三轮旧 snapshot 传播让当前刺激可经 Eye→普通 Block→Hand 到达动作，最后一帧保留当前局部图产生 Eye eligibility。
前两轮无动作、无 eligibility 观察，图被丢弃；最后一轮之后也丢弃局部图，不跨历史反传。
等待 reward 期间没有其他动作，不做 wall-clock sleep；时延真实进入 eligibility 的逻辑时间计算。

| 环节 | 实际源码函数 |
|---|---|
| 渲染 RGB | environment.VisualEnvironment.render → runner.run_episode |
| Eye 事件 | StageZero.act → update_frame → 原 Runtime.encode_readin → EyeAdapter.forward |
| 传播 | Core.source_snapshot（每轮一次）→ Core.update_block → Block.update / _update |
| 脉冲消费 | 原 Core.update_block 将 Eye o 替换为零 buffer；全零图像仍强制更新 |
| Hand | 原 Runtime.decode_readout → HandAdapter.forward → acnt.control.sample_discrete |
| eligibility | 原 Runtime._observe_discrete → EligibilityBank.observe_control；adapter-only score 梯度；固定 B_i @ score → observe_vector 的 local VJP |
| 延迟 Goodness | 环境 exact_goodness → StageZero.deliver_goodness → 原 Plasticity.apply_goodness |
| 衰减与更新 | 原 EligibilityBank.advance → 原参数更新/乘 rho/EMA baseline |

保持 ticktime=tau=0.25 s、hold_tick=4、learning_rate=0.001、rho=0.9、EMA alpha=0.1、initial g_bar=0.5，
不裁剪参数。250 ms 衰减为 `exp(-1)=0.367879`，更新后 eligibility 再乘 0.9。
Eye adapter 的参数在 Eye Block group，Hand 离散直达梯度只进入 Hand adapter；Block 内部仍使用原固定随机反馈。

## D. 测试与审计

- 修改前原始 `pytest -q`：151 passed、2 fixture errors（默认 Windows pytest 临时目录拒绝访问）。
- 用仓库内新 basetemp/cache 路径重跑完整旧测试：**153 passed**。
- 正式运行前完整测试：**201 passed、0 skipped、0 failed**；实际 CUDA smoke 通过。
- 加入事后审计及损坏日志测试后，交付前完整测试：**208 passed、0 skipped、0 failed**，11.14 秒。

覆盖全部用户要求：27 类与映射、27/0 输出、不同 Eye/Hand Block、10 Block active、禁止 ReadOut 和物理输出、
strict exact reward/多按失败、非零延迟、原 eligibility 衰减、learn=False 不变、正 scalar 合法更新、
参数与 traces 有限、冻结不更新、禁止监督分类损失；另测旧 snapshot 执行顺序不变、零帧 pulse、
固定 feedback 不影响前向、不受 Goodness 更新、真实原始数据落盘及损坏记录拒绝。
正 scalar 注入只用于单元测试验证更新链，**未替代主实验 strict reward**。

事后从 8,100 行 raw CSV 独立重算 action→reward、500/250 ms 时序、baseline、原始 eligibility 衰减、
各类样本数和阶段统计，全部一致；初始与冻结阶段参数逐位不变；所有正式日志 NaN=Inf=0。
五个 checkpoint 均重新载入，参数 hash 与原 summary 相同，固定 feedback 与同 seed 初始化逐位相同。

## E. 正式运行配置与预算

运行开始 UTC：`2026-09-20T12:46:36.886232+00:00`；本地时区 Asia/Shanghai。
设备：NVIDIA GeForce RTX 5080，CUDA 13.0，PyTorch 2.10.0+cu130，Python 3.11.9。
FP32；TF32 关闭；确定性算法打开；CUBLAS_WORKSPACE_CONFIG=:4096:8；CPU 调度线程=1。
CPU 与 CUDA 两种 smoke 均通过，正式实验选择 CUDA，未改模型。

| 项目 | 数量 |
|---|---:|
| Seeds | 11,22,33,44,55，共 5 个 |
| 每 seed 初始化测评 | 270；每类 10 次 |
| 每 seed 训练 | 1,080；每类 40 次 |
| 每 seed 冻结测评 | 270；每类 10 次 |
| 总训练 | 5,400 |
| 总无更新测评 | 2,700 |
| 总动作 / 总 Eye 帧 | 8,100 / 24,300 |
| 纯训练时间 | 933.351 秒，15.56 分钟 |
| 完整正式运行时间 | 1185.067 秒，19.75 分钟 |

每 27 次将全部类别均衡洗牌，phase 的 target RNG 和 Hand RNG 独立，各阶段使用不同种子。
每阶段重置 dynamics/history/eligibility，保留权重；baseline 保留但 evaluation 不使用或更新它。
冻结阶段重新排列的序列不参与参数更新；冻结前后整向量逐位比较。
预算在训练前按 smoke 耗时固定，无挑 seed、无奖励调参、无中途挑选最佳窗口或提前终止。

## F. 全部结果

### 总体表现

| 阶段 | episodes | exact success | mean Goodness | target p | non-target p | target-bit hit | non-target false | 平均 active bits |
|---|---|---|---|---|---|---|---|---|
| initial | 1350 | 0.00% | 0.0 | 0.499772 | 0.501918 | 50.3704% | 50.1624% | 13.5459 |
| training | 5400 | 0.00% | 0.0 | 0.500945 | 0.502109 | 50.2037% | 50.3533% | 13.5939 |
| frozen | 1350 | 0.00% | 0.0 | 0.500098 | 0.502189 | 48.5185% | 49.9316% | 13.4674 |

target p 由 0.499772 到 0.500098 的微小变化没有成功率支持；non-target p 也由 0.501918 上升至 0.502189。
不存在“目标提高且非目标降低，冻结后保留并多 seed 复现”的证据组合。

### 全部 seed（没有删除失败项）

| seed | 初始/训练/冻结 exact | target p 初始→冻结 | non-target p 初始→冻结 | 参数总 Δ L2 |
|---|---|---|---|---|
| 11 | 0 / 0 / 0 | 0.507993 → 0.507432 | 0.513001 → 0.514518 | 0.173757 |
| 22 | 0 / 0 / 0 | 0.517241 → 0.514497 | 0.515013 → 0.516408 | 0.174884 |
| 33 | 0 / 0 / 0 | 0.509285 → 0.514978 | 0.510855 → 0.509904 | 0.165530 |
| 44 | 0 / 0 / 0 | 0.480433 → 0.475947 | 0.477394 → 0.478456 | 0.177460 |
| 55 | 0 / 0 / 0 | 0.483909 → 0.487635 | 0.493327 → 0.491659 | 0.174350 |

只有 seed 33、55 的冻结 tendency 方向同时符合期望，但它们没有一个 exact success，且无跨 seed 稳定趋势。
随机 recurrent 状态、序列与动作抽样的变化也会改变这些均值，不能将微小方向变化认定为学习。

### 训练曲线（五 seed 合并，每 seed 的 episode 编号）

| 训练窗口 | 总样本 | mean g / exact | target p | non-target p | active bits | 窗口最大 Δ参数 L2 |
|---|---|---|---|---|---|---|
| 1–270 | 1350 | 0 / 0 | 0.500217 | 0.501990 | 13.6074 | 0.0920466 |
| 271–540 | 1350 | 0 / 0 | 0.498129 | 0.502192 | 13.5963 | 0 |
| 541–810 | 1350 | 0 / 0 | 0.502119 | 0.502006 | 13.6496 | 0 |
| 811–1080 | 1350 | 0 / 0 | 0.503315 | 0.502248 | 13.5222 | 0 |

每 seed 的全部窗口另存 `learning_curve.csv`，未平滑或删去失败窗口。
窗口后的 tendency 波动不代表继续学习：第二个窗口开始，所有 seed 记录的参数增量 L2 都是零（FP32 日志精度）。

### 27 类逐类结果

每类合并分母：初始 50、训练 200、冻结 50；五个 seed 单独的每类 exact success 也全部为 0。
以下展示所有类，逐 seed 完整数据保存在各 `per_class.csv` 和 `summary.json`。

| 动作 | 初始/训练/冻结成功数 | target p 初始 | target p 冻结 | non-target p 冻结 | 冻结 target-bit hit |
|---|---|---|---|---|---|
| A | 0/50；0/200；0/50 | 0.57614 | 0.58009 | 0.49715 | 56% |
| B | 0/50；0/200；0/50 | 0.43882 | 0.42994 | 0.50989 | 44% |
| C | 0/50；0/200；0/50 | 0.52973 | 0.53702 | 0.50217 | 54% |
| D | 0/50；0/200；0/50 | 0.44850 | 0.45261 | 0.50439 | 40% |
| E | 0/50；0/200；0/50 | 0.49908 | 0.51252 | 0.50295 | 52% |
| F | 0/50；0/200；0/50 | 0.43090 | 0.44929 | 0.50256 | 50% |
| G | 0/50；0/200；0/50 | 0.55668 | 0.55906 | 0.49899 | 62% |
| H | 0/50；0/200；0/50 | 0.42894 | 0.45484 | 0.50408 | 52% |
| I | 0/50；0/200；0/50 | 0.48672 | 0.49173 | 0.50374 | 46% |
| J | 0/50；0/200；0/50 | 0.52161 | 0.50100 | 0.49910 | 44% |
| K | 0/50；0/200；0/50 | 0.47723 | 0.48521 | 0.50199 | 50% |
| L | 0/50；0/200；0/50 | 0.46833 | 0.47890 | 0.50478 | 52% |
| M | 0/50；0/200；0/50 | 0.51715 | 0.50817 | 0.49804 | 44% |
| N | 0/50；0/200；0/50 | 0.53940 | 0.53547 | 0.50115 | 50% |
| O | 0/50；0/200；0/50 | 0.48606 | 0.49225 | 0.50696 | 50% |
| P | 0/50；0/200；0/50 | 0.55012 | 0.54347 | 0.50099 | 56% |
| Q | 0/50；0/200；0/50 | 0.55255 | 0.53266 | 0.50081 | 60% |
| R | 0/50；0/200；0/50 | 0.51809 | 0.52443 | 0.49814 | 44% |
| S | 0/50；0/200；0/50 | 0.48542 | 0.50545 | 0.50655 | 40% |
| T | 0/50；0/200；0/50 | 0.45307 | 0.45485 | 0.50375 | 46% |
| U | 0/50；0/200；0/50 | 0.51348 | 0.50402 | 0.50086 | 34% |
| V | 0/50；0/200；0/50 | 0.43584 | 0.42620 | 0.50369 | 34% |
| W | 0/50；0/200；0/50 | 0.53167 | 0.50227 | 0.49817 | 50% |
| X | 0/50；0/200；0/50 | 0.53220 | 0.53509 | 0.49821 | 58% |
| Y | 0/50；0/200；0/50 | 0.57384 | 0.55343 | 0.49971 | 50% |
| Z | 0/50；0/200；0/50 | 0.42805 | 0.43566 | 0.50675 | 44% |
| MOUSE_LEFT | 0/50；0/200；0/50 | 0.51424 | 0.51704 | 0.50353 | 48% |

### 参数、eligibility 与 baseline

| seed | 改变的参数张量 | 参数 ΔL2 | 最大单元素 |Δ| | 动作时 eligibility norm 范围 | 最后非零更新 episode |
|---|---|---|---|---|---|
| 11 | 200/200 | 0.173757 | 0.0022488 | 321.274–500.417 | 232 |
| 22 | 200/200 | 0.174884 | 0.0027049 | 328.662–481.161 | 235 |
| 33 | 200/200 | 0.165530 | 0.0024169 | 315.772–453.886 | 235 |
| 44 | 200/200 | 0.177460 | 0.0019604 | 330.610–477.596 | 252 |
| 55 | 200/200 | 0.174350 | 0.0023264 | 317.679–499.916 | 241 |

五 seed 的 Eye、Core、Hand 参数均包含合法更新，所有 200 个可训练参数张量都改变过。
全体单元素改变量范围：-0.0024169 至
0.0027049；未裁剪、未出现 NaN/Inf。
每个参数张量的 norm、delta norm、极值与变化元素数保存在 `parameter_summary.json`。

## G. 失败分析与证据边界

**最强证据指向严格奖励稀疏，以及零奖励下 baseline 导致的学习信号衰减。**

1. **Hand sampling / reward sparsity**：平均同时激活约 13.5 个动作，非目标 bit false activation 约 50%。
   若所有 p=0.5，指定正确 one-hot 事件概率为 `2^-27 ≈ 7.45×10^-9`，并不是 `1/27`。
   用每次真实 p 计算 `p_target × ∏(1-p_non_target)`，5,400 次训练的概率和仅
   **3.37148091e-05**；全 8,100 次概率和约
   **5.05432271e-05**。
   这是沿实际轨迹的条件事件概率总和，用于衡量稀疏性；不是对自适应轨迹给出的独立同分布显著性证明。
   实际正奖励为零，与该极低可达概率一致。strict binary goodness 在本次运行中造成了正奖励完全缺失。

2. **Learning signal / optimization**：所有 g*=0，因此原有机制精确满足
   `g_bar(n)=0.5×0.9^n`，训练末端为 **1.90932649e-50**。
   初期 delta<0 仍会改变参数，但没有成功动作作为正向反馈；这不能证明学到了 target 对应关系。
   第 232–252 episode 后记录到的参数增量全部为零。eligibility 依然非零，消失的是更新标量的有效尺度。
   1,080 个训练 episode 中后续约 800 多次没有可测参数学习；未据此提前停止或改奖励。

3. **Eligibility decay**：动作时 norm 为 315.77–500.42，250 ms 后按 exp(-1) 衰减、更新后乘 rho=0.9。
   单元测试逐参数验证，正式所有 CSV 行也验证了 norm 比率。没有证据显示 traces 被意外清零或时延未生效。

4. **Visual encoding**：27 个编码在五 seed 均可区分；最小字母 pairwise L2 为
   0.014099–0.153502。
   独立只读 checkpoint 诊断在相同重置状态下换 A/B，Hand q 的 L2 差为
   0.031088–0.093317；
   A/绿屏也均不同。说明像素经过正式 Eye/Core 能影响动作 tendency；不证明编码足够支持学习。

5. **Block dynamics / fixed feedback / parameter scale**：所有 snapshot/pulse 测试、数值稳定性、参数更新和
   feedback 完整性通过，没有找到需要修改数学的 bug。但这次没有获得正奖励，无法将学习失败单独归因于
   Block dynamics、随机反馈方向质量或学习率。不能凭 finite/nonzero eligibility 宣称这些因素已被充分验证。

诊断仅包括不更新参数的编码检查、checkpoint 一致性、冻结前向视觉敏感性，以及独立测试中的 synthetic
positive scalar。主训练和冻结测评未改 reward、action mechanism 或 architecture。
后续可另立协议研究严格奖励的首次命中时间、受控 scalar 对参数方向的影响；本次没有运行奖励塑形或监督分类实验。

## H. 结论、限制与复现

**NOT SUPPORTED：当前 Stage Zero 条件下未观察到 ACNT 的真实 end-to-end mapping learning。**
已支持的是执行链能够接受真实图像、产生独立 Hand 动作、生成 local eligibility，并在延迟 scalar 到达后更新参数。
没有支持的是这些更新带来视觉→正确 one-hot 动作的稳定学习。五 seed 的零成功和冻结结果必须保留。
该有限预算不能证明无限训练、其他明确协议或整个架构不可能学习。

训练代码中声明了 SUPPORTED/PARTIAL 的趋势与 Wilson 区间门槛。Wilson 在 recurrent、均衡序列环境下
仅为保守描述性门槛，不能当作严格独立样本推断；本次结论直接由零成功与无稳定趋势得到。

复现时从仓库根执行（输出目录必须不存在）：

```powershell
python -m experiments.stage_zero --output runs/stage_zero/reproduction --device cuda --seeds 11 22 33 44 55 --train-episodes 1080 --evaluation-episodes 270 --window 270
python -m experiments.stage_zero.audit runs/stage_zero/reproduction
pytest -q --basetemp=.test-tmp/stage-zero-reproduction-tests -o cache_dir=.test-tmp/stage-zero-pytest-cache
```

CPU 功能测试和 smoke 可用 `--device cpu`。CUDA/CPU action RNG 不保证跨设备逐位相同；本次两个 smoke 分开保存。
同设备复现应保留 PyTorch/CUDA 版本、配置、源码与所有 seed；不得覆盖或仅选择成功 run。

原始目录：`runs/stage_zero/formal-20260920/`，仅本地保留；仓库忽略 runs 内容，未上传 bulk logs/checkpoints。

| 文件 | 内容 |
|---|---|
| config.json / environment_config.json | 固定协议、glyph 字体表、全部图像指纹 |
| run_metadata.json | commit、源码 hash、Python/torch/CUDA、seeds、超参数、时间与完成状态 |
| seed_*/episode_log.csv | 全部 q/p、sampled actions、target、奖励、时延、参数/eligibility/NaN/Inf |
| seed_*/summary.json / parameter_summary.json | 阶段/每类统计、参数变化明细 |
| seed_*/learning_curve.csv / per_class.csv | 每个 seed 的原始窗口和逐类结果 |
| seed_*/visual_diagnostics.json / final_state.pt | 只读视觉诊断与最终参数/feedback checkpoint |
| aggregate.csv / aggregate.json / report.md | runner 原始汇总，未用事后报告替代 |
| audit.json / checkpoint_audit.json / checkpoint_audit.py | 8,100 行独立审计、checkpoint/视觉敏感性诊断及脚本 |
| final_pytest.txt / formal_report.md | 最终完整测试输出、本报告副本 |

五个 seed 的原始文件共 35 个、108,881,538 bytes
（约 103.84 MiB，不含根目录元数据和审计文件）。
每个文件的 SHA256 和字节数保存于 audit.json；最大 checkpoint 约 18 MiB。
单独的 CPU/CUDA smoke 目录也保留，未并入正式训练统计。完整测试/审计结果与原始数据足以回答本轮问题。
