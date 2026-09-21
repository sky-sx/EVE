# Stage Zero frozen Teacher 正式实验报告

**结论：NOT SUPPORTED。**

在保持 27 维独立 Bernoulli 动作、现有 ACNT/e-prop、250 ms delayed goodness 和原正式预算不变时，本次冻结 Teacher 表没有使 ACNT 学出稳定的视觉到 one-hot 动作映射。结论仅限这张表及本次预算。

基线与核验的远端 HEAD：`43677b0bce2d10dfdbe217fc8c7670583054644e`。设备：NVIDIA GeForce RTX 5080，CUDA FP32；运行耗时 1248.09 秒。

五个 seeds：11、22、33、44、55。每个 seed：270 initial + 1080 training + 270 frozen，window=270。无筛选、提前停止或结果驱动调参。

## 五 seeds 三阶段结果

| Seed | Phase | Episodes | Exact success | Target probability | Non-target probability | Mean Teacher goodness |
|---|---|---:|---:|---:|---:|---:|
| 11 | initial | 270 | 0.000000000 | 0.507992832 | 0.513001362 | 0.043544733 |
| 11 | training | 1080 | 0.000000000 | 0.514604133 | 0.514988647 | 0.037260034 |
| 11 | frozen | 270 | 0.000000000 | 0.513444277 | 0.516553595 | 0.039347215 |
| 22 | initial | 270 | 0.000000000 | 0.517240667 | 0.515012953 | 0.036716872 |
| 22 | training | 1080 | 0.000000000 | 0.517756066 | 0.514863593 | 0.038022006 |
| 22 | frozen | 270 | 0.000000000 | 0.508959930 | 0.514936189 | 0.036770316 |
| 33 | initial | 270 | 0.000000000 | 0.509285138 | 0.510854934 | 0.036793224 |
| 33 | training | 1080 | 0.000000000 | 0.512512169 | 0.511454045 | 0.035240003 |
| 33 | frozen | 270 | 0.000000000 | 0.508488064 | 0.514050957 | 0.035633251 |
| 44 | initial | 270 | 0.000000000 | 0.480432831 | 0.477393804 | 0.031740466 |
| 44 | training | 1080 | 0.000000000 | 0.481969843 | 0.479728633 | 0.035447671 |
| 44 | frozen | 270 | 0.000000000 | 0.480381244 | 0.480549744 | 0.032458752 |
| 55 | initial | 270 | 0.000000000 | 0.483908865 | 0.493326991 | 0.034538409 |
| 55 | training | 1080 | 0.000000000 | 0.493513198 | 0.494370524 | 0.038576022 |
| 55 | frozen | 270 | 0.000000000 | 0.493133669 | 0.494473417 | 0.031717901 |

## 合并指标

| Phase | Episodes | Exact success | Target p | Non-target p | Mean Teacher goodness | Target-bit hit | Non-target false rate | Mean active actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 1350 | 0.000000000 | 0.499772067 | 0.501918009 | 0.036666741 | 0.503703704 | 0.501623932 | 13.545925926 |
| training | 5400 | 0.000000000 | 0.504071082 | 0.503081088 | 0.036909147 | 0.506111111 | 0.502706553 | 13.576481481 |
| frozen | 1350 | 0.000000000 | 0.500881437 | 0.504112780 | 0.035185487 | 0.491111111 | 0.502792023 | 13.563703704 |

## 参数与 eligibility（只作路径审计）

| Seed | Parameter delta norm | Training updates | Eligibility norm min/max | Exact successes | Positive Teacher scores |
|---|---:|---:|---|---:|---:|
| 11 | 0.277309770 | 1080 | 308.002716 / 500.416931 | 0 | 1452 |
| 22 | 0.280969667 | 1080 | 334.621857 / 481.161316 | 0 | 1441 |
| 33 | 0.260750303 | 1080 | 317.358337 / 449.642059 | 0 | 1430 |
| 44 | 0.276537393 | 1080 | 321.615448 / 477.595734 | 0 | 1389 |
| 55 | 0.286863406 | 1080 | 320.723083 / 499.916077 | 0 | 1395 |

全部 5,400 次训练有参数更新，每个 seed 的 200/200 参数张量发生变化。7,107/8,100 次动作的 Teacher goodness 大于 0；exact success 为 0/8,100。参数变化和正分数均未被当作学习成功。

## 审计结果

- 全部 8,100 行重新计算 bucket、Teacher mean、独立 exact match 和 training delta，完全一致。
- 全部 250 ms delivery timing 和 eligibility exp(-1) × rho 衰减检查通过。
- 所有 initial/frozen 阶段参数逐位不变；固定反馈不变、设备约束通过，NaN/Inf=0。
- 运行源码 SHA256 全部匹配；`acnt/*.py` 对初始提交无修改，runtime dependency 无修改。
- 五 seeds 的 initial 参数哈希和全部 1,350 次 initial 的目标顺序、q/p、action bits、时间与旧正式实验逐项一致。
- 仓库表、run 表及五份 seed 表字节一致，SHA256 和 calibration metadata 通过审计。

## Teacher calibration

使用 DeepSeek `deepseek-flash`、固定中文 system prompt v1、固定本地 RNG seed=20260921。Teacher 只收到实际 render() 图像及 27-bit Hand ReadOut/按键名称；隐藏标签仅在本地审计记录中存在。无 Goodness ReadOut 训练，无 GoodnessAdapter，无额外 shaping。

Smoke：54 cells × 1；正式：54 cells × 5，270 次有效评价、270 次 HTTP 尝试，无重试。所有 270 个随机实例、实际图片和完整请求哈希独立重建一致。所有评分有限且在 [0,1]；54 cells 完整；mean/原始评分/总体方差及标准差一致。

冻结表 SHA256：`7c40b1d32f4cc67a5ab64341bfc29c58a3297878395274dd7133a0792a421435`。

`M[1,0]=0.58`，原始分数 `[0,1,1,0.9,0]`，std=0.4749736834815167；`M[0,0]=0`，五个原始分数均为 0。

训练前已输出并逐项检查所有 54 个 cell。存在非单调性，以及 k=14、16、22、23、25 时 M[1,k]<M[0,k]。这一 Teacher 语义不一致性是本轮结果的限制；未手工修正、强制单调、筛除样本或依据训练结果重校准。方差和标准差仅用于审计，学习只查询 mean。

API Key 只读加载到校准进程内存；未复制到其他文件、未写入代码/配置/日志/Git。训练期间无 DeepSeek 调用。

完整 54-cell 表及每 cell 标准差：

| k | M[1,k] | std[1,k] | M[0,k] | std[0,k] |
|---|---|---|---|---|
| 0 | 0.58 | 0.474973683 | 0 | 0 |
| 1 | 0.1 | 0.2 | 0 | 0 |
| 2 | 0.202666667 | 0.165494545 | 0.02 | 0.04 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 0.04 | 0.08 | 0 | 0 |
| 5 | 0.08 | 0.116619038 | 0 | 0 |
| 6 | 0.028 | 0.056 | 0 | 0 |
| 7 | 0.044 | 0.0595315043 | 0.02 | 0.04 |
| 8 | 0.0444 | 0.0888 | 0 | 0 |
| 9 | 0.08 | 0.04 | 0 | 0 |
| 10 | 0.02 | 0.04 | 0.0076 | 0.0152 |
| 11 | 0.016 | 0.032 | 0 | 0 |
| 12 | 0.14 | 0.174355958 | 0.0234 | 0.0317212862 |
| 13 | 0.0274074074 | 0.03902797 | 0.01 | 0.02 |
| 14 | 0.02 | 0.04 | 0.03 | 0.04 |
| 15 | 0.1525 | 0.209821353 | 0.006 | 0.012 |
| 16 | 0.01 | 0.02 | 0.02 | 0.0244948974 |
| 17 | 0.044 | 0.0542586399 | 0 | 0 |
| 18 | 0.1 | 0.2 | 0 | 0 |
| 19 | 0.036 | 0.072 | 0.02 | 0.04 |
| 20 | 0.03 | 0.06 | 0.0074 | 0.0148 |
| 21 | 0.04 | 0.08 | 0.006 | 0.012 |
| 22 | 0 | 0 | 0.2164 | 0.343292645 |
| 23 | 0 | 0 | 0.008 | 0.016 |
| 24 | 0.19 | 0.365184885 | 0.02 | 0.04 |
| 25 | 0 | 0 | 0.116 | 0.192831533 |
| 26 | 0 | 0 | 0 | 0 |

## 测试

完整通过：**246 passed in 12.12s**，无失败、错误或跳过；CUDA 测试执行。

首次按要求执行 `pytest -q`：215 passed、1 failed、30 errors、2 warnings。1 个新增测试误把固定 prompt 的 reward shaping 短语识别为隐藏标签，已修正；30 个 fixture 错误及 cache 警告是既有 Windows 临时目录权限问题。

完整重跑命令：`pytest -q --basetemp=.test-tmp/teacher-20260921-a -o cache_dir=.test-tmp/teacher-cache-20260921`。详细输出见原始数据目录的 `pytest.txt`。

## 修改文件与核心 diff

| 文件 | 改动 |
|---|---|
| experiments/stage_zero/environment.py | 严格 27-bit bucket、冻结表加载/校验/均值查询；exact_goodness 只作审计 |
| experiments/stage_zero/harness.py | 仅把 goodness 验证改为有限 [0,1]；delayed/plasticity 逻辑不变 |
| experiments/stage_zero/runner.py | 三阶段查同一表、独立 exact 指标、修正 successes、路径参数和表/哈希/元数据快照 |
| experiments/stage_zero/evaluation.py | mean_goodness 明确聚合 Teacher 标量，行为判据不变 |
| experiments/stage_zero/audit.py | 重算表值/bucket/exact/delta，检查表哈希及全部旧约束 |
| experiments/stage_zero/calibrate_teacher.py（新增） | 独立 stdlib HTTP/PNG 校准，严格解析、有限重试、只读密钥、无假数据回退 |
| experiments/stage_zero/teacher_goodness.json / teacher_goodness.md（生成） | 正式冻结表、原始样本和可读 54-cell 表 |
| experiments/stage_zero/DESIGN.md / README.md | 更新现有协议和运行说明，保留历史实验 |
| tests/test_stage_zero_environment.py | bucket、完整表、边界、实际 PNG/I/O、无 key 与失败重试 |
| tests/test_stage_zero_harness.py | 连续标量完整 delayed e-prop 数学核验及非法值拒绝 |
| tests/test_stage_zero_runner.py | 表查询/独立 exact/运行 artifacts，ACNT 与基线一致 |
| tests/test_stage_zero_evaluation.py | Teacher goodness 与成功率解耦、仅分数上升不能判定学会 |
| tests/test_stage_zero_audit.py | table/value/bucket/exact/delta/timing 篡改检测 |
| reports/stage_zero_teacher_preflight.md / stage_zero_teacher_report.md（新增） | 测试、预检、校准、正式结果和限制 |

## 原始数据

- 正式 run：`D:\EVE\EVE0.6\runs\stage_zero\teacher-formal-20260921`。完整逐 episode CSV、q/p/bits、学习曲线、per-class、参数变化、final_state.pt、aggregate、report、audit.json、冻结表和元数据。
- 校准与验证：`D:\EVE\EVE0.6\runs\stage_zero\teacher-validation-20260921`。pytest、校准控制台、270-request 独立校准审计、旧新 initial 对照、source_snapshot 和 tracked_changes.patch。
- Smoke：`D:\EVE\EVE0.6\runs\stage_zero\teacher-smoke-20260921`。

API 图像格式依据 [DeepSeek 官方 Vision 文档](https://api-docs.deepseek.com/guides/vision/)。
