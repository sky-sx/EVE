# Persistent Delta-W Plasticity 实现与验证报告

日期：2026-09-27。状态：**IMPLEMENTED / NOT YET FORMALLY VALIDATED**。OR/XOR 200,000 step 机制 sanity：两个任务各 3/3 成功。尚无正式 ACNT 行为学习结论。

## 起点与基线

- 实际起始 HEAD：`be8a8fa34ec8fcc9dd8963142a6268fafc77e1e2`，main，工作区干净；未 reset 或回退。
- 起始最近五条提交：`be8a8fa ACNT_new_try,code_change_0002`、`2cc8c35 ACNT_stage0_ABC_0001`、`13f4fc9 ACNT_code_change_0003`、`d19e8f1 ACNT_code_change_0003`、`f1e63db fix(acnt): restore per-neuron CTM NLM with real-time history`。
- 原样运行 `python -m pytest -q`：162 passed、7 failed、2 errors。7 项失败为旧 Block/Plasticity API 与旧实验调用不一致；2 项错误为系统 pytest 临时目录 PermissionError，另有 cache 权限警告。
- 本地沙箱启动失败，获准使用沙箱外执行；后续测试用 `PYTEST_ADDOPTS="--basetemp=.test-tmp/delta-w-final-02 -o cache_dir=.test-tmp/delta-w-cache"` 避开不可写的系统临时目录。未改动或删除那个系统目录。

## 实现与时序

普通 Block / Adapter 的所有参数元素（含 bias）统一注册；Goodness designated Block + Adapter 排除。每个普通张量有同 shape/device 的 FP32、非 Parameter、无梯度 delta_w。

1. 初始化 `Δw_i = s_i δ`，独立等概率正负符号。
2. 首次有效 Goodness 仅建立 previous_goodness。
3. 下一次 forward 前，从全部普通元素统一无放回抽 `max(1, floor(p*N))` 个元素，执行 `W[S] += ΔW[S]`。
4. 正常系统继续运行，消费对应唯一 Goodness，计算 `M=G-current_previous_G`。
5. `M<0` 时只执行 `ΔW[S] *= -1`；否则不变。W 永不回滚。更新 previous_goodness 并消费 pending。

默认 δ=0.001、p=0.001、parameter_clip=None、plasticity_seed=0。独立 CPU torch.Generator 初始化符号，独立 random.Random 采样；注册顺序决定 flat offset。ABC 和 sanity 的学习 seed=实验 seed+300000，不复用动作 RNG。

learn=False 清 runtime credit，保留 W/ΔW/RNG；恢复后首轮重建 baseline。最多一个 unresolved trial，第二次 begin_trial 明确抛 RuntimeError。只裁剪被移动的参数元素；有限性检查在裁剪前检测溢出。

删除旧 eligibility 状态、衰减/EMA baseline、accumulate、control score/observer、adapter hooks、Block learning flags、SynapseFrame、learning_frames、LN_vjp、NLM local_vjp 及其缓存。未恢复 z_bar 或 perturbation_scale，未保留 compatibility mode。Block/Core 的神经动力学与角色绑定保持不变。

## OR/XOR：实际短预算结果

命令：`python -m experiments.delta_w_sanity --steps 1000 --seeds 11 22 33 --output runs/delta_w_sanity/smoke.json`。CPU，2→3→1 MLP，tanh/sigmoid，13 个参数元素，每轮选 1 个，δ=.001、p=.001。Goodness=`1-MSE` 是 toy truth table 目标，仅用于机制 sanity，不是 ACNT 正式 Goodness。没有梯度、optimizer、双边比较或回滚。

| 任务 | seed | initial loss | final loss | initial accuracy | final accuracy | success |
|---|---:|---:|---:|---:|---:|---|
| OR | 11 | 0.283388048410 | 0.236609116197 | 0.50 | 0.50 | False |
| OR | 22 | 0.280968815088 | 0.240457445383 | 0.00 | 0.75 | False |
| OR | 33 | 0.309220641851 | 0.258890002966 | 0.25 | 0.50 | False |
| XOR | 11 | 0.255980789661 | 0.251420259476 | 0.75 | 0.50 | False |
| XOR | 22 | 0.253921478987 | 0.251045763493 | 0.25 | 0.50 | False |
| XOR | 33 | 0.260133177042 | 0.251538544893 | 0.50 | 0.75 | False |

success 定义为四个输入全部分类正确。OR 0/3、XOR 0/3 成功；六次 MSE 均下降，部分 accuracy 下降。这仅证明短实验产生了可测量的损失变化，不能宣称学会 OR/XOR，更不能宣称 ACNT 行为学习已支持。当时仅运行了短预算；后续 200,000 step 结果见下一节。

本地原始 JSON：`runs/delta_w_sanity/smoke.json`；SHA256：`4c6b94b9e5ab067b4b7e51a439d1c8692ecdb13af493b977029f0eb96d51704d`。原始运行文件按仓库规则留在 runs，不上传；本表保留全部六次结果。

## OR/XOR：200,000 step 结果更新

更新日期：2026-09-27。依据用户已完成的 `runs/delta_w_sanity/200k.json` 重新汇总；本次未重跑训练。当前审阅代码提交为 `84f4bd6`；JSON 未记录运行时 commit、耗时、逐步轨迹或首次成功时间，因此不能据此确认这些信息。

**结果：OR 3/3、XOR 3/3 成功，六次最终 truth-table accuracy 均为 100%。** 相同 seed 的初始 loss/accuracy 与此前 1,000 step 结果逐项一致；两个预算的成功数从各 0/3 提升到各 3/3。两份文件是独立结果快照，不能视为一次训练的连续日志。

与结果配置对应的复现命令：

```powershell
python -m experiments.delta_w_sanity --device cpu --steps 200000 --seeds 11 22 33 --subset-fraction 0.001 --delta-magnitude 0.001 --output runs/delta_w_sanity/200k.json
```

CPU；2→3→1 MLP，tanh/sigmoid；13 个参数元素，每轮选择 1 个；每个任务/seed 200,000 次更新，共 1,200,000 次。plasticity_seed 为实验 seed + 300000。Goodness 为完整四行真值表的 `1-MSE`，success 为以 0.5 为阈值时四个预测全部正确。生产 Plasticity 算法用于该 toy 模型；它并非完整 ACNT 架构。

| 任务 | seed | initial MSE | final MSE | initial accuracy | final accuracy | success |
|---|---:|---:|---:|---:|---:|---|
| OR | 11 | 0.28338804841 | 0 | 0.50 | 1.00 | True |
| OR | 22 | 0.280968815088 | 0 | 0.00 | 1.00 | True |
| OR | 33 | 0.309220641851 | 4.37886568949e-35 | 0.25 | 1.00 | True |
| XOR | 11 | 0.255980789661 | 2.97609686895e-10 | 0.75 | 1.00 | True |
| XOR | 22 | 0.253921478987 | 3.11442788226e-10 | 0.25 | 1.00 | True |
| XOR | 33 | 0.260133177042 | 3.11481618276e-10 | 0.50 | 1.00 | True |

| 任务 | 1,000 step 成功数 | 200,000 step 成功数 | 200,000 step 平均 final MSE |
|---|---:|---:|---:|
| OR | 0/3 | 3/3 | 1.4596218965e-35 |
| XOR | 0/3 | 3/3 | 3.06844697799e-10 |

OR 的两项 MSE 在存储精度下为 0，另一项约 4.38e-35；XOR 三项均约 3e-10。这支持该机制在所测三个 seed、固定真值表和 200,000 step 预算下学会 OR/XOR。文件仅包含初始与最终指标，无法判断中途是否单调改善、达到成功所需最小预算或收敛后长期稳定性；数值 0 也不代表任意精度下误差严格为零。

该结果不提供 held-out 泛化、更多 seed 的成功率或 ACNT ABC 行为学习证据。正式 ACNT 状态仍为 **NOT YET FORMALLY VALIDATED**；此前短预算与历史 Stage Zero 结果保留作为不同实验的记录。

原始文件保留在本地 `runs/delta_w_sanity/200k.json`，SHA256：`3035f4b072c1ce33e59cfc90d6633a39df44bf70cc04e21ba867d7d093fe0f20`。按现有仓库约定上传本报告中的全部六项汇总，不上传 runs 原始文件。

## 实现提交的历史验证记录

- `python -m pytest -q`（上述环境变量）：**178 passed in 4.98s**。覆盖初始化、全局元素抽样、首次 baseline、正/负/零变化、不回滚、冻结恢复、Goodness 排除、无 autograd、pending、裁剪与有限性、Runtime forward 前移动及 action RNG 独立性。
- `python -m acnt --core-only --steps 12`：通过，12 tick。
- `python -m acnt --steps 2`：通过；第一次不移动，第二次选 106/106949 个元素，Goodness 差 -0.3，反转 106 个方向。
- `python -m experiments.stage_zero_abc.runner --device cpu --seeds 11 --initial 3 --training 6 --frozen 3 --output runs/delta_w_abc_smoke`：通过，12 episode，NaN/Inf=0；首次训练不移动，随后五轮各选 3 元素，最后两轮各翻转 3。初始/训练/冻结 exact success 为 1/3、0/6、0/3，仅为集成 smoke。
- CUDA 可用；`python -m experiments.delta_w_sanity --device cuda --steps 12 --seeds 11 --output runs/delta_w_sanity/cuda_smoke.json`：OR/XOR 均完成，有限值，无 autograd。仅设备 smoke。
- stale-reference audit：生产 acnt、活动实验与 active production tests 中无旧学习 API；当前规范只在否定约束中出现 eligibility；其他文档仅在明确历史说明中提旧机制。

## 历史事实

此前 Local Plasticity / CorrelationRule Stage Zero 结论仍为 **NOT SUPPORTED**。strict-reward exact success：初始 0/1350、训练 0/5400、冻结 0/1350。原 reports 内容未修改；原阶段报告与旧实验源码/说明逐字节保存到 `archive/stage_zero_superseded`。归档依赖当时 Git 版本，不保证在当前 production API 上重新训练。当前唯一活动 Stage Zero 是 ABC。

## 修改文件与边界

- 生产：acnt/plasticity.py、block.py、runtime.py、control.py、__main__.py。
- 实验：新增 experiments/delta_w_sanity.py；更新 stage_zero_abc 的 harness.py、runner.py、__init__.py、README.md；旧 stage_zero 的 6 个 Python 实现文件移至 archive/stage_zero_superseded，原入口 README/DESIGN 改为历史指引。
- 测试：test_plasticity.py、test_control_plasticity.py、test_block.py、test_goodness.py、test_readin.py、test_runtime_smoke.py、test_stage_zero_abc.py、test_stage_zero_local.py。conftest.py 已审查，无旧 API，不需修改。保留历史环境/Goodness 数学审计测试，不运行历史学习器。
- 文档：README.md、docs/canonical_architecture.txt、README.github.template.md、adapter_choices.md、implementation_task.md、phase_report.md、stage_zero_plan.md。

未解决的科学问题：连续 G 差值同时受输入、动作噪声和世界变化影响；长期稳定性、固定步幅与稀疏比例适用范围未验证；overlapping delayed Goodness 尚未定义，当前只接受单 pending trial。通用 checkpoint 恢复、世界轨迹恢复不在本轮范围。

与提示词的处理说明：算法语义无有意偏离。ABC 阶段切换保留神经历史，以满足连续不可逆轨迹；不再用 frozen/initial 阶段均值初始化训练 baseline。未升级版本。附件末尾“不要 git push”由用户本条最新“完成后上传 github”覆盖，因此完成验证后上传。原实现提交未运行数小时正式训练；本次仅汇总用户提供的 200,000 step sanity 结果，未重跑训练。
