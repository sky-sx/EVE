# ACNT 训练研究入口

2026-10-02 更新。当前实验主线为原版前向上的在线全参数自修改，最新版本把 AB 动作和自评价合成单一有界总 goodness。

- [本对话实验总报告](../../reports/acnt_self_write_session_2026-10-02.md)：全部阶段、三次十分钟运行、成绩定义、边界与复现。
- [公开精简证据与源码版本](../../reports/acnt_self_write_session_2026-10-02/manifest.json)：56 条结果记录、执行源码按 SHA256 索引；完整轨迹和权重留在本地。
- [当前写入与信用算法](CALIBRATED_WRITE.md)、[有界总 goodness 数学](BOUNDED_TOTAL_GOODNESS.md)。
- [向量输出头自覆盖预算](WRITE_VECTOR_BUDGET.md)、[分组方向假设](GROUPED_DIRECTION_WRITE.md)、[本机长期轨迹计划](LOCAL_LONG_TRAJECTORY_PLAN.md)。
- [原版在线训练](ORIGINAL_ACNT_ONLINE.md)、[局部自修改启动](SELF_WRITE_BOOTSTRAP.md)、[地址写入](ADDRESS_WRITE_BOOTSTRAP.md)、[首轮分组写入](GROUPED_WRITE_BOOTSTRAP.md)：保留各版结果，不混同为当前候选。
- [短窗 BPTT 基线](ORIGINAL_ACNT_TRAINING.md)：已有短时任务训练前身，独立于连续自修改主线。
- [G0–G3 原交接](EVENT_FLOW_HANDOFF.md)、[集成及本地 smoke](INTEGRATION.md)：独立 event-flow 参考线。

完整回归 265 项通过。连续轨迹内无重置/重放，全部有效参数可写；Core H 外还存在固定规模训练数值状态。最新十分钟末 1,000 次总 G=0.845592、动作正确率81.1%、预测最终G的MSE=0.016041；尚未满足末段平台条件，不把截止时成绩当作稳定最终成绩。

```powershell
python -m pip install -e ".[test,reports]"
python -m experiments.summarize_session --verify-published
python -m experiments.calibrated_write_duration --seconds 600 --seed 66 --width 4 --threads 1 --bounded-total --output runs/new_bounded_life
```

以下保留原交接内容，日期及“current/canonical/next”用语只对应当时的 G0–G3 参考线；其研究建议不是本轮新增运行指令，不能覆盖上面的最新实测报告。

---

# Historical ACNT Training Handoff — G0/G1/G2 + G3 Frontier

**Date:** 2026-10-01
**Purpose:** executable handoff package for continuing ACNT training research in Codex.

## Important provenance note

The exploratory runs in the ChatGPT conversation were executed in transient analysis sessions and were **not originally persisted as a repository/raw-log directory**. This package therefore contains:

1. **reconstructed, runnable reference implementations** of the mechanisms developed in the conversation;
2. **fresh smoke/regression tests** that are run again before packaging;
3. a verbatim-style summary of the **numbers reported in the chat**, explicitly marked as *reported, not raw persisted logs*.

Do not treat `reports/training_handoff_2026-10-01/reported_chat_results.md` as raw experimental evidence. Re-run the scripts and have Codex produce raw logs for any formal claim.

## Current research state

- **G0: PASS** — local mathematical / derivative correctness.
- **G1: PASS at POC level** — block-local exact eligibility plus streaming low-rank cross-Block influence can restore cross-Block credit without a BPTT tape/full persistent RTRL Jacobian.
- **G2: PASS at POC level** — mixed histories and real physical time work after replacing the old time gate with **semantic-event jump + exponential continuous flow**.
- **G3: PARTIAL** — asynchronous eager/lazy runtime semantics are mathematically consistent; robust developmental bootstrap across random births remains unresolved.

## Canonical post-G2 time semantics

Pure physical time passage:

```math
z(t+\Delta t)=\bar z+\exp(-\lambda\Delta t)\odot(z(t)-\bar z)
```

A scheduler tick is **not** a semantic event. During pure time passage:

```text
H unchanged
target unchanged
lambda unchanged
```

Only a semantic event (ReadIn, Block message event, action/internal event under the chosen runtime definition) may:

1. materialize `z` at the event time;
2. shift history once;
3. perform an event jump;
4. recompute target and lambda.

Allowed function family is therefore:

```text
linear, tanh, sigmoid, exp_time
```

`exp_time` is not a general activation; it is reserved for physical-time flow.

## Canonical credit state

```math
\hat J_t = E^{local}_t + U_t V_t^T
```

- `E_local`: exact Block-local online eligibility / forward sensitivity.
- `U,V`: streaming low-rank cross-Block influence.
- `rank` is a capacity parameter, **not a fixed architectural constant**.
- pure time passage propagates `U` through the time-flow Jacobian but must **not** inject/recompress stochastic cross credit.
- semantic events may inject new local→cross influence and trigger stochastic low-rank merge.

Policy/action score then gives an online parameter-direction trace:

```math
q_t = \nabla_H \log \pi(a_t\mid H_t)\; \hat J_t
```

Delayed Goodness can be carried by one or more real-time traces:

```math
Q_k(t+\Delta t)=e^{-\Delta t/\tau_k}Q_k(t)+q_t
```

and the slow plasticity update is:

```math
\dot\theta = \eta (G-\bar G) \sum_k \alpha_k Q_k
```

`G` remains the **only final evaluation scalar**. `\bar G` is only a baseline statistic.

## Run

```bash
python -m pip install -e ".[test]"
python -m experiments.training.run_all
python -m pytest -q
```

## Files

- `ALGORITHM_SPEC.md` — current algorithm and invariants.
- `STATUS_AND_GATES.md` — G0/G1/G2/G3 evidence/status.
- `CODEX_HANDOFF.md` — implementation continuation checklist.
- `acnt/event_flow.py` — current event/continuous-time Block.
- `acnt/rtrl.py` — exact oracle and streaming low-rank reference.
- `experiments/training/` — G0–G3 smoke/regression programs.
- `reports/training_handoff_2026-10-01/reported_chat_results.md` — chat-reported numbers, **not raw logs**.

## What Codex should do next

Do **not** invent a new training theory first. The next target is **G3 developmental robustness**:

1. implement production/manual local Jacobian/JVP kernels (do not depend on global autograd graphs);
2. preserve the event-vs-time semantics exactly;
3. sweep a task-agnostic neutral birth initialization;
4. measure distant signal/credit transmissibility at birth;
5. compare exact-RTRL oracle vs low-rank cross credit;
6. keep Freeze-Core and no-cross ablations;
7. persist all raw seeds/configs/curves.

See [INTEGRATION.md](INTEGRATION.md) for current local verification and runtime migration boundaries.
