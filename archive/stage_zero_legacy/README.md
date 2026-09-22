# 历史 Stage 0 归档

此目录保留在 Local Plasticity 架构替换之前完成的 Stage 0 工作：`experiments/stage_zero/` 是旧 e-prop 实验程序，`tests/test_stage_zero_*.py` 是专属测试，`reports/stage_zero_*.md` 是预检与两轮正式结果。迁移保留了原相对目录结构和报告内容；`docs/` 另存了旧 canonical、澄清与阶段记录的逐字快照，供核对当时依据。旧实验的两轮五 seed 正式结论均为 **NOT SUPPORTED**，只适用于当时的实现、Teacher 表和预算。

这些文件不属于当前 ACNT 的生产运行或活动测试入口。旧代码依赖已删除的 `EligibilityBank`、固定反馈和 `g_bar` 等 API；若需复现历史结果，应使用报告中记录的当时 Git 提交及环境，不能直接以当前 `acnt/` 运行。此处的旧命令仅是历史记录，不是新 Stage 0 的启动说明。

被忽略的原始运行产物仍留在本机 `runs/stage_zero/`，未上传、移动或重算。当前规范见[架构原文](../../docs/canonical_architecture.txt)；尚未实施的新 Stage 0 条件见[新协议页](../../docs/stage_zero_plan.md)。
