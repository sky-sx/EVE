"""Report the defined grouped-write cohort, not a biological/long-term proof."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    base = root/'runs/grouped_write_online'
    paths = [base/folder/name for folder, name in (
        ('group_controls', 'cue_seed_11_g1_core.json'),
        ('group_controls', 'cue_seed_11_g8_core.json'),
        ('core_g32', 'cue_seed_11_g32_core.json'),
        ('core_g32', 'cue_seed_22_g32_core.json'),
        ('core_g32', 'cue_seed_33_g32_core.json'),
        ('independent_capacity', 'cue_seed_11_g32_independent.json'),
        ('reward_controls', 'cue_seed_11_g32_anchored.json'),
        ('reward_controls', 'cue_seed_11_g32_broadcast.json'),
        ('rms_diagnostic', 'cue_seed_11_g32_broadcast_rms.json'),
        ('short_trace_diagnostic', 'cue_seed_11_g32_broadcast_rms_short_trace.json'))]
    runs = []
    for path in paths:
        run = json.loads(path.read_text(encoding='utf-8'))
        for name, digest in run['source_hashes'].items():
            candidates = [root/name, base/'source_snapshot'/name, base/'source_snapshot_latest'/name]
            if not any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in candidates):
                raise ValueError(f'source mismatch: {name} in {path}')
        run['raw_path'] = path.relative_to(root).as_posix()
        run['generator'] = run.get('generator', 'shared address generator')
        run['actuator'] = run.get('actuator', 'raw')
        run['taus_ms'] = run.get('taus_ms', [8., 32., 128.])
        run['feedback_per_writer_parameter'] = run['decisions']/run['writer_parameters']
        runs.append(run)
    destination = root/'reports/grouped_write_online_2026-10-01'
    compact = [{k: v for k, v in run.items() if k != 'rows'} for run in runs]
    destination.with_suffix('.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for run in runs:
        core = run['modulation'] == 'core'
        axis = axes[0 if core else 1]
        if core:
            label = f"g={run['group_size']}, seed {run['seed']}"
            if run['generator'] != 'shared address generator':
                label = 'independent heads, g=32, seed 11'
        else:
            label = run['modulation']
            if run['actuator'] != 'raw':
                label += ', RMS'
            if run['taus_ms'] == [.1, .1, .1]:
                label += ', short trace'
        values = [r['expected_goodness'] for r in run['rows']]
        x = list(range(100, len(values)+1, 100))
        y = [sum(values[i-100:i])/100 for i in x]
        axis.plot(x, y, marker='o', markersize=3, linewidth=1.5, label=label)
    for axis, title in zip(axes, ('Core-generated group F', 'Controls with explicit reward signal')):
        axis.axhline(.5, color='gray', linestyle=':', linewidth=1)
        axis.set(title=title, xlabel='Continuous decisions', ylim=(.0, 1.05))
        axis.grid(alpha=.15)
        axis.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('Expected correctness, non-overlapping 100 decisions')
    figure.suptitle('Grouped write bootstrap: capacity and eligibility diagnostics')
    figure.tight_layout()
    figure.savefig(destination.with_suffix('.png'), dpi=160)
    figure.savefig(destination.with_suffix('.svg'))
    plt.close(figure)

    def config(run):
        if run['generator'] != 'shared address generator':
            return '独立输出头 / Core F / g32'
        if run['modulation'] == 'core':
            return f"共享头 / Core F / g{run['group_size']}"
        label = run['modulation']
        if run['actuator'] != 'raw':
            label += ' / RMS'
        if run['taus_ms'] == [.1, .1, .1]:
            label += ' / 极短迹'
        return label

    lines = ['# E × 分组 F：首轮结果', '',
        '2026-10-01。十条独立单向轨迹，每条 1,000 次决策、4,000 个事件。',
        '实际执行了训练，主体不重置、不反事实重放、不读取观察者日志。', '',
        '**结论：全参数分组执行和自身写入路径成立；本轮 Core 生成的 F 尚未学会条件任务。**', '',
        '先去掉 Hand 中未用于二元任务的 774 个参数，保留原已用输出的精确初始映射。',
        '共享版总参数 3,122、Write 81；每组最多 32 个参数时只有 125 个 F。',
        '独立输出头的自身参数也纳入预算：总参数 3,756、Write 715、143 个 F。', '',
        '|配置|种子|Write参数|F数|末100期望正确率|末100采样正确率|自身实际写过/可写|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        lines.append(f"|{config(run)}|{run['seed']}|{run['writer_parameters']}|{run['groups']}|"
            f"{run['last']['expected_goodness']:.4f}|{run['last']['sampled_accuracy']:.4f}|"
            f"{run['self_written_coordinates']}/{run['writer_parameters']}|")
    lines += ['', '![实验曲线](grouped_write_online_2026-10-01.png)', '',
        '## 容量假说能判断到哪里', '',
        '把查询从 3,122 个减少到 392/125 个，未出现条件任务学习收益；增加为每组',
        '独立系数的 715 参数输出头，同样未解决本轮任务。权限测试覆盖所有实际坐标，',
        '包含所有 Write 坐标，实际轨迹中是否修改某个坐标还取决于其 E 是否非零。',
        '不能把“可写”混同为“每个参数都应该在每轮修改”。独立头实际自写 564/715，',
        '不声称轨迹中每个自身参数都已经获得有效信用。', '',
        '这些结果不排除容量不足：共享头仍是共同函数，独立头只测一个种子、',
        '且每参数反馈量从约 12.35 降为约 1.40，并未匹配训练自由度后的反馈预算。',
        '反馈/参数比是预算描述，不是判断收敛的定理。当前只有学习机制探测，',
        '没有严格的容量/数据量因果结论。', '',
        '## 为什么暂不判定分组方向失败', '',
        '统一 G 调制对照也未学会条件任务；RMS 和极短迹诊断有一些改善，仍未',
        '达到稳定可靠的条件动作。其成功或改善也不能归因于 Core 自行分配信用。',
        '因此需要先校准 E 的有效方向、时间混合、写入幅度和反馈信用，再比较',
        '控制器容量。不能单凭这一训练组合的负结果否定生理类比或所有分组方案。', '',
        'RMS 保留原 E 递推，但更改幅度转换及 nominal eta；极短迹另改时间常数。',
        '这两个是独立诊断，不属于只改分组的干净比较。', '',
        '## 机制和局限', '',
        '同组共享 F；最终增减仍由有符号 E 决定，F 是沿 E/反 E 的区域调制。',
        'E 的指数递推保留；actor score 与包含写入路径的 writer score 的结合是',
        '本轮混合近似，不称为已证正确的全系统资格迹或钙浓度模型。',
        '外部 Write bootstrap 仍存在；它不直接更新行为参数。Core F 不读直接 G 旁路，',
        'anchored/broadcast 则明确读取 G；不能把后两者结果称为自主分析 G。', '',
        '稳定性依赖明确的参数/更新/迹/敏感度限制；无非有限数不证明长期稳定。',
        '神经活动量、延迟反馈、任务切换、长期遗忘和现实身体后果尚未验证。', '',
        '所有候选包含全参数分组映射，没有永久只读的可训练控制网络。',
        '主体只有当前 Core 状态、参数位移和固定规模训练数值状态；',
        '没有动作/反馈历史槽、经验回放或外部经历日记。', '',
        '## 验证与资源', '',
        '全仓 240 项测试通过。新增验证包括同组 F、全部坐标权限、保持原动作映射、',
        '真实 self-write 信用、资格迹递推、固定权重 actor 梯度、条件自写导数、',
        'RMS 保持原始 E、独立输出头自覆盖固定点、固定资源等。', '',
        '共享主体/训练张量固定为 2,510,008 bytes；RMS 为 2,534,984；独立头为',
        '12,826,456。另有模型、地址/索引和单事件工作空间，未测峰值进程内存。',
        'J/T 是额外数值状态，不冒充普通 Core H。', '',
        '[机制说明](../docs/training/GROUPED_WRITE_BOOTSTRAP.md) · '
        '[汇总数据](grouped_write_online_2026-10-01.json)', '', '## 原始数据', '']
    lines += ['以下为本地完整记录路径，未上传逐决策日志。公开精简记录与源码版本见',
              '[本对话证据包](acnt_self_write_session_2026-10-02/manifest.json)。', '']
    lines += [f"- `{run['raw_path']}`" for run in runs]
    destination.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps([dict(configuration=config(run), seed=run['seed'], last=run['last'],
        controls=run['groups'], writer_parameters=run['writer_parameters']) for run in runs], indent=2))


if __name__ == '__main__':
    main()
