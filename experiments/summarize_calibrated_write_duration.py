"""Validate and plot a completed wall-clock life; never rerun the subject."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from experiments.calibrated_write_duration import stats, class_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='runs/calibrated_write_duration/fixed_cue_seed_44_600s_v1')
    parser.add_argument('--report', default='reports/calibrated_write_10min_2026-10-01')
    args = parser.parse_args()
    folder, base = Path(args.input), Path(args.report)
    result_path = folder/'result.json'
    r = json.loads(result_path.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (folder/'observer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == r['decisions']
    assert r['forward_events'] == 4*len(rows)
    assert r['last_event_synthetic_ms'] == 2*(r['forward_events']-1)
    assert r['observed_wall_seconds'] >= r['requested_wall_seconds']
    assert r['subject_reset_count'] == r['replay_count'] == r['outer_parameter_updates'] == 0
    assert r['total_parameters'] == 3756 and r['writer_parameters'] == 715 and r['F'] == 143
    previous = -1.
    for i, row in enumerate(rows, 1):
        assert row['decision'] == i and row['cue_class'] == ('B' if row['target'] else 'A')
        assert row['goodness'] == float(row['action'] == row['target'])
        assert .0474 <= row['probability_action_1'] <= .9526
        expected = row['probability_action_1'] if row['target'] else 1-row['probability_action_1']
        assert abs(expected-row['expected_goodness']) < 1e-12
        assert row['elapsed_wall_seconds'] > previous
        previous = row['elapsed_wall_seconds']
    assert stats(rows[-1000:]) == r['last_1000']
    assert class_stats(rows[-1000:]) == r['last_1000_classes']
    assert stats(rows[:200]) == r['first_200']
    current_matches = {}
    for name, digest in r['source_hashes'].items():
        saved = folder/'source_snapshot'/name
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
        current = Path(name)
        current_matches[name] = current.exists() and hashlib.sha256(current.read_bytes()).hexdigest() == digest
    checkpoint = torch.load(folder/'final_checkpoint.pt', map_location='cpu', weights_only=True)
    assert checkpoint['completed_decisions'] == len(rows)
    assert checkpoint['learner_scalars']['steps'] == r['forward_events']
    for tensor in checkpoint['learner_tensors'].values():
        assert torch.isfinite(tensor).all()
    assert int(checkpoint['learner_tensors']['ever_written'].sum()) == r['written_coordinates']
    base.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # A final partial block has fewer samples; do not present it as a full
    # 100-decision window (e.g. 19 correct samples would create a false 1.0 spike).
    windows = [w for w in r['windows'] if w['count'] == 100]
    axes[0].plot([w['elapsed_wall_seconds']/60 for w in windows],
        [w['goodness'] for w in windows], label='Sampled goodness', color='#1976d2')
    axes[0].plot([w['elapsed_wall_seconds']/60 for w in windows],
        [w['expected_goodness'] for w in windows], label='Correct-action probability', color='#ef6c00')
    axes[0].axhline(.5, color='gray', linestyle='--', label='Constant-action baseline')
    axes[0].axhline(.95257413, color='gray', linestyle=':', label='Exploration design ceiling')
    axes[0].set_xlabel('Compute wall time (minutes)')
    axes[0].set_ylabel('Mean, 100-decision windows')
    axes[0].set_title(f'One fixed-cue life: seed {r["seed"]}, {r["decisions"]} decisions')
    axes[0].legend(fontsize=8)
    labels = ['A -> 0', 'B -> 1']
    for j, key in enumerate(('goodness', 'expected_goodness')):
        axes[1].bar([i+(-.18 if j==0 else .18) for i in range(2)],
            [r['last_1000_classes'][name][key] for name in ('A','B')],
            width=.36, label='Sampled accuracy' if j==0 else 'Correct-action probability')
    axes[1].set_xticks([0,1], labels)
    axes[1].set_title('Last 1,000 decisions, by cue class')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_ylim(0, 1.04)
        axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(base.with_suffix('.png'), dpi=170)
    fig.savefig(base.with_suffix('.svg'))
    plt.close(fig)
    last = r['last_1000']
    lines = ['# 十分钟固定规则在线学习实验', '',
        f'新种子{r["seed"]}的一条连续生命，实际计算{r["observed_wall_seconds"]:.2f}秒，'
        f'{r["decisions"]}次决策、{r["forward_events"]}次前向事件。', '',
        f'最后1000次实际goodness均值 **{last["goodness"]:.3f}**，'
        f'正确动作概率均值 **{last["expected_goodness"]:.6f}**。', '',
        f'末段五个200次窗口是否满足预设平台观察条件：**{r["observed_terminal_plateau"]}**。'
        '这是有限窗口观察，不代表数学收敛。', '',
        '## 任务与算法', '',
        '两类合成信号，幅度随机，A选择0、B选择1，各以约一半概率独立出现。'
        '输入在第一次前向事件到达，第四次事件先采样动作再接收标量G；'
        '规则、类别编号和正确动作只在环境/观察器中使用，模型接收信号、自身动作和G。', '',
        '7个普通Block，每个4个神经元。全部3756个实际参数包括Block之间的连接；'
        'Write有715参数、143个独立F。保留慢Core、Leaky Hand和置信度限制探索版，'
        '没有在生命中更换算法或超参数。探索使正确动作概率最多约0.9526。', '',
        '## 实测成绩', '',
        '|区间|次数|实际G均值|正确动作概率均值|',
        '|---|---:|---:|---:|']
    for name, item in [('最初200次',r['first_200']), ('最后200次',r['last_200']),
                       ('最后1000次',last), ('全程',r['all']),
                       ('末段A',r['last_1000_classes']['A']),('末段B',r['last_1000_classes']['B'])]:
        lines.append(f'|{name}|{item["count"]}|{item["goodness"]:.4f}|{item["expected_goodness"]:.6f}|')
    lines += ['', '最后五个200次窗口：', '',
        '|窗口|实际G均值|正确动作概率均值|', '|---|---:|---:|']
    for i, item in enumerate(r['terminal_200_decision_windows'], 1):
        lines.append(f'|{i}|{item["goodness"]:.4f}|{item["expected_goodness"]:.6f}|')
    lines += ['', '## 连续性与数值状态', '',
        '生命内0次重置、0次经验重放、0次外部参数优化。源文件快照哈希、动作反馈、'
        '行数、模拟时钟与最终检查点均校验通过。日志及检查点仅供观察器事后使用。', '',
        f'实际发生过修改的参数坐标：{r["written_coordinates"]}/{r["total_parameters"]}；'
        f'其中Write自身为{r["self_written_coordinates"]}/715。写入权限覆盖全部有效参数；'
        '本次未激活/零eligibility坐标未必产生实际修改。', '',
        f'额外状态及训练数值张量固定为{r["persistent_state_and_training_tensor_bytes"]:,}字节，'
        f'进程峰值工作集约{r["process_peak_working_set_bytes"]/1e6:.1f} MB。'
        f'J裁剪{r["tangent_clips"]}次，更新范数裁剪{r["update_clips"]}次。'
        '有J、迹、矩和归一化统计等额外数值状态，不能说全部记忆都位于Core H。', '',
        f'最终状态检查为有限值；每100次及末尾抽样观察的最大值：{r["sampled_state_maxima"]}。'
        '这些是抽样最大值，不能当作逐事件最大值。', '',
        '## 解释边界', '',
        f'十分钟为实际计算时间；单调模拟时钟最后到{r["last_event_synthetic_ms"]/1000:.3f}秒，'
        '没有按真实世界时间等待或接入麦克风。本次支持的范围是一个固定、即时反馈的'
        '合成条件选择任务；不覆盖规则反转、多任务、长期保留和匿名延迟奖励。', '',
        '中间G下降作为探索过程记录，主要看稳定后的末段。规定的校准损失、条件温度'
        '求导及停止写入/统计导数等近似仍在，不能据此宣布一般长期学习已解决。', '',
        '## 复现及产物', '',
        '```powershell',
        f'python -m experiments.calibrated_write_duration --seconds 600 --seed {r["seed"]} '
        '--width 4 --threads 1 --output runs/calibrated_write_duration/new_life', '```', '',
        '墙钟截点依赖本机负载，重跑的总决策数可以不同。原始轨迹为observer.jsonl，'
        '最终训练状态和随机数状态为final_checkpoint.pt，源码快照为source_snapshot。', '',
        f'原始结果SHA256：`{hashlib.sha256(result_path.read_bytes()).hexdigest()}`。', '',
        f'![学习曲线]({base.name}.png)', '']
    base.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    summary = {k:v for k,v in r.items() if k!='windows'}
    summary.update(raw_result=str(result_path), raw_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        validation_passed=True, current_source_hash_matches=current_matches)
    base.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(validation_passed=True, last_1000=last,
        classes=r['last_1000_classes'], plateau=r['observed_terminal_plateau'], report=str(base.with_suffix('.md')))))


if __name__ == '__main__':
    main()
