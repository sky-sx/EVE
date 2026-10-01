"""Validate a completed joint action/evaluation life; no subject replay."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from experiments.calibrated_write_duration import stats


def mean(values):
    return sum(values)/len(values) if values else None


def metrics(rows):
    if not rows:
        return dict(count=0)
    p = [r['predicted_goodness'] for r in rows]
    g = [r['goodness'] for r in rows]
    actual_mean = mean(g)
    by_g = {str(label):[r for r in rows if r['goodness']==label] for label in (0,1)}
    pair_errors, pair_correct = [], []
    for row in rows:
        for action in (0,1):
            prediction, target = row[f'goodness_if_{action}'], int(action==row['target'])
            pair_errors.append((prediction-target)**2)
            pair_correct.append(int((prediction>=.5)==bool(target)))
    return dict(count=len(rows), mse=mean([(a-b)**2 for a,b in zip(p,g)]),
        mae=mean([abs(a-b) for a,b in zip(p,g)]),
        accuracy=mean([int((a>=.5)==bool(b)) for a,b in zip(p,g)]),
        best_constant_fit_mse=actual_mean*(1-actual_mean),
        constant_half_mse=.25, always_predict_reward_accuracy=actual_mean,
        pair_mse=mean(pair_errors), pair_accuracy=mean(pair_correct),
        by_teacher={label:dict(count=len(part),
            prediction_mean=mean([r['predicted_goodness'] for r in part]),
            mse=mean([r['prediction_squared_error'] for r in part]),
            accuracy=mean([r['prediction_correct'] for r in part])) for label,part in by_g.items()},
        cue_action_prediction_means={f'{cue}_action_{action}':mean(
            [r[f'goodness_if_{action}'] for r in rows if r['cue_class']==cue])
            for cue in ('A','B') for action in (0,1)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='runs/calibrated_write_duration/goodness_seed_55_600s_v1')
    parser.add_argument('--report', default='reports/goodness_prediction_10min_2026-10-01')
    args = parser.parse_args()
    folder, base = Path(args.input), Path(args.report)
    raw = folder/'result.json'
    result = json.loads(raw.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (folder/'observer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==result['decisions'] and result['forward_events']==4*len(rows)
    assert result['observed_wall_seconds']>=result['requested_wall_seconds']
    assert result['task']=='fixed_cue_and_goodness'
    assert result['actor_modulator']=='teacher_goodness_only'
    assert result['subject_reset_count']==result['replay_count']==result['outer_parameter_updates']==0
    assert result['total_parameters']==3839 and result['F']==148 and result['writer_parameters']==740
    assert result['goodness_adapter_parameters']==58
    previous=-1.
    for i,row in enumerate(rows,1):
        assert row['decision']==i and row['elapsed_wall_seconds']>previous
        previous=row['elapsed_wall_seconds']
        assert row['goodness']==float(row['action']==row['target'])
        assert 0<=row['predicted_goodness']<=1
        assert row['predicted_goodness']==row[f'goodness_if_{row["action"]}']
        assert abs(row['prediction_squared_error']-(row['predicted_goodness']-row['goodness'])**2)<1e-12
        assert row['prediction_correct']==int((row['predicted_goodness']>=.5)==bool(row['goodness']))
        assert all(0<=row[f'goodness_if_{a}']<=1 for a in (0,1))
    assert stats(rows[-1000:])==result['last_1000']
    for name,digest in result['source_hashes'].items():
        assert hashlib.sha256((folder/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    checkpoint=torch.load(folder/'final_checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['learner_scalars']['steps']==result['forward_events']
    assert checkpoint['learner_scalars']['pending_prediction'] is None
    assert checkpoint['learner_scalars']['teacher_for_commit'] is None
    assert checkpoint['learner_tensors']['evaluation_traces'].shape==(3,3839)
    assert all(torch.isfinite(t).all() for t in checkpoint['learner_tensors'].values())
    # Decode the saved flat mask using a static architecture; no subject steps
    # or environment actions are executed by this inspection.
    from acnt.goodness_prediction import GoodnessPredictionKernel
    assert hashlib.sha256(Path('acnt/goodness_prediction.py').read_bytes()).hexdigest()==result['source_hashes']['acnt/goodness_prediction.py']
    structure=GoodnessPredictionKernel(seed=result['seed'],width=4,hold=3,group_size=32)
    structure.load_state_dict(checkpoint['kernel_state'])
    written=checkpoint['learner_tensors']['ever_written']
    goodness_written=sum(int(written[a:b].sum()) for name,a,b,_ in structure.schema if name.startswith('goodness.'))
    assert int(written.sum())==result['written_coordinates']
    final=metrics(rows[-1000:])
    assert abs(final['mse']-result['evaluation']['last_1000']['mse'])<1e-12
    base.parent.mkdir(parents=True,exist_ok=True)
    windows=[rows[i:i+100] for i in range(0,len(rows)-99,100)]
    xs=[part[-1]['elapsed_wall_seconds']/60 for part in windows]
    wm=[metrics(part) for part in windows]
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.2))
    axes[0].plot(xs,[stats(part)['goodness'] for part in windows],label='Sampled action goodness')
    axes[0].plot(xs,[stats(part)['expected_goodness'] for part in windows],label='Correct-action probability')
    axes[0].axhline(.5,color='gray',linestyle='--')
    axes[0].set_ylim(0,1.04)
    axes[0].set_title('AB action task')
    axes[0].set_ylabel('Mean, 100-decision windows')
    axes[1].plot(xs,[m['mse'] for m in wm],label='Actual selected-action MSE')
    axes[1].plot(xs,[m['pair_mse'] for m in wm],label='Both outputs MSE (observer only)')
    axes[1].plot(xs,[m['best_constant_fit_mse'] for m in wm],label='Best constant fit in window',linestyle=':')
    axes[1].axhline(.25,color='gray',linestyle='--',label='Constant 0.5 on balanced pairs')
    axes[1].set_ylim(bottom=0)
    axes[1].set_title('Pre-teacher goodness prediction')
    axes[1].set_ylabel('Squared error (lower is better)')
    matrix=[[final['cue_action_prediction_means'][f'{cue}_action_{a}'] for a in (0,1)] for cue in ('A','B')]
    im=axes[2].imshow(matrix,vmin=0,vmax=1,cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[2].text(j,i,f'{matrix[i][j]:.3f}\nTeacher: {int(i==j)}',ha='center',va='center',
                         color='white' if matrix[i][j]>.6 else 'black')
    axes[2].set_xticks([0,1],['Action 0','Action 1'])
    axes[2].set_yticks([0,1],['Cue A','Cue B'])
    axes[2].set_title('Last 1,000: both predicted scores')
    fig.colorbar(im,ax=axes[2],fraction=.046,pad=.04)
    for axis in axes[:2]:
        axis.set_xlabel('Compute wall time (minutes)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle(f'Joint action + goodness prediction: seed {result["seed"]}, {len(rows)} decisions',fontsize=12)
    fig.tight_layout()
    fig.savefig(base.with_suffix('.png'),dpi=170)
    fig.savefig(base.with_suffix('.svg'))
    plt.close(fig)
    lines=['# AB动作与自评价双任务：十分钟实验','',
        f'一条新种子{result["seed"]}生命，实际计算{result["observed_wall_seconds"]:.2f}秒，'
        f'{len(rows)}次决策、{result["forward_events"]}次前向事件。','',
        '## 两个任务与因果顺序','',
        '动作任务为A选0、B选1。自评价任务输出g0、g1，分别判断当前输入下选0/1能否拿分。'
        '动作选定后、当前teacher G生成和交给模型之前，两路预测已经算完并记录。'
        '只用实际动作的teacher G监督对应输出；另一路只供观察器事后检验，不反馈标签。','',
        '系统goodness仍按动作是否正确提供。预测值没有替换系统G，也没有作为策略baseline、'
        'actor F目标或奖励调制量；它只进入自评价输出的teacher监督误差。'
        '辅助学习会改变共享网络未来的表现，这是双任务共同学习的实际影响。','',
        '## 结构与训练','',
        '仍为7个普通Block、28个神经元。现有普通Block 5接Goodness Adapter，'
        '结构为4→8→2，58个参数；输入只来自它的神经状态，当前动作通过选择输出通道体现。'
        '没有把输入类别、答案、teacher G或Hand概率直接塞给该Adapter。','',
        '全部3839个实际参数含Block间连接；Write为4→148、740参数。'
        '新增Goodness和扩容Write全部加入分组写入。原有Write行保留初始化，新增行独立初始化。','',
        '自评价训练使用teacher监督的BCE：负梯度为(G-p)乘反馈前捕获的所选logit信用。'
        '信用含直接参数项与在线J项，保存于三条指数衰减迹，生命内不留计算图。'
        '辅助信用仅写入Block 5和Goodness参数，由其对应正值F控制写入；'
        '既有校准训练仍更新Write等控制参数。所有有效参数均可写，没有外部optimizer。','',
        'Block 5从原动作信用分区改为评价分区，学习率为0.001；Goodness Adapter为0.01。'
        '因此这是新增任务及信用分区一起改变的候选，和此前种子44 AB版不构成单因素对照。','',
        '## 实测末段','',
        '|区间|动作G均值|所选评分MSE|两路评分MSE|两路判分正确率|',
        '|---|---:|---:|---:|---:|']
    for name,part in [('最初200次',rows[:200]),('最后200次',rows[-200:]),('最后1000次',rows[-1000:]),('全程',rows)]:
        m=metrics(part)
        lines.append(f'|{name}|{stats(part)["goodness"]:.4f}|{m["mse"]:.6f}|{m["pair_mse"]:.6f}|{m["pair_accuracy"]:.4f}|')
    lines += ['',
        f'最后1000次：所选预测MAE={final["mae"]:.6f}，判分正确率={final["accuracy"]:.4f}；'
        f'总是预测得分的分类基线为{final["always_predict_reward_accuracy"]:.4f}。',
        f'同一末段最佳常数拟合MSE={final["best_constant_fit_mse"]:.6f}，'
        '它利用了该窗口的结果均值，是事后描述基线，不是被试的前向预测。','',
        '按实际teacher结果分层：','',
        '|Teacher G|样本数|平均预测G|MSE|判分正确率|','|---|---:|---:|---:|---:|']
    for label,m in final['by_teacher'].items():
        fmt=lambda v:'无样本' if v is None else f'{v:.6f}'
        lines.append(f'|{label}|{m["count"]}|{fmt(m["prediction_mean"])}|{fmt(m["mse"])}|{fmt(m["accuracy"])}|')
    lines += ['', '当前输入的两路输出均值（观察器检验）：','',
        '|输入|选0预测|选1预测|Teacher规则|','|---|---:|---:|---|',
        f'|A|{matrix[0][0]:.6f}|{matrix[0][1]:.6f}|1 / 0|',
        f'|B|{matrix[1][0]:.6f}|{matrix[1][1]:.6f}|0 / 1|','',
        '## 连续性与限制','',
        '0次重置、重放或外部参数优化。源代码快照、行数、teacher反馈、预测通道、'
        '统计误差和最终状态均校验通过。日志与最终检查点没有提供给学习器。','',
        f'曾实际写入{result["written_coordinates"]}/{result["total_parameters"]}坐标，'
        f'Write自身{result["self_written_coordinates"]}/740；'
        f'新增Goodness参数{goodness_written}/58；'
        f'有限训练数值张量固定{result["persistent_state_and_training_tensor_bytes"]:,}字节。'
        f'最终均为有限值，J裁剪{result["tangent_clips"]}次，更新裁剪{result["update_clips"]}次。','',
        '除Core H外仍有有限J、指数迹、矩和归一化统计，以及当前尚待反馈的一项预测标量。'
        '这些不是经验日记，但需要披露，不能说全部记忆只有Core H。','',
        '十分钟是计算时间，事件时钟为单调模拟时间；仅一个固定规则、即时反馈合成任务。'
        '中间回落不作为机制失败。当前预设平台指标只评估动作任务，不能据此宣称评分头已收敛。'
        'BCE、校准目标、信用分区和部分停止导数为人为规定的训练机制；'
        '本轮没有让模型自行产生或替换goodness effect。','',
        f'动作末段平台判据满足：{result["observed_terminal_plateau"]}。', '',
        '新增四项因果顺序、系统评分隔离、指数迹和全参数可写性检查通过；'
        '全套回归253项通过（系统临时目录拒绝访问后，使用工作区新目录完成）。','',
        f'![双任务曲线]({base.name}.png)','',
        '原始日志、结果及最终训练/随机数状态在实验目录；执行源码以source_snapshot为准。',
        f'原始结果SHA256：`{hashlib.sha256(raw.read_bytes()).hexdigest()}`。','']
    base.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    summary={k:v for k,v in result.items() if k!='windows'}
    summary.update(evaluation_verified_metrics=final,validation_passed=True,
                   goodness_adapter_written_coordinates=goodness_written,
                   raw_result=str(raw),raw_result_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    base.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(validation_passed=True,last_1000_action=result['last_1000'],
        evaluation=final,report=str(base.with_suffix('.md')))))


if __name__=='__main__':
    main()
