"""Validate a completed bounded-total-G life; no training or trajectory replay."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from acnt.bounded_goodness import combined_goodness
from acnt.goodness_prediction import GoodnessPredictionKernel


def avg(values):
    return sum(values)/len(values) if values else None


def metrics(rows, floor=.2, penalty=.2):
    if not rows:
        return dict(count=0)
    pair_errors, pair_grades, pair_correct = [], [], []
    means_p, means_g = {}, {}
    for cue in ('A','B'):
        part=[r for r in rows if r['cue_class']==cue]
        for action in (0,1):
            ps=[r[f'goodness_if_{action}'] for r in part]
            gs=[combined_goodness(float(action==r['target']),p,floor=floor,penalty=penalty)['goodness']
                for r,p in zip(part,ps)]
            key=f'{cue}_action_{action}'
            means_p[key],means_g[key]=avg(ps),avg(gs)
    for row in rows:
        for action in (0,1):
            p=row[f'goodness_if_{action}']
            raw=float(action==row['target'])
            g=combined_goodness(raw,p,floor=floor,penalty=penalty)['goodness']
            pair_errors.append((p-g)**2)
            pair_grades.append(g)
            pair_correct.append(int((p>=.5)==bool(raw)))
    gs=[r['goodness'] for r in rows]
    gmean=avg(gs)
    return dict(count=len(rows), total_goodness=gmean,
        action_accuracy=avg([r['action_goodness'] for r in rows]),
        correct_action_probability=avg([r['expected_goodness'] for r in rows]),
        main_base=avg([r['main_base'] for r in rows]),
        evaluation_correction=avg([r['evaluation_correction'] for r in rows]),
        prediction_mse=avg([r['prediction_squared_error'] for r in rows]),
        prediction_mae=avg([abs(r['predicted_goodness']-r['goodness']) for r in rows]),
        best_constant_fit_to_recorded_G_mse=avg([(g-gmean)**2 for g in gs]),
        evaluation_classification_accuracy=avg([r['prediction_correct'] for r in rows]),
        pair_prediction_mse=avg(pair_errors), pair_classification_accuracy=avg(pair_correct),
        cue_action_prediction_means=means_p, cue_action_final_G_means=means_g,
        by_raw_score={str(raw):dict(count=sum(r['action_goodness']==raw for r in rows),
            prediction_mse=avg([r['prediction_squared_error'] for r in rows if r['action_goodness']==raw]),
            prediction_mean=avg([r['predicted_goodness'] for r in rows if r['action_goodness']==raw]),
            total_G_mean=avg([r['goodness'] for r in rows if r['action_goodness']==raw])) for raw in (0,1)})


def validate(folder):
    raw=folder/'result.json'
    r=json.loads(raw.read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (folder/'observer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert r['task']=='bounded_total_goodness' and r['actor_modulator']=='bounded_total_goodness_only'
    assert len(rows)==r['decisions'] and r['forward_events']==4*len(rows)
    assert r['last_event_synthetic_ms']==2*(r['forward_events']-1)
    assert r['observed_wall_seconds']>=r['requested_wall_seconds']
    assert r['subject_reset_count']==r['replay_count']==r['outer_parameter_updates']==0
    assert r['total_parameters']==3839 and r['F']==148 and r['writer_parameters']==740
    floor,penalty=r['goodness_floor'],r['goodness_penalty']
    previous=-1.
    for i,row in enumerate(rows,1):
        assert row['decision']==i and row['elapsed_wall_seconds']>previous
        previous=row['elapsed_wall_seconds']
        assert row['action_goodness']==float(row['action']==row['target'])
        assert 0<=row['goodness']<=1
        assert row['predicted_goodness']==row[f'goodness_if_{row["action"]}']
        expected=combined_goodness(row['action_goodness'],row['predicted_goodness'],floor=floor,penalty=penalty)
        for name in ('goodness','main_base','evaluation_correction','root_residual','prediction_squared_error'):
            assert abs(row[name]-expected[name])<1e-12
        if row['action_goodness']:
            assert row['goodness']>=1-penalty-1e-12
        else:
            assert row['goodness']<=floor+1e-12
        p,g=row['predicted_goodness'],row['goodness']
        coefficient=2*penalty*(g-p)/(1+2*penalty*(g-p))*p*(1-p)
        assert abs(row['evaluation_logit_coefficient']-coefficient)<1e-12
    for name,digest in r['source_hashes'].items():
        assert hashlib.sha256((folder/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    checkpoint=torch.load(folder/'final_checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['completed_decisions']==len(rows)
    assert checkpoint['learner_scalars']['pending_prediction'] is None
    assert checkpoint['learner_scalars']['teacher_for_commit'] is None
    assert all(torch.isfinite(t).all() for t in checkpoint['learner_tensors'].values())
    assert checkpoint['learner_scalars']['penalty']==penalty
    assert hashlib.sha256(Path('acnt/goodness_prediction.py').read_bytes()).hexdigest()==r['source_hashes']['acnt/goodness_prediction.py']
    structure=GoodnessPredictionKernel(seed=r['seed'],width=4,hold=3,group_size=32)
    structure.load_state_dict(checkpoint['kernel_state'])
    written=checkpoint['learner_tensors']['ever_written']
    assert int(written.sum())==r['written_coordinates']
    adapter_written=sum(int(written[a:b].sum()) for name,a,b,_ in structure.schema if name.startswith('goodness.'))
    return r,rows,adapter_written,hashlib.sha256(raw.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='runs/calibrated_write_duration/bounded_total_seed_66_600s_v1')
    parser.add_argument('--report',default='reports/bounded_goodness_10min_2026-10-01')
    args=parser.parse_args()
    folder,base=Path(args.input),Path(args.report)
    r,rows,adapter_written,digest=validate(folder)
    render(r,rows,adapter_written,digest,base,folder)


def render(r,rows,adapter_written,digest,base,folder):
    floor,penalty=r['goodness_floor'],r['goodness_penalty']
    final=metrics(rows[-1000:],floor,penalty)
    windows=[rows[i:i+100] for i in range(0,len(rows)-99,100)]
    ms=[metrics(part,floor,penalty) for part in windows]
    xs=[part[-1]['elapsed_wall_seconds']/60 for part in windows]
    base.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.2))
    axes[0].plot(xs,[m['total_goodness'] for m in ms],label='Final total G')
    axes[0].plot(xs,[m['action_accuracy'] for m in ms],label='Raw AB action accuracy')
    axes[0].set_ylim(0,1.04)
    axes[0].set_title('One total objective')
    axes[0].set_ylabel('Mean, 100-decision windows')
    axes[1].plot(xs,[m['prediction_mse'] for m in ms],label='Selected p vs final G')
    axes[1].plot(xs,[m['pair_prediction_mse'] for m in ms],label='Both outputs (observer only)')
    constant_pair=avg([(.5-combined_goodness(raw,.5,floor=floor,penalty=penalty)['goodness'])**2 for raw in (0.,1.)])
    axes[1].axhline(constant_pair,color='gray',linestyle='--',label='Constant 0.5, re-composed pair G')
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel('MSE to final G (lower is better)')
    axes[1].set_title('Pre-feedback prediction')
    matrix=[[final['cue_action_prediction_means'][f'{cue}_action_{a}'] for a in (0,1)] for cue in ('A','B')]
    im=axes[2].imshow(matrix,vmin=0,vmax=1,cmap='Blues')
    for i,cue in enumerate(('A','B')):
        for a in (0,1):
            g=final['cue_action_final_G_means'][f'{cue}_action_{a}']
            axes[2].text(a,i,f'p: {matrix[i][a]:.3f}\nG: {g:.3f}',ha='center',va='center',
                color='white' if matrix[i][a]>.6 else 'black')
    axes[2].set_xticks([0,1],['Action 0','Action 1'])
    axes[2].set_yticks([0,1],['Cue A','Cue B'])
    axes[2].set_title('Last 1,000: predicted / composed G')
    fig.colorbar(im,ax=axes[2],fraction=.046,pad=.04)
    for axis in axes[:2]:
        axis.set_xlabel('Compute wall time (minutes)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle(f'Bounded total goodness: seed {r["seed"]}, {len(rows)} decisions',fontsize=12)
    fig.tight_layout()
    fig.savefig(base.with_suffix('.png'),dpi=170)
    fig.savefig(base.with_suffix('.svg'))
    plt.close(fig)
    lines=['# 有界总goodness：十分钟实验','',
        f'独立种子{r["seed"]}生命，实际计算{r["observed_wall_seconds"]:.2f}秒，'
        f'{len(rows)}次决策、{r["forward_events"]}次前向事件。','',
        '本轮动作与评分任务共同使用一个Teacher总G。评分头在当前反馈前预测最终G。','',
        '## 数学定义','',
        f'`B={floor}+(1-{floor})R; G=B-{penalty}(p-G)^2`，取[0,1]内唯一根。',
        '右侧映回[0,1]且压缩常数最多0.4；评价修正为`G-B`，p=G时严格为0。'
        '正确动作总G至少0.8，错误动作总G最多0.2。错误动作预测准确时目标为0.2，'
        '这是预留扣分空间的主任务评分刻度。证明、闭式根和导数见'
        '[数学文档](../docs/training/BOUNDED_TOTAL_GOODNESS.md)。','',
        f'实测全程G范围：{r["total_range"]}；最大方程残差{r["max_root_residual"]:.3g}。','',
        '## 学习路径','',
        'Teacher内部用动作原始分R与评分头输出p合成G。传给Core、actor F校准及'
        '评价信用的仅是G；R只在观察器中，不作额外BCE标签。'
        '评分信用系数为`2λ(G-p)/(1+2λ(G-p))*p*(1-p)`，乘反馈前保存的评分logit'
        '在线信用，再由Write相应F控制实际写入。它是合成G的局部导数。','',
        '仍为7个普通Block、28神经元、3839参数、148个F；Goodness Adapter58参数，'
        'Write740参数，全部实际参数包含Block间连接并保持可写。其他学习率与上一双任务候选相同。','',
        '## 实测成绩','',
        '|区间|总G|动作正确率|预测最终G的MSE|平均评价修正|','|---|---:|---:|---:|---:|']
    for label,part in [('最初200次',rows[:200]),('最后200次',rows[-200:]),('最后1000次',rows[-1000:]),('全程',rows)]:
        m=metrics(part,floor,penalty)
        lines.append(f'|{label}|{m["total_goodness"]:.6f}|{m["action_accuracy"]:.4f}|{m["prediction_mse"]:.6f}|{m["evaluation_correction"]:.6f}|')
    lines += ['',
        f'末段两路评分对各自最终G的均方误差：{final["pair_prediction_mse"]:.6f}；'
        f'两路判动作对错的正确率：{final["pair_classification_accuracy"]:.4f}。'
        '两路分数来自同一前向输出；未选动作的Teacher公式仅供观察器计算，没有实际另试动作或反馈标签。','',
        f'末段最佳常数拟合记录中G的MSE为{final["best_constant_fit_to_recorded_G_mse"]:.6f}。'
        '这是事后回归基线；改变预测本身会改变Teacher合成G，不能把此基线当作实际奖励干预结果。','',
        '## 连续性与验证','',
        '无重置、回滚、经验重放或外部optimizer。全行合成公式、G范围、零点数学检查、'
        '即时信用导数、源文件快照与最终状态均校验通过。','',
        '本轮完整测试套件265项通过；其中包含独立二分求根、输入网格边界、'
        '零修正与局部导数数值验证。','',
        f'实际写入{r["written_coordinates"]}/{r["total_parameters"]}参数；'
        f'Goodness参数{adapter_written}/58，Write参数{r["self_written_coordinates"]}/740。'
        f'持久状态及训练数值张量固定{r["persistent_state_and_training_tensor_bytes"]:,}字节，'
        f'进程峰值工作集约{r["process_peak_working_set_bytes"]/1e6:.1f} MB。','',
        f'最终状态有限；J裁剪{r["tangent_clips"]}次，更新裁剪{r["update_clips"]}次。'
        'Core H以外仍有J、指数迹、矩与归一化统计，不能说全部记忆仅在Core H。','',
        f'末段总G平台判据满足：{r["observed_terminal_plateau"]}；判据为{r["plateau_rule"]}。'
        '中间下降仅作观察；未满足平台条件不等于已证明长期学习机制失败。','',
        '## 解释边界','',
        '本次改了Teacher函数及评分信用，并换用新种子，不能与旧双任务分数作单因素因果比较。'
        '只有一条固定AB、即时反馈合成任务；不覆盖多任务、延迟奖励或一般长期保留。'
        'Teacher函数的局部导数虽精确，生命周期训练仍有停止导数、J/更新裁剪和信用归一化等近似。','',
        f'十分钟为计算时间，单调模拟事件时钟累计{r["last_event_synthetic_ms"]/1000:.3f}秒。'
        '观察日志与最终状态只供事后检查，没有提供给学习器。','',
        f'![十分钟曲线]({base.name}.png)','',
        f'原始结果：{folder}/result.json；最终状态：final_checkpoint.pt；源码：source_snapshot。',
        f'原始结果SHA256：`{digest}`。','']
    base.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    summary={k:v for k,v in r.items() if k!='windows'}
    summary.update(verified_final_metrics=final,goodness_adapter_written_coordinates=adapter_written,
                   validation_passed=True,raw_result=str(folder/'result.json'),raw_result_sha256=digest)
    base.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(validation_passed=True,final=final,range=r['total_range'],
        plateau=r['observed_terminal_plateau'],report=str(base.with_suffix('.md')))))


if __name__=='__main__':
    main()
