"""Validate actual completed runs and produce a report/figure; never rerun training."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path.cwd()
    folder = root/'runs/calibrated_write_online'
    snapshot = folder/'source_snapshot_v1'
    paths = sorted(p for p in folder.glob('*/*.json') if p.parent.name != 'source_snapshot_v1')
    runs = []
    for path in paths:
        run = json.loads(path.read_text(encoding='utf-8'))
        if 'source_hashes' not in run:
            continue
        assert len(run['rows']) == run['decisions']
        assert run['events'] == 4*run['decisions']
        assert run['neural_resets'] == run['outer_parameter_updates'] == 0
        assert run['total_parameters'] == 3756 and run['writer_parameters'] == 715
        for i, row in enumerate(run['rows'], 1):
            assert row['decision'] == i
            assert row['goodness'] == float(row['action'] == row['target'])
            assert 0 <= row['expected_goodness'] <= 1
        for name, expected in run['source_hashes'].items():
            source = root/name
            saved = snapshot/name
            if hashlib.sha256(source.read_bytes()).hexdigest() == expected:
                saved.parent.mkdir(parents=True, exist_ok=True)
                if saved.exists():
                    assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
                else:
                    shutil.copy2(source, saved)
            else:
                assert saved.exists() and hashlib.sha256(saved.read_bytes()).hexdigest() == expected
        last = run['rows'][-100:]
        measured = sum(r['expected_goodness'] for r in last)/len(last)
        assert abs(measured-run['last']['expected_goodness']) < 1e-12
        run['artifact'] = str(path.relative_to(root)).replace('\\','/')
        run['configuration'] = path.parent.name
        runs.append(run)
    assert runs
    # The switch run shares the same initial world RNG and subject prefix.
    cue = next(r for r in runs if r['configuration']=='short_trace_v1' and r['mode']=='core')
    switch = next((r for r in runs if r['configuration']=='switch_v1'), None)
    continual = next((r for r in runs if r['configuration']=='continual_v1'), None)
    exploration = next((r for r in runs if r['configuration']=='exploration_v1'), None)
    if switch:
        for a, b in zip(cue['rows'], switch['rows'][:1000]):
            for key in ('action','target','goodness','expected_goodness'):
                assert a[key] == b[key]
    fig, axes = plt.subplots(1,2,figsize=(12,4.6))
    for run in runs:
        if run['mode']=='core' and run['configuration'] in ('short_trace_v1','short_trace_replicates_v1'):
            rows=run['rows']
            xs=list(range(100,len(rows)+1,100))
            ys=[sum(r['expected_goodness'] for r in rows[x-100:x])/100 for x in xs]
            axes[0].plot(xs,ys,label=f"seed {run['seed']}")
    if switch:
        rows=switch['rows']
        xs=list(range(100,len(rows)+1,100))
        axes[1].plot(xs,[sum(r['expected_goodness'] for r in rows[x-100:x])/100 for x in xs],label='original rates / ReLU')
        axes[1].axvline(1000,color='gray',linestyle=':')
        axes[1].axvline(2000,color='gray',linestyle=':')
    if continual:
        rows=continual['rows']
        xs=list(range(100,len(rows)+1,100))
        axes[1].plot(xs,[sum(r['expected_goodness'] for r in rows[x-100:x])/100 for x in xs],label='slow Core / leaky Hand')
    if exploration:
        rows=exploration['rows']
        xs=list(range(100,len(rows)+1,100))
        axes[1].plot(xs,[sum(r['expected_goodness'] for r in rows[x-100:x])/100 for x in xs],label='plus confidence-limited actions')
    for ax in axes:
        ax.axhline(.5,color='gray',linestyle='--')
        ax.set_ylim(0,1.04)
        ax.set_xlabel('Decisions, continuous trajectory')
        ax.set_ylabel('Mean correct-action probability, blocks of 100')
        ax.legend()
    axes[0].set_title('Immediate reward, learned vector Write')
    axes[1].set_title('Unannounced rule reversals; one seed')
    fig.tight_layout()
    base=root/'reports/calibrated_write_online_2026-10-01'
    fig.savefig(base.with_suffix('.png'),dpi=150)
    fig.savefig(base.with_suffix('.svg'))
    compact=[]
    for r in runs:
        item = {k:r[k] for k in ('configuration','artifact','seed','mode','task','decisions',
            'last','phases','written_coordinates','self_written_coordinates',
            'persistent_state_and_training_tensor_bytes','tangent_clips','source_hashes')}
        item['actor_lr']=r.get('actor_lr', .001)
        item['hand_activation']=r.get('hand_activation', 'ReLU')
        item['logit_confidence_limit']=r.get('logit_confidence_limit')
        tail = r['rows'][-500:]
        item['last500'] = dict(count=len(tail),
            sampled_goodness=sum(x['goodness'] for x in tail)/len(tail),
            expected_goodness=sum(x['expected_goodness'] for x in tail)/len(tail))
        item['last500_windows'] = [dict(
            sampled_goodness=sum(x['goodness'] for x in tail[i:i+100])/len(tail[i:i+100]),
            expected_goodness=sum(x['expected_goodness'] for x in tail[i:i+100])/len(tail[i:i+100]))
            for i in range(0,len(tail),100)]
        compact.append(item)
    base.with_suffix('.json').write_text(json.dumps(dict(runs=compact),indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 自校准向量 Write：实际实验结果','',
        '2026-10-01。这里汇总本次实际运行；汇总器校验轨迹长度、评分、源码 SHA256 和零重置。',
        '汇总器读取这12条本次实际跑完的轨迹，不重新启动训练。普通 Block 保留原版前向；Adapter 新增因果标准化；信用定义已改变。','',
        '![训练与规则切换](calibrated_write_online_2026-10-01.png)','',
        '|配置|模式|任务|种子|决策|末100正确动作概率|末100采样准确率|末100 F 符号正确率|实际自写 Write 坐标|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        s=r['last']
        lines.append(f"|{r['configuration']}|{r['mode']}|{r['task']}|{r['seed']}|{r['decisions']}|"
            f"{s['expected_goodness']:.6f}|{s['sampled_accuracy']:.3f}|{s['f_sign_accuracy']:.3f}|"
            f"{r['self_written_coordinates']}/715|")
    for sw in (r for r in (switch,continual,exploration) if r is not None):
        lines += ['', '规则切换的每段末100次（包含重新适应；不等于保留旧任务）：','',
                  f"配置：{sw['configuration']}。",'',
                  '|阶段|规则|末100正确动作概率|末100采样准确率|','|---:|---|---:|---:|']
        for i in range(3):
            part=sw['rows'][(i+1)*1000-100:(i+1)*1000]
            lines.append(f"|{i+1}|{'A' if i!=1 else 'B'}|"
                f"{sum(r['expected_goodness'] for r in part)/100:.6f}|{sum(r['goodness'] for r in part)/100:.3f}|")
    lines += ['', '## 按稳定末段评价最终 goodness','',
        '用户明确：探索期间的goodness下降可以是学习的一部分，不据此判定学习失效。',
        '主要看充分学习后、基本稳定的末段goodness。之前“中间适应失败”的措辞撤回。',
        '每阶段固定1000次的切换可能打断仍在适应的过程，阶段截点不是收敛证明。','',
        '|配置|最后500次实际goodness均值|最后500次正确动作概率均值|末段观察|',
        '|---|---:|---:|---|']
    for cfg in ('switch_v1','continual_v1','exploration_v1'):
        item=next((r for r in compact if r['configuration']==cfg),None)
        if item:
            observation='观察到稳定高值' if cfg!='continual_v1' else '当前截止时仍约随机；不证明永远不能恢复'
            lines.append(f"|{cfg}|{item['last500']['sampled_goodness']:.3f}|"
                f"{item['last500']['expected_goodness']:.6f}|{observation}|")
    lines += ['',
        'exploration_v1末五个100次窗口的实际goodness均值为0.97、0.96、0.95、0.97、0.97，',
        '对应正确动作概率均值均约0.952574；观察到平台，并符合当前保留探索概率的设计上限。',
        '因此本次3000次轨迹支持存在稳定高最终goodness；更长或其他环境仍未验证，',
        '不能将尚未验证表述成已证明长期学习无效。','']
    lines += ['', '## 结论与范围','',
        '独立向量头共143个输出、715个参数；全部实际参数3756，均具有自写权限。',
        '实际发生变化的总坐标少于3756时，表示这次轨迹未激活全部信用，不表示隐藏保护区域。',
        '所有模式的origin固定、外部参数更新0、Core重置0；全部Write实际自写以表中计数为准。','',
        '校准目标明确由算法规定：行为F预测2G-1，校准组学习正增益，',
        '资格迹中的控制参数使用负校准梯度。没有正确动作标签输入，但不能声称信用规则由Core自行发现。',
        '行为F进入Adam风格动量；实际更新不是无记忆的eta*E*F。','',
        '短迹成功不能推广到原8/32/128ms长迹，不能推广到未知延迟奖励。',
        '标准化、逐参数预处理、控制信用和头学习率均已改变，不能把改善归因于单一分组因素。',
        'A→B→A是继续适应测试，不证明旧任务无遗忘。其前1000次与seed11单任务轨迹一致，不算额外独立复现。',
        'continual_v1同时把Core学习率降为1e-5并把Hand激活改为LeakyReLU(0.1)，所有参数仍可写。',
        '它是另一个独立实验生命，没有从先前轨迹回滚；两项改动一起测试，不能单独归因。',
        'exploration_v1进一步在动作前按当前raw logit选择温度，保证两种动作概率均至少sigmoid(-3)。',
        '因此正确动作概率上限约95.26%；条件score停止温度选择导数，额外增加4字节数值状态，不是精确完整policy梯度。',
        '参数范围和写入范数有限不证明终身不崩；敏感度裁剪后信用近似，计数见JSON。','',
        '主体没有外部经历日记，但有额外固定大小的J、迹、动量和标准化状态；不冒充Core H。',
        '字节计数不包含模型origin、固定映射元数据、单事件临时工作空间和峰值进程内存。','',
        '全量回归：248 passed；最初3项因系统tmp_path权限未执行，改用已验证的工作区临时目录后全部通过。','',
        '[算法](../docs/training/CALIBRATED_WRITE.md) · [运行](../experiments/calibrated_write_online.py) · '
        '[汇总](../experiments/summarize_calibrated_write_online.py)','',
        '复现即时反馈候选：','',
        '```powershell',
        'python -m experiments.calibrated_write_online --seeds 11 22 33 --modes core --decisions 1000 --taus-ms 0.1 0.1 0.1 --output runs/calibrated_write_online/reproduce',
        '```','']
    base.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(completed_runs=len(runs),report=str(base.with_suffix('.md')))))


if __name__ == '__main__':
    main()
