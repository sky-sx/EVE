"""Aggregate the defined first-round address-write runs and verify their sources."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [root/'runs/address_write_online'/folder/name for folder, name in (
        ('constant_v1', 'constant_seed_11_full.json'),
        ('constant_control', 'constant_seed_11_no_write.json'),
        ('cue_v1', 'cue_seed_11_full.json'),
        ('cue_v1', 'cue_seed_22_full.json'),
        ('cue_v1', 'cue_seed_33_full.json'),
        ('cue_small_step', 'cue_seed_11_full.json'),
        ('ablations_v1', 'cue_seed_11_no_write.json'),
        ('ablations_v1', 'cue_seed_11_cut_write.json'))]
    runs = []
    for path in paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        for name, digest in result['source_hashes'].items():
            actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
            if actual != digest:
                snapshot = root/'runs/address_write_online/source_snapshot'/name
                if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'source drift for {name} in {path}')
        result['raw_path'] = path.relative_to(root).as_posix()
        runs.append(result)
    destination = root/'reports/address_write_online_2026-10-01'
    compact = [{k: v for k, v in run.items() if k != 'rows'} for run in runs]
    destination.with_suffix('.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for run in runs:
        axis = axes[0 if run['task'] == 'constant' else 1]
        label = f"{run['mode']}, seed {run['seed']}"
        if run['write_scale'] != .002:
            label += ', 10x smaller write'
        values = [r['expected_goodness'] for r in run['rows']]
        x = list(range(100, len(values)+1, 100))
        y = [sum(values[i-100:i])/100 for i in x]
        axis.plot(x, y, marker='o', markersize=3, label=label,
                  linestyle='--' if run['mode'] != 'full' else '-')
    for axis, title in zip(axes, ('Constant action', 'Cue-conditioned action')):
        axis.axhline(.5, color='gray', linewidth=1, linestyle=':')
        axis.set(title=title, xlabel='Continuous decisions', ylim=(0., 1.05))
        axis.legend(fontsize=8, loc='lower right')
        axis.grid(alpha=.15)
    axes[0].set_ylabel('Expected correctness, non-overlapping 100 decisions')
    figure.suptitle('Full-parameter address self-write: bootstrap test, not lifelong validation')
    figure.tight_layout()
    figure.savefig(destination.with_suffix('.png'), dpi=160)
    figure.savefig(destination.with_suffix('.svg'))
    plt.close(figure)

    lines = ['# 全参数地址自写：首轮结果', '',
        '2026-10-01。执行了实际前向与在线启动训练，未清空状态、重放经历或回滚。',
        '每条轨迹 1,000 次决策、4,000 个同步事件；即时二元 G。', '',
        '**结论：地址执行与自身参数覆盖通过；常量任务学会；条件信号任务尚未学会。**', '',
        '模型实际参数 3,896 个，每次写入覆盖 3,896 个地址，Write 自身 81 个参数也可写。',
        '外部启动只更新 Write 坐标，不能直接训练行为参数；它仍是手写训练规则。', '',
        '|任务/配置|种子|末100期望正确率|末100采样正确率|信用裁剪次数|实际写过坐标|',
        '|---|---:|---:|---:|---:|---:|']
    for run in runs:
        name = f"{run['task']} / {run['mode']}"
        if run['write_scale'] != .002:
            name += ' / 写入幅度÷10'
        lines.append(f"|{name}|{run['seed']}|{run['last']['expected_goodness']:.4f}|"
                     f"{run['last']['sampled_accuracy']:.4f}|{run['meta_clips']}|{run['written_coordinates']}|")
    lines += ['', '![首轮曲线](address_write_online_2026-10-01.png)', '',
        '## 可据此判断的内容', '',
        '所有 full 轨迹都实际写过全部 3,896 个坐标，包括全部 81 个自身 Write 坐标。',
        '这不是只写某个 Core 区域，也不是用注册参数测试替代实际权重。',
        '关闭写入和切断写入信用时，零初始化 Write 得不到启动梯度，二者产生相同轨迹；',
        '不能把它们当成两个独立随机重复实验。常量成功不能证明条件信用分配已解决。', '',
        '常量 full 轨迹的实际参数总变化范数为 250.38，伴随大幅权重漂移。',
        '任务达到恒定正确动作，不足以说明网络学会了精细而可持续的参数修改。', '',
        '条件任务三个种子未显示持续高于随机的学习收益。降低写入幅度十倍未恢复学习，',
        '因此本轮失败不能仅归因于数值裁剪或幅度太大；控制器信息、共享生成器的',
        '地址表达能力以及外部信用规则均尚未分离诊断。未给这些原因下定论。', '',
        '参数裁剪为 [-4,4]，信用范数裁剪为 1,000；无 NaN 不代表无内在信用放大。',
        '权重长期有界来自显式裁剪，不是自主稳定性证明。还没有任务切换/遗忘长程验证。', '',
        '## 后续方向', '',
        '保留地址执行器作为全参数自覆盖工具，本轮地址生成训练组合暂不视为通过。',
        '下一候选可比较 E × 分组方向 F；先绑定实际 E 并测组内方向相容性，',
        '保持同一任务、真实单向轨迹及全参数（包含 Write 自身）覆盖约束。',
        '本轮结果既没有否定所有地址方案，也没有验证分组方向方案。', '',
        '## 验证与资源', '',
        '全仓 229 项测试通过。针对性六项验证了原版前向一致性、所有实际参数写入、',
        '含自身写入的四事件导数与完整展开一致、信用切断、固定资源和单调时间。',
        '完整展开是独立测试 oracle，不在在线主体内使用。', '',
        '联合状态、敏感度与优化矩固定为 1,315,272 bytes；地址元数据另占 46,752 bytes。',
        '另有模型 origin 参数、固定索引/缓冲区和单事件工作空间；未测峰值进程内存。',
        '主体不读取观察者的标签、概率正确率或 JSON 日志，没有经历日记。', '',
        '[机制说明](../docs/training/ADDRESS_WRITE_BOOTSTRAP.md) · '
        '[汇总数据](address_write_online_2026-10-01.json)', '',
        '## 原始记录', '']
    lines += ['以下为本地完整记录路径，未上传逐决策日志。公开精简记录与源码版本见',
              '[本对话证据包](acnt_self_write_session_2026-10-02/manifest.json)。', '']
    lines += [f"- `{run['raw_path']}`" for run in runs]
    destination.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
