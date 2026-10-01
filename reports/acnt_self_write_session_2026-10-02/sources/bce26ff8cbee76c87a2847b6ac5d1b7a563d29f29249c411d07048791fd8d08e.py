"""Compare group F, per-coordinate F, and broadcast G on the same eligibility."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from acnt.grouped_write import GroupedWriteKernel, GroupedWriteLearner


def run(seed=11, *, group_size=32, modulation='core', decisions=1000, lr=.001,
        write_scale=.01, task='cue', output='runs/grouped_write_online/pilot'):
    kernel = GroupedWriteKernel(seed=seed, group_size=group_size, modulation=modulation,
                               write_scale=write_scale)
    model = GroupedWriteLearner(kernel, lr=lr)
    rng = random.Random(seed+70000)
    actions = torch.Generator().manual_seed(seed+10000)
    rows, bit, started = [], 0, time.perf_counter()
    budget = model.persistent_tensor_bytes
    initial = kernel.effective(model.state).detach().clone()
    ever_written = torch.zeros(kernel.parameter_size, dtype=torch.bool)
    sources = ('acnt/grouped_write.py', 'acnt/address_write.py', 'acnt/full_write.py',
               'acnt/self_write.py', 'acnt/block.py', 'acnt/adapters.py', __file__)
    hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\', '/'):
              hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    for event in range(decisions*4):
        phase, cue, goodness, action = event%4, None, None, None
        if phase == 0:
            bit = rng.randrange(2)
            sign, amplitude = 2*bit-1, rng.uniform(.5, 1.)
            cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude,
                                          .5*sign*amplitude, -.5*sign*amplitude]])
        if phase == 3:
            target = bit if task == 'cue' else 1
            action, probability, score = model.sample(actions)
            goodness = float(action == target)
            model.record_action(score, now_ms=event*2)
            bootstrap_score = score[kernel.writer_indices].clone()
            row = dict(decision=len(rows)+1, target=target, action=action, goodness=goodness,
                       expected_goodness=probability if target else 1-probability,
                       score_norm=float(score.norm()), writer_score_norm=float(bootstrap_score.norm()))
        before = model.state[kernel.neural_size:].clone()
        model.step(now_ms=event*2, cue=cue, goodness=goodness, action=action)
        ever_written |= model.state[kernel.neural_size:] != before
        if goodness is not None:
            row['bootstrap_update_norm'] = model.learn(bootstrap_score, goodness)
            rows.append(row)
            if len(rows)%100 == 0:
                print(json.dumps(dict(seed=seed, group_size=group_size, modulation=modulation,
                    decisions=len(rows), expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    sampled_accuracy=sum(r['goodness'] for r in rows[-100:])/100,
                    meta_clips=model.meta_clips, actor_clips=model.actor_clips,
                    seconds=time.perf_counter()-started)), flush=True)
        assert budget == model.persistent_tensor_bytes
    def stats(part):
        return dict(count=len(part), expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
                    sampled_accuracy=sum(r['goodness'] for r in part)/len(part))
    result = dict(seed=seed, task=task, group_size=group_size, groups=kernel.group_count,
        modulation=modulation, decisions=decisions, events=model.steps, lr=lr, write_scale=write_scale,
        total_effective_parameters=kernel.parameter_size, writer_parameters=model.param_size,
        writer_parameters_per_control=model.param_size/kernel.group_count,
        neural_resets=0, persistent_state_and_training_tensor_bytes=budget,
        first=stats(rows[:min(100, decisions)]), last=stats(rows[-min(100, decisions):]), all=stats(rows),
        written_coordinates=int(ever_written.sum()), self_written_coordinates=int(ever_written[kernel.writer_indices].sum()),
        write_path_length=model.write_path, self_write_path_length=model.self_write_path,
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        meta_clips=model.meta_clips, actor_clips=model.actor_clips, trace_clips=model.trace_clips,
        score_clips=model.score_clips, source_hashes=hashes, seconds=time.perf_counter()-started,
        rows=rows, limitations=['hybrid actor/writer score eligibility; no claim of biological identity',
            'exponential trace rule preserved; origin/writes stopped in actor eligibility',
            'eligibility value stopped in bootstrap derivatives', 'external 81-parameter writer bootstrap remains',
            'anchored/broadcast modes explicitly use direct G; not autonomous Core-only credit',
            'immediate reward toy; no lifelong/no-forgetting validation'])
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'{task}_seed_{seed}_g{group_size}_{modulation}.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('seed', 'group_size', 'groups', 'modulation',
        'first', 'last', 'written_coordinates', 'self_written_coordinates', 'seconds')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[11])
    parser.add_argument('--sizes', nargs='+', type=int, default=[32])
    parser.add_argument('--modulations', nargs='+', choices=['core', 'anchored', 'broadcast'], default=['core'])
    parser.add_argument('--decisions', type=int, default=1000)
    parser.add_argument('--task', choices=['cue', 'constant'], default='cue')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--write-scale', type=float, default=.01)
    parser.add_argument('--output', default='runs/grouped_write_online/pilot')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for modulation in args.modulations:
        for size in args.sizes:
            for seed in args.seeds:
                run(seed, group_size=size, modulation=modulation, decisions=args.decisions,
                    lr=args.lr, write_scale=args.write_scale, task=args.task, output=args.output)
