"""Amplitude diagnostic with the same actor and eligibility as grouped v1."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from acnt.grouped_write_rms import RMSGroupedWriteKernel, RMSGroupedWriteLearner


def run(seed=11, *, modulation='broadcast', group_size=32, decisions=1000,
        output='runs/grouped_write_online/rms_diagnostic'):
    kernel = RMSGroupedWriteKernel(seed=seed, group_size=group_size,
                                  modulation=modulation, write_scale=.001)
    model = RMSGroupedWriteLearner(kernel)
    rng = random.Random(seed+70000)
    actions = torch.Generator().manual_seed(seed+10000)
    rows, bit, started = [], 0, time.perf_counter()
    initial = kernel.effective(model.state).detach().clone()
    budget = model.persistent_tensor_bytes
    ever = torch.zeros(kernel.parameter_size, dtype=torch.bool)
    sources = ('acnt/grouped_write_rms.py', 'acnt/grouped_write.py', 'acnt/address_write.py',
               'acnt/full_write.py', 'acnt/self_write.py', 'acnt/block.py', 'acnt/adapters.py', __file__)
    hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\', '/'):
              hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    for event in range(decisions*4):
        cue, goodness, action = None, None, None
        if event%4 == 0:
            bit = rng.randrange(2)
            sign, amplitude = 2*bit-1, rng.uniform(.5, 1.)
            cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude,
                                          .5*sign*amplitude, -.5*sign*amplitude]])
        if event%4 == 3:
            action, probability, score = model.sample(actions)
            goodness = float(action == bit)
            model.record_action(score, now_ms=event*2)
            bootstrap = score[kernel.writer_indices].clone()
            row = dict(decision=len(rows)+1, action=action, target=bit, goodness=goodness,
                       expected_goodness=probability if bit else 1-probability)
        before = model.state[kernel.neural_size:].clone()
        model.step(now_ms=event*2, cue=cue, goodness=goodness, action=action)
        ever |= model.state[kernel.neural_size:] != before
        if goodness is not None:
            row['bootstrap_update_norm'] = model.learn(bootstrap, goodness)
            rows.append(row)
            if len(rows)%100 == 0:
                print(json.dumps(dict(seed=seed, modulation=modulation, decisions=len(rows),
                    expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    meta_clips=model.meta_clips, actor_clips=model.actor_clips,
                    seconds=time.perf_counter()-started)), flush=True)
        assert budget == model.persistent_tensor_bytes
    def stats(part):
        return dict(count=len(part), expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
                    sampled_accuracy=sum(r['goodness'] for r in part)/len(part))
    result = dict(seed=seed, task='cue', group_size=group_size, groups=kernel.group_count,
        modulation=modulation, actuator='detached RMS', write_scale=.001, lr=.001,
        decisions=decisions, events=model.steps, neural_resets=0,
        total_effective_parameters=kernel.parameter_size, writer_parameters=model.param_size,
        first=stats(rows[:100]), last=stats(rows[-100:]), all=stats(rows),
        written_coordinates=int(ever.sum()), self_written_coordinates=int(ever[kernel.writer_indices].sum()),
        self_write_path_length=model.self_write_path,
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        persistent_state_and_training_tensor_bytes=budget,
        meta_clips=model.meta_clips, actor_clips=model.actor_clips,
        source_hashes=hashes, rows=rows, seconds=time.perf_counter()-started,
        limitations=['amplitude AND nominal scale differ from raw v1; not a pure capacity comparison',
            'raw exponential eligibility unchanged; normalization state and its derivatives stopped',
            'hybrid eligibility with conditional self-write credit', 'external writer bootstrap remains',
            'broadcast/anchored explicitly use G', 'toy immediate-reward stream, not lifelong validation'])
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'cue_seed_{seed}_g{group_size}_{modulation}_rms.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('seed', 'modulation', 'first', 'last',
        'written_coordinates', 'self_written_coordinates', 'seconds')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[11])
    parser.add_argument('--modulations', nargs='+', choices=['core', 'anchored', 'broadcast'], default=['broadcast'])
    parser.add_argument('--decisions', type=int, default=1000)
    parser.add_argument('--output', default='runs/grouped_write_online/rms_diagnostic')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for modulation in args.modulations:
        for seed in args.seeds:
            run(seed, modulation=modulation, decisions=args.decisions, output=args.output)
