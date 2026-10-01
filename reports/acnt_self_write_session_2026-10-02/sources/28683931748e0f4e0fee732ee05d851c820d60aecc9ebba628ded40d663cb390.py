"""Independent readout capacity test; same raw trace and actuator as grouped v1."""
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from acnt.grouped_write import GroupedWriteLearner
from acnt.grouped_write_independent import IndependentGroupedWriteKernel


def run(decisions=1000, seed=11):
    kernel = IndependentGroupedWriteKernel(seed=seed, group_size=32, modulation='core')
    model = GroupedWriteLearner(kernel)
    rng = random.Random(seed+70000)
    actions = torch.Generator().manual_seed(seed+10000)
    rows, bit, started = [], 0, time.perf_counter()
    budget = model.persistent_tensor_bytes
    initial = kernel.effective(model.state).detach().clone()
    ever = torch.zeros(kernel.parameter_size, dtype=torch.bool)
    sources = ('acnt/grouped_write_independent.py', 'acnt/grouped_write.py', 'acnt/address_write.py',
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
                print(json.dumps(dict(seed=seed, decisions=len(rows), writer_parameters=model.param_size,
                    groups=kernel.group_count, expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    meta_clips=model.meta_clips, actor_clips=model.actor_clips,
                    seconds=time.perf_counter()-started)), flush=True)
        assert budget == model.persistent_tensor_bytes
    def stats(part):
        return dict(count=len(part), expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
                    sampled_accuracy=sum(r['goodness'] for r in part)/len(part))
    result = dict(seed=seed, task='cue', group_size=32, groups=kernel.group_count,
        modulation='core', generator='independent linear heads', actuator='raw', write_scale=.01, lr=.001,
        decisions=decisions, events=model.steps, neural_resets=0,
        total_effective_parameters=kernel.parameter_size, writer_parameters=model.param_size,
        first=stats(rows[:100]), last=stats(rows[-100:]), all=stats(rows),
        written_coordinates=int(ever.sum()), self_written_coordinates=int(ever[kernel.writer_indices].sum()),
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        meta_clips=model.meta_clips, actor_clips=model.actor_clips,
        persistent_state_and_training_tensor_bytes=budget, source_hashes=hashes, rows=rows,
        seconds=time.perf_counter()-started,
        limitations=['one seed', 'new independently parameterized outputs increase total mutable parameters',
            'raw exponential eligibility shared with grouped v1; hybrid stopped derivative approximation',
            'external writer bootstrap remains', 'immediate reward toy, not lifelong validation'])
    folder = Path('runs/grouped_write_online/independent_capacity')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'cue_seed_{seed}_g32_independent.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('seed', 'writer_parameters', 'groups', 'first', 'last',
        'self_written_coordinates', 'seconds')}), flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
