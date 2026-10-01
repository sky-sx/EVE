"""Full-coverage address self-write bootstrap on one uninterrupted toy stream."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from acnt.address_write import AddressWriteKernel, AddressWriteLearner


def run(seed=11, *, task='cue', mode='full', decisions=1000, gap=3,
        lr=.001, write_scale=.002, output='runs/address_write_online/pilot'):
    kernel = AddressWriteKernel(seed=seed, write_scale=write_scale)
    model = AddressWriteLearner(kernel, lr=0 if mode == 'frozen' else lr,
        write_enabled=mode != 'no_write', cut_write_credit=mode == 'cut_write',
        constant_controller=mode == 'constant_controller')
    rng = random.Random(seed+70000)
    action_rng = torch.Generator().manual_seed(seed+10000)
    rows, bit, started = [], 0, time.perf_counter()
    resource = model.persistent_tensor_bytes
    initial = kernel.effective(model.state).detach().clone()
    ever_written = torch.zeros(kernel.parameter_size, dtype=torch.bool)  # Observer only.
    for event in range(decisions*(gap+1)):
        phase, cue, goodness, action = event%(gap+1), None, None, None
        if phase == 0:
            bit = rng.randrange(2)
            sign, amplitude = 2*bit-1, rng.uniform(.5, 1.)
            cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude,
                                          .5*sign*amplitude, -.5*sign*amplitude]])
        if phase == gap:
            decision = len(rows)
            target = 1 if task == 'constant' else bit if task == 'cue' else int(decision//250%2 == 0)
            action, probability, score = model.sample(action_rng)
            goodness = float(action == target)
            row = dict(decision=decision+1, action=action, target=target, probability=probability,
                       goodness=goodness, expected_goodness=probability if target else 1-probability,
                       score_norm=float(score.norm()))
        # No labels, expected reward or observer statistics enter the subject.
        before = model.state[kernel.neural_size:].clone()
        model.step(now_ms=event*2, cue=cue, goodness=goodness, action=action)
        ever_written |= model.state[kernel.neural_size:] != before
        if goodness is not None:
            row['bootstrap_update_norm'] = model.learn(score, goodness)
            rows.append(row)
            if len(rows)%100 == 0:
                print(json.dumps(dict(seed=seed, task=task, mode=mode, decisions=len(rows),
                    expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    sampled_accuracy=sum(r['goodness'] for r in rows[-100:])/100,
                    meta_clips=model.meta_clips, seconds=time.perf_counter()-started)), flush=True)
        assert resource == model.persistent_tensor_bytes
        assert model.state.grad_fn is None and model.meta.grad_fn is None
    def stats(part):
        return dict(count=len(part), expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
                    sampled_accuracy=sum(r['goodness'] for r in part)/len(part))
    source_names = ('acnt/address_write.py', 'acnt/full_write.py', 'acnt/self_write.py',
                    'acnt/block.py', 'acnt/adapters.py', __file__)
    hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\', '/'):
              hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_names}
    result = dict(seed=seed, task=task, mode=mode, decisions=decisions, events=model.steps,
        gap=gap, lr=model.lr, write_scale=write_scale, neural_resets=0,
        first=stats(rows[:min(100, decisions)]), last=stats(rows[-min(100, decisions):]),
        all=stats(rows), total_effective_parameters=kernel.parameter_size,
        write_controls=kernel.parameter_size, bootstrap_parameters=model.param_size,
        written_coordinates=int(ever_written.sum()),
        self_written_coordinates=int(ever_written[kernel.writer_indices].sum()),
        write_path_length=model.write_path, self_write_path_length=model.self_write_path,
        effective_parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        persistent_state_and_training_tensor_bytes=resource,
        address_metadata_bytes=kernel.addresses.numel()*kernel.addresses.element_size(),
        meta_clips=model.meta_clips, bootstrap_updates=model.updates,
        seconds=time.perf_counter()-started, source_hashes=hashes, rows=rows,
        limitations=['external writer bootstrap remains', 'immediate reward only',
                     'conditional common-origin tangent with stopped outer updates and clipping',
                     'all-active synchronous seven original Blocks', 'synthetic cue task is not lifelong validation'])
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{task}_seed_{seed}_{mode}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('seed', 'task', 'mode', 'first', 'last',
        'written_coordinates', 'self_written_coordinates', 'seconds')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[11])
    parser.add_argument('--modes', nargs='+', choices=['full', 'frozen', 'no_write', 'cut_write',
        'constant_controller'], default=['full'])
    parser.add_argument('--task', choices=['constant', 'cue', 'switch'], default='cue')
    parser.add_argument('--decisions', type=int, default=1000)
    parser.add_argument('--gap', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--write-scale', type=float, default=.002)
    parser.add_argument('--output', default='runs/address_write_online/pilot')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for mode in args.modes:
        for seed in args.seeds:
            run(seed, task=args.task, mode=mode, decisions=args.decisions, gap=args.gap,
                lr=args.lr, write_scale=args.write_scale, output=args.output)
