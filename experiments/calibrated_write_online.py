"""One uninterrupted cue/switch trajectory for self-calibrating vector Write."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from acnt.calibrated_write import CalibratedWriteKernel, CalibratedWriteLearner


def run(seed=11, *, decisions=1000, mode='core', task='cue', normalize=True,
        taus_ms=(8., 32., 128.), head_lr=.01, control_lr=.003,
        output='runs/calibrated_write_online/pilot'):
    kernel = CalibratedWriteKernel(seed=seed, group_size=32)
    model = CalibratedWriteLearner(kernel, mode=mode, normalize=normalize,
        taus_ms=taus_ms, head_lr=head_lr, control_lr=control_lr)
    rng = random.Random(seed+70000)
    generator = torch.Generator().manual_seed(seed+10000)
    initial = kernel.effective(model.state).detach().clone()
    budget = model.persistent_tensor_bytes
    sources = ('acnt/calibrated_write.py', 'acnt/grouped_write_independent.py',
        'acnt/grouped_write.py', 'acnt/address_write.py', 'acnt/self_write.py',
        'acnt/full_write.py', 'acnt/block.py', 'acnt/adapters.py', __file__)
    hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\', '/'):
              hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    rows, bit, target, started = [], 0, 0, time.perf_counter()
    for event in range(decisions*4):
        cue, goodness, action = None, None, None
        phase = event%4
        if phase == 0:
            bit = rng.randrange(2)
            mapping = (len(rows)//max(1, decisions//3))%2 if task == 'switch' else 0
            target = bit^mapping
            sign, amplitude = 2*bit-1, rng.uniform(.5, 1.)
            cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude,
                                          .5*sign*amplitude, -.5*sign*amplitude]])
        if phase == 3:
            action, probability, score = model.sample(generator)
            goodness = float(action == target)
            model.record_action(score, now_ms=event*2)
            row = dict(decision=len(rows)+1, action=action, target=target,
                mapping=mapping, goodness=goodness,
                expected_goodness=probability if target else 1-probability)
        diagnostics = model.step(now_ms=event*2, cue=cue, goodness=goodness, action=action)
        if goodness is not None:
            row.update(diagnostics)
            rows.append(row)
            if len(rows)%100 == 0:
                print(json.dumps(dict(seed=seed, mode=mode, task=task, decisions=len(rows),
                    expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    f_sign_accuracy=sum(r['actor_f_sign_accuracy'] for r in rows[-100:])/100,
                    tangent_clips=model.tangent_clips,
                    seconds=time.perf_counter()-started)), flush=True)
        assert budget == model.persistent_tensor_bytes
        assert model.state.grad_fn is None and model.tangent.grad_fn is None
    def stats(part):
        return dict(count=len(part), expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
            sampled_accuracy=sum(r['goodness'] for r in part)/len(part),
            f_sign_accuracy=sum(r['actor_f_sign_accuracy'] for r in part)/len(part),
            calibration_loss=sum(r['calibration_loss'] for r in part)/len(part))
    result = dict(seed=seed, mode=mode, task=task, decisions=decisions, events=model.steps,
        normalize=normalize, taus_ms=list(taus_ms), head_lr=head_lr, control_lr=control_lr,
        total_parameters=kernel.parameter_size, writer_parameters=kernel.writer_indices.numel(),
        groups=kernel.group_count, neural_resets=0, outer_parameter_updates=0,
        first=stats(rows[:100]), last=stats(rows[-100:]), all=stats(rows),
        phases=[stats([r for r in rows if (r['decision']-1)//max(1, decisions//3)==i])
                for i in range(3)] if task == 'switch' else [],
        written_coordinates=int(model.ever_written.sum()),
        self_written_coordinates=int(model.ever_written[kernel.writer_indices].sum()),
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        persistent_state_and_training_tensor_bytes=budget, tangent_clips=model.tangent_clips,
        update_clips=model.update_clips, seconds=time.perf_counter()-started,
        source_hashes=hashes, rows=rows,
        limitations=['task-independent calibration loss explicitly prescribed, not discovered by Core',
            'controller eligibility uses calibration gradient; changes eligibility binding',
            'finite dense neural sensitivity/moments/normalization are extra numerical state, not Core H',
            'stop-update and statistic derivatives; not a complete self-modifying lifetime gradient',
            'normalization changes Adapter input mapping, ordinary Block forward retained',
            'immediate scalar reward toy; delayed anonymous rewards and lifelong retention unvalidated'])
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    target_path = folder/f'{task}_seed_{seed}_{mode}.json'
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('seed','mode','task','last','written_coordinates',
        'self_written_coordinates','seconds')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[11])
    parser.add_argument('--modes', nargs='+', default=['core'])
    parser.add_argument('--task', choices=['cue','switch'], default='cue')
    parser.add_argument('--decisions', type=int, default=1000)
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--taus-ms', type=float, nargs='+', default=[8.,32.,128.])
    parser.add_argument('--head-lr', type=float, default=.01)
    parser.add_argument('--control-lr', type=float, default=.003)
    parser.add_argument('--output', default='runs/calibrated_write_online/pilot')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for mode in args.modes:
        for seed in args.seeds:
            run(seed, decisions=args.decisions, mode=mode, task=args.task,
                normalize=not args.no_normalize, taus_ms=tuple(args.taus_ms),
                head_lr=args.head_lr, control_lr=args.control_lr, output=args.output)
