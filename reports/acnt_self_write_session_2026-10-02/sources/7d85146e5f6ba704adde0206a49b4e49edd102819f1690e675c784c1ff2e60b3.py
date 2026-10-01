"""Slower Core / leaky Hand variant; a new uninterrupted life, not rollback.

Changes two design choices together. A positive 1e-5 Core rate leaves every
actual parameter writable. No task phase or target enters the learner.
"""
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from acnt.calibrated_write import CalibratedWriteKernel, CalibratedWriteLearner


def run(seed=11, decisions=3000):
    kernel = CalibratedWriteKernel(seed=seed, group_size=32)
    kernel.body.hand.network[1] = nn.LeakyReLU(.1)
    model = CalibratedWriteLearner(kernel, taus_ms=(.1,.1,.1), actor_lr=.00001)
    rng = random.Random(seed+70000)
    generator = torch.Generator().manual_seed(seed+10000)
    initial = kernel.effective(model.state).detach().clone()
    budget, started, rows = model.persistent_tensor_bytes, time.perf_counter(), []
    source_files = ('acnt/calibrated_write.py', 'acnt/grouped_write_independent.py',
        'acnt/grouped_write.py','acnt/address_write.py','acnt/self_write.py',
        'acnt/full_write.py','acnt/block.py','acnt/adapters.py',__file__)
    hashes = {str(Path(p).resolve().relative_to(Path.cwd())).replace('\\','/'):
              hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files}
    segment = decisions//3
    assert segment*3 == decisions
    for event in range(decisions*4):
        cue, goodness, action = None,None,None
        if event%4 == 0:
            bit = rng.randrange(2)
            mapping = (len(rows)//segment)%2
            target = bit^mapping
            sign, amplitude = 2*bit-1,rng.uniform(.5,1.)
            cue = model.state.new_tensor([[sign*amplitude,-sign*amplitude,
                                          .5*sign*amplitude,-.5*sign*amplitude]])
        if event%4 == 3:
            action, probability, score = model.sample(generator)
            goodness = float(action == target)
            model.record_action(score,now_ms=event*2)
            row = dict(decision=len(rows)+1,action=action,target=target,mapping=mapping,
                goodness=goodness,expected_goodness=probability if target else 1-probability)
        diagnostics = model.step(now_ms=event*2,cue=cue,goodness=goodness,action=action)
        if goodness is not None:
            row.update(diagnostics)
            # Observer-only diagnostics, never read by the subject.
            stride = kernel.width*(1+kernel.hold)
            z = model.state[2*stride:2*stride+kernel.width]
            normalized = (z-kernel.hand_mean)/kernel.hand_scale
            with torch.no_grad():
                params = kernel.mapping(kernel.effective(model.state))
                pre = params['hand.network.0.weight']@normalized+params['hand.network.0.bias']
            row['positive_hidden_fraction'] = float((pre>0).float().mean())
            rows.append(row)
            if len(rows)%100 == 0:
                print(json.dumps(dict(seed=seed,decisions=len(rows),mapping=mapping,
                    expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    f_sign_accuracy=sum(r['actor_f_sign_accuracy'] for r in rows[-100:])/100,
                    tangent_clips=model.tangent_clips,seconds=time.perf_counter()-started)),flush=True)
        assert model.persistent_tensor_bytes == budget
    def stats(part):
        return dict(count=len(part),expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
            sampled_accuracy=sum(r['goodness'] for r in part)/len(part),
            f_sign_accuracy=sum(r['actor_f_sign_accuracy'] for r in part)/len(part),
            calibration_loss=sum(r['calibration_loss'] for r in part)/len(part))
    result = dict(seed=seed,mode='core',task='switch',decisions=decisions,events=model.steps,
        normalize=True,taus_ms=[.1,.1,.1],actor_lr=.00001,head_lr=.01,control_lr=.003,
        hand_activation='LeakyReLU(0.1)',neural_resets=0,outer_parameter_updates=0,
        total_parameters=kernel.parameter_size,writer_parameters=kernel.writer_indices.numel(),
        groups=kernel.group_count,first=stats(rows[:100]),last=stats(rows[-100:]),all=stats(rows),
        phases=[stats(rows[i*segment:(i+1)*segment]) for i in range(3)],
        phase_ends=[stats(rows[(i+1)*segment-100:(i+1)*segment]) for i in range(3)],
        written_coordinates=int(model.ever_written.sum()),
        self_written_coordinates=int(model.ever_written[kernel.writer_indices].sum()),
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        persistent_state_and_training_tensor_bytes=budget,tangent_clips=model.tangent_clips,
        update_clips=model.update_clips,source_hashes=hashes,rows=rows,seconds=time.perf_counter()-started,
        limitations=['two concurrent changes: slow Core and leaky Hand; not a isolated ablation',
            'one seed immediate-feedback switch; not retention or lifelong stability validation',
            'prescribed calibration gradient and extra numerical state remain',
            'all actual parameters writable; slower is not frozen'])
    folder = Path('runs/calibrated_write_online/continual_v1')
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'switch_seed_{seed}_core.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('last','phase_ends','self_written_coordinates','seconds')}),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
