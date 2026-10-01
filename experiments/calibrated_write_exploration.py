"""Confidence-limited action readout on a new irreversible toy trajectory.

Temperature is chosen from the current raw logit before action/reward. Score
conditions on that temperature (stops its selection derivative). This is an
explicit additional approximation, not an unbiased full-policy gradient.
"""
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from acnt.calibrated_write import CalibratedWriteKernel, CalibratedWriteLearner


class ExploringKernel(CalibratedWriteKernel):
    def __init__(self, **settings):
        super().__init__(**settings)
        self.register_buffer('temperature', torch.ones(()))

    def logits(self, state):
        return super().logits(state)/self.temperature


class ExploringLearner(CalibratedWriteLearner):
    def sample(self, generator):
        stride = self.kernel.width*(1+self.kernel.hold)
        self.normalize_role(0, self.state[2*stride:2*stride+self.kernel.width])
        with torch.no_grad():
            raw = CalibratedWriteKernel.logits(self.kernel, self.state)
            self.kernel.temperature.copy_((raw.abs()/3.).clamp_min(1.))
            probability = float(torch.sigmoid(self.kernel.logits(self.state)))
            action = int(torch.bernoulli(self.state.new_tensor(probability),generator=generator))
        return action,probability,self.policy_score(action)

    @property
    def persistent_tensor_bytes(self):
        return super().persistent_tensor_bytes+self.kernel.temperature.element_size()


def run(seed=11, decisions=3000):
    kernel=ExploringKernel(seed=seed,group_size=32)
    kernel.body.hand.network[1]=nn.LeakyReLU(.1)
    model=ExploringLearner(kernel,taus_ms=(.1,.1,.1),actor_lr=.00001)
    rng=random.Random(seed+70000)
    generator=torch.Generator().manual_seed(seed+10000)
    initial=kernel.effective(model.state).detach().clone()
    budget,started,rows=model.persistent_tensor_bytes,time.perf_counter(),[]
    sources=('acnt/calibrated_write.py','acnt/grouped_write_independent.py',
        'acnt/grouped_write.py','acnt/address_write.py','acnt/self_write.py',
        'acnt/full_write.py','acnt/block.py','acnt/adapters.py',__file__)
    hashes={str(Path(p).resolve().relative_to(Path.cwd())).replace('\\','/'):
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    segment=decisions//3
    assert segment*3==decisions
    for event in range(decisions*4):
        cue,goodness,action=None,None,None
        if event%4==0:
            bit=rng.randrange(2)
            mapping=(len(rows)//segment)%2
            target=bit^mapping
            sign,amplitude=2*bit-1,rng.uniform(.5,1.)
            cue=model.state.new_tensor([[sign*amplitude,-sign*amplitude,
                                       .5*sign*amplitude,-.5*sign*amplitude]])
        if event%4==3:
            action,probability,score=model.sample(generator)
            goodness=float(action==target)
            model.record_action(score,now_ms=event*2)
            row=dict(decision=len(rows)+1,action=action,target=target,mapping=mapping,
                goodness=goodness,expected_goodness=probability if target else 1-probability,
                temperature=float(kernel.temperature))
            assert .0474 <= probability <= .9526
        diagnostics=model.step(now_ms=event*2,cue=cue,goodness=goodness,action=action)
        if goodness is not None:
            row.update(diagnostics)
            rows.append(row)
            if len(rows)%100==0:
                print(json.dumps(dict(seed=seed,decisions=len(rows),mapping=mapping,
                    expected_goodness=sum(r['expected_goodness'] for r in rows[-100:])/100,
                    f_sign_accuracy=sum(r['actor_f_sign_accuracy'] for r in rows[-100:])/100,
                    tangent_clips=model.tangent_clips,seconds=time.perf_counter()-started)),flush=True)
        assert budget==model.persistent_tensor_bytes
    def stats(part):
        return dict(count=len(part),expected_goodness=sum(r['expected_goodness'] for r in part)/len(part),
            sampled_accuracy=sum(r['goodness'] for r in part)/len(part),
            f_sign_accuracy=sum(r['actor_f_sign_accuracy'] for r in part)/len(part),
            calibration_loss=sum(r['calibration_loss'] for r in part)/len(part))
    result=dict(seed=seed,mode='core',task='switch',decisions=decisions,events=model.steps,
        normalize=True,taus_ms=[.1,.1,.1],actor_lr=.00001,head_lr=.01,control_lr=.003,
        hand_activation='LeakyReLU(0.1)',logit_confidence_limit=3.,
        neural_resets=0,outer_parameter_updates=0,total_parameters=kernel.parameter_size,
        writer_parameters=kernel.writer_indices.numel(),groups=kernel.group_count,
        first=stats(rows[:100]),last=stats(rows[-100:]),all=stats(rows),
        phases=[stats(rows[i*segment:(i+1)*segment]) for i in range(3)],
        phase_ends=[stats(rows[(i+1)*segment-100:(i+1)*segment]) for i in range(3)],
        written_coordinates=int(model.ever_written.sum()),
        self_written_coordinates=int(model.ever_written[kernel.writer_indices].sum()),
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        persistent_state_and_training_tensor_bytes=budget,tangent_clips=model.tangent_clips,
        update_clips=model.update_clips,source_hashes=hashes,rows=rows,seconds=time.perf_counter()-started,
        limitations=['new confidence-limited readout; keeps both actions probability at least sigmoid(-3)',
            'temperature selection derivative stopped; conditional score, not exact full-policy gradient',
            'one seed immediate switch, not lifelong retention proof',
            'prescribed calibration gradient and fixed numerical training state remain'])
    folder=Path('runs/calibrated_write_online/exploration_v1')
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'switch_seed_{seed}_core.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('last','phase_ends','self_written_coordinates','seconds')}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    run()
