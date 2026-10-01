"""Bounded differentiable rollouts and scalar-Goodness training for original ACNT.

No EventFlowBlock, exponential target dynamics, or delta-W updates are used.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math
import torch
from torch import Tensor, nn
from .runtime import Runtime, READINS, READOUTS

@dataclass(frozen=True)
class TrainingState:
    z: Tensor
    r: Tensor
    a: Tensor
    history: tuple[Tensor, ...]
    times: tuple[int, ...]
    o: Tensor | None

    def detached(self):
        return TrainingState(self.z.detach(), self.r.detach(), self.a.detach(),
                             tuple(x.detach() for x in self.history), self.times,
                             None if self.o is None else self.o.detach())

@dataclass(frozen=True)
class PolicySample:
    action: Tensor
    log_prob: Tensor
    entropy: Tensor

class OriginalTrainingRuntime(nn.Module):
    """Same parameters and equations as Runtime, separate differentiable state.

    This object never calls Runtime.step or its old Plasticity. It does not
    execute mechanical actions. External environments consume action samples.
    Call truncate after each backward/optimizer update. Pending losses must not
    survive an optimizer update. Use reset only at a genuine episode boundary.
    """
    def __init__(self, runtime: Runtime, *, max_window: int = 64):
        super().__init__()
        if type(max_window) is not int or max_window < 1:
            raise ValueError("max_window must be a positive integer")
        self.runtime = runtime
        self.max_window = max_window
        self.states = tuple(self._capture(b) for b in runtime.core.blocks)
        self.active = tuple(b.active for b in runtime.core.blocks)
        self.events_in_window = 0
        self.last_time = max((t for s in self.states for t in s.times), default=None)

    @staticmethod
    def _capture(b):
        return TrainingState(b.z.detach().clone(), b.r.detach().clone(), b.a.detach().clone(),
                             tuple(x.detach().clone() for x in b.A), tuple(b.At),
                             None if b.o is None else b.o.detach().clone())

    def reset(self):
        """Zero neural state for an independent episode; parameters are preserved."""
        states=[]
        for b in self.runtime.core.blocks:
            states.append(TrainingState(torch.zeros_like(b.z), torch.zeros_like(b.r),
                                        torch.zeros_like(b.a), (), (),
                                        None if b.o is None else torch.zeros_like(b.o)))
        self.states=tuple(states)
        self.active=tuple(b.active for b in self.runtime.core.blocks)
        self.events_in_window=0
        self.last_time=None

    def truncate(self):
        """Carry state/history values across windows, cutting only derivative links."""
        self.states=tuple(s.detached() for s in self.states)
        self.events_in_window=0

    def set_active_mask(self, mask: Sequence[bool]):
        values=list(mask)
        if len(values)!=len(self.states) or any(type(x) is not bool for x in values):
            raise ValueError("one bool per Block is required")
        values[self.runtime.organ_blocks["route"]]=True
        values[self.runtime.organ_blocks["goodness"]]=self.runtime.goodness_active
        self.active=tuple(values)

    def step(self, *, now_ms: int, readins: Mapping[str, Tensor] | None = None):
        if type(now_ms) is not int or (self.last_time is not None and now_ms < self.last_time):
            raise ValueError("time must be monotonic integer milliseconds")
        if self.events_in_window >= self.max_window:
            raise RuntimeError("training window exhausted; update and truncate before continuing")
        inputs={} if readins is None else dict(readins)
        if not set(inputs).issubset(READINS):
            raise ValueError("ReadIn must be eye or ear")
        forced={self.runtime.organ_blocks[n] for n in inputs}
        encoded={self.runtime.organ_blocks[n]:self.runtime.adapters[n].forward_train(x)
                 for n,x in inputs.items()}
        active=list(self.active)
        active[self.runtime.organ_blocks["route"]]=True
        active[self.runtime.organ_blocks["goodness"]]=self.runtime.goodness_active
        sources={i:s.z for i,s in enumerate(self.states) if active[i] or i in forced}
        states=list(self.states)
        updated=[]
        for i,(b,s) in enumerate(zip(self.runtime.core.blocks,self.states)):
            due=not s.times or now_ms-s.times[-1]>=b.ticktime
            if i not in forced and (not active[i] or not due):
                continue
            o=encoded.get(i,s.o)
            r,a,z,history,times=b.transition(history=s.history,times=s.times,
                                           now_ms=now_ms,active_z=sources,o=o)
            states[i]=TrainingState(z,r,a,history,times,None if o is None else torch.zeros_like(o))
            updated.append(i)
        self.states=tuple(states)
        self.active=tuple(active)
        self.last_time=now_ms
        self.events_in_window+=1
        return tuple(updated)

    def decode(self, name: str):
        if name not in READOUTS:
            raise ValueError("unknown ReadOut")
        return self.runtime.adapters[name].forward_train(self.states[self.runtime.organ_blocks[name]].z)

    @staticmethod
    def bernoulli(raw: Tensor, *, tau: float = 1.0, threshold: float = 0.0,
                  generator: torch.Generator | None = None, effective_mask: Tensor | None = None):
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        logits=(raw-threshold)/tau
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite policy logits")
        dist=torch.distributions.Bernoulli(logits=logits)
        action=torch.bernoulli(dist.probs.detach(),generator=generator)
        logp=dist.log_prob(action)
        entropy=dist.entropy()
        if effective_mask is not None:
            logp=logp*effective_mask
            entropy=entropy*effective_mask
        return PolicySample(action.detach(),logp.sum(),entropy.sum())

    @staticmethod
    def joint_sample(*samples: PolicySample):
        """Combine Hand/Speak/Route decisions at one timestamp for one reward."""
        if not samples:
            raise ValueError("at least one action sample is required")
        return PolicySample(torch.cat([s.action.reshape(-1) for s in samples]),
                            torch.stack([s.log_prob for s in samples]).sum(),
                            torch.stack([s.entropy for s in samples]).sum())

    def sample_hand(self, *, generator=None, continuous_std: float | None = None):
        adapter=self.runtime.adapters["hand"]
        raw=self.decode("hand")
        count=adapter.discrete_controls
        discrete=self.bernoulli(raw[:count],tau=self.runtime.noise_scale,generator=generator)
        if not adapter.continuous_controls:
            return discrete
        if continuous_std is None or not math.isfinite(continuous_std) or continuous_std <= 0:
            raise ValueError("continuous Hand training requires a positive exploration std")
        continuous=self.gaussian(raw[count:],std=continuous_std,generator=generator)
        return PolicySample(torch.cat((discrete.action,continuous.action)),
                            discrete.log_prob+continuous.log_prob,
                            discrete.entropy+continuous.entropy)

    @staticmethod
    def gaussian(mean: Tensor, *, std: float, generator=None):
        """Explicit training exploration for continuous outputs; deploy the mean."""
        if not math.isfinite(std) or std <= 0 or not torch.isfinite(mean).all():
            raise ValueError("finite mean and positive finite std are required")
        action=mean.detach()+std*torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=generator)
        dist=torch.distributions.Normal(mean,std)
        return PolicySample(action,dist.log_prob(action).sum(),dist.entropy().sum())

    def sample_speak(self, *, std: float, generator=None):
        return self.gaussian(self.decode("speak"),std=std,generator=generator)

    def sample_route(self, *, generator=None):
        raw=self.decode("route")
        mask=torch.ones_like(raw)
        rid=self.runtime.organ_blocks["route"]
        gid=self.runtime.organ_blocks["goodness"]
        mask[rid]=0; mask[gid]=0
        sample=self.bernoulli(raw,tau=self.runtime.noise_scale,generator=generator,effective_mask=mask)
        actual=sample.action.clone()
        actual[rid]=1; actual[gid]=float(self.runtime.goodness_active)
        if self.runtime.execution_enabled["route"]:
            self.set_active_mask([bool(x) for x in actual.tolist()])
        else:
            return PolicySample(actual,sample.log_prob*0,sample.entropy*0)
        return PolicySample(actual,sample.log_prob,sample.entropy)

    @torch.no_grad()
    def commit(self):
        """Copy learned rollout values back to original inference state."""
        self.truncate()
        for b,s,active in zip(self.runtime.core.blocks,self.states,self.active):
            b.z=s.z.clone(); b.r=s.r.clone(); b.a=s.a.clone()
            b.A.clear(); b.A.extend(x.clone() for x in s.history)
            b.At.clear(); b.At.extend(s.times)
            b.o=None if s.o is None else s.o.clone()
            b.active=active
        if self.runtime.plasticity is not None:
            self.runtime.plasticity.set_learning(False)
        self.runtime.learning=False

class GoodnessTrainer:
    """On-policy REINFORCE with a past-only scalar baseline and bounded BPTT.

    Each sample is a joint action event. Goodness values are scalar rewards at
    matching event times; reward-to-go credits earlier actions. A terminal-only
    Goodness is represented by zero intermediate rewards and the final value.
    No learned value head, extra reward axis, or old delta-W updates are used.
    """
    def __init__(self, model: OriginalTrainingRuntime, *, lr=0.003, baseline_decay=0.95,
                 grad_clip=1.0, entropy_weight=0.0):
        if not math.isfinite(lr) or lr<=0 or not 0<=baseline_decay<1:
            raise ValueError("invalid optimizer or baseline settings")
        if not math.isfinite(grad_clip) or grad_clip<=0 or not math.isfinite(entropy_weight) or entropy_weight<0:
            raise ValueError("invalid gradient/entropy settings")
        self.model=model
        self.baseline_decay=baseline_decay
        self.grad_clip=grad_clip
        self.entropy_weight=entropy_weight
        self.baseline=0.0
        gid=model.runtime.organ_blocks["goodness"]
        excluded={id(p) for p in model.runtime.core.blocks[gid].parameters()}
        excluded.update(id(p) for p in model.runtime.adapters["goodness"].parameters())
        self.parameters=[p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
        self.optimizer=torch.optim.Adam(self.parameters,lr=lr)
        self.goodness_parameters=[p for p in list(model.runtime.core.blocks[gid].parameters())+list(model.runtime.adapters["goodness"].parameters()) if p.requires_grad]
        self.goodness_optimizer=torch.optim.Adam(self.goodness_parameters,lr=lr) if self.goodness_parameters else None

    @staticmethod
    def returns(goodness: Sequence[float], times_ms: Sequence[int], *, discount_tau_ms: float | None = None):
        if len(goodness)!=len(times_ms) or not goodness:
            raise ValueError("Goodness and action timestamps must correspond")
        if any(not math.isfinite(g) or not 0<=g<=1 for g in goodness):
            raise ValueError("Goodness must be finite and in [0,1]")
        if any(type(t) is not int for t in times_ms) or any(b<a for a,b in zip(times_ms,times_ms[1:])):
            raise ValueError("action timestamps must be monotonic integer milliseconds")
        if discount_tau_ms is not None and (not math.isfinite(discount_tau_ms) or discount_tau_ms<=0):
            raise ValueError("discount time constant must be positive")
        out=[0.0]*len(goodness); value=0.0
        for i in range(len(goodness)-1,-1,-1):
            discount=1.0 if discount_tau_ms is None or i==len(goodness)-1 else math.exp(-(times_ms[i+1]-times_ms[i])/discount_tau_ms)
            value=float(goodness[i])+discount*value
            out[i]=value
        return out

    def update(self, episodes, *, discount_tau_ms=None):
        """episodes: sequence of (PolicySamples, scalar Goodness values, times_ms).

        Independent episode graphs may be accumulated for one on-policy batch.
        Weights remain fixed while sampling the complete batch.
        """
        if not episodes:
            raise ValueError("at least one rollout is required")
        losses=[]; observed=[]
        for samples,goodness,times in episodes:
            if len(samples)!=len(goodness):
                raise ValueError("one Goodness value per joint action sample is required")
            values=self.returns(goodness,times,discount_tau_ms=discount_tau_ms)
            loss=sum(-(v-self.baseline)*sample.log_prob-self.entropy_weight*sample.entropy
                     for sample,v in zip(samples,values))
            losses.append(loss)
            observed.extend(values)
        loss=torch.stack(losses).mean()
        result=self.update_loss(loss)
        self.baseline=self.baseline_decay*self.baseline+(1-self.baseline_decay)*(sum(observed)/len(observed))
        result["baseline"]=self.baseline
        return result

    def update_loss(self, loss: Tensor):
        """Also supports differentiable supervised calibration/pretraining losses."""
        if loss.ndim!=0 or not torch.isfinite(loss):
            raise ValueError("loss must be finite and scalar")
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters):
            self.model.truncate()
            raise FloatingPointError("nonfinite gradient")
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.grad_clip,error_if_nonfinite=True)
        self.optimizer.step()
        self.model.truncate()
        return {"loss":float(loss.detach()),"gradient_norm":float(norm.detach())}

    def calibrate_goodness(self, teacher: float):
        """External teacher calibrates only Goodness Block/Adapter.

        Run in a separate calibration rollout, after actor losses are consumed.
        The raw scalar is fit to [0,1]; the inference clamp remains unchanged.
        No self-generated Goodness is used as a calibration target.
        """
        if not math.isfinite(teacher) or not 0<=teacher<=1:
            raise ValueError("external teacher must be in [0,1]")
        if not self.model.runtime.goodness_active or self.goodness_optimizer is None:
            raise RuntimeError("Goodness calibration requires an enabled trainable Goodness organ")
        prediction=self.model.decode("goodness").reshape(())
        loss=.5*(prediction-teacher).square()
        gradients=torch.autograd.grad(loss,self.goodness_parameters,allow_unused=True)
        self.goodness_optimizer.zero_grad(set_to_none=True)
        for p,g in zip(self.goodness_parameters,gradients):
            if g is not None and not torch.isfinite(g).all():
                raise FloatingPointError("nonfinite calibration gradient")
            p.grad=g
        torch.nn.utils.clip_grad_norm_(self.goodness_parameters,self.grad_clip,error_if_nonfinite=True)
        self.goodness_optimizer.step()
        self.model.truncate()
        return float(loss.detach())

    def save_checkpoint(self, path, *, generator=None):
        """Save after an update, never with outstanding action graphs."""
        from pathlib import Path
        states=[{"z":s.z.detach(),"r":s.r.detach(),"a":s.a.detach(),
                 "history":[x.detach() for x in s.history],"times":list(s.times),
                 "o":None if s.o is None else s.o.detach()} for s in self.model.states]
        signature=[(b.neuron_size,b.source_sizes,b.hold_tick,b.ticktime,b.nlm_hidden_dim) for b in self.model.runtime.core.blocks]
        payload={"version":1,"signature":signature,"model":self.model.runtime.state_dict(),
                 "optimizer":self.optimizer.state_dict(),"goodness_optimizer":None if self.goodness_optimizer is None else self.goodness_optimizer.state_dict(),
                 "baseline":self.baseline,"baseline_decay":self.baseline_decay,
                 "grad_clip":self.grad_clip,"entropy_weight":self.entropy_weight,
                 "states":states,"active":self.model.active,"last_time":self.model.last_time,
                 "max_window":self.model.max_window,"generator":None if generator is None else generator.get_state(),
                 "goodness_active":self.model.runtime.goodness_active,
                 "execution_enabled":dict(self.model.runtime.execution_enabled),
                 "organ_blocks":dict(self.model.runtime.organ_blocks),
                 "noise_scale":self.model.runtime.noise_scale}
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,target)

    def load_checkpoint(self, path, *, generator=None):
        """Restore weights, optimizer, neural histories, baseline and optional RNG."""
        device=next(self.model.parameters()).device
        payload=torch.load(path,map_location=device,weights_only=True)
        signature=[(b.neuron_size,b.source_sizes,b.hold_tick,b.ticktime,b.nlm_hidden_dim) for b in self.model.runtime.core.blocks]
        if payload["version"]!=1 or payload["signature"]!=signature or payload["organ_blocks"]!=self.model.runtime.organ_blocks:
            raise ValueError("checkpoint architecture does not match")
        self.model.runtime.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.goodness_optimizer is not None and payload["goodness_optimizer"] is not None:
            self.goodness_optimizer.load_state_dict(payload["goodness_optimizer"])
        self.baseline=payload["baseline"]; self.baseline_decay=payload["baseline_decay"]
        self.grad_clip=payload["grad_clip"]; self.entropy_weight=payload["entropy_weight"]
        self.model.states=tuple(TrainingState(s["z"],s["r"],s["a"],tuple(s["history"]),tuple(s["times"]),s["o"]) for s in payload["states"])
        self.model.active=tuple(payload["active"]); self.model.last_time=payload["last_time"]
        self.model.max_window=payload["max_window"]; self.model.events_in_window=0
        self.model.runtime.goodness_active=payload["goodness_active"]
        self.model.runtime.execution_enabled=payload["execution_enabled"]
        self.model.runtime.noise_scale=payload["noise_scale"]
        if generator is not None and payload["generator"] is not None:
            generator.set_state(payload["generator"].cpu())
