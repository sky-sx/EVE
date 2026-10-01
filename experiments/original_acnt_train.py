"""Scalar-Goodness delayed-cue training on the original six-organ ACNT.

EarAdapter -> original Core/GLU/LN/private NLM/history -> HandAdapter.
A terminal action chooses the previously observed bit. No label loss is used.
"""
import argparse
import json
from pathlib import Path
import time
import hashlib
import torch
from acnt import Block, Core, Runtime, OriginalTrainingRuntime, GoodnessTrainer
from acnt.adapters import EyeAdapter,EarAdapter,HandAdapter,SpeakAdapter,GoodnessAdapter,RouteAdapter
from acnt.runtime import ORGANS

def build(seed,device="cpu"):
    torch.manual_seed(seed)
    n=6; count=6
    core=Core([Block(i,n,[n]*count,readin=i<2,ticktime=1,hold_tick=4) for i in range(count)])
    adapters={"eye":EyeAdapter(n),"ear":EarAdapter(n,window_samples=4),
              "hand":HandAdapter(n,hidden_size=12),"speak":SpeakAdapter(n),
              "goodness":GoodnessAdapter(n),"route":RouteAdapter(n,count)}
    runtime=Runtime(core,adapters,dict(zip(ORGANS,range(6))))
    runtime.execution_enabled["route"]=False
    return OriginalTrainingRuntime(runtime.to(device),max_window=8)

def rollout(model,bit,amplitude=1.0,interval=2):
    model.reset()
    sign=2*bit-1
    audio=model.states[0].z.new_tensor([[sign*amplitude,-sign*amplitude,.5*sign*amplitude,-.5*sign*amplitude]])
    for t in range(4):
        model.step(now_ms=t*interval,readins={"ear":audio} if t==0 else None)
    return model.decode("hand")[0:1]

@torch.no_grad()
def evaluate(model,seed):
    # Balanced held-out amplitudes, both seen and unseen physical intervals.
    rows=[]
    for bit in (0,1):
        for amplitude in (.35,.65,.85,1.15):
            for interval in (1,2,3):
                logits=rollout(model,bit,amplitude,interval)
                p=float(torch.sigmoid(logits/model.runtime.noise_scale).item())
                rows.append({"bit":bit,"amplitude":amplitude,"interval_ms":interval,
                             "probability_one":p,"correct":bool((p>=.5)==bool(bit))})
    return {"accuracy":sum(r["correct"] for r in rows)/len(rows),
            "expected_goodness":sum(r["probability_one"] if r["bit"] else 1-r["probability_one"] for r in rows)/len(rows),
            "rows":rows}

def train(seed,updates,batch_size,lr,output,freeze_core=False):
    model=build(seed)
    if freeze_core:
        for p in model.runtime.core.parameters(): p.requires_grad_(False)
    trainer=GoodnessTrainer(model,lr=lr,entropy_weight=.001)
    generator=torch.Generator().manual_seed(seed+10000)
    before={n:p.detach().clone() for n,p in model.runtime.named_parameters()}
    initial=evaluate(model,seed)
    curve=[]; started=time.perf_counter()
    for step in range(updates):
        episodes=[]; rewards=[]
        for _ in range(batch_size):
            bit=int(torch.randint(2,(),generator=generator))
            amplitude=.5+float(torch.rand((),generator=generator))*.5
            logits=rollout(model,bit,amplitude,2)
            sample=model.bernoulli(logits,tau=model.runtime.noise_scale,generator=generator)
            goodness=float(int(sample.action.item())==bit)
            episodes.append(([sample],[goodness],[6]))
            rewards.append(goodness)
        stats=trainer.update(episodes)
        row={"update":step+1,"sample_goodness":sum(rewards)/len(rewards),**stats}
        if step==0 or (step+1)%25==0 or step+1==updates:
            val=evaluate(model,seed)
            row.update({"eval_accuracy":val["accuracy"],"eval_expected_goodness":val["expected_goodness"]})
            print(json.dumps({"seed":seed,**row}),flush=True)
        curve.append(row)
    final=evaluate(model,seed)
    changed={n:float((p.detach()-before[n]).norm()) for n,p in model.runtime.named_parameters()
             if not torch.equal(p.detach(),before[n])}
    result={"seed":seed,"algorithm":"bounded BPTT + scalar-Goodness REINFORCE",
            "architecture":"original ACNT, six organ bindings, no G2", "freeze_core":freeze_core,
            "updates":updates,"batch_size":batch_size,"seconds":time.perf_counter()-started,
            "initial":initial,"final":final,"curve":curve,"parameter_changes":changed,
            "limitations":["Ear cue task only; Eye/Speak/Route learning not benchmarked", "finite 4-event credit window", "episode resets", "full architecture held-out generalization not established"]}
    target=Path(output)/f"seed_{seed}{'_freeze_core' if freeze_core else ''}.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    torch.save(model.runtime.state_dict(),target.with_suffix(".pt"))
    trainer.save_checkpoint(target.with_name(target.stem+"_training.pt"),generator=generator)
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path("acnt/original_training.py"),Path("acnt/block.py"),Path("acnt/adapters.py")]}
    target.with_name(target.stem+"_source_hashes.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(json.dumps({"seed":seed,"initial_goodness":initial["expected_goodness"],
                      "final_goodness":final["expected_goodness"],"accuracy":final["accuracy"],"seconds":result["seconds"]}),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
    parser.add_argument("--updates",type=int,default=300)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--lr",type=float,default=.003)
    parser.add_argument("--output",default="runs/original_acnt_training")
    parser.add_argument("--freeze-core",action="store_true")
    args=parser.parse_args()
    if args.updates<1 or args.batch_size<1: parser.error("budgets must be positive")
    torch.set_num_threads(1)
    results=[train(seed,args.updates,args.batch_size,args.lr,args.output,args.freeze_core) for seed in args.seeds]
    summary=[{"seed":r["seed"],"initial_goodness":r["initial"]["expected_goodness"],
              "final_goodness":r["final"]["expected_goodness"],"accuracy":r["final"]["accuracy"]} for r in results]
    (Path(args.output)/("summary_freeze_core.json" if args.freeze_core else "summary.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__=="__main__": main()
