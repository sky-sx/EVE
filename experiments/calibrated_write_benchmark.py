"""Local CPU throughput, not a behavioral or lifelong learning validation.

Each case is a separate process/life. Writes, action credit and scalar feedback
are active. Observer measurements are never supplied to the learner.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch import nn
from experiments.calibrated_write_exploration import ExploringKernel, ExploringLearner


def peak_working_set():
    if sys.platform != 'win32':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage')]
    kernel32, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize


def case(width, threads, warmup, decisions):
    torch.set_num_threads(threads)
    start = time.perf_counter()
    kernel = ExploringKernel(seed=11, width=width, hold=3, group_size=32)
    kernel.body.hand.network[1] = nn.LeakyReLU(.1)
    model = ExploringLearner(kernel, taus_ms=(.1, .1, .1), actor_lr=.00001)
    setup_seconds = time.perf_counter()-start
    rng, generator = random.Random(70011), torch.Generator().manual_seed(10011)
    budget, timings, goods = model.persistent_tensor_bytes, [], []
    for decision in range(warmup+decisions):
        tick = time.perf_counter()
        bit, amplitude = rng.randrange(2), rng.uniform(.5, 1.)
        sign = 2*bit-1
        cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude,
                                      .5*sign*amplitude, -.5*sign*amplitude]])
        for phase in range(4):
            event = 4*decision+phase
            goodness, action = None, None
            if phase == 3:
                action, _, score = model.sample(generator)
                goodness = float(action == bit)
                model.record_action(score, now_ms=event*2)
            model.step(now_ms=event*2, cue=cue if phase == 0 else None,
                       goodness=goodness, action=action)
            assert model.state.grad_fn is None and model.tangent.grad_fn is None
            assert model.persistent_tensor_bytes == budget
        if decision >= warmup:
            timings.append(time.perf_counter()-tick)
            goods.append(goodness)
    mean = sum(timings)/len(timings)
    return dict(width=width, core_neurons=7*width, cpu_threads=threads,
        parameters=kernel.parameter_size, write_parameters=kernel.writer_indices.numel(),
        F=kernel.group_count, persistent_tensor_bytes=budget,
        process_peak_working_set_bytes=peak_working_set(), setup_seconds=setup_seconds,
        warmup_decisions=warmup, measured_decisions=decisions,
        mean_seconds_per_decision=mean, decisions_per_second=1/mean,
        first_half_seconds_per_decision=sum(timings[:decisions//2])/(decisions//2),
        second_half_seconds_per_decision=sum(timings[decisions//2:])/(decisions-decisions//2),
        estimated_hours_100k=mean*100000/3600, estimated_hours_1m=mean*1000000/3600,
        neural_resets=0, events=model.steps, self_written_coordinates=int(
            model.ever_written[kernel.writer_indices].sum()),
        benchmark_goodness=sum(goods)/len(goods), per_decision_seconds=timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', action='store_true')
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--output', default='reports/calibrated_write_benchmark_2026-10-01.json')
    args = parser.parse_args()
    if args.warmup < 0 or args.decisions < 2:
        raise ValueError('nonnegative warmup and at least two measured decisions required')
    if args.case:
        print(json.dumps(case(args.width, args.threads, args.warmup, args.decisions), allow_nan=False))
        return
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    sources = ('acnt/calibrated_write.py', 'acnt/grouped_write_independent.py',
        'acnt/grouped_write.py', 'acnt/address_write.py', 'acnt/self_write.py',
        'acnt/full_write.py', 'acnt/block.py', 'acnt/adapters.py',
        'experiments/calibrated_write_exploration.py', __file__)
    hashes = {str(Path(p)).replace('\\', '/'): hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in sources}
    rows = []
    for width, threads in ((4, 1), (8, 1), (12, 1), (16, 1), (8, 4)):
        completed = subprocess.run([sys.executable, '-m', 'experiments.calibrated_write_benchmark',
            '--case', '--width', str(width), '--threads', str(threads),
            '--warmup', str(args.warmup), '--decisions', str(args.decisions)],
            check=True, capture_output=True, text=True)
        row = json.loads(completed.stdout)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('width', 'cpu_threads', 'parameters',
            'mean_seconds_per_decision', 'estimated_hours_100k',
            'process_peak_working_set_bytes')}), flush=True)
    result = dict(kind='CPU throughput benchmark; not behavioral validation',
        torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
        cuda_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        device='cpu', source_hashes=hashes, rows=rows,
        limitations=['short warmed-up benchmark; long-run hours are extrapolations',
            'peak working set includes interpreter/framework and transient tensors',
            'no GPU benchmark: current learner creates numerical buffers on CPU',
            'no replay or reset within a life; each benchmark case is a separate life',
            'full training active, but benchmark goodness is not a convergence result'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
