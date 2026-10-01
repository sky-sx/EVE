"""A wall-clock-limited, uninterrupted fixed-cue life.

Observer logs and the final checkpoint never feed the subject. No task switch,
evaluation reset, replay, or optimizer outside the self-write actuator occurs.
Wall-clock duration and synthetic monotonic event time are reported separately.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from experiments.calibrated_write_exploration import ExploringKernel, ExploringLearner
from experiments.calibrated_write_benchmark import peak_working_set


def stats(rows):
    if not rows:
        return dict(count=0, goodness=None, expected_goodness=None)
    return dict(count=len(rows), goodness=sum(r['goodness'] for r in rows)/len(rows),
        expected_goodness=sum(r['expected_goodness'] for r in rows)/len(rows))


def class_stats(rows):
    return {name: stats([r for r in rows if r['target'] == bit])
            for bit, name in ((0, 'A'), (1, 'B'))}


def run(*, seconds=600., seed=44, width=4, threads=1, output, self_evaluate=False):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite duration required')
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    if self_evaluate:
        from acnt.goodness_prediction import GoodnessPredictionKernel, GoodnessPredictionLearner
        kernel = GoodnessPredictionKernel(seed=seed, width=width, hold=3, group_size=32)
        model = GoodnessPredictionLearner(kernel, taus_ms=(.1, .1, .1), actor_lr=.00001)
    else:
        kernel = ExploringKernel(seed=seed, width=width, hold=3, group_size=32)
        kernel.body.hand.network[1] = nn.LeakyReLU(.1)
        model = ExploringLearner(kernel, taus_ms=(.1, .1, .1), actor_lr=.00001)
    rng, generator = random.Random(seed+70000), torch.Generator().manual_seed(seed+10000)
    initial = kernel.effective(model.state).detach().clone()
    sources = ('acnt/calibrated_write.py', 'acnt/grouped_write_independent.py',
        'acnt/grouped_write.py', 'acnt/address_write.py', 'acnt/self_write.py',
        'acnt/full_write.py', 'acnt/block.py', 'acnt/adapters.py',
        'experiments/calibrated_write_exploration.py',
        'experiments/calibrated_write_benchmark.py', __file__)
    if self_evaluate:
        sources += ('acnt/goodness_prediction.py',)
    hashes = {}
    for name in sources:
        source = Path(name).resolve()
        relative = source.relative_to(Path.cwd())
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        saved = folder/'source_snapshot'/relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, saved)
    config = dict(seed=seed, width=width, ordinary_blocks=7, core_neurons=7*width,
        task='fixed_cue_and_goodness' if self_evaluate else 'fixed_cue',
        mapping='A(negative signal)->0; B(positive signal)->1',
        stimulus='[s*a, -s*a, 0.5*s*a, -0.5*s*a], a uniform [0.5,1]',
        target_input_distribution='independent fair coin, randomized amplitude',
        requested_wall_seconds=seconds, cpu_threads=threads, device='cpu',
        total_parameters=kernel.parameter_size, writer_parameters=kernel.writer_indices.numel(),
        F=kernel.group_count, group_size=32, taus_ms=[.1, .1, .1],
        actor_lr=.00001, head_lr=.01, control_lr=.003,
        hand_activation='LeakyReLU(0.1)', logit_confidence_limit=3.,
        wall_started_at=datetime.now().astimezone().isoformat(),
        subject_reset_count=0, replay_count=0, outer_parameter_updates=0,
        source_hashes=hashes)
    if self_evaluate:
        config.update(goodness_adapter_parameters=sum(p.numel() for p in kernel.body.goodness.parameters()),
            goodness_source_block=5, goodness_outputs=2, goodness_lr=.01,
            evaluation_core_lr=.001, evaluation_weight=1.,
            actor_modulator='teacher_goodness_only',
            prediction_timing='after action selection, before teacher feedback',
            evaluation_credit='selected-logit BCE credit cached before feedback; exponential traces',
            evaluation_update_region='ordinary Block 5 and Goodness Adapter, positive Write-controlled gain',
            seed_comparison=f'new seed {seed} life, not a controlled comparison with prior seed 44')
    (folder/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    budget, rows = model.persistent_tensor_bytes, []
    max_neural, max_parameter, max_tangent = 0., 0., 0.
    started, next_progress = time.perf_counter(), 30.
    print(json.dumps(dict(status='started', **{k: config[k] for k in
        ('seed', 'total_parameters', 'F', 'requested_wall_seconds')})), flush=True)
    with (folder/'observer.jsonl').open('x', encoding='utf-8') as observer:
        while not rows or time.perf_counter()-started < seconds:
            bit, amplitude = rng.randrange(2), rng.uniform(.5, 1.)
            sign = 2*bit-1
            cue = model.state.new_tensor([[sign*amplitude, -sign*amplitude,
                                         .5*sign*amplitude, -.5*sign*amplitude]])
            for phase in range(4):
                event = 4*len(rows)+phase
                action, goodness = None, None
                if phase == 3:
                    action, probability, score = model.sample(generator)
                    model.record_action(score, now_ms=event*2)
                    judgment = model.judge(action, now_ms=event*2) if self_evaluate else {}
                    goodness = float(action == bit)
                    row = dict(decision=len(rows)+1, target=bit, cue_class='B' if bit else 'A',
                        action=action, probability_action_1=probability, goodness=goodness,
                        expected_goodness=probability if bit else 1-probability,
                        temperature=float(kernel.temperature))
                    row.update(judgment)
                    if self_evaluate:
                        row['prediction_squared_error'] = (judgment['predicted_goodness']-goodness)**2
                        row['prediction_correct'] = int((judgment['predicted_goodness'] >= .5) == bool(goodness))
                    assert .0474 <= probability <= .9526
                diagnostics = model.step(now_ms=event*2, cue=cue if phase == 0 else None,
                                         goodness=goodness, action=action)
                assert model.state.grad_fn is None and model.tangent.grad_fn is None
                assert budget == model.persistent_tensor_bytes
            row.update(diagnostics)
            row['elapsed_wall_seconds'] = time.perf_counter()-started
            rows.append(row)
            observer.write(json.dumps(row, allow_nan=False)+'\n')
            if len(rows)%100 == 0:
                max_neural = max(max_neural, float(model.state[:kernel.neural_size].abs().max()))
                max_parameter = max(max_parameter, float(kernel.effective(model.state).detach().abs().max()))
                max_tangent = max(max_tangent, float(model.tangent.norm()))
            elapsed = time.perf_counter()-started
            if elapsed >= next_progress:
                observer.flush()
                progress = dict(status='running', elapsed_wall_seconds=elapsed,
                    decisions=len(rows), last_200=stats(rows[-200:]),
                    last_200_classes=class_stats(rows[-200:]),
                    persistent_tensor_bytes=budget, tangent_clips=model.tangent_clips)
                if self_evaluate:
                    recent = rows[-200:]
                    progress['prediction_mse_last_200'] = sum(r['prediction_squared_error'] for r in recent)/len(recent)
                print(json.dumps(progress), flush=True)
                next_progress += 30.
    elapsed = time.perf_counter()-started
    assert model.steps == 4*len(rows) and model.last_time == 2*(model.steps-1)
    assert torch.isfinite(model.state).all() and torch.isfinite(model.tangent).all()
    max_neural = max(max_neural, float(model.state[:kernel.neural_size].abs().max()))
    max_parameter = max(max_parameter, float(kernel.effective(model.state).detach().abs().max()))
    max_tangent = max(max_tangent, float(model.tangent.norm()))
    checkpoint = dict(config=config, kernel_state=kernel.state_dict(),
        learner_tensors={name: getattr(model, name) for name in
            ('state', 'tangent', 'traces', 'moment', 'square', 'rates', 'stats', 'ever_written')},
        learner_scalars={name: getattr(model, name) for name in ('stats_count', 'times',
            'last_time', 'trace_time', 'steps', 'updates', 'tangent_clips', 'update_clips')},
        initial_effective_parameters=initial, environment_rng_state=rng.getstate(),
        action_rng_state=generator.get_state(), completed_decisions=len(rows))
    if self_evaluate:
        checkpoint['learner_tensors'].update(evaluation_traces=model.evaluation_traces,
                                             evaluation_stats=model.evaluation_stats)
        checkpoint['learner_scalars'].update(evaluation_stats_count=model.evaluation_stats_count,
            pending_prediction=model.pending_prediction, teacher_for_commit=model.teacher_for_commit,
            auxiliary_direction_norm=model.auxiliary_direction_norm, evaluation_weight=model.evaluation_weight)
    torch.save(checkpoint, folder/'final_checkpoint.pt')
    windows = [dict(first_decision=i+1, last_decision=min(i+100, len(rows)),
        elapsed_wall_seconds=rows[min(i+100, len(rows))-1]['elapsed_wall_seconds'],
        **stats(rows[i:i+100])) for i in range(0, len(rows), 100)]
    tail = rows[-1000:]
    tail_windows = [stats(tail[i:i+200]) for i in range(0, len(tail), 200)]
    plateau = len(tail)==1000 and len(tail_windows)==5 and (
        max(w['goodness'] for w in tail_windows)-min(w['goodness'] for w in tail_windows) <= .05
        and abs(tail_windows[-1]['expected_goodness']-tail_windows[0]['expected_goodness']) <= .01)
    result = dict(**config, observed_wall_seconds=elapsed, decisions=len(rows),
        forward_events=model.steps, last_event_synthetic_ms=model.last_time,
        first_200=stats(rows[:200]), last_200=stats(rows[-200:]), last_1000=stats(tail),
        all=stats(rows), last_1000_classes=class_stats(tail), all_classes=class_stats(rows),
        terminal_200_decision_windows=tail_windows,
        observed_terminal_plateau=plateau,
        plateau_rule='last 1000, five 200-decision windows: sampled G range <=0.05, expected G endpoint change <=0.01',
        written_coordinates=int(model.ever_written.sum()),
        self_written_coordinates=int(model.ever_written[kernel.writer_indices].sum()),
        parameter_change_norm=float((kernel.effective(model.state).detach()-initial).norm()),
        persistent_state_and_training_tensor_bytes=budget,
        process_peak_working_set_bytes=peak_working_set(),
        tangent_clips=model.tangent_clips, update_clips=model.update_clips,
        sampled_state_maxima=dict(neural_abs=max_neural, parameter_abs=max_parameter,
                                 tangent_norm=max_tangent, interval_decisions=100),
        final_finite=True, windows=windows,
        limitations=['one new life and one fixed immediate-feedback synthetic task',
            '600s is compute wall time; event timestamps advance synthetically by 2ms, not real-world paced time',
            'no task reversal, real audio input, delayed reward, or lifelong retention validation',
            'finite-window plateau observation is not mathematical convergence',
            'conditional temperature score and stopped write/statistic derivatives remain',
            'prescribed calibration loss and extra finite J/traces/moments/statistics remain',
            'observer logs and final checkpoint are never read by the learning subject'])
    path = folder/'result.json'
    if self_evaluate:
        def prediction_stats(part):
            if not part:
                return dict(count=0, mse=None, accuracy=None)
            return dict(count=len(part), mse=sum(r['prediction_squared_error'] for r in part)/len(part),
                accuracy=sum(r['prediction_correct'] for r in part)/len(part),
                counterfactual_pair_mse=sum(((r['goodness_if_0']-(r['target']==0))**2+
                    (r['goodness_if_1']-(r['target']==1))**2)/2 for r in part)/len(part),
                prediction_mean=sum(r['predicted_goodness'] for r in part)/len(part))
        result['evaluation'] = dict(first_200=prediction_stats(rows[:200]),
            last_200=prediction_stats(rows[-200:]), last_1000=prediction_stats(tail),
            all=prediction_stats(rows),
            last_1000_by_teacher={str(g):prediction_stats([r for r in tail if r['goodness']==g]) for g in (0,1)},
            last_1000_matrix={f'{cue}_action_{a}':prediction_stats([r for r in tail if r['cue_class']==cue and r['action']==a])
                             for cue in ('A','B') for a in (0,1)})
        result['limitations'] += ['joint task changes Block 5 credit/rate and Write budget; not an isolated adapter-only ablation',
            'counterfactual pair scores are observer-only; only actual-action teacher supervised the learner',
            'selected BCE gradient is prescribed; prediction never replaces system goodness']
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status='completed', result=str(path), observed_wall_seconds=elapsed,
        decisions=len(rows), last_1000=result['last_1000'], last_1000_classes=result['last_1000_classes'],
        observed_terminal_plateau=plateau, self_written_coordinates=result['self_written_coordinates'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--self-evaluate', action='store_true')
    parser.add_argument('--output', default='runs/calibrated_write_duration/fixed_cue_seed_44_600s_v1')
    args = parser.parse_args()
    run(seconds=args.seconds, seed=args.seed, width=args.width, threads=args.threads,
        output=args.output, self_evaluate=args.self_evaluate)


if __name__ == '__main__':
    main()
