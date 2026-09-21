"""CLI orchestration, raw audit logs and frozen evaluation for Stage Zero."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from .environment import ACTIONS, VisualEnvironment, balanced_targets, exact_goodness, action_quality_bucket, TeacherTable, DEFAULT_TEACHER_TABLE
from .evaluation import summarize, learning_curve, classify_evidence, EVIDENCE_CRITERIA
from .harness import Protocol, StageZero


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_hash(tensor):
    """Hash torch serialization; requires no numpy or image dependency."""
    stream = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), stream)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def parameter_summary(model, before):
    result = {}
    total_squared = 0.0
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        delta = value - before[name]
        stats = torch.stack((value.norm(), delta.norm(), delta.abs().max(),
                             delta.min(), delta.max(), (delta != 0).sum())).cpu().tolist()
        result[name] = dict(zip(("norm_after", "delta_norm", "max_abs_delta", "min_delta",
                                "max_delta", "changed_elements"), stats))
        result[name]["elements"] = parameter.numel()
        total_squared += stats[1] ** 2
    return {"delta_norm": total_squared ** 0.5,
            "changed_parameter_tensors": sum(v["changed_elements"] > 0 for v in result.values()),
            "parameter_tensors": len(result), "per_parameter": result}


def stability_and_norms(model, old_parameters):
    parameters = model.parameter_vector()
    eligibility = model.eligibility_vector()
    # Logging reductions only; no tensor here enters learning or decision making.
    state = torch.cat([b.z.flatten() for b in model.core.blocks])
    stats = torch.stack((
        parameters.norm(), (parameters - old_parameters).norm(), eligibility.norm(),
        torch.isnan(parameters).sum() + torch.isnan(eligibility).sum() + torch.isnan(state).sum(),
        torch.isinf(parameters).sum() + torch.isinf(eligibility).sum() + torch.isinf(state).sum(),
    )).cpu().tolist()
    if stats[3] or stats[4]:
        raise FloatingPointError("non-finite parameters, eligibility or states")
    return dict(zip(("parameter_norm", "parameter_delta_norm", "eligibility_norm",
                     "nan_count", "inf_count"), stats))


def run_episode(model, environment, target, *, episode, phase_episode, phase,
                start_ms, generator, learn, teacher_table):
    before = model.parameter_vector()
    # Target remains in the environment; model.act accepts only the RGB image.
    image = environment.render(target, device=model.device)
    action_ms, hand = model.act(image, start_ms=start_ms, generator=generator, learn=learn)
    signal = hand.discrete
    probabilities = signal.p.cpu().tolist()
    tendencies = signal.q.cpu().tolist()
    actions = signal.a.cpu().tolist()
    correct_pressed, wrong_count = action_quality_bucket(target, signal.a)
    goodness = teacher_table.lookup(correct_pressed, wrong_count)
    exact_match = exact_goodness(target, signal.a)
    eligibility_before = float(model.eligibility_vector().norm())
    baseline_before = model.plasticity.g_bar
    delivery_ms = action_ms + model.protocol.goodness_delay_ms
    delta = model.deliver_goodness(goodness, now_ms=delivery_ms)
    # Analytical independent-bit event probability is an audit statistic only.
    event_probability = probabilities[target]
    for i, p in enumerate(probabilities):
        if i != target:
            event_probability *= 1.0 - p
    row = {
        "episode": episode, "phase_episode": phase_episode, "phase": phase,
        "visual_time_ms": start_ms, "logical_time_ms": action_ms,
        "target_class": target, "target_action": ACTIONS[target],
        "sampled_actions": json.dumps([name for name, bit in zip(ACTIONS, actions) if bit]),
        "goodness": goodness, "goodness_delivery_time": delivery_ms,
        "teacher_goodness": goodness, "correct_pressed": correct_pressed,
        "wrong_count": wrong_count, "correct_exact_match": int(exact_match),
        "target_q": tendencies[target], "target_probability": probabilities[target],
        "non_target_probability": sum(p for i, p in enumerate(probabilities) if i != target) / 26,
        "target_bit_correct": int(actions[target]),
        "non_target_false_rate": sum(bit for i, bit in enumerate(actions) if i != target) / 26,
        "active_action_count": sum(actions), "exact_probability": event_probability,
        "eligibility_norm_at_action": eligibility_before,
        "g_bar_before": baseline_before, "g_bar": model.plasticity.g_bar,
        "delta": delta, "learning_enabled": learn,
        "q": json.dumps(tendencies), "p": json.dumps(probabilities),
        "action_bits": json.dumps(actions),
    }
    row.update(stability_and_norms(model, before))
    return row


@torch.no_grad()
def visual_diagnostics(model, environment):
    """Read-only encoding check; never used to choose actions or update weights."""
    encoded = torch.stack([model.adapters["eye"](environment.render(i, device=model.device))
                           for i in range(27)])
    letter_distances = torch.pdist(encoded[:26])
    all_distances = torch.pdist(encoded)
    return {"eye_shape": [3, 1080, 1920], "encoded_shape": list(encoded.shape),
            "finite": bool(torch.isfinite(encoded).all()),
            "minimum_pairwise_letter_encoding_l2": float(letter_distances.min()),
            "minimum_pairwise_all_encoding_l2": float(all_distances.min()),
            "maximum_pairwise_encoding_l2": float(all_distances.max()),
            "meaning": "Distinct encodings are not proof of learnability."}

def run_seed(seed, *, output, device, train_episodes, evaluation_episodes, window, protocol, teacher_table):
    directory = output / f"seed_{seed:03d}"
    directory.mkdir()
    teacher_table.save(directory / "teacher_goodness.json")
    model = StageZero(seed, device=device, protocol=protocol)
    environment = VisualEnvironment()
    write_json(directory / "visual_diagnostics.json", visual_diagnostics(model, environment))
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    feedback_before = {n: t.clone() for n, t in model.named_buffers() if n.startswith("feedback_")}
    initial_hash = tensor_hash(model.parameter_vector())
    phase_summaries, rows, curves = {}, [], []
    episode = 0
    training_seconds = 0.0
    started = time.perf_counter()
    log_handle = (directory / "episode_log.csv").open("w", newline="", encoding="utf-8")
    writer = None
    try:
        for phase_index, (phase, count, learn) in enumerate((
            ("initial", evaluation_episodes, False),
            ("training", train_episodes, True),
            ("frozen", evaluation_episodes, False),
        )):
            model.train(learn)
            model.reset_dynamics()
            frozen_before = model.parameter_vector()
            generator = torch.Generator(device=model.device).manual_seed(seed * 10000 + 1000 + phase_index)
            sequence_seed = seed * 10000 + 2000 + phase_index
            phase_rows = []
            phase_started = time.perf_counter()
            for index, target in enumerate(balanced_targets(count, sequence_seed)):
                row = run_episode(
                    model, environment, target, episode=episode, phase_episode=index,
                    phase=phase, start_ms=episode * protocol.episode_interval_ms,
                    generator=generator, learn=learn, teacher_table=teacher_table,
                )
                if writer is None:
                    writer = csv.DictWriter(log_handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                log_handle.flush()
                phase_rows.append(row)
                rows.append(row)
                episode += 1
                if (index + 1) % 27 == 0 or index + 1 == count:
                    print(json.dumps({"seed": seed, "phase": phase, "episodes": index + 1,
                                      "total": count, "successes": sum(r["correct_exact_match"] for r in phase_rows),
                                      "elapsed_seconds": round(time.perf_counter() - phase_started, 2)}), flush=True)
            elapsed = time.perf_counter() - phase_started
            unchanged = torch.equal(frozen_before, model.parameter_vector())
            if not learn and not unchanged:
                raise AssertionError("evaluation changed parameters")
            phase_summaries[phase] = summarize(phase_rows)
            phase_summaries[phase].update({"seconds": elapsed, "parameters_unchanged": unchanged,
                                           "sequence_seed": sequence_seed,
                                           "action_seed": seed * 10000 + 1000 + phase_index})
            curves.extend(learning_curve(phase_rows, window))
            if learn:
                training_seconds = elapsed
    finally:
        log_handle.close()
    changes = parameter_summary(model, initial)
    feedback_unchanged = all(torch.equal(t, dict(model.named_buffers())[n]) for n, t in feedback_before.items())
    if not feedback_unchanged or not model.check_devices():
        raise AssertionError("fixed feedback or device invariant failed")
    phase_summaries.update({"teacher_table_sha256": teacher_table.sha256, "seed": seed, "device": str(model.device),
                           "training_seconds": training_seconds,
                           "total_seconds": time.perf_counter() - started,
                           "initial_parameter_hash": initial_hash,
                           "final_parameter_hash": tensor_hash(model.parameter_vector()),
                           "parameter_delta_norm": changes["delta_norm"],
                           "feedback_unchanged": feedback_unchanged,
                           "all_tensors_on_device": model.check_devices()})
    write_json(directory / "summary.json", phase_summaries)
    write_json(directory / "parameter_summary.json", changes)
    write_csv(directory / "learning_curve.csv", curves)
    class_rows = [{"phase": phase, **row} for phase in ("initial", "training", "frozen")
                  for row in phase_summaries[phase]["per_class"]]
    write_csv(directory / "per_class.csv", class_rows)
    torch.save({"state_dict": model.state_dict(), "protocol": asdict(protocol),
                "g_bar": model.plasticity.g_bar, "seed": seed}, directory / "final_state.pt")
    return phase_summaries


def source_metadata():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    paths = [*sorted((root / "acnt").glob("*.py")),
             *sorted((root / "experiments" / "stage_zero").glob("*.py")),
             *sorted((root / "tests").glob("test_stage_zero*.py")),
             root / "experiments" / "stage_zero" / "DESIGN.md"]
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACNT Stage Zero frozen Teacher delayed scalar experiment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-table", type=Path, default=DEFAULT_TEACHER_TABLE)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--train-episodes", type=int, default=2700)
    parser.add_argument("--evaluation-episodes", type=int, default=270)
    parser.add_argument("--window", type=int, default=270)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if any(v < 1 for v in (args.train_episodes, args.evaluation_episodes, args.window, args.threads)):
        parser.error("episode counts, window and threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    if args.train_episodes % 27 or args.evaluation_episodes % 27:
        parser.error("phase counts must be multiples of 27 for exactly balanced sampling")
    if args.output.exists():
        parser.error("output must be a new directory; existing data are never overwritten")
    teacher_table = TeacherTable.load(args.teacher_table)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    teacher_table.save(args.output / "teacher_goodness.json")
    (args.output / "teacher_goodness.sha256").write_text(teacher_table.sha256 + "\n", encoding="ascii")
    write_json(args.output / "teacher_calibration_metadata.json", teacher_table.metadata)
    protocol = Protocol()
    config = {**asdict(protocol), "seeds": args.seeds, "training_episodes": args.train_episodes,
              "evaluation_episodes": args.evaluation_episodes, "window": args.window,
              "teacher_table_sha256": teacher_table.sha256,
              "threads": args.threads, "device": device, "threshold": 0.0,
              "parameter_clip": None, "dtype": "float32", "deterministic_algorithms": True}
    write_json(args.output / "config.json", config)
    environment = VisualEnvironment()
    environment_config = environment.config()
    environment_config["image_sha256_torch_serialization"] = {
        action: tensor_hash(environment.render(i)) for i, action in enumerate(ACTIONS)
    }
    write_json(args.output / "environment_config.json", environment_config)
    metadata = {**source_metadata(), "date_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
                "seeds": args.seeds, "number_of_blocks": protocol.blocks,
                "neurons_per_block": protocol.neurons, "goodness_delay_ms": protocol.goodness_delay_ms,
                "learning_rate": protocol.learning_rate,
                "eligibility_settings": {"tau_seconds": protocol.ticktime, "rho": protocol.rho,
                                         "ema_alpha": protocol.ema_alpha, "initial_g_bar": protocol.initial_g_bar},
                "number_of_training_episodes_per_seed": args.train_episodes,
                "number_of_evaluation_episodes_per_phase_per_seed": args.evaluation_episodes,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "evidence_criteria": EVIDENCE_CRITERIA,
                "teacher_table_sha256": teacher_table.sha256, "teacher_calibration": teacher_table.metadata,
                "completed": False}
    write_json(args.output / "run_metadata.json", metadata)
    summaries = []
    started = time.perf_counter()
    for seed in args.seeds:
        summaries.append(run_seed(seed, output=args.output, device=device,
                                  train_episodes=args.train_episodes,
                                  evaluation_episodes=args.evaluation_episodes,
                                  window=args.window, protocol=protocol, teacher_table=teacher_table))
    aggregate = [{"seed": s["seed"], "phase": phase,
                  **{k: v for k, v in s[phase].items() if not isinstance(v, (dict, list))},
                  "parameter_delta_norm": s["parameter_delta_norm"]}
                 for s in summaries for phase in ("initial", "training", "frozen")]
    write_csv(args.output / "aggregate.csv", aggregate)
    conclusion = classify_evidence(summaries)
    metadata.update({"completed": True, "elapsed_seconds": time.perf_counter() - started,
                     "training_seconds": sum(s["training_seconds"] for s in summaries),
                     "conclusion": conclusion})
    write_json(args.output / "run_metadata.json", metadata)
    write_json(args.output / "aggregate.json", {"conclusion": conclusion, "seeds": summaries})
    (args.output / "report.md").write_text(
        f"# Stage Zero raw run\n\nConclusion: **{conclusion}** under this protocol.\n\n"
        f"Base commit: `{metadata['git_commit']}`; inspect source hashes for experiment code.\n\n"
        "Full raw episodes, per-class results and windowed curves are retained for every seed.\n\n"
        "| Seed | Initial exact | Training exact | Frozen exact | Target p initial/frozen | Non-target p initial/frozen | Parameter delta norm |\n"
        "|---|---|---|---|---|---|---|\n" + "".join(
            f"| {s['seed']} | {s['initial']['exact_success_rate']:.6g} | {s['training']['exact_success_rate']:.6g} | "
            f"{s['frozen']['exact_success_rate']:.6g} | {s['initial']['target_probability']:.6g}/{s['frozen']['target_probability']:.6g} | "
            f"{s['initial']['non_target_probability']:.6g}/{s['frozen']['non_target_probability']:.6g} | {s['parameter_delta_norm']:.6g} |\n"
            for s in summaries) + "\n| Seed | Phase | Exact success | Target p | Non-target p | Mean Teacher goodness |\n"
        "|---|---|---|---|---|---|\n" + "".join(
            f"| {s['seed']} | {phase} | {s[phase]['exact_success_rate']:.9g} | "
            f"{s[phase]['target_probability']:.9g} | {s[phase]['non_target_probability']:.9g} | "
            f"{s[phase]['mean_goodness']:.9g} |\n"
            for s in summaries for phase in ("initial", "training", "frozen")), encoding="utf-8")
    print(json.dumps({"completed": True, "conclusion": conclusion, "output": str(args.output),
                      "seconds": metadata["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
