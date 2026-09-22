"""Stage Zero boundary and actual ACNT learning-path integration tests.

The positive reward below is an explicitly synthetic wiring test. It does not
replace the fractional environment Goodness used by the experiment runner.
"""

import builtins
import inspect
import json
import math
import os

import pytest
import torch

from torch.nn import functional as F

import acnt.adapters as adapters
from acnt.control import sample_discrete
from acnt.mechanical import MechanicalLog
from acnt.plasticity import EligibilityBank, Plasticity
from acnt.runtime import Runtime
import experiments.stage_zero.harness as harness
from experiments.stage_zero.environment import VisualEnvironment
from experiments.stage_zero.harness import Protocol, StageZero
from experiments.stage_zero.runner import run_episode


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    return StageZero(17)


@pytest.fixture
def frame():
    return VisualEnvironment().render(2)


def _generator(device="cpu"):
    return torch.Generator(device=device).manual_seed(431)


def _forbidden(*args, **kwargs):
    raise AssertionError("Stage Zero entered a forbidden path")


def _assert_finite(model):
    assert model.check_devices()
    assert torch.isfinite(model.parameter_vector()).all()
    assert torch.isfinite(model.eligibility_vector()).all()
    assert all(torch.isfinite(t).all() for t in model.buffers())
    assert all(torch.isfinite(t).all() for b in model.core.blocks for t in b.A)


def test_stage_zero_dimensions_and_canonical_components(model):
    assert len(model.core.blocks) == 10
    assert all(block.neuron_size == 100 for block in model.core.blocks)
    assert model.core.active_ids == tuple(range(10))
    assert model.organ_blocks == {"eye": 0, "hand": 1}
    assert model.organ_blocks["eye"] != model.organ_blocks["hand"]
    assert set(model.adapters) == {"eye", "hand"}
    assert isinstance(model.adapters["eye"], adapters.EyeAdapter)
    assert isinstance(model.adapters["hand"], adapters.HandAdapter)
    assert model.adapters["hand"].discrete_controls == 27
    assert model.adapters["hand"].continuous_controls == 0
    assert [b.o is not None for b in model.core.blocks] == [True] + [False] * 9
    assert StageZero.encode_readin is Runtime.encode_readin
    assert StageZero.decode_readout is Runtime.decode_readout
    assert StageZero._observe_discrete is Runtime._observe_discrete
    assert harness.sample_discrete is sample_discrete
    assert isinstance(model.plasticity, Plasticity)
    _assert_finite(model)


def test_only_eye_hand_path_runs_and_never_injects_physical_input(monkeypatch, frame):
    for name in ("step", "generate_route", "generate_speak", "generate_goodness",
                 "generate_hand", "_emit", "__init__"):
        monkeypatch.setattr(Runtime, name, _forbidden)
    for cls in (adapters.EarAdapter, adapters.RouteAdapter,
                adapters.SpeakAdapter, adapters.GoodnessAdapter):
        monkeypatch.setattr(cls, "forward", _forbidden)
        monkeypatch.setattr(cls, "__init__", _forbidden)
    monkeypatch.setattr(MechanicalLog, "append", _forbidden)
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"pyautogui", "pynput", "keyboard", "mouse",
                                  "win32api", "win32gui", "pydirectinput"}:
            _forbidden()
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    # Also intercept native Windows executor construction, if introduced later.
    import ctypes
    for name in ("WinDLL", "windll"):
        if hasattr(ctypes, name):
            monkeypatch.setattr(ctypes, name, _forbidden)
    model = StageZero(17)
    eye_inputs, hand_inputs = [], []
    eye_hook = model.adapters["eye"].register_forward_pre_hook(
        lambda module, args: eye_inputs.append(args[0]))
    hand_hook = model.adapters["hand"].register_forward_pre_hook(
        lambda module, args: hand_inputs.append(args[0].detach().clone()))
    try:
        action_ms, hand = model.act(frame, start_ms=0, generator=_generator(), learn=True)
        model.deliver_goodness(0.0, now_ms=action_ms + 250)
    finally:
        eye_hook.remove()
        hand_hook.remove()
    assert len(eye_inputs) == 3 and all(t is frame for t in eye_inputs)
    assert all(t.shape == (3, 1080, 1920) for t in eye_inputs)
    assert len(hand_inputs) == 1
    torch.testing.assert_close(hand_inputs[0], model.core.blocks[1].z)
    assert hand.discrete.a.shape == (27,) and hand.discrete.a.dtype == torch.bool
    assert hand.continuous.shape == (0,)
    assert model.core.active_ids == tuple(range(10))
    assert not hasattr(model, "executors") and not hasattr(model, "step")
    assert all(torch.count_nonzero(t) == 0 for bank in model.plasticity.continuous.values()
               for t in bank.values.values())
    _assert_finite(model)


def test_uses_real_independent_logistic_sampling(model, frame, monkeypatch):
    calls = []

    def observe_sample(q, **kwargs):
        signal = sample_discrete(q, **kwargs)
        calls.append(signal)
        return signal

    monkeypatch.setattr(harness, "sample_discrete", observe_sample)
    _, hand = model.act(frame, start_ms=0, generator=_generator(), learn=True)
    assert len(calls) == 1
    expected = sample_discrete(calls[0].q.detach(), tau=0.25, generator=_generator())
    torch.testing.assert_close(hand.discrete.noise, expected.noise, rtol=0, atol=0)
    torch.testing.assert_close(hand.discrete.p, expected.p, rtol=0, atol=0)
    assert torch.equal(hand.discrete.a, expected.a)
    assert torch.equal(hand.discrete.a, hand.discrete.q + hand.discrete.noise > hand.discrete.threshold)
    assert hand.discrete.tau == 0.25
    # Fixed test seed samples several controls, proving the result is not argmax.
    assert hand.discrete.a.sum() > 1
    assert all(not getattr(hand.discrete, name).requires_grad
               for name in ("q", "p", "noise", "a", "threshold"))
    assert model._local_z == {}


def test_no_supervised_loss_optimizer_or_target_enters_network(model, monkeypatch):
    assert "target" not in inspect.signature(model.act).parameters
    for name in ("cross_entropy", "nll_loss", "binary_cross_entropy",
                 "binary_cross_entropy_with_logits", "softmax", "log_softmax"):
        monkeypatch.setattr(F, name, _forbidden)
    for name in ("softmax", "argmax"):
        monkeypatch.setattr(torch, name, _forbidden)
        monkeypatch.setattr(torch.Tensor, name, _forbidden)
    monkeypatch.setattr(torch.distributions.Categorical, "__init__", _forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", _forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", _forbidden)
    monkeypatch.setattr(torch.autograd, "backward", _forbidden)
    deliveries = []
    original_deliver = model.deliver_goodness
    def trace_delivery(value, *, now_ms):
        deliveries.append((value, now_ms))
        return original_deliver(value, now_ms=now_ms)
    monkeypatch.setattr(model, "deliver_goodness", trace_delivery)
    row = run_episode(model, VisualEnvironment(), 2, episode=0, phase_episode=0,
                      phase="training", start_ms=0, generator=_generator(), learn=True)
    bits = json.loads(row["action_bits"])
    assert row["correct_exact_match"] == float(bits[2] and sum(bits) == 1)
    expected = int(bits[2]) / sum(bits) if sum(bits) else 0.0
    assert row["goodness"] == row["fractional_goodness"] == expected
    assert deliveries == [(expected, row["logical_time_ms"] + 250)]
    assert row["delta"] == expected - row["g_bar_before"]
    assert row["parameter_delta_norm"] > 0
    assert row["goodness_delivery_time"] - row["logical_time_ms"] == 250
    assert row["nan_count"] == row["inf_count"] == 0


def test_zero_visual_event_forces_early_update_and_consumes_pulse(model):
    zero = torch.zeros(3, 1080, 1920)
    calls = []
    hook = model.adapters["eye"].register_forward_hook(lambda *_: calls.append(1))
    try:
        model.update_frame(now_ms=0, image=zero)
        eye = model.core.blocks[0]
        assert list(eye.At) == [0]
        early = model.update_frame(now_ms=1, image=zero)
        assert set(early) == {0} and list(eye.At) == [0, 1]
        assert torch.count_nonzero(eye.o) == 0 and not eye.o.requires_grad
        old = eye.z.clone()
        assert model.update_frame(now_ms=2) == {}
        assert list(eye.At) == [0, 1]
        assert torch.equal(old, eye.z) and len(calls) == 2
        model.update_frame(now_ms=251)
        signal = sample_discrete(model.decode_readout("hand"), tau=0.25, generator=_generator())
        model._observe_discrete("hand", signal, now_ms=251)
        # No new image at 251: a consumed pulse cannot regenerate Eye gradients.
        tags = model.plasticity.internal[0].values
        assert all(torch.count_nonzero(t) == 0 for n, t in tags.items() if n.startswith("adapter."))
        assert len(calls) == 2
    finally:
        hook.remove()


def test_committed_old_snapshot_and_order_independence(frame):
    left, right = StageZero(29), StageZero(29)
    for now_ms in (0, 250, 500):
        for model, order in ((left, range(10)), (right, reversed(range(10)))):
            old = model.core.source_snapshot()
            model.update_frame(now_ms=now_ms, image=frame, order=order)
            for block in model.core.blocks[1:]:
                expected = block.b.detach().clone()
                for i, z in old.items():
                    expected += block.W_ij[i].detach() @ z
                torch.testing.assert_close(block.r, expected, rtol=0, atol=0)
        for a, b in zip(left.core.blocks, right.core.blocks):
            for name in ("r", "a", "h", "z"):
                torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
            assert list(a.At) == list(b.At)
    assert left.core.blocks[0].z.norm() > 0
    assert left.core.blocks[1].z.norm() > 0


@pytest.mark.parametrize("offset", [0, 1, 249, 251])
def test_goodness_rejects_wrong_delivery_time_without_update(model, frame, offset):
    action_ms, _ = model.act(frame, start_ms=1000, generator=_generator(), learn=True)
    before = model.parameter_vector()
    trace = model.eligibility_vector()
    with pytest.raises(ValueError, match="250 ms"):
        model.deliver_goodness(1.0, now_ms=action_ms + offset)
    assert torch.equal(before, model.parameter_vector())
    assert torch.equal(trace, model.eligibility_vector())
    assert model.pending == (action_ms, True)


def test_actual_continuous_eprop_delay_decay_update_baseline_and_fixed_feedback(model, frame, monkeypatch):
    control_calls, vjp_calls, goodness_calls = [], [], []
    observe_control = EligibilityBank.observe_control
    observe_vector = EligibilityBank.observe_vector
    apply_goodness = Plasticity.apply_goodness

    def trace_control(self, signal, **kwargs):
        control_calls.append((self, kwargs["now_ms"]))
        return observe_control(self, signal, **kwargs)

    def trace_vjp(self, output, learning_signal, **kwargs):
        vjp_calls.append((self, output.shape, kwargs["now_ms"]))
        return observe_vector(self, output, learning_signal, **kwargs)

    def trace_goodness(self, value, **kwargs):
        goodness_calls.append((value, kwargs["now_ms"]))
        return apply_goodness(self, value, **kwargs)

    monkeypatch.setattr(EligibilityBank, "observe_control", trace_control)
    monkeypatch.setattr(EligibilityBank, "observe_vector", trace_vjp)
    monkeypatch.setattr(Plasticity, "apply_goodness", trace_goodness)
    before = {i: {n: p.detach().clone() for n, p in group.items()}
              for i, group in model.plasticity.groups.items()}
    feedback = {n: t.clone() for n, t in model.named_buffers() if n.startswith("feedback_")}
    assert all(not t.requires_grad for t in feedback.values())
    assert not set(feedback).intersection(dict(model.named_parameters()))
    action_ms, _ = model.act(frame, start_ms=0, generator=_generator(), learn=True)
    assert action_ms == 500
    assert goodness_calls == []
    assert control_calls == [(model.plasticity.control[1], action_ms)]
    assert {id(bank) for bank, shape, _ in vjp_calls if shape == (100,)} == {
        id(bank) for bank in model.plasticity.internal.values()}
    assert all(t == action_ms for _, _, t in vjp_calls)
    traces = [{i: {n: t.clone() for n, t in bank.values.items()}
               for i, bank in channel.items()} for channel in model.plasticity.banks()]
    assert model.eligibility_vector().norm() > 0
    assert sum(t.abs().sum() for n, t in traces[0][0].items() if n.startswith("adapter.")) > 0
    # Discrete adapter eligibility must not leak direct Hand gradients into Block.
    assert all(torch.count_nonzero(t) == 0 for n, t in traces[2][1].items() if n.startswith("block."))
    _assert_finite(model)
    # Synthetic positive scalar tests the existing e-prop wiring independently.
    delta = model.deliver_goodness(0.75, now_ms=action_ms + 250)
    assert goodness_calls == [(0.75, 750)]
    assert delta == 0.25 and model.plasticity.g_bar == pytest.approx(0.525)
    changed = 0
    for block_id, group in model.plasticity.groups.items():
        for name, parameter in group.items():
            expected = before[block_id][name] + 0.001 * 0.25 * sum(
                trace[block_id][name] * math.exp(-1) for trace in traces)
            torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-8)
            changed += not torch.equal(parameter, before[block_id][name])
    assert changed > 0
    for index, channel in enumerate(model.plasticity.banks()):
        for block_id, bank in channel.items():
            assert bank.last_ms == 750
            for name, value in bank.values.items():
                torch.testing.assert_close(value, traces[index][block_id][name] * math.exp(-1) * 0.9)
                assert not value.requires_grad
    for name, value in feedback.items():
        assert torch.equal(value, dict(model.named_buffers())[name])
    assert model.pending is None
    _assert_finite(model)


def test_feedback_cannot_affect_forward_or_action(frame):
    left, right = StageZero(17), StageZero(17)
    with torch.no_grad():
        for name, value in right.named_buffers():
            if name.startswith("feedback_"):
                value.zero_()
    _, a = left.act(frame, start_ms=0, generator=_generator(), learn=True)
    _, b = right.act(frame, start_ms=0, generator=_generator(), learn=True)
    assert torch.equal(a.discrete.q, b.discrete.q)
    assert torch.equal(a.discrete.a, b.discrete.a)
    assert all(torch.equal(x.z, y.z) for x, y in zip(left.core.blocks, right.core.blocks))
    assert sum(t.abs().sum() for bank in left.plasticity.internal.values() for t in bank.values.values()) > 0
    assert all(torch.count_nonzero(t) == 0 for bank in right.plasticity.internal.values() for t in bank.values.values())


def test_no_learning_and_frozen_evaluation_never_update(model, frame, monkeypatch):
    initial = model.parameter_vector()
    action_ms, _ = model.act(frame, start_ms=0, generator=_generator(), learn=False)
    assert model.deliver_goodness(1.0, now_ms=action_ms + 250) is None
    assert torch.equal(initial, model.parameter_vector())
    assert torch.count_nonzero(model.eligibility_vector()) == 0
    assert model.plasticity.g_bar == 0.5
    model.reset_dynamics()
    action_ms, _ = model.act(frame, start_ms=0, generator=_generator(), learn=True)
    model.deliver_goodness(1.0, now_ms=action_ms + 250)
    trained = model.parameter_vector()
    assert not torch.equal(initial, trained)
    baseline = model.plasticity.g_bar
    model.reset_dynamics()
    monkeypatch.setattr(Plasticity, "apply_goodness", _forbidden)
    monkeypatch.setattr(EligibilityBank, "observe_control", _forbidden)
    for index, target in enumerate((0, 25, 26)):
        row = run_episode(model, VisualEnvironment(), target, episode=index,
                          phase_episode=index, phase="frozen", start_ms=index * 1000,
                          generator=_generator(), learn=False)
        assert row["learning_enabled"] is False and row["parameter_delta_norm"] == 0
        assert row["eligibility_norm"] == 0 and row["delta"] is None
        assert row["nan_count"] == row["inf_count"] == 0
        assert torch.equal(trained, model.parameter_vector())
        assert model.plasticity.g_bar == baseline


def test_pending_action_prevents_new_frame_or_reset(model, frame):
    action_ms, _ = model.act(frame, start_ms=0, generator=_generator(), learn=True)
    with pytest.raises(RuntimeError, match="pending goodness"):
        model.update_frame(now_ms=750, image=frame)
    with pytest.raises(RuntimeError, match="undelivered"):
        model.reset_dynamics()
    model.deliver_goodness(0.0, now_ms=action_ms + 250)
    with pytest.raises(RuntimeError, match="no pending"):
        model.deliver_goodness(0.0, now_ms=1000)


def test_all_blocks_must_remain_active(model, frame):
    model.core.set_active(9, False)
    with pytest.raises(RuntimeError, match="all Blocks active"):
        model.update_frame(now_ms=0, image=frame)


@pytest.mark.parametrize("field,value", [("blocks", 9), ("neurons", 10),
                                         ("goodness_delay_ms", 0)])
def test_protocol_rejects_architecture_or_delay_change(field, value):
    with pytest.raises(ValueError, match="Stage Zero fixes"):
        Protocol(**{field: value})


def test_cuda_real_path_smoke():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU Stage Zero integration is covered")
    model = StageZero(17, device="cuda")
    frame = VisualEnvironment().render(26, device=model.device)
    before = model.parameter_vector()
    action_ms, hand = model.act(frame, start_ms=0, generator=_generator(model.device), learn=True)
    assert hand.discrete.q.device == model.device == frame.device
    assert hand.continuous.device == model.device
    assert model.check_devices()
    _assert_finite(model)
    model.deliver_goodness(1.0, now_ms=action_ms + 250)
    assert not torch.equal(before, model.parameter_vector())
    _assert_finite(model)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_goodness_rejects_nonfinite_or_out_of_range_without_update(model, frame, value):
    action_ms, _ = model.act(frame, start_ms=0, generator=_generator(), learn=True)
    before, trace = model.parameter_vector(), model.eligibility_vector()
    with pytest.raises(ValueError, match="finite"):
        model.deliver_goodness(value, now_ms=action_ms + 250)
    assert torch.equal(before, model.parameter_vector())
    assert torch.equal(trace, model.eligibility_vector())
    assert model.pending == (action_ms, True)
