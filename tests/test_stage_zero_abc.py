import torch

from experiments.stage_zero_abc.environment import (
    ACTIONS,
    CLASS_COUNT,
    DELAYS_MS,
    balanced_targets,
    encode_target,
    exact_success,
    phase_schedule,
    potential_goodness,
)

from experiments.stage_zero_abc.harness import (
    StageZeroABC,
)


def test_abc_environment_is_exactly_three_classes():
    assert ACTIONS == (
        "A",
        "B",
        "C",
    )

    assert CLASS_COUNT == 3

    assert DELAYS_MS == (0,)


def test_direct_readin_encoding_is_deterministic_and_one_hot():
    for target in range(3):
        o = encode_target(
            target,
            neuron_size=10,
        )

        assert o.shape == (20,)
        assert o.dtype == torch.float32

        assert o[:10].sum().item() == 1.0
        assert o[target].item() == 1.0

        expected = torch.zeros(10)
        expected[target] = 1.0

        torch.testing.assert_close(
            o[:10],
            expected,
            rtol=0,
            atol=0,
        )

        torch.testing.assert_close(
            o[10:],
            torch.zeros(10),
            rtol=0,
            atol=0,
        )


def test_goodness_is_single_scalar_only():
    target = 0

    action = torch.tensor(
        [False, False, False]
    )

    assert potential_goodness(
        target,
        action,
    ) == 0.0

    action = torch.tensor(
        [True, False, False]
    )

    assert potential_goodness(
        target,
        action,
    ) == 1.0

    assert exact_success(
        target,
        action,
    )

    action = torch.tensor(
        [True, True, False]
    )

    assert potential_goodness(
        target,
        action,
    ) == 0.5

    assert not exact_success(
        target,
        action,
    )

    action = torch.tensor(
        [True, True, True]
    )

    assert potential_goodness(
        target,
        action,
    ) == 1.0 / 3.0

    assert not exact_success(
        target,
        action,
    )


def test_balanced_abc_target_schedule():
    import random

    targets = balanced_targets(
        300,
        random.Random(7),
    )

    assert len(targets) == 300

    assert all(
        targets.count(target) == 100
        for target in range(3)
    )


def test_phase_schedule_uses_independent_target_and_hand_seeds():
    schedule, streams = phase_schedule(
        300,
        11,
        1,
    )

    assert len(schedule) == 300

    assert streams["target"] != streams["hand"]

    assert all(
        delay == 0
        for _, delay
        in schedule
    )


def test_abc_harness_is_exactly_two_blocks_of_ten_neurons():
    model = StageZeroABC(
        11,
    )

    assert len(
        model.core.blocks
    ) == 2

    assert all(
        block.neuron_size == 10
        for block
        in model.core.blocks
    )

    assert all(
        block.hold_tick == 4
        for block
        in model.core.blocks
    )

    assert all(
        block.ticktime == 250
        for block
        in model.core.blocks
    )

    assert (
        model.core.blocks[0].o
        is not None
    )

    assert (
        model.core.blocks[1].o
        is None
    )

    assert (
        model.hand.discrete_controls
        == 3
    )

    assert (
        model.hand.continuous_controls
        == 0
    )


def test_abc_has_no_trainable_input_adapter():
    model = StageZeroABC(
        11,
    )

    block0_group = (
        model.groups[0]
    )

    assert all(
        not name.startswith(
            "adapter."
        )
        for name
        in block0_group
    )


def test_frozen_abc_preserves_parameters_directions_and_rng():
    model = StageZeroABC(11)
    model.reset_phase(False)
    before = {k:p.clone() for k,p in model.plasticity.parameters.items()}
    delta = {k:d.clone() for k,d in model.plasticity.delta_w.items()}
    rng = model.plasticity.rng.getstate()
    model.episode(phase="frozen", episode=0, target=0, delay_ms=0, start_ms=0,
                  action_generator=torch.Generator().manual_seed(19))
    assert model.plasticity.pending_indices is None
    assert model.plasticity.rng.getstate() == rng
    assert all(torch.equal(p,before[k]) for k,p in model.plasticity.parameters.items())
    assert all(torch.equal(d,delta[k]) for k,d in model.plasticity.delta_w.items())


def test_training_abc_moves_only_after_first_goodness_without_autograd():
    model = StageZeroABC(11)
    model.reset_phase(True)
    generator = torch.Generator().manual_seed(23)
    for episode in range(3):
        row = model.episode(phase="training", episode=episode, target=1, delay_ms=0,
                            start_ms=episode*750, action_generator=generator)
        assert row["selected_parameter_count"] == (0 if episode == 0 else 3)
        assert row["parameter_delta_norm"] == 0 if episode == 0 else row["parameter_delta_norm"] > 0
        assert row["nan_count"] == row["inf_count"] == 0
    assert all(p.grad is None for p in model.parameters())
