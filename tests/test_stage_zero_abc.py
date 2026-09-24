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


def test_frozen_abc_has_no_perturbation_trace_or_parameter_update():
    model = StageZeroABC(
        11,
    )

    model.reset_phase(
        False
    )

    before = {
        id(parameter):
        parameter.clone()
        for parameter
        in model.plasticity
        .parameters
        .values()
    }

    action_generator = (
        torch.Generator()
        .manual_seed(19)
    )

    model.episode(
        phase="frozen",
        episode=0,
        target=0,
        delay_ms=0,
        start_ms=0,
        action_generator=
            action_generator,
    )

    assert all(
        torch.equal(
            block.z,
            block.z_bar,
        )
        for block
        in model.core.blocks
    )

    assert all(
        trace.count_nonzero()
        == 0
        for trace
        in model.plasticity
        .traces
        .values()
    )

    assert all(
        torch.equal(
            parameter,
            before[
                id(parameter)
            ],
        )
        for parameter
        in model.plasticity
        .parameters
        .values()
    )


def test_training_abc_generates_nonzero_eligibility_without_autograd():
    model = StageZeroABC(
        11,
    )

    model.reset_phase(
        True
    )

    action_generator = (
        torch.Generator()
        .manual_seed(23)
    )

    row = model.episode(
        phase="training",
        episode=0,
        target=1,
        delay_ms=0,
        start_ms=0,
        action_generator=
            action_generator,
    )

    assert (
        row[
            "pre_goodness_eligibility_trace"
        ]["total"]["l2"]
        > 0
    )

    assert all(
        parameter.grad is None
        for parameter
        in model.parameters()
    )

    assert (
        row["nan_count"]
        == 0
    )

    assert (
        row["inf_count"]
        == 0
    )