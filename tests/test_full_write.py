import pytest
import torch
from torch import nn
from acnt.block import Block
from acnt.full_write import SharedWriteAdapter, FullParameterWriteSurface


def model():
    torch.manual_seed(11)
    root = nn.Module()
    root.block = Block(0, 2, [2], hold_tick=2, nlm_hidden_dim=2)
    root.write = SharedWriteAdapter(2)
    return root, FullParameterWriteSurface(root, root.write)


def test_complete_surface_includes_generator_and_frozen_parameters():
    root, surface = model()
    assert sum(p.numel() for p in root.write.parameters()) == 65
    assert sum(p.numel() for p in root.block.parameters()) == 80
    assert surface.parameter_count == surface.control_count == 145
    root.block.b.requires_grad_(False)
    initial = surface.pack().detach().clone()
    command = torch.linspace(-.01, .01, surface.control_count)
    surface.apply(command)
    torch.testing.assert_close(surface.pack(), initial + command, rtol=0, atol=0)


def test_each_control_has_exactly_one_parameter_effect():
    root, surface = model()
    jacobian = torch.autograd.functional.jacobian(surface.proposed, torch.zeros(surface.control_count))
    torch.testing.assert_close(jacobian, torch.eye(surface.parameter_count), rtol=0, atol=0)
    # Every slot, including Write's own output weights/bias, can be addressed alone.
    for slot in surface.slots:
        initial = surface.pack().detach().clone()
        command = torch.zeros(surface.control_count)
        command[slot.start] = .125
        surface.apply(command)
        torch.testing.assert_close(surface.pack(), initial + command, rtol=0, atol=0)


def test_generate_all_commands_before_self_modifying_generator():
    root, surface = model()
    hidden = torch.tensor([.3, -.5])
    initial = surface.pack().detach().clone()
    command = surface.generate(hidden, chunk_size=7)
    whole = surface.generate(hidden, chunk_size=1000)
    torch.testing.assert_close(command, whole, atol=1e-9, rtol=1e-5)
    writer_grad = torch.autograd.grad(command.sum(), tuple(root.write.parameters()))
    assert sum(float(g.norm()) for g in writer_grad) > 0
    writer_before = torch.cat([p.detach().flatten().clone() for p in root.write.parameters()])
    surface.apply(command)
    torch.testing.assert_close(surface.pack(), initial + command.detach(), rtol=0, atol=0)
    writer_after = torch.cat([p.detach().flatten() for p in root.write.parameters()])
    assert not torch.equal(writer_before, writer_after)


def test_invalid_command_and_new_unregistered_parameter_are_rejected():
    root, surface = model()
    initial = surface.pack().detach().clone()
    command = torch.zeros(surface.control_count)
    command[-1] = float('nan')
    with pytest.raises(ValueError):
        surface.apply(command)
    torch.testing.assert_close(surface.pack(), initial, rtol=0, atol=0)
    root.extra = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match='registry changed'):
        surface.generate(torch.zeros(2))
