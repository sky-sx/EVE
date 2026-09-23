import pytest
import torch

from acnt import Block, Core, Runtime
from acnt.adapters import EarAdapter, EyeAdapter, GoodnessAdapter, HandAdapter, RouteAdapter, SpeakAdapter
from acnt.runtime import ORGANS


@pytest.fixture
def make_runtime():
    def create(*, neuron_size=3, block_count=8, ticktime=250, continuous_controls=2):
        torch.manual_seed(73)
        sizes = [neuron_size] * block_count
        core = Core([Block(i, neuron_size, sizes, readin=i < 2, ticktime=ticktime) for i in range(block_count)])
        adapters = {
            "eye": EyeAdapter(neuron_size),
            "ear": EarAdapter(neuron_size, window_samples=32),
            "hand": HandAdapter(neuron_size, continuous_controls=continuous_controls),
            "speak": SpeakAdapter(neuron_size),
            "goodness": GoodnessAdapter(neuron_size),
            "route": RouteAdapter(neuron_size, max_blocks=block_count),
        }
        return Runtime(core, adapters, {name: i for i, name in enumerate(ORGANS)})
    return create
