"""Snapshot-based Block scheduler; no organ or adapter behavior."""

from collections.abc import Mapping, Sequence
from time import monotonic_ns
from torch import Tensor, nn

from .block import Block


class Core(nn.Module):
    """An existing Block stays in this collection even while inactive.

    Every update in one step reads the same committed source snapshot,
    including self-edges. New states are visible in the next step.
    """

    def __init__(self, blocks: Sequence[Block]) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("Core requires at least one Block")
        sizes = tuple(block.neuron_size for block in blocks)
        for index, block in enumerate(blocks):
            if block.block_id != index:
                raise ValueError("Block ids must be contiguous and in order")
            if block.source_sizes != sizes:
                raise ValueError("every Block must declare all source neuron sizes")
            if block.b.device != blocks[0].b.device:
                raise ValueError("all Blocks must use the same device")
        self.blocks = nn.ModuleList(blocks)

    def _check_id(self, block_id: int) -> None:
        if type(block_id) is not int or not 0 <= block_id < len(self.blocks):
            raise ValueError("block_id must index an existing Block")

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(block.block_id for block in self.blocks if block.active)

    def set_active(self, block_id: int, active: bool) -> None:
        """Change participation without deleting the Block or its history."""
        self._check_id(block_id)
        if type(active) is not bool:
            raise TypeError("active must be bool")
        self.blocks[block_id].active = active

    def set_active_mask(self, mask: Sequence[bool]) -> None:
        """Apply one full mask; validate it before changing any Block."""
        values = list(mask)
        if len(values) != len(self.blocks) or any(type(value) is not bool for value in values):
            raise ValueError("mask must contain one bool for each existing Block")
        for block, active in zip(self.blocks, values):
            block.active = active

    def is_due(self, block_id: int, now_ms: int) -> bool:
        """ticktime is a positive interval in ms, timestamps are also in ms."""
        self._check_id(block_id)
        if type(now_ms) is not int:
            raise ValueError("now_ms must be integer milliseconds")
        block = self.blocks[block_id]
        if not block.At:
            return True
        elapsed_ms = now_ms - block.At[-1]
        if elapsed_ms < 0:
            raise ValueError("time must not move backwards")
        return elapsed_ms >= block.ticktime

    def source_snapshot(self) -> dict[int, Tensor]:
        """Detached old states, captured once for the whole scheduling event."""
        return {block.block_id: block.z.detach().clone() for block in self.blocks if block.active}

    def update_block(self, block_id: int, *, now_ms: int | None = None, force: bool = False, source_snapshot: Mapping[int, Tensor] | None = None) -> Tensor:
        """Update a due active Block; ReadIn may later force one early update."""
        self._check_id(block_id)
        block = self.blocks[block_id]
        check_ms = monotonic_ns() // 1_000_000 if now_ms is None else now_ms
        due = self.is_due(block_id, check_ms)
        if not block.active or (not force and not due):
            return block.z
        active_z = self.source_snapshot() if source_snapshot is None else source_snapshot
        output = block.update(now_ms=now_ms, active_z=active_z)
        if block.o is not None:
            # The current ReadIn contribution is consumed once.
            block.o = block.o.new_zeros(block.o.shape)
        return output

    def step(self, *, now_ms: int | None = None, order: Sequence[int] | None = None) -> dict[int, Tensor]:
        """Visit all Blocks, or an explicit subset/order for asynchronous runs.

        An explicit now_ms is shared; otherwise each Block samples its clock
        after computing a. Inactive Blocks leave state/history untouched.
        """
        ids = list(range(len(self.blocks))) if order is None else list(order)
        for block_id in ids:
            self._check_id(block_id)
        if len(set(ids)) != len(ids):
            raise ValueError("a Block may appear only once per step")
        check_ms = monotonic_ns() // 1_000_000 if now_ms is None else now_ms
        due_ids = [i for i in ids if self.blocks[i].active and self.is_due(i, check_ms)]
        sources = self.source_snapshot()
        return {i: self.update_block(i, now_ms=now_ms, source_snapshot=sources) for i in due_ids}
