"""The six selected adapters wired to six distinct canonical Blocks."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from time import monotonic_ns

import torch
from torch import Tensor, nn

from .control import DiscreteSignal, sample_discrete
from .core import Core
from .hand import HAND_DISCRETE_NAMES
from .mechanical import MechanicalLog
from .plasticity import LocalEvent, Plasticity


READINS = ("eye", "ear")
READOUTS = ("hand", "speak", "goodness", "route")
ORGANS = READINS + READOUTS


@dataclass(frozen=True)
class GoodnessSignal:
    """One scalar prediction plus trainer metadata, never extra reward axes."""

    time_ms: int
    g: Tensor
    g_eff: float | None
    teacher: float | None
    calibration_loss: float | None


@dataclass(frozen=True)
class HandSignal:
    discrete: DiscreteSignal
    continuous: Tensor


class Runtime(nn.Module):
    def __init__(self, core: Core, adapters: Mapping[str, nn.Module], organ_blocks: Mapping[str, int], *, noise_scale: float = 1.0) -> None:
        super().__init__()
        if set(adapters) != set(ORGANS) or set(organ_blocks) != set(ORGANS):
            raise ValueError("exactly the six canonical organs must be bound")
        ids = list(organ_blocks.values())
        if len(set(ids)) != len(ORGANS):
            raise ValueError("each organ must use a different Block")
        for name, block_id in organ_blocks.items():
            core._check_id(block_id)
            block = core.blocks[block_id]
            adapter = adapters[name]
            if adapter.neuron_size != block.neuron_size:
                raise ValueError("Adapter and Block neuron sizes must match")
            if (block.o is not None) != (name in READINS):
                raise ValueError("only eye and ear Blocks have a ReadIn o vector")
            if any(p.device != block.b.device for p in adapter.parameters()):
                raise ValueError("Adapter and Block must use the same device")
        if adapters["route"].max_blocks != len(core.blocks):
            raise ValueError("route dimension must equal the configured Block capacity")
        readin_ids = {organ_blocks[name] for name in READINS}
        for block in core.blocks:
            if (block.o is not None) != (block.block_id in readin_ids):
                raise ValueError("only bound eye/ear Blocks may have a ReadIn o vector")
        if not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError("noise_scale must be finite and positive")
        self.noise_scale = float(noise_scale)
        self.core = core
        self.adapters = nn.ModuleDict(adapters)
        self.organ_blocks = dict(organ_blocks)
        self.goodness_active = core.blocks[self.organ_blocks["goodness"]].active
        self.core.set_active(self.organ_blocks["route"], True)
        self.mechanical_log = MechanicalLog()
        self.execution_enabled = {"hand": False, "speak": False, "route": True, "goodness": self.goodness_active}
        self.executors: dict = {}
        self.plasticity: Plasticity | None = None

    def enable_plasticity(self, **hyperparameters) -> Plasticity:
        """Install a replaceable local rule; no e-prop state is created."""
        if self.plasticity is not None:
            self.plasticity.detach_adapters()
        reverse = {block_id: name for name, block_id in self.organ_blocks.items()}
        groups = {}
        for i, block in enumerate(self.core.blocks):
            parameters = {f"block.{name}": parameter for name, parameter in block.named_parameters()}
            organ = reverse.get(i)
            if organ is not None:
                parameters.update({f"adapter.{name}": parameter for name, parameter in self.adapters[organ].named_parameters()})
            groups[i] = parameters
        self.plasticity = Plasticity(groups, self.organ_blocks["goodness"], **hyperparameters)
        self.plasticity.attach_adapters(self.adapters, route_id=self.organ_blocks["route"])
        for block in self.core.blocks:
            block.local_observer = lambda parameter, pre, post: self.plasticity.observe(
                parameter, LocalEvent(pre, post, "dense" if parameter.ndim == 2 else "bias")
            )
        return self.plasticity

    @torch.no_grad()
    def step(
        self, *, now_ms: int, readins: Mapping[str, Tensor] | None = None,
        teacher: float | None = None, teacher_time_ms: int | None = None,
        generator: torch.Generator | None = None, learn: bool = True,
    ) -> dict:
        """One mock/world tick. Returned diagnostics contain no live graphs."""
        if learn and self.plasticity is None:
            self.enable_plasticity()
        updated = self.update_blocks(now_ms=now_ms, readins=readins)
        route = self.generate_route(now_ms=now_ms, generator=generator)
        hand = self.generate_hand(now_ms=now_ms, generator=generator)
        speak = self.generate_speak(now_ms=now_ms)
        goodness = self.generate_goodness(
            now_ms=now_ms, teacher=teacher, teacher_time_ms=teacher_time_ms, calibrate=learn,
        )
        modulation = self.learn_goodness(goodness) if learn else None
        return {
            "time_ms": now_ms, "updated": list(updated), "next_active": list(self.core.active_ids),
            "history_lengths": [len(block.A) for block in self.core.blocks],
            "z_norms": [float(block.z.norm()) for block in self.core.blocks],
            "g": float(goodness.g), "g_eff": goodness.g_eff, "teacher": goodness.teacher,
            "calibration_loss": goodness.calibration_loss, "goodness_modulation": modulation,
            "route_sampled": route.a.tolist(),
            "hand_discrete": hand.discrete.a.tolist(), "hand_continuous": hand.continuous.detach().tolist(),
            "speak": speak.detach().tolist(), "mechanical_records": len(self.mechanical_log.records),
        }

    def learn_goodness(self, signal: GoodnessSignal, *, delivered_ms: int | None = None) -> float | None:
        """Delivery time may be later than the paired prediction/teacher time."""
        if self.plasticity is None:
            raise RuntimeError("enable plasticity before consuming goodness")
        delivered_ms = signal.time_ms if delivered_ms is None else delivered_ms
        if type(delivered_ms) is not int or delivered_ms < signal.time_ms:
            raise ValueError("delivery cannot precede the goodness event")
        return None if signal.g_eff is None else self.plasticity.apply_goodness(signal.g_eff, now_ms=delivered_ms)

    def set_execution_enabled(self, name: str, enabled: bool) -> None:
        if name not in READOUTS or type(enabled) is not bool:
            raise ValueError("execution switch requires a ReadOut name and bool")
        self.execution_enabled[name] = enabled
        if name == "goodness":
            self.set_goodness_active(enabled)

    def _emit(self, name: str, signal: dict, now_ms: int) -> None:
        """Only the mechanical boundary sees executor returns/errors."""
        record = {"time_ms": now_ms, "organ": name, "signal": deepcopy(signal),
                  "execution_enabled": self.execution_enabled[name]}
        executor = self.executors.get(name)
        if self.execution_enabled[name] and executor is not None:
            try:
                executor(deepcopy(signal))  # Return value is deliberately discarded.
            except Exception as error:
                record["executor_error"] = str(error)
        self.mechanical_log.append(record)

    @torch.no_grad()
    def generate_hand(self, *, now_ms: int = 0, generator: torch.Generator | None = None, threshold: float | Tensor = 0.0) -> HandSignal:
        """Generate all 82 keyboard keys, three mouse buttons and dx/dy."""
        adapter = self.adapters["hand"]
        if adapter.discrete_controls != len(HAND_DISCRETE_NAMES) or adapter.continuous_controls != 2:
            raise ValueError("runtime hand requires 82 keys, three mouse buttons and dx/dy")
        raw = self.decode_readout("hand")
        count = adapter.discrete_controls
        discrete = sample_discrete(raw[:count], tau=self.noise_scale, threshold=threshold, generator=generator)
        continuous = raw[count:]
        if self.plasticity is not None:
            self._observe_discrete("hand", discrete, now_ms=now_ms)
        self._emit("hand", {
            "discrete": dict(zip(HAND_DISCRETE_NAMES, discrete.a.tolist())),
            "continuous": {"dx": float(continuous[0].detach()), "dy": float(continuous[1].detach())},
        }, now_ms)
        return HandSignal(discrete, continuous)

    def _observe_discrete(self, name: str, signal: DiscreteSignal, *, now_ms: int) -> None:
        # The terminal itself sees q, its own noise, and the actual event.
        # No score function or global feedback is constructed.
        self.plasticity.observe_control(self.adapters[name], signal)

    @torch.no_grad()
    def generate_speak(self, *, now_ms: int = 0) -> Tensor:
        controls = self.decode_readout("speak")
        if controls.shape != (30,):
            raise ValueError("speak requires 30 continuous controls")
        self._emit("speak", {"controls": controls.detach().tolist()}, now_ms)
        return controls

    @torch.no_grad()
    def generate_goodness(
        self, *, now_ms: int, teacher: float | None = None,
        teacher_time_ms: int | None = None, calibrate: bool = True,
    ) -> GoodnessSignal:
        """One clamped scalar; same-time teacher calibrates B_g and A_g locally."""
        if type(now_ms) is not int:
            raise ValueError("now_ms must be integer milliseconds")
        if teacher is not None:
            teacher = float(teacher)
            if not math.isfinite(teacher) or not 0 <= teacher <= 1:
                raise ValueError("teacher must be in [0,1]")
            if teacher_time_ms is not None and (type(teacher_time_ms) is not int or teacher_time_ms != now_ms):
                raise ValueError("teacher and g must have the same timestamp")
        elif teacher_time_ms is not None:
            raise ValueError("a teacher timestamp requires a teacher")
        block = self.core.blocks[self.organ_blocks["goodness"]]
        raw = self.adapters["goodness"](block.z)
        if raw.shape != (1,):
            raise ValueError("goodness must produce exactly one scalar")
        prediction = raw.reshape(()).clamp(0, 1).detach().clone()
        loss = None
        if teacher is not None and self.execution_enabled["goodness"]:
            c_g = teacher - float(prediction)
            loss = 0.5 * c_g * c_g
            if calibrate and self.plasticity is not None:
                self.plasticity.calibrate_goodness(c_g)
        self._emit("goodness", {"g": float(prediction)}, now_ms)
        effective = teacher if teacher is not None else (float(prediction) if self.execution_enabled["goodness"] else None)
        return GoodnessSignal(now_ms, prediction, effective, teacher, loss)

    def set_goodness_active(self, active: bool) -> None:
        """The goodness Block's active state is human-controlled, not routed."""
        if type(active) is not bool:
            raise TypeError("goodness active must be bool")
        self.goodness_active = active
        if hasattr(self, "execution_enabled"):
            self.execution_enabled["goodness"] = active
        self.core.set_active(self.organ_blocks["goodness"], active)

    @torch.no_grad()
    def generate_route(self, *, now_ms: int = 0, generator: torch.Generator | None = None, threshold: float | Tensor = 0.0) -> DiscreteSignal:
        """q + independent Logistic noise + threshold -> next active set."""
        signal = sample_discrete(self.decode_readout("route"), tau=self.noise_scale, threshold=threshold, generator=generator)
        if self.plasticity is not None:
            # Observe the actual terminal event before role overrides.
            self._observe_discrete("route", signal, now_ms=now_ms)
        mask = signal.a.tolist()
        # Role invariants take precedence over route's stochastic proposals.
        mask[self.organ_blocks["route"]] = True
        mask[self.organ_blocks["goodness"]] = self.goodness_active
        self._emit("route", {"active": mask, "sampled": signal.a.tolist()}, now_ms)
        if self.execution_enabled["route"]:
            self.core.set_active_mask(mask)
        self.core.set_active(self.organ_blocks["route"], True)
        return signal

    @torch.no_grad()
    def encode_readin(self, name: str, external_input: Tensor) -> Tensor:
        """External sample -> adapter -> the unique ReadIn Block's o."""
        if name not in READINS:
            raise ValueError("only eye and ear are ReadIn organs")
        block = self.core.blocks[self.organ_blocks[name]]
        o = self.adapters[name](external_input)
        block._check_vector(o, 2 * block.neuron_size, "ReadIn output")
        block.o = o
        return o

    @torch.no_grad()
    def decode_readout(self, name: str) -> Tensor:
        """The unique ReadOut Block's z -> its raw adapter output."""
        if name not in READOUTS:
            raise ValueError("only hand, speak, goodness and route are ReadOut organs")
        block = self.core.blocks[self.organ_blocks[name]]
        return self.adapters[name](block.z)

    @torch.no_grad()
    def update_blocks(self, *, now_ms: int | None = None, readins: Mapping[str, Tensor] | None = None, order: Sequence[int] | None = None) -> dict[int, Tensor]:
        """New input forces one update, even if route did not select its Block.

        Forced ReadIns contribute only their OLD committed source states.
        Their new states can propagate next round; active choices are restored.
        """
        inputs = {} if readins is None else dict(readins)
        if not set(inputs).issubset(READINS):
            raise ValueError("external inputs may only target eye or ear")
        ids = list(range(len(self.core.blocks))) if order is None else list(order)
        if sorted(ids) != list(range(len(self.core.blocks))):
            raise ValueError("order must visit every Block exactly once")
        now_ms = monotonic_ns() // 1_000_000 if now_ms is None else now_ms
        self.core.set_active(self.organ_blocks["route"], True)
        self.core.set_active(self.organ_blocks["goodness"], self.goodness_active)
        # Validate time before accepting an external input or changing flags.
        for i in range(len(self.core.blocks)):
            self.core.is_due(i, now_ms)
        for name, external in inputs.items():
            self.encode_readin(name, external)
        forced = {self.organ_blocks[name] for name in inputs}
        previous_active = {i: self.core.blocks[i].active for i in forced}
        for i in forced:
            self.core.set_active(i, True)
        try:
            outputs = {}
            sources = self.core.source_snapshot()
            for i in ids:
                block = self.core.blocks[i]
                if block.active and (i in forced or self.core.is_due(i, now_ms)):
                    outputs[i] = self.core.update_block(i, now_ms=now_ms, force=i in forced, source_snapshot=sources)
            return outputs
        finally:
            for i, active in previous_active.items():
                self.core.set_active(i, active)
