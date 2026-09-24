"""Connection-local plasticity for the production ACNT runtime.

Every parameter tensor owns one local eligibility trace q. Real elapsed time
decays that trace, while each Block supplies only its own analytic local VJP
contribution from the instantaneous perturbation. The one global Goodness
scalar is converted to M = g_eff - g_bar before the ordinary parameters are
updated. No autograd, Jacobian, e-prop state, credit assignment machinery, or
global backward signal is used.
"""

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn


class Plasticity:
    """One eligibility trace per parameter element, owned by its Block or Adapter.

    When present, the goodness Block and Adapter only receive c_g. Every other
    connection receives the one global g_eff when that scalar arrives.
    """

    def __init__(
        self,
        groups: Mapping[int, Mapping[str, nn.Parameter]],
        goodness_id: int | None,
        *,
        learning_rate: float = 0.001,
        tau_q_s: float = 1.0,
        tau_g_s: float = 5.0,
        perturbation_scale: float = 0.1,
        parameter_clip: tuple[float, float] | None = None,
    ) -> None:
        self.groups = {i: dict(parameters) for i, parameters in groups.items()}
        if goodness_id is not None and goodness_id not in self.groups:
            raise ValueError("goodness Block must have a parameter group")
        self.goodness_id = goodness_id
        self.parameters = {id(p): p for group in self.groups.values() for p in group.values()}
        if len(self.parameters) != sum(len(group) for group in self.groups.values()):
            raise ValueError("each parameter must belong to exactly one Block group")
        if parameter_clip is not None and (
            len(parameter_clip) != 2 or
            not all(math.isfinite(v) for v in parameter_clip) or
            parameter_clip[0] > parameter_clip[1]
        ):
            raise ValueError("parameter_clip must be ordered finite bounds")
        self.parameter_clip = parameter_clip

        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")

        if not math.isfinite(tau_q_s) or tau_q_s <= 0:
            raise ValueError("tau_q_s must be finite and positive")

        if not math.isfinite(tau_g_s) or tau_g_s <= 0:
            raise ValueError("tau_g_s must be finite and positive")

        if not math.isfinite(perturbation_scale) or perturbation_scale <= 0:
            raise ValueError(
                "perturbation_scale must be finite and positive"
            )

        self.learning_rate = float(learning_rate)
        self.tau_q_s = float(tau_q_s)
        self.tau_g_s = float(tau_g_s)
        self.perturbation_scale = float(perturbation_scale)

        self.traces = {
            key: torch.zeros_like(p)
            for key, p in self.parameters.items()
        }

        self.trace_times_ms: dict[int, int | None] = {
            key: None
            for key in self.parameters
        }
        self.g_bar = 0.5
        self.goodness_time_ms: int | None = None
        self._event_time_ms: int | None = None
        self._excluded_controls: dict[int, int] = {}
        self._hooks = []
        self._discrete_adapters: dict[
            str,
            tuple[nn.Module, dict],
        ] = {}

    def attach_adapters(
        self,
        adapters: Mapping[str, nn.Module],
        *,
        route_id: int | None = None,
    ) -> None:
        self.detach_adapters()

        for name in ("hand", "route"):
            if name not in adapters:
                continue

            adapter = adapters[name]

            linears = [
                module
                for module in adapter.modules()
                if isinstance(module, nn.Linear)
            ]

            if not linears:
                raise ValueError(
                    f"{name} requires at least one Linear layer"
                )

            cache: dict = {
                "layers": [],
            }

            def make_hook(cache_ref):
                @torch.no_grad()
                def hook(module, inputs, output):
                    cache_ref["layers"].append({
                        "module": module,
                        "input": inputs[0].detach().clone(),
                        "output": output.detach().clone(),
                    })
                return hook

            hook = make_hook(cache)
            for layer in linears:
                self._hooks.append(
                    layer.register_forward_hook(hook)
                )

            self._discrete_adapters[name] = (
                adapter,
                cache,
            )

        if (
            "route" in adapters
            and route_id is not None
        ):
            self._excluded_controls[
                id(adapters["route"])
            ] = route_id

    def detach_adapters(self) -> None:
        for hook in self._hooks:
            hook.remove()

        self._hooks.clear()

        if hasattr(self, "_discrete_adapters"):
            self._discrete_adapters.clear()

        self._excluded_controls.clear()

    def set_event_time(self, now_ms: int) -> None:
        """Record the real timestamp of the current scheduling event."""
        self._validate_time(now_ms, "local event time")
        self._event_time_ms = now_ms

    @staticmethod
    def _validate_time(now_ms: int | None, label: str) -> None:
        if type(now_ms) is not int:
            raise ValueError(f"{label} must be integer milliseconds")

    def _decay_trace(
        self,
        key: int,
        now_ms: int,
    ) -> None:
        last_ms = self.trace_times_ms[key]

        if last_ms is not None:
            if now_ms < last_ms:
                raise ValueError(
                    "eligibility time must not move backwards"
                )

            dt_s = (now_ms - last_ms) / 1000.0
            decay = math.exp(-dt_s / self.tau_q_s)

            self.traces[key].mul_(decay)

        self.trace_times_ms[key] = now_ms

    def _decay_groups(
        self,
        block_ids,
        now_ms: int,
    ) -> tuple[int, ...]:
        ids = tuple(block_ids)

        for block_id in ids:
            for parameter in self.groups[block_id].values():
                self._decay_trace(
                    id(parameter),
                    now_ms,
                )

        return ids

    @torch.no_grad()
    def accumulate(
        self,
        parameter: nn.Parameter,
        contribution: Tensor,
        *,
        now_ms: int,
    ) -> None:
        key = id(parameter)

        if key not in self.traces:
            raise ValueError(
                "eligibility parameter is not registered"
            )

        self._validate_time(
            now_ms,
            "local eligibility time",
        )

        if contribution.shape != parameter.shape:
            raise ValueError(
                "eligibility contribution shape mismatch"
            )

        if contribution.dtype != parameter.dtype:
            raise ValueError(
                "eligibility contribution dtype mismatch"
            )

        if contribution.device != parameter.device:
            raise ValueError(
                "eligibility contribution device mismatch"
            )

        if not torch.isfinite(contribution).all():
            raise FloatingPointError(
                "non-finite eligibility contribution"
            )

        self._decay_trace(
            key,
            now_ms,
        )

        self.traces[key].add_(
            contribution.detach()
        )

        if not torch.isfinite(self.traces[key]).all():
            raise FloatingPointError(
                "non-finite eligibility trace"
            )

    @torch.no_grad()
    def observe_control(
        self,
        name: str,
        adapter: nn.Module,
        signal,
        *,
        now_ms: int,
    ) -> None:
        self._validate_time(
            now_ms,
            "local eligibility time",
        )

        if name not in self._discrete_adapters:
            raise ValueError(
                f"{name} discrete adapter is not attached"
            )

        attached_adapter, cache = self._discrete_adapters[name]

        if attached_adapter is not adapter:
            raise ValueError(
                "discrete adapter mismatch"
            )

        layers = cache["layers"]

        if not layers:
            raise ValueError(
                "discrete adapter must run before observe_control"
            )

        score = (
            signal.a.to(signal.p.dtype)
            - signal.p
        ) / signal.tau

        excluded = self._excluded_controls.get(
            id(adapter)
        )

        if excluded is not None:
            score = score.clone()
            if 0 <= excluded < score.numel():
                score[excluded] = 0.0

        terminal_info = layers[-1]
        terminal = terminal_info["module"]
        terminal_input = terminal_info["input"]

        count = signal.q.numel()

        if terminal.out_features < count:
            raise ValueError(
                "terminal layer has fewer outputs than discrete controls"
            )

        delta = torch.zeros(
            terminal.out_features,
            dtype=terminal_input.dtype,
            device=terminal_input.device,
        )

        delta[:count] = score

        self.accumulate(
            terminal.weight,
            delta.unsqueeze(1)
            * terminal_input.unsqueeze(0),
            now_ms=now_ms,
        )

        if terminal.bias is not None:
            self.accumulate(
                terminal.bias,
                delta,
                now_ms=now_ms,
            )

        if len(layers) >= 2:
            previous_info = layers[-2]
            previous = previous_info["module"]
            previous_input = previous_info["input"]
            previous_output = previous_info["output"]

            hidden_delta = (
                terminal.weight.detach().T
                @ delta
            )

            hidden_delta = hidden_delta * (
                previous_output > 0
            ).to(hidden_delta.dtype)

            self.accumulate(
                previous.weight,
                hidden_delta.unsqueeze(1)
                * previous_input.unsqueeze(0),
                now_ms=now_ms,
            )

            if previous.bias is not None:
                self.accumulate(
                    previous.bias,
                    hidden_delta,
                    now_ms=now_ms,
                )

        cache["layers"].clear()

    @torch.no_grad()
    def _apply(
        self,
        block_ids,
        scalar: float,
        *,
        teacher: bool = False,
    ) -> float:
        if not math.isfinite(scalar):
            raise ValueError(
                "plasticity modulator must be finite"
            )

        proposals = []

        for block_id in block_ids:
            for parameter in self.groups[block_id].values():
                trace = self.traces[id(parameter)]

                value = (
                    parameter.detach()
                    + self.learning_rate
                    * scalar
                    * trace
                )

                if self.parameter_clip is not None:
                    value = value.clamp(
                        *self.parameter_clip
                    )

                if (
                    value.shape != parameter.shape
                    or not torch.isfinite(value).all()
                ):
                    raise FloatingPointError(
                        "plasticity proposed an invalid parameter"
                    )

                proposals.append(
                    (parameter, value)
                )

        for parameter, value in proposals:
            parameter.copy_(value)

        return scalar

    def apply_goodness(
        self,
        g_eff: float,
        *,
        now_ms: int,
    ) -> float:
        if (
            not math.isfinite(g_eff)
            or not 0 <= g_eff <= 1
        ):
            raise ValueError(
                "effective goodness must be in [0,1]"
            )

        self._validate_time(
            now_ms,
            "goodness delivery time",
        )

        block_ids = self._decay_groups(
            (
                i
                for i in self.groups
                if i != self.goodness_id
            ),
            now_ms,
        )

        if (
            self.goodness_time_ms is not None
            and now_ms < self.goodness_time_ms
        ):
            raise ValueError(
                "goodness time must not move backwards"
            )

        modulation = (
            float(g_eff)
            - self.g_bar
        )

        self._apply(
            block_ids,
            modulation,
        )

        delta_ms = (
            0
            if self.goodness_time_ms is None
            else now_ms - self.goodness_time_ms
        )

        retention = math.exp(
            -(delta_ms / 1000.0)
            / self.tau_g_s
        )

        self.g_bar = (
            retention * self.g_bar
            + (1.0 - retention)
            * float(g_eff)
        )

        self.goodness_time_ms = now_ms

        return modulation

    @torch.no_grad()
    def calibrate_goodness(
        self,
        c_g: float,
        *,
        now_ms: int,
    ) -> float:
        if self.goodness_id is None:
            raise RuntimeError(
                "no goodness Block is registered"
            )

        if (
            not math.isfinite(c_g)
            or not -1 <= c_g <= 1
        ):
            raise ValueError(
                "calibration scalar must be in [-1,1]"
            )

        self._validate_time(
            now_ms,
            "goodness calibration time",
        )

        return float(c_g)

    def clear(self) -> None:
        for trace in self.traces.values():
            trace.zero_()

        for key in self.trace_times_ms:
            self.trace_times_ms[key] = None

        self.g_bar = 0.5
        self.goodness_time_ms = None
        self._event_time_ms = None

        for _, cache in getattr(
            self,
            "_discrete_adapters",
            {},
        ).values():
            cache["layers"].clear()