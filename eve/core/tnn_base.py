"""
EVE TNN (Tiny Neural Network) 基类。

TNN 是小型可训练 PyTorch 神经网络，支持运行时加载/运行/卸载。
- TNN 不是规则插件 —— 必须有参数、forward()、训练路径、保存/加载、版本。
- TNN 有独立的 TNN ID 空间（与 MemoryID 分离）。
- TNN Store 管理 TNN 权重/描述/结构，Memory 可引用 TNN 但权重在 TNN Store。
- QNN 仅存在于 Dock 训练阶段，不出现在运行时图中。
- 不同 TNN 可有不同时间尺度（ms 级动作、秒级策略等）。
- TNN 输出到 blackboard，不直接到 output。
"""
from __future__ import annotations

import json
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class SourceRef:
    """描述 TNN 输入来源。"""
    source_type: str  # "state" | "world" | "myself" | "blackboard" | "tnn_output" | "memory"
    source_id: str    # e.g., "screen", "cursor", "tnn:detector.target_position"
    field: str = ""   # 具体字段


@dataclass
class TNNDescriptor:
    """TNN 描述 — 事实源。"""
    tnn_id: str
    version: int = 1
    purpose: str = ""
    inputs: list[SourceRef] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)          # named output fields
    run_frequency_hz: float = 10.0                            # 0 = event-driven
    trigger_condition: str = ""                               # event trigger description
    observation_window_s: float = 1.0                         # how much history to observe
    action_horizon_s: float = 0.0                             # 0 = not action TNN
    latency_budget_ms: float = 30.0
    device: str = "cuda"
    precision: str = "float32"
    structure_summary: str = ""                               # brief description of architecture
    training_source: str = ""                                 # which training produced this
    eval_metrics: dict[str, float] = field(default_factory=dict)
    parent_tnn_id: str | None = None                          # version lineage
    replaced_by: str | None = None
    status: str = "available"                                 # "available" | "disabled" | "deprecated"
    tnn_class: str = ""                                       # fully qualified class name for deserialization


# ── 工具函数 ──────────────────────────────────────────────


def _descriptor_from_dict(d: dict[str, Any]) -> TNNDescriptor:
    """从字典重建 TNNDescriptor，正确处理嵌套 SourceRef。"""
    items: dict[str, Any] = {}
    for k, v in d.items():
        if k not in TNNDescriptor.__dataclass_fields__:
            continue
        if k == "inputs" and isinstance(v, list):
            items[k] = [SourceRef(**s) if isinstance(s, dict) else s for s in v]
        else:
            items[k] = v
    return TNNDescriptor(**items)


# ── TNN 基类 ──────────────────────────────────────────────


class TNNBase(ABC, nn.Module):
    """所有 TNN 的稳定基类。

    继承自 ABC 和 nn.Module，确保每个 TNN 都是真正的神经网络，
    具备 forward()、参数、保存/加载能力。
    """

    def __init__(self, descriptor: TNNDescriptor):
        super().__init__()
        self.descriptor = descriptor

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """前向推理。输入 dict 的 key 对应 SourceRef 的 source_id。"""
        ...

    def get_parameters(self) -> dict[str, Any]:
        """返回参数统计。"""
        return {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def save(self, path: str | Path) -> None:
        """保存权重 + descriptor 到指定目录。

        保存格式：
          {path}/descriptor.json
          {path}/weights.pt
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        from dataclasses import asdict

        desc_dict = asdict(self.descriptor)
        desc_dict["tnn_class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

        with open(path / "descriptor.json", "w", encoding="utf-8") as f:
            json.dump(desc_dict, f, indent=2, ensure_ascii=False)

        torch.save(self.state_dict(), path / "weights.pt")

    @classmethod
    def load(cls, path: str | Path) -> TNNBase:
        """从文件加载 TNN。

        读取 descriptor.json 确定 TNN 类型，调用对应子类的 _build_from_descriptor，
        然后加载 weights.pt。
        """
        path = Path(path)

        with open(path / "descriptor.json", "r", encoding="utf-8") as f:
            desc_dict = json.load(f)

        tnn_class_path = desc_dict.get("tnn_class", "")
        descriptor = _descriptor_from_dict(desc_dict)

        if tnn_class_path:
            module_path, class_name = tnn_class_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            target_cls = getattr(mod, class_name)
        else:
            target_cls = cls

        instance = target_cls._build_from_descriptor(descriptor)
        state_dict = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        instance.load_state_dict(state_dict)
        return instance

    @staticmethod
    @abstractmethod
    def _build_from_descriptor(descriptor: TNNDescriptor) -> TNNBase:
        """子类实现：根据 descriptor 构建网络结构。"""
        ...


# ── 最小测试 TNN ──────────────────────────────────────────


class DummyTNN(TNNBase):
    """最小测试 TNN：2 层 MLP，用于验证接口。"""

    def __init__(
        self,
        descriptor: TNNDescriptor,
        input_dim: int = 64,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__(descriptor)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._output_names = descriptor.outputs

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = list(inputs.values())[0]
        x = x.flatten(1)[:, : self._input_dim]
        if x.shape[1] < self._input_dim:
            x = torch.nn.functional.pad(x, (0, self._input_dim - x.shape[1]))
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return {self._output_names[0]: out} if self._output_names else {"output": out}

    @staticmethod
    def _build_from_descriptor(descriptor: TNNDescriptor) -> TNNBase:
        return DummyTNN(descriptor)


class ConvTNN(TNNBase):
    """用于图像输入的小型 CNN TNN。"""

    def __init__(self, descriptor: TNNDescriptor, input_channels: int = 3):
        super().__init__(descriptor)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 16, 64)
        self._input_channels = input_channels
        self._output_names = descriptor.outputs

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = list(inputs.values())[0]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != self._input_channels and x.shape[1] == 4:
            x = x[:, : self._input_channels]
        h = self.conv(x)
        h = h.flatten(1)
        out = self.fc(h)
        return {self._output_names[0]: out} if self._output_names else {"output": out}

    @staticmethod
    def _build_from_descriptor(descriptor: TNNDescriptor) -> TNNBase:
        return ConvTNN(descriptor)
