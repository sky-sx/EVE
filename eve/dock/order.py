"""Dock 训练订单与结果。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingOrder:
    """训练订单 — Dock 接收的训练任务描述。"""
    order_id: str
    target_tnn_id: str          # 要训练/创建的 TNN ID
    purpose: str                # 训练目的
    training_data: list[str]    # MemoryID 列表，用于获取训练数据
    teacher: str                # "local_llm" | "vlm" | "yolo" | "human" | "rule" | "existing_tnn:xxx"
    teacher_prompt: str = ""    # LLM 教师专用 prompt
    input_sources: list[Any] = field(default_factory=list)  # SourceRef list
    output_fields: list[str] = field(default_factory=list)
    run_frequency_hz: float = 10.0
    latency_budget_ms: float = 30.0
    observation_window_s: float = 1.0
    action_horizon_s: float = 0.0
    structure_hint: str = "mlp"  # "mlp" | "cnn" | "conv_small"
    create_new: bool = True     # True=创建新TNN, False=补训已有TNN
    priority: str = "medium"    # "low" | "medium" | "high"
    created_at_ns: int = 0
    status: str = "pending"     # "pending" | "running" | "completed" | "failed"

    def __post_init__(self):
        if self.created_at_ns == 0:
            self.created_at_ns = time.monotonic_ns()


@dataclass
class TrainingResult:
    order_id: str
    success: bool
    tnn_id: str = ""
    version: int = 0
    eval_metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error_message: str = ""
    memory_reports: list[str] = field(default_factory=list)  # MemoryID of training reports
