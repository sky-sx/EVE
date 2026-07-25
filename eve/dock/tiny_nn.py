"""TNN 构建工具 — 根据训练订单创建合适的 TNN 结构。"""
from __future__ import annotations

import torch.nn as nn
from eve.core.tnn_base import TNNDescriptor, TNNBase, DummyTNN, ConvTNN


def create_tnn_from_order(order, input_dim_estimate: int = 64) -> TNNBase:
    """根据训练订单创建 TNN。

    根据 structure_hint 选择合适的网络：
    - "mlp" → DummyTNN (2-layer MLP)
    - "cnn" → ConvTNN (small CNN)
    - "conv_small" → 更小的 CNN
    """
    # 类型导入延迟到函数内以避免循环引用
    from eve.dock.order import TrainingOrder

    descriptor = TNNDescriptor(
        tnn_id=order.target_tnn_id,
        purpose=order.purpose,
        inputs=order.input_sources,
        outputs=order.output_fields,
        run_frequency_hz=order.run_frequency_hz,
        latency_budget_ms=order.latency_budget_ms,
        observation_window_s=order.observation_window_s,
        action_horizon_s=order.action_horizon_s,
    )

    hint = order.structure_hint
    if hint == "mlp":
        return DummyTNN(descriptor, input_dim=input_dim_estimate)
    elif hint == "cnn":
        return ConvTNN(descriptor)
    elif hint == "conv_small":
        return ConvTNN(descriptor, input_channels=1)
    else:
        return DummyTNN(descriptor, input_dim=input_dim_estimate)
