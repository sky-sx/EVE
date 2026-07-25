"""Tests for Dock training system: TrainingOrder, TrainingResult, Trainer, create_tnn_from_order."""

import torch
from pathlib import Path

from eve.dock.order import TrainingOrder, TrainingResult
from eve.dock.trainer import Trainer
from eve.dock.tiny_nn import create_tnn_from_order
from eve.core.tnn_base import SourceRef, TNNDescriptor, DummyTNN, ConvTNN
from eve.core.tnn_store import TNNStore
from eve.memory.memorizer import Memorizer


# ── TrainingOrder ─────────────────────────────────────────────

def test_training_order_fields() -> None:
    order = TrainingOrder(
        order_id="ord_001",
        target_tnn_id="tnn_target",
        purpose="classify images",
        training_data=["mem_a", "mem_b"],
        teacher="rule",
        teacher_prompt="classify this",
        structure_hint="cnn",
        priority="high",
    )
    assert order.order_id == "ord_001"
    assert order.target_tnn_id == "tnn_target"
    assert order.purpose == "classify images"
    assert order.training_data == ["mem_a", "mem_b"]
    assert order.teacher == "rule"
    assert order.structure_hint == "cnn"
    assert order.priority == "high"
    assert order.status == "pending"


def test_training_result_structure() -> None:
    result = TrainingResult(
        order_id="ord_x",
        success=True,
        tnn_id="tnn_y",
        version=3,
        eval_metrics={"loss": 0.05},
        latency_ms=123.4,
    )
    assert result.success is True
    assert result.tnn_id == "tnn_y"
    assert result.version == 3
    assert result.eval_metrics["loss"] == 0.05
    assert result.latency_ms == 123.4
    assert result.error_message == ""

    failed = TrainingResult(order_id="fail_order", success=False, error_message="boom")
    assert failed.success is False
    assert failed.error_message == "boom"


# ── Trainer enqueue / has_pending / priority ──────────────────

def test_trainer_enqueue(tmp_path: Path) -> None:
    memorizer = Memorizer(tmp_path / "mem")
    # Empty store
    store = TNNStore(tmp_path / "store")
    trainer = Trainer(tnn_store=store, memorizer=memorizer)

    order = TrainingOrder(
        order_id="o1", target_tnn_id="t1", purpose="test",
        training_data=[], teacher="rule", priority="medium",
    )
    trainer.enqueue(order)
    assert trainer.has_pending() is True
    assert trainer.get_queue_size() == 1


def test_trainer_has_pending() -> None:
    class FakeStore:
        pass
    class FakeMem:
        pass
    trainer = Trainer(tnn_store=FakeStore(), memorizer=FakeMem())
    assert trainer.has_pending() is False
    order = TrainingOrder(
        order_id="o1", target_tnn_id="t1", purpose="test",
        training_data=[], teacher="rule", priority="medium",
    )
    trainer.enqueue(order)
    assert trainer.has_pending() is True


def test_trainer_queue_priority() -> None:
    class FakeStore:
        pass
    class FakeMem:
        pass
    trainer = Trainer(tnn_store=FakeStore(), memorizer=FakeMem())

    low = TrainingOrder(
        order_id="low", target_tnn_id="t1", purpose="low",
        training_data=[], teacher="rule", priority="low",
    )
    high = TrainingOrder(
        order_id="high", target_tnn_id="t2", purpose="high",
        training_data=[], teacher="rule", priority="high",
    )
    medium = TrainingOrder(
        order_id="med", target_tnn_id="t3", purpose="med",
        training_data=[], teacher="rule", priority="medium",
    )
    trainer.enqueue(low)
    trainer.enqueue(high)
    trainer.enqueue(medium)
    assert trainer.get_queue_size() == 3
    # Check queue order: high should be first
    assert trainer._queue[0].order_id == "high"
    assert trainer._queue[1].order_id == "med"
    assert trainer._queue[2].order_id == "low"


# ── create_tnn_from_order ─────────────────────────────────────

def test_create_tnn_from_order_mlp() -> None:
    order = TrainingOrder(
        order_id="o_mlp", target_tnn_id="tnn_mlp", purpose="mlp test",
        training_data=[], teacher="rule", structure_hint="mlp",
        input_sources=[SourceRef(source_type="state", source_id="screen")],
        output_fields=["cls"],
    )
    tnn = create_tnn_from_order(order, input_dim_estimate=64)
    assert isinstance(tnn, DummyTNN)
    assert tnn.descriptor.tnn_id == "tnn_mlp"
    assert tnn.descriptor.purpose == "mlp test"


def test_create_tnn_from_order_cnn() -> None:
    order = TrainingOrder(
        order_id="o_cnn", target_tnn_id="tnn_cnn", purpose="cnn test",
        training_data=[], teacher="rule", structure_hint="cnn",
        input_sources=[SourceRef(source_type="state", source_id="screen")],
        output_fields=["feat"],
    )
    tnn = create_tnn_from_order(order, input_dim_estimate=64)
    assert isinstance(tnn, ConvTNN)
    assert tnn.descriptor.tnn_id == "tnn_cnn"
    assert tnn.descriptor.outputs == ["feat"]


# ── Trainer process_one (rule teacher) ────────────────────────

def test_trainer_process_one_rule_teacher(tmp_path: Path) -> None:
    mem = Memorizer(tmp_path / "mem")
    store = TNNStore(tmp_path / "store")
    trainer = Trainer(tnn_store=store, memorizer=mem)

    # Create training data in memory
    mid1 = mem.create("sample data for training", payload_type="text")
    mid2 = mem.create("another training sample here", payload_type="text")

    order = TrainingOrder(
        order_id="train_ord_1",
        target_tnn_id="tnn_rule",
        purpose="text classifier",
        training_data=[mid1, mid2],
        teacher="rule",
        teacher_prompt="",
        input_sources=[SourceRef(source_type="memory", source_id="text")],
        output_fields=["cls"],
        structure_hint="mlp",
    )

    trainer.enqueue(order)
    result = trainer.process_one()

    assert isinstance(result, TrainingResult)
    assert result.success is True
    assert result.order_id == "train_ord_1"
    assert result.tnn_id == "tnn_rule"
    assert result.version > 0
    assert "eval_loss" in result.eval_metrics
