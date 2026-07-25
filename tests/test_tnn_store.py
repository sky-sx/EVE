"""Tests for TNN Store: TNNBase, DummyTNN, ConvTNN, TNNDescriptor, SourceRef, TNNStore."""

import torch
from pathlib import Path

from eve.core.tnn_base import TNNDescriptor, SourceRef, TNNBase, DummyTNN, ConvTNN
from eve.core.tnn_store import TNNStore, TNNStoreEntry


# ── TNN Forward ───────────────────────────────────────────────

def test_dummy_tnn_forward() -> None:
    desc = TNNDescriptor(tnn_id="dummy_1", outputs=["out"])
    tnn = DummyTNN(desc, input_dim=64, hidden_dim=32, output_dim=16)
    x = torch.randn(1, 64)
    out = tnn({"out": x})
    assert isinstance(out, dict)
    assert "out" in out
    assert out["out"].shape[1] == 16


def test_conv_tnn_forward() -> None:
    desc = TNNDescriptor(tnn_id="conv_1", outputs=["feat"])
    tnn = ConvTNN(desc, input_channels=3)
    x = torch.randn(1, 3, 64, 64)
    out = tnn({"feat": x})
    assert isinstance(out, dict)
    assert "feat" in out
    assert out["feat"].shape[1] == 64


# ── Descriptor / SourceRef ────────────────────────────────────

def test_descriptor_fields() -> None:
    desc = TNNDescriptor(
        tnn_id="tnn_test",
        version=2,
        purpose="classify",
        inputs=[SourceRef(source_type="state", source_id="screen")],
        outputs=["cls"],
        run_frequency_hz=10.0,
    )
    assert desc.tnn_id == "tnn_test"
    assert desc.version == 2
    assert desc.purpose == "classify"
    assert len(desc.inputs) == 1
    assert desc.inputs[0].source_type == "state"
    assert desc.outputs == ["cls"]
    assert desc.run_frequency_hz == 10.0


def test_source_ref_parsing() -> None:
    sr = SourceRef(source_type="tnn_output", source_id="tnn:detector.bbox", field="x1")
    assert sr.source_type == "tnn_output"
    assert sr.source_id == "tnn:detector.bbox"
    assert sr.field == "x1"

    sr2 = SourceRef(source_type="blackboard", source_id="cursor_pos")
    assert sr2.source_type == "blackboard"
    assert sr2.source_id == "cursor_pos"
    assert sr2.field == ""


# ── Save / Load Roundtrip ─────────────────────────────────────

def test_tnn_save_load_roundtrip(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="roundtrip", outputs=["out"])
    tnn = DummyTNN(desc, input_dim=64, hidden_dim=32, output_dim=16)
    x = torch.randn(1, 64)
    result1 = tnn({"out": x})["out"]

    save_dir = tmp_path / "roundtrip_v1"
    tnn.save(save_dir)

    loaded = DummyTNN.load(save_dir)
    loaded.eval()
    with torch.no_grad():
        result2 = loaded({"out": x})["out"]
    assert torch.allclose(result1, result2)
    assert loaded.descriptor.tnn_id == "roundtrip"


# ── TNNStore register / list ──────────────────────────────────

def test_tnn_store_register(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="store_reg", outputs=["out"])
    tnn = DummyTNN(desc)
    save_dir = tmp_path / "src"
    tnn.save(save_dir)

    store = TNNStore(tmp_path / "store")
    entry = store.register(
        tnn_id="store_reg",
        descriptor=desc,
        weights_path=str(save_dir / "weights.pt"),
        structure_version="1",
    )
    assert entry.tnn_id == "store_reg"
    assert entry.version == desc.version

    available = store.list_available()
    assert "store_reg" in available


# ── TNNStore load / unload ────────────────────────────────────

def test_tnn_store_load_unload(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="load_test", outputs=["out"])
    tnn = DummyTNN(desc)
    save_dir = tmp_path / "src2"
    tnn.save(save_dir)

    store = TNNStore(tmp_path / "store2")
    store.register(
        tnn_id="load_test",
        descriptor=desc,
        weights_path=str(save_dir / "weights.pt"),
        structure_version="1",
    )

    # Initially not loaded
    assert "load_test" not in store.list_loaded()

    instance = store.load_tnn("load_test", device="cpu")
    assert instance is not None
    assert "load_test" in store.list_loaded()
    assert store.get_descriptor("load_test") is not None

    store.unload_tnn("load_test")
    assert "load_test" not in store.list_loaded()

    # get_descriptor still works even after unload
    d = store.get_descriptor("load_test")
    assert d is not None
    assert d.tnn_id == "load_test"


# ── Save new version / Rollback ───────────────────────────────

def test_tnn_store_save_new_version(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="version_tnn", outputs=["out"], version=1)
    tnn = DummyTNN(desc)
    save_dir = tmp_path / "src_v"
    tnn.save(save_dir)

    store = TNNStore(tmp_path / "store_v")
    store.register(
        tnn_id="version_tnn",
        descriptor=desc,
        weights_path=str(save_dir / "weights.pt"),
        structure_version="1",
    )

    # Save new version
    desc2 = TNNDescriptor(tnn_id="version_tnn", outputs=["out"], version=2)
    tnn2 = DummyTNN(desc2)
    saved_path = store.save_new_version("version_tnn", tnn2, new_version=2)
    assert saved_path

    entry = store._entries.get("version_tnn")
    assert entry is not None
    assert entry.version == 2


def test_tnn_store_rollback(tmp_path: Path) -> None:
    desc = TNNDescriptor(tnn_id="rb_tnn", outputs=["out"], version=1)
    tnn = DummyTNN(desc)
    save_dir = tmp_path / "src_rb"
    tnn.save(save_dir)

    store = TNNStore(tmp_path / "store_rb")
    store.register(
        tnn_id="rb_tnn",
        descriptor=desc,
        weights_path=str(save_dir / "weights.pt"),
        structure_version="1",
    )

    # Save v2
    desc2 = TNNDescriptor(tnn_id="rb_tnn", outputs=["out"], version=2)
    tnn2 = DummyTNN(desc2)
    store.save_new_version("rb_tnn", tnn2, new_version=2)

    # Rollback to v1
    ok = store.rollback("rb_tnn", 1)
    assert ok is True
    entry = store._entries.get("rb_tnn")
    assert entry is not None
    assert entry.version == 1

    # Rollback to nonexistent version fails
    ok = store.rollback("rb_tnn", 99)
    assert ok is False


# ── Parameter Count ───────────────────────────────────────────

def test_tnn_parameter_count() -> None:
    desc = TNNDescriptor(tnn_id="param_test", outputs=["out"])
    tnn = DummyTNN(desc, input_dim=64, hidden_dim=32, output_dim=16)
    params = tnn.get_parameters()
    assert params["total_params"] > 0
    assert params["trainable_params"] > 0
    # DummyTNN: fc1 (64→32) + fc2 (32→16) = 64*32+32 + 32*16+16 = 2080+32+512+16
    expected = 64 * 32 + 32 + 32 * 16 + 16
    assert params["total_params"] == expected


# ── TNNStore _scan ────────────────────────────────────────────

def test_tnn_store_scan(tmp_path: Path) -> None:
    # Pre-create a TNN directory structure manually
    weights_dir = tmp_path / "scan_store" / "TNNweights" / "scanned_tnn" / "v1"
    weights_dir.mkdir(parents=True, exist_ok=True)

    desc = TNNDescriptor(tnn_id="scanned_tnn", outputs=["out"], version=1)
    tnn = DummyTNN(desc)
    tnn.save(weights_dir)

    # Now create TNNStore pointing at this base_dir — _scan should find it
    store = TNNStore(tmp_path / "scan_store")
    available = store.list_available()
    assert "scanned_tnn" in available

    d = store.get_descriptor("scanned_tnn")
    assert d is not None
    assert d.tnn_id == "scanned_tnn"
    assert d.version == 1
