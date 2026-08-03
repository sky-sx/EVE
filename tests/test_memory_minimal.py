from __future__ import annotations

import hashlib
import json

import numpy as np

from eve.memory.memorizer import CatalogRecord, MemoryUnit
from eve.memory.memorizer import Memorizer


def test_memory_crud_and_minimal_retrieval(tmp_path):
    memory = Memorizer(tmp_path)
    first = memory.create({"text": "red ball"}, "observation")
    second = memory.create({"text": "blue square"}, "observation")
    result = memory.create({"ok": True}, "output_result")

    assert memory.read(first) == {"text": "red ball"}
    assert memory.search(payload_type="output_result") == [result]
    assert memory.search(keyword="blue") == [second]
    memory.load_to_mtm(first)
    memory.persist_to_ltm(first)
    assert memory.mtm == {first}
    assert memory.ltm == {first}
    assert first in memory.stm
    assert memory.delete(second)
    assert memory.read(second) is None

    reloaded = Memorizer(tmp_path)
    assert reloaded.read(first) == {"text": "red ball"}
    assert first in reloaded.catalog


def test_payloads_views_events_and_catalog_restore_without_payload_copies(tmp_path):
    memory = Memorizer(tmp_path / "memory")
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image_id = memory.create(image, "screen_frame")
    json_id = memory.create({"kind": "output", "ok": True}, "output_result")
    memory.load_to_mtm(image_id)
    memory.persist_to_ltm(image_id)
    event = memory.create_event([image_id, json_id], summary="action closure")

    assert isinstance(memory.get_record(image_id), CatalogRecord)
    unit = memory.get_unit(json_id)
    assert isinstance(unit, MemoryUnit)
    assert unit.payload == {"kind": "output", "ok": True}
    assert np.array_equal(memory.read(image_id), image)
    assert len(list((tmp_path / "memory" / "objects").glob(f"{image_id}.*"))) == 1

    reloaded = Memorizer(tmp_path / "memory")
    assert reloaded.mtm == {image_id}
    assert reloaded.ltm == {image_id}
    assert reloaded.read_event(event.event_id).memory_ids == (image_id, json_id)
    assert json.loads((tmp_path / "memory" / "views" / "ltm.json").read_text()) == {
        "memory_ids": [image_id]
    }


def test_verified_legacy_memory_copy_keeps_source_and_ids(tmp_path):
    source = tmp_path / "runs" / "memory"
    legacy = Memorizer(source)
    memory_id = legacy.create({"legacy": True}, "json")
    before = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*") if path.is_file()
    }
    destination = tmp_path / "eve" / "memory"
    result = Memorizer.migrate_legacy_directory(source, destination)

    assert result["migrated"] is True
    assert source.is_dir()
    assert result["backup"]
    migrated = Memorizer(destination)
    assert migrated.read(memory_id) == {"legacy": True}
    after = {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.rglob("*") if path.is_file()
        and str(path.relative_to(destination)) in before
    }
    assert after == before
