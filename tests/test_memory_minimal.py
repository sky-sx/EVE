from __future__ import annotations

from eve.memory.memorizer import Memorizer


def test_memory_crud_and_minimal_retrieval(tmp_path):
    memory = Memorizer(tmp_path)
    first = memory.create({"text": "red ball"}, "observation")
    second = memory.create({"text": "blue square"}, "observation")
    result = memory.create({"ok": True}, "output_result")

    assert memory.read(first) == {"text": "red ball"}
    assert memory.search(payload_type="output_result") == [result]
    assert memory.search(keyword="blue") == [second]
    memory.promote_to_mtm(first)
    assert memory.mtm == {first}
    assert memory.delete(second)
    assert memory.read(second) is None

    reloaded = Memorizer(tmp_path)
    assert reloaded.read(first) == {"text": "red ball"}
    assert first in reloaded.catalog
