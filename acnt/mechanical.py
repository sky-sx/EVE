"""Mechanical records are external observations, never feedback to Core."""

from copy import deepcopy
import json
from pathlib import Path


class MechanicalLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.records: list[dict] = []
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        record = deepcopy(record)
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
