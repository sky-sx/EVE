"""Small, explicit configuration for the active EVE runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EVEConfig:
    run_dir: Path = Path("runs")
    memory_dir: Path = Path("runs/memory")
    snapshot_path: Path = Path("runs/state_snapshot.json")
    loop_interval_s: float = 0.02
    screen_fps: float = 10.0
    cursor_hz: float = 20.0

    @classmethod
    def default(cls) -> "EVEConfig":
        return cls()
