"""
EVE 配置系统。

提供 EVEConfig dataclass，支持默认值、JSON 文件加载。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HormoneDefaults:
    """激素基础水平与恢复率。"""
    cortisol: float = 0.1
    adrenaline: float = 0.05
    dopamine: float = 0.3
    serotonin: float = 0.4
    oxytocin: float = 0.2
    endorphin: float = 0.15
    recovery_rate: float = 0.01  # 每秒恢复率


@dataclass
class EVEConfig:
    """EVE 项目全局配置。"""

    # ── 路径 ──
    runs_dir: str = "runs"
    memory_dir: str = "eve/memory/LTM"
    tnn_weights_dir: str = "eve/memory/TNNweights"
    snapshot_dir: str = "snapshots"
    log_dir: str = "logs"

    # ── 捕获频率 ──
    screen_fps: int = 30
    cursor_hz: int = 60

    # ── Safegate ──
    freeze_duration_s: float = 5.0

    # ── 激素 ──
    hormones: HormoneDefaults = field(default_factory=HormoneDefaults)

    # ── 主循环 ──
    llm_loop_min_s: float = 10.0
    llm_loop_max_s: float = 20.0

    # ── 输出 ──
    default_mode: str = "disabled"

    # ── 模型路径（默认 None，未配置） ──
    local_llm_path: Optional[str] = None
    vlm_path: Optional[str] = None
    yolo_path: Optional[str] = None

    @classmethod
    def from_json(cls, path: str | Path) -> "EVEConfig":
        """从 JSON 文件加载配置，缺失字段使用默认值。"""
        config_path = Path(path)
        config = cls()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _apply_json_fields(config, data)
        return config

    @classmethod
    def default(cls) -> "EVEConfig":
        """返回全默认配置。"""
        return cls()


def _apply_json_fields(config: EVEConfig, data: dict) -> None:
    for key, value in data.items():
        if key == "hormones":
            for hk, hv in value.items():
                if hasattr(config.hormones, hk):
                    setattr(config.hormones, hk, hv)
        elif hasattr(config, key):
            setattr(config, key, value)
