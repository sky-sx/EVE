"""
EVE 运行时状态管理器。

持有 world/myself/blackboard 三体，支持快照保存/加载、
LLM 结构化输出更新。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from eve.config import EVEConfig
from eve.state import Blackboard, MyselfState, WorldState


class RuntimeStateManager:
    """持有并管理 WorldState、MyselfState、Blackboard。"""

    def __init__(self, config: EVEConfig) -> None:
        self.world = WorldState()
        self.myself = MyselfState()
        self.blackboard = Blackboard()
        self._config = config

    def save_snapshot(self, path: Path) -> None:
        """将 world.md、self.md、blackboard.md 写入指定目录。"""
        path.mkdir(parents=True, exist_ok=True)

        world_lines = [
            f"# World State",
            f"",
            f"- **scene**: {self.world.scene}",
            f"- **sub_scene**: {self.world.sub_scene}",
            f"- **active_window**: {self.world.active_window}",
            f"- **visible_objects**: {', '.join(self.world.visible_objects)}",
            f"- **detected_text**: {self.world.detected_text}",
            f"- **uncertainty**: {self.world.uncertainty}",
            f"- **updated_at_ns**: {self.world.updated_at_ns}",
        ]
        if self.world.visual_results:
            world_lines.append(f"- **visual_results**:")
            for k, v in self.world.visual_results.items():
                world_lines.append(f"  - {k}: {v}")
        (path / "world.md").write_text("\n".join(world_lines), encoding="utf-8")

        myself_lines = [
            f"# Myself State",
            f"",
            f"- **what_im_thinking**: {self.myself.what_im_thinking}",
            f"- **current_task**: {self.myself.current_task}",
            f"- **task_progress**: {self.myself.task_progress}",
            f"- **loaded_tnn**: {', '.join(self.myself.loaded_tnn)}",
            f"- **available_tnn_summary**: {', '.join(self.myself.available_tnn_summary)}",
            f"- **control_summary**: {self.myself.control_summary}",
            f"- **updated_at_ns**: {self.myself.updated_at_ns}",
        ]
        if self.myself.resource_status:
            myself_lines.append(f"- **resource_status**: {json.dumps(self.myself.resource_status, ensure_ascii=False)}")
        if self.myself.hormone_levels:
            myself_lines.append(f"- **hormone_levels**: {json.dumps(self.myself.hormone_levels, ensure_ascii=False)}")
        if self.myself.tendencies:
            myself_lines.append(f"- **tendencies**: {json.dumps(self.myself.tendencies, ensure_ascii=False)}")
        (path / "self.md").write_text("\n".join(myself_lines), encoding="utf-8")

        bb_lines = ["# Blackboard"]
        for kind, entries in self.blackboard.entries.items():
            bb_lines.append(f"")
            bb_lines.append(f"## {kind}")
            for e in entries:
                bb_lines.append(f"- [{e.entry_id}] producer={e.producer} valid_until={e.valid_until_ns} payload={e.payload}")
        (path / "blackboard.md").write_text("\n".join(bb_lines), encoding="utf-8")

    def load_snapshot(self, path: Path) -> bool:
        """从快照目录加载 world/self/blackboard，成功返回 True。"""
        if not path.exists():
            return False

        world_path = path / "world.md"
        if not world_path.exists():
            return False

        now_ns = time.monotonic_ns()
        text = world_path.read_text(encoding="utf-8")
        self.world = WorldState(updated_at_ns=now_ns)
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("- **scene**:"):
                self.world.scene = _extract_md_value(line)
            elif line.startswith("- **sub_scene**:"):
                self.world.sub_scene = _extract_md_value(line)
            elif line.startswith("- **active_window**:"):
                self.world.active_window = _extract_md_value(line)
            elif line.startswith("- **visible_objects**:"):
                val = _extract_md_value(line)
                self.world.visible_objects = [v.strip() for v in val.split(",") if v.strip()]
            elif line.startswith("- **detected_text**:"):
                self.world.detected_text = _extract_md_value(line)
            elif line.startswith("- **uncertainty**:"):
                self.world.uncertainty = _extract_md_value(line)

        self_path = path / "self.md"
        if self_path.exists():
            text = self_path.read_text(encoding="utf-8")
            self.myself = MyselfState(updated_at_ns=now_ns)
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("- **what_im_thinking**:"):
                    self.myself.what_im_thinking = _extract_md_value(line)
                elif line.startswith("- **current_task**:"):
                    self.myself.current_task = _extract_md_value(line)
                elif line.startswith("- **task_progress**:"):
                    self.myself.task_progress = _extract_md_value(line)
                elif line.startswith("- **loaded_tnn**:"):
                    val = _extract_md_value(line)
                    self.myself.loaded_tnn = [v.strip() for v in val.split(",") if v.strip()]
                elif line.startswith("- **available_tnn_summary**:"):
                    val = _extract_md_value(line)
                    self.myself.available_tnn_summary = [v.strip() for v in val.split(",") if v.strip()]
                elif line.startswith("- **control_summary**:"):
                    self.myself.control_summary = _extract_md_value(line)
                elif line.startswith("- **resource_status**:"):
                    try:
                        self.myself.resource_status = json.loads(_extract_md_value(line))
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("- **hormone_levels**:"):
                    try:
                        self.myself.hormone_levels = json.loads(_extract_md_value(line))
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("- **tendencies**:"):
                    try:
                        self.myself.tendencies = json.loads(_extract_md_value(line))
                    except json.JSONDecodeError:
                        pass

        bb_path = path / "blackboard.md"
        if bb_path.exists():
            self.blackboard = Blackboard()

        return True

    def update_from_llm_world(self, llm_output: dict[str, Any]) -> None:
        """从 LLM 结构化输出更新 world。"""
        now_ns = time.monotonic_ns()
        self.world.scene = llm_output.get("scene", self.world.scene)
        self.world.sub_scene = llm_output.get("sub_scene", self.world.sub_scene)
        self.world.active_window = llm_output.get("active_window", self.world.active_window)
        if "visible_objects" in llm_output:
            self.world.visible_objects = list(llm_output["visible_objects"])
        self.world.detected_text = llm_output.get("detected_text", self.world.detected_text)
        if "visual_results" in llm_output:
            self.world.visual_results = dict(llm_output["visual_results"])
        self.world.uncertainty = llm_output.get("uncertainty", self.world.uncertainty)
        self.world.updated_at_ns = now_ns

    def update_from_llm_myself(self, llm_output: dict[str, Any]) -> None:
        """从 LLM 结构化输出更新 myself。"""
        now_ns = time.monotonic_ns()
        self.myself.what_im_thinking = llm_output.get("what_im_thinking", self.myself.what_im_thinking)
        self.myself.current_task = llm_output.get("current_task", self.myself.current_task)
        self.myself.task_progress = llm_output.get("task_progress", self.myself.task_progress)
        if "loaded_tnn" in llm_output:
            self.myself.loaded_tnn = list(llm_output["loaded_tnn"])
        if "available_tnn_summary" in llm_output:
            self.myself.available_tnn_summary = list(llm_output["available_tnn_summary"])
        if "resource_status" in llm_output:
            self.myself.resource_status = dict(llm_output["resource_status"])
        if "hormone_levels" in llm_output:
            self.myself.hormone_levels = dict(llm_output["hormone_levels"])
        if "tendencies" in llm_output:
            self.myself.tendencies = dict(llm_output["tendencies"])
        self.myself.control_summary = llm_output.get("control_summary", self.myself.control_summary)
        self.myself.updated_at_ns = now_ns


def _extract_md_value(line: str) -> str:
    """从 markdown 行提取 `- **key**: value` 中的 value。"""
    idx = line.find(":")
    if idx == -1:
        return ""
    return line[idx + 1:].strip()
