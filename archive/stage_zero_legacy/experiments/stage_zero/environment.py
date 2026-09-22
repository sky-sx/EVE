"""Fixed visual stimuli and scalar feedback for the Stage Zero experiment.

Class indices exist only on the environment side.  The network receives the
rendered RGB tensor; exact match is an audit metric, never learning feedback.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from dataclasses import dataclass

import random
import string

import torch


ACTIONS = tuple(string.ascii_uppercase) + ("MOUSE_LEFT",)

# This table is the complete font specification, not a system-font reference.
# Every glyph occupies the same 5 x 7 cell with nearest-neighbour pixel scaling.
GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _validate_target(target: int) -> None:
    if isinstance(target, bool) or not isinstance(target, int):
        raise TypeError("target must be an integer class index")
    if not 0 <= target < len(ACTIONS):
        raise ValueError("target must be in [0, 26]")


class VisualEnvironment:
    """Deterministic, dependency-free RGB Eye frames at the real Eye resolution."""

    height = 1080
    width = 1920
    pixel_size = 64

    def render(self, target: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
        _validate_target(target)
        frame = torch.zeros((3, self.height, self.width), dtype=torch.float32, device=device)
        if target == 26:
            frame[1].fill_(1.0)
            return frame
        glyph = torch.tensor(
            [[int(bit) for bit in row] for row in GLYPHS[ACTIONS[target]]],
            dtype=torch.float32,
            device=device,
        ).repeat_interleave(self.pixel_size, dim=0).repeat_interleave(self.pixel_size, dim=1)
        top = (self.height - glyph.shape[0]) // 2
        left = (self.width - glyph.shape[1]) // 2
        frame[:, top : top + glyph.shape[0], left : left + glyph.shape[1]] = glyph
        return frame

    def config(self) -> dict:
        """Serializable specification sufficient to reconstruct every image."""
        return {
            "actions": list(ACTIONS),
            "shape": [3, self.height, self.width],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "channel_order": "RGB",
            "background_rgb": [0.0, 0.0, 0.0],
            "letter_rgb": [1.0, 1.0, 1.0],
            "click_rgb": [0.0, 1.0, 0.0],
            "font": "stage_zero_builtin_5x7",
            "font_table": {letter: list(rows) for letter, rows in GLYPHS.items()},
            "pixel_size": self.pixel_size,
            "glyph_size": [7 * self.pixel_size, 5 * self.pixel_size],
            "glyph_top_left": [(self.height - 7 * self.pixel_size) // 2,
                               (self.width - 5 * self.pixel_size) // 2],
            "augmentation": None,
            "goodness": "frozen Teacher table mean indexed by (correct_pressed, wrong_count)",
        }


def balanced_targets(episodes: int, seed: int) -> list[int]:
    """Shuffle each complete class cycle using a separate local Python RNG.

    Incomplete final cycles are allowed for short smoke runs.  For all lengths,
    the most and least frequent class counts differ by at most one.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer")
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    rng = random.Random(seed)
    targets: list[int] = []
    while len(targets) < episodes:
        cycle = list(range(len(ACTIONS)))
        rng.shuffle(cycle)
        targets.extend(cycle[: episodes - len(targets)])
    return targets


def action_quality_bucket(target: int, actions: torch.Tensor) -> tuple[int, int]:
    """Environment-only bucket for a strict 27-bit boolean Hand action."""
    _validate_target(target)
    if not isinstance(actions, torch.Tensor):
        raise TypeError("actions must be a torch.Tensor")
    if actions.shape != (len(ACTIONS),):
        raise ValueError("actions must have shape [27]")
    if actions.dtype != torch.bool:
        raise TypeError("actions must have dtype torch.bool")
    correct = int(actions[target].item())
    return correct, int(actions.sum().item()) - correct


def exact_goodness(target: int, actions: torch.Tensor) -> float:
    """Exact-one-hot audit metric only; never used as learning feedback."""
    return float(action_quality_bucket(target, actions) == (1, 0))


DEFAULT_TEACHER_TABLE = Path(__file__).with_name("teacher_goodness.json")


def _score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Teacher score must be a finite number in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class TeacherTable:
    """Immutable means for learning; original bytes retain audit-only metadata."""

    means: tuple[tuple[float, ...], ...]
    raw: bytes
    sha256: str

    @classmethod
    def load(cls, path=DEFAULT_TEACHER_TABLE):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data.get("schema_version") != 1 or data.get("frozen") is not True:
            raise ValueError("Teacher table must be complete and frozen schema version 1")
        metadata = data["calibration"]
        if not all(metadata.get(key) for key in ("model", "prompt_version", "system_prompt")):
            raise ValueError("Teacher calibration metadata missing")
        cells = data["cells"]
        if set(cells) != {f"{c},{k}" for c in (0, 1) for k in range(27)}:
            raise ValueError("Teacher table must contain exactly 54 cells")
        means = [[], []]
        for c in (0, 1):
            for k in range(27):
                cell = cells[f"{c},{k}"]
                mean = _score(cell["mean"])
                scores = [_score(x) for x in cell["raw_scores"]]
                if type(cell["sample_count"]) is not int or cell["sample_count"] < 1 or len(scores) != cell["sample_count"]:
                    raise ValueError("Teacher sample count mismatch")
                for name, expected in (("mean", statistics.mean(scores)),
                                       ("variance", statistics.pvariance(scores)),
                                       ("std", statistics.pstdev(scores))):
                    value = cell[name]
                    if type(value) not in (float, int) or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
                        raise ValueError(f"Teacher {name} disagrees with raw scores")
                if cell.get("model") != metadata["model"] or cell.get("prompt_version") != metadata["prompt_version"]:
                    raise ValueError("Teacher cell provenance mismatch")
                means[c].append(mean)
        return cls(tuple(tuple(row) for row in means), raw, hashlib.sha256(raw).hexdigest())

    def lookup(self, correct_pressed: int, wrong_count: int) -> float:
        if type(correct_pressed) is not int or correct_pressed not in (0, 1):
            raise ValueError("correct_pressed must be 0 or 1")
        if type(wrong_count) is not int or not 0 <= wrong_count <= 26:
            raise ValueError("wrong_count must be in [0, 26]")
        return self.means[correct_pressed][wrong_count]

    @property
    def metadata(self):
        return json.loads(self.raw)["calibration"]

    def save(self, path):
        Path(path).write_bytes(self.raw)
