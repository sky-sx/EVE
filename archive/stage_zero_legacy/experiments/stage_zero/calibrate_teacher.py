"""Offline DeepSeek visual Teacher calibration. Never imported by the runner."""

import argparse
import base64
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import struct
import time
import urllib.error
import urllib.request
import zlib

import torch

from .environment import ACTIONS, DEFAULT_TEACHER_TABLE, TeacherTable, VisualEnvironment

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
PROMPT_VERSION = "stage-zero-visual-teacher-v1"
SYSTEM_PROMPT = """你是 EVE 幼年阶段的 Teacher。
你只评价截至当前时刻已经发生的 ReadIn / ReadOut 信息。
不要使用未来信息，不要假设额外环境结果，不要提供正确动作、梯度、子奖励或置信度。

当前 Stage Zero 中，视觉出现哪个字母，就表示希望 EVE 按下对应字母按钮；
绿色 stimulus 表示 MOUSE_LEFT。
27 个动作除当前应响应的按钮外没有额外优先级，错误按钮之间等价。

输出一个 [0,1] 的整体 Goodness：
0 表示明显背离当前目标；
1 表示当前行为完全符合目标；
允许中间值表达部分正确。
不要为了让神经网络更容易训练而人为设计 reward shaping。

只输出数值。"""
REQUEST_SETTINGS = {"model": MODEL, "temperature": 1.0, "max_tokens": 128,
                    "thinking": {"type": "disabled"}, "stream": False}


def read_api_key(path=None):
    """Read-only in memory. The key and its source never enter persisted metadata."""
    if path is not None:
        try:
            key = Path(path).read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            raise ValueError("Cannot read API key file") from None
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or any(ch.isspace() for ch in key):
        raise ValueError("Missing/invalid DeepSeek API key: set DEEPSEEK_API_KEY or pass --api-key-file")
    return key


def parse_score(content):
    if not isinstance(content, str) or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", content.strip(), flags=re.ASCII) is None:
        raise ValueError("Teacher must return only a numeric scalar")
    score = float(content.strip())
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Teacher score must be finite and in [0, 1]")
    return score


def png_bytes(frame):
    """Losslessly encode actual render() RGB pixels using stdlib PNG chunks."""
    if frame.dtype != torch.float32 or frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("Expected actual RGB float32 Eye frame")
    height, width = frame.shape[1:]
    pixels = (frame.cpu().permute(1, 2, 0).contiguous() * 255).to(torch.uint8)
    # Tensor storage -> bytes needs no numpy or Pillow dependency.
    raw = ctypes.string_at(pixels.data_ptr(), pixels.numel())
    scanlines = b"".join(b"\x00" + raw[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def build_messages(image, actions):
    if actions.dtype != torch.bool or actions.shape != (27,):
        raise ValueError("Expected 27-bit Hand ReadOut")
    observed = {"button_order": list(ACTIONS), "hand_readout": actions.tolist(),
                "pressed_buttons": [name for name, bit in zip(ACTIONS, actions.tolist()) if bit]}
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii"), "detail": "original"}},
                {"type": "text", "text": "当前实际视觉 ReadIn 与已经发生的 Hand ReadOut：\n" + json.dumps(observed, ensure_ascii=False)}]}]


def request_score(key, messages, *, max_attempts=3, timeout=120):
    payload = json.dumps({**REQUEST_SETTINGS, "messages": messages}, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, max_attempts + 1):
        try:
            request = urllib.request.Request(BASE_URL + "/chat/completions", data=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete Teacher answer")
            score = parse_score(choice["message"]["content"])
            return score, {"response_id": result.get("id"), "response_model": result.get("model"),
                           "attempts": attempt, "usage": result.get("usage"),
                           "request_sha256": hashlib.sha256(payload).hexdigest()}
        except urllib.error.HTTPError as error:
            # Never print response bodies, request headers, or exception reprs.
            reason = f"HTTP {error.code}"
            if error.code in (400, 401, 402, 403, 404):
                raise RuntimeError("DeepSeek calibration failed: " + reason) from None
        except (ValueError, KeyError, IndexError, TypeError):
            reason = "invalid Teacher response"
        except (urllib.error.URLError, TimeoutError, OSError):
            reason = "transport failure"
        if attempt < max_attempts:
            time.sleep(min(attempt, 3))
    raise RuntimeError(f"DeepSeek calibration failed after {max_attempts} attempts: {reason}")


def calibrate(key, *, samples_per_cell=5, seed=20260921, max_attempts=3, progress=None):
    if type(samples_per_cell) is not int or samples_per_cell < 1 or max_attempts < 1:
        raise ValueError("samples per cell and attempts must be positive")
    rng = random.Random(seed)
    environment = VisualEnvironment()
    cache, cells = {}, {}
    started = datetime.now(timezone.utc).isoformat()
    for correct in (0, 1):
        for wrong in range(27):
            scores, instances = [], []
            for _ in range(samples_per_cell):
                target = rng.randrange(27)
                actions = torch.zeros(27, dtype=torch.bool)
                actions[rng.sample([i for i in range(27) if i != target], wrong)] = True
                actions[target] = bool(correct)
                if target not in cache:
                    cache[target] = png_bytes(environment.render(target))
                image = cache[target]
                score, info = request_score(key, build_messages(image, actions), max_attempts=max_attempts)
                scores.append(score)
                # Local audit labels are never part of the API request.
                instances.append({"target": target, "action_bits": actions.tolist(),
                                  "image_sha256": hashlib.sha256(image).hexdigest(), **info})
            cells[f"{correct},{wrong}"] = {
                "mean": statistics.mean(scores), "sample_count": len(scores), "raw_scores": scores,
                "variance": statistics.pvariance(scores), "std": statistics.pstdev(scores),
                "model": MODEL, "prompt_version": PROMPT_VERSION, "instances": instances}
            if progress:
                progress(correct, wrong, cells[f"{correct},{wrong}"])
    return {"schema_version": 1, "frozen": True, "calibration": {
        "model": MODEL, "base_url": BASE_URL, "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT, "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "request_settings": REQUEST_SETTINGS, "seed": seed, "samples_per_cell": samples_per_cell,
        "max_attempts": max_attempts, "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "variance_convention": "population",
        "environment": environment.config(), "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("environment.py", "calibrate_teacher.py")}}, "cells": cells}


def table_report(table):
    cells = json.loads(table.raw)["cells"]
    lines = ["# Frozen Teacher calibration", "", f"SHA256: `{table.sha256}`", "",
             f"Model: {table.metadata['model']}; samples/cell: {table.metadata['samples_per_cell']}", "",
             "| k | M[1,k] | std[1,k] | M[0,k] | std[0,k] |", "|---|---|---|---|---|"]
    for k in range(27):
        a, b = cells[f"1,{k}"], cells[f"0,{k}"]
        lines.append(f"| {k} | {a['mean']:.9g} | {a['std']:.9g} | {b['mean']:.9g} | {b['std']:.9g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_TEACHER_TABLE)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--samples-per-cell", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        key = read_api_key(args.api_key_file)
        if args.output.exists():
            raise ValueError("Output already exists; frozen tables are never overwritten")
        torch.set_num_threads(1)
        def progress(c, k, cell):
            print(json.dumps({"cell": [c, k], "mean": cell["mean"], "std": cell["std"], "samples": cell["sample_count"]}), flush=True)
        data = calibrate(key, samples_per_cell=args.samples_per_cell, seed=args.seed,
                         max_attempts=args.max_attempts, progress=progress)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        table = TeacherTable.load(args.output)
        report = table_report(table)
        args.output.with_suffix(".md").write_text(report, encoding="utf-8")
        print(report, flush=True)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
