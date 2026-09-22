"""Deterministic 1080p RGB stimuli; target and reward stay outside ACNT."""
import hashlib
import json
import random
import string

import torch

ACTIONS = tuple(string.ascii_uppercase) + ("MOUSE_LEFT",)
GLYPHS = {'A': ('01110', '10001', '10001', '11111', '10001', '10001', '10001'), 'B': ('11110', '10001', '10001', '11110', '10001', '10001', '11110'), 'C': ('01111', '10000', '10000', '10000', '10000', '10000', '01111'), 'D': ('11110', '10001', '10001', '10001', '10001', '10001', '11110'), 'E': ('11111', '10000', '10000', '11110', '10000', '10000', '11111'), 'F': ('11111', '10000', '10000', '11110', '10000', '10000', '10000'), 'G': ('01111', '10000', '10000', '10111', '10001', '10001', '01111'), 'H': ('10001', '10001', '10001', '11111', '10001', '10001', '10001'), 'I': ('11111', '00100', '00100', '00100', '00100', '00100', '11111'), 'J': ('00111', '00010', '00010', '00010', '10010', '10010', '01100'), 'K': ('10001', '10010', '10100', '11000', '10100', '10010', '10001'), 'L': ('10000', '10000', '10000', '10000', '10000', '10000', '11111'), 'M': ('10001', '11011', '10101', '10101', '10001', '10001', '10001'), 'N': ('10001', '11001', '10101', '10011', '10001', '10001', '10001'), 'O': ('01110', '10001', '10001', '10001', '10001', '10001', '01110'), 'P': ('11110', '10001', '10001', '11110', '10000', '10000', '10000'), 'Q': ('01110', '10001', '10001', '10001', '10101', '10010', '01101'), 'R': ('11110', '10001', '10001', '11110', '10100', '10010', '10001'), 'S': ('01111', '10000', '10000', '01110', '00001', '00001', '11110'), 'T': ('11111', '00100', '00100', '00100', '00100', '00100', '00100'), 'U': ('10001', '10001', '10001', '10001', '10001', '10001', '01110'), 'V': ('10001', '10001', '10001', '10001', '10001', '01010', '00100'), 'W': ('10001', '10001', '10001', '10101', '10101', '10101', '01010'), 'X': ('10001', '10001', '01010', '00100', '01010', '10001', '10001'), 'Y': ('10001', '10001', '01010', '00100', '00100', '00100', '00100'), 'Z': ('11111', '00001', '00010', '00100', '01000', '10000', '11111')}
HEIGHT, WIDTH, PIXEL_SIZE = 1080, 1920, 64
DELAYS_MS = (250, 500, 1000)


def environment_config():
    return {
        "actions": ACTIONS, "shape": [3, HEIGHT, WIDTH], "dtype": "float32",
        "range": [0.0, 1.0], "channel_order": "RGB",
        "background_rgb": [0, 0, 0], "red_rgb": [1, 0, 0],
        "blue_rgb": [0, 0, 1], "green_rgb": [0, 1, 0],
        "font": "builtin_5x7", "font_table": GLYPHS,
        "pixel_size": PIXEL_SIZE, "augmentation": None,
    }


def render(target, color="red", device="cpu"):
    if type(target) is not int or target not in range(27):
        raise ValueError("target must be a class index")
    if target != 26 and color not in ("red", "blue"):
        raise ValueError("letter color must be red or blue")
    frame = torch.zeros((3, HEIGHT, WIDTH), dtype=torch.float32, device=device)
    if target == 26:
        frame[1].fill_(1)
        return frame
    glyph = torch.tensor([[int(c) for c in row] for row in GLYPHS[ACTIONS[target]]],
                         dtype=torch.float32, device=device)
    glyph = glyph.repeat_interleave(PIXEL_SIZE, 0).repeat_interleave(PIXEL_SIZE, 1)
    top, left = (HEIGHT - glyph.shape[0]) // 2, (WIDTH - glyph.shape[1]) // 2
    frame[0 if color == "red" else 2, top:top+glyph.shape[0], left:left+glyph.shape[1]] = glyph
    return frame


def stimulus_hash(frame):
    return hashlib.sha256(frame.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def balanced_targets(count, rng):
    order = []
    while len(order) < count:
        cycle = list(range(27))
        rng.shuffle(cycle)
        order.extend(cycle[:count-len(order)])
    return order


def balanced_delays(count, rng):
    order = []
    while len(order) < count:
        cycle = list(DELAYS_MS)
        rng.shuffle(cycle)
        order.extend(cycle[:count-len(order)])
    return order


def balanced_colors(targets, rng):
    # Each letter's repeated presentations alternate after one independent draw.
    first = {letter: rng.randrange(2) for letter in range(26)}
    seen = {letter: 0 for letter in range(26)}
    colors = []
    for target in targets:
        if target == 26:
            colors.append("green")
        else:
            colors.append(("red", "blue")[(first[target] + seen[target]) % 2])
            seen[target] += 1
    return colors


def exact_goodness(target, action):
    return float(bool(action[target]) and int(action.sum()) == 1)


def phase_schedule(count, seed, phase_index):
    # Four disjoint Python streams; Hand uses a fifth independent torch generator.
    bases = {name: seed * 1000003 + phase_index * 10007 + offset
             for offset, name in enumerate(("target", "color", "delay", "hand"), 1)}
    targets = balanced_targets(count, random.Random(bases["target"]))
    colors = balanced_colors(targets, random.Random(bases["color"]))
    delays = balanced_delays(count, random.Random(bases["delay"]))
    return list(zip(targets, colors, delays)), bases
