"""Verify the SM64DS Mario physics struct at 0x0217F74C-0x0217F764.

Runs controlled phases (straight slide, hold LEFT, hold RIGHT, jump) and prints
each word's trace so the axes can be named:
  * lateral position: delta sign follows LEFT/RIGHT
  * forward speed: ~saturates at 409600 (100.0 units/s, the slide speed cap)
  * vertical: drops large during falls; clamps at -307200 (-75 kill plane)

Usage:
    python -m src.ram_verify --state ds2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from desmume.emulator import DeSmuME  # noqa: E402
from desmume.controls import Keys, keymask  # noqa: E402

WORDS = [0x0217F74C, 0x0217F750, 0x0217F754, 0x0217F75C, 0x0217F760, 0x0217F764]


def step_hold(emu, keys, steps, frameskip=4):
    for k in keys:
        emu.input.keypad_add_key(keymask(k))
    for _ in range(steps):
        for _ in range(frameskip):
            emu.cycle()
    for k in keys:
        emu.input.keypad_rm_key(keymask(k))


def poll(emu):
    return [emu.memory.read(a, a, 4, False) for a in WORDS]


def s32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="ds2")
    args = ap.parse_args()

    rom = os.path.join(ROOT, "data/Super Mario 64 DS (USA) (Rev 1).nds")
    state = os.path.join(ROOT, f"data/Super Mario 64 DS (USA) (Rev 1).{args.state}")
    emu = DeSmuME()
    emu.open(rom)
    emu.savestate.load_file(state)

    phases = [
        ("straight", [Keys.KEY_UP], 60),
        ("LEFT", [Keys.KEY_UP, Keys.KEY_LEFT], 60),
        ("straight", [Keys.KEY_UP], 40),
        ("RIGHT", [Keys.KEY_UP, Keys.KEY_RIGHT], 60),
        ("straight", [Keys.KEY_UP], 40),
        ("JUMP", [Keys.KEY_B], 30),
        ("straight", [Keys.KEY_UP], 40),
    ]
    rows = []
    step_hold(emu, [Keys.KEY_UP], 80)      # get up to slide speed
    for name, keys, n in phases:
        base = poll(emu)
        rows.append((name, "start", [s32(v) for v in base]))
        step_hold(emu, keys, n)
        after = poll(emu)
        rows.append((name, "end", [s32(v) for v in after]))
    emu.destroy()

    print(f"{'phase':<10}{'point':<7}" + "".join(f"{hex(a):>14}" for a in WORDS))
    for name, point, vals in rows:
        print(f"{name:<10}{point:<7}" + "".join(f"{v:>14}" for v in vals))
    print("\nper-phase deltas (end - start):")
    for i in range(0, len(rows), 2):
        name = rows[i][0]
        d = [rows[i + 1][2][j] - rows[i][2][j] for j in range(len(WORDS))]
        print(f"{name:<10}" + "".join(f"{v:>14}" for v in d))
    print("\nas /4096 (20.12 units):")
    for name, point, vals in rows:
        print(f"{name:<10}{point:<7}" + "".join(f"{v/4096.0:>14.2f}" for v in vals))


if __name__ == "__main__":
    main()
