"""Fine-grained RAM trace of a candidate struct region during a real descent.

Polls every word in a window around the wall-pinned lateral candidates
(0x0217F600-0x0217F900) every `every` env-steps while sliding with UP held,
printing the words that change -- to identify forward/vertical coordinates
(monotone), lateral position, velocities (finite differences) and death flags.

Usage:
    python -m src.ram_trace --state ds2 --steps 400 --every 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from desmume.emulator import DeSmuME  # noqa: E402
from desmume.controls import Keys, keymask  # noqa: E402

RAM_BASE = 0x02000000


def step_hold(emu, keys, steps, frameskip=4):
    for k in keys:
        emu.input.keypad_add_key(keymask(k))
    for _ in range(steps):
        for _ in range(frameskip):
            emu.cycle()
    for k in keys:
        emu.input.keypad_rm_key(keymask(k))


def read_words(emu, lo, hi):
    raw = bytes(emu.memory.read(lo, hi, 1, False))
    return np.frombuffer(raw, dtype="<u4").copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="ds2")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--lo", type=lambda x: int(x, 16), default=0x0217F600)
    ap.add_argument("--hi", type=lambda x: int(x, 16), default=0x0217F900)
    ap.add_argument("--keys", default="up", help="comma list: up,down,left,right")
    args = ap.parse_args()

    rom = os.path.join(ROOT, "data/Super Mario 64 DS (USA) (Rev 1).nds")
    state = os.path.join(ROOT, f"data/Super Mario 64 DS (USA) (Rev 1).{args.state}")
    emu = DeSmuME()
    emu.open(rom)
    emu.savestate.load_file(state)

    keymap = {"up": Keys.KEY_UP, "down": Keys.KEY_DOWN,
              "left": Keys.KEY_LEFT, "right": Keys.KEY_RIGHT,
              "jump": Keys.KEY_B}
    keys = [keymap[k] for k in args.keys.split(",") if k in keymap]

    n_polls = args.steps // args.every + 1
    polls = []
    brights = []
    for p in range(n_polls):
        w = read_words(emu, args.lo, args.hi)
        polls.append(w)
        import cv2
        frame = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8) \
            .reshape((384, 256, 4))[:192, :, [2, 1, 0]]
        brights.append(float(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).mean()))
        if p < n_polls - 1:
            step_hold(emu, keys, args.every)
    emu.destroy()

    P = np.stack(polls)                      # (n_polls, W) u32
    S = P.view(np.int32)
    changed = np.any(P != P[0], 0)
    print(f"{n_polls} polls, brightness: {[round(b,1) for b in brights]}", flush=True)
    print(f"changing words in [0x{args.lo:X}, 0x{args.hi:X}): {int(changed.sum())}",
          flush=True)
    for i in np.nonzero(changed)[0]:
        addr = args.lo + i * 4
        ser = S[:, i]
        d = ser[1:] - ser[:-1]
        tags = []
        if (d > 0).all():
            tags.append("MONO+")
        if (d < 0).all():
            tags.append("MONO-")
        if len(d) > 3 and (np.abs(d[1:] - d[:-1]) <= 2).all():
            tags.append(f"lin(d~{int(np.median(d))})")
        print(f"0x{addr:08X} s32={ser.tolist()} {' '.join(tags)}", flush=True)
        print(f"            /4096={[round(float(x)/4096.0, 3) for x in ser]}",
              flush=True)


if __name__ == "__main__":
    main()
