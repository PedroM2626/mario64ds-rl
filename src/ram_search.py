"""Differential RAM search for Super Mario 64 DS: wall-pinning gate.

On slide courses Mario cannot stop (braking never fully halts him), so the
motion gate is wall-pinning instead: holding LEFT/RIGHT pins the lateral
coordinate against the wall while the forward descent continues. Motion state
words freeze along the pinned axis; timers keep their constant rate. Choose a
narrow enclosed chute as the search ground (ds2) -- ds1's opening is an open
slope where steering off means falling, not pinning.

Usage:
    python -m src.ram_search --state ds2
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

RAM_BASE, RAM_SIZE = 0x02000000, 4 * 1024 * 1024


def dump_ram(emu):
    return np.frombuffer(
        bytes(emu.memory.read(RAM_BASE, RAM_BASE + RAM_SIZE, 1, False)),
        dtype=np.uint8).copy()


def hold(emu, keys, steps, frameskip=4):
    for k in keys:
        emu.input.keypad_add_key(keymask(k))
    for _ in range(steps):
        for _ in range(frameskip):
            emu.cycle()
    for k in keys:
        emu.input.keypad_rm_key(keymask(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="ds2",
                    help="ds2 = Peach's Secret Slide: a narrow enclosed chute, "
                         "walls in reach of the line (ds1's opening is an open "
                         "slope where steering off means falling, not pinning)")
    ap.add_argument("--free-steps", type=int, default=120)
    ap.add_argument("--pin-steps", type=int, default=150,
                    help="long enough to reach the wall and settle on it")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    rom = os.path.join(ROOT, "data/Super Mario 64 DS (USA) (Rev 1).nds")
    state = os.path.join(ROOT, f"data/Super Mario 64 DS (USA) (Rev 1).{args.state}")
    emu = DeSmuME()
    emu.open(rom)
    emu.savestate.load_file(state)
    t0 = time.time()

    def brightness():
        import cv2
        frame = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8) \
            .reshape((384, 256, 4))[:192, :, [2, 1, 0]]
        return float(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).mean())

    hold(emu, [Keys.KEY_UP], 200)          # warm-up past any intro scripting
    dumps, pinned, brights = [], [], []
    for r in range(args.rounds):
        for pin_key in (Keys.KEY_LEFT, Keys.KEY_RIGHT):
            hold(emu, [Keys.KEY_UP], args.free_steps)          # free slide
            dumps.append(dump_ram(emu)); pinned.append(0)
            brights.append(brightness())
            hold(emu, [Keys.KEY_UP, pin_key], args.pin_steps)   # pinned at wall
            dumps.append(dump_ram(emu)); pinned.append(1)
            brights.append(brightness())
            hold(emu, [Keys.KEY_UP, pin_key], args.pin_steps)   # still pinned
            dumps.append(dump_ram(emu)); pinned.append(1)
            brights.append(brightness())
    emu.destroy()
    print(f"{len(dumps)} dumps ({time.time()-t0:.0f}s)", flush=True)
    print(f"brightness per dump: {[round(b,1) for b in brights]}", flush=True)

    D = np.stack(dumps)
    U = np.stack([np.frombuffer(d.tobytes(), dtype="<u4") for d in D])
    S = U.view(np.int32)
    F = np.stack([np.frombuffer(d.tobytes(), dtype="<f4") for d in D])
    m = np.array(pinned)

    dS = S[1:].astype(np.int64) - S[:-1].astype(np.int64)
    free = m[1:] == 0
    pin = m[1:] == 1
    mag_free = np.abs(dS[free]).mean(0)
    mag_pin = np.abs(dS[pin]).mean(0)
    # gated: moves while free, (nearly) frozen while pinned
    gated = (mag_free > 8) & (mag_pin < mag_free / 8.0) & (dS[free] != 0).all(0)
    print(f"wall-pinned-gated candidates: {int(gated.sum())}", flush=True)

    idx = np.nonzero(gated)[0]
    for i in idx[:80]:
        print(f"  0x{RAM_BASE + int(i)*4:08X}", flush=True)
        print(f"      s32={S[:, i].tolist()}", flush=True)
        print(f"      f32={[round(float(x), 2) for x in F[:, i]]}", flush=True)

    np.savez_compressed(os.path.join(ROOT, "data/ram_search4.npz"),
                        dumps=D, pinned=m, gate_idx=idx)
    print(f"saved -> data/ram_search4.npz ({len(idx)} candidates)", flush=True)


if __name__ == "__main__":
    main()
