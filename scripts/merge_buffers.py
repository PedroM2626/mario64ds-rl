"""Merge several world-model replay buffers into one (e.g. base + checkpoint-
seeded exploration + on-policy MPC failure episodes).

    python scripts/merge_buffers.py --out data/wm_buffer_all.pkl \
        data/wm_buffer3.pkl data/wm_buffer4.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.wm_replay import EpisodeBuffer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Merge replay buffers")
    p.add_argument("inputs", nargs="+", help="buffer .pkl files to merge, in order")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = EpisodeBuffer()
    for path in args.inputs:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        b = EpisodeBuffer()
        b.load(full)
        out.episodes.extend(b.episodes)
        out.total_steps += b.total_steps
        deaths = sum(1 for e in b.episodes
                     if len(e["continues"]) and e["continues"][-1] < 0.5)
        print(f"+ {path}: {b.total_steps} steps / {len(b)} episodes ({deaths} deaths)")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    out.save(out_path)
    deaths = sum(1 for e in out.episodes if len(e["continues"]) and e["continues"][-1] < 0.5)
    print(f"= merged: {out.total_steps} steps / {len(out)} episodes ({deaths} deaths) -> {out_path}")


if __name__ == "__main__":
    main()
