"""Merge state-transition pickles (collect_state / state_mpc --collect-to).

    python scripts/merge_state.py --out data/state_all.pkl data/state_transitions.pkl ...
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    merged = {}
    total_steps = 0
    for path in args.inputs:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        with open(full, "rb") as f:
            d = pickle.load(f)
        for tr, eps in d.items():
            merged.setdefault(tr, []).extend(eps)
            n = sum(len(e["actions"]) for e in eps)
            total_steps += n
            print(f"+ {path} [{tr}]: {len(eps)} episodes / {n} transitions")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    falls = sum(int(e["falls"].sum()) for eps in merged.values() for e in eps)
    terms = sum(int(e["dones"].any()) for eps in merged.values() for e in eps)
    print(f"= merged: {total_steps} transitions ({falls} falls, {terms} "
          f"terminal deaths) -> {out_path}")


if __name__ == "__main__":
    main()
