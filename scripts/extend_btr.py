"""Extend a saved BTR agent to a larger action set (e.g. ref10 -> ref13) without changing what it does.

The new actions start as copies of existing ones with a slightly lower output bias (src/btr.BTRAgent.load), so a
noise-free greedy policy picks the same actions as before; training can then learn to use the new ones.

Usage:
    python scripts/extend_btr.py --model models/btr_ttc_s9/step_750000.pt --action-set ref13 --out models/btr_ttc_s9_ref13.pt
"""
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--action-set", default="ref13")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from src.btr import BTRAgent
    from src.env import ACTION_SETS, ACTION_SET_EXTENDS
    base, sources = ACTION_SET_EXTENDS[args.action_set]
    n = len(ACTION_SETS[args.action_set][0])
    agent = BTRAgent(n, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.model), sources=sources)
    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, args.out)), exist_ok=True)
    agent.save(os.path.join(ROOT_DIR, args.out))
    print(f"{args.model} ({len(ACTION_SETS[base][0])} actions) -> {args.out} ({n} actions, {args.action_set})")


if __name__ == "__main__":
    main()
