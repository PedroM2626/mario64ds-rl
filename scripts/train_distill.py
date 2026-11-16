"""Distil the two-network TTC agent into ONE 13-action network (policy/quantile distillation, supervised).

Data: scripts/collect_distill_data.py (newest frames, teacher id, the teacher's quantile values at 16 fixed levels for
13 actions). The student (IQNNet, 13 actions) starts as the main policy extended to 13 actions and is trained, on the
frame stacks rebuilt from the recorded frames, to reproduce the quantile values of whichever teacher acted at that
decision: loss = MSE(student quantiles, teacher quantiles) + w * cross-entropy of softmax(mean quantile / T) against
the teacher's greedy action. Runs on the CPU (the GPU is used by another job); scripts/lowprio.py lowers its priority.

Usage:
    python scripts/lowprio.py scripts/train_distill.py --data data/distill --out models/btr_ttc_merged --epochs 12
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

SOURCES = [1, 8, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default="models/btr_ttc_s9/step_750000.pt", help="10-action main policy, extended to 13 actions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--filter-parts", default="",
                    help="comma-separated numbers of DAgger part files (part_NN.npz): of these, only the episodes in which the specialist "
                         "acted at some point are kept (the main policy is not a reliable teacher in the states where the student got stuck before the hand)")
    ap.add_argument("--val-every", type=int, default=6, help="every n-th collected episode is held out for validation")
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from src.btr import BTRAgent

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    parts = []
    drop = {int(x) for x in args.filter_parts.split(",") if x}
    for f in sorted(glob.glob(os.path.join(ROOT_DIR, args.data, "part_*.npz"))):
        d = {k: v for k, v in np.load(f).items()}
        if int(os.path.basename(f)[5:-4]) in drop:
            good = np.unique(d["ep"][d["teacher"] == 1])
            keep = np.isin(d["ep"], good)
            print(f"{os.path.basename(f)}: kept {int(keep.sum())} of {len(keep)} decisions ({len(good)} of {len(np.unique(d['ep']))} episodes reached the specialist)", flush=True)
            d = {k: v[keep] for k, v in d.items()}
        parts.append(d)
    frames = np.concatenate([p["frames"] for p in parts])
    q = np.concatenate([p["q"] for p in parts]).astype(np.float32)
    teacher = np.concatenate([p["teacher"] for p in parts]).astype(np.int64)
    ep = np.concatenate([p["ep"] for p in parts])
    t = np.concatenate([p["t"] for p in parts])
    N = len(frames)
    print(f"{N} decisions, {len(np.unique(ep))} episodes, teacher 0: {(teacher == 0).sum()}, teacher 1: {(teacher == 1).sum()}", flush=True)

    mean_q = q.mean(1)                                                       # (N, 13)
    top2 = np.sort(mean_q, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    temp = float(np.median(gap)) / 2.0
    print(f"greedy gap median {np.median(gap):.4f}, mean {gap.mean():.4f}; softmax temperature {temp:.4f}", flush=True)
    target_a = mean_q.argmax(1)

    episodes = np.unique(ep)
    val_eps = set(episodes[:: args.val_every].tolist())
    is_val = np.array([e in val_eps for e in ep])
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"train {len(train_idx)}, validation {len(val_idx)} ({len(val_eps)} held-out episodes)", flush=True)

    def stacks(idx):
        """(B, 4, 84, 84) uint8 from the recorded frames; frames before the start of the episode are zeros."""
        offs = np.arange(-3, 1)
        j = idx[:, None] + offs[None, :]
        valid = (t[idx][:, None] + offs[None, :]) >= 0
        x = frames[np.clip(j, 0, N - 1)] * valid[:, :, None, None]
        return x

    agent = BTRAgent(13, device="cpu")
    agent.load(os.path.join(ROOT_DIR, args.init), sources=SOURCES)
    net = agent.net
    net.train(False)                    # the layers use their means only, as in act(noisy=False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0, eps=1.5e-4)
    mid = ((torch.arange(16, dtype=torch.float32) + 0.5) / 16).view(1, 16)
    os.makedirs(os.path.join(ROOT_DIR, args.out), exist_ok=True)

    def evaluate(idx):
        net.train(False)
        agree = {0: [0, 0], 1: [0, 0]}
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                b = idx[s:s + 256]
                x = torch.as_tensor(stacks(b))
                pred = net(x, mid.expand(len(b), 16)).mean(1).argmax(-1).numpy()
                for tid in (0, 1):
                    m = teacher[b] == tid
                    agree[tid][0] += int((pred[m] == target_a[b][m]).sum())
                    agree[tid][1] += int(m.sum())
        return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in agree.items()}

    t0, best = time.time(), -1.0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_idx)
        tot_mse = tot_ce = 0.0
        nb = 0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            x = torch.as_tensor(stacks(b))
            pred = net(x, mid.expand(len(b), 16))                            # (B, 16, 13)
            tgt = torch.as_tensor(q[b])
            mse = torch.nn.functional.mse_loss(pred, tgt)
            ce = torch.nn.functional.cross_entropy(pred.mean(1) / temp, torch.as_tensor(target_a[b]))
            loss = mse + args.ce_weight * ce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            tot_mse += float(mse)
            tot_ce += float(ce)
            nb += 1
            if args.minutes and time.time() - t0 > args.minutes * 60:
                break
        tr_acc = evaluate(rng.choice(train_idx, min(4000, len(train_idx)), replace=False))
        va_acc = evaluate(val_idx) if len(val_idx) else {0: float("nan"), 1: float("nan")}
        score = min(va_acc[0], va_acc[1]) if len(val_idx) else min(tr_acc[0], tr_acc[1])
        print(f"epoch {epoch}: mse {tot_mse / nb:.5f} ce {tot_ce / nb:.4f} | train agree main {tr_acc[0]:.4f} spec {tr_acc[1]:.4f} | "
              f"val agree main {va_acc[0]:.4f} spec {va_acc[1]:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save({"net": net.state_dict(), "n_actions": 13}, os.path.join(ROOT_DIR, args.out, f"epoch_{epoch:02d}.pt"))
        if score > best:
            best = score
            torch.save({"net": net.state_dict(), "n_actions": 13}, os.path.join(ROOT_DIR, args.out, "best.pt"))
        if args.minutes and time.time() - t0 > args.minutes * 60:
            break
    torch.save({"net": net.state_dict(), "n_actions": 13}, os.path.join(ROOT_DIR, args.out, "final.pt"))


if __name__ == "__main__":
    main()
