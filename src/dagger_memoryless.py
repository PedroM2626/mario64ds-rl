"""Memoryless DAgger on the world model's learned perception.

Plain behavioural cloning of the expert plateaus at ~0.99 aggregate action
accuracy but *systematically* fails the narrow monkey slide (ds3): once the
memoryless student takes a single wrong action its future frames diverge from the
expert's trajectory and the error cascades (classic imitation-compounding). This
module runs DAgger -- roll out the *student*, and at every state it actually
visits ask the expert PPO for the correct action -- so the head learns the
corrections for its own failure states, which is exactly what removes the
compounding (reference §10.25). After every round the candidate head is validated
on the live emulator one-track-per-subprocess and the best is kept.

    python -m src.dagger_memoryless --bundle models/wm_mario64ds_wm.pt \
        --buffer data/wm_buffer.pkl --ppo-model models/curriculum_flow25_r3_best.zip \
        --rounds 8 --eps-per-track 3 --finetune-enc
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import deque

import numpy as np
import torch

from src.env import Mario64DSEnv
from src.wm_replay import EpisodeBuffer
from src.distill_memoryless import (ROOT, STACK, MemorylessPolicy, build_dataset,
                                    train_head, _fresh_model)


def _save_head(path, model, head, args, finetune_enc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(head=head.state_dict(),
                    enc=(model.encoder.state_dict() if finetune_enc else None),
                    enc_dim=model.encoder.out_dim, hidden=args.hidden,
                    finetune_enc=finetune_enc), path)


def _collect_round(model, head, env, expert, device, tracks, max_steps, eps_per_track):
    """Roll the *student*; label each visited state with the expert's action.

    The observation stack is built exactly as at deployment (4 frames, oldest->
    newest, zero-padded start) so the expert labels the true decision states.
    """
    Xs, Ys = [], []
    for tr in tracks:
        env.state_path = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{tr}")
        for _ in range(eps_per_track):
            obs, _ = env.reset()
            first = obs[:, :, 0]
            dq = deque([np.zeros_like(first)] * (STACK - 1) + [first], maxlen=STACK)
            steps, done = 0, False
            while not done and steps < max_steps:
                stack = np.stack(list(dq), axis=0)                 # (4,84,84) uint8
                stackf = stack.astype(np.float32) / 255.0
                ea, _ = expert.predict(stack, deterministic=True)  # expert label here
                Xs.append(stackf); Ys.append(int(ea))
                with torch.no_grad():
                    feat = model.encoder(torch.from_numpy(stackf)[None].to(device))
                    a = int(head.act(feat).item())
                obs, _, term, trunc, _ = env.step(a)
                dq.append(obs[:, :, 0])
                done = bool(term or trunc); steps += 1
    return np.stack(Xs).astype(np.float32), np.array(Ys, dtype=np.int64)


def _validate(head_path, tracks, args):
    """One fresh subprocess per track; return (#cleared, {track: steps})."""
    cleared, res = 0, {}
    for tr in tracks:
        child = [sys.executable, "-m", "src.distill_memoryless", "--evalresult",
                 "--single", tr, "--bundle", args.bundle, "--rom", args.rom,
                 "--max-steps", str(args.max_steps), "--flow-weight", str(args.flow_weight),
                 "--device", args.device, "--save", args.save, "--out-head", head_path]
        if args.finetune_enc:
            child.append("--finetune-enc")
        r = subprocess.run(child, cwd=ROOT, capture_output=True, text=True)
        m = re.search(rf"EVALRESULT {tr} (\d+) (-?\d+)", (r.stdout or "") + (r.stderr or ""))
        surv = int(m.group(1)) if m else 0
        res[tr] = (surv, int(m.group(2)) if m else -1)
        cleared += surv
    return cleared, res


def main():
    p = argparse.ArgumentParser(description="Memoryless DAgger on world-model features")
    p.add_argument("--bundle", default="models/wm_mario64ds_wm.pt")
    p.add_argument("--buffer", default="data/wm_buffer.pkl")
    p.add_argument("--ppo-model", default="models/curriculum_flow25_r3_best.zip")
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--tracks", default="ds1,ds2,ds3")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--eps-per-track", type=int, default=3)
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    p.add_argument("--save", default="models/wm_mario64ds_mem.pt")
    p.add_argument("--finetune-enc", action="store_true")
    args = p.parse_args()

    import shutil
    device = args.device if torch.cuda.is_available() else "cpu"
    bundle = args.bundle if os.path.isabs(args.bundle) else os.path.join(ROOT, args.bundle)
    head_path = args.save if os.path.isabs(args.save) else os.path.join(ROOT, args.save)
    best_path = head_path + ".best"
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    from stable_baselines3 import PPO
    expert = PPO.load(args.ppo_model if os.path.isabs(args.ppo_model)
                      else os.path.join(ROOT, args.ppo_model))

    model = _fresh_model(bundle, device)
    head = MemorylessPolicy(model.encoder.out_dim, 6, args.hidden).to(device)

    buffer = EpisodeBuffer()
    buffer.load(args.buffer if os.path.isabs(args.buffer) else os.path.join(ROOT, args.buffer))
    X, Y = build_dataset(buffer, np.random.default_rng(0))
    train_head(model, head, X, Y, device, args.iters, args.lr, finetune_enc=args.finetune_enc)

    rom = os.path.join(ROOT, args.rom)
    ss0 = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{tracks[0]}")
    env = Mario64DSEnv(rom, ss0, max_steps=args.max_steps, flow_weight=args.flow_weight)

    try:
        _save_head(head_path, model, head, args, args.finetune_enc)
        cleared, res = _validate(head_path, tracks, args)
        best = cleared
        print(f"[dag] round 0 (offline BC): cleared {cleared}/{len(tracks)}  "
              + "  ".join(f"{t}={res[t][1]}" for t in tracks), flush=True)
        if best > 0:
            shutil.copyfile(head_path, best_path)

        for r in range(1, args.rounds + 1):
            if cleared == len(tracks):
                break
            Xd, Yd = _collect_round(model, head, env, expert, device, tracks,
                                    args.max_steps, args.eps_per_track)
            X = np.concatenate([X, Xd], 0); Y = np.concatenate([Y, Yd], 0)
            train_head(model, head, X, Y, device, args.iters, args.lr,
                       finetune_enc=args.finetune_enc, seed=r)
            _save_head(head_path, model, head, args, args.finetune_enc)
            cleared, res = _validate(head_path, tracks, args)
            print(f"[dag] round {r}: +{len(Xd)} on-policy states (total {len(X)}) -> "
                  f"cleared {cleared}/{len(tracks)}  "
                  + "  ".join(f"{t}={res[t][1]}" for t in tracks), flush=True)
            if cleared > best:
                best = cleared; shutil.copyfile(head_path, best_path)
            if cleared == len(tracks):
                shutil.copyfile(head_path, best_path)
                best = cleared
    finally:
        env.close()

    if os.path.exists(best_path):
        shutil.copyfile(best_path, head_path)
        os.remove(best_path)
    print(f"\n[dag] DONE best cleared {best}/{len(tracks)} -> {head_path}", flush=True)


if __name__ == "__main__":
    main()
