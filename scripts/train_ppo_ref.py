"""PPO on the same task as scripts/train_btr.py, for the algorithm ablation.

Same environments, episode rules and reward as the BTR run (scripts/train_btr.ref_env_kwargs: +1 per coin,
--goal-reward at the finish, nothing else), same 15 environments (5 per slide), same 4-frame stack of 84x84
grayscale frames and the same action set; only the learner differs. Standard Atari-style PPO settings
(not tuned): 128 steps per environment per rollout, batch 256, 4 epochs, lr 2.5e-4, gamma 0.99, GAE 0.95,
clip 0.1, entropy 0.01, SB3's default CNN. Stops after --max-minutes and saves final.zip.

Usage:
    python scripts/train_ppo_ref.py --run abl_ppo --goal-reward 20 --max-minutes 60
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="abl_ppo")
    ap.add_argument("--tracks", default="ds1,ds2,ds3")
    ap.add_argument("--envs-per-track", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--goal-reward", type=float, default=20.0)
    ap.add_argument("--action-set", default="ref10", choices=["ref10", "legacy6", "compound12"])
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--log-every", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

    from scripts.train_btr import make_ref_env

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    names = [t for t in tracks for _ in range(args.envs_per_track)]
    E = len(names)
    out = os.path.join(ROOT_DIR, "models", args.run)
    os.makedirs(out, exist_ok=True)
    vec = SubprocVecEnv([make_ref_env(t, rank=i, seed=args.seed, max_steps=args.max_steps,
                                      goal_reward=args.goal_reward, action_set=args.action_set)
                         for i, t in enumerate(names)])
    vec = VecTransposeImage(VecFrameStack(vec, 4))

    log_path = os.path.join(ROOT_DIR, "results", f"{args.run}_train.csv")
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if os.path.getsize(log_path) == 0:
        writer.writerow(["steps", "minutes"] + [f"{t}_return" for t in tracks] + [f"{t}_len" for t in tracks])

    class Stop(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0, self.next_log = time.time(), args.log_every
            self.ret, self.len = np.zeros(E), np.zeros(E, int)
            self.recent = {t: [] for t in tracks}

        def _on_step(self):
            self.ret += self.locals["rewards"]
            self.len += 1
            for i in np.flatnonzero(self.locals["dones"]):
                r = self.recent[names[i]]
                r.append((self.ret[i], self.len[i]))
                del r[:-30]
                self.ret[i], self.len[i] = 0.0, 0
            if self.num_timesteps >= self.next_log:
                self.next_log += args.log_every
                row = [self.num_timesteps, round((time.time() - self.t0) / 60, 1)]
                row += [round(float(np.mean([a for a, _ in self.recent[t]])), 2) if self.recent[t] else "" for t in tracks]
                row += [round(float(np.mean([b for _, b in self.recent[t]]))) if self.recent[t] else "" for t in tracks]
                writer.writerow(row)
                log.flush()
                print("steps", row[0], "|", row[1], "min |", row[2:], flush=True)
            return (time.time() - self.t0) / 60 < args.max_minutes

    model = PPO("CnnPolicy", vec, n_steps=128, batch_size=256, n_epochs=4, learning_rate=2.5e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, device=args.device, seed=args.seed, verbose=0)
    model.learn(total_timesteps=10 ** 9, callback=Stop())
    model.save(os.path.join(out, "final.zip"))
    vec.close()
    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
