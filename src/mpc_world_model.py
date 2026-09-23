"""Closed-loop Model Predictive Control (CEM-MPC) over the learned RSSM world model.

This is the reference's flagship world-model controller (``smw-pinn`` §10.6
"Closed-Loop Model-Based RL via MPC", §10.19/§10.24 autonomous MPC navigation):
instead of learning a reactive policy, at every *real* step we plan an action
sequence by Cross-Entropy Method (CEM) search through the learned latent prior,
score each candidate by the discounted predicted reward (penalised by epistemic
uncertainty and discounted by the learned continue probability), and execute the
first action of the best sequence. Because we re-plan from the true belief state
every step (receding horizon), MPC is immune to the open-loop compounding error
that limits the distilled policy.

Only the fitted world model is needed -- no additional policy training and no
gradient steps -- so the agent is usable immediately after ``train_world_model``
fits the model from a small real dataset.

    python -m src.mpc_world_model --bundle models/wm_mario64ds_wm.pt \
        --tracks ds1,ds2,ds3 --episodes 3            # evaluate on the real game
    python -m src.mpc_world_model --bundle ... --record --fps 15   # + videos
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np
import torch
from torch.distributions import Categorical

from src.env import Mario64DSEnv
from src.world_model import Latent
from src.world_model_controller import load_world_model_actor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = 4


class MPCController:
    def __init__(self, model, device="cuda", horizon=12, candidates=240, elites=24,
                 iters=4, gamma=0.995, pessimism=0.0, terminal_cost=100.0,
                 death_thresh=0.5, alpha=0.5, seed=0, warmstart=True):
        self.m = model.to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.H = horizon
        self.N = candidates
        self.E = elites
        self.iters = iters
        self.gamma = gamma
        self.pessimism = pessimism
        self.terminal_cost = terminal_cost
        self.death_thresh = death_thresh
        self.alpha = alpha
        self.warmstart = warmstart
        self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)
        self.A = self.m.num_actions
        self.latent = None
        self.prev_action = None
        self.dq = None
        self._last_best = None

    @torch.no_grad()
    def _posterior_update(self, frame=None, first=False):
        if first:
            arr = np.stack(list(self.dq), axis=0).astype(np.float32) / 255.0
        else:
            self.dq.append(frame)
            arr = np.stack(list(self.dq), axis=0).astype(np.float32) / 255.0
        obs = torch.from_numpy(arr)[None].to(self.device)
        enc = self.m.encoder(obs)
        post, _ = self.m.observe_step(self.latent, self.prev_action, enc)
        self.latent = post
        return post

    def reset(self, first_frame):
        self.latent = self.m.initial_state(1, self.device)
        self.prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self.dq = deque([np.zeros_like(first_frame)] * (STACK - 1) + [first_frame],
                        maxlen=STACK)
        self._last_best = None
        self._posterior_update(first_frame, first=True)
        action = self._plan()
        self.prev_action = torch.tensor([action], dtype=torch.long, device=self.device)
        return action

    @torch.no_grad()
    def _plan(self):
        start = Latent(self.latent.deter, self.latent.mean, self.latent.std,
                       self.latent.mean)  # deterministic belief (z = prior/post mean)
        logits = torch.zeros(self.H, self.A, device=self.device)
        if self.warmstart and self._last_best is not None:
            shifted = torch.cat([self._last_best[1:],
                                 torch.zeros(1, dtype=torch.long, device=self.device)])
            logits.scatter_(1, shifted.unsqueeze(1), 0.5)
        best_seq, best_val = None, -1e18
        for _ in range(self.iters):
            probs = torch.softmax(logits, dim=-1)
            seq = Categorical(probs=probs).sample((self.N,)).T.contiguous()  # (H, N)
            values = self.m.rollout_value(start, seq, self.gamma, self.pessimism,
                                          self.terminal_cost, self.death_thresh)  # (N,)
            top = values.topk(min(self.E, self.N)).indices
            elite = seq[:, top]                                            # (H, E)
            counts = torch.nn.functional.one_hot(elite, self.A).float().sum(1)  # (H, A)
            logits = torch.log(counts + self.alpha)
            b = int(values.argmax())
            if float(values[b]) > best_val:
                best_val, best_seq = float(values[b]), seq[:, b].clone()
        self._last_best = best_seq
        return int(best_seq[0].item())

    @torch.no_grad()
    def act(self, frame):
        self._posterior_update(frame)
        action = self._plan()
        self.prev_action = torch.tensor([action], dtype=torch.long, device=self.device)
        return action


def _run_episode(env, model, args, seed):
    ctrl = MPCController(model, device=args.device, horizon=args.horizon,
                         candidates=args.candidates, elites=args.elites,
                         iters=args.iters, gamma=args.gamma, pessimism=args.pessimism,
                         terminal_cost=args.terminal_cost, death_thresh=args.death_thresh,
                         seed=seed)
    obs, _ = env.reset()
    frame = obs[:, :, 0]
    # initial action via reset path (cold belief + plan)
    action = ctrl.reset(frame)
    steps, R, done = 0, 0.0, False
    frames = [env.get_screen_rgb()] if args.record else []
    while not done and steps < args.max_steps:
        obs, reward, term, trunc, _ = env.step(action)
        frame = obs[:, :, 0]
        action = ctrl.act(frame)
        R += float(reward); steps += 1
        done = bool(term or trunc)
        if args.record:
            frames.append(env.get_screen_rgb())
    return steps, R, frames


def main():
    p = argparse.ArgumentParser(description="CEM-MPC control over the RSSM world model")
    p.add_argument("--bundle", default="models/wm_mario64ds_wm.pt")
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--tracks", default="ds1,ds2,ds3")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    p.add_argument("--device", default="cuda")
    # CEM / planning
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--candidates", type=int, default=240)
    p.add_argument("--elites", type=int, default=24)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--pessimism", type=float, default=0.0)
    p.add_argument("--terminal-cost", type=float, default=100.0,
                   help="one-off penalty applied when the model predicts death within "
                        "the horizon; makes MPC avoid cliffs (higher = more cautious)")
    p.add_argument("--death-thresh", type=float, default=0.5,
                   help="continue-probability below which a state is deemed fatal")
    p.add_argument("--no-warmstart", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.warmstart = not args.no_warmstart

    import subprocess, sys
    if args.single is None and args.record:
        os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
        for tr in [t.strip() for t in args.tracks.split(",") if t.strip()]:
            subprocess.run([sys.executable, __file__, "--bundle", args.bundle,
                            "--rom", args.rom, "--max-steps", str(args.max_steps),
                            "--flow-weight", str(args.flow_weight), "--device", args.device,
                            "--horizon", str(args.horizon), "--candidates", str(args.candidates),
                            "--elites", str(args.elites), "--iters", str(args.iters),
                            "--gamma", str(args.gamma), "--pessimism", str(args.pessimism),
                            "--terminal-cost", str(args.terminal_cost),
                            "--death-thresh", str(args.death_thresh),
                            "--record", "--fps", str(args.fps), "--single", tr], cwd=ROOT)
        return

    model, _, _ = load_world_model_actor(args.bundle, device=args.device)
    rom = os.path.join(ROOT, args.rom)

    if args.single:
        ss = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{args.single}")
        env = Mario64DSEnv(rom, ss, max_steps=args.max_steps, flow_weight=args.flow_weight)
        steps, R, frames = _run_episode(env, model, args, seed=0)
        env.close()
        if frames:
            import imageio
            out = os.path.join(ROOT, "videos", f"mpc_{args.single}.mp4")
            with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=1) as w:
                for fr in frames:
                    w.append_data(fr)
            print(f"{args.single}: reward={R:.1f} steps={steps}/{args.max_steps} "
                  f"{'COMPLETED' if steps>=args.max_steps else 'FAILED'} | {out}", flush=True)
        return

    # evaluation mode: N episodes per track on one persistent emulator
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    env = None
    summary = {}
    try:
        for tr in tracks:
            ss = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{tr}")
            if env is None:
                env = Mario64DSEnv(rom, ss, max_steps=args.max_steps,
                                   flow_weight=args.flow_weight)
            env.state_path = ss
            surv, dists = 0, []
            for ep in range(args.episodes):
                steps, R, _ = _run_episode(env, model, args, seed=ep)
                dists.append(steps); surv += int(steps >= args.max_steps)
                print(f"  [{tr} ep{ep}] steps={steps}/{args.max_steps} R={R:.1f} "
                      f"{'SURVIVED' if steps>=args.max_steps else 'died'}", flush=True)
            summary[tr] = (surv, np.mean(dists))
            print(f"== {tr}: survival {surv}/{args.episodes} mean_steps {np.mean(dists):.0f} ==",
                  flush=True)
    finally:
        if env:
            env.close()
    print("\n=== CEM-MPC WORLD-MODEL AGENT — REAL GAME ===")
    allc = all(s == args.episodes for s, _ in summary.values())
    for tr, (sv, md) in summary.items():
        print(f"  {tr}: {sv}/{args.episodes} completed  (mean {md:.0f}/1350)")
    print(f"  ALL THREE TRACKS COMPLETED: {allc}")


if __name__ == "__main__":
    main()
