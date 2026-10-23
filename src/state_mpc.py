"""Expert-free CEM-MPC entirely in RAM-state space (the smw-pinn paradigm).

The pixel-only pipeline could not achieve reliable control because every head
was action-blind on imagined latents (docs/WORLD_MODEL.md section 9). With
Mario's physics struct read directly from RAM (src/ram_state.py), actions move
the STATE ITSELF: the learned dynamics f(s, a) -> s' give the planner precise
fall foresight (fall/terminal heads at recall ~1.0), and no imagination drift
exists because replanning starts from the exact state every step.

No policy network, no action labels, no pixels in the control loop.

    python -m src.state_mpc --tracks ds1,ds2,ds3 --episodes 3
    python -m src.state_mpc --tracks ds1 --record          # video
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
from torch.distributions import Categorical

from src.device import resolve_device
from src.env import Mario64DSEnv
from src.ram_state import Sm64DsState
from src.state_model import (StateDynamicsModel, STATE_COLS, GAMMA, FALL_PENALTY)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StateMPC:
    def __init__(self, model: StateDynamicsModel, norm, device,
                 horizon=6, candidates=256, elites=32, iters=4,
                 gamma=0.97, fall_penalty=50.0, persistence=0.6,
                 speed_weight=1.0, seed=0):
        self.m = model.to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.mu = np.asarray(norm["mu"], dtype=np.float64)
        self.sd = np.asarray(norm["sd"], dtype=np.float64)
        # kill plane in normalized state units (height is STATE_COLS index 2)
        height_raw = -307200.0
        self.kill_norm = torch.tensor((height_raw - self.mu[2]) / self.sd[2],
                                      dtype=torch.float32, device=device)
        self.H = horizon
        self.N = candidates
        self.E = elites
        self.iters = iters
        self.gamma = gamma
        self.fall_penalty = fall_penalty
        self.speed_weight = speed_weight
        self.persistence = persistence
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def _score(self, s0, seq):
        """Physics-grounded objective: sum of predicted slide speed, with a
        large penalty the first time the predicted height clamps at the kill
        plane (a deterministic physical event in the learned dynamics) plus the
        learned terminal-death head as a safety net.

        With exact state there is no need for a learned value bootstrap: falls
        develop within a few steps and are predicted directly by the dynamics
        (the measured learned heads read fall_p=0.065 AT the kill plane and
        V(respawn) > V(on-track) -- semantics learned from transitions, not
        states; the physics itself is the reliable signal)."""
        B = seq.shape[1]
        s = s0.expand(B, *s0.shape[1:]).contiguous()
        disc = torch.ones(B, device=self.device)
        alive = torch.ones(B, device=self.device)
        score = torch.zeros(B, device=self.device)
        SPEED_CAP = 409600.0
        for t in range(seq.shape[0]):
            ds, _, tmp, _, _ = self.m(s, seq[t])
            s = s + ds
            clamp_hit = (s[:, 2] <= self.kill_norm).float()
            # speed is state column 0; normalize to [0, ~1] per step
            speed_raw = s[:, 0] * float(self.sd[0]) + float(self.mu[0])
            r = self.speed_weight * (speed_raw / SPEED_CAP)
            score = score + alive * disc * (r - self.fall_penalty * clamp_hit)
            alive = alive * (1.0 - torch.sigmoid(tmp).clamp(max=0.9))
            disc = disc * self.gamma
        return score

    def plan(self, s_raw):
        """CEM over H-step action sequences; returns the best first action."""
        s_np = np.asarray(s_raw, dtype=np.float64)[list(STATE_COLS)]
        s0 = torch.tensor((s_np - self.mu) / self.sd, dtype=torch.float32,
                          device=self.device)[None]
        logits = torch.zeros(self.H, 6, device=self.device)
        for _ in range(self.iters):
            probs = torch.softmax(logits, dim=-1)
            seq = Categorical(probs=probs).sample((self.N,)).T.contiguous()
            if self.persistence > 0:
                keep = torch.rand(self.H, self.N, device=self.device,
                                  generator=self.gen) < self.persistence
                keep[0] = False
                for t in range(1, self.H):
                    seq[t] = torch.where(keep[t], seq[t - 1], seq[t])
            values = self._score(s0, seq)
            top = values.topk(min(self.E, self.N)).indices
            elite = seq[:, top]
            counts = torch.nn.functional.one_hot(elite, 6).float().sum(1)
            logits = torch.log(counts + 0.5)
        mode = torch.softmax(logits, dim=-1).argmax(-1)
        return int(mode[0].item())


def load_state_bundle(path, device):
    ck = torch.load(path, map_location=device)
    out = {}
    for tr, d in ck.items():
        m = StateDynamicsModel()
        m.load_state_dict(d["model"])
        out[tr] = StateMPC(m, d["norm"], device)
    return out


def _probe_here(env, idx, out_dir, existing, track, branch_len=10):
    """Savestate-branch: apply each of the 6 probe actions once at the current
    real state, then coast (noop) for branch_len steps. Records the true
    fall/terminal outcome per action -- which arms the planner's emergency
    fall-prediction where the model alone misfires.
    All offline: 6 RAM polls per step (no image processing), so cheap."""
    import uuid
    path = os.path.join(out_dir, f"smpc_{track}_{idx}_{uuid.uuid4().hex[:6]}.dsx")
    env.emu.savestate.save_file(path)
    orig_state_path = env.state_path
    prev_gray = env.prev_gray
    S0 = np.asarray(env.ram.poll(), dtype=np.float64)
    n_add = 0
    try:
        for a_probe in range(6):
            env.state_path = path
            try:
                env.reset()
            except Exception:
                break
            env.prev_gray = prev_gray
            states = [S0]
            actions, rewards, falls, dones = [], [], [], []
            dead = False
            a = a_probe
            for k in range(branch_len):
                obs, r, term, trunc, info = env.step(int(a))
                s1 = info.get("ram")
                s1 = np.asarray(s1, dtype=np.float64) if s1 is not None else states[-1]
                states.append(s1)
                actions.append(int(a))
                rewards.append(float(r))
                from src.ram_state import Sm64DsState
                falls.append(bool(Sm64DsState.is_dead(s1)))
                dones.append(bool(term))
                dead = bool(term or trunc)
                if dead:
                    break
                a = 0  # coast afterwards
            ep = dict(behavior="state_mpc_probe",
                      states=np.array(states),
                      actions=np.array(actions, dtype=np.int64),
                      rewards=np.array(rewards, dtype=np.float32),
                      falls=np.array(falls), dones=np.array(dones))
            existing.setdefault(track, []).append(ep)
            n_add += 1
    finally:
        env.state_path = path
        try:
            env.reset()
            env.prev_gray = prev_gray
        except Exception:
            pass
        env.state_path = orig_state_path
        try:
            os.remove(path)
        except OSError:
            pass
    return n_add


def run_episode(env, ctrl: StateMPC, max_steps, record=False,
                probe_every=0, probe_dir=None, existing=None, track=None,
                probe_branch_len=10):
    obs, info = env.reset()
    frames = [env.get_screen_rgb()] if record else None
    s = info["ram"]
    action = ctrl.plan(s)
    steps, R, done = 0, 0.0, False
    falls = 0
    states = [np.asarray(s, dtype=np.float64)]
    actions, rewards, falls_l, dones = [], [], [], []
    while not done and steps < max_steps:
        obs, r, term, trunc, info = env.step(action)
        R += float(r)
        steps += 1
        s1 = info.get("ram")
        s1 = np.asarray(s1, dtype=np.float64) if s1 is not None else states[-1]
        states.append(s1)
        actions.append(int(action))
        rewards.append(float(r))
        is_fall = bool(Sm64DsState.is_dead(s1))
        falls_l.append(is_fall)
        dones.append(bool(term))
        falls += int(is_fall)
        done = bool(term or trunc)
        if (probe_dir is not None and existing is not None and not done
                and probe_every > 0 and steps % probe_every == 0):
            _probe_here(env, steps, probe_dir, existing, track,
                        branch_len=probe_branch_len)
        if record:
            frames.append(env.get_screen_rgb())
        if not done:
            action = ctrl.plan(info["ram"])
    ep = dict(behavior="state_mpc",
              states=np.array(states), actions=np.array(actions, dtype=np.int64),
              rewards=np.array(rewards, dtype=np.float32),
              falls=np.array(falls_l), dones=np.array(dones))
    return steps, R, falls, frames, ep


def main():
    p = argparse.ArgumentParser(description="RAM-state CEM-MPC (expert-free)")
    p.add_argument("--bundle", default="models/state_wm.pt")
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--tracks", default="ds1,ds2,ds3")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    p.add_argument("--device", default="auto",
                   help="torch device: auto (cuda if available), cpu or cuda")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--elites", type=int, default=32)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--fall-penalty", type=float, default=50.0)
    p.add_argument("--speed-weight", type=float, default=1.0)
    p.add_argument("--persistence", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--record-single", default=None, help=argparse.SUPPRESS)
    p.add_argument("--collect-to", default=None,
                   help="append episodes (state transitions) to a pickle in the "
                        "collect_state format -- the DAgger loop for the state "
                        "model: run, collect the planner's own visited states, "
                        "retrain (seconds), repeat")
    p.add_argument("--probe-every", type=int, default=0,
                   help="probe all 6 actions from the CURRENT real state every N "
                        "steps (savestate branching; counterfactual fall labels "
                        "for the planner's visited states; needs --collect-to)")
    p.add_argument("--probe-branch-len", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    device = resolve_device(args.device)
    ctrls = load_state_bundle(os.path.join(ROOT, args.bundle), device)
    rom = os.path.join(ROOT, args.rom)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    if args.record:
        import subprocess
        import sys
        os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
        for tr in tracks:
            cmd = [sys.executable, "-m", "src.state_mpc",
                   "--bundle", args.bundle, "--tracks", tr, "--episodes", "1",
                   "--seed", str(args.seed), "--record-single", tr,
                   "--fps", str(args.fps),
                   "--horizon", str(args.horizon), "--candidates", str(args.candidates),
                   "--elites", str(args.elites), "--iters", str(args.iters),
                   "--fall-penalty", str(args.fall_penalty),
                   "--speed-weight", str(args.speed_weight),
                   "--persistence", str(args.persistence)]
            subprocess.run(cmd, cwd=ROOT)
        return

    env = None
    summary = {}
    single_record = args.record_single
    if single_record:
        tracks = [single_record]

    def load_collect(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    collector = None
    if args.collect_to:
        cpath = args.collect_to if os.path.isabs(args.collect_to) \
            else os.path.join(ROOT, args.collect_to)
        collector = cpath
        if os.path.exists(cpath):
            try:
                existing = load_collect(cpath)
            except Exception as e:
                print(f"[state-mpc] WARNING: unreadable/corrupt {cpath} ({e}); "
                      f"starting a fresh collection", flush=True)
                existing = {}
        else:
            existing = {}

    probe_dir = None
    if args.probe_every > 0 and collector is not None:
        probe_dir = os.path.join(ROOT, "data", "_state_probe")
        os.makedirs(probe_dir, exist_ok=True)

    try:
        for tr in tracks:
            if tr not in ctrls:
                print(f"[state-mpc] no model for {tr}; skipping", flush=True)
                continue
            ctrl = ctrls[tr]
            ctrl.H, ctrl.N, ctrl.E, ctrl.iters = (args.horizon, args.candidates,
                                                 args.elites, args.iters)
            ctrl.fall_penalty = args.fall_penalty
            ctrl.speed_weight = args.speed_weight
            ctrl.persistence = args.persistence
            ss = os.path.join(ROOT, f"data/Super Mario 64 DS (USA) (Rev 1).{tr}")
            if env is None:
                env = Mario64DSEnv(rom, ss, max_steps=args.max_steps,
                                   flow_weight=args.flow_weight)
            env.state_path = ss
            surv, dists, fall_counts = 0, [], []
            for ep in range(args.episodes):
                ctrl.gen.manual_seed(args.seed + ep)
                steps, R, falls, frames, ep_data = run_episode(
                    env, ctrl, args.max_steps, record=bool(single_record),
                    probe_every=args.probe_every,
                    probe_dir=probe_dir if single_record is None else None,
                    existing=existing if collector is not None else None,
                    track=tr, probe_branch_len=args.probe_branch_len)
                dists.append(steps)
                surv += int(steps >= args.max_steps)
                fall_counts.append(falls)
                print(f"  [{tr} ep{ep}] steps={steps}/{args.max_steps} R={R:.1f} "
                      f"falls={falls} "
                      f"{'SURVIVED' if steps >= args.max_steps else 'died'}",
                      flush=True)
                if collector is not None:
                    existing.setdefault(tr, []).append(ep_data)
                    with open(collector, "wb") as f:
                        pickle.dump(existing, f)
                    print(f"    collected -> {collector}", flush=True)
                if single_record and frames:
                    import imageio
                    out = os.path.join(ROOT, "videos",
                                       f"state_mpc_{single_record}.mp4")
                    with imageio.get_writer(out, fps=args.fps, codec="libx264",
                                            quality=8, macro_block_size=1) as w:
                        for fr in frames:
                            w.append_data(fr)
                    print(f"  saved video -> {out}", flush=True)
            summary[tr] = (surv, float(np.mean(dists)), float(np.mean(fall_counts)))
            print(f"== {tr}: survival {surv}/{args.episodes} "
                  f"mean_steps {np.mean(dists):.0f} "
                  f"mean_falls {np.mean(fall_counts):.1f} ==", flush=True)
    finally:
        if env:
            env.close()
    print("\n=== RAM-STATE CEM-MPC (EXPERT-FREE) — REAL GAME ===")
    for tr, (sv, md, fl) in summary.items():
        print(f"  {tr}: {sv}/{args.episodes} completed (mean {md:.0f}/1350, "
              f"falls/ep {fl:.1f})")


if __name__ == "__main__":
    main()
