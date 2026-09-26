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
from src.wm_replay import EpisodeBuffer
from src.world_model import Latent
from src.world_model_controller import load_world_model_actor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = 4


class MPCController:
    def __init__(self, model, device="cuda", horizon=12, candidates=240, elites=24,
                 iters=4, gamma=0.995, pessimism=0.0, terminal_cost=100.0,
                 death_thresh=0.5, alpha=0.5, seed=0, warmstart=True,
                 value_boot=False, q_weight=0.0, reward_cap=0.0, replan_every=1,
                 q_cont=1, cem_persistence=0.0):
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
        # Terminal bootstrap from the value head (fit on real returns): the
        # effective horizon becomes unbounded without lengthening the prior
        # rollout, which is what previously forced the horizon-vs-drift dilemma.
        self.value_boot = bool(value_boot and self.m.value_net is not None)
        # Q grounding: the first action is scored on the TRUE posterior state
        # with the Q head (real-return targets) -- the only evaluation in the
        # whole planner that never touches imagined latents.
        self.q_weight = float(q_weight) if self.m.q_net is not None else 0.0
        # Continuation semantics for the Q read (0 = noop-coast, 1 = hold).
        self.q_cont = int(q_cont)
        # Clamp predicted per-step reward (anti reward-hack for flow spikes).
        self.reward_cap = float(reward_cap)
        # Candidate persistence: with prob p each sampled step repeats the
        # previous action. The behaviors that survive these slides are momentum
        # PATTERNS (jump-spam holds the racing line, a held steer slides off the
        # edge); iid per-step sampling almost never proposes a coherent 12-step
        # hold, so CEM could not even evaluate the patterns the data is about.
        self.cem_persistence = float(cem_persistence)
        # Commit to the plan's prefix for K steps (receding horizon with action
        # persistence): re-planning every single step chases planner noise and
        # produced high-frequency weaving near hazards.
        self.replan_every = max(1, int(replan_every))
        self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)
        self.A = self.m.num_actions
        self.latent = None
        self.prev_action = None
        self.dq = None
        self._last_best = None
        self._queue = []

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
        self._queue = []
        self._posterior_update(first_frame, first=True)
        seq = self._plan()
        self._queue = [int(a) for a in seq[:self.replan_every]]
        self.prev_action = torch.tensor([self._queue.pop(0)], dtype=torch.long,
                                         device=self.device)
        return int(self.prev_action.item())

    @torch.no_grad()
    def _plan(self):
        """CEM search; returns the *modal* plan of the final elite distribution.

        Acting on the argmax-sampled candidate (as before) chases evaluation
        noise: the max over ~1000 noisy value estimates is biased upward and
        changes erratically step-to-step (measured as weaving near cliffs). The
        per-step mode of the converged elite distribution is the standard, far
        less noise-sensitive CEM control law.
        """
        start = Latent(self.latent.deter, self.latent.mean, self.latent.std,
                       self.latent.mean)  # deterministic belief (z = prior/post mean)
        logits = torch.zeros(self.H, self.A, device=self.device)
        if self.warmstart and self._last_best is not None:
            shifted = torch.cat([self._last_best[1:],
                                 torch.zeros(1, dtype=torch.long, device=self.device)])
            logits.scatter_(1, shifted.unsqueeze(1), 0.5)
        for _ in range(self.iters):
            probs = torch.softmax(logits, dim=-1)
            seq = Categorical(probs=probs).sample((self.N,)).T.contiguous()  # (H, N)
            if self.cem_persistence > 0.0:
                # momentum-biased proposals: repeat the previous action with prob p
                keep = torch.rand(self.H, self.N, device=self.device,
                                  generator=self.gen) < self.cem_persistence
                keep[0] = False
                # vectorized repeat: seq[t] = seq[t-1] where keep
                for t in range(1, self.H):
                    seq[t] = torch.where(keep[t], seq[t - 1], seq[t])
            values = self.m.rollout_value(start, seq, self.gamma, self.pessimism,
                                          self.terminal_cost, self.death_thresh,
                                          value_boot=self.value_boot,
                                          q_weight=self.q_weight,
                                          reward_cap=self.reward_cap,
                                          q_cont=self.q_cont)  # (N,)
            top = values.topk(min(self.E, self.N)).indices
            elite = seq[:, top]                                            # (H, E)
            counts = torch.nn.functional.one_hot(elite, self.A).float().sum(1)  # (H, A)
            logits = torch.log(counts + self.alpha)
        seq_mode = torch.softmax(logits, dim=-1).argmax(-1)                 # (H,)
        self._last_best = seq_mode
        return seq_mode

    @torch.no_grad()
    def act(self, frame):
        self._posterior_update(frame)
        if not self._queue:
            seq = self._plan()
            self._queue = [int(a) for a in seq[:self.replan_every]]
        action = self._queue.pop(0)
        self.prev_action = torch.tensor([action], dtype=torch.long, device=self.device)
        return action


class _RunCollector:
    """In-memory episode collector with atomic, batched flushes.

    Reloading + rewriting the whole pickle per probe branch made collection
    O(n^2) in I/O (hours) and a mid-write kill corrupted the buffer. The
    collector owns one EpisodeBuffer for the whole process, adds in memory,
    and flushes atomically (tmp + replace) once per episode.
    """

    def __init__(self, path):
        self.path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        self.buf = EpisodeBuffer()
        if os.path.exists(self.path):
            self.buf.load(self.path)

    def add(self, *args, **kwargs):
        self.buf.add(*args, **kwargs)

    def flush(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        self.buf.save(tmp)
        os.replace(tmp, self.path)
        print(f"[mpc] collected -> {self.path} "
              f"({self.buf.total_steps} steps / {len(self.buf)} episodes)", flush=True)


def _probe_here(env, traj, ppo, steps_done, out_dir, collector, branch_len,
                prefix_len, branch_policy, tag):
    """Counterfactual probe from the CURRENT emulator state during an MPC episode.

    Saves a savestate, tries every action once (continuation: noop/hold/ppo),
    stores each branch into the collector with probe metadata, then reloads the
    savestate so the episode continues from the exact same state. The MPC
    controller's own belief/frame stack is untouched -- only the emulator is
    rewound between branches.
    """
    import uuid
    path = os.path.join(out_dir, f"mpcprobe_{tag}_{uuid.uuid4().hex[:6]}.dsx")
    env.emu.savestate.save_file(path)
    prev_gray = env.prev_gray
    orig_state_path = env.state_path
    frames_all = [np.asarray(f, np.uint8) for f in traj["frames"]]
    actions_all = [int(a) for a in traj["actions"]]
    rewards_all = [float(r) for r in traj["rewards"]]
    continues_all = [float(c) for c in traj["continues"]]
    n_probe = 0
    try:
        for a_probe in range(6):
            env.state_path = path
            try:
                env.reset()
            except Exception:
                break
            env.prev_gray = prev_gray
            frames = frames_all[-prefix_len:]
            actions = actions_all[-prefix_len:]
            rewards = rewards_all[-prefix_len:]
            continues = continues_all[-prefix_len:]
            a = a_probe
            dead = False
            for k in range(branch_len):
                obs_b, r_b, term_b, trunc_b, _ = env.step(int(a))
                frames.append(obs_b[:, :, 0])
                actions.append(int(a))
                rewards.append(float(r_b))
                continues.append(0.0 if term_b else 1.0)
                dead = bool(term_b or trunc_b)
                if dead:
                    break
                if branch_policy == "noop":
                    a = 0
                elif branch_policy == "hold":
                    a = a_probe
                else:
                    recent = frames[-4:]
                    if len(recent) < 4:  # early-episode probes: zero-pad the stack
                        recent = ([np.zeros_like(frames[-1])] * (4 - len(recent))
                                  + recent)
                    stack = np.stack(recent, axis=0)[np.newaxis, :]
                    pred, _ = ppo.predict(stack, deterministic=True)
                    a = int(pred[0])
            collector.add(np.array(frames), np.array(actions), np.array(rewards),
                           np.array(continues), source="probe_mpc", completed=not dead,
                           probe_at=len(frames_all[-prefix_len:]), probe_id=steps_done,
                           branch_cont={"noop": 0, "hold": 1, "ppo": 2}[branch_policy])
            n_probe += 1
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
    return n_probe


def _run_episode(env, model, args, seed, value_boot=False, q_weight=0.0,
                 reward_cap=0.0, replan_every=1, collector=None, q_cont=1,
                 cem_persistence=0.0):
    ctrl = MPCController(model, device=args.device, horizon=args.horizon,
                         candidates=args.candidates, elites=args.elites,
                         iters=args.iters, gamma=args.gamma, pessimism=args.pessimism,
                         terminal_cost=args.terminal_cost, death_thresh=args.death_thresh,
                         seed=seed, value_boot=value_boot, q_weight=q_weight,
                         reward_cap=reward_cap, replan_every=replan_every, q_cont=q_cont,
                         cem_persistence=cem_persistence)
    obs, _ = env.reset()
    frame = obs[:, :, 0]
    # initial action via reset path (cold belief + plan)
    action = ctrl.reset(frame)
    steps, R, done = 0, 0.0, False
    frames = [env.get_screen_rgb()] if args.record else []
    # raw trajectory for --collect-to (same layout as src/collect_data.py _rollout:
    # frames[t] is the observation AFTER executing actions[t])
    traj = dict(frames=[], actions=[], rewards=[], continues=[])
    probe_dir = None
    probe_ppo = None
    if getattr(args, "probe_every", 0) and collector is not None:
        probe_dir = os.path.join(ROOT, "data", "_mpc_probe")
        os.makedirs(probe_dir, exist_ok=True)
        if getattr(args, "probe_branch_policy", "noop") == "ppo":
            from stable_baselines3 import PPO
            probe_ppo = PPO.load(os.path.join(ROOT, args.probe_ppo))
    try:
        while not done and steps < args.max_steps:
            obs, reward, term, trunc, _ = env.step(action)
            frame = obs[:, :, 0]
            traj["frames"].append(frame)
            traj["actions"].append(int(action))
            traj["rewards"].append(float(reward))
            traj["continues"].append(0.0 if term else 1.0)
            if (probe_dir is not None and steps % args.probe_every == 0
                    and not (term or trunc)):
                _probe_here(env, traj, probe_ppo, steps, probe_dir, collector,
                            args.probe_branch_len, args.probe_prefix_len,
                            args.probe_branch_policy, tag=f"{steps}")
            action = ctrl.act(frame)
            R += float(reward); steps += 1
            done = bool(term or trunc)
            if args.record:
                frames.append(env.get_screen_rgb())
    finally:
        if probe_dir is not None:
            import shutil
            shutil.rmtree(probe_dir, ignore_errors=True)
    traj = {k: np.asarray(v) for k, v in traj.items()}
    if collector is not None:
        completed = steps >= args.max_steps
        collector.add(traj["frames"], traj["actions"], traj["rewards"],
                      traj["continues"], source="mpc", completed=completed)
        collector.flush()
    return steps, R, frames, traj


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
    p.add_argument("--no-value-boot", action="store_true",
                   help="disable the terminal bootstrap from the value head "
                        "(enabled automatically when the bundle trained one)")
    p.add_argument("--q-weight", type=float, default=None,
                   help="weight of the Q-head grounding of the first action "
                        "(real-state, real-return Q). Default: 1.0 when the "
                        "bundle trained a Q head, else 0.")
    p.add_argument("--q-cont", type=int, choices=[0, 1, 2, 3], default=1,
                   help="continuation semantics for the Q read: 1 = hold/commit, "
                        "0 = noop-coast, 2 = expert recovery, 3 = hold + recovery "
                        "(pattern preference on-line, escape gradient off-line)")
    p.add_argument("--cem-persistence", type=float, default=0.75,
                   help="probability that a sampled candidate step repeats the "
                        "previous action (momentum-biased proposals: surviving "
                        "behaviors on these slides are action PATTERNS, and iid "
                        "per-step sampling almost never proposes a coherent hold)")
    p.add_argument("--reward-cap", type=float, default=2.0,
                   help="clamp the predicted per-step reward inside the planner "
                        "so flow spikes (fast falling motion earns up to ~3.8/step, "
                        "triple a legitimate fast slide) cannot outbid safety; "
                        "2.0 ~ the expert's 95th percentile per-step reward. "
                        "0 disables the cap.")
    p.add_argument("--replan-every", type=int, default=3,
                   help="execute the first K actions of each CEM plan before "
                        "re-planning (action persistence; re-planning every step "
                        "chases planner noise and weaves near hazards)")
    p.add_argument("--collect-to", default=None,
                   help="append every evaluated episode to this replay buffer "
                        "(on-policy failure collection: the agent's own deaths are "
                        "exactly the near-cliff data the model is missing)")
    p.add_argument("--probe-every", type=int, default=0,
                   help="probe all 6 actions from the current real state every N "
                        "steps (counterfactual branch data along the deployment "
                        "trajectory; requires --collect-to)")
    p.add_argument("--probe-branch-policy", choices=["noop", "hold", "ppo"],
                   default="noop", help="probe branch continuation policy")
    p.add_argument("--probe-branch-len", type=int, default=14)
    p.add_argument("--probe-prefix-len", type=int, default=24)
    p.add_argument("--probe-ppo", default="models/curriculum_flow25_r3_best.zip",
                   help="PPO model for --probe-branch-policy ppo")
    p.add_argument("--no-warmstart", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0,
                   help="controller seed for --single/--record episodes")
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.warmstart = not args.no_warmstart

    import subprocess, sys
    if args.single is None and args.record:
        os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
        for tr in [t.strip() for t in args.tracks.split(",") if t.strip()]:
            cmd = [sys.executable, "-m", "src.mpc_world_model",
                   "--bundle", args.bundle,
                   "--rom", args.rom, "--max-steps", str(args.max_steps),
                   "--flow-weight", str(args.flow_weight), "--device", args.device,
                   "--horizon", str(args.horizon), "--candidates", str(args.candidates),
                   "--elites", str(args.elites), "--iters", str(args.iters),
                   "--gamma", str(args.gamma), "--pessimism", str(args.pessimism),
                   "--terminal-cost", str(args.terminal_cost),
                   "--death-thresh", str(args.death_thresh),
                   "--reward-cap", str(args.reward_cap),
                   "--replan-every", str(args.replan_every),
                   "--record", "--fps", str(args.fps), "--single", tr]
            if args.no_value_boot:
                cmd.append("--no-value-boot")
            if args.q_weight is not None:
                cmd += ["--q-weight", str(args.q_weight)]
            cmd += ["--q-cont", str(args.q_cont),
                    "--cem-persistence", str(args.cem_persistence),
                    "--seed", str(args.seed)]
            if args.collect_to:
                cmd += ["--collect-to", args.collect_to,
                        "--probe-every", str(args.probe_every),
                        "--probe-branch-policy", args.probe_branch_policy]
            subprocess.run(cmd, cwd=ROOT)
        return

    model, _, meta = load_world_model_actor(args.bundle, device=args.device)
    # The value bootstrap is the core of the short-horizon planner: use it
    # whenever the bundle actually trained a value head (and not --no-value-boot).
    value_boot = (not args.no_value_boot) and bool(meta.get("value_trained", False))
    q_weight = (1.0 if meta.get("q_trained", False) else 0.0) \
        if args.q_weight is None else args.q_weight
    print(f"[mpc] value bootstrap: {'ON' if value_boot else 'OFF'} "
          f"(horizon {args.horizon} + V-terminal) | "
          f"Q grounding: weight {q_weight:.2f}", flush=True)

    collector = _RunCollector(args.collect_to) if args.collect_to else None

    rom = os.path.join(ROOT, args.rom)

    if args.single:
        ss = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{args.single}")
        env = Mario64DSEnv(rom, ss, max_steps=args.max_steps, flow_weight=args.flow_weight)
        steps, R, frames, traj = _run_episode(env, model, args, seed=args.seed,
                                             value_boot=value_boot, q_weight=q_weight,
                                             reward_cap=args.reward_cap,
                                             replan_every=args.replan_every,
                                             collector=collector, q_cont=args.q_cont,
                                             cem_persistence=args.cem_persistence)
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
        else:
            print(f"{args.single}: reward={R:.1f} steps={steps}/{args.max_steps} "
                  f"{'COMPLETED' if steps>=args.max_steps else 'FAILED'}", flush=True)
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
                steps, R, _, traj = _run_episode(env, model, args, seed=ep,
                                                 value_boot=value_boot,
                                                 q_weight=q_weight,
                                                 reward_cap=args.reward_cap,
                                                 replan_every=args.replan_every,
                                                 collector=collector, q_cont=args.q_cont,
                                             cem_persistence=args.cem_persistence)
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
