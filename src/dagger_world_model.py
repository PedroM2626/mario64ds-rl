"""Interactive DAgger refinement of the world-model policy (reference §10.25).

Offline distillation reproduces the expert only on the states the *expert*
visits; once the student drifts, compounding error ends the episode early
(exactly what we observed on the harder tracks). DAgger fixes this by rolling
out the CURRENT student on the real game, asking the expert (the trained PPO
agent) what it would do at each *student-visited* state, and distilling the
student onto those on-policy labels.

The world model itself stays frozen (already fit); only the latent actor is
refined, so every DAgger round is cheap. Loads a bundle produced by
``train_world_model.py`` and overwrites it with the improved actor.

    python -m src.dagger_world_model --bundle models/wm_mario64ds_wm.pt \
        --buffer data/wm_buffer.pkl --rounds 3 --episodes-per-track 3
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np
import torch

from src.env import Mario64DSEnv
from src.wm_replay import EpisodeBuffer
from src.world_model_controller import WorldModelController, load_world_model_actor
from src.world_model_agent import actor_bc_update
from src.train_world_model import make_eval_env, real_eval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dagger_rollout(env, ctrl, expert_ppo):
    """One on-policy rollout: collect student frames/actions + expert labels."""
    obs, _ = env.reset()
    frame = obs[:, :, 0]
    first = frame.copy()
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)

    def stack():
        return np.stack(list(dq), axis=0)[np.newaxis, :]

    preL, preE, frames, continues = [], [], [], []
    action = ctrl.reset(frame)  # student decision at the initial observation
    done, steps = False, 0
    while not done and steps < env.max_steps:
        expert_a = int(expert_ppo.predict(stack())[0])   # expert at same state
        preL.append(action)
        preE.append(expert_a)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = obs[:, :, 0]
        dq.append(frame)
        frames.append(frame)
        continues.append(0.0 if terminated else 1.0)
        done = bool(terminated or truncated)
        action = ctrl.act(frame)   # student's next decision
        steps += 1
    frames = np.array(frames)
    actions = np.array(preL, dtype=np.int64)
    # label aligned to frames[t] (the decision to take AT that observation)
    expert_next = np.array(preE[1:] + [preE[-1]], dtype=np.int64)
    if len(frames) == 0:
        return None
    return dict(frames=frames, actions=actions, labels=expert_next,
                rewards=np.zeros(len(frames), np.float32),
                continues=np.array(continues, np.float32))


def main():
    p = argparse.ArgumentParser(description="DAgger refinement of the world-model actor")
    p.add_argument("--bundle", required=True)
    p.add_argument("--buffer", default="data/wm_buffer.pkl")
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--states", default="ds1,ds2,ds3")
    p.add_argument("--ppo-model", default="models/curriculum_flow25_r3_best.zip")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes-per-track", type=int, default=3)
    p.add_argument("--bc-iters", type=int, default=800, help="distillation iters per round")
    p.add_argument("--demo-weight", type=int, default=2,
                   help="how many times offline demos are oversampled (anti-forgetting)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    bundle = args.bundle if os.path.isabs(args.bundle) else os.path.join(ROOT, args.bundle)
    model, actor, meta = load_world_model_actor(bundle, device=device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, eps=1e-5)

    from stable_baselines3 import PPO
    expert_ppo = PPO.load(os.path.join(ROOT, args.ppo_model))

    # expert demonstration pool = offline completed descents + growing DAgger set
    base_buf = EpisodeBuffer()
    base_buf.load(args.buffer if os.path.isabs(args.buffer) else os.path.join(ROOT, args.buffer))
    expert_eps = list(base_buf.expert_episodes()) or list(
        e for e in base_buf.episodes if e.get("continues")[-1] > 0.5)
    dagger_eps = []
    all_expert = expert_eps + dagger_eps

    rom_path = os.path.join(ROOT, args.rom)
    states = [s.strip() for s in args.states.split(",") if s.strip()]
    eval_env, existing_states = make_eval_env(rom_path, states, args.max_steps, args.flow_weight)
    ctrl_env = eval_env  # reuse the same persistent emulator for rollout + eval

    best = -1e18
    rng = np.random.default_rng(0)
    try:
        # Baseline eval of the incoming (seed) actor so DAgger can NEVER regress: we
        # only ever overwrite the bundle with a strictly better policy.
        rb = real_eval(model, actor, eval_env, existing_states, device, n_episodes=1,
                       max_steps=args.max_steps)
        best = rb["survival"] * 1e6 + rb["mean_steps"]
        print(f"[dagger] baseline (seed actor): {rb}", flush=True)
        for rnd in range(1, args.rounds + 1):
            # 1. on-policy data collection (labels from the expert)
            ctrl_env.state_path = os.path.join(ROOT, "data",
                                               f"Super Mario 64 DS (USA) (Rev 1).{existing_states[0]}")
            collected = 0
            for st in existing_states:
                ctrl_env.state_path = os.path.join(ROOT, "data",
                                                   f"Super Mario 64 DS (USA) (Rev 1).{st}")
                for _ in range(args.episodes_per_track):
                    ctrl = WorldModelController(model, actor, device=device, deterministic=True)
                    roll = _dagger_rollout(ctrl_env, ctrl, expert_ppo)
                    if roll is not None:
                        dagger_eps.append(roll)
                        collected += 1
            # oversample offline demos (anti-forgetting) + accumulate on-policy data
            all_expert = expert_eps * args.demo_weight + dagger_eps
            print(f"[dagger] round {rnd}: +{collected} on-policy eps "
                  f"(expert pool total {len(all_expert)})", flush=True)

            # 2. distill the latent actor on the combined on/off-policy expert set
            for _ in range(args.bc_iters):
                obs, ein, elab = _sample_all(all_expert, args.batch, args.seq_len, rng)
                actor_bc_update(model, actor, opt,
                                obs.to(device), ein.to(device), elab.to(device))

            # 3. evaluate on the real game, keep the best actor
            print(f"[dagger] === REAL EVAL @ round {rnd} ===", flush=True)
            r = real_eval(model, actor, eval_env, existing_states, device,
                          n_episodes=1, max_steps=args.max_steps)
            sig = r["survival"] * 1e6 + r["mean_steps"]
            if sig > best:
                best = sig
                torch.save(dict(world_model_cfg=meta.get("world_model_cfg",
                                             _cfg_from_model(model)),
                                model=model.state_dict(), actor=actor.state_dict(),
                                meta=dict(stage=f"dagger_r{rnd}", eval=r)), bundle)
                print(f"[dagger] saved improved actor -> {bundle}", flush=True)
    finally:
        if eval_env is not None:
            eval_env.close()


def _cfg_from_model(model):
    return dict(deter_dim=model.deter_dim, stoch_dim=model.stoch_dim,
                enc_dim=model.encoder.out_dim, num_actions=model.num_actions,
                obs_channels=4, hidden=model.prior_net.net[0].in_features)


def _sample_all(episodes, batch, seq_len, rng):
    obs = np.zeros((batch, seq_len, 4, 84, 84), np.float32)
    act_in = np.zeros((batch, seq_len), np.int64)
    label = np.zeros((batch, seq_len), np.int64)
    usable = [e for e in episodes if len(e["frames"]) >= seq_len + 2] or episodes
    for b in range(batch):
        e = usable[rng.integers(len(usable))]
        frames, T = e["frames"], len(e["frames"])
        padded = np.concatenate([np.zeros((3,) + frames.shape[1:], np.uint8), frames], 0)
        start = 0  # start-aligned so the belief matches the deployment controller
        labels = e.get("labels", None)
        for t in range(seq_len):
            i = start + t
            obs[b, t] = padded[i:i + 4].transpose(0, 1, 2).astype(np.float32) / 255.0
            act_in[b, t] = e["actions"][min(i, T - 1)]
            label[b, t] = (labels[min(i, len(labels) - 1)] if labels is not None
                           else e["actions"][min(i + 1, T - 1)])
    return (torch.from_numpy(obs), torch.from_numpy(act_in), torch.from_numpy(label))


if __name__ == "__main__":
    main()
