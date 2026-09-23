"""Train the Mario 64 DS world model and an actor inside its imagination.

Pipeline (the fast-training core of the model-based approach):

  1. Fit the RSSM world model on a (small) buffer of REAL emulator transitions —
     learn prior/posterior dynamics, a reward head and a continue head.
  2. Freeze the model and optimize an actor-critic on *imagined* latent
     rollouts. No emulator is stepped here, so thousands of policy updates run
     per second of wall-clock (the reason this trains "very quickly" in real
     samples, exactly like the reference's Dyna-PPO in the PINN sim env).
  3. Periodically deploy the actor on the REAL game (belief-state controller)
     to measure survival / stage completion.

Example:
    python -m src.train_world_model --buffer data/wm_buffer.pkl --run-id wm_mario64ds \
        --model-iters 4000 --actor-iters 4000 --eval-every 1000
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from src.env import Mario64DSEnv
from src.wm_replay import EpisodeBuffer
from src.world_model import WorldModel, Latent, model_loss, _gauss_sample
from src.world_model_agent import ActorCritic, actor_critic_update, actor_bc_update

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _infer_expert_episodes(buffer, episodes_per_state=None):
    """Fallback expert labeling for buffers saved before source tagging.

    A random-policy episode never reaches the horizon (they die early), so any
    episode that ends with ``continue`` set (i.e. survived to ``max_steps``) is a
    successful expert descent and a valid distillation demonstration.
    """
    out = []
    for e in buffer.episodes:
        cont = e["continues"]
        if len(cont) and float(cont[-1]) > 0.5:
            out.append(e)
    return out


def _posterior_starts(model, obs_seq, action_seq):
    """Detach posterior latents at a random time index to seed imagination rollouts."""
    with torch.no_grad():
        out = model.observe_sequence(obs_seq, action_seq)
    T = out["post_mean"].shape[1]
    t = int(torch.randint(0, T, (1,)).item())
    d = out["post_deter"][:, t]
    m = out["post_mean"][:, t]
    s = out["post_std"][:, t]
    z = _gauss_sample(m, s)
    return Latent(d, m, s, z)


def make_eval_env(rom_path, states, max_steps, flow_weight):
    """Create the ONE persistent emulator used for every in-training eval.

    DeSmuME cannot be (re)initialized more than once per process (access
    violation), so a single instance is kept alive for the whole run and the
    loaded savestate is swapped per track inside real_eval().
    """
    existing = [s for s in states
                if os.path.exists(os.path.join(ROOT, "data",
                                               f"Super Mario 64 DS (USA) (Rev 1).{s}"))]
    if not existing:
        return None, []
    env = Mario64DSEnv(rom_path, os.path.join(ROOT, "data",
                       f"Super Mario 64 DS (USA) (Rev 1).{existing[0]}"),
                       max_steps=max_steps, flow_weight=flow_weight)
    return env, existing


def real_eval(model, actor, env, existing_states, device, n_episodes=1, max_steps=1350,
              deterministic=True, verbose=True):
    from src.world_model_controller import WorldModelController
    if env is None or not existing_states:
        return dict(survival=0, n=0, mean_steps=0.0, mean_reward=0.0)
    results = []
    for state in existing_states:
        env.state_path = os.path.join(ROOT, "data",
                                      f"Super Mario 64 DS (USA) (Rev 1).{state}")
        for ep in range(n_episodes):
            ctrl = WorldModelController(model, actor, device=device, deterministic=deterministic)
            obs, _ = env.reset()
            frame = obs[:, :, 0]
            action = ctrl.reset(frame)
            steps, R = 0, 0.0
            done = False
            while not done and steps < max_steps:
                obs, reward, terminated, truncated, _ = env.step(action)
                frame = obs[:, :, 0]
                action = ctrl.act(frame)
                R += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            survived = steps >= max_steps
            results.append((state, ep, R, steps, survived))
            if verbose:
                print(f"  [eval] {state} ep{ep}: R={R:.1f} steps={steps}/{max_steps} "
                      f"{'SURVIVED' if survived else 'died'}", flush=True)
    surv = sum(1 for r in results if r[4])
    mean_steps = np.mean([r[3] for r in results]) if results else 0
    mean_R = np.mean([r[2] for r in results]) if results else 0
    return dict(survival=surv, n=len(results), mean_steps=float(mean_steps),
                mean_reward=float(mean_R))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    buffer = EpisodeBuffer()
    buf_path = args.buffer if os.path.isabs(args.buffer) else os.path.join(ROOT, args.buffer)
    buffer.load(buf_path)
    print(f"[wm] loaded buffer: {buffer.total_steps} steps / {len(buffer)} episodes on {device}")

    wm_cfg = dict(deter_dim=args.deter_dim, stoch_dim=args.stoch_dim,
                  enc_dim=args.enc_dim, num_actions=6, obs_channels=4, hidden=args.hidden)
    model = WorldModel(**wm_cfg).to(device)
    actor = ActorCritic(args.deter_dim, args.stoch_dim, 6, hidden=args.hidden).to(device)
    model_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, eps=1e-5)

    rng = np.random.default_rng(args.seed)
    rom_path = os.path.join(ROOT, args.rom)
    eval_states = [s.strip() for s in args.states.split(",") if s.strip()]

    # Expert full-descent demonstrations for amortized policy distillation.
    expert_eps = buffer.expert_episodes()
    if not expert_eps:
        # Backward-compat: buffers collected before source-tagging are labeled by
        # their deterministic collection order (per state: N random then N ppo).
        expert_eps = _infer_expert_episodes(buffer, args.episodes_per_state)
    print(f"[wm] expert demonstration episodes for distillation: {len(expert_eps)}")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    out_bundle = os.path.join(ROOT, "models", f"{args.run_id}_wm.pt")

    best_sig = -1e18
    history = []
    t0 = time.time()

    def batch():
        obs, acts, rews, conts = buffer.sample(args.batch, args.seq_len, rng)
        return (obs.to(device), acts.to(device), rews.to(device), conts.to(device))

    n_iters = args.model_iters + max(args.actor_iters, args.bc_iters)
    # Persistent emulator for all evaluations (never re-initialized mid-process).
    eval_env, existing_eval_states = make_eval_env(rom_path, eval_states,
                                                    args.max_steps, args.flow_weight)
    try:
      for it in range(1, n_iters + 1):
        obs, acts, rews, conts = batch()
        am = bcm = None

        # --- world-model learning phase (real data) ---
        if it <= args.model_iters:
            model.train()
            mloss = model_loss(model, obs, acts, rews, conts, free_nats=args.free_nats)
            model_opt.zero_grad()
            mloss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            model_opt.step()

        # Both agent-learning phases run only after the model has been fit.
        if it > args.model_iters:
            model.eval()
            # (a) imagination RL (optional; can be disabled for pure distillation)
            if args.actor_updates_per_iter > 0 and it <= args.model_iters + args.actor_iters:
                starts = _posterior_starts(model, obs, acts)
                for _ in range(args.actor_updates_per_iter):
                    am = actor_critic_update(model, actor, actor_opt, starts,
                                             horizon=args.horizon, gamma=args.gamma,
                                             lam=args.lam, ent_coef=args.ent_coef,
                                             pessimism_beta=args.pessimism_beta,
                                             uncertainty_trunc=args.unc_trunc)
            # (b) amortized policy distillation (behaviour cloning on expert descents)
            if expert_eps and it <= args.model_iters + args.bc_iters:
                eobs, ein, elab = buffer.sample_expert(args.batch, args.bc_seq_len,
                                                       expert_eps, rng,
                                                       from_start=args.bc_from_start)
                bcm = actor_bc_update(model, actor, actor_opt,
                                      eobs.to(device), ein.to(device), elab.to(device))

        if it % args.log_every == 0 or it == 1:
            msg = f"[wm] iter {it}/{n_iters} ({time.time()-t0:.0f}s)"
            if it <= args.model_iters:
                msg += (f" model loss={float(mloss['loss']):.4f} kl={float(mloss['kl']):.3f} "
                        f"rew={float(mloss['rew']):.4f} cont={float(mloss['cont']):.4f}")
            if am is not None:
                msg += f" | actor R={am['mean_return']:.2f} vloss={am['value_loss']:.3f}"
            if bcm is not None:
                msg += f" | bc_acc={bcm['bc_acc']:.3f} bc_loss={bcm['bc_loss']:.3f}"
            print(msg, flush=True)

        if it % args.eval_every == 0 or it == n_iters:
            print(f"[wm] === REAL-EMULATOR EVAL @ iter {it} ===", flush=True)
            r = real_eval(model, actor, eval_env, existing_eval_states, device,
                          n_episodes=args.eval_episodes, max_steps=args.max_steps)
            r["iter"] = it
            history.append(r)
            sig = r["survival"] * 1e6 + r["mean_steps"]
            if sig > best_sig:
                best_sig = sig
                torch.save(dict(world_model_cfg=wm_cfg, model=model.state_dict(),
                                actor=actor.state_dict(), meta=dict(iter=it, eval=r)),
                           out_bundle)
                print(f"[wm] saved best world-model+actor bundle -> {out_bundle}", flush=True)
    finally:
        if eval_env is not None:
            eval_env.close()

    # final save (if never better than baseline, still persist)
    if not os.path.exists(out_bundle):
        torch.save(dict(world_model_cfg=wm_cfg, model=model.state_dict(),
                        actor=actor.state_dict(), meta=dict(iter=n_iters)), out_bundle)
    print(f"[wm] done in {time.time()-t0:.0f}s. Bundle: {out_bundle}")
    return out_bundle


def main():
    p = argparse.ArgumentParser(description="Train Mario 64 DS world model + imagination actor")
    p.add_argument("--buffer", default="data/wm_buffer.pkl")
    p.add_argument("--run-id", default="wm_mario64ds")
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--states", default="ds1,ds2,ds3", help="tracks for periodic real eval")
    p.add_argument("--seed", type=int, default=0)
    # model / actor architecture
    p.add_argument("--deter-dim", type=int, default=128)
    p.add_argument("--stoch-dim", type=int, default=32)
    p.add_argument("--enc-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    # training
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--bc-seq-len", type=int, default=250,
                   help="distillation window length")
    p.add_argument("--bc-from-start", action="store_true",
                   help="start-align distillation windows (belief matches deployment); "
                        "default False uses diverse mid-episode windows")
    p.add_argument("--model-iters", type=int, default=4000)
    p.add_argument("--actor-iters", type=int, default=4000)
    p.add_argument("--actor-updates-per-iter", type=int, default=2)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--free-nats", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # safe-MBRL pessimism (reference: deep-ensemble epistemic truncation)
    p.add_argument("--pessimism-beta", type=float, default=1.0,
                   help="subtract beta*model-uncertainty from imagined reward")
    p.add_argument("--unc-trunc", type=float, default=1.5,
                   help="truncate imagined rollout when prior std exceeds this")
    # amortized policy distillation (behaviour cloning on expert descents)
    p.add_argument("--bc-iters", type=int, default=6000,
                   help="run policy-distillation updates for the first N iterations")
    p.add_argument("--episodes-per-state", type=int, default=6,
                   help="fallback expert-labeling hint for untagged buffers")
    p.add_argument("--log-every", type=int, default=100)
    # evaluation
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
