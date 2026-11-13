"""One value-based agent for the three slides (ds1, ds2, ds3) with the reference's setup.

Setup of the agent in the video this follows: reward only for collecting a coin (no penalty for dying
or falling, no distance or finish reward, no demonstrations), ten actions (None, Up, Up-R, R, Down-R,
Down, Down-L, L, Up-L, A), the same network for every slide, gamma < 1 so that earlier coins are worth
more. The algorithm is BTR-style (src/btr.py: IQN + dueling + noisy nets + n-step + Munchausen +
prioritised replay + IMPALA). The coin counter is read from RAM (src/ram_state.COIN_COUNT_ADDR).

Episode end rules (not rewards): a fall below the course's kill_y, the game's own respawn after a fall
(a displacement above --respawn-jump in one decision) and, on ds3 only, a long free fall
(--ds3-fall-speed); the finish region ends the episode; a black screen does not.

Usage:
    python scripts/train_btr.py --run btr_ref10 --steps 20000000
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402


TTC_ROUTE = "config/routes/ttc_human_v1.json"


def ref_env_kwargs(track, respawn_jump=2000.0, ds3_fall_speed=140.0, goal_reward=0.0, action_set="ref10",
                   waypoint_reward=1.0, ttc_star_reward=100.0, stall_steps=0, fall_drop=0.0, start_states=None,
                   full_start_prob=1.0):
    """Mario64DSEnv settings of the coin-only agent (everything but the coin counter pays nothing, unless
    goal_reward > 0: a one-time reward when the episode ends in the finish region)."""
    kw = dict(action_set=action_set, death_penalty=0.0, step_penalty=0.0, flow_weight=0.0,
              dist_reward_enabled=False, goal_reward=goal_reward, star_reward=0.0, ram_coin_reward=1.0,
              use_ram_death=True, black_screen_death=False, respawn_jump=respawn_jump)
    if track == "ds3":
        kw.update(fall_speed=ds3_fall_speed, fall_steps=15)
    if track == "ds8":
        # Tick Tock Clock: +waypoint_reward for every waypoint of the recorded human ascent reached for the first time
        # (the route progress only ratchets forward) and `ttc_star_reward` once for the star; no coins, no height term.
        kw.update(ram_coin_reward=0.0, dist_reward_enabled=True, dist_reward_mode="waypoint",
                  route_path=os.path.join(ROOT_DIR, TTC_ROUTE), waypoint_reward=waypoint_reward, star_reward=ttc_star_reward,
                  stall_steps=stall_steps, fall_drop=fall_drop)
        if start_states:
            # training only: with probability 1 - full_start_prob an episode starts from one of these savestates
            # (its own place on the route; the route behind it is never paid). Evaluation never passes them.
            kw.update(start_states=list(start_states), full_start_prob=full_start_prob)
    return kw


def make_ref_env(track, rank=0, seed=0, max_steps=1500, **kw):
    from src.envs import make_env, rom_path, savestate_path
    k = ref_env_kwargs(track, **kw)
    return make_env(rom_path(), savestate_path(track), rank=rank, seed=seed, max_steps=max_steps, monitor=False,
                    **{a: k.pop(a) for a in ("step_penalty", "flow_weight", "dist_reward_enabled", "goal_reward",
                                             "star_reward", "use_ram_death")}, **k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="btr_ref10")
    ap.add_argument("--tracks", default="ds1,ds2,ds3")
    ap.add_argument("--envs-per-track", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20_000_000, help="environment steps (decisions) in total")
    ap.add_argument("--max-steps", type=int, default=1500, help="decisions per episode")
    ap.add_argument("--capacity", type=int, default=400_000, help="replay capacity in steps (all envs)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--utd", type=int, default=1, help="gradient updates per step of all envs")
    ap.add_argument("--learning-starts", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--ckpt-every", type=int, default=250_000)
    ap.add_argument("--log-every", type=int, default=20_000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--goal-reward", type=float, default=0.0,
                    help="one-time reward for ending an episode in the finish region (0 = the reference setup)")
    ap.add_argument("--action-set", default="ref10", choices=["ref10", "ref13", "legacy6", "compound12"])
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop after this many minutes of training (0 = no limit)")
    ap.add_argument("--stall-steps", type=int, default=0,
                    help="ds8: end the episode when no new waypoint was reached for this many decisions (0 = off)")
    ap.add_argument("--fall-drop", type=float, default=0.0,
                    help="ds8: end the episode when Mario falls this many units below the highest point "
                         "reached (0 = off); the human route's largest drop is about 516")
    ap.add_argument("--start-states", default="",
                    help="ds8: directory of *.dst savestates used as extra episode starts (default: none, full start only)")
    ap.add_argument("--full-start-prob", type=float, default=0.5,
                    help="with --start-states: probability that an episode starts from the full ds8 savestate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from stable_baselines3.common.vec_env import SubprocVecEnv

    from src.btr import BTRAgent, PrioritizedStepBuffer

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    names = [t for t in tracks for _ in range(args.envs_per_track)]
    E = len(names)
    out = os.path.join(ROOT_DIR, "models", args.run)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
    np.random.seed(args.seed)

    start_states = sorted(glob.glob(os.path.join(ROOT_DIR, args.start_states, "*.dst"))) if args.start_states else []
    if args.start_states and not start_states:
        raise SystemExit(f"no .dst files in {args.start_states}")
    vec = SubprocVecEnv([make_ref_env(t, rank=i, seed=args.seed, max_steps=args.max_steps, goal_reward=args.goal_reward,
                         action_set=args.action_set, stall_steps=args.stall_steps,
                         fall_drop=args.fall_drop, start_states=start_states, full_start_prob=args.full_start_prob)
                         for i, t in enumerate(names)])
    agent = BTRAgent(int(vec.action_space.n), device=args.device, lr=args.lr, gamma=args.gamma,
                     n_step=args.n_step)
    if args.resume:
        from src.env import ACTION_SET_EXTENDS
        ext = ACTION_SET_EXTENDS.get(args.action_set)
        agent.load(args.resume, sources=ext[1] if ext else None)
    buf = PrioritizedStepBuffer(E, args.capacity // E, n_step=args.n_step, gamma=args.gamma)

    obs = vec.reset()                                         # (E, 84, 84, 1)
    stacks = [deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[i, :, :, 0]], maxlen=4) for i in range(E)]
    ep_ret, ep_len = np.zeros(E), np.zeros(E, int)
    recent = {t: deque(maxlen=30) for t in tracks}            # (return, decisions, route arc length) of the last episodes
    routed = [t for t in tracks if t == "ds8"]
    log_path = os.path.join(ROOT_DIR, "results", f"{args.run}_train.csv")
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if os.path.getsize(log_path) == 0:
        writer.writerow(["steps", "minutes", "updates", "loss"] + [f"{t}_coins" for t in tracks] +
                        [f"{t}_len" for t in tracks] + [f"{t}_route_s" for t in routed])
    t0, step, next_log, next_ckpt, loss = time.time(), 0, args.log_every, args.ckpt_every, float("nan")
    while step < args.steps:
        if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
            break
        batch_stacks = np.stack([np.stack(list(s)) for s in stacks])           # (E, 4, 84, 84)
        actions = agent.act(batch_stacks, noisy=True)
        obs, rewards, dones, infos = vec.step(actions)
        buf.add(batch_stacks[:, -1], actions, rewards, dones)
        ep_ret += rewards
        ep_len += 1
        for i in range(E):
            if dones[i]:
                recent[names[i]].append((ep_ret[i], ep_len[i], infos[i].get("route_progress") or 0.0))
                ep_ret[i], ep_len[i] = 0.0, 0
                stacks[i] = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[i, :, :, 0]], maxlen=4)
            else:
                stacks[i].append(obs[i, :, :, 0])
        step += E
        if step >= args.learning_starts:
            beta = 0.4 + 0.6 * min(1.0, step / args.steps)
            for _ in range(args.utd):
                b = buf.sample(args.batch, beta)
                loss, td = agent.learn(b)
                buf.update(b["e"], b["s"], td)
        if step >= next_log:
            next_log += args.log_every
            row = [step, round((time.time() - t0) / 60, 1), agent.updates, round(loss, 4)]
            row += [round(float(np.mean([r[0] for r in recent[t]])), 2) if recent[t] else "" for t in tracks]
            row += [round(float(np.mean([r[1] for r in recent[t]]))) if recent[t] else "" for t in tracks]
            row += [round(float(np.mean([r[2] for r in recent[t]]))) if recent[t] else "" for t in routed]
            writer.writerow(row)
            log.flush()
            print("steps {} | {} min | loss {} | coins {} | decisions {}".format(
                row[0], row[1], row[3], dict(zip(tracks, row[4:4 + len(tracks)])),
                dict(zip(tracks, row[4 + len(tracks):4 + 2 * len(tracks)]))) +
                ("" if not routed else " | route arc length {}".format(row[4 + 2 * len(tracks):])), flush=True)
        if step >= next_ckpt:
            next_ckpt += args.ckpt_every
            agent.save(os.path.join(out, f"step_{step}.pt"))
    agent.save(os.path.join(out, "final.pt"))
    vec.close()
    print("Training complete")


if __name__ == "__main__":
    main()
