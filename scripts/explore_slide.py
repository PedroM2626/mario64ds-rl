"""Find a way down a slide without a human: savestate-branching search around a policy.

The reference PPO jumps over the side wall of the `ds3` slide at a curve and
gets stuck outside the track. This search starts where the policy is still on
the slide and repeats, from the furthest safe point found so far:

  1. run N rollouts in parallel from the current savestate: the policy's greedy
     action, replaced with probability --eps by a random one held for 1-6
     decisions;
  2. score each decision by the straight-line distance to the finish, and keep
     the best decision that the rollout then survived for --safety more
     decisions (a fall into the void ends the episode within that margin);
  3. continue from there (one savestate per stage), unless the gain is below
     --min-gain for --patience stages in a row.

The emulator is deterministic, so the result is just the action sequence from
the track's start; it is replayed from scratch at the end, and written in the
recorder's attempt format (with checkpoint savestates) so that
scripts/prepare_human_route.py can turn it into a route and a curriculum.

Usage:
    python scripts/explore_slide.py --track ds3 --prefix 100
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

_W = {}


def _env(track, state):
    from src.env import Mario64DSEnv
    from src.envs import rom_path
    return Mario64DSEnv(rom_path=rom_path(), state_path=state, max_steps=100000, flow_weight=0.0,
                        dist_reward_enabled=False, track_id=track)


def _init(track, model):
    from scripts.eval_finish_line import make_policy
    from src.envs import savestate_path
    _W["env"] = _env(track, savestate_path(track))
    _W["policy"] = make_policy("ppo", model, None, "cpu")


def _play(env, policy, state, history, n, eps, rng, forced=None):
    """From savestate `state` with frame-stack `history`: forced actions, or policy+epsilon."""
    reset, act = policy
    env.state_path = state
    obs, info = env.reset()
    reset(obs, history)
    frames = list(history) if history is not None else [np.zeros((84, 84), np.uint8)] * 3
    actions, pos, died, goal, hold, held = [], [], False, False, 0, 0
    for t in range(len(forced) if forced is not None else n):
        greedy = act(obs)
        frames.append(obs[:, :, 0].copy())
        if forced is not None:
            a = int(forced[t])
        else:
            if hold == 0 and rng is not None and rng.random() < eps:
                held, hold = int(rng.integers(env.action_space.n)), int(rng.integers(1, 7))
            a = held if hold > 0 else greedy
            hold = max(hold - 1, 0)
        obs, _, term, _, info = env.step(a)
        actions.append(a)
        pos.append(info["pos_3d"].copy())
        if term:
            died, goal = bool(info.get("died")), bool(info.get("reached_goal"))
            break
    return actions, np.array(pos), died, goal, frames[-3:]


def _rollout(job):
    state, history, n, eps, seed = job
    actions, pos, died, goal, _ = _play(_W["env"], _W["policy"], state, history, n, eps,
                                        np.random.default_rng(seed))
    return actions, pos, died, goal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="ds3")
    ap.add_argument("--model", default="models/curriculum_flow25_r3_best.zip")
    ap.add_argument("--prefix", type=int, default=100,
                    help="Greedy policy decisions before the search starts (still on the slide)")
    ap.add_argument("--rollouts", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=160)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--safety", type=int, default=40, help="Decisions a branch point must be survived by")
    ap.add_argument("--min-gain", type=float, default=150.0)
    ap.add_argument("--patience", type=int, default=3, help="Stages without gain before stopping")
    ap.add_argument("--max-stages", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/explore")
    args = ap.parse_args()

    from scripts.eval_finish_line import make_policy
    from src.envs import savestate_path
    from src.ram_state import TRACK_GOALS
    goal = TRACK_GOALS[args.track]["target_pos"]
    start = savestate_path(args.track)
    model = os.path.join(ROOT_DIR, args.model)
    run = os.path.join(ROOT_DIR, args.out, f"{args.track}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run)
    env = _env(args.track, start)
    policy = make_policy("ppo", model, None, "cpu")

    def advance(state, history, actions, name):
        """Replay `actions` from `state`, save the resulting savestate, return it and the history."""
        acts, p, d, g, hist = _play(env, policy, state, history, 0, 0.0, None, forced=actions)
        assert not d and len(acts) == len(actions), "replaying a kept branch died"
        path = os.path.join(run, name)
        env.emu.savestate.save_file(path)
        return path, hist, p[-1]

    acts, _, died, _, _ = _play(env, policy, start, None, args.prefix, 0.0, None)
    assert not died, "the greedy prefix already dies; use a shorter --prefix"
    full = list(acts)
    state, history, p_last = advance(start, None, acts, "stage-000.dst")
    best = float(np.linalg.norm(p_last - goal))
    print(f"prefix {len(full)} decisions, distance {best:.1f}", flush=True)
    log, stale, finished = [], 0, False
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.track, model)) as pool:
        for stage in range(1, args.max_stages + 1):
            jobs = [(state, history, args.horizon, args.eps, 1000 * stage + i) for i in range(args.rollouts)]
            cand, deaths = None, 0
            for actions, p, d, g in pool.map(_rollout, jobs):
                deaths += int(d)
                if g:
                    cand = (0.0, actions, len(actions), True)
                    break
                alive = len(actions) - (args.safety if d else 0)
                if alive <= 0:
                    continue
                dist = np.linalg.norm(p[:alive] - goal, axis=1)
                i = int(np.argmin(dist))
                if cand is None or dist[i] < cand[0]:
                    cand = (float(dist[i]), actions, i + 1, False)
            entry = {"stage": stage, "deaths": deaths}
            if cand is None:
                stale += 1
                entry["best"] = None
            else:
                dist, actions, k, g = cand
                gain = best - dist
                entry.update(best=round(dist, 1), gain=round(gain, 1))
                if g or gain >= args.min_gain:
                    full += actions[:k]
                    best, stale = dist, 0
                    if g:
                        finished = True
                    else:
                        state, history, _ = advance(state, history, actions[:k], f"stage-{stage:03d}.dst")
                else:
                    stale += 1
            entry.update(decisions=len(full), finish=finished)
            log.append(entry)
            print(json.dumps(entry), flush=True)
            with open(os.path.join(run, "search.json"), "w", encoding="utf-8") as fh:
                json.dump({"track": args.track, "args": vars(args), "actions": [int(a) for a in full],
                           "log": log}, fh)
            if finished or stale >= args.patience:
                break

    # Independent replay of the whole sequence from the track start.
    acts, pos, died, reached, _ = _play(env, policy, start, None, 0, 0.0, None, forced=full)
    env.close()
    result = {"track": args.track, "decisions": len(acts), "finish": reached, "died": died,
              "min_dist": round(float(np.min(np.linalg.norm(pos - goal, axis=1))), 1),
              "final_pos": [round(float(v), 1) for v in pos[-1]]}
    with open(os.path.join(run, "search.json"), "w", encoding="utf-8") as fh:
        json.dump({"track": args.track, "args": vars(args), "actions": [int(a) for a in full],
                   "log": log, "replay": result}, fh, indent=1)
    np.save(os.path.join(run, "positions.npy"), pos)
    print("RESULT " + json.dumps(result), flush=True)
    print(f"-> {run}", flush=True)


if __name__ == "__main__":
    main()
