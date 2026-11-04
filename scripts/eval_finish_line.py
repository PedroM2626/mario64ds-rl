"""Did the controller reach the end of the slide? Measured against the RAM finish line.

The slide results in the README were scored by surviving 1350 decisions. This
replays a controller deterministically on each track and reports what
src/ram_state.TRACK_GOALS says about the finish (goal radius 250 units): whether
it was reached and at which decision, the closest straight-line distance to it,
and how the episode ended. Two environments:

  historical  black-screen death only (a fall into the void respawns Mario),
              the setting the published slide results were produced under
  ram         a fall below the course's kill_y ends the episode (position rule)
  struct      legacy rule: the 0x0217F760 fast-fall clamp also ends it, which
              fires on legitimate long drops (e.g. the ds1 exit into the end room)

The finish is target_pos within 250 units or the measured end area
(TRACK_GOALS finish_region).

Each track runs in its own process (two DeSmuME cores in one process crash).

Usage:
    python scripts/eval_finish_line.py --controller ppo --model models/curriculum_flow25_r3_best.zip
    python scripts/eval_finish_line.py --controller mem --model models/wm_mario64ds_mem.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

TRACKS = ("ds1", "ds2", "ds3")


def make_policy(controller, model_path, bundle, device):
    """Returns (reset_fn(obs), act_fn(obs) -> int) sharing SB3's [0,0,0,frame] stack."""
    import torch
    dq = deque(maxlen=4)
    if controller == "ppo":
        from src.eval import _load_sb3_with_legacy_patch
        from stable_baselines3 import PPO
        model = _load_sb3_with_legacy_patch(PPO, model_path, device=device)

        def act(obs):
            dq.append(obs[:, :, 0])
            x = np.stack(list(dq), axis=0)[None]
            return int(model.predict(x, deterministic=True)[0][0])
    else:
        from src.distill_memoryless import _fresh_model, _load_head
        wm = _fresh_model(bundle, device)
        head = _load_head(wm, torch.load(model_path, map_location=device), device)

        def act(obs):
            dq.append(obs[:, :, 0])
            a = np.stack(list(dq), axis=0).astype(np.float32) / 255.0
            return int(head.act(wm.encoder(torch.from_numpy(a)[None].to(device))).item())

    def reset(obs, history=None):
        """history: the 3 frames before obs (to resume from a mid-run savestate)."""
        dq.clear()
        dq.extend(list(history) if history is not None else [np.zeros_like(obs[:, :, 0])] * 3)

    return reset, act


def run_track(args):
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.ram_state import TRACK_GOALS, Sm64DsState

    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(args.track), max_steps=args.max_steps,
                       flow_weight=1.0, use_ram_death=args.env != "historical", dist_reward_enabled=False,
                       track_id=args.track, death_rule="struct" if args.env == "struct" else "position")
    kill_y = TRACK_GOALS[args.track]["kill_y"]
    reset, act = make_policy(args.controller, os.path.join(ROOT_DIR, args.model),
                             os.path.join(ROOT_DIR, args.bundle), args.device)
    goal = TRACK_GOALS[args.track]["target_pos"]
    try:
        obs, info = env.reset()
        reset(obs)
        d0 = float(np.linalg.norm(info["pos_3d"] - goal))
        dmin, fast, void, was_fast, was_void, t = d0, 0, 0, False, False, 0
        for t in range(1, args.max_steps + 1):
            obs, _, term, trunc, info = env.step(act(obs))
            p = info.get("pos_3d")
            if p is not None:
                dmin = min(dmin, float(np.linalg.norm(p - goal)))
            # fast: entries into the 0x0217F760 clamp (any long drop; the
            # dashboard's "in-track falls"). void: drops below kill_y.
            f_now = info.get("ram") is not None and Sm64DsState.is_dead(info["ram"])
            v_now = Sm64DsState.fell_out_of_course(p, kill_y)
            fast += int(f_now and not was_fast)
            void += int(v_now and not was_void)
            was_fast, was_void = f_now, v_now
            if term or trunc:
                break
        end = ("finish" if info.get("reached_goal") else "died" if info.get("died")
               else "horizon")
        row = {"track": args.track, "env": args.env, "controller": args.controller, "model": args.model,
               "end": end, "decisions": t, "start_dist": round(d0, 1), "min_dist": round(dmin, 1),
               "fast_falls": fast, "void_falls": void}
    finally:
        env.close()
    print("__ROW__" + json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=["ppo", "mem"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", default="models/wm_mario64ds_wm.pt", help="RSSM bundle (mem only)")
    ap.add_argument("--tracks", default=",".join(TRACKS))
    ap.add_argument("--envs", default="historical,ram")
    ap.add_argument("--max-steps", type=int, default=1350)
    ap.add_argument("--device", default="cpu", help="cpu is the published instrument")
    ap.add_argument("--out", default=None)
    ap.add_argument("--track", help=argparse.SUPPRESS)
    ap.add_argument("--env", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.track:
        return run_track(args)

    rows = []
    for env_name in args.envs.split(","):
        for track in args.tracks.split(","):
            cmd = [sys.executable, os.path.abspath(__file__), "--controller", args.controller,
                   "--model", args.model, "--bundle", args.bundle, "--max-steps", str(args.max_steps),
                   "--device", args.device, "--track", track, "--env", env_name]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
            line = next((ln for ln in out.stdout.splitlines() if ln.startswith("__ROW__")), None)
            if line is None:
                raise RuntimeError(f"{track}/{env_name} produced no result:\n{out.stderr[-2000:]}")
            row = json.loads(line[len("__ROW__"):])
            rows.append(row)
            print(f"{env_name:10s} {track}  {row['end']:8s} decisions={row['decisions']:5d} "
                  f"min_dist={row['min_dist']:9.1f} / start {row['start_dist']:9.1f} "
                  f"void_falls={row['void_falls']} fast_falls={row['fast_falls']}", flush=True)
    name = os.path.splitext(os.path.basename(args.model))[0]
    out = args.out or os.path.join(ROOT_DIR, "results", f"finish_line_{name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"max_steps": args.max_steps, "goal_radius": 250.0, "device": args.device,
                   "rows": rows}, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
