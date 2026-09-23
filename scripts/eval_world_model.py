"""Multi-episode real-game evaluation of a world-model agent bundle.

Because DeSmuME's multithreaded rasterizer makes pixel-level death detection
slightly non-deterministic at knife-edge states, a single rollout can complete or
fail; this harness averages across N episodes per track for an honest picture of
the distilled world-model policy's real-time performance.

    python scripts/eval_world_model.py --model models/wm_mario64ds_wm.pt --episodes 5
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TRACKS = ["ds1", "ds2", "ds3"]


def main():
    p = argparse.ArgumentParser(description="Evaluate world-model agent on the real game")
    p.add_argument("--model", required=True)
    p.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1350)
    p.add_argument("--flow-weight", type=float, default=2.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--tracks", default=",".join(TRACKS))
    args = p.parse_args()

    from src.env import Mario64DSEnv
    from src.world_model_controller import WorldModelController, load_world_model_actor

    model, actor, _ = load_world_model_actor(args.model, device=args.device)
    rom = os.path.join(ROOT, args.rom)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    env = None
    report = {}
    try:
        for ti, tr in enumerate(tracks):
            ss = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1).{tr}")
            if env is None:
                env = Mario64DSEnv(rom, ss, max_steps=args.max_steps,
                                   flow_weight=args.flow_weight)
            env.state_path = ss
            steps_list, rew_list, done_ct = [], [], 0
            for ep in range(args.episodes):
                ctrl = WorldModelController(model, actor, device=args.device,
                                            deterministic=not args.stochastic)
                obs, _ = env.reset()
                action = ctrl.reset(obs[:, :, 0])
                steps, R, done = 0, 0.0, False
                while not done and steps < args.max_steps:
                    obs, reward, term, trunc, _ = env.step(action)
                    action = ctrl.act(obs[:, :, 0])
                    R += float(reward); steps += 1
                    done = bool(term or trunc)
                survived = steps >= args.max_steps
                done_ct += int(survived)
                steps_list.append(steps); rew_list.append(R)
                print(f"  [{tr} ep{ep}] steps={steps}/{args.max_steps} "
                      f"R={R:.1f} {'SURVIVED' if survived else 'died'}", flush=True)
            report[tr] = dict(mean_steps=float(np.mean(steps_list)),
                              survival=f"{done_ct}/{args.episodes}",
                              mean_reward=float(np.mean(rew_list)))
            print(f"== {tr}: {report[tr]} ==", flush=True)
    finally:
        if env is not None:
            env.close()

    print("\n=== WORLD-MODEL AGENT REAL-GAME EVALUATION ===")
    for tr, v in report.items():
        print(f"  {tr}: mean_steps={v['mean_steps']:.0f}/{args.max_steps} "
              f"survival={v['survival']} mean_reward={v['mean_reward']:.1f}")


if __name__ == "__main__":
    main()
