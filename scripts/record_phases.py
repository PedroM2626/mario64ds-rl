"""Records 1 video per track (ds1, ds2, ds3) showcasing the trained agent playing.

Each track runs in an isolated subprocess (DeSmuME causes access violations
if multiple emulators run concurrently in the same process). Records full-color
RGB top screen (256x192) via env.get_screen_rgb() rather than the agent's
84x84 grayscale observation. Outputs: videos/phase_ds1.mp4, phase_ds2.mp4, phase_ds3.mp4.

Examples:
    python scripts/record_phases.py --model models/curriculum_flow25_r3_best.zip
    python scripts/record_phases.py --model models/grid_ppo_nature_s0_500k_best.zip --fps 15
"""

import argparse
import os
import subprocess
import sys

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PHASES = ["ds1", "ds2", "ds3"]


def main():
    parser = argparse.ArgumentParser(description="Record videos of the agent playing each track")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SB3 model checkpoint (.zip)")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=15,
                        help="Video FPS (frameskip=4 on 60 Hz -> 15 = real time)")
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--flow-weight", type=float, default=2.5,
                        help="Flow reward weight (should match training)")
    parser.add_argument("--step-penalty", type=float, default=0.02)
    parser.add_argument("--single", type=str, default=None,
                        help=argparse.SUPPRESS)  # Internal use: records single track and exits
    args = parser.parse_args()

    out_dir_full = os.path.join(ROOT_DIR, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir

    if args.single is None:
        # Orchestrator mode: 1 subprocess per phase (isolates DeSmuME C++ memory space)
        os.makedirs(out_dir_full, exist_ok=True)
        for phase in PHASES:
            out = os.path.join(out_dir_full, f"phase_{phase}.mp4")
            print(f"\n=== Recording {phase} -> {out} ===", flush=True)
            r = subprocess.run(
                [sys.executable, __file__, "--model", args.model, "--rom", args.rom,
                 "--out-dir", args.out_dir, "--fps", str(args.fps),
                 "--max-steps", str(args.max_steps), "--single", phase,
                 "--flow-weight", str(args.flow_weight),
                 "--step-penalty", str(args.step_penalty)],
                cwd=ROOT_DIR
            )
            if r.returncode != 0:
                print(f"WARNING: recording {phase} failed (exit={r.returncode})", flush=True)
        return

    # Single mode: record ONE phase in this process
    _record_single(args, out_dir_full)


def _record_single(args, out_dir_full):
    import imageio
    import numpy as np
    from stable_baselines3 import PPO

    from src.env import Mario64DSEnv

    rom_path = os.path.join(ROOT_DIR, args.rom) if not os.path.isabs(args.rom) else args.rom
    savestate_path = os.path.join(ROOT_DIR, "data", f"Super Mario 64 DS (USA) (Rev 1).{args.single}")
    out_path = os.path.join(out_dir_full, f"phase_{args.single}.mp4")
    model_path = os.path.join(ROOT_DIR, args.model) if not os.path.isabs(args.model) else args.model

    model = PPO.load(model_path)

    env = Mario64DSEnv(rom_path=rom_path, state_path=savestate_path,
                       step_penalty=args.step_penalty, flow_weight=args.flow_weight)
    obs, info = env.reset()

    # Accurate FrameStack initialization matching SB3 VecFrameStack: on reset,
    # SB3 initializes the stack as [0, 0, 0, reset_frame] (3 zeros + current frame in last slot),
    # NOT 4 copies of reset_frame (convention used in gymnasium FrameStackObservation).
    # Replicating this exact initialization ensures trajectory parity with training.
    from collections import deque
    first = obs[:, :, 0]
    stack_deque = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)

    def make_stack():
        # (4, 84, 84) CHW — matches training tensor format (VecFrameStack + VecTransposeImage)
        return np.stack(list(stack_deque), axis=0)[np.newaxis, :]

    frames = [env.get_screen_rgb()]
    done, steps, total_reward = False, 0, 0.0
    while not done and steps < args.max_steps:
        action, _ = model.predict(make_stack(), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        stack_deque.append(obs[:, :, 0])
        total_reward += float(reward)
        done = bool(terminated or truncated)
        frames.append(env.get_screen_rgb())
        steps += 1

    env.close()

    # frameskip=4: env advances at 15 simulation steps/s; fps=15 corresponds to real-time playback
    with imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    survived = steps >= args.max_steps
    print(f"{args.single}: reward={total_reward:.2f}, steps={steps}, "
          f"{'TIMEOUT (survived)' if survived else 'DEATH'} | "
          f"{out_path} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
