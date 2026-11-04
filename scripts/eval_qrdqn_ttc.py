"""Academic evaluation script for QR-DQN models on Tick Tock Clock (ds8).
Evaluates models deterministically on ds8 and logs exact RAM trajectories,
action distributions, initial Q-values, and progress milestones.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import collections
import numpy as np
import torch
from sb3_contrib import QRDQN

from src.env import Mario64DSEnv
from src.envs import savestate_path, rom_path
from src.ram_state import TRACK_GOALS

ROM_PATH = rom_path()
STATE_PATH = savestate_path("ds8")

def evaluate_model(model_path: str, dist_mode: str = "progress", max_steps: int = 1350,
                   flow_weight: float = 0.2, route_path: str | None = None):
    print(f"\n=======================================================")
    print(f"Evaluating Model: {os.path.basename(model_path)}")
    print(f"Distance Mode: {dist_mode} | flow_weight={flow_weight}")
    print(f"=======================================================")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    model = QRDQN.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    # The action set is recoverable from the checkpoint's action space size.
    action_set = {6: "legacy6", 12: "compound12"}[int(model.action_space.n)]

    env = Mario64DSEnv(
        rom_path=ROM_PATH,
        state_path=STATE_PATH,
        max_steps=max_steps,
        frameskip=4,
        step_penalty=0.02,
        flow_weight=flow_weight,
        dist_reward_enabled=True,
        dist_weight=0.05,
        dist_reward_mode=dist_mode,
        goal_reward=200.0,
        star_reward=500.0,
        use_ram_death=True,
        track_id="ds8",
        action_set=action_set,
        route_path=route_path,
    )
    ACTION_NAMES = env.action_names

    obs, _ = env.reset()
    
    # 4-frame buffer stack matching training VecFrameStack
    from collections import deque
    # SB3's VecFrameStack opens as [0, 0, 0, reset_frame] (not four copies of
    # it); matching that keeps this evaluator on the EvalCallback trajectory.
    frame_stack = deque([np.zeros_like(obs.squeeze(-1))] * 3, maxlen=4)
    frame_stack.append(obs.squeeze(-1))
    
    def get_stacked_obs():
        arr = np.stack(list(frame_stack), axis=0) # shape (4, 84, 84)
        return arr[None, ...] # shape (1, 4, 84, 84)

    # Initial state Q-values
    initial_stack = get_stacked_obs()
    with torch.no_grad():
        obs_t = model.policy.obs_to_tensor(initial_stack)[0]
        quantiles = model.policy.quantile_net(obs_t) # [1, n_quantiles, n_actions]
        q_vals_s0 = quantiles.mean(dim=1).cpu().numpy()[0]

    init_pos = env.ram.get_pos_3d()
    init_y = init_pos[1]
    target_pos = TRACK_GOALS["ds8"]["target_pos"]
    delta_y_total = target_pos[1] - init_y # 7145.0 - (-4710.0) = 11855.0

    print(f"Initial Mario Position: X={init_pos[0]:.1f}, Y={init_pos[1]:.1f}, Z={init_pos[2]:.1f}")
    print(f"Target Star Position:   X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
    print(f"Total Course Height:    Delta Y = {delta_y_total:.1f} units")
    print("\nInitial State Q-values Q(s0, a):")
    for a_idx, a_name in enumerate(ACTION_NAMES):
        print(f"  Action {a_idx} ({a_name:20s}): Q = {q_vals_s0[a_idx]:+.4f}")
    best_a0 = int(np.argmax(q_vals_s0))
    print(f"  Greedy Choice at s0: Action {best_a0} ({ACTION_NAMES[best_a0]})")

    action_counts = collections.Counter()
    trajectory_y = [init_y]
    trajectory_pos = [init_pos]
    total_reward = 0.0
    flow_reward_total = 0.0
    dist_reward_total = 0.0
    steps = 0
    termination_reason = "truncated"

    done = False
    while not done:
        stacked = get_stacked_obs()
        action, _ = model.predict(stacked, deterministic=True)
        action = int(action[0] if isinstance(action, (np.ndarray, list)) else action)
        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        frame_stack.append(obs.squeeze(-1))
        steps += 1
        total_reward += reward

        pos = env.ram.get_pos_3d()
        trajectory_y.append(pos[1])
        trajectory_pos.append(pos)

        flow_reward_total += info.get("flow_reward", 0.0)
        dist_reward_total += info.get("dist_reward", 0.0)

        if terminated:
            if info.get("star_collected", False):
                termination_reason = "star_collected"
            elif info.get("died", False):
                termination_reason = "died (abyss/RAM death)"
            else:
                termination_reason = "terminated (goal)"
            done = True
        elif truncated:
            termination_reason = "truncated (max steps reached)"
            done = True

    env.close()

    max_y = float(np.max(trajectory_y))
    min_y = float(np.min(trajectory_y))
    final_pos = trajectory_pos[-1]
    delta_y_climbed = max_y - init_y
    climb_pct = (delta_y_climbed / delta_y_total) * 100.0

    # Minimum distance to star
    dists_to_star = [float(np.linalg.norm(np.array(p) - target_pos)) for p in trajectory_pos]
    min_dist_to_star = float(np.min(dists_to_star))
    init_dist_to_star = float(np.linalg.norm(np.array(init_pos) - target_pos))

    print(f"\n--- Episode Summary ---")
    print(f"Steps Survived:         {steps} / {max_steps}")
    print(f"Termination Reason:     {termination_reason}")
    print(f"Final Mario Position:   X={final_pos[0]:.1f}, Y={final_pos[1]:.1f}, Z={final_pos[2]:.1f}")
    print(f"Max Height Y Reached:   {max_y:.1f} (Delta Y = {delta_y_climbed:+.1f} units)")
    print(f"Tower Ascent Climax:    {climb_pct:.2f}% of full course")
    print(f"Distance to Star:       Initial={init_dist_to_star:.1f} -> Min={min_dist_to_star:.1f} -> Final={dists_to_star[-1]:.1f}")
    print(f"Total Episode Return:   {total_reward:+.4f}")
    print(f"  Flow Reward Sum:      {flow_reward_total:+.4f}")
    print(f"  Dist Reward Sum:      {dist_reward_total:+.4f}")
    print(f"  Step Penalties:       {-steps * 0.02:+.4f}")
    print(f"  Terminal/Other:       {total_reward - flow_reward_total - dist_reward_total + steps * 0.02:+.4f}")
    
    print("\nAction Distribution:")
    for a_idx, a_name in enumerate(ACTION_NAMES):
        cnt = action_counts.get(a_idx, 0)
        pct = (cnt / steps) * 100.0 if steps > 0 else 0.0
        print(f"  Action {a_idx} ({a_name:20s}): {cnt:5d} ({pct:5.1f}%)")

    return {
        "model": os.path.basename(model_path),
        "dist_mode": dist_mode,
        "steps": steps,
        "termination": termination_reason,
        "init_pos": init_pos,
        "final_pos": final_pos,
        "max_y": max_y,
        "delta_y": delta_y_climbed,
        "climb_pct": climb_pct,
        "init_dist": init_dist_to_star,
        "min_dist": min_dist_to_star,
        "total_reward": total_reward,
        "flow_reward": flow_reward_total,
        "dist_reward": dist_reward_total,
        "flow_weight": flow_weight,
        "action_set": action_set,
        "action_counts": dict(action_counts),
        "q_vals_s0": q_vals_s0.tolist()
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate QR-DQN model on ds8")
    parser.add_argument("--model", type=str, default="models/qrdqn_impala_ds8_v2_best.zip")
    parser.add_argument("--dist-mode", type=str, default="progress")
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--flow-weight", type=float, default=0.2,
                        help="Flow weight of the instrument (0.2 = the v1-v3 training weight)")
    parser.add_argument("--route", type=str, default=None, help="Route JSON for --dist-mode route")
    args = parser.parse_args()

    m_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT_DIR, args.model)
    r_path = None if args.route is None else (
        args.route if os.path.isabs(args.route) else os.path.join(ROOT_DIR, args.route))
    evaluate_model(m_path, dist_mode=args.dist_mode, max_steps=args.max_steps,
                   flow_weight=args.flow_weight, route_path=r_path)
