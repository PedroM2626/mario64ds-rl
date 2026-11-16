"""Data for distilling the two-network TTC agent into one network.

Rolls out the composite agent (the main policy until its route progress reaches --handoff-wp waypoints, then the transfer
specialist) from the normal ds8 start and records, for every decision, the newest frame the agent saw, which network was
the teacher and that teacher's quantile values at 16 fixed quantile levels for all 13 actions. The frame stack is
continuous (it is NOT restarted at the hand-over): the single student network has to work from the stack it sees.

Both teachers are used as 13-action agents; the main policy (10 actions) is extended with the rows of ref13's three new
actions as in training, so its greedy choice is unchanged.

Usage (use scripts/lowprio.py when the CPU is shared):
    python scripts/lowprio.py scripts/collect_distill_data.py --main models/btr_ttc_s9/step_750000.pt \
        --spec models/btr_ttc_spec5/final.pt --episodes 8 --seed 1 --main-fixed-taus --out data/distill/part_01.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402

SOURCES = [1, 8, 2]   # ref13's new actions start as copies of these ref10 actions (src/env.ACTION_SET_EXTENDS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--handoff-wp", type=int, default=340)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--main-fixed-taus", action="store_true", help="the main policy acts with fixed quantile levels (its path to the hand is then always the same)")
    ap.add_argument("--student", default=None,
                    help="DAgger mode: this single 13-action network acts, and every state it visits is labelled by the teacher of its phase")
    ap.add_argument("--student-fixed-taus", action="store_true")
    ap.add_argument("--max-steps", type=int, default=2700)
    ap.add_argument("--stall", type=int, default=300, help="stop an episode whose main policy gained no waypoint for this many decisions before the hand-over")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from scripts.train_btr import ref_env_kwargs
    from src.btr import BTRAgent
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    from src.route import Route

    torch.manual_seed(args.seed)
    route = Route.load(os.path.join(ROOT_DIR, "config/routes/ttc_human_v1.json"))
    main_agent = BTRAgent(13, device="cpu")
    main_agent.load(os.path.join(ROOT_DIR, args.main), sources=SOURCES)
    spec = BTRAgent(13, device="cpu")
    spec.load(os.path.join(ROOT_DIR, args.spec))
    student = None
    if args.student:
        student = BTRAgent(13, device="cpu")
        student.load(os.path.join(ROOT_DIR, args.student))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path("ds8"), max_steps=args.max_steps, track_id="ds8",
                       **ref_env_kwargs("ds8", action_set="ref13"))
    mid = ((torch.arange(16, dtype=torch.float32) + 0.5) / 16).view(1, 16)
    frames, qv, teacher, ep_id, t_id, acts = [], [], [], [], [], []
    summary = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        dq = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
        phase, best_wp, last_gain, end = 0, 0, 0, "horizon"
        for t in range(args.max_steps):
            agent = spec if phase else main_agent
            stack = np.stack(list(dq))[None]
            with torch.no_grad():
                agent.net.train(False)
                q = agent.net(torch.as_tensor(stack), mid)[0]                       # (16, 13)
            if student is not None:
                a = int(student.act(stack, noisy=False, fixed_taus=args.student_fixed_taus)[0])
            else:
                a = int(agent.act(stack, noisy=False, fixed_taus=bool(args.main_fixed_taus and not phase))[0])
            frames.append(dq[-1].copy()); qv.append(q.numpy().astype(np.float16)); teacher.append(phase)
            ep_id.append(len(summary)); t_id.append(t); acts.append(a)
            obs, _, term, _, info = env.step(a)
            dq.append(obs[:, :, 0])
            wp = route.passed(0, float(info.get("route_progress") or 0.0))
            if wp > best_wp:
                best_wp, last_gain = wp, t
            if not phase and best_wp >= args.handoff_wp:
                phase = 1
            if term:
                end = "star" if info.get("success") else "died"
                break
            if (student is not None or (not phase and best_wp < args.handoff_wp)) and t - last_gain >= args.stall:
                end = "stalled"
                break
        summary.append((end, t + 1, best_wp, phase))
        print(f"episode {ep}: {end}, {t + 1} decisions, best waypoint {best_wp}, specialist reached {bool(phase)}", flush=True)
        os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, args.out)), exist_ok=True)
        np.savez(os.path.join(ROOT_DIR, args.out), frames=np.stack(frames), q=np.stack(qv), teacher=np.array(teacher, np.int8),
                 ep=np.array(ep_id, np.int32) + 1000 * args.seed, t=np.array(t_id, np.int32), action=np.array(acts, np.int8))
    env.close()


if __name__ == "__main__":
    main()
