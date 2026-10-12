"""Why one checkpoint can score 1350/1350 on one machine and die at 218 on another.

The published benchmark table says curriculum_flow25_r3_best.zip completes all
three slide tracks, and videos/phase_ds3.mp4 (1351 frames, 2026-09-21) shows a
full-horizon run. On this machine today, src/eval.py --deterministic reproduces
ds1 and ds2 to the cent but ds3 ends at decision 218 with reward -3.06. The env
has no randomness and the policy runs with deterministic=True, so what is left is
arithmetic: the same weights convolve through cuDNN on the GPU and through a
different kernel on the CPU, and a near-tie broken the other way one decision
early puts the slide on a different line.

This probe measures that. Same savestate, same checkpoint, same VecFrameStack
convention the recorder and src/eval.py use -- device is the only variable. Each
(track, device) arm runs in its own process, because two DeSmuME cores in one
process take an access violation, and the job list is executed by
rust/sm64_sup.exe.

    python tools/probe_device_divergence.py                 # both arms, all tracks
    python tools/probe_device_divergence.py --device cpu --track ds3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULT_RE = re.compile(r"^RESULT\t(.*)$", re.MULTILINE)
SUP = os.path.join(ROOT, "rust", "sm64_sup.exe")


def rollout(device, track, rom, state, max_steps):
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from src.env import Mario64DSEnv

    model = PPO.load(os.path.join(ROOT, "models", "curriculum_flow25_r3_best.zip"),
                     device=device)
    pol = model.policy
    env = Mario64DSEnv(rom_path=rom, state_path=state)
    obs, _ = env.reset()
    first = obs[:, :, 0]
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)
    acts, margins, rewards = [], [], []
    done = trunc = False
    while not (done or trunc) and len(acts) < max_steps:
        stack = np.stack(list(dq), axis=0)[np.newaxis, :]
        a, _ = model.predict(stack, deterministic=True)
        with torch.no_grad():
            t = pol.obs_to_tensor(stack)[0]
            p = pol.get_distribution(t).distribution.probs[0].detach().float().cpu().numpy()
        srt = np.sort(p)[::-1]
        margins.append(round(float(srt[0] - srt[1]), 6))
        obs, r, done, trunc, _ = env.step(int(a[0]))
        dq.append(obs[:, :, 0])
        acts.append(int(a[0]))
        rewards.append(float(r))
    env.close()
    return {"device": str(next(pol.parameters()).device), "track": track,
            "steps": len(acts), "reward": round(float(np.sum(rewards)), 6),
            "actions_sha1": hashlib.sha1(bytes(acts)).hexdigest()[:12],
            "actions": acts, "margins": margins}


def run_arms(tracks, devices, max_steps):
    jobfile = os.path.join(ROOT, "out", "divergence_jobs.tsv")
    os.makedirs(os.path.dirname(jobfile), exist_ok=True)
    with open(jobfile, "w", encoding="utf-8") as f:
        for track in tracks:
            for dev in devices:
                argv = [sys.executable, os.path.abspath(__file__),
                        "--device", dev, "--track", track,
                        "--max-steps", str(max_steps)]
                f.write("\t".join([f"{track}-{dev}", "900"] + argv) + "\n")
    p = subprocess.run([SUP, jobfile, os.path.join(ROOT, "out", "divergence_log.json")],
                       cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    results = {}
    for m in RESULT_RE.findall(out):
        d = json.loads(m)
        results[(d["track"], d["device"])] = d
    if not results:
        print(out[-3000:], file=sys.stderr)
        raise SystemExit("no RESULT lines from the supervisor")
    return results


def arm_path(track, device):
    return os.path.join(ROOT, "out", f"divergence_{track}_{device}.json")


def build_report(tracks, max_steps, out):
    report = {"max_steps": max_steps, "tracks": {}}
    for track in tracks:
        arms = {}
        for dev in ("cuda", "cpu"):
            p = arm_path(track, dev)
            # torch reports the CUDA device as "cuda:0"; the key is the device kind
            if os.path.exists(p):
                d = json.load(open(p))
                d["device"] = str(d["device"]).split(":")[0]
                arms[dev] = d
        if len(arms) != 2:
            report["tracks"][track] = {"error": f"missing arm: {sorted(arms)}"}
            continue
        ga, ca = arms["cuda"], arms["cpu"]
        div = next((i for i, (x, y) in enumerate(zip(ga["actions"], ca["actions"]))
                    if x != y), None)
        row = {"cuda": {"steps": ga["steps"], "reward": ga["reward"],
                        "actions_sha1": ga["actions_sha1"]},
               "cpu": {"steps": ca["steps"], "reward": ca["reward"],
                       "actions_sha1": ca["actions_sha1"]},
               "same_action_sequence": ga["actions"] == ca["actions"],
               "first_divergent_decision": div,
               "margin_at_divergence": (ga["margins"][div] if div is not None else None),
               "min_margin_over_run": round(min(ga["margins"] + ca["margins"]), 6)}
        report["tracks"][track] = row
        print(f"[probe] {track}: cuda {row['cuda']['steps']} steps "
              f"R={row['cuda']['reward']:+.2f} | cpu {row['cpu']['steps']} steps "
              f"R={row['cpu']['reward']:+.2f} | first divergence at decision "
              f"{row['first_divergent_decision']} (top-2 margin "
              f"{row['margin_at_divergence']}, min over run "
              f"{row['min_margin_over_run']:.4f})", flush=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] wrote {out}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="both", choices=["both", "cuda", "cpu"])
    ap.add_argument("--track", default=None, help="single-track worker mode")
    ap.add_argument("--tracks", default="ds1,ds2,ds3")
    ap.add_argument("--max-steps", type=int, default=1350)
    ap.add_argument("--report-only", action="store_true",
                    help="build the report from the per-arm files already on disk")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "device_divergence.json"))
    args = ap.parse_args()

    rom = os.path.join(ROOT, "data", "Super Mario 64 DS (USA) (Rev 1).nds")
    tracks = [t.strip() for t in args.tracks.split(",")]

    if args.report_only:
        return build_report(tracks, args.max_steps, args.out)

    if args.device != "both":
        # worker mode: one (track, device) arm in this process, then exit
        state = os.path.join(ROOT, "data", f"Super Mario 64 DS (USA) (Rev 1)."
                                            f"{args.track}")
        r = rollout(args.device, args.track, rom, state, args.max_steps)
        with open(arm_path(args.track, args.device), "w") as f:
            json.dump(r, f)
        slim = {k: r[k] for k in ("device", "track", "steps", "reward", "actions_sha1")}
        print("RESULT\t" + json.dumps(slim), flush=True)
        return 0

    run_arms(tracks, ["cuda", "cpu"], args.max_steps)
    return build_report(tracks, args.max_steps, args.out)


if __name__ == "__main__":
    sys.exit(main())
