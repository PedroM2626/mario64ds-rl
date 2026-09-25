"""Diagnose a fitted world model + controller *before* spending emulator time on it.

Two questions this answers, both of which cost hours to discover by end-to-end
runs alone:

1. ``--check model``  Is the learned model *discriminative*?
   Rolls each constant action sequence through the prior and reports predicted
   survival and mean reward per step. Pathologies this catches:
     * survive=1.0 for every action  -> the continue head never saw a death
       (terminal deaths sit on the FINAL frame; a window sampler capped at
       T-seq_len-1 silently excludes all of them), so planners walk off cliffs;
     * survive=0.0 for every action  -> over-corrected / death-saturated data,
       imagination says nothing survives so there is no usable gradient;
     * mean reward ~0 for every action -> the -100 terminal spike swamped the
       reward MSE; the dense flow signal is gone.

2. ``--check actor``  Has an imagination-trained policy collapsed?
   Reports the actor's action entropy and argmax histogram over real belief
   states. Near-zero entropy on one action explains byte-identical evaluation
   numbers across training checkpoints.

    python scripts/diagnose_world_model.py --check model --bundle models/wm_native_wm.pt
    python scripts/diagnose_world_model.py --check actor  --bundle models/wm_dreamer_wm.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.wm_replay import EpisodeBuffer                     # noqa: E402
from src.world_model import WorldModel, Latent, _gauss_sample  # noqa: E402

NAMES = ["noop", "left", "right", "up(accel)", "down(brake)", "jump"]


def _load(bundle, device):
    ck = torch.load(bundle, map_location=device)
    cfg = ck["world_model_cfg"]
    model = WorldModel(**cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, cfg, ck


def _alive_start_states(model, buf, device, seq_len=40, n=32):
    """Belief states that the DATA says are still alive (a dead start state makes
    every rollout trivially 'survive=0' and would be misread as model failure)."""
    obs, acts, rews, conts = buf.sample(64, seq_len, np.random.default_rng(0))
    with torch.no_grad():
        out = model.observe_sequence(obs.to(device), acts.to(device))
    T = out["post_mean"].shape[1]
    keep = (conts[:, -1] > 0.5).nonzero(as_tuple=True)[0]
    if len(keep) < 4:
        keep = torch.arange(out["post_mean"].shape[0])
    g = keep[:n]
    t = T - 1
    m, s = out["post_mean"][g, t], out["post_std"][g, t]
    return Latent(out["post_deter"][g, t], m, s, _gauss_sample(m, s)), len(g)


def check_model(model, buf, device, horizon):
    start, n = _alive_start_states(model, buf, device)
    B = start.deter.shape[0]
    print(f"predicted dynamics over H={horizon} constant-action rollouts")
    print(f"  from {B} belief states that are ALIVE in the data\n")
    print(f"{'action':<12}{'value(no term)':>16}{'value(term)':>13}"
          f"{'survive frac':>14}{'mean rew/step':>15}")
    with torch.no_grad():
        for a in range(6):
            seq = torch.full((horizon, B), a, dtype=torch.long, device=device)
            v0 = model.rollout_value(start, seq, gamma=0.98, terminal_cost=0.0)
            v1 = model.rollout_value(start, seq, gamma=0.98,
                                     terminal_cost=150.0, death_thresh=0.5)
            lat, surv, rw = start, torch.ones(B, device=device), torch.zeros(B, device=device)
            for _ in range(horizon):
                rw += surv * model.reward(lat)
                surv = surv * (model.continue_prob(lat) >= 0.5).float()
                lat = model.imagine_step(lat, seq[0], deterministic=True)
            print(f"{NAMES[a]:<12}{float(v0.mean()):>16.3f}{float(v1.mean()):>13.3f}"
                  f"{float(surv.mean()):>14.3f}{float((rw/horizon).mean()):>15.4f}")

    print("\nverdict:")
    print("  survival is discriminative across actions:",
          "YES" if len({round(float(model.rollout_value(
              start, torch.full((horizon, B), a, dtype=torch.long, device=device),
              gamma=0.98).mean()), 2) for a in range(6)}) > 2 else "NO (flat)")


def check_actor(model, cfg, ck, buf, device):
    from src.world_model_agent import ActorCritic
    ac = ActorCritic(cfg["deter_dim"], cfg["stoch_dim"], cfg["num_actions"],
                     hidden=cfg.get("hidden", 128)).to(device)
    ac.load_state_dict(ck["actor"])
    ac.eval()
    obs, acts, _, _ = buf.sample(64, 40, np.random.default_rng(0))
    with torch.no_grad():
        out = model.observe_sequence(obs.to(device), acts.to(device))
        T = out["post_mean"].shape[1]
        ents, argmax, mean_p = [], {}, None
        for t in range(T):
            img = torch.cat([out["post_deter"][:, t], out["post_mean"][:, t]], -1)
            d = ac.dist(img)
            ents.append(float(d.entropy().mean()))
            for k, v in zip(*torch.unique(d.probs.argmax(-1), return_counts=True)):
                argmax[int(k)] = argmax.get(int(k), 0) + int(v)
            p = d.probs.mean(0)
            mean_p = p if mean_p is None else mean_p + p
        mean_p = mean_p / T
    print(f"actor entropy over states: {np.mean(ents):.4f}  (uniform = {np.log(6):.3f})")
    print("mean action prob [noop,left,right,up,down,jump]:",
          np.round(mean_p.cpu().numpy(), 4).tolist())
    print("argmax histogram:", dict(sorted(argmax.items())))
    dom = float(max(mean_p.cpu().numpy()))
    print(f"dominant action share: {dom:.3f} ->",
          "COLLAPSED (one action everywhere)" if dom > 0.9 else "diverse")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="models/wm_native_wm.pt")
    p.add_argument("--buffer", default="data/wm_buffer3.pkl")
    p.add_argument("--check", choices=["model", "actor", "both"], default="both")
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--device", default="cpu", help="cpu keeps this clear of GPU training jobs")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model, cfg, ck = _load(args.bundle, device)
    buf = EpisodeBuffer()
    buf.load(args.buffer if os.path.isabs(args.buffer) else os.path.join(ROOT, args.buffer))
    print(f"bundle: {args.bundle}\nbuffer: {args.buffer} ({buf.total_steps} steps)\n")
    if args.check in ("model", "both"):
        check_model(model, buf, device, args.horizon)
    if args.check in ("actor", "both"):
        print()
        check_actor(model, cfg, ck, buf, device)


if __name__ == "__main__":
    main()
