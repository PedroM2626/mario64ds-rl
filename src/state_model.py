"""State-space dynamics model for the SM64DS slide courses (the smw-pinn paradigm).

Learns f(s, a) -> s' directly on Mario's RAM physics struct (see
src/ram_state.py), plus heads that make a planner out of it:

    delta_s  residual dynamics (the state model proper)
    fall     P(kill-plane hit on the next step) -- a real in-game fall
             (slide courses respawn mid-track; falls cost time, not the episode)
    terminal P(env black-screen death) -- the true episode-ending event
    reward   the env's reward (flow + death penalty) predicted from state:
             with exact state this is ~perfectly learnable, unlike the
             pixel-only pipeline where every head was action-blind
    value    V(s) on Monte-Carlo discounted returns of the real reward

Trained per track (each course has its own geometry). The bundle stores
normalization constants (raw fixed-point words are large).

Usage:
    python -m src.train_state --data data/state_transitions.pkl --run-id state_wm
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.device import resolve_device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_ACTIONS = 6
GAMMA = 0.995
FALL_PENALTY = 30.0          # planner penalty per predicted in-game fall

# Informative physics words (indices into the raw 6-word struct):
#   1 speed, 3 lateral, 4 height, 5 vvel.
# Dropped: 'heading' (angle with wraparound -- unlearnable by plain MSE and a
# compounding-error source, measured 0.84 sigma drift after 5 steps) and
# 'const' (a near-constant word whose jitter the model amplifies, 2.87 sigma
# drift after 5 steps).
STATE_COLS = (1, 3, 4, 5)
STATE_DIM = len(STATE_COLS)


class StateDynamicsModel(nn.Module):
    """MLP: [s_norm (STATE_DIM), onehot(a)] -> (delta_s, fall, terminal, reward, value)."""

    def __init__(self, hidden=128, layers=3):
        super().__init__()
        in_dim = STATE_DIM + NUM_ACTIONS
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*mods)
        self.dyn = nn.Linear(d, STATE_DIM)
        self.fall = nn.Linear(d, 1)
        self.term = nn.Linear(d, 1)
        self.rew = nn.Linear(d, 1)
        self.val = nn.Linear(d, 1)

    def forward(self, s, a):
        x = torch.cat([s, F.one_hot(a, NUM_ACTIONS).float()], dim=-1)
        h = self.trunk(x)
        return (self.dyn(h), self.fall(h).squeeze(-1), self.term(h).squeeze(-1),
                self.rew(h).squeeze(-1), self.val(h).squeeze(-1))

    @torch.no_grad()
    def step(self, s, a):
        """One planning step: returns (s', fall_p, term_p, r_hat)."""
        ds, fl, tm, rw, _ = self.forward(s, a)
        return s + ds, torch.sigmoid(fl), torch.sigmoid(tm), rw

    @torch.no_grad()
    def rollout_score(self, s0, actions, gamma=GAMMA, fall_penalty=FALL_PENALTY,
                      term_penalty=100.0):
        """CEM candidate scoring: discounted reward + value bootstrap -
        penalties for predicted falls / terminal death inside the horizon.

        actions: (H, B) long tensor. Returns (B,) scores.
        """
        B = actions.shape[1]
        if s0.shape[0] != B:
            s0 = s0.expand(B, *s0.shape[1:])
        disc = torch.ones(B, device=actions.device)
        alive = torch.ones(B, device=actions.device)
        score = torch.zeros(B, device=actions.device)
        s = s0
        for t in range(actions.shape[0]):
            s, fall_p, term_p, r = self.step(s, actions[t])
            # expected penalties (probabilities; each candidate penalized once
            # per step where death is likely)
            score = score + alive * disc * (r - fall_penalty * fall_p
                                            - term_penalty * term_p)
            alive = alive * (1.0 - term_p.clamp(max=0.9))
            disc = disc * gamma
        _, _, _, _, v = self.forward(s, torch.zeros(B, dtype=torch.long,
                                                    device=actions.device))
        score = score + alive * disc * v
        return score


def compute_returns(rewards, dones, gamma=GAMMA):
    """MC discounted returns per step within an episode (respawn falls are NOT
    terminals -- the episode continues, so returns flow across them)."""
    T = len(rewards)
    g = np.zeros(T, dtype=np.float64)
    acc = 0.0
    for t in reversed(range(T)):
        acc = float(rewards[t]) + gamma * acc * (0.0 if dones[t] else 1.0)
        g[t] = acc
    return g


def build_dataset(episodes):
    """Per-episode transition arrays -> flat (s, a, s', fall, term, r, G),
    restricted to the informative state columns."""
    S, A, S2, FL, TM, R, G = [], [], [], [], [], [], []
    for ep in episodes:
        st, ac, rw, fl, dn = (ep["states"], ep["actions"], ep["rewards"],
                              ep["falls"], ep["dones"])
        st = st[:, STATE_COLS]
        T = len(ac)
        S.append(st[:T]); A.append(ac); S2.append(st[1:T + 1])
        FL.append(fl); TM.append(dn); R.append(rw)
        G.append(compute_returns(rw, dn))
    cat = lambda xs: np.concatenate(xs, 0)
    return cat(S), cat(A), cat(S2), cat(FL), cat(TM), cat(R), cat(G)


def chain_batches(episodes, K=8, batch=64, rng=None, max_stride=1):
    """Multi-step training anchors: (episode, t) with K real actions following.

    One-step fitting alone lets error compound (measured: 0.1 sigma one-step ->
    ~1 sigma after 10-20 steps); training on teacher-forced K-step chains of
    the model's OWN rollouts is the standard fix.
    """
    rng = rng or np.random.default_rng(0)
    anchors = []
    for ei, ep in enumerate(episodes):
        T = len(ep["actions"])
        for t in range(0, T - K, max_stride):
            anchors.append((ei, t))
    rng.shuffle(anchors)
    for i in range(0, len(anchors), batch):
        chunk = anchors[i:i + batch]
        yield chunk


def train_track(episodes, device, epochs=60, batch=512, lr=1e-3, chain_k=8,
                fall_oversample=0.4, verbose=True):
    # Normalization constants must come from REAL trajectories only: the
    # probe-branch episodes are mostly frozen/clamp states; including them
    # shifts mu/sd so that every real state reads as OOD to the controller.
    real_eps = [e for e in episodes
                if e.get("behavior") not in ("state_mpc_probe", "probe")]
    SR, _, _, _, _, _, _ = build_dataset(real_eps)
    S, A, S2, FL, TM, R, G = build_dataset(episodes)
    n = len(S)
    mu = SR.mean(0)
    sd = SR.std(0) + 1e-6
    rmu, rsd = float(R.mean()), float(R.std() + 1e-6)
    gmu, gsd = float(G.mean()), float(G.std() + 1e-6)

    def norm(x):
        return (x - mu) / sd
    Sn = norm(S)
    S2n = norm(S2)
    Rn = (R - rmu) / rsd
    Gn = (G - gmu) / gsd

    # pre-death windows: the steps leading INTO a kill-plane transition are the
    # whole point (a planner blind to them is lured off edges) -- oversample them.
    dead_steps = np.nonzero(FL > 0.5)[0]                # the step WHERE the fall lands
    pre_death = np.unique(np.clip(dead_steps[:, None] - np.arange(1, 13)[None, :],
                                  0, n - 1).ravel())
    # AND mark the pre-death *approach* as fall-labeled too: the learnable
    # fall signature is the steady -8/step height ramp of a fatal fall, which
    # shows up ~12 steps BEFORE the clamp state did. Without this the fall head
    # only fires once the fall is already unstoppable.
    FL = FL.copy()
    pre_ramp = np.unique(np.clip(dead_steps[:, None] - np.arange(1, 13)[None, :],
                                 0, n - 1).ravel())
    FL[pre_ramp] = 1.0

    model = StateDynamicsModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    idx_all = np.arange(n)
    hold = rng.permutation(n)[:max(200, n // 10)]
    # per-episode normalized state arrays for the chain loss
    ep_states_n = [norm(e["states"][:, STATE_COLS]) for e in episodes]
    t0 = time.time()
    for ep_i in range(epochs):
        # --- pass 1: one-step heads (dynamics + fall/term/reward/value) ---
        perm = rng.permutation(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            if fall_oversample > 0 and len(pre_death):
                # replace a slice of the batch with pre-death steps
                k_rep = max(1, int(batch * fall_oversample))
                j[-k_rep:] = pre_death[rng.integers(0, len(pre_death), k_rep)]
                j = np.clip(j, 0, n - 1)
            s = torch.from_numpy(Sn[j]).float().to(device)
            a = torch.from_numpy(A[j]).long().to(device)
            s2 = torch.from_numpy(S2n[j]).float().to(device)
            fl = torch.from_numpy(FL[j]).float().to(device)
            tm = torch.from_numpy(TM[j]).float().to(device)
            r = torch.from_numpy(Rn[j]).float().to(device)
            g = torch.from_numpy(Gn[j]).float().to(device)
            ds, flp, tmp, rp, vp = model(s, a)
            loss = (F.mse_loss(ds, s2 - s)
                    + 8.0 * F.binary_cross_entropy_with_logits(flp, fl)
                    + 8.0 * F.binary_cross_entropy_with_logits(tmp, tm)
                    + F.mse_loss(rp, r)
                    + 0.5 * F.mse_loss(vp, g))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1

        # --- pass 2: K-step chain loss (compounding-error fix) ---
        chains = list(chain_batches(episodes, K=chain_k, batch=64, rng=rng,
                                    max_stride=4))
        tot2 = 0.0
        nb2 = 0
        for chunk in chains[:: max(1, len(chains) // 40)][:40]:
            s_list, act_list, target_list = [], [], []
            for ei, t in chunk:
                s_list.append(ep_states_n[ei][t])
                act_list.append(episodes[ei]["actions"][t:t + chain_k])
                target_list.append(ep_states_n[ei][t + 1:t + 1 + chain_k])
            sb = torch.from_numpy(np.stack(s_list)).float().to(device)
            ab = torch.from_numpy(np.stack(act_list)).long().to(device)   # (B,K)
            tg = torch.from_numpy(np.stack(target_list)).float().to(device)
            s = sb
            loss2 = torch.zeros((), device=device)
            for k in range(chain_k):
                ds, _, _, _, _ = model(s, ab[:, k])
                s = s + ds
                w_k = 0.9 ** k
                loss2 = loss2 + w_k * F.mse_loss(s, tg[:, k])
            loss2 = loss2 / chain_k
            opt.zero_grad()
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot2 += float(loss2); nb2 += 1

        if verbose and (ep_i + 1) % 10 == 0:
            with torch.no_grad():
                s = torch.from_numpy(Sn[hold]).float().to(device)
                a = torch.from_numpy(A[hold]).long().to(device)
                s2 = torch.from_numpy(S2n[hold]).float().to(device)
                ds, flp, tmp, rp, vp = model(s, a)
                perr = float((ds - (s2 - s)).abs().mean())
                fl_prob = torch.sigmoid(flp)
                fl_mask = torch.from_numpy(FL[hold] > 0.5).to(device)
                if fl_mask.any():
                    fl_recall = float((fl_prob[fl_mask] > 0.5).float().mean())
                else:
                    fl_recall = float("nan")
                g_corr = float(np.corrcoef(vp.cpu().numpy(), Gn[hold])[0, 1])
            print(f"    epoch {ep_i+1}/{epochs} loss={tot/nb:.4f} "
                  f"|s'|err={perr:.3f} fall_recall={fl_recall:.2f} "
                  f"v_corr={g_corr:.3f} chain={tot2/max(1,nb2):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    return model, dict(mu=mu.tolist(), sd=sd.tolist(), rmu=rmu, rsd=float(rsd),
                      gmu=gmu, gsd=float(gsd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/state_transitions.pkl")
    ap.add_argument("--run-id", default="state_wm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="auto",
                    help="torch device: auto (cuda if available), cpu or cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    with open(os.path.join(ROOT, args.data), "rb") as f:
        data = pickle.load(f)

    out = {}
    for tr, eps in data.items():
        print(f"[state] training {tr}: {len(eps)} episodes, "
              f"{sum(len(e['actions']) for e in eps)} transitions", flush=True)
        model, norm = train_track(eps, device, epochs=args.epochs)
        out[tr] = dict(model=model.state_dict(), norm=norm)
    path = os.path.join(ROOT, "models", f"{args.run_id}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    print(f"[state] saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
