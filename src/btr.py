"""A BTR-style value-based agent ("Beyond The Rainbow", Clark et al. 2024), written for this project.

Components: IMPALA-CNN encoder, IQN quantile head with dueling streams and noisy layers, n-step
returns, Munchausen reward, prioritised replay (proportional, importance-weighted), soft target
update, AdamW. The replay buffer is kept per environment so that frame stacks and n-step windows are
rebuilt from single 84x84 frames (about 7 KB a step instead of 28 KB per observation).

This is an independent implementation from the paper's description, not the authors' code.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.impala_cnn import ImpalaCNN


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy layer (Fortunato et al. 2018)."""

    def __init__(self, n_in, n_out, std_init=0.5):
        super().__init__()
        self.n_in, self.n_out, self.std_init = n_in, n_out, std_init
        self.w_mu = nn.Parameter(torch.empty(n_out, n_in))
        self.w_sigma = nn.Parameter(torch.empty(n_out, n_in))
        self.b_mu = nn.Parameter(torch.empty(n_out))
        self.b_sigma = nn.Parameter(torch.empty(n_out))
        self.register_buffer("w_eps", torch.zeros(n_out, n_in))
        self.register_buffer("b_eps", torch.zeros(n_out))
        self.noise = True
        bound = 1 / math.sqrt(n_in)
        nn.init.uniform_(self.w_mu, -bound, bound)
        nn.init.uniform_(self.b_mu, -bound, bound)
        nn.init.constant_(self.w_sigma, std_init / math.sqrt(n_in))
        nn.init.constant_(self.b_sigma, std_init / math.sqrt(n_out))
        self.resample()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def resample(self):
        e_in = self._f(torch.randn(self.n_in, device=self.w_mu.device))
        e_out = self._f(torch.randn(self.n_out, device=self.w_mu.device))
        self.w_eps.copy_(e_out.outer(e_in))
        self.b_eps.copy_(e_out)

    def forward(self, x):
        if self.noise and self.training:
            return F.linear(x, self.w_mu + self.w_sigma * self.w_eps, self.b_mu + self.b_sigma * self.b_eps)
        return F.linear(x, self.w_mu, self.b_mu)


class IQNNet(nn.Module):
    """Frames (B, 4, 84, 84) uint8 -> quantile values (B, N, A) at the given taus (B, N)."""

    def __init__(self, n_actions, frames=4, feat=512, hidden=512, n_cos=64, noisy_std=0.5):
        super().__init__()
        self.n_actions, self.n_cos = n_actions, n_cos
        self.encoder = ImpalaCNN(c=frames, h=84, w=84, features_dim=feat)
        self.cos_embed = nn.Sequential(nn.Linear(n_cos, feat), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(feat, hidden, noisy_std), nn.ReLU(), NoisyLinear(hidden, 1, noisy_std))
        self.adv = nn.Sequential(NoisyLinear(feat, hidden, noisy_std), nn.ReLU(),
                                 NoisyLinear(hidden, n_actions, noisy_std))
        self.register_buffer("_i", torch.arange(1, n_cos + 1).float().view(1, 1, -1))

    def noisy_layers(self):
        return [m for m in self.modules() if isinstance(m, NoisyLinear)]

    def resample_noise(self):
        for m in self.noisy_layers():
            m.resample()

    def forward(self, x, taus):
        feat, _ = self.encoder(x.float() / 255.0)                       # (B, F)
        cos = torch.cos(math.pi * self._i * taus.unsqueeze(-1))         # (B, N, n_cos)
        h = feat.unsqueeze(1) * self.cos_embed(cos)                     # (B, N, F)
        v, a = self.value(h), self.adv(h)                               # (B, N, 1), (B, N, A)
        return v + a - a.mean(-1, keepdim=True)


class PrioritizedStepBuffer:
    """Per-environment ring buffers of single frames with proportional prioritised sampling.

    Step t of env e stores the frame the agent saw, its action, the reward and done that followed, and
    the absolute index of the episode's first step (to blank the frames before it in a stack). The next
    state of step t is step t+1 of the same env, so a transition is only sampled once its n following
    steps exist; a `done` inside the window ends the return without bootstrapping.
    """

    def __init__(self, n_envs, capacity_per_env, n_step=3, gamma=0.99, alpha=0.2, stack=4):
        self.E, self.C, self.n, self.gamma, self.alpha, self.stack = n_envs, capacity_per_env, n_step, gamma, alpha, stack
        self.frames = np.zeros((n_envs, capacity_per_env, 84, 84), np.uint8)
        self.actions = np.zeros((n_envs, capacity_per_env), np.int64)
        self.rewards = np.zeros((n_envs, capacity_per_env), np.float32)
        self.dones = np.zeros((n_envs, capacity_per_env), bool)
        self.ep_start = np.zeros((n_envs, capacity_per_env), np.int64)
        self.tabs = np.full((n_envs, capacity_per_env), -1, np.int64)
        self.prio = np.zeros((n_envs, capacity_per_env), np.float64)
        self.pos = np.zeros(n_envs, np.int64)          # steps written per env
        self._start = np.zeros(n_envs, np.int64)       # abs index of each env's current episode start
        self.max_prio = 1.0

    def __len__(self):
        return int(np.minimum(self.pos, self.C).sum())

    def add(self, frames, actions, rewards, dones):
        """One step of every env: frames (E, 84, 84) the agents saw, then what followed."""
        for e in range(self.E):
            t = int(self.pos[e])
            s = t % self.C
            self.frames[e, s], self.actions[e, s] = frames[e], actions[e]
            self.rewards[e, s], self.dones[e, s] = rewards[e], dones[e]
            self.ep_start[e, s], self.tabs[e, s] = self._start[e], t
            self.prio[e, s] = self.max_prio ** self.alpha
            self.pos[e] = t + 1
            if dones[e]:
                self._start[e] = t + 1

    def _valid(self):
        pos = self.pos[:, None]
        return (self.tabs >= 0) & (self.tabs <= pos - 1 - self.n) & (self.tabs >= pos - self.C + self.stack)

    def _stack_at(self, e, t):
        """(stack, 84, 84) uint8 for env e at absolute step t (frames before the episode start are 0)."""
        out = np.zeros((self.stack, 84, 84), np.uint8)
        start = self.ep_start[e, t % self.C]
        for j in range(self.stack):
            tj = t - (self.stack - 1) + j
            if tj >= start:
                out[j] = self.frames[e, tj % self.C]
        return out

    def sample(self, batch, beta):
        valid = self._valid()
        p = np.where(valid, self.prio, 0.0).ravel()
        total = p.sum()
        flat = np.random.choice(p.size, size=batch, p=p / total)
        e, s = flat // self.C, flat % self.C
        obs = np.zeros((batch, self.stack, 84, 84), np.uint8)
        nxt = np.zeros_like(obs)
        act, ret = np.zeros(batch, np.int64), np.zeros(batch, np.float32)
        boot = np.zeros(batch, np.float32)
        for i in range(batch):
            ei, t = int(e[i]), int(self.tabs[e[i], s[i]])
            obs[i] = self._stack_at(ei, t)
            act[i] = self.actions[ei, t % self.C]
            r, g, nonterm, k = 0.0, 1.0, 1.0, self.n
            for j in range(self.n):
                sj = (t + j) % self.C
                r += g * self.rewards[ei, sj]
                g *= self.gamma
                if self.dones[ei, sj]:
                    nonterm, k = 0.0, j + 1
                    break
            ret[i], boot[i] = r, nonterm * g
            nxt[i] = self._stack_at(ei, t + k) if nonterm else 0
        probs = p[flat] / total
        w = (len(self) * probs) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        return {"obs": obs, "act": act, "ret": ret, "boot": boot, "next": nxt, "w": w, "e": e, "s": s}

    def update(self, e, s, td):
        pr = (np.abs(td) + 1e-6)
        self.max_prio = max(self.max_prio, float(pr.max()))
        self.prio[e, s] = pr ** self.alpha


class BTRAgent:
    def __init__(self, n_actions, device="cuda", lr=1e-4, gamma=0.99, n_step=3, n_tau=8, n_tau_target=8,
                 munchausen=(0.9, 0.03, -1.0), tau_soft=0.005, weight_decay=0.01, kappa=1.0):
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_actions, self.gamma, self.n_step, self.kappa = n_actions, gamma, n_step, kappa
        self.n_tau, self.n_tau_t = n_tau, n_tau_target
        self.m_alpha, self.m_tau, self.m_lo = munchausen
        self.tau_soft = tau_soft
        self.net = IQNNet(n_actions).to(self.dev)
        self.target = IQNNet(n_actions).to(self.dev)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay, eps=1.5e-4)
        self.updates = 0

    @torch.no_grad()
    def act(self, stacks, noisy=True, fixed_taus=False):
        """Greedy action on the mean of the quantiles; the noisy layers explore unless noisy=False.

        fixed_taus=True replaces the 16 random quantile levels by their midpoints, so with noisy=False the
        action is a deterministic function of the observation."""
        self.net.train(noisy)
        if noisy:
            self.net.resample_noise()
        x = torch.as_tensor(stacks, device=self.dev)
        if fixed_taus:
            taus = ((torch.arange(16, device=self.dev, dtype=torch.float32) + 0.5) / 16).expand(x.shape[0], 16)
        else:
            taus = torch.rand(x.shape[0], 16, device=self.dev)
        return self.net(x, taus).mean(1).argmax(-1).cpu().numpy()

    def learn(self, b):
        dev = self.dev
        obs = torch.as_tensor(b["obs"], device=dev)
        nxt = torch.as_tensor(b["next"], device=dev)
        act = torch.as_tensor(b["act"], device=dev)
        ret = torch.as_tensor(b["ret"], device=dev)
        boot = torch.as_tensor(b["boot"], device=dev)
        w = torch.as_tensor(b["w"], device=dev)
        B = obs.shape[0]
        self.net.train()
        self.net.resample_noise()
        self.target.train()
        self.target.resample_noise()
        with torch.no_grad():
            tau_t = torch.rand(B, self.n_tau_t, device=dev)
            q_next = self.target(nxt, tau_t)                                  # (B, N', A)
            pi_next = F.softmax(q_next.mean(1) / self.m_tau, -1)              # (B, A)
            logpi_next = F.log_softmax(q_next.mean(1) / self.m_tau, -1)
            soft_next = (pi_next.unsqueeze(1) * (q_next - self.m_tau * logpi_next.unsqueeze(1))).sum(-1)  # (B, N')
            # Munchausen bonus of the first action, from the target policy at s
            q_now = self.target(obs, torch.rand(B, self.n_tau_t, device=dev)).mean(1)
            logpi = F.log_softmax(q_now / self.m_tau, -1).gather(1, act[:, None]).squeeze(1)
            m = torch.clamp(self.m_alpha * self.m_tau * logpi, min=self.m_lo, max=0.0)
            target = (ret + m).unsqueeze(1) + boot.unsqueeze(1) * soft_next   # (B, N')
        taus = torch.rand(B, self.n_tau, device=dev)
        z = self.net(obs, taus).gather(2, act[:, None, None].expand(B, self.n_tau, 1)).squeeze(-1)   # (B, N)
        td = target.unsqueeze(1) - z.unsqueeze(2)                              # (B, N, N')
        huber = torch.where(td.abs() <= self.kappa, 0.5 * td ** 2, self.kappa * (td.abs() - 0.5 * self.kappa))
        qloss = (taus.unsqueeze(2) - (td.detach() < 0).float()).abs() * huber / self.kappa
        per_sample = qloss.mean(2).sum(1)                                      # (B,)
        loss = (w * per_sample).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for pt, p in zip(self.target.parameters(), self.net.parameters()):
                pt.mul_(1 - self.tau_soft).add_(self.tau_soft * p)
        self.updates += 1
        return float(loss), per_sample.detach().cpu().numpy()

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "n_actions": self.n_actions}, path)

    def load(self, path, sources=None, gap=0.05):
        """Load weights. A checkpoint with fewer actions than this agent is extended: `sources[i]` is the old
        action whose output row the i-th new action starts as a copy of, with its bias lowered by `gap` so that
        a greedy (noise-free) policy keeps choosing exactly what it chose before (the dueling mean removes the
        same amount from every old action)."""
        d = torch.load(path, map_location=self.dev)
        net = d["net"]
        old_n = int(net["adv.2.w_mu"].shape[0])
        if old_n != self.n_actions:
            if old_n > self.n_actions or sources is None or len(sources) != self.n_actions - old_n:
                raise ValueError(f"checkpoint has {old_n} actions, agent {self.n_actions}; pass one source per new action")
            net = dict(net)
            for k in ("adv.2.w_mu", "adv.2.w_sigma", "adv.2.b_mu", "adv.2.b_sigma", "adv.2.w_eps", "adv.2.b_eps"):
                if k in net:
                    net[k] = torch.cat([net[k], net[k][list(sources)].clone()], dim=0)
            net["adv.2.b_mu"][old_n:] -= gap
        self.net.load_state_dict(net)
        self.target.load_state_dict(net)


def checkpoint_actions(path):
    """Number of actions of a saved agent (its advantage head)."""
    d = torch.load(path, map_location="cpu")
    return int(d["net"]["adv.2.w_mu"].shape[0])
