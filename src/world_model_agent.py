"""Actor-critic trained *inside* the latent world model (imagination).

Mirrors the reference ``dyna_ppo.py`` (Dyna-PPO) but, because our dynamics live
in a latent RSSM space, the policy is optimized on imagined latent trajectories
chained through the model prior. Gradients are produced with a REINFORCE-style
policy gradient (discrete actions) with a learned value baseline and GAE/lambda
returns, plus an entropy regularizer. The world-model parameters are kept frozen
during the imagination phase (detached latents), so this is a clean "train the
agent inside a fixed learned world" loop — extremely cheap in real samples.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.world_model import Latent, WorldModel, _gauss_sample


class ActorCritic(nn.Module):
    def __init__(self, deter_dim: int, stoch_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        img_dim = deter_dim + stoch_dim
        self.actor = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def dist(self, img: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(img))

    def value(self, img: torch.Tensor) -> torch.Tensor:
        return self.critic(img).squeeze(-1)

    @torch.no_grad()
    def act(self, img: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        logits = self.actor(img)
        return (logits.argmax(-1) if deterministic else Categorical(logits=logits).sample())


@torch.no_grad()
def imagine(model: WorldModel, ac: ActorCritic, start: Latent, horizon: int,
            gamma: float, lam: float, pessimism_beta: float = 0.0,
            uncertainty_trunc: float = 1e9, terminal_cost: float = 0.0,
            death_thresh: float = 0.5) -> Dict[str, torch.Tensor]:
    """Roll the agent's policy forward inside the frozen model prior.

    ``start`` is a detached latent (batch of belief states seeded from real
    data). Returns stacked per-step images, actions, log-probs, values,
    lambda-returns and discount masks.

    Reference-style safe-MBRL: ``pessimism_beta`` subtracts the model's epistemic
    proxy (prior std) from the imagined reward and ``uncertainty_trunc`` ends the
    imagined rollout once uncertainty exceeds a threshold — this prevents the
    actor from exploiting regions where the learned model is unreliable.

    ``terminal_cost`` applies a one-off penalty the first time the model predicts
    death inside the horizon and stops accruing reward afterwards. This is
    essential on these slides: the raw optical-flow reward keeps *paying* while
    Mario falls off a cliff (falling is fast downward motion), so without an
    in-horizon death term the actor is rewarded for suicide.
    """
    B = start.deter.shape[0]
    device = start.deter.device
    latent = start
    alive = torch.ones(B, device=device)
    imgs, acts, logps, vals, rews, conts = [], [], [], [], [], []
    for _ in range(horizon):
        img = model.img(latent)
        dist = ac.dist(img)
        action = dist.sample()
        imgs.append(img)
        acts.append(action)
        logps.append(dist.log_prob(action))
        vals.append(ac.value(img))
        uncertainty = latent.std.mean(-1)  # epistemic proxy
        cont = model.continue_prob(latent) * (uncertainty < uncertainty_trunc).float()
        r = alive * (model.reward(latent) - pessimism_beta * uncertainty)
        dying = ((cont < death_thresh) & (alive > 0.5)).float()
        r = r - terminal_cost * dying
        alive = alive * (1.0 - dying)
        conts.append(cont * alive)  # cut the return once dead
        rews.append(r)
        latent = model.imagine_step(latent, action)

    img = model.img(latent)
    last_value = ac.value(img)

    imgs = torch.stack(imgs, 1)          # (B,T,D)
    acts = torch.stack(acts, 1)          # (B,T)
    logps = torch.stack(logps, 1)        # (B,T)
    vals = torch.stack(vals, 1)          # (B,T)
    rews = torch.stack(rews, 1)          # (B,T)
    conts = torch.stack(conts, 1)        # (B,T)

    # lambda-returns with bootstrapped terminal value.
    returns = torch.zeros_like(rews)
    last_gae = 0.0
    next_value = last_value
    next_cont = 1.0
    for t in reversed(range(horizon)):
        delta = rews[:, t] + gamma * next_value * next_cont - vals[:, t]
        last_gae = delta + gamma * lam * next_cont * last_gae
        returns[:, t] = vals[:, t] + last_gae
        next_value = vals[:, t]
        next_cont = conts[:, t]
    advantages = returns - vals
    return dict(imgs=imgs, acts=acts, logps=logps, vals=vals,
                returns=returns, advantages=advantages)


def actor_critic_update(model: WorldModel, ac: ActorCritic, optimizer: torch.optim.Optimizer,
                        start: Latent, horizon: int = 15, gamma: float = 0.995,
                        lam: float = 0.95, ent_coef: float = 3e-3,
                        max_grad_norm: float = 0.5, pessimism_beta: float = 0.0,
                        uncertainty_trunc: float = 1e9, terminal_cost: float = 0.0,
                        death_thresh: float = 0.5) -> Dict[str, float]:
    """One imagination PPO-free (REINFORCE + baseline) update of actor & critic."""
    data = imagine(model, ac, start, horizon, gamma, lam, pessimism_beta,
                   uncertainty_trunc, terminal_cost, death_thresh)
    imgs = data["imgs"].reshape(-1, data["imgs"].shape[-1])
    acts = data["acts"].reshape(-1)
    returns = data["returns"].reshape(-1)
    advantages = data["advantages"].reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dist = ac.dist(imgs)
    logp = dist.log_prob(acts)
    entropy = dist.entropy().mean()

    policy_loss = -(logp * advantages.detach()).mean()
    value_loss = F.mse_loss(ac.value(imgs), returns.detach())
    loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
    optimizer.step()
    return dict(policy_loss=float(policy_loss), value_loss=float(value_loss),
                entropy=float(entropy), mean_return=float(returns.mean()))


def actor_bc_update(model: WorldModel, ac: ActorCritic, optimizer: torch.optim.Optimizer,
                    obs_seq: torch.Tensor, action_in: torch.Tensor, label: torch.Tensor,
                    max_grad_norm: float = 0.5) -> Dict[str, float]:
    """Amortized policy distillation (behaviour cloning) in the learned latent space.

    The frozen world model turns each real observation sequence into belief
    vectors; the actor is trained to reproduce the expert action taken at each
    observation (reference: "amortized policy distillation" / DAgger). Only the
    actor is updated; the model stays frozen. This trains a *deployable* policy
    extremely fast (no environment) and anchors it against imagination collapse.
    """
    with torch.no_grad():
        imgs = model.posterior_imgs(obs_seq, action_in)  # (B,T,D) model frozen
    B, T, D = imgs.shape
    flat = imgs.reshape(B * T, D)
    lab = label.reshape(B * T)
    dist = ac.dist(flat)
    logp = dist.log_prob(lab)
    loss = -logp.mean()
    acc = (dist.probs.argmax(-1) == lab).float().mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
    optimizer.step()
    return dict(bc_loss=float(loss), bc_acc=float(acc))
