"""Latent World Model for Super Mario 64 DS (Dreamer / PlaNet-style RSSM).

This module adapts the model-based RL recipe of the reference repository
``smw-pinn`` (see ``_ref_smw_pinn/src/environment/pinn_sim_env.py`` and
``_ref_smw_pinn/src/training/dyna_ppo.py``) to the *pixel-only* observation
space of ``src/env.py``.

The reference learns a low-dimensional state dynamics model ``f(s, a) -> s'``
from console RAM, wraps it in a GPU-vectorized simulation environment, and
trains an actor-critic **entirely inside the learned model** ("Dyna-PPO"),
achieving ultra-fast policy learning because no emulator is stepped during
training. Here the game state is only observable as pixels, so we learn the
dynamics in a compact *latent* space via a Recurrent State-Space Model (RSSM):

    deterministic  h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    prior          p(z_t | h_t)               (used for imagination rollouts)
    posterior      q(z_t | h_t, o_t)          (used when a real frame o_t exists)
    reward         r_t ~ R(h_t, z_t)
    continue       c_t ~ C(h_t, z_t)

The policy and value networks consume the latent state ``(h_t, z_t)`` and are
trained on imagined trajectories produced by chaining the *prior* — i.e. the
agent is trained "inside its own mind", which is what makes learning fast in
terms of real (emulator) samples. A single real rollout (~tens of thousands of
emulator steps) is reused for many GPU imagination updates.

Key design choices for robustness on this task:
* Gaussian diagonal latents (no discrete EMAs to tune).
* Reconstruction-free model loss (KL + reward + continue). We never decode
  pixels, so the model only needs to be predictive for control, matching the
  reference's state-based world-model philosophy.
* Free-nats KL and gradient clipping for stability.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Latent helpers
# --------------------------------------------------------------------------- #
class Latent(NamedTuple):
    deter: torch.Tensor   # (..., deter_dim)
    mean: torch.Tensor    # (..., stoch_dim)
    std: torch.Tensor     # (..., stoch_dim)
    sample: torch.Tensor  # (..., stoch_dim)


def _gauss_sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mean)
    return mean + std * eps


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 layers: int = 2, act: str = "relu", out_act: bool = False):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh, "silu": nn.SiLU}[act]
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), act_fn()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        if out_act:
            mods += [nn.Tanh()]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Convolutional encoder (pixels -> feature vector)
# --------------------------------------------------------------------------- #
class ConvEncoder(nn.Module):
    """Encodes a 4-frame stacked grayscale observation (B, 4, 84, 84) in [0, 1]."""

    def __init__(self, in_channels: int = 4, feat_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ReLU(),   # 84 -> 41
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),            # 41 -> 20
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),           # 20 -> 9
            nn.Conv2d(128, 128, 3, stride=1), nn.ReLU(),          # 9 -> 7
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat = self.cnn(dummy).flatten(1).shape[1]
        self.fc = nn.Linear(flat, feat_dim)
        self.out_dim = feat_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self.cnn(obs).flatten(1)))


# --------------------------------------------------------------------------- #
# The recurrent state-space model ("world model")
# --------------------------------------------------------------------------- #
class WorldModel(nn.Module):
    def __init__(self, deter_dim: int = 128, stoch_dim: int = 32,
                 enc_dim: int = 128, num_actions: int = 6,
                 obs_channels: int = 4, hidden: int = 128):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_actions = num_actions

        self.encoder = ConvEncoder(in_channels=obs_channels, feat_dim=enc_dim)
        # GRU consumes [z, onehot(a)]
        self.gru = nn.GRUCell(stoch_dim + num_actions, deter_dim)

        self.prior_net = MLP(deter_dim, 2 * stoch_dim, hidden=hidden, layers=1, act="silu")
        self.post_net = MLP(enc_dim + deter_dim, 2 * stoch_dim, hidden=hidden, layers=1, act="silu")
        self.reward_net = MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
        self.continue_net = MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")

        self.ln_post = nn.LayerNorm(stoch_dim)
        self.ln_prior = nn.LayerNorm(stoch_dim)

    # ---- latent construction ------------------------------------------------
    def _dist(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = params.chunk(2, dim=-1)
        std = 0.5 * (log_std.clamp(-8, 8)).exp()  # softplus-like, bounded
        return mean, std.clamp(0.05, 10.0)

    def initial_state(self, batch: int, device: torch.device) -> Latent:
        deter = torch.zeros(batch, self.deter_dim, device=device)
        mean = torch.zeros(batch, self.stoch_dim, device=device)
        std = torch.ones(batch, self.stoch_dim, device=device)
        return Latent(deter, mean, std, mean)

    def img(self, latent: Latent) -> torch.Tensor:
        return torch.cat([latent.deter, latent.sample], dim=-1)

    # ---- one stochastic transition (prior only: imagination / deployment) ---
    def observe_step(self, prev: Latent, action: torch.Tensor,
                     enc_feat: torch.Tensor) -> Tuple[Latent, Latent]:
        """One RSSM step given a real observation encoding.

        Returns (posterior_latent, prior_latent) for time t (both share deter h_t).
        """
        a_onehot = F.one_hot(action, self.num_actions).float()
        gru_in = torch.cat([prev.sample, a_onehot], dim=-1)
        deter = torch.tanh(self.gru(gru_in, prev.deter))
        prior_params = self.prior_net(deter)
        prior_mean, prior_std = self._dist(prior_params)
        prior_mean = self.ln_prior(prior_mean)
        prior = Latent(deter, prior_mean, prior_std, _gauss_sample(prior_mean, prior_std))

        post_params = self.post_net(torch.cat([enc_feat, deter], dim=-1))
        post_mean, post_std = self._dist(post_params)
        post_mean = self.ln_post(post_mean)
        post = Latent(deter, post_mean, post_std, _gauss_sample(post_mean, post_std))
        return post, prior

    def imagine_step(self, prev: Latent, action: torch.Tensor) -> Latent:
        """One RSSM step using the prior only (no observation available)."""
        a_onehot = F.one_hot(action, self.num_actions).float()
        gru_in = torch.cat([prev.sample, a_onehot], dim=-1)
        deter = torch.tanh(self.gru(gru_in, prev.deter))
        prior_params = self.prior_net(deter)
        prior_mean, prior_std = self._dist(prior_params)
        prior_mean = self.ln_prior(prior_mean)
        return Latent(deter, prior_mean, prior_std, _gauss_sample(prior_mean, prior_std))

    # ---- heads --------------------------------------------------------------
    def reward(self, latent: Latent) -> torch.Tensor:
        return self.reward_net(self.img(latent)).squeeze(-1)

    def posterior_imgs(self, obs_seq: torch.Tensor, action_seq: torch.Tensor
                       ) -> torch.Tensor:
        """Belief vectors [h, z] from the posterior at every step of a real seq.

        Used for supervised policy distillation (behaviour cloning) in latent space.
        """
        B, T = action_seq.shape
        latent = self.initial_state(B, obs_seq.device)
        imgs = []
        for t in range(T):
            enc_feat = self.encoder(obs_seq[:, t])
            post, _ = self.observe_step(latent, action_seq[:, t], enc_feat)
            imgs.append(self.img(post))
            latent = post
        return torch.stack(imgs, 1)  # (B, T, deter+stoch)

    def continue_prob(self, latent: Latent) -> torch.Tensor:
        return torch.sigmoid(self.continue_net(self.img(latent))).squeeze(-1)

    # ---- sequence forward over a real trajectory ----------------------------
    def observe_sequence(self, obs_seq: torch.Tensor, action_seq: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
        """Run the observer model over (B, T, C, H, W) frames + (B, T) actions.

        Returns stacked posteriors/priors for KL, and per-step reward/continue
        predictions made from the *posterior* state (teacher forcing).
        """
        B, T = action_seq.shape
        device = obs_seq.device
        post_list, prior_list = [], []
        latent = self.initial_state(B, device)
        for t in range(T):
            enc_feat = self.encoder(obs_seq[:, t])
            post, prior = self.observe_step(latent, action_seq[:, t], enc_feat)
            post_list.append(post)
            prior_list.append(prior)
            latent = post
        post_d = torch.stack([p.deter for p in post_list], 1)      # (B,T,D)
        post_m = torch.stack([p.mean for p in post_list], 1)
        post_s = torch.stack([p.std for p in post_list], 1)
        prior_d = torch.stack([p.deter for p in prior_list], 1)
        prior_m = torch.stack([p.mean for p in prior_list], 1)
        prior_s = torch.stack([p.std for p in prior_list], 1)
        return dict(post_deter=post_d, post_mean=post_m, post_std=post_s,
                    prior_deter=prior_d, prior_mean=prior_m, prior_std=prior_s)


# --------------------------------------------------------------------------- #
# KL divergence (posterior || prior) for diagonal Gaussians, with free nats
# --------------------------------------------------------------------------- #
def gauss_kl(post_mean, post_std, prior_mean, prior_std, free_nats: float = 1.0):
    var_p = prior_std.pow(2)
    var_q = post_std.pow(2)
    kl = 0.5 * ((var_q / var_p).log() + (post_mean - prior_mean).pow(2) / var_p
                + var_q / var_p - 1.0)
    kl = kl.sum(-1)  # (B, T)
    return kl.clamp(min=free_nats)


def model_loss(model: WorldModel, obs_seq: torch.Tensor, action_seq: torch.Tensor,
               reward_seq: torch.Tensor, continue_seq: torch.Tensor,
               free_nats: float = 1.0) -> Dict[str, torch.Tensor]:
    out = model.observe_sequence(obs_seq, action_seq)
    kl = gauss_kl(out["post_mean"], out["post_std"],
                  out["prior_mean"], out["prior_std"], free_nats).mean()

    post = Latent(out["post_deter"], out["post_mean"], out["post_std"],
                  _gauss_sample(out["post_mean"], out["post_std"]))
    # Teacher-forced per-step prediction targets aligned with (B, T).
    r_pred = torch.stack([model.reward(_index_latent(post, t))
                          for t in range(reward_seq.shape[1])], 1)
    c_pred = torch.stack([model.continue_prob(_index_latent(post, t))
                          for t in range(continue_seq.shape[1])], 1)
    rew_loss = F.mse_loss(r_pred, reward_seq)
    cont_loss = F.binary_cross_entropy(c_pred.clamp(1e-4, 1 - 1e-4), continue_seq)
    total = kl + rew_loss + cont_loss
    return dict(loss=total, kl=kl.detach(), rew=rew_loss.detach(), cont=cont_loss.detach())


def _index_latent(latent: Latent, t: int) -> Latent:
    return Latent(latent.deter[:, t], latent.mean[:, t], latent.std[:, t],
                  _gauss_sample(latent.mean[:, t], latent.std[:, t]))
