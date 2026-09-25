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
# Convolutional decoder (latent -> pixels), used only as a representation
# regularizer. Reconstruction keeps the stochastic latent informative about the
# *visual* scene (e.g. the fall-to-black transition at a cliff), which makes the
# prior rollouts used by MPC / imagination calibrated -- the absence of this is
# what let a reconstruction-free latent be exploited by the flow-reward hack.
# --------------------------------------------------------------------------- #
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int = 4, hidden: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden * 6 * 6)
        self.hidden = hidden

        def up(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                                 nn.Upsample(scale_factor=2, mode="bilinear",
                                             align_corners=False))
        self.net = nn.Sequential(
            up(hidden, 64), up(64, 32), up(32, 16), nn.Conv2d(16, out_channels, 3, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.hidden, 6, 6)
        x = self.net(x)
        return F.interpolate(x, size=(84, 84), mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# The recurrent state-space model ("world model")
# --------------------------------------------------------------------------- #
class WorldModel(nn.Module):
    def __init__(self, deter_dim: int = 128, stoch_dim: int = 32,
                 enc_dim: int = 128, num_actions: int = 6,
                 obs_channels: int = 4, hidden: int = 128, use_decoder: bool = True):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_actions = num_actions
        self.use_decoder = use_decoder

        self.encoder = ConvEncoder(in_channels=obs_channels, feat_dim=enc_dim)
        # GRU consumes [z, onehot(a)]
        self.gru = nn.GRUCell(stoch_dim + num_actions, deter_dim)

        self.prior_net = MLP(deter_dim, 2 * stoch_dim, hidden=hidden, layers=1, act="silu")
        self.post_net = MLP(enc_dim + deter_dim, 2 * stoch_dim, hidden=hidden, layers=1, act="silu")
        self.reward_net = MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
        self.continue_net = MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
        self.decoder = ConvDecoder(deter_dim + stoch_dim, obs_channels) if use_decoder else None

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

    def imagine_step(self, prev: Latent, action: torch.Tensor,
                     deterministic: bool = False) -> Latent:
        """One RSSM step using the prior only (no observation available).

        ``deterministic=True`` uses the prior mean as the latent sample (no noise),
        which is required by MPC planners for stable value estimates of candidate
        action sequences.
        """
        a_onehot = F.one_hot(action, self.num_actions).float()
        gru_in = torch.cat([prev.sample, a_onehot], dim=-1)
        deter = torch.tanh(self.gru(gru_in, prev.deter))
        prior_params = self.prior_net(deter)
        prior_mean, prior_std = self._dist(prior_params)
        prior_mean = self.ln_prior(prior_mean)
        z = prior_mean if deterministic else _gauss_sample(prior_mean, prior_std)
        return Latent(deter, prior_mean, prior_std, z)

    # ---- MPC helper: deterministic discounted value of an action sequence ----
    @torch.no_grad()
    def rollout_value(self, latent: Latent, actions: torch.Tensor,
                      gamma: float = 0.99, pessimism: float = 0.0,
                      terminal_cost: float = 0.0, death_thresh: float = 0.5) -> torch.Tensor:
        """Deterministically roll ``actions`` (T, B) from ``latent`` and return the
        discounted cumulative predicted reward (B,).

        Safety: once the continue head predicts death (``c < death_thresh``) we apply
        a single large ``terminal_cost`` and stop accumulating reward for that
        candidate. This makes the planner avoid cliffs without the flow-reward being
        exploited (falling into the abyss produces *large* downward optical flow, so
        an un-terminated rollout would wrongly reward falling off).
        """
        B = actions.shape[1]
        if latent.deter.shape[0] != B:  # tile a single start state to the batch
            latent = Latent(*[t.expand(B, *t.shape[1:]) for t in latent])
        disc = torch.ones(B, device=actions.device)
        alive = torch.ones(B, device=actions.device)
        value = torch.zeros(B, device=actions.device)
        for t in range(actions.shape[0]):
            latent = self.imagine_step(latent, actions[t], deterministic=True)
            img = self.img(latent)
            c = torch.sigmoid(self.continue_net(img)).squeeze(-1)
            r = self.reward_net(img).squeeze(-1) - pessimism * latent.std.mean(-1)
            dying = ((c < death_thresh) & (alive > 0.5)).float()
            value = value + alive * disc * (r - terminal_cost * dying)
            alive = alive * (1.0 - dying)
            disc = disc * gamma
        return value

    # ---- heads --------------------------------------------------------------
    def reward(self, latent: Latent) -> torch.Tensor:
        return self.reward_net(self.img(latent)).squeeze(-1)

    def recon(self, latent: Latent) -> Optional[torch.Tensor]:
        """Decode the observation from a latent (representation regularizer)."""
        if self.decoder is None:
            return None
        return self.decoder(self.img(latent))

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
               free_nats: float = 1.0, recon_weight: float = 0.0,
               death_weight: float = 1.0, reward_clip: float = 0.0,
               mask_terminal_reward: bool = False) -> Dict[str, torch.Tensor]:
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
    # Terminality is owned by the continue head (and injected by the planner as a
    # terminal cost). The -100 death spike must not enter the reward MSE: with the
    # clip it drags the predicted dense flow negative (~-0.2/step vs a true +0.5),
    # destroying the progress gradient the planner needs. Masking terminal steps
    # out keeps the flow signal crisp while the continue head still supplies death.
    if mask_terminal_reward:
        m = (continue_seq > 0.5).float()
        num = (m * (r_pred - reward_seq) ** 2).sum()
        rew_loss = num / (m.sum() + 1e-6)
    else:
        rew_target = (reward_seq.clamp(-reward_clip, reward_clip)
                      if reward_clip > 0 else reward_seq)
        rew_loss = F.mse_loss(r_pred, rew_target)
    if death_weight > 1.0:
        # Deaths are ~0.1% of steps; without up-weighting the continue head learns
        # "never die", which blinds MPC/imagination to cliffs.
        w = torch.where(continue_seq < 0.5, torch.full_like(continue_seq, death_weight),
                        torch.ones_like(continue_seq))
        bce = F.binary_cross_entropy(c_pred.clamp(1e-4, 1 - 1e-4), continue_seq,
                                     reduction="none")
        cont_loss = (w * bce).sum() / w.sum()
    else:
        cont_loss = F.binary_cross_entropy(c_pred.clamp(1e-4, 1 - 1e-4), continue_seq)
    total = kl + rew_loss + cont_loss
    result = dict(loss=total, kl=kl.detach(), rew=rew_loss.detach(), cont=cont_loss.detach())
    with torch.no_grad():
        result["cont_recall"] = (((c_pred < 0.5) & (continue_seq < 0.5)).float().sum()
                                 / ((continue_seq < 0.5).float().sum() + 1e-6))

    if recon_weight > 0.0 and model.decoder is not None:
        B, T = reward_seq.shape
        img = torch.cat([post.deter, post.sample], dim=-1)      # (B,T,det+stoch)
        dec = model.decoder(img.reshape(B * T, -1)).view(B, T, *obs_seq.shape[2:])
        recon_loss = F.mse_loss(dec, obs_seq)
        result["loss"] = total + recon_weight * recon_loss
        result["recon"] = recon_loss.detach()
    return result


def _index_latent(latent: Latent, t: int) -> Latent:
    return Latent(latent.deter[:, t], latent.mean[:, t], latent.std[:, t],
                  _gauss_sample(latent.mean[:, t], latent.std[:, t]))
