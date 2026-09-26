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
                 obs_channels: int = 4, hidden: int = 128, use_decoder: bool = True,
                 use_value: bool = True, use_q: bool = True):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_actions = num_actions
        self.use_decoder = use_decoder
        self.use_value = use_value
        self.use_q = use_q

        self.encoder = ConvEncoder(in_channels=obs_channels, feat_dim=enc_dim)
        # GRU consumes [z, onehot(a)]
        self.gru = nn.GRUCell(stoch_dim + num_actions, deter_dim)

        self.prior_net = MLP(deter_dim, 2 * stoch_dim, hidden=hidden, layers=1, act="silu")
        self.post_net = MLP(enc_dim + deter_dim, 2 * stoch_dim, hidden=hidden, layers=1, act="silu")
        self.reward_net = MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
        self.continue_net = MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
        self.decoder = ConvDecoder(deter_dim + stoch_dim, obs_channels) if use_decoder else None
        # Value head: a critic fit on REAL discounted returns (posterior states,
        # teacher-forced -- zero imagination drift). The planner bootstraps it as
        # the terminal value of a short rollout, so death foresight comes from
        # real data instead of long, drifting prior rollouts. It learns the one
        # progress signal that *cannot* be earned by falling: the true return of
        # a fall is capped at the -100 death penalty.
        # It is an ENSEMBLE: hazard-approach states carry MIXED data (expert
        # survivals + agent deaths), and a single mean-fit head reads the average
        # (+45 while a fall is seconds away). Bootstrapped heads diverge exactly
        # on those ambiguous states, and the planner bootstraps the MINIMUM --
        # the "survival-weighted, ensemble-uncertainty-pessimistic" terminal value.
        self.value_net = (MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
                          if use_value else None)
        self.value_net2 = (MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
                           if use_value else None)
        self.value_net3 = (MLP(deter_dim + stoch_dim, 1, hidden=hidden, layers=2, act="relu")
                           if use_value else None)
        # Q head: Q(s, a) fit on the SAME real discounted returns, with the taken
        # action folded into the input. It is evaluated on the TRUE posterior of
        # the current state (never on an imagined one), so the planner's first
        # action is grounded in real (s, a) pairs -- near-cliff data shows up
        # here exactly, e.g. "jump while approaching this edge = death", even
        # when everything the model can simulate is blind. The input also
        # carries a continuation token: probe branches exist under several
        # continuation semantics (noop-coast / hold / expert recovery) and the
        # same (s, a) pair has very different returns under each -- without
        # the token, Q learns their (flat, useless) average. The planner
        # evaluates with the noop-coast token (raw consequence of the action).
        self.q_net = (MLP(deter_dim + stoch_dim + num_actions + 3, 1, hidden=hidden,
                          layers=2, act="relu") if use_q else None)
        # normalization of the return targets (set by train_world_model; the head
        # predicts (G - ret_mean) / ret_std)
        self.register_buffer("ret_mean", torch.zeros(()))
        self.register_buffer("ret_std", torch.ones(()))

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
                      terminal_cost: float = 0.0, death_thresh: float = 0.5,
                      value_boot: bool = False, q_weight: float = 0.0,
                      reward_cap: float = 0.0, q_cont: int = 1) -> torch.Tensor:
        """Deterministically roll ``actions`` (T, B) from ``latent`` and return the
        discounted cumulative predicted reward (B,).

        Safety: once the continue head predicts death (``c < death_thresh``) we apply
        a single large ``terminal_cost`` and stop accumulating reward for that
        candidate. This makes the planner avoid cliffs without the flow-reward being
        exploited (falling into the abyss produces *large* downward optical flow, so
        an un-terminated rollout would wrongly reward falling off).

        ``value_boot=True`` adds ``gamma^T * V(s_T)`` from the value head (fit on
        REAL discounted returns) for candidates that are still alive. This extends
        the effective horizon far beyond what the prior can roll out reliably: the
        long-horizon consequences (cliff ahead, safe line) live in V, which was
        learned from teacher-forced posterior states and therefore never suffers
        from prior drift.

        ``q_weight > 0`` grounds the FIRST action with the Q head evaluated on the
        TRUE start belief (real state, never imagined): score = Q(s_0, a_0) +
        model terms for steps 1..T-1 + bootstrap. Q's target is the same real
        return, so the death penalty for lethal first actions is seen even when
        everything the model can simulate is blind (measured: imagined latents
        lose the edge-proximity information after ~2 prior steps).

        ``reward_cap > 0`` clamps the *predicted* per-step reward inside the
        planner. The dense flow reward is a progress gradient, but its spikes
        (fast downward pixel motion -- including while falling off a cliff)
        must not be able to outbid safety inside the search: a fall at full
        speed earns up to +3.8/step, roughly triple a legitimate fast slide.
        Capping keeps the reward as progress signal without paying for falls.
        """
        B = actions.shape[1]
        if latent.deter.shape[0] == 1:  # tile a single start state to the batch
            latent = Latent(*[t.expand(B, *t.shape[1:]) for t in latent])
        elif latent.deter.shape[0] != B:
            raise ValueError(f"latent batch {latent.deter.shape[0]} does not match "
                             f"action batch {B} (only single-state starts are tiled)")
        q_ground = q_weight > 0.0 and self.q_net is not None
        disc = torch.ones(B, device=actions.device)
        alive = torch.ones(B, device=actions.device)
        value = torch.zeros(B, device=actions.device)
        if q_ground:
            # The Q grounding is evaluated under the COMMITTED-continuation
            # semantics (hold token) by default: the hold-probe branches measure
            # "commit to this action", which both carries the expert's
            # momentum patterns (jump-spam holds the racing line) and exposes
            # lethal directions (holding toward an edge falls). The noop-coast
            # token measured raw caution instead and taught the planner to
            # brake through the opening, lose the line's momentum, and get
            # pushed into hazards.
            tok = torch.full((B,), int(q_cont), dtype=torch.long, device=actions.device)
            value = value + q_weight * (self.q(latent, actions[0], cont_token=tok)
                                         * self.ret_std + self.ret_mean)
        for t in range(actions.shape[0]):
            if t == 0 and q_ground:
                # step 0 is carried by Q(s_0, a_0) (its target includes r_0 and
                # everything after); still step the model to reach s_1.
                latent = self.imagine_step(latent, actions[t], deterministic=True)
                disc = disc * gamma
                continue
            latent = self.imagine_step(latent, actions[t], deterministic=True)
            img = self.img(latent)
            c = torch.sigmoid(self.continue_net(img)).squeeze(-1)
            r = self.reward_net(img).squeeze(-1) - pessimism * latent.std.mean(-1)
            if reward_cap > 0.0:
                r = r.clamp(max=reward_cap)
            dying = ((c < death_thresh) & (alive > 0.5)).float()
            value = value + alive * disc * (r - terminal_cost * dying)
            alive = alive * (1.0 - dying)
            disc = disc * gamma
        if value_boot and self.value_net is not None:
            # pessimistic (min) ensemble read: hazard-approach states with mixed
            # data get the death-side interpretation, not the optimistic average
            v = self.value_raw(latent, agg="min")
            value = value + alive * disc * v
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

    def value(self, latent: Latent) -> torch.Tensor:
        """Value-head prediction (normalized return units); None if disabled."""
        if self.value_net is None:
            return None
        return self.value_net(self.img(latent)).squeeze(-1)

    def value_heads(self) -> list:
        """All ensemble value heads that exist in this checkpoint (older bundles
        trained only the first one)."""
        return [n for n in (self.value_net, getattr(self, "value_net2", None),
                           getattr(self, "value_net3", None)) if n is not None]

    def value_raw(self, latent: Latent, agg: str = "mean") -> torch.Tensor:
        """Ensemble value in RAW return units. agg='min' is the pessimistic read
        the planner bootstraps (a hazard-approach state with mixed data gets the
        death-side interpretation from the most pessimistic head)."""
        heads = self.value_heads()
        if not heads:
            return None
        img = self.img(latent)
        outs = [(h(img).squeeze(-1) * self.ret_std + self.ret_mean) for h in heads]
        stacked = torch.stack(outs, 0)
        return stacked.min(0).values if agg == "min" else stacked.mean(0)

    def q(self, latent: Latent, action: torch.Tensor,
           cont_token: torch.Tensor = None) -> torch.Tensor:
        """Q(s, a) prediction (normalized return units); None if disabled.

        ``cont_token`` selects the continuation semantics the probe branches
        were collected under (0 = noop-coast [what the planner evaluates],
        1 = hold, 2 = expert recovery). Without it the historical/BC-style Q
        reading is returned for backward compatibility.
        """
        if self.q_net is None:
            return None
        a_onehot = F.one_hot(action, self.num_actions).float()
        img = self.img(latent)
        if cont_token is None:
            c = torch.zeros(img.shape[0], 3, device=img.device)
        else:
            c = F.one_hot(cont_token, 3).float()
        return self.q_net(torch.cat([img, a_onehot, c], dim=-1)).squeeze(-1)

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
               mask_terminal_reward: bool = False,
               returns_seq: torch.Tensor = None, value_weight: float = 0.0,
               prior_rollout_max: int = 12,
               branch: tuple = None, branch_q_weight: float = 0.0,
               branch_gamma: float = 0.995
               ) -> Dict[str, torch.Tensor]:
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

    # Value head: MSE against the *real* discounted return of each posterior
    # state (already normalized by the caller with model.ret_mean/ret_std).
    # Unlike the reward head this target legitimately includes the -100 death
    # spike -- that is exactly the "cannot be earned by falling" signal: a fall
    # caps the true return, and V learned on teacher-forced states sees cliffs
    # without any prior-rollout drift.
    if value_weight > 0.0 and returns_seq is not None and model.value_net is not None:
        heads = model.value_heads()
        value_loss = torch.zeros((), device=obs_seq.device)
        v_pred = None
        for head in heads:
            # bootstrapped rows per head (deep-ensemble diversity): heads diverge
            # exactly on ambiguous (mixed-outcome) states, which is the point.
            m = (torch.rand(returns_seq.shape, device=obs_seq.device) > 0.2).float()
            v_pred = torch.stack([head(model.img(_index_latent(post, t))).squeeze(-1)
                                  for t in range(returns_seq.shape[1])], 1)
            num = (m * (v_pred - returns_seq) ** 2).sum()
            value_loss = value_loss + num / (m.sum() + 1e-6)
        value_loss = value_loss / max(1, len(heads))
        # Q head: same real-return targets, with the TAKEN action folded into the
        # input. Fit on posterior states only -- this is the reality-grounded
        # term the planner uses for its first action.
        q_loss_val = None
        if model.q_net is not None:
            q_pred = torch.stack([model.q(_index_latent(post, t), action_seq[:, t])
                                  for t in range(action_seq.shape[1])], 1)
            q_loss_val = F.mse_loss(q_pred, returns_seq)
            value_loss = value_loss + q_loss_val
        # Prior-anchoring: V is bootstrapped on *imagined* latents by the
        # planner, but a latent reached by k prior steps has drifted off the
        # posterior manifold. So V must also be fit on prior-rollout states:
        # roll the real actions k steps from a posterior anchor and supervise
        # V(imagined s_k) with the real return at t+k. The rollout is DETACHED:
        # it calibrates the value head on the model's real imagined manifold and
        # must never back-propagate into the dynamics (doing so lets the model
        # move latents to where V reads comfortably and degrades the dynamics).
        T = returns_seq.shape[1]
        K = min(prior_rollout_max, T - 1) if T > 2 else 0
        if K >= 1:
            B = returns_seq.shape[0]
            k = int(torch.randint(1, K + 1, (1,)).item())
            hi_t = T - k - 1
            if hi_t >= 0:
                t0 = torch.randint(0, hi_t + 1, (B,), device=obs_seq.device)
                rows = torch.arange(B, device=obs_seq.device)
                with torch.no_grad():
                    lat = Latent(out["post_deter"][rows, t0], out["post_mean"][rows, t0],
                                 out["post_std"][rows, t0], out["post_mean"][rows, t0])
                    idx = t0[:, None] + torch.arange(k, device=obs_seq.device)[None, :]
                    acts_k = action_seq.gather(1, idx)
                    for j in range(k):
                        lat = model.imagine_step(lat, acts_k[:, j], deterministic=True)
                target = returns_seq.gather(1, (t0 + k)[:, None]).squeeze(1)
                img_lat = model.img(lat)
                # every ensemble head must be calibrated on imagined latents:
                # the planner bootstraps the MINIMUM over heads there
                for head in heads:
                    value_loss = value_loss + F.mse_loss(head(img_lat).squeeze(-1), target)
        result["loss"] = result["loss"] + value_weight * value_loss
        result["value"] = value_loss.detach()
        with torch.no_grad():
            result["value_corr"] = _safe_corr(v_pred, returns_seq)
            if q_loss_val is not None:
                result["q"] = q_loss_val.detach()
                result["q_corr"] = _safe_corr(q_pred, returns_seq)

    # Counterfactual branch Q: from probe-branch episodes (all 6 actions tried
    # from the SAME savestate, real outcomes), fit Q(s_0, a_probe) to the REAL
    # branch return -- discounted branch rewards plus the value head's
    # (detached) bootstrap on the branch's final REAL posterior state. This is
    # the only source of true action contrast: MC returns from ordinary
    # trajectories never see the same state twice, and every head measured
    # action-blind on imagined latents (the prior itself is action-sensitive;
    # posterior-supervised heads just never learned to read those directions).
    if (branch_q_weight > 0.0 and branch is not None and model.q_net is not None
            and model.value_net is not None):
        bobs, bacts, brews, bconts, bpa, bvalid, btok = branch
        out_b = model.observe_sequence(bobs, bacts)
        B, T = bacts.shape
        rows = torch.arange(B, device=bobs.device)
        idx0 = (bpa - 1).clamp(min=0)                       # pre-action state
        s0 = Latent(out_b["post_deter"][rows, idx0], out_b["post_mean"][rows, idx0],
                    out_b["post_std"][rows, idx0], out_b["post_mean"][rows, idx0])
        # discounted real branch return (raw units)
        target = torch.zeros(B, device=bobs.device)
        disc = torch.ones(B, device=bobs.device)
        for k in range(T):
            idx_k = (bpa + k).clamp(max=T - 1)     # masked-out steps may index past T
            m = (bpa + k < bvalid).float()
            target = target + disc * m * brews[rows, idx_k]
            disc = disc * branch_gamma * m
        with torch.no_grad():
            vend = (bvalid - 1).clamp(max=T - 1)
            lat_end = Latent(out_b["post_deter"][rows, vend],
                             out_b["post_mean"][rows, vend],
                             out_b["post_std"][rows, vend],
                             out_b["post_mean"][rows, vend])
            # pessimistic (min) ensemble read, consistent with the planner's
            # terminal bootstrap
            v_end_raw = model.value_raw(lat_end, agg="min")
            alive_end = (bconts[rows, vend] > 0.5).float()
        # raw target = branch rewards + gamma^len * V(end) if the branch survived
        k_len = (bvalid - bpa).clamp(min=0).float()
        target = target + (branch_gamma ** k_len) * alive_end * v_end_raw
        target = (target - model.ret_mean) / model.ret_std
        q_pred = model.q(s0, bacts[rows, bpa], cont_token=btok)
        # Lethal branches carry the decision-critical signal and are ~1% of
        # rows; up-weight them so the mean-reduction cannot average them away.
        w = 1.0 + 9.0 * (1.0 - alive_end)
        branch_loss = (w * (q_pred - target) ** 2).sum() / w.sum()
        result["loss"] = result["loss"] + branch_q_weight * branch_loss
        result["branch_q"] = branch_loss.detach()
    return result


def _safe_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred.flatten()
    t = target.flatten()
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.std() * t.std()).clamp(min=1e-8)
    return (p * t).mean() / denom


def _index_latent(latent: Latent, t: int) -> Latent:
    return Latent(latent.deter[:, t], latent.mean[:, t], latent.std[:, t],
                  _gauss_sample(latent.mean[:, t], latent.std[:, t]))
