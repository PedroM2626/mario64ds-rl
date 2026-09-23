"""Deploy the world-model actor on the real emulator (belief-state controller).

At every real step the controller encodes the incoming frame with the *posterior*
observer model (we have a true observation) and rolls the recurrent belief state
``(h, z)`` forward with the action taken at the previous step — the same
``observe_step`` used during training. The frozen actor then maps the latent
``[h, z]`` to an action. This is the model-to-real transfer step the reference
calls "closed-loop control on the real console".
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

from src.world_model import WorldModel
from src.world_model_agent import ActorCritic

STACK = 4


class WorldModelController:
    def __init__(self, model: WorldModel, actor: ActorCritic, device: str = "cuda",
                 deterministic: bool = True):
        self.model = model.to(device).eval()
        self.actor = actor.to(device).eval()
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.latent = None
        self.prev_action = None
        self.dq = None

    @torch.no_grad()
    def reset(self, first_frame: np.ndarray):
        self.latent = self.model.initial_state(1, self.device)
        self.prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self.dq = deque([np.zeros_like(first_frame)] * (STACK - 1) + [first_frame], maxlen=STACK)
        return self._act_from_stack()

    def _stack(self) -> torch.Tensor:
        arr = np.stack(list(self.dq), axis=0).astype(np.float32) / 255.0  # (4,84,84)
        return torch.from_numpy(arr)[None].to(self.device)                  # (1,4,84,84)

    @torch.no_grad()
    def _act_from_stack(self) -> int:
        enc = self.model.encoder(self._stack())
        post, _ = self.model.observe_step(self.latent, self.prev_action, enc)
        self.latent = post
        img = self.model.img(post)
        a = self.actor.act(img, deterministic=self.deterministic)
        self.prev_action = a
        return int(a.item())

    @torch.no_grad()
    def act(self, frame: np.ndarray) -> int:
        """frame: (84,84) uint8 single grayscale frame for the *current* step."""
        self.dq.append(frame)
        return self._act_from_stack()


def load_world_model_actor(model_path: str, device: str = "cuda"):
    """Load a saved world model + actor bundle (see train_world_model.py)."""
    ckpt = torch.load(model_path, map_location=device)
    wm_cfg = ckpt["world_model_cfg"]
    model = WorldModel(**wm_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    actor = ActorCritic(wm_cfg["deter_dim"], wm_cfg["stoch_dim"], wm_cfg["num_actions"])
    actor.load_state_dict(ckpt["actor"])
    return model, actor, ckpt.get("meta", {})
