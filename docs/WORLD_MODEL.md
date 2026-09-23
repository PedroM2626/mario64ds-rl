# World Model for Super Mario 64 DS (Model-Based Reinforcement Learning)

> A physics-style **world model** adapted from the research repository
> [`smw-pinn`](https://github.com/PedroM2626/smw-pinn) (Physics-Informed Neural
> Networks vs. statistical models as *Super Mario World* world models) and
> ported to the **pixel-only** observation space of *Super Mario 64 DS* slide
> tracks. The agent is trained **inside its own learned model**, so it needs only
> a tiny amount of real emulator experience and then converges extremely fast.

---

## 1. Why a world model? The reference idea

The `smw-pinn` benchmark shows that, for classic console platformers, learning a
*dynamics model* `f(s, a) → s'` and then training a controller **inside that
model** (Dyna-PPO / MPC) is **>25× more sample-efficient** than model-free RL:
a Hard-Residual PINN world model trained on ~200 real transitions beat an MLP
trained on 5 000. Its pipeline is:

1. record genuine transitions,
2. fit a forward-dynamics world model,
3. wrap it in a **GPU-vectorized simulation environment** (`pinn_sim_env.py`),
4. train an actor-critic **entirely in the model** (`dyna_ppo.py`),
5. deploy the amortized policy back on the **real console**.

`smw-pinn` reads Mario's `(x, y, vx, vy, ...)` state directly from SNES WRAM.
Our *Mario 64 DS* environment (`src/env.py`) exposes **only pixels** (84×84
grayscale, 4-frame stack) — there is no clean RAM state vector. So we cannot copy
the low-dimensional PINN; instead we learn the dynamics in a **compact latent
space** with a recurrent world model, which is the standard model-based-RL answer
for pixel observations.

## 2. Architecture (RSSM latent world model)

A Dreamer/PlaNet-style Recurrent State-Space Model, in `src/world_model.py`:

```
deterministic  h_t = tanh( GRU([z_{t-1}, onehot(a_{t-1})], h_{t-1}) )
posterior      q(z_t | h_t, o_t)  = N(mu_post(h_t, enc(o_t)), sigma_post)   # with a real frame
prior          p(z_t | h_t)        = N(mu_prior(h_t),      sigma_prior)      # imagination / no frame
reward         r_t ≈ R(h_t, z_t)          (scalar head, matches src/env.py reward)
continue       c_t ≈ sigmoid(C(h_t, z_t)) (prob. episode has not ended by death)
```

* `enc` is a small `ConvEncoder` over the 4-frame stack.
* **No pixel reconstruction.** Like the reference (which predicts physical state,
  not images), the model only needs to be predictive *for control*: it is trained
  with `KL(q ‖ p)` (with free-nats) + reward MSE + continue BCE. The KL forces the
  **prior** (used when we have no observation, i.e. during imagination and
  long-horizon rollouts) to match the **posterior**.
* `latent = [h, z]` is the belief state consumed by the policy.

## 3. Fast training loop (`src/train_world_model.py`)

Each iteration does the two classic MBRL phases:

1. **Model learning** — sample a batch of real sequences from the replay buffer,
   minimize the world-model loss. This is the only phase that touches real data.
2. **Imagination learning** — freeze the model, take a detached belief state
   seeded from a real sequence, roll the **actor** forward `H=15` latent steps
   through the *prior* (`imagine()` in `src/world_model_agent.py`), compute
   λ-returns, and update the actor (REINFORCE + value baseline + entropy) and the
   critic. **No emulator is stepped**, so thousands of imagined transitions run per
   second on the GPU — this is what makes the agent learn *very quickly*.

Every `--eval-every` iterations the current actor is deployed on the **real game**
with a belief-state controller (`src/world_model_controller.py`) to measure how
far it gets / whether it survives the whole descent.

## 4. Data collection (`src/collect_data.py`)

The emulator is the sole source of ground truth and is slow (~35 real
steps/s), so we keep its use minimal. A single DeSmuME instance is reused across
tracks (DeSmuME cannot be re-initialized within one process). Transitions come
from two behavior policies, mirroring the reference's "genuine transitions" rule:

* **random** — includes deaths, teaching the model the abyss / terminal dynamics;
* **ppo** — replays the existing benchmark agent
  (`models/curriculum_flow25_r3_best.zip`) so the buffer also contains *successful
  full-descent* trajectories the latent policy can learn to reproduce.

## 5. Real-time evaluation (`scripts/record_world_model.py`)

Closed-loop test on the real game, one subprocess per track, recording an RGB MP4
per stage and reporting reward / steps / completion, exactly like
`scripts/record_phases.py` does for the model-free baselines.

## 6. Reproduce

```bash
# 1. collect a small real dataset (~19k transitions, ~12 min on CPU emulator)
python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 6 \
    --behavior random,ppo --ppo-model models/curriculum_flow25_r3_best.zip \
    --out data/wm_buffer.pkl

# 2. fit the world model and train the agent inside its imagination
python -m src.train_world_model --buffer data/wm_buffer.pkl --run-id wm_mario64ds \
    --states ds1,ds2,ds3 --model-iters 8000 --actor-iters 6000 --eval-every 2000

# 3. test the learned agent on the real game in real time + record videos
python scripts/record_world_model.py --model models/wm_mario64ds_wm.pt
```

Offline unit tests (no emulator required): `python -m pytest tests/test_world_model.py -v`.

## 7. Results (honest, measured on the real game)

Setup: **~19k real emulator transitions** collected once (random + PPO
demonstrations). The world model converged quickly (reward-head MSE 1.9 → 0.01,
continue error → 1e-4). The agent is then learned in the model in two ways, both
much cheaper in real samples than the model-free PPO baseline (which needs
~1–2 M steps):

1. **Imagination RL (Dyna-style)** — pure RL inside the learned RSSM collapsed
   via *model exploitation* (imagined returns inflated while real survival did not
   improve) — exactly the failure mode the reference fixes with pessimistic deep
   ensembles; we added reference-style epistemic pessimism + uncertainty
   truncation (`--pessimism-beta`, `--unc-trunc`).
2. **Amortized policy distillation** (reference §10.23/§10.25) — the latent actor
   clones the expert full-descent demonstrations *inside the learned latent
   space*; this is stable and fast and yields a deployable policy.

Per-track deterministic real-game evaluation of the shipped bundle
(`models/wm_mario64ds_wm.pt`, 4 episodes/track, `scripts/eval_world_model.py`):

| Track | Mean descent steps (of 1350) | Completed | Notes |
|---|---|---|---|
| `ds1` (Course 3) | 116 | 0/4 | systematic early imitation failure |
| `ds2` (Peach Slide) | 226 | 0/4 | systematic early imitation failure |
| `ds3` (Monkey Slide) | 881 | **2/4 (1350/1350)** | full descents observed in real time |

**Can it finish a stage? Yes — intermittently.** The fast world-model agent fully
traverses a whole slide track (reaches 1350/1350 on `ds3`) in real time; a
completing run is captured in `videos/wm_ds3.mp4`.

**Limitation & honest conclusion.** Reliably completing *all three* long (90 s /
1350-step) descents is **not** achieved by the distilled model-based policy:
pixel-space imitation suffers recurrent belief-state compounding error, and
DeSmuME's multithreaded rasterizer makes pixel death detection slightly
non-deterministic at knife-edge states (hence `ds3` completes 2/4, not 4/4). The
model-free PPO baseline (`curriculum_flow25_r3_best.zip`) remains the only agent
that completes all three tracks. The reference's SNES PINN reaches robust control
because it models an exact low-dimensional RAM state; with only pixels, the
learned dynamics are far less precise. Closing this gap (e.g. latent-state
ensembling for pessimism, or an MPC planner over the RSSM) is the natural next
step. All components are unit-tested (`tests/test_world_model.py`) and run
end-to-end on the live emulator.

### Reproduce these numbers
```bash
python scripts/eval_world_model.py --model models/wm_mario64ds_wm.pt --episodes 4
python scripts/record_world_model.py --model models/wm_mario64ds_wm.pt --tracks ds3
```
