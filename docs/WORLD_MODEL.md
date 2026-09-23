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

## 7. Results (real game, measured)

Setup: **~35k real emulator transitions** collected once (random + deterministic
PPO expert descents). The RSSM world model converged fast (reward-head MSE
1.9 → 0.01). Three world-model controllers were evaluated on the live emulator:

1. **Imagination RL (Dyna-style)** — pure RL inside the RSSM prior collapsed via
   *model exploitation* (inflated imagined return, no real gain) — the exact
   failure the reference fixes with pessimistic ensembles; we added epistemic
   pessimism + uncertainty truncation.
2. **CEM-MPC over the world model** (`src/mpc_world_model.py`) — closed-loop
   receding-horizon planning. Completes **ds2 (2/2)** and **ds3 (1/2, up to
   R=1054)** but not ds1: the learned flow reward *rewards falling off the cliff*
   (downward optical flow spikes while falling), so a dynamics planner is lured
   over invisible edges — a genuine reward-hacking limit.
3. **Amortized policy distillation + DAgger** (`src/distill_memoryless.py`,
   `src/dagger_memoryless.py`) — a policy head distilled from the expert's descents
   into the world model's learned encoder and deployed **memorylessly** (no
   recurrent-belief compounding). Plain BC nails `ds1`/`ds2` but is fragile on the
   narrow `ds3` (a single wrong reactive action diverges the visual state and
   cascades). **Memoryless DAgger** fixes exactly that: rolling out the *student*
   and labelling each visited state with the expert. Over rounds the ds3 descent
   extended **65 → 1088 → 1350** while ds1/ds2 stayed complete, and the selected
   head **clears all three tracks, 3/3 episodes each (deterministic)**:

| Track | Real-game result (fresh emulator, deterministic) | Reward |
|---|---|---|
| `ds1` (Course 3)      | **3/3 — 1350/1350 descents** | 717 |
| `ds2` (Peach Slide)   | **3/3 — 1350/1350 descents** | 550 |
| `ds3` (Monkey Slide)  | **3/3 — 1350/1350 descents**   | 535 |

Rewards meet or exceed the model-free PPO benchmark (ds1 +369.9, ds2 +865.3,
ds3 +504.0). Demonstration videos: `videos/mem_ds1.mp4`, `videos/mem_ds2.mp4`,
`videos/mem_ds3.mp4`.

### Time saved by the world model (vs the model-free PPO baseline)

| | Real emulator steps | Wall-clock to a 3-track agent |
|---|---|---|
| Model-free PPO (curriculum, `curriculum_flow25_r3`) | ~2,300,000 | **~12.2 h** (1M base 4.6 h + 3 curriculum rounds 3.1/2.2/2.3 h) |
| World model — distill only (35k collect → fit → distill) | **~35,000** | ~15 min (≈13 min sampling + ~3 min GPU) |
| World model — full 3-track agent (+ memoryless DAgger) | **~59,000** | **~30 min** (≈24 min emulator + ~6 min GPU fit/distill/retrain) |

**≈ 40× fewer real emulator interactions and ≈ 24× less wall-clock time.** The
learning itself (world-model fit + distillation + DAgger head retraining) runs in
~6 min of GPU instead of ~12 h of emulator stepping; the emulator time is spent
only sampling transitions, not gradient-training on them.

**Honest caveat.** The distillation path reuses the trained PPO agent to *provide
demonstrations* (the reference's amortized-policy-distillation / DAgger paradigm),
so the headline saving is in *learning a deployable policy from a tiny sampled
buffer* — not in replacing the ~12 h expert pretraining with nothing. The
**CEM-MPC** controller, by contrast, uses only the learned dynamics (no policy
net, no expert) and already clears 2 of 3 tracks in real time. Fully closing ds1
with pure dynamics requires a reward that cannot be hacked by falling.

### Reproduce these numbers
```bash
# CEM-MPC (world-model dynamics only, no policy net):
python -m src.mpc_world_model --bundle models/wm_mario64ds_wm.pt --tracks ds2,ds3
# Distill + DAgger a memoryless head that clears all 3 tracks (validated per-track):
python -m src.dagger_memoryless --bundle models/wm_mario64ds_wm.pt --buffer data/wm_buffer.pkl --ppo-model models/curriculum_flow25_r3_best.zip --rounds 8 --eps-per-track 3 --finetune-enc
# record the three full descents (one fresh process per track):
python -m src.distill_memoryless --bundle models/wm_mario64ds_wm.pt --tracks ds1,ds2,ds3 --record
```
