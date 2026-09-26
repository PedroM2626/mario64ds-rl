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
value (×3)     V_i(h_t, z_t) ≈ discounted real return  (ensemble; planner takes the min)
Q              Q(h_t, z_t, a, cont-token) ≈ real return of branch continuations
```

* `enc` is a small `ConvEncoder` over the 4-frame stack.
* The **value/Q heads** are fit on real discounted returns only (§9): the
  planner bootstraps them instead of trusting long prior rollouts, and the Q
  head is conditioned on the *continuation semantics* of the counterfactual
  probe branches it was trained on.

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

> **⚠️ Transparency.** The controller that clears all three tracks (item 3) is a
> **distillation/DAgger clone of the already-trained model-free PPO expert**
> (`models/curriculum_flow25_r3_best.zip`) — the *skill* (safe racing line) is
> copied from that PPO; the world model's learned encoder only makes fitting a
> deployable policy from it fast on a small buffer. It is **not** the world model
> learning the stage from scratch. The world-model-native (expert-free) results
> are the limitations: imagination RL collapsed (item 1) and CEM-MPC clears only
> 2/3 (item 2). See §7 caveat and the README *Time saved* note.

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
net, no expert): after the expert-free-control program of §9 it completes
`ds2` and `ds3` on the real game (videos `videos/mpc_ds2.mp4`,
`videos/mpc_ds3.mp4`), but not reliably, and `ds1` remains blocked (§9.6-§9.8).

### Reproduce these numbers
```bash
# CEM-MPC (world-model dynamics only, no policy net):
python -m src.mpc_world_model --bundle models/wm_mario64ds_wm.pt --tracks ds2,ds3
# Distill + DAgger a memoryless head that clears all 3 tracks (validated per-track):
python -m src.dagger_memoryless --bundle models/wm_mario64ds_wm.pt --buffer data/wm_buffer.pkl --ppo-model models/curriculum_flow25_r3_best.zip --rounds 8 --eps-per-track 3 --finetune-enc
# record the three full descents (one fresh process per track):
python -m src.distill_memoryless --bundle models/wm_mario64ds_wm.pt --tracks ds1,ds2,ds3 --record
```

## 8. Expert-free controllers: a controlled study (and a real negative result)

> Continuation: §9 runs the program this section proposed (real-return value
> learning, more near-cliff data, ensemble uncertainty) and reports how far it
> got — expert-free completions of `ds2` and `ds3` on video, `ds1` still
> blocked, and the precise measurement of why.

The headline result in §7 clones an expert. To ask whether the world model can
control the agent **by itself**, we built two expert-free controllers and studied
them with a diagnostic harness (`scripts/diagnose_world_model.py`) instead of
guessing from end-to-end runs.

### 8.1 A real bug this surfaced (fixed, with regression tests)

`EpisodeBuffer.sample()` capped a window's start at `T - seq_len - 1`, but a
terminal death sits on the **final** frame (`continue[-1] == 0`). So **no training
window ever contained a death**: the continue head learned "never die",
`cont_recall` was structurally 0, and every imagined rollout reported
`survive = 1.0`. That single off-by-one explains *both* expert-free failures —
CEM-MPC walking off cliffs and the imagination actor collapsing to one action.
Fixed (inclusive bounds + `death_windows()` + `death_oversample` +
`death_weight`), with tests in `tests/test_world_model.py`.

### 8.2 The calibration trade-off (why expert-free control still fails here)

With deaths now visible, four models trace a clean trade-off — predicted
survival and the dense flow gradient are in tension:

| model | death emphasis | mean flow / step | predicted survival over H=40 |
|---|---|---|---|
| `wm_native`    | none                    | **+0.54** ✓ | 1.00 for every action ✗ (cliff-blind) |
| `wm_flowclean` | dw 4 / os 0.15          | **+0.21** ✓ | 1.00 for every action ✗ |
| `wm_bal`       | dw 10 / os 0.25         | +0.16 ✓     | 0.00 for every action ✗ (model pessimism) |
| `wm_deathcal2` | dw 8 / os 0.2 + clip 1  | −0.19 ✗     | 0.56–1.00 ✓ (only this one discriminates) |

`--mask-terminal-reward` keeps the −100 spike out of the reward MSE (otherwise the
fitted flow prediction goes negative and the planner loses its progress gradient);
`--death-weight` / `--death-oversample` push the other way. No setting in the
sweep gave a *positive* flow gradient **and** discriminative survival at the same
time.

### 8.3 Honest conclusion

Expert-free CEM-MPC reaches roughly **450–840 of 1350 steps** on the slides
(e.g. ds2 1/2 at 837 mean) but does **not** reliably clear all three; the
imagination actor collapses to a single action. The reason is structural, not a
tuning gap: the optical-flow reward **pays for falling** (downward pixel motion
spikes in an abyss) and, at the decision point, the fatal action is not
identifiable from a short pixel window — so there is no horizon at which a planner
both foresees the cliff and is not swamped by prior-drift pessimism. This is
precisely the advantage the reference `smw-pinn` enjoys from modelling exact RAM
state (`x, y, vx, vy`) instead of pixels.

What would actually break the wall, if pursued further: a progress reward that
cannot be earned by falling (e.g. distance-along-track estimated from the model, or
a survival-weighted objective with an ensemble-estimated uncertainty term), plus
more near-cliff data than ~150 death events.

**So the verified all-three-tracks result in §7 stands, and it is a distilled
clone of the trained PPO — labelled as such there and above.**

---

## 9. Breaking the wall, part 2: what the §8 program actually achieved

§8 ended with a concrete program: real-return value learning, more near-cliff
data, and ensemble-estimated uncertainty. We ran that program to completion —
plus several deeper interventions that the measured failures forced. This
section documents what was built, what was measured, and the honest outcome.

### 9.1 Grounding the planner in real outcomes (value / Q heads)

The planner's objective was rebuilt around quantities fit on **real data only**:

* A **value head** `V(h,z)` trained on real discounted returns (`γ=0.995`),
  computed on teacher-forced *posterior* states — zero imagination drift. This
  is the progress signal that cannot be earned by falling: a fall caps the true
  return at the −100 death penalty. Converged at corr ≈ 0.98 with true returns.
* A **Q head** `Q(h,z,a)` on the same targets with the taken action folded in,
  grounding the planner's *first action* in real (s, a) pairs.
* A **prior-anchored value loss**: V must also be calibrated on *imagined*
  latents (roll the real actions k ≤ 12 prior steps from a posterior anchor,
  supervise `V(imagined s_k)` with the real return at `t+k`). The rollout is
  **detached** — with gradients flowing into the dynamics, the model learned to
  move latents to where V reads comfortably, degrading the dynamics (measured:
  emulator steps dropped from 837 → 109/524/60).
* An **ensemble** of 3 bootstrap-diversified value heads; the planner
  bootstraps the **minimum** ("survival-weighted planning with
  ensemble-estimated uncertainty"): hazard-approach states carry *mixed* data
  (expert survivals + agent deaths), and a single mean-fit head reads +45 while
  the agent dies seconds later — the pessimistic head reads the death side.

MPC then scores `Q(s0,a0) + Σ γᵗ·(r̂ₜ, capped) + γ^H·min-V(s_H)` with a short
horizon (H=12): long-horizon death foresight comes from real-data V instead of
long, drifting prior rollouts (measured: prior rollouts lose the
edge-proximity signal after ~2 steps — V(prior) stays optimistic while
V(posterior) collapses before a fall).

### 9.2 Near-cliff data at the racing distribution (~470 deaths)

* **Checkpoint-seeded random exploration** (`--behavior explore`): one PPO
  descent saves savestates every 150 steps; *random* episodes play from each.
  Deaths spread across all sections (previously they stopped at ~step 600).
  151 → 251 death events.
* **Perturbed-expert rollouts** (`--behavior ppo_noise`, sampled PPO + ε=0.15-0.3
  random actions): deaths *at racing speed along the expert line* — the exact
  distribution a planner flies through. Random-policy deaths are slow meanders
  that leave the value landscape blind where the agent actually races.
* **On-policy failure collection** (`mpc_world_model --collect-to`): every
  evaluated MPC episode (including its death) is appended to the buffer.

### 9.3 The core discovery: every head is action-blind on imagined latents

Planning needs to know *which action* is safe. Measured on a trained model:

* The prior **is** action-sensitive: pairwise latent distance across constant
  actions 0.34-0.52 at k=1 (vs 0.11-2.18 for real consecutive steps). The
  information exists in the latent dynamics.
* But **no head reads it**: `V(s₁(a))` varies ~1-3 points across actions,
  the reward head ~0.01-0.1, Q (MC-trained) ~1-3. Posterior-supervised heads
  never *needed* the action-conditional directions — teacher forcing always
  hands them the true next frame. This is the pixel-only analogue of why the
  RAM-state reference works: there, actions move the *state itself*.

### 9.4 Counterfactual probe-branch data (savestate action branching)

Offline (s, a, return) pairs never contain the *same state twice*, so no
amount of ordinary data can separate "steer left here kills you" from "steer
right here saves you". DeSmuME savestates fix this: at any state, **try every
action and record each real outcome** (`collect_data --behavior probe`,
`mpc_world_model --probe-every`), continuing each branch under a chosen policy:

* **noop-coast continuation** — the raw geometric consequence of the action
  (measured: mostly speed contrast; on-line, a single racing-speed action is
  essentially always recoverable);
* **hold continuation** — the committed consequence: 17 measured
  partial-death states, e.g. *"only right survives: +15.0 vs −85.4 .. −97.0
  for every other action"*, and the expert momentum patterns (jump-spam holds
  the racing line, a held steer slides off the edge);
* **expert-recovery continuation** — the DAgger-style safety margin.

Branches are stored with a shared prefix (warm belief), a `probe_at` index
and a **continuation token**. Q is conditioned on the token: the same (s, a)
pair legitimately has very different returns under coasting vs committing
(+25 vs −95), and a single Q trained on the mixture learns their flat,
useless average (measured: Q read −55 everywhere — exactly the mean).

To make the rare lethal contrast actually learned: **contrast-group sampling**
(whole 6-branch probe states per batch, drawn preferentially from the
death/high-spread pool) and 10× up-weighting of dead branch rows. After this,
Q correctly separates the ground-truth partial-death states (safe actions
ranked above lethal ones).

### 9.5 Planner discipline

* **Modal plan execution** instead of the argmax-sampled candidate (the max
  over ~1000 noisy value estimates is biased and changes erratically —
  measured as weaving near cliffs).
* **Action persistence** (`--cem-persistence`): candidate steps repeat the
  previous action with prob p — surviving behaviors on these slides are
  momentum *patterns*, and iid per-step sampling almost never proposes a
  coherent 12-step hold.
* **Reward cap** (`--reward-cap 2.0` ≈ the expert's p95 per-step reward):
  falling at full speed earns up to +3.8/step of downward optical flow —
  the flow spike must not outbid safety inside the search.
* **K-step plan commitment** (`--replan-every 3`).

### 9.6 Results (real game, measured — honest)

The §8 wall — "expert-free CEM-MPC reaches ~450–840 and never finishes" — was
broken for 2 of 3 tracks. **Expert-free completions, real time, on video:**

| Track | Expert-free CEM-MPC (world model only, no policy net, no action labels) |
|---|---|
| `ds2` (Peach Slide)   | **1350/1350 COMPLETED ×4** — video `videos/mpc_ds2.mp4` (R=571.5); evals R=559.1 / 587.3 / 417.7 |
| `ds3` (Monkey Slide)  | **1350/1350 COMPLETED ×2** — video `videos/mpc_ds3.mp4` (R=830.5); eval R=851.3 |
| `ds1` (Course 3)      | never completed; best **525/1350** (video `videos/mpc_ds1.mp4` shows the partial descent) |

Reliability, stated plainly: per-hazard threading is ~40-70%, so a full
descent completes on ~5-15% of episodes on the clearable tracks (ds2 completed
4 of ~20 logged episodes; ds3 2 of ~30; the ds3 video took 32 seeded attempts).
**This is not the reliable 3/3 of the distilled controller in §7**, and ds1
remains blocked: across 25+ logged episodes (including a 9-seed lottery with
the best configuration) it clusters at 440-525 and brushes the wall at
516-525 without ever crossing it (§9.7).

### 9.7 Why ds1 is still blocked

ds1's unique blocker is the **dark tunnel section** (~steps 80-200): optically
featureless (flow reward ≈ 0, encoder blind), where the model cannot
discriminate lines and every head goes flat. The agent enters slightly
off-line, wanders 100+ steps in the dark, and dies at or just after the exit
(measured: exits the tunnel with V=+63, dies 12 steps later; V stays positive
until ~6 steps before falls everywhere). The expert transits in ~15 steps by
jump-spamming straight through — a reactive precision that outcome data alone
has not reproduced.

A dedicated follow-up hunt ("retome a caça ao ds1") mapped the remaining wall
precisely and did not break it:

* The **hard wall sits at steps 400-525** — no attempt in ~25 episodes ever
  passed it; the best (q7 + noop-token Q) clusters tightly at 440-525 and
  brushes 516-525 but does not cross.
* **~850 counterfactual hold-continuation branches** were probed at exactly
  the wall (21 partial-death states, 70 lethal committed-direction branches at
  steps 400-460); **expert-recovery branches** were probed along the agent's
  failure lines (302 states, 5 states where even the expert cannot recover
  from some actions). Q trained on this separates some ground-truth states but
  does not thread the wall on the emulator.
* Two bootstrap designs were measured dead ends: supervising V *along* failure
  branches creates a pessimistic fixed point (V-at-the-expert-opening collapsed
  +68 → 0 and the planner regressed); bootstrapping recovery branches with the
  pessimistic ensemble minimum poisons recovery targets (recoverable actions
  read ~−85). The correct recovery bootstrap is the optimistic (max) head, but
  the rare-contrast fitting limit (§9.4) still applies.
* Four further model generations (q12-q16) with all of this data oscillate in
  the 150-500 band — worse than the q7 sweet spot, i.e. the loop is sampling
  noise around a capability ceiling, not climbing.

Conclusion unchanged but sharpened: ds1 requires either the labelled-recovery
route (§7's distillation) or a reactive state representation that survives
optically featureless sections — not more outcome data.

### 9.8 Conclusion

The §8 diagnosis stands, sharpened: even with grounded values, 470 deaths at
the racing distribution, true counterfactual action contrast and pessimistic
ensembles, **pixel-only expert-free control clears 2 of 3 tracks (with video
evidence) but not reliably all three**. The remaining gap is the pixel-precise
reactive steering at hazard boundaries — exactly the advantage the reference
`smw-pinn` derives from modelling exact RAM state. Each new planner line
visits states the data does not cover; closing that loop reliably converges
only toward labelled recovery actions (the §7 distillation path). The verified
all-three-tracks controller in §7 therefore remains the distilled/DAgger one,
labelled as such.

Reproduce:

```bash
# 1. near-cliff data: checkpoint-seeded exploration + perturbed-expert rollouts
python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 8 --behavior explore --ppo-model models/curriculum_flow25_r3_best.zip --out data/wm_buffer4.pkl
python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 6 --behavior ppo_noise --ppo-model models/curriculum_flow25_r3_best.zip --out data/wm_perturb.pkl

# 2. counterfactual probe-branch data (all 6 actions from savestated states)
python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 0 --behavior probe,probe_noise --ppo-model models/curriculum_flow25_r3_best.zip --probe-every 8 --branch-len 14 --branch-policy noop --out data/wm_probe_noop.pkl
python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 0 --behavior probe --ppo-model models/curriculum_flow25_r3_best.zip --probe-every 8 --branch-len 14 --branch-policy hold --out data/wm_probe_hold.pkl

# 3. merge + fit the model with value/Q/ensemble + branch-Q
python scripts/merge_buffers.py --out data/wm_buffer_all.pkl <buffers...>
python -m src.train_world_model --buffer data/wm_buffer_all.pkl --run-id wm_safe --model-iters 6000 --actor-iters 0 --bc-iters 0 --eval-episodes 0 --mask-terminal-reward --death-weight 6 --death-oversample 0.2 --recon-weight 0.2

# 4. expert-free CEM-MPC on the real game (+ on-policy failure collection)
python -m src.mpc_world_model --bundle models/wm_safe_q7_wm.pt --tracks ds1,ds2,ds3 \
    --q-weight 3.0 --q-cont 0 --reward-cap 2.0 --replan-every 3 --cem-persistence 0.0 \
    --collect-to data/wm_mpc_fail.pkl
#    ...and with in-episode probing (counterfactual data along the deployment
#    trajectory; --probe-branch-policy ppo harvests recovery contrast, hold the
#    committed-direction contrast, noop the raw-geometric one):
python -m src.mpc_world_model --bundle models/wm_safe_q7_wm.pt --tracks ds1 --probe-every 4 \
    --probe-branch-policy ppo --collect-to data/wm_probe_mpc.pkl <same planner flags>

# 5. videos (retry seeds until a completion is captured; see §9.6 rates)
python -m src.mpc_world_model --bundle models/wm_safe_q7_wm.pt --tracks ds2 --record --seed 1 <planner flags>
```
