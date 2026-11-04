# 📐 3D Proximity Reward System, RAM State Architecture & Course Completion in SM64DS

This document specifies the academic foundations, reverse-engineered RAM architecture, mathematical formulation, and empirical verification of the goal-directed 3D reward system and RAM-level state detection implemented for *Super Mario 64 DS* (USA Rev 1).

---

## 1. Problem Formulation: Survival vs. Goal Completion

### 1.1 The Survival Fallacy in Slide Courses
Historically, the reinforcement learning agent was trained primarily using a combination of:
1. **Optical Flow Forward Velocity**: $\bar{v}_y \cdot w_{\text{flow}}$, rewarding visual downward pixel shift on the top screen.
2. **Abyss Death Penalty**: $R_{\text{death}} = -100.0$.
3. **Temporal Step Cost**: $c_{\text{step}} = -0.02$.

Under this specification, policy optimization maximizes discounted expected return:
$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

Because the environment terminated only upon **abyss death** (screen fade) or **time horizon exhaustion** ($T = 1350$ steps, 90 seconds), the policy learned that *staying alive until timeout yields higher expected return than risking downhill speed*. Reaching the end of the course was neither explicitly detected nor incentivized, leading to:
- **Plateau Camping**: Loitering on safe terrain (e.g. flat segments on `ds3`) where optical flow was near zero and death risk was zero.
- **Ambiguous Evaluation**: Inability to differentiate whether an agent surviving 1350 steps reached the finish line or merely got stuck against a wall early in the course.

---

## 2. Reverse-Engineered RAM Memory Map (ARM9 Architecture)

Through differential memory tracing and hardware-level state inspection in DeSmuME on *Super Mario 64 DS* (`NTR-ASME-USA`, Rev 1), the canonical memory addresses for actor coordinates, save registers, and level context were isolated:

| Address | Data Type | Representation | Semantic Description |
|---|---|---|---|
| `0x0217F714` | `int32` | $s20.12$ Fixed-Point | **Mario 3D Position $X$** (Track units: value / 4096.0) |
| `0x0217F718` | `int32` | $s20.12$ Fixed-Point | **Mario 3D Position $Y$** (Height: value / 4096.0) |
| `0x0217F71C` | `int32` | $s20.12$ Fixed-Point | **Mario 3D Position $Z$** (Lateral/Depth: value / 4096.0) |
| `0x0217F760` | `int32` | $s20.12$ Fixed-Point | **Kill-Plane / Height Clamp** (Clamps to $-307200 = -75.0$ on void fall) |
| `0x02285238` | `uint32` | Integer ID | **Active Course / Level ID** (`11`=PSS, `5`=TTM, `3`=CCM, `14`=TTC) |
| `0x0209FB64` | `uint32` | Integer Counter | **Global Star Counter** in persistent save RAM |
| `0x02169B10` | `int32[3]` | $s20.12$ Fixed-Point | **Tick Tock Clock Star Object Position** $(X, Y, Z)$ |

All coordinate values in SM64DS utilize the native Nintendo DS Nitro SDK fixed-point format `fx32` ($20$ bits integer, $12$ bits fractional):
$$\text{World Units} = \frac{\text{RAM Value}}{4096.0}$$

---

## 3. Mathematical Formulation of the 3D Proximity Reward

### 3.1 Potential-Based Reward Shaping
To guide the agent toward the goal without distorting the optimal policy $\pi^*$, we employ potential-based reward shaping (Ng, Harada & Russell, 1999):

Let $\mathbf{p}_t = [X_t, Y_t, Z_t]^T \in \mathbb{R}^3$ be Mario's spatial position at decision step $t$, and let $\mathbf{g} \in \mathbb{R}^3$ be the course terminus. The distance metric is Euclidean:
$$D(\mathbf{p}_t, \mathbf{g}) = \|\mathbf{p}_t - \mathbf{g}\|_2 = \sqrt{(X_t - X_g)^2 + (Y_t - Y_g)^2 + (Z_t - Z_g)^2}$$

The potential function $\Phi(s)$ is defined as negative scaled distance:
$$\Phi(s_t) = - w_{\text{dist}} \cdot D(\mathbf{p}_t, \mathbf{g})$$

The shaping term added to the environment reward is:
$$F(s_t, a_t, s_{t+1}) = \Phi(s_{t+1}) - \Phi(s_t) = w_{\text{dist}} \cdot \left[ D(\mathbf{p}_t, \mathbf{g}) - D(\mathbf{p}_{t+1}, \mathbf{g}) \right]$$

### 3.2 Theoretical Guarantees
1. **Monotonic Progress Incentive**: Moving closer to the terminus yields positive reward ($D_{t+1} < D_t \implies F > 0$); retreating or drifting away penalizes the agent ($F < 0$).
2. **Telescoping Sum (No Reward Loops)**:
   $$\sum_{t=0}^{K-1} F(s_t, a_t, s_{t+1}) = w_{\text{dist}} \cdot \left[ D(\mathbf{p}_0, \mathbf{g}) - D(\mathbf{p}_K, \mathbf{g}) \right]$$
   The agent cannot exploit cyclical behaviors (such as oscillating back and forth) to accumulate unbounded artificial reward.
3. **Total Composite Reward**:
   $$R_t = -c_{\text{step}} + w_{\text{flow}} \cdot \max(0, \bar{v}_y) + F(s_t, a_t, s_{t+1}) + R_{\text{terminal}}$$

---

## 4. Course Terminus & Star Coordinates

Using the reference savestates in `data/`, the exact boundary coordinates were extracted directly from the game's RAM:

| Track Key | Course Name | Start State | Terminus State | Start Coordinates $(X, Y, Z)$ | Goal Coordinates $(X, Y, Z)$ | Goal Type | Success Threshold |
|---|---|---|---|---|---|---|---|
| `ds1` | Cool, Cool Mountain slide | `.ds1` | `.ds4` | $(-4444.53, +6409.80, -5976.62)$ | $(-6342.00, -5700.00, -6625.27)$ | Position | $d \le 250.0$ units |
| `ds2` | The Princess's Secret Slide | `.ds2` | `.ds5` | $(+2144.91, +5802.77, -5500.00)$ | $(-6240.47, -4479.99, +4506.09)$ | Position | $d \le 250.0$ units |
| `ds3` | Tall, Tall Mountain slide | `.ds3` | `.ds6` | $(+10012.89, +12697.51, +11114.62)$ | $(-514.75, -13645.99, -11697.84)$ | Position | $d \le 250.0$ units |
| `ds8` | Tick Tock Clock | `.ds8` | `.ds9` | $(+1470.00, -4710.00, +128.00)$ | $(+2350.00, +7145.00, +2350.00)$ | Star | Star $+1$ or $d \le 80.0$ |
| `ds9` | TTC Star Jump | `.ds9` | Target Star | $(+2342.50, +6885.69, +2351.12)$ | $(+2350.00, +7145.00, +2350.00)$ | Star | Star $+1$ or $d \le 80.0$ |

---

## 5. Investigation of Tick Tock Clock (Course 14)

### 5.1 Clock Speed Setting Identification
In *Super Mario 64 DS*, the mechanical speed of moving pendulums, rotating triangular pegs, and conveyor treadmills in Tick Tock Clock depends on the minute hand's angle on the grandfather clock face when entering the portal:
- **12 o'clock ($0^\circ$)**: Stopped / Still (Mechanisms frozen).
- **3 o'clock ($90^\circ$)**: Slow mode.
- **6 o'clock ($180^\circ$)**: Fast mode (Double speed).
- **9 o'clock ($270^\circ$)**: Random mode (Erratic fluctuations).

#### Spectral & Autocorrelation Verification in `ds8`:
To determine the exact setting of `ds8` without unfounded assumptions, we captured an unperturbed 600-frame video sequence ($10$ seconds at 60 Hz) and performed spatial-temporal autocorrelation on the dynamic hazards:
- **Variance Peak**: Moving hazard centered at pixel $(y=158, x=180)$.
- **Autocorrelation Lags**: $\tau \in \{60, 120, 180\}$ steps of 3 frames.
- **Fundamental Oscillation Period**:
  $$T = 60 \times 3 = 180 \text{ frames} \quad (3.00 \text{ seconds at } 60\text{ Hz})$$
Because $T$ is strictly periodic and equal to 180 frames (rather than 90 frames for Fast or non-periodic for Random), `ds8` is empirically proven to be configured in **Slow Mode**.

### 5.2 Star Collection Mechanics (`ds9`)
Savestate `ds9` places Mario on the upper clock platform directly beneath the level's Power Star:
- **Mario Position**: $(X=2342.50, Y=6885.69, Z=2351.12)$
- **Star Actor Position**: $(X=2350.00, Y=7145.00, Z=2350.00)$ at RAM address `0x02169B10`
- **Horizontal Distance**: $\Delta_{xz} = \sqrt{(2350 - 2342.5)^2 + (2350 - 2351.1)^2} = 7.58$ units.
- **Vertical Gap**: $\Delta_y = 7145.00 - 6885.69 = 259.31$ units.

A standard single jump ($B$) reaches a maximum height of $Y = 7091.19$, which is insufficient to intersect the star bounding box. Executing a **Crouch Jump / Backflip** ($R + B$) reaches $Y = 7346.19$, intersecting the Star and triggering:
1. Star collection event: RAM address `0x0209FB64` (Global Star Counter) increments from $25 \to 26$.
2. Environment completion signal: `done = True`, `info["star_collected"] = True`, `reward += 500.0`.

---

## 6. RAM-Based Death Detection

Visual death detection via black-screen pixel thresholding (`obs < 10` on $>95\%$ pixels) suffers from an intrinsic delay of $\sim 60\text{--}80$ frames during the camera tracking and screen fade-out sequence. 

### 6.1 Instantaneous Detection via Hardware Register
In the SM64DS ARM9 physics pipeline:
- Register `0x0217F760` acts as a vertical descent clamp.
- During any fatal abyss fall, this register clamps at exactly $-307200$ (representing $-75.0$ units in $20.12$ fixed-point).
- By monitoring `0x0217F760 <= -300000` (or $Y < Y_{\text{kill}}$ for the course), fatal falls are flagged instantaneously on the exact frame the agent crosses the void plane, prior to camera lag or screen fading.

---

## 7. Legal & Engineering Analysis: Can Savestates Be Committed to Git?

**Verdict: NO. Savestates (`.ds*`) must NOT be committed to the Git repository.**

### Reasons:
1. **Copyright & Intellectual Property**:
   A savestate dump of a commercial Nintendo DS title contains complete unencrypted snapshots of main memory (ARM9 RAM), VRAM (textures, tiles, character sprites), and audio banks. Distributing these binary dumps in a public repository constitutes redistribution of copyrighted proprietary assets.
2. **Repository Size & History Bloat**:
   Each `.ds*` savestate file is $\approx 2.7\text{--}2.8\text{ MB}$. Git stores binary delta revisions inefficiently. Committing savestates expands repository clone sizes significantly and irreversibly without Git LFS.
3. **Existing Project Architecture**:
   The repository's `.gitignore` explicitly prohibited savestates:
   ```gitignore
   # ROMs and savestates (copyrighted / locally generated files)
   data/*.nds
   data/*.ds*
   data/*.dst
   ```
   Files `.ds4`, `.ds5`, `.ds6`, `.ds8`, and `.ds9` appeared as untracked strictly due to an incomplete enumeration (`.ds1`, `.ds2`, `.ds3`) in the original `.gitignore`. Updating the rule to `data/*.ds*` restores the intended architecture.
4. **Best Practice Solution**:
   Store deterministic numerical coordinates, RAM offsets, and verification checksums in the codebase and documentation. Binary savestates must remain in external storage or be generated locally via deterministic emulator scripts.

---

## 8. Color System Audit & Gymnasium Specification Compliance

A systematic audit of color handling across all scripts and pipelines revealed:

1. **DeSmuME Native Layout**: `py-desmume`'s `display_buffer_as_rgbx()` outputs native **BGR(X)** byte ordering.
2. **Grayscale Observations**: Correctly computed via `cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)` after the `[2, 1, 0]` channel reversal. Grayscale calculations are mathematically identical whether computed from BGR or RGB.
3. **OpenCV Video Recording**: `scripts/record_dashboard.py` uses `grab_bgr()` directly for `cv2.VideoWriter`, preserving true colors.
4. **ImageIO MP4 Export**: `scripts/record_phases.py` uses `env.get_screen_rgb()` which swaps `[2, 1, 0]`, feeding true RGB to `imageio`.
5. **Gymnasium API Specification Violation (FIXED)**:
   - *Previous behavior*: In `src/env.py`, `render(mode='rgb_array')` returned `self._get_obs()`, which produced an $84 \times 84 \times 1$ grayscale tensor.
   - *Gymnasium Standard*: `render_mode='rgb_array'` requires an $(H, W, 3)$ uint8 RGB array.
   - *Correction*: `render()` now returns `self.get_screen_rgb()` (shape $192 \times 256 \times 3$, RGB order), restoring full compliance with external evaluation harnesses and video recorders.

---

## 9. Savestate Lifecycle, Episode Reset Latency & Asset Management Pipeline

### 9.1 The Computational Imperative of Savestates in RL
A fundamental question is whether savestates can be completely eliminated from the RL training pipeline in favor of resetting via emulator soft-boot from the ROM.

We conducted empirical timing benchmarks on *Super Mario 64 DS* booting under `py-desmume`:
1. **Cold-Boot ROM Latency**:
   - Nintendo DS firmware & health/safety touch barrier: $\approx 210\text{ frames}$ ($3.5\text{ s}$).
   - Title sequence & interactive Mario face: $\approx 700\text{ frames}$ ($11.6\text{ s}$).
   - File select menu & save file confirmation: $\approx 600\text{ frames}$ ($10.0\text{ s}$).
   - Castle grounds load & character fade-in: $\approx 200\text{ frames}$ ($3.3\text{ s}$).
   - **Total cold-boot reset duration**: $\approx 1700\text{--}2400\text{ frames}$ ($\approx 28\text{--}40\text{ seconds}$).
2. **Savestate Reset Latency (`env.reset()`)**:
   - Memory deserialization via `emu.savestate.load_file(...)`: **$1.2\text{--}1.8\text{ ms}$** ($<0.002\text{ seconds}$).
3. **Training Feasibility Impact**:
   - Over a moderate RL training run of $100{,}000$ episodes, cold-boot resetting would consume $\approx 3{,}500{,}000\text{ seconds}$ ($\mathbf{40.5\text{ days}}$) solely idling through boot menus. With savestates, reset overhead across $10^5$ episodes totals just $\mathbf{2.5\text{ minutes}}$.
   - Therefore, in alignment with OpenAI Gym Retro and ALE benchmarks, **savestates are structurally indispensable for tractable RL training**.

### 9.2 Separation of Runtime States vs. Reference Coordinates
- **Terminus States (`ds4`, `ds5`, `ds6`) are Obsolete**: These states were captured to locate the finish boundaries of courses 11, 5, and 3. Because their 3D coordinates are now hardcoded into `TRACK_GOALS` (`src/ram_state.py`), **`ds4`, `ds5`, and `ds6` are never loaded by `Mario64DSEnv`** and need not exist on disk.
- **Starting States (`ds1`, `ds2`, `ds3`, `ds8`, `ds9`)**: These 5 files represent the true episode initialization points.

### 9.3 Infeasibility of Open-Loop Blind TAS Generators
Attempting to generate `ds8` and `ds9` programmatically from a blank ROM via open-loop keystroke injection is technically unfeasible:
- Course 14 (Tick Tock Clock) is gated behind the 3rd floor 50-star door; cold-booting requires completing 50 stars across the game.
- Savestate `ds9` requires executing a flawless multi-minute 3D platforming sequence over moving gears, pendulums, and conveyor belts to reach a tiny floating platform at height $Y = 6885$. Without an existing frame-by-frame TAS movie or human replay, open-loop macros diverge due to floating-point subpixel drift.

### 9.4 Automated Tooling (`scripts/manage_states.py`)
To maintain zero copyrighted binaries in Git while providing seamless reproducibility, the `scripts/manage_states.py` utility is provided:
- `python scripts/manage_states.py check`: Validates presence and SHA-256 integrity of the ROM and active savestates in `data/`.
- `python scripts/manage_states.py verify`: Boots DeSmuME, loads each state, and confirms Mario's 3D coordinates and course ID against `TRACK_GOALS`.
- `python scripts/manage_states.py bundle --out assets/sm64ds_states.tar.gz`: Compresses active states with a cryptographically hashed `manifest.json` for hosting on external storage (e.g. Hugging Face Hub / GitHub Releases).
- `python scripts/manage_states.py unpack --bundle assets/sm64ds_states.tar.gz`: Restores and verifies states into `data/` in a single command.

---

## 10. Hyperparameter Calibration & Training Regimes for Tick Tock Clock (`ds8`)

### 10.1 Structural Contrast: Slide Gravity Courses vs. Vertical Platforming Tower
The transition from slide courses (`ds1`, `ds2`, `ds3`) to Tick Tock Clock (`ds8`) represents a fundamental shift in environment topology, physical dynamics, and reward landscapes:

| Property | Slide Courses (`ds1`, `ds2`, `ds3`) | Tick Tock Clock (`ds8`) |
|---|---|---|
| **Dominant Motion Vector** | Downward vertical descent ($\Delta Y < 0$) with forward slide | Upward vertical ascent ($\Delta Y = +11{,}855.0$ units) |
| **Terminal Objective** | Spatial finish coordinate plane ($d \le 250.0$) | Power Star actor acquisition ($d \le 80.0$ or Star $+1$) |
| **Environmental Dynamics** | Static terrain geometry (ramps, ice boundaries) | Dynamic periodic hazards (pendulums, rotating gears, conveyor belts) |
| **Optical Flow Correlation** | Downward pixel flow ($\bar{v}_y > 0$) directly correlates with progress | Downward pixel flow ($\bar{v}_y > 0$) corresponds to catastrophic abyss falling |
| **Initial 3D Distance ($D_0$)** | $7{,}974$ (`ds1`), $14{,}644$ (`ds2`), $29{,}352$ (`ds3`) | $12{,}093.0$ units |

### 10.2 Mathematical Derivation of Calibrated Hyperparameters

```
                ┌────────────────────────────────────────────────────────┐
                │         Power Star Goal: (2350, 7145, 2350)            │
                └───────────────────────────▲────────────────────────────┘
                                            │  ΔY = +11,855.0
                                            │  Total Distance D_0 ≈ 12,093.0
                                            │  Potential: +604.65 (w_dist = 0.05)
                                            │
                ┌───────────────────────────┴────────────────────────────┐
                │         Mario Start: (1470, -4710, 128) [ds8]          │
                └────────────────────────────────────────────────────────┘
```

#### 1. Optical Flow Weight ($w_{\text{flow}} = 0.2$)
In slide courses, $w_{\text{flow}} = 2.5$ acted as the primary surrogate reward. In `ds8`, setting $w_{\text{flow}} \ge 1.0$ is counterproductive: Mario must jump vertically to ascend. Under camera follow mechanics, falling down into the abyss creates rapid downward visual flow on the upper screen, which would award substantial artificial reward immediately prior to death. 
We dampen $w_{\text{flow}}$ from $2.5 \to 0.2$ (or $0.0$). A slight positive coefficient ($0.2$) provides gentle exploratory kinetic incentive without overpowering the directional distance signal.

#### 2. Potential Distance Shaping ($w_{\text{dist}} = 0.05$)
With Mario starting at $(1470, -4710, 128)$ and the Star at $(2350, 7145, 2350)$, initial Euclidean distance is:
$$D_0 = \sqrt{(2350 - 1470)^2 + (7145 - (-4710))^2 + (2350 - 128)^2} \approx 12{,}093.0 \text{ units}$$
Applying potential-based shaping with $w_{\text{dist}} = 0.05$:
$$\Delta \Phi_{\text{total}} = w_{\text{dist}} \cdot D_0 = 0.05 \times 12{,}093.0 = +604.65$$
This theoretical cumulative reward:
- Matches the order of magnitude of the terminal star reward ($R_{\text{star}} = +500.0$).
- Grants $+5.0$ points per $100$ units climbed vertically, providing dense gradient feedback for landing jumps on successive ascending platforms.

#### 3. Temporal Step Cost ($c_{\text{step}} = 0.02$)
Tick Tock Clock contains several stationary safe ledges. Without a step cost, an agent could achieve optimal return by idling indefinitely on the starting platform ($R = 0.0$). At $c_{\text{step}} = 0.02$, an episode idling to timeout ($T_{\max} = 1350$) incurs a penalty of $-27.0$. This balances urgency against the requirement to pause briefly for periodic hazards ($T = 180$ frames = 45 decision steps).

#### 4. Instantaneous Void Death Detection (`use_ram_death = True`)
In `ds8`, the baseline platform is at $Y = -4710.0$. Slipping off any gear precipitates an irreversible fall into the void. Visual black-screen detection wastes $60\text{--}80$ frames in camera-tracking animation. By monitoring RAM register `0x0217F760 <= -300000` or $Y < -5500.0$, death is registered on frame 0 with an immediate $-100.0$ penalty, truncating wasteful uninformative transitions.

#### 5. Terminal Star Acquisition Bonus ($R_{\text{star}} = +500.0$)
Verified on `ds9`: when Mario's bounding box intersects the Power Star ($d \le 80.0$ or global star counter increments $25 \to 26$), a $+500.0$ terminal bonus is granted and the episode terminates successfully.

---

### 10.3 Algorithmic Paradigm: QR-DQN + IMPALA CNN vs. PPO + NatureCNN

For sample-efficient training in Tick Tock Clock, the value-based **Quantile Regression DQN (QR-DQN)** paired with the **IMPALA CNN** visual representation architecture was selected over standard on-policy PPO + NatureCNN:

```
+-----------------------------------------------------------------------------------------+
|                                    QR-DQN ARCHITECTURE                                  |
|                                                                                         |
| Observation (4x84x84) ---> [ IMPALA CNN Extractor ] ---> Feature Vector (256-d)         |
|                                                                   │                     |
|                                                                   ▼                     |
|                                                       [ Quantile Network ]              |
|                                                                   │                     |
|                                                                   ▼                     |
|                                                        θ_i(s, a) ∈ R^{200 x 6}          |
|                                                                   │                     |
|                                                 Q(s, a) = (1/N) ∑_i θ_i(s, a)           |
+-----------------------------------------------------------------------------------------+
```

#### 1. Replay Buffer Sample Efficiency
On-policy algorithms (e.g. PPO) collect batches of rollouts, execute 4 mini-batch gradient epochs, and discard all collected transitions. In a 3D platformer with sparse jumping milestones, successfully scaling a moving platform occurs infrequently during early exploration; PPO forgets these rare events rapidly. QR-DQN maintains an experience replay buffer ($N = 20{,}000$ transitions), allowing critical ascent transitions to be sampled and trained upon hundreds of times.

#### 2. Distributional Value Representation under Risk
Platforming over the void exhibits severe multi-modal risk: jumping from an oscillating clock hand either reaches safety ($V \approx +200$) or results in death ($V \approx -100$). Standard DQN collapses this to an expected scalar mean $\mathbb{E}[Q]$, losing information about return variance. QR-DQN models the full return distribution via $N = 200$ discrete quantiles:
$$\mathcal{L}_{\text{QR}}(\theta) = \sum_{i=1}^N \mathbb{E}_{j} \left[ \rho_{\tau_i}^{\kappa} \left( r + \gamma \theta_j(s', a^*) - \theta_i(s, a) \right) \right]$$
where $\rho_\tau^\kappa$ is the Huber quantile loss at target quantile $\tau_i = \frac{2i-1}{2N}$.

#### 3. Visual Representation: IMPALA CNN (15 Layers) vs NatureCNN (3 Layers)
NatureCNN utilizes 3 sequential non-residual convolutions (`conv(8x8, s4) -> conv(4x4, s2) -> conv(3x3, s1)`), which severely downsamples spatial resolution and loses fine geometric features. In TTC, the agent must resolve:
- The phase and rotation speed of thin clock hands and pendulums.
- Depth boundaries between foreground rotating gears and patterned background tower walls.
The IMPALA CNN architecture incorporates 3 residual convolutional stacks (each containing 2 residual blocks with skip connections, 15 convolutional layers total), preserving multi-scale spatial features essential for 3D visual navigation.

#### 4. Calibrated Hyperparameter Summary for `ds8`

| Parameter | Calibrated Value | Rationale |
|---|---|---|
| Algorithm | **QR-DQN** | Sample-efficient off-policy replay with quantile risk representation |
| Feature Extractor | **IMPALA CNN** | 15-layer residual blocks preserve thin gear edges and pendulum phases |
| States | `ds8` | Full-course ascent from base platform |
| Parallel Envs (`n_envs`) | $4$ | Efficient multi-process DeSmuME data collection |
| Replay Buffer Size | $20{,}000$ | Sized to prevent process OOM while retaining historical transitions |
| Quantiles ($N$) | $200$ | Standard Dabney et al. (2018) quantile discretization |
| Learning Rate ($\alpha$) | $5 \times 10^{-5}$ | Conservative Adam learning rate for stable Q-network convergence |
| Batch Size | $32$ | Standard mini-batch update size |
| Frame Skip | $4$ | 15 Hz decision rate (60 Hz emulator physics) |
| Max Steps ($T_{\max}$) | $1350$ | 90 seconds wall-clock episode horizon |
| Optical Flow Weight ($w_{\text{flow}}$) | $0.2$ | Dampened to prevent rewarding void falls |
| Distance Weight ($w_{\text{dist}}$) | $0.05$ | Scales total potential to $+604.65 \approx R_{\text{star}}$ |
| Star Reward ($R_{\text{star}}$) | $+500.0$ | Terminal milestone bonus on star acquisition |
| Step Penalty ($c_{\text{step}}$) | $0.02$ | Anti-camping cost ($-27.0$ for full-horizon idling) |
| Abyss Death Penalty ($R_{\text{death}}$) | $-100.0$ | Punishes void falls |
| RAM Death Detection | `True` | Instantaneous void clamp detection (`0x0217F760 <= -300000`) |

---

### 10.5 Diagnosis of the Jumping Invariant & The High-Water Mark Progression Solution

#### 1. Empirical Discovery: The In-Place Jump Collapse
When evaluating the initial QR-DQN baseline on `ds8`, the agent achieved an apparent reward of $+487.92$, yet spatial tracking revealed it only reached height $Y = -4468.5$ ($+241.5$ units, or **2.04%** of the tower climb) before falling at step 801. Action frequency analysis revealed a degenerate policy:
$$\pi(s) = \text{Action 5 (Jump [B])} \quad \text{for } 100\% \text{ of decisions}$$

Inspection of the initial Q-network distribution confirmed extreme value bias:
* $Q(s_0, \text{Jump [B]}) = \mathbf{+6.66}$
* $Q(s_0, \text{Down [Brake]}) = -0.18$
* $Q(s_0, \text{Up [Accelerate]}) = -1.57$
* $Q(s_0, \text{Left}) = -1.86$
* $Q(s_0, \text{Noop}) = -2.03$
* $Q(s_0, \text{Right}) = -3.42$

#### 2. Mathematical Proof of Potential Exploitation under Discounted Returns ($\gamma < 1$)
In classical potential-based shaping (Ng et al., 1999), policy invariance requires the potential shaping term to incorporate the algorithm's discount factor $\gamma$:
$$F(s_t, a_t, s_{t+1}) = \gamma \Phi(s_{t+1}) - \Phi(s_t)$$
When the environment implements undiscounted potential differences $F_t = \Phi(s_{t+1}) - \Phi(s_t)$ while the Q-network optimizes discounted return with $\gamma = 0.99$:
$$J(\pi) = \sum_{t=0}^\infty \gamma^t \left( \Phi(s_{t+1}) - \Phi(s_t) \right) = -\Phi(s_0) + (1 - \gamma) \sum_{t=1}^\infty \gamma^{t-1} \Phi(s_t)$$
Because $\Phi(s_t) = -w_{\text{dist}} D_t$, maximizing $J(\pi)$ directly incentivizes maximizing the time-discounted average potential $\Phi(s_t)$. By executing a jump, Mario's height rises $+241.5$ units, reducing distance to the star and yielding an immediate reward $+8.25$ at $t$. When landing $k$ frames later, the distance increases, incurring $-8.25$ at $t+k$.
Because future penalties are discounted by $\gamma^k = (0.99)^k < 1.0$:
$$\Delta Q_{\text{cycle}} \approx 8.25 - (0.99)^k \cdot 8.25 > 0$$
Thus, repeated jumping in place created a net positive discounted reward loop, which completely dominated horizontal exploration.

#### 3. The High-Water Mark Progress Solution (`dist_reward_mode="progress"`)
To definitively eliminate reward loops without requiring synchronized discount factors, we replace symmetric potential differences with a **monotonically ratcheted record progression**:

Let $D_{\min}$ be the closest Euclidean distance to the Power Star attained thus far in the episode:
$$D_{\min}^{(0)} = \|\mathbf{p}_0 - \mathbf{g}\|_2$$
At decision step $t$:
$$r_{\text{dist}} = \begin{cases} w_{\text{dist}} \cdot (D_{\min} - D_t) & \text{if } D_t < D_{\min} \\ 0.0 & \text{otherwise} \end{cases}$$
$$D_{\min} \leftarrow \min(D_{\min}, D_t)$$

**Formal Invariants:**
1. **Zero Oscillatory Gain**: On the first jump, Mario sets a new record $D_{\min}^{(1)} = D_0 - 165$, earning $+8.25$. When falling back to the floor, $D > D_{\min}$, earning $0.0$. On every subsequent jump to the same height, $D \ge D_{\min}^{(1)}$; thus, $r_{\text{dist}} = 0.0$ and Mario incurs solely the step penalty $-c_{\text{step}} = -0.02$.
2. **Strict Escalation Requirement**: To earn any positive distance reward beyond $+8.25$, the agent is mathematically required to leap past the initial jump apex and secure footing on higher platforms.
3. **Bounded Return**: Total distance return across any episode remains strictly bounded:
   $$\sum_{t=0}^T r_{\text{dist}} = w_{\text{dist}} \cdot \left( D_0 - D_{\min}^{(\text{terminal})} \right) \le +604.65$$

#### 4. Scaled Training Regime Specification

| Hyperparameter | Baseline Initial | Scaled Production (`v2`) | Impact |
|---|---|---|---|
| Replay Buffer Size | $20{,}000$ | **$100{,}000$ transitions** | Expands replay horizon from ~15 to ~75 full episodes (~5.6 GB RAM) |
| Total Timesteps | $20{,}000$ | **$100{,}000$ steps** | Increases active training from 22 min to ~1.8 hours equivalent gameplay |
| Exploration Fraction | $0.10$ (2k steps) | **$0.30$ (30k steps)** | Prevents premature policy collapse; maintains broad exploratory entropy |
| Distance Reward Mode | `potential` | **`progress` (Ratchet)** | Eliminates in-place jumping exploit; enforces true territorial ascent |
| Final Exploration Rate | $0.01$ | **$0.05$** | Sustains exploratory perturbation during plateau navigation |

---

### 10.6 Empirical Evaluation of Models & Resolution of the False Star Trigger

#### 1. Quantitative Benchmark: Baseline v1 vs Scaled v2

Using our reproducible evaluation protocol (`scripts/eval_qrdqn_ttc.py`) on `ds8` (Tick Tock Clock, Slow Mode), we measured exact spatial trajectories, action histograms, and initial Q-values:

| Metric | Baseline (`v1 Best`) | Scaled v2 Best | Scaled v2 Final | Scaled v3 Best (`ds8_v3_best`) | Scaled v3 Final (`ds8_v3_final`) |
|---|---|---|---|---|---|
| **Distance Reward Mode** | `potential` (Bidirectional) | `progress` (Ratchet) | `progress` (Ratchet) | `progress` (Ratchet) | `progress` (Ratchet) |
| **False Star Bug Active?** | Yes | Yes (during training) | Yes (during training) | **No (Fixed $D \le 500$)** | **No (Fixed $D \le 500$)** |
| **Initial Q(s0, Jump [B])** | **$+6.66$** | **$+0.3435$** | **$+1.0988$** | **$+0.4627$** | **$+0.5307$** |
| **Initial Q(s0, Up [Accel])** | $-1.57$ | $+0.2173$ | $+0.9595$ | $+0.4187$ | $+0.4905$ |
| **Initial Q(s0, Down [Decel])**| $-0.18$ | $+0.2240$ | $+0.9728$ | $+0.4430$ | $+0.5075$ |
| **Initial Q(s0, Right)** | $-3.42$ | $+0.2285$ | $+1.0004$ | $+0.4227$ | $+0.5013$ |
| **Initial Q(s0, Left)** | $-1.86$ | $+0.1939$ | $+0.9803$ | $+0.4121$ | $+0.4900$ |
| **Initial Q(s0, Noop)** | $-2.03$ | $+0.1985$ | $+0.9551$ | $+0.4390$ | $+0.5048$ |
| **Action Distribution** | 100% Jump (B) | 100% Jump (B) | 100% Jump (B) | **41.9% Noop, 57.6% Jump** | **56.0% Noop, 43.1% Jump** |
| **Max Height ($Y_{\max}$)** | $-4468.5$ ($\Delta Y = +241.5$) | $-4468.5$ ($\Delta Y = +241.5$) | $-4468.5$ ($\Delta Y = +241.5$) | **$-4332.0$ ($\Delta Y = +378.0$)** | $-4468.5$ ($\Delta Y = +241.5$) |
| **Ascent Climax (% course)**| **2.04%** | **2.04%** | **2.04%** | **3.19%** | **2.04%** |
| **Mario Final Displacement**| $\Delta = 0$ (Frozen) | $\Delta = 0$ (Frozen) | $\Delta = 0$ (Frozen) | $\Delta Z = +55.5$ | **$\Delta Z = +419.1$ units** |
| **Distance Reward Sum** | $+0.00$ (net cycle) | $+11.83$ (first jump) | $+11.83$ (first jump) | **$+18.98$** (escalated ratchet) | **$+15.49$** |
| **Flow Reward Sum** ($w_{\text{flow}} = 0.2$) | not recorded | $+6.44$ | not recorded | **$+9.06$** | not recorded |
| **Step Penalties** | $-16.02$ (801 steps) | $-27.00$ (1350 steps) | $-27.00$ (1350 steps) | $-27.00$ (1350 steps) | $-27.00$ (1350 steps) |
| **Episode Return (Evaluated)**| $+487.92$ (spurious star) | $-8.73$ (fixed horizon) | $-8.73$ (fixed horizon) | **$+1.04$ (Net Positive)** | $-4.83$ |

#### 2. Root Cause Discovery: The False Star Collection Trigger at Step 801
During evaluations on `ds8`, episodes were observed to terminate at step 801 with `star_collected = True` and $+500.0$ bonus, despite Mario remaining on the starting floor at $Y = -4468.5$ ($12{,}093.5$ units away from the Star at $Y = 7145.0$).

A systematic differential RAM trace (`scripts/diff_ram_star.py`) across the 4MB ARM9 address space revealed the mechanism:
1. **Dynamic Volatility of `0x0209FB64`**: Address `0x0209FB64` was previously hypothesized to represent the global save file star counter. However, cycle-by-cycle logging demonstrated that `0x0209FB64` is an internal dynamic animation/scratch register:
   * At savestate boot: `0x0209FB64 = 86`.
   * At step 38 ($f=152$ frames): drops to `1`.
   * Across 3,500 frames: oscillates through $\{19, 31, 32, 53, 125, 0, 13\}$.
   * At step 801 ($f=3204$ frames): register spikes to `113` (and later `125`).
2. **Environment Condition Failure**: In `src/env.py`, the star completion check was defined as:
   ```python
   if curr_stars > self.start_star_count or curr_dist <= success_dist:
       done = True
       self.star_collected = True
       reward += self.star_reward
   ```
   Because `113 > 86`, the condition `curr_stars > self.start_star_count` evaluated to `True` at step 801, abruptly terminating the episode with $+500.0$ reward!
3. **Training Policy Reinforcement**: During the training of `qrdqn_impala_ds8_v2`, this spurious reward created a massive attractor: the agent discovered that simply jumping or idling safely on the starting platform yielded a guaranteed $+500.0$ terminal payout at step 801 with zero risk of falling into the abyss.

#### 3. Mathematical & Architectural Fix
To definitively prevent spurious memory fluctuations from triggering completion from across the course, the star collection logic in `src/env.py` was constrained with a spatial proximity gate:
```python
if goal_type == "star":
    curr_stars = self.ram.get_star_count()
    if curr_dist <= success_dist or (curr_stars > self.start_star_count and curr_dist <= 500.0):
        done = True
        self.star_collected = True
        reward += self.star_reward
```
Under this formulation:
* In `ds9` (Star Platform), Mario is at $D = 259.4 \le 500.0$ units from the star; executing the jump brings $D \le 80.0 \le success\_dist$, legitimately triggering star collection and the $+500.0$ bonus.
* In `ds8` (Base of TTC), Mario is at $D \approx 12{,}093.5 \gg 500.0$ units; memory spikes at `0x0209FB64` are safely rejected. As verified in post-fix evaluation, the agent now survives the full $1{,}350$ steps, receiving $-27.0$ in step costs, $+11.83$ in initial ratchet progress and $+6.44$ in optical-flow reward (net return $-8.73$).

> **Correction (2026-11-03).** Earlier versions of this section omitted the flow term. `scripts/eval_qrdqn_ttc.py` read `info["flow_reward"]`, a key `Mario64DSEnv._get_info()` did not return, so it printed a flow sum of $+0.0000$; the per-term sums above come from the dashboard recorder (`videos/dashboard_ds8_v{2,3}/dashboard_ds8.json`, `flow_term_sum`). The env now reports `flow_reward` and the evaluator also read `info["is_dead"]` instead of `info["died"]`; both are fixed.

#### 4. Dashboard 60 Hz Video Generation on `ds8`
We recorded the full 60 Hz dashboard verification video for `qrdqn_impala_ds8_v2_best.zip` on `ds8` using `scripts/record_dashboard.py`:
* **Video Output**: `videos/dashboard_ds8_v2/dashboard_ds8.mp4` (10.7 MB, 5,401 frames, 90.02s duration).
* **Metadata & Parity**: `dashboard_ds8.json` records decomposition residual $5.8 \times 10^{-7}$ (exact mathematical zero).
* **Observed Dynamics**: Demonstrates the ratcheted progress mode correctly holding distance reward flat at $+11.83$ after the initial jump, visually confirming the complete elimination of the cyclic reward exploit.

---

### 10.7 Model v3: Training with Fixed Proximity Constraints & Dynamic Decoupling

Following the elimination of the false $+500.0$ star trigger, model `qrdqn_impala_ds8_v3` was trained under identical hyperparameters ($100{,}000$ timesteps, $100{,}000$ buffer size, 4 parallel environments, CUDA).

#### Key Empirical Results:
1. **Elimination of the Freeze / Holding Invariant**:
   In `v2`, the agent held Action 5 (Jump) 100% of the time, which due to lack of a key-release edge caused Mario to land and freeze motionless. In `v3 Best`, the policy learned to interleave Action 0 (NOOP, 41.9%) and Action 5 (Jump, 57.6%). This release edge allows Mario to actively re-jump and maintain mobility throughout the entire episode.
2. **Height Escalation**:
   The peak climb height increased from $-4468.5$ to **$-4332.0$** ($\Delta Y = +378.0$ units, climbing $3.19\%$ of the tower). Ratchet distance reward rose from $+11.83$ to **$+18.98$**. The episode return $+1.04$ decomposes as $+18.98$ (distance) $+9.06$ (optical flow) $-27.00$ (step cost): without the flow term it would be $-8.02$.
3. **Horizontal Territorial Displacement**:
   In `v3 Final`, Mario achieved substantial horizontal travel on the starting platform, moving from initial $Z = 128.0$ to final $Z = 547.1$ ($\Delta Z = +419.1$ units), confirming active exploratory motion rather than in-place freezing.
4. **Dashboard Video**:
   Recorded in `videos/dashboard_ds8_v3/dashboard_ds8.mp4` (11.1 MB, 5,401 frames, 90.02s), with exact zero decomposition residual ($4.46 \times 10^{-7}$).
5. **Checkpoint selection used a different reward**: v1-v3 were trained with `--flow-weight 0.2`, but `src/train_qrdqn.py` defaulted `--eval-flow-weight` to $1.0$, so `EvalCallback` chose `*_best.zip` under a 5x larger flow term than the one optimised. The default is now "same as `--flow-weight`".

---

### 10.8 Why v3 stays on the base platform, and what is now in place

#### 1. Diagnosis (measured facts only)
* **Standing still is paid.** The clock's gears move without Mario moving, so optical flow is non-zero while he idles: $+9.06$ of v3's $+1.04$ return is flow. On `ds8` the flow term rewards the safe plateau rather than the climb.
* **The straight-line distance is almost all height.** Of the initial $12{,}093.5$ units to the star, $11{,}855$ are $\Delta Y$. The ascent winds around the tower, so horizontal moves along the real path rarely set a new $D_{\min}$ and the ratchet pays $0$.
* **Random exploration falls quickly.** Holding Up from `ds8` drops Mario below the start platform after $147$ frames ($\approx 37$ decisions). In v3 training (`tensorboard_logs/qrdqn_impala_ds8_v3`) the first logged episodes averaged $54.5$-$92.2$ decisions with mean return $-83.6$ to $-91.8$, and at 99k steps still $280.1$ decisions and $-81.5$: most $\varepsilon$-greedy episodes end in a fall. Evaluation means were $17.04, 17.04, 37.26, 34.92, 31.31$ at 20k-100k steps, priced at flow weight $1.0$ (point 5 of 10.7; the $37.26$ equals the v3 dashboard re-priced at weight $1.0$, $37.2634$).
* **The action set cannot express a running jump.** Measured from `ds8` over 40 frames: the legacy sequence Up (16 frames) $\to$ B (4) $\to$ release (20) travels $62.2$ units; holding Up throughout with Up+B for the jump travels $148.8$ ($2.4\times$). Holding B longer jumps higher ($262.5$ vs $142.2$ units of peak height for 24 vs 4 frames). Up+Y (run) walks $190$ units in 40 frames vs $135$ for Up. Pressing Up and B on the same frame from standstill does not jump; with B one frame after Up it does (peak $+234.5$). R crouches, and R then B backflips ($+546.8$ peak height). The `compound12` set has no crouch, so the agent cannot backflip.

#### 2. Changes (no model has been trained with them yet)
| Change | Where | Default |
|---|---|---|
| `compound12` action set: legacy six in the same indices + up+left, up+right, up+B, up+left+B, up+right+B, up+Y | `src/env.py` `ACTION_SETS`, `--action-set` | `legacy6` (old checkpoints load unchanged) |
| `dist_reward_mode="route"`: ratcheted arc length along an ordered human route, searched only $[-300, +600]$ units around the best progress and within a $250$-unit corridor; whole route pays `route_total_reward` ($500$) | `src/route.py`, `--dist-mode route --route` | off |
| Start-state curriculum: reset loads the full start with probability `full_start_prob` ($0.25$), otherwise a uniform pick of checkpoints saved along the human run (cf. Salimans & Chen, 2018). Route progress at reset is the checkpoint's own route position, so earlier route is never paid. Evaluation always starts from `ds8` | `src/env.py`, `--start-states route` | off |
| Demo replay-buffer prefill: the human actions are re-executed under the training reward and frame stack and added to the QR-DQN buffer before learning (no large-margin loss, no permanent demo partition) | `src/demos.py`, `--demo` | off |
| `--eval-flow-weight` follows `--flow-weight` | `src/train_qrdqn.py` | changed |
| `info["flow_reward"]`, evaluator death key | `src/env.py`, `scripts/eval_qrdqn_ttc.py` | fixed |

The route and checkpoints come from one human run recorded with `scripts/record_human.py` (`record-ttc.bat`) and converted by `scripts/prepare_human_route.py`, which refuses a recording that does not replay to identical RAM positions. See [HUMAN_RECORDING.md](HUMAN_RECORDING.md). None of this is evidence that the agent can climb TTC; it removes the causes measured above.





---

### 10.9 Model v4: human route, curriculum and demonstrations (2026-11-03)

`qrdqn_impala_ds8_v4` is the first run with everything from 10.8: `compound12`, flow weight $0$, the route reward from one human ascent (`config/routes/ttc_human_v1.json`, 53.2k units, see [HUMAN_RECORDING.md](HUMAN_RECORDING.md)), 75% of training resets at one of the 89 checkpoints along that ascent, 1,732 exact human decisions prefilled into the replay buffer, horizon $3000$. Everything else as v3 (QR-DQN + IMPALA, 4 envs, buffer $100{,}000$, $\varepsilon$ $1.0 \to 0.05$ over the first 30%), but $300{,}000$ steps ($1.03$ h on the RTX 4070 laptop GPU). Config: `models/qrdqn_impala_ds8_v4/run_config.json`.

#### 1. Full start (`ds8`, deterministic, 15 evaluations every 20k steps)
Every evaluation ended in a fall within $47$-$128$ decisions. Returns: $-99.25$ (20k, 40k), $-96.94$ (60k-200k, 240k, 280k, 300k), $-98.24$ (220k), best $-90.79$ (260k, kept as `*_best.zip`). The dashboard of the best checkpoint (`videos/dashboard_ds8_v4/`) reproduces it exactly: 128 decisions, route reward $+11.77$ (1,253 of 53,246 route units), then the kill plane; decomposition residual $1.6 \times 10^{-8}$. Peak height gain from `ds8` is $+268.5$ (v3 best: $+378.0$ without dying). `scripts/eval_qrdqn_ttc.py` used to start its frame stack with four copies of the first frame instead of SB3's $[0, 0, 0, 	ext{frame}]$; for v4 that gave a different trajectory (137 decisions, $+299.2$). It now follows SB3 and reproduces EvalCallback; v3's numbers are identical under both conventions. On the full start, v4 is not better than v3.

#### 2. From the human checkpoints (`scripts/eval_route_starts.py`, 13 starts, 600 decisions each)
| Policy | Route units gained (median) | Starts ending in a fall | Star |
|---|---|---|---|
| do nothing (`--model noop`) | $8.2$ | 2 / 13 | 1 / 13 |
| v4 best (260k) | $336.7$ | 11 / 13 | 1 / 13 |
| v4 final (300k) | $398.2$ | 11 / 13 | 1 / 13 |

The single "star" is the last checkpoint (decision 2,670 of the human run, 4 decisions before the star), which collects it even when idle; it is not a learned completion. v4 gets 41-49x further along the route than standing still, but falls from 11 of 13 starts within 25-583 decisions. Training episodes ended the same way: at 300k the mean training episode lasted $152.4$ decisions with return $-89.7$.

#### 3. What this does and does not show
* The route reward and compound actions changed the behaviour: v3 alternated noop/jump on the start platform; v4 moves along the human path with running jumps (up+B, up+left+B make up 49-87% of its full-start decisions).
* None of the changes taught it to survive the moving platforms. With route reward $\approx 0.0094$ per unit, a typical segment pays a few points while a fall costs $100$, so the measured failure is fall avoidance, not reward sparsity.
* One seed, one human route, 300k steps. These results do not separate the contributions of the curriculum, the demonstrations and the action set.

#### 4. Planned next step (not started): adaptive reverse curriculum
v4 samples its 89 checkpoints uniformly. The planned change follows Salimans & Chen (2018, "Learning Montezuma's Revenge from a single demonstration"): start training only from the checkpoints nearest the star, measure per-checkpoint survival with `scripts/eval_route_starts.py`, and move the start window back along the human run once the agent survives from the current window (for example, success on most starts). Two related changes to try with it: keep the human transitions in the buffer for the whole run with a large-margin supervised loss as in DQfD (Hester et al., 2018) instead of letting them age out, and record more human runs so that more than one route and seed can be compared. No TTC training is planned before this is implemented.
