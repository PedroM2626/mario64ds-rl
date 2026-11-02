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
| `ds1` | Peach's Secret Slide | `.ds1` | `.ds4` | $(-4444.53, +6409.80, -5976.62)$ | $(-6342.00, -5700.00, -6625.27)$ | Position | $d \le 250.0$ units |
| `ds2` | Tall, Tall Mountain | `.ds2` | `.ds5` | $(+2144.91, +5802.77, -5500.00)$ | $(-6240.47, -4479.99, +4506.09)$ | Position | $d \le 250.0$ units |
| `ds3` | Cool, Cool Mountain | `.ds3` | `.ds6` | $(+10012.89, +12697.51, +11114.62)$ | $(-514.75, -13645.99, -11697.84)$ | Position | $d \le 250.0$ units |
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
