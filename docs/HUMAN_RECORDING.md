# Recording a human run of Tick Tock Clock

The QR-DQN agent does not leave the `ds8` start platform (see
[REWARD_SYSTEM_AND_RAM.md, section 10.8](REWARD_SYSTEM_AND_RAM.md#108-why-v3-stays-on-the-base-platform-and-what-is-now-in-place)).
One human run up to the star provides three things the agent cannot find on
its own: the route (for the `route` progress reward), savestates along that
route (for a start-state curriculum) and demonstration transitions (for the
replay buffer).

## 1. Record

Requirements: Windows, the project `.venv`, the ROM and `ds8` in `data/`
(`python scripts/manage_states.py check`). `ffmpeg` is optional (without it
there is no `gameplay.mp4`).

Double-click `record-ttc.bat` in the repository root, or run:

```powershell
.venv\Scripts\python.exe scripts/record_human.py
```

1. Wait for the **SM64DS RL - Human recorder** window, click it and press
   **Enter**. The game only advances while that window has focus.
2. Controls (any combination, read every frame):

   | Keyboard | DS button | In game (measured on `ds8`) |
   |---|---|---|
   | Arrows or WASD | D-pad | move |
   | Space or K | B | jump (hold longer = higher) |
   | Shift or J | Y | run while moving / punch |
   | Ctrl or C | R | crouch; crouch then jump = backflip (547 units high vs 234) |
   | E or U | A | punch |
   | Q | L | camera behind Mario |
   | X | X | camera |

   **F5** restarts the attempt, **F8** saves the current attempt as a partial
   demo and stops, **F9/Esc** stops without saving. Start/Select are not
   mapped (the pause menu can exit the course).
3. A death discards the attempt and reloads `ds8`; release the keys to start
   the next one. Collecting the star (RAM event, as in training) saves the
   attempt and prints `validated_recording` with its folder.

The game runs at 60 Hz and the buttons are applied frame by frame, as in
smg-rl. The recording keeps every frame's button mask, so the run replays
exactly. Each 4-frame decision also gets a `compound12` label (noop, left,
right, up, down, B, up+left, up+right, up+B, up+left+B, up+right+B, up+Y). It
is `exact` only when the 4 frames held one constant combination that such an
action expresses exactly. Only exact decisions become replay-buffer demos; all
of them count for the route and the checkpoints.

`--input agent` restores the earlier mode: one action per decision, every
decision exact, but presses less than 4 frames apart merge. Measured: if Up
and B start on the same frame from standstill, Mario does not jump; one frame
apart, he does. That is why moving forward and jumping together often failed
in this mode.

Defaults: 120-minute session, 200 attempts, 9000 decisions (10 min) per
attempt (reaching it saves a partial demo), a checkpoint savestate every 30
decisions (2 s). Change them with `--max-minutes`, `--max-attempts`,
`--max-decisions` and `--checkpoint-every`.

Each saved attempt (`runs/human-demonstrations/<time>/completed/attempt-NNNNNN/`) contains:

| File | Content |
|---|---|
| `demo.npz` | `frame_masks` (one button mask per emulated frame), `actions` (per-decision `compound12` labels), `exact`, `rewards`, `positions` (RAM XYZ, one more than actions), `obs` (84x84 uint8 frames the agent would see), `dist_reward`, `flow_reward`, `dist_to_goal` |
| `trajectory.csv`, `trajectory.png` | per-decision position/action table; top-down path and height plot |
| `checkpoints/step-NNNNNN.dst` | DeSmuME savestates along the run |
| `gameplay.mp4` | silent 60 Hz top screen |
| `manifest.json` | source savestate and ROM SHA-256, input mode, action set, frameskip, counts, frames per button, action histogram, file hashes |

`runs/` and `data/curriculum/` are git-ignored: savestates are game RAM dumps
and are not committed.

## 2. Build the route and curriculum

```powershell
.venv\Scripts\python.exe scripts/prepare_human_route.py runs/human-demonstrations/<time>/completed/attempt-000001
```

The script checks the file hashes, replays the recorded actions from `ds8` in a
fresh emulator and refuses the recording unless every position matches and the
star fires again. It then writes `config/routes/ttc_human_v1.json` (waypoints
every 150 units of path, corridor 250, lookahead 600; coordinates only, safe to
commit) and copies the checkpoints to `data/curriculum/ttc_human_v1/`.
A partial demo needs `--allow-partial`.

## 3. Train (when you decide to)

```powershell
.venv\Scripts\python.exe -m src.train_qrdqn --run-id qrdqn_impala_ds8_v4 --states ds8 --features impala `
    --n-envs 4 --timesteps 100000 --buffer-size 100000 --exploration-fraction 0.3 --exploration-final-eps 0.05 `
    --action-set compound12 --flow-weight 0 --dist-mode route --route config/routes/ttc_human_v1.json `
    --start-states route --full-start-prob 0.25 `
    --demo runs/human-demonstrations/<time>/completed/attempt-000001 --use-ram-death --device cuda
```

* `--flow-weight 0`: the flow term paid v3 for idling (+9.06 of its +1.04).
* `--start-states route`: 75% of training resets start at a uniformly chosen
  checkpoint; evaluation (`EvalCallback`, `best_model.zip`) always starts at `ds8`.
* `--demo` replays the recording under these reward settings and adds its
  exact decisions to the replay buffer before learning. They age out of the
  circular buffer like any other transition.
* `--max-steps` stays 1350 (90 s). A human run longer than that still works for
  the curriculum, but a full-start episode cannot reach the star within the
  horizon; check `decisions` in the manifest and raise `--max-steps` if needed.
* The run config is written to `models/<run-id>/run_config.json`.

Evaluate and record the result with the same instrument:

```powershell
.venv\Scripts\python.exe scripts/eval_qrdqn_ttc.py --model models/qrdqn_impala_ds8_v4_best.zip --flow-weight 0 --dist-mode route --route config/routes/ttc_human_v1.json
.venv\Scripts\python.exe scripts/record_dashboard.py --model models/qrdqn_impala_ds8_v4_best.zip --tracks ds8 --flow-weight 0 --dist-mode route --route config/routes/ttc_human_v1.json --out-dir videos/dashboard_ds8_v4
```

Both detect the action set from the checkpoint.

## What has been verified (2026-11-03, no training)

* A scripted 70-decision `compound12` run stored through the recorder's
  `Attempt` class replayed to identical RAM positions (max error 0.000000).
* A scripted 45-decision free-input run (Up then B one frame later, crouch,
  run) also replayed exactly; the running jump from standstill rose 234.5
  units; 36 of 45 decisions were exact and exactly those 36 were prefilled.
* A route-mode env reset from the step-30 checkpoint started at the
  checkpoint's route position (982.14 vs 982.13 recorded) and paid
  (1231.11 - 982.14) / 1231.11 x 500 = 101.1 for the rest of the route.
* Demo prefill replayed 70 transitions into a QR-DQN buffer (68 added, grouped
  by 4 envs) with actions identical to the recording and return 498.6
  (500 route - 70 x 0.02 step cost).
* The interactive window opened, waited for Enter and exited at its time limit.

No human run of TTC has been recorded yet, and no model has been trained with
the route reward, the curriculum, the demo prefill or `compound12`.
