"""Record a dashboard-style video of the SM64DS slide agent: gameplay at the
emulator's true 60 Hz, what the CNN actually receives, the policy's real action
probabilities, the critic's value, the RAM physics and the reward instrument.

The point of this tool is evidence. Each frame carries the track, the step count
against the 1350-decision horizon, the reward terms that produced the published
numbers and the controller identity, so a viewer can tell what a run proved
without opening a metrics file. Every run also prints a RESULT line and writes
out/dashboard_<track>.json with the same numbers it displayed -- the README
gate re-derives its cells from those files.

Two reward instruments exist in this repo (src/eval.py uses the env default
flow_weight=1.0; scripts/record_phases.py records at the training weight 2.5).
The recorder therefore logs the weight-free flow sum, so one recording re-prices
the episode under either instrument and both published columns stay checkable.

Each track runs in its own subprocess because two DeSmuME instances in one
process take an access violation (see scripts/record_phases.py). Supervising
those subprocesses -- launch, stdout stream, per-run budget, kill on overrun,
result collection -- is the std-only Rust binary in rust/. Without it the same
job list runs through Python; both arms parse the identical RESULT lines.

Usage:
    python scripts/record_dashboard.py --model models/curriculum_flow25_r3_best.zip
    python scripts/record_dashboard.py --tracks ds1 --out-dir media
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.env import ACTION_BUTTONS, ACTION_NAMES  # noqa: E402
from src.ram_state import Sm64DsState, NAMES, KILL_PLANE, SCALE  # noqa: E402

ALL_TRACKS = ["ds1", "ds2", "ds3", "ds8", "ds9"]
DEFAULT_TRACKS = ["ds1", "ds2", "ds3"]
TRACKS = ALL_TRACKS
TRACK_NAMES = {
    "ds1": "Cool, Cool Mountain slide",
    "ds2": "Princess Peach's Secret Slide",
    "ds3": "Tall, Tall Mountain slide (Monkey)",
    "ds8": "Tick Tock Clock (Course 14) - Slow Mode",
    "ds9": "Tick Tock Clock - Star Platform",
}
STATE_PATH = {t: os.path.join(ROOT_DIR, "data",
                              f"Super Mario 64 DS (USA) (Rev 1).{t}") for t in ALL_TRACKS}
RUST_SUP = os.path.join(ROOT_DIR, "rust", "sm64_sup.exe")

W, H = 1280, 720
BG = (24, 20, 14)
PANEL = (34, 30, 22)
FG = (216, 210, 200)
DIM = (140, 134, 124)
ACCENT = (60, 190, 245)
GOOD = (90, 200, 120)
BAD = (80, 90, 235)
GRID = (52, 48, 42)

# action_space = Discrete(6); src/env.py holds the mapped key for the whole
# frameskip window, so "held" below is literally the keypad state. The names and
# button labels live in src/env.py so the dashboard cannot drift from the env.
WINDOW = 60  # decisions behind the rolling action mix


# ------------------------------------------------------------------ drawing
def panel_box(img, x, y, w, h, title=""):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 56, 48), 1)
    if title:
        cv2.putText(img, title.upper(), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, ACCENT, 1, cv2.LINE_AA)


def text(img, s, x, y, scale=0.45, color=FG, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_agent_view(img, x, y, st):
    """The pixels the policy consumes: the 84x84 grayscale CNN input (top DS
    screen, one channel of the 4-stack) and the 32x32 window the flow reward is
    measured on, with the mean vertical flow drawn as the arrow that scored it."""
    obs84 = st["obs84"]
    view = cv2.resize(obs84, (168, 168), interpolation=cv2.INTER_NEAREST)
    img[y:y + 168, x:x + 168] = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (x, y), (x + 168, y + 168), (60, 56, 48), 1)
    text(img, "84x84 gray CNN input (2x)", x, y - 6, 0.32, DIM)
    dark = float(np.mean(obs84 < 10))
    text(img, f"dark px {dark * 100:.0f}% of the input (env calls >95% a death)",
         x, y + 182, 0.30, BAD if dark > 0.95 else DIM)

    fx, fw = x + 188, 128
    top = y + 20
    gv = st["flow_win"]
    if gv is not None:
        g = cv2.resize(gv, (fw, fw), interpolation=cv2.INTER_NEAREST)
        img[top:top + fw, fx:fx + fw] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (fx, top), (fx + fw, top + fw), (60, 56, 48), 1)
        text(img, "32x32 flow window (2px)", fx, y - 6, 0.32, DIM)
        flow = st["flow"]
        if flow is not None:
            # the reward uses mean(flow_y) alone; the arrow shows that mean, and
            # is clamped inside the box -- an unclamped one drew across the frame
            cx, cy = fx + fw // 2, top + fw // 2
            my, mx = float(np.mean(flow[..., 1])), float(np.mean(flow[..., 0]))
            r = fw // 2 - 6
            dx = max(-r, min(r, mx * fw * 2.0))
            dy = max(-r, min(r, my * fw * 2.0))
            cv2.arrowedLine(img, (cx, cy), (int(cx + dx), int(cy + dy)), ACCENT, 2,
                            cv2.LINE_AA, tipLength=0.3)
            tx = fx + fw + 10
            text(img, "mean flow_y", tx, top + 26, 0.30, DIM)
            text(img, f"{my:+.4f} px", tx, top + 44, 0.34, GOOD if my > 0 else BAD)
            text(img, f"x{st['flow_weight']:g} reward", tx, top + 66, 0.30, DIM)
            text(img, f"{st['flow_reward']:+.4f}", tx, top + 84, 0.34, FG)
            text(img, "per decision", tx, top + 104, 0.26, DIM)


def draw_probs(img, x, y, w, h, probs, taken, recent):
    """pi(a|s) straight from the policy's categorical head. The deployed policy is
    argmax here (deterministic eval), so the bar is the decision, not a hint."""
    counts = collections.Counter(recent)
    n = max(1, len(recent))
    bh = (h - 6) / len(ACTION_NAMES)
    for i, name in enumerate(ACTION_NAMES):
        yy = int(y + 4 + i * bh)
        text(img, name, x, yy + 12, 0.36, FG if i == taken else DIM)
        bx, bw = x + 118, w - 118 - 92
        cv2.rectangle(img, (bx, yy + 4), (bx + bw, yy + 13), GRID, -1)
        if probs is not None:
            p = float(probs[i])
            fill = int(bw * min(1.0, p))
            if fill:
                cv2.rectangle(img, (bx, yy + 4), (bx + fill, yy + 13),
                              GOOD if i == taken else ACCENT, -1)
            text(img, f"{p * 100:5.1f}%", bx + bw + 4, yy + 13, 0.32,
                 GOOD if i == taken else DIM)
        text(img, f"mix {100.0 * counts.get(i, 0) / n:3.0f}%", bx + bw + 50, yy + 13,
             0.28, DIM)


def draw_series(img, x, y, w, h, arr, color, lo, hi, thick=2):
    a = np.asarray(arr, dtype=float)
    if len(a) < 2 or np.any(np.isnan(a)):
        return
    xs = np.linspace(x, x + w, len(a))
    ys = y + h - (a - lo) / ((hi - lo) or 1.0) * h
    cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, color, thick,
                  cv2.LINE_AA)


def draw_value(img, x, y, w, h, st):
    hv, hr = st["hist_v"][-200:], st["hist_r"][-200:]
    if len(hv) < 2:
        text(img, "collecting", x, y + h // 2, 0.40, DIM)
        return
    draw_series(img, x, y, w, h, hr, (72, 72, 72), float(min(np.min(hr), 0.0)),
                float(max(np.max(hr), 1e-6)), 1)
    lo, hi = float(np.min(hv)), float(np.max(hv))
    draw_series(img, x, y, w, h, hv, GOOD, lo, hi, 2)
    if st["hist_wm"] and len(st["hist_wm"][-200:]) > 1:
        draw_series(img, x, y, w, h, st["hist_wm"][-200:], ACCENT, lo, hi, 1)
    text(img, f"green: critic V(s) {st['value']:+.2f}   grey: per-step r "
              f"{st['step_reward']:+.3f}", x, y + h + 14, 0.32, DIM)
    text(img, f"cumulative R {st['total_reward']:+.2f}", x, y + h + 30, 0.34, FG)
    if st["wm_pred"] is not None:
        r_hat, fall_p = st["wm_pred"]
        text(img, f"cyan: RAM model r_hat {r_hat:+.3f}  fall_p {fall_p * 100:.0f}%",
             x, y + h + 46, 0.30, ACCENT)


def draw_buttons(img, held):
    """Only the keys this action space can press: the crosspad and B."""
    def btn(label, cx, cy, r):
        on = label in held
        cv2.circle(img, (cx, cy), r, ACCENT if on else GRID, -1)
        text(img, label, cx - 8 if len(label) < 2 else cx - 14, cy + 5, 0.36,
             BG if on else DIM)

    for lab, dx, dy in [("up", 0, -28), ("dn", 0, 28), ("lf", -28, 0), ("rt", 28, 0)]:
        btn(lab, 1074 + dx, 148 + dy, 14)
    btn("B", 1172, 148, 22)
    text(img, "crosspad          B", 1036, 200, 0.30, DIM)
    text(img, f"held: {', '.join(held) or 'no input'}", 1036, 222, 0.32, FG)
    text(img, "A, X, Y, L, R and Start are never", 1036, 248, 0.26, DIM)
    text(img, "pressed: the action space is 6 keys", 1036, 262, 0.26, DIM)


def draw_descent(img, x, y, w, h, st):
    hist = st["hist_height"][-600:]
    if len(hist) < 2:
        return
    v = np.asarray(hist, dtype=float)
    if st["track"] in ("ds8", "ds9"):
        target_y = 7145.0
        lo = min(float(v.min()), st["kill_plane"], -4800.0)
        hi = max(float(v.max()), target_y + 200.0)
        draw_series(img, x, y, w, h, v, ACCENT, lo, hi, 1)
        yt = int(y + h - (target_y - lo) / ((hi - lo) or 1.0) * h)
        cv2.line(img, (x, yt), (x + w, yt), GOOD, 1, cv2.LINE_AA)
        yk = int(y + h - (st["kill_plane"] - lo) / ((hi - lo) or 1.0) * h)
        cv2.line(img, (x, yk), (x + w, yk), BAD, 1, cv2.LINE_AA)
        text(img, f"height {v[-1]:+.1f} (star {target_y:.0f}, kill {st['kill_plane']:.0f})",
             x, y + h + 14, 0.28, DIM)
    else:
        lo, hi = min(float(v.min()), st["kill_plane"]), float(v.max())
        draw_series(img, x, y, w, h, v, ACCENT, lo, hi, 1)
        yk = int(y + h - (st["kill_plane"] - lo) / ((hi - lo) or 1.0) * h)
        cv2.line(img, (x, yk), (x + w, yk), BAD, 1, cv2.LINE_AA)
        text(img, f"height {v[-1]:+.2f} (kill plane {st['kill_plane']:.1f})", x, y + h + 14,
             0.30, DIM)


# ----------------------------------------------------------------- compose
def compose(game_bgr, st):
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    header_title = ("Reinforcement learning on Super Mario 64 DS (Tick Tock Clock)"
                    if st["track"] in ("ds8", "ds9")
                    else "Reinforcement learning on Super Mario 64 DS slide courses")
    text(img, header_title, 24, 30, 0.70, FG, 2)
    algo_lbl = st.get("algo_label", "PPO + NatureCNN")
    text(img, f"{st['track_name']}   ·   savestate {st['track']}   ·   {algo_lbl}, "
              f"deterministic, policy on {st['device']}", 24, 48, 0.42, DIM)
    text(img, os.path.basename(st["model"]), W - 340, 34, 0.38, ACCENT)
    text(img, f"decision {st['step']} / {st['max_steps']}", W - 340, 52, 0.38, DIM)

    gx, gy = 24, 66
    panel_box(img, gx - 8, gy - 8, 528, 400)
    img[gy:gy + 384, gx:gx + 512] = cv2.resize(game_bgr, (512, 384),
                                               interpolation=cv2.INTER_NEAREST)
    text(img, f"top DS screen 256x192 at 2x, 1 frame per emulated cycle "
              f"({st['cycles']} cycles so far)", gx, 490, 0.40, DIM)
    text(img, f"R = {st['total_reward']:+.2f}   survived {st['step']} / {st['max_steps']} "
              f"decisions   in-track falls {st['falls']}", gx, 512, 0.46, FG)

    if st.get("dist_reward_enabled"):
        star_str = f"  +  star {st.get('star_bonus', 0.0):+.1f}" if st.get("star_bonus", 0.0) > 0 else ""
        goal_str = f"  +  goal {st.get('goal_bonus', 0.0):+.1f}" if st.get("goal_bonus", 0.0) > 0 else ""
        text(img, f"R = dist {st.get('dist_term', 0.0):+.2f}  +  flow {st['flow_term']:+.2f}  "
                  f"-  step cost {st['step_penalty'] * st['step']:.2f}  -  death "
                  f"{st['deaths']} x {st['death_penalty']:g}{star_str}{goal_str}", gx, 534, 0.36, ACCENT)
    else:
        text(img, f"R = flow {st['flow_term']:+.2f}  -  step cost "
                  f"{st['step_penalty'] * st['step']:.2f}  -  death "
                  f"{st['deaths']} x {st['death_penalty']:g}", gx, 534, 0.38, ACCENT)

    text(img, f"action {ACTION_NAMES[st['action_idx']]}  "
              f"[{', '.join(st['held']) or 'no input'}]  "
              f"pi(chosen)={st['pi_chosen'] * 100:.1f}%", gx, 556, 0.38, DIM)

    x0 = 572
    panel_box(img, x0, 66, 440, 226, "what the agent sees (pixels only)")
    draw_agent_view(img, x0 + 12, 104, st)

    panel_box(img, x0, 302, 440, 160, "policy pi(a|s) now  +  mix of last 60")
    draw_probs(img, x0 + 12, 316, 416, 138, st["probs"], st["action_idx"], st["recent"])

    panel_box(img, x0, 472, 440, 144, "value and reward")
    draw_value(img, x0 + 12, 500, 416, 66, st)

    panel_box(img, 1024, 66, 248, 216, "buttons")
    draw_buttons(img, st["held"])

    panel_box(img, 1024, 294, 248, 224, "SciML / instrument")
    for i, (k, v) in enumerate(st["physics"]):
        yy = 320 + i * 24
        text(img, k, 1034, yy, 0.28, DIM)
        text(img, v, 1034, yy + 14, 0.28, FG)

    box_title = "vertical ascent" if st["track"] in ("ds8", "ds9") else "descent"
    panel_box(img, 1024, 528, 248, 88, box_title)
    draw_descent(img, 1034, 556, 228, 34, st)

    y = 640
    if st.get("dist_reward_enabled"):
        goal_desc = "target: Power Star (height 7145)" if st["track"] in ("ds8", "ds9") else "target: course terminus"
        text(img, f"horizon: {st['max_steps']} decisions; 3D proximity reward shaping enabled ({goal_desc})", 24, y - 8, 0.38, DIM)
    else:
        text(img, "horizon: 1350 decisions; the env has no finish-line detector, so a "
                  "timeout records survival, not a verified traversal", 24, y - 8, 0.38, DIM)
    frac = max(0.0, min(1.0, st["step"] / float(st["max_steps"])))
    cv2.rectangle(img, (24, y), (W - 24, y + 22), GRID, -1)
    cv2.rectangle(img, (24, y), (int(24 + (W - 48) * frac), y + 22), ACCENT, -1)
    text(img, f"{frac * 100:.1f}%", W - 92, y + 17, 0.48, BG if frac > 0.06 else FG, 2)
    if st["outcome"]:
        is_success = bool(st["completed"] or "STAR" in st["outcome"] or "GOAL" in st["outcome"])
        text(img, st["outcome"], 24, y + 50, 0.52, GOOD if is_success else BAD, 2)
    if st["completed"]:
        text(img, f"{st['stationary']} of {st['max_steps']} decisions had |speed| < 0.5 "
                  f"(the flow reward pays for sliding, not for arriving)",
             640, y + 50, 0.36, DIM)
    return img


# ----------------------------------------------------------------- inference
def load_policy_model(model_path, device):
    """Load either a PPO or QR-DQN policy and detect its feature extractor."""
    import zipfile
    from stable_baselines3 import PPO

    is_qrdqn = False
    try:
        with zipfile.ZipFile(model_path, "r") as z:
            if "data" in z.namelist():
                raw = z.read("data").decode("utf-8", errors="ignore")
                if "QRDQN" in raw or "qrdqn" in raw:
                    is_qrdqn = True
    except Exception:
        pass

    if is_qrdqn:
        from sb3_contrib import QRDQN
        model = QRDQN.load(model_path, device=device)
        algo_name = "QR-DQN"
    else:
        try:
            model = PPO.load(model_path, device=device)
            algo_name = "PPO"
        except Exception:
            from sb3_contrib import QRDQN
            model = QRDQN.load(model_path, device=device)
            algo_name = "QR-DQN"

    feat_name = "NatureCNN"
    ext = getattr(model.policy, "features_extractor", None)
    if ext is None and hasattr(model.policy, "quantile_net"):
        ext = getattr(model.policy.quantile_net, "features_extractor", None)
    if ext is not None:
        cls_name = ext.__class__.__name__
        if "Impala" in cls_name:
            feat_name = "IMPALA CNN"
        elif "Nature" in cls_name:
            feat_name = "NatureCNN"

    return model, f"{algo_name} + {feat_name}"


def policy_head(model, stack):
    """pi(a|s) and V(s) from the loaded policy (PPO or QR-DQN), on a tensor built
    by obs_to_tensor() -- the same conversion predict() applies, so the numbers describe
    the decision the agent actually makes rather than a re-implementation of preprocessing."""
    import torch
    pol = model.policy
    with torch.no_grad():
        obs_t = pol.obs_to_tensor(stack)[0]
        if hasattr(pol, "quantile_net"):
            quantiles = pol.quantile_net(obs_t)
            q_values = quantiles.mean(dim=1)
            mode = int(torch.argmax(q_values, dim=1)[0].item())
            value = float(q_values.max().item())
            probs_t = torch.softmax(q_values, dim=1)[0]
            probs = probs_t.detach().float().cpu().numpy()
            ent = float(-(probs_t * torch.log(probs_t.clamp(min=1e-8))).sum().item())
            return probs, value, ent, mode
        else:
            dist = pol.get_distribution(obs_t)
            probs = dist.distribution.probs[0].detach().float().cpu().numpy()
            ent = float(dist.distribution.entropy()[0].item())
            mode = int(dist.mode()[0].item())
            value = float(pol.predict_values(obs_t).reshape(-1)[0])
            return probs, value, ent, mode


def load_state_model(track, device):
    """Optional RAM dynamics model (docs/WORLD_MODEL.md section 10): its r_hat and
    fall probability for the state the dashboard is showing."""
    path = os.path.join(ROOT_DIR, "models", "state_wm.pt")
    if not os.path.exists(path) or track not in TRACKS:
        return None
    try:
        import torch
        from src.state_model import StateDynamicsModel, STATE_COLS
        ck = torch.load(path, map_location=device, weights_only=True)
        m = StateDynamicsModel()
        m.load_state_dict(ck[track]["model"])
        m.eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        n = ck[track]["norm"]
        return {"m": m, "mu": np.asarray(n["mu"], float), "sd": np.asarray(n["sd"], float),
                "rmu": float(n["rmu"]), "rsd": float(n["rsd"]), "cols": STATE_COLS,
                "torch": torch, "device": device}
    except Exception as e:
        print(f"[dash] RAM dynamics model unavailable ({type(e).__name__}: {str(e)[:90]})",
              flush=True)
        return None


def state_model_pred(sm, ram_raw, action):
    if sm is None:
        return None
    torch = sm["torch"]
    s = np.asarray(ram_raw, dtype=np.float64)[list(sm["cols"])]
    sn = torch.tensor((s - sm["mu"]) / sm["sd"], dtype=torch.float32,
                      device=sm["device"])[None]
    an = torch.tensor([int(action)], dtype=torch.long, device=sm["device"])
    with torch.no_grad():
        _, fl, _, rw, _ = sm["m"](sn, an)
    return float(rw.item() * sm["rsd"] + sm["rmu"]), float(torch.sigmoid(fl).item())


# ------------------------------------------------------------------ record
def grab_bgr(emu):
    """py-desmume's display buffer is BGR-native despite the method name
    (see src/env.py); the dashboard writes through cv2, which wants BGR."""
    buf = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8).reshape((384, 256, 4))
    return np.ascontiguousarray(buf[:192, :, :3])


def _record_single(args, out_dir_full):
    import torch
    from src.env import Mario64DSEnv

    track = args.single
    rom = (args.rom if os.path.isabs(args.rom) else os.path.join(ROOT_DIR, args.rom))
    model_path = (args.model if os.path.isabs(args.model)
                  else os.path.join(ROOT_DIR, args.model))
    out_mp4 = os.path.join(out_dir_full, f"dashboard_{track}.mp4")

    # One worker = one tiny 84x84 policy forward. Left alone, torch gives every
    # process all 32 cores: three parallel workers then took 144.8 s for what
    # sequential did in 24.3 s (measured, out/supervisor_parity.json).
    torch.set_num_threads(max(1, int(args.torch_threads)))
    model, algo_label = load_policy_model(model_path, device=args.device)
    device = str(next(model.policy.parameters()).device)
    print(f"[dash] {track}: policy device {device} ({algo_label})", flush=True)
    sm = load_state_model(track, device) if not args.no_state_model else None

    dist_reward = args.dist_reward if args.dist_reward is not None else (track in ("ds8", "ds9"))
    use_ram_death = args.use_ram_death if args.use_ram_death is not None else (track in ("ds8", "ds9"))
    flow_weight = args.flow_weight
    if track in ("ds8", "ds9") and args.flow_weight == 1.0:
        flow_weight = 0.2

    env = Mario64DSEnv(rom_path=rom, state_path=STATE_PATH[track], max_steps=args.max_steps,
                       frameskip=args.frameskip, flow_weight=flow_weight,
                       step_penalty=args.step_penalty, death_penalty=args.death_penalty,
                       timeout_penalty=args.timeout_penalty,
                       dist_reward_enabled=dist_reward, dist_weight=args.dist_weight,
                       goal_reward=args.goal_reward, star_reward=args.star_reward,
                       use_ram_death=use_ram_death,
                       action_set={6: "legacy6", 12: "compound12"}[int(model.action_space.n)],
                       dist_reward_mode=args.dist_mode,
                       route_path=os.path.join(ROOT_DIR, args.route) if args.route else None)
    # One track per process (see _job_cmd), so the panel labels can follow the
    # checkpoint's action set without threading it through every draw call.
    global ACTION_NAMES, ACTION_BUTTONS
    ACTION_NAMES, ACTION_BUTTONS = env.action_names, env.action_buttons
    obs, info = env.reset()

    # SB3 VecFrameStack convention: the stack opens as [0, 0, 0, reset_frame],
    # not four copies of it -- replicating that keeps the trajectory identical to
    # src/eval.py's SubprocVecEnv + VecFrameStack(n_stack=4) + VecTransposeImage.
    first = obs[:, :, 0]
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)

    def make_stack():
        return np.stack(list(dq), axis=0)[np.newaxis, :]

    tmp_avi = os.path.join(ROOT_DIR, "out", f"dashboard_{track}_tmp.avi")
    os.makedirs(os.path.dirname(tmp_avi), exist_ok=True)
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (W, H))
    if not writer.isOpened():
        print(f"[dash] {track}: VideoWriter could not open {tmp_avi}", file=sys.stderr)
        return 2

    cycle_frames = []
    env.on_cycle = lambda: cycle_frames.append(grab_bgr(env.emu))

    hist_v, hist_r, hist_wm, hist_h, hist_speed = [], [], [], [], []
    trace = []
    recent, probs, value = [], None, float("nan")
    units = np.zeros(len(NAMES))
    total_reward = 0.0
    flow_sum_raw = 0.0     # weight-free sum of clip(mean flow_y, 0): re-prices R
    flow_term = 0.0        # the term actually added, at this run's weight
    dist_term = 0.0
    goal_bonus = 0.0
    star_bonus = 0.0
    deaths = falls = 0
    stationary = 0
    pi_mismatch = 0
    cycles = 0
    frames_written = 0
    step = 0
    done = trunc = False
    ram_prev = info.get("ram")
    t0 = time.time()
    wm_pred = state_model_pred(sm, ram_prev, 0) if sm else None

    st = {"track": track, "track_name": TRACK_NAMES[track], "model": model_path,
          "device": device, "algo_label": algo_label,
          "max_steps": args.max_steps, "flow_weight": flow_weight,
          "step_penalty": args.step_penalty, "death_penalty": args.death_penalty,
          "kill_plane": KILL_PLANE / SCALE, "physics": [], "probs": None,
          "obs84": first, "flow": None, "flow_win": None, "flow_reward": 0.0,
          "step": 0, "cycles": 0, "held": [], "action_idx": 0, "pi_chosen": 0.0,
          "total_reward": 0.0, "flow_term": 0.0, "falls": 0, "deaths": 0,
          "dist_reward_enabled": dist_reward, "dist_term": 0.0,
          "goal_bonus": 0.0, "star_bonus": 0.0,
          "stationary": 0,
          "value": float("nan"), "step_reward": 0.0, "wm_pred": wm_pred,
          "hist_v": hist_v, "hist_r": hist_r, "hist_wm": hist_wm,
          "hist_height": hist_h, "recent": recent, "outcome": "", "completed": False}

    # one decision frame before the first step so the clip opens with panels
    game = grab_bgr(env.emu)
    st["physics"] = physics_rows(env, NAMES, KILL_PLANE, SCALE, ram_prev, 0.0, 0.0)
    writer.write(compose(game, st))
    frames_written += 1

    while not (done or trunc) and step < args.max_steps:
        cycle_frames.clear()
        action, _ = model.predict(make_stack(), deterministic=True)
        a = int(action[0])
        probs, value, ent, mode = policy_head(model, make_stack())
        # the bar is labelled as the decision, so prove it: the distribution's own
        # mode must be the action deterministic predict() returned.
        if mode != a:
            pi_mismatch += 1
        wm_pred = state_model_pred(sm, ram_prev, a) if sm else None

        obs, reward, done, trunc, info = env.step(a)
        dq.append(obs[:, :, 0])
        cycles += env.frameskip
        step += 1
        total_reward += float(reward)

        ram = info.get("ram")
        fell = bool(ram is not None and Sm64DsState.is_dead(ram)
                    and (ram_prev is None or not Sm64DsState.is_dead(ram_prev)))
        if fell:
            falls += 1
        if done:
            deaths += 1
        flow_sum_raw += env.last_flow_mean if env.last_flow_mean > 0 else 0.0
        flow_term += env.last_flow_reward
        dist_term += getattr(env, "last_dist_reward", 0.0)
        if info.get("star_collected"):
            star_bonus += args.star_reward
        if info.get("reached_goal"):
            goal_bonus += args.goal_reward

        units = Sm64DsState.units(ram) if ram is not None else np.zeros(6)
        if abs(float(units[1])) < 0.5:
            stationary += 1
        if ram is not None:
            ram_prev = np.asarray(ram, dtype=np.float64)
        trace.append([step, a, round(float(units[1]), 3), round(float(units[3]), 3),
                      round(float(units[4]), 3), round(float(units[5]), 3),
                      round(float(env.last_flow_mean), 5), round(float(reward), 5),
                      round(total_reward, 4), 1 if fell else 0])
        hist_v.append(value)
        hist_r.append(float(reward))
        hist_wm.append(wm_pred[0] if wm_pred else float("nan"))
        hist_h.append(float(units[4]))
        hist_speed.append(float(units[1]))
        recent.append(a)
        if len(recent) > WINDOW:
            recent.pop(0)

        completed = bool(trunc) and step >= args.max_steps
        if info.get("star_collected"):
            outcome = f"STAR COLLECTED at decision {step}! (+{args.star_reward:g})"
        elif info.get("reached_goal"):
            outcome = f"GOAL REACHED at decision {step}! (+{args.goal_reward:g})"
        elif completed:
            outcome = (f"TIMEOUT at decision {step} - survived 1350 horizon" if dist_reward
                       else f"TIMEOUT at decision {step} - the agent did not die; the env has no finish-line detector")
        elif done:
            outcome = f"DEATH at decision {step} - {'RAM kill plane' if use_ram_death else 'black screen'}, -{args.death_penalty:g}"
        else:
            outcome = ""

        st.update({
            "step": step, "cycles": cycles, "held": ACTION_BUTTONS[a], "action_idx": a,
            "probs": probs, "pi_chosen": float(probs[a]), "value": value,
            "step_reward": float(reward), "total_reward": total_reward,
            "flow_term": flow_term, "flow_reward": env.last_flow_reward,
            "dist_term": dist_term, "goal_bonus": goal_bonus, "star_bonus": star_bonus,
            "flow": env.last_flow, "obs84": obs[:, :, 0],
            "flow_win": (cv2.resize(obs[:, :, 0], (32, 32), interpolation=cv2.INTER_AREA)
                         if env.last_flow is not None else None),
            "falls": falls, "deaths": deaths, "wm_pred": wm_pred,
            "stationary": stationary,
            "physics": physics_rows(env, NAMES, KILL_PLANE, SCALE, ram,
                                    total_reward, flow_sum_raw),
            "completed": completed,
            "outcome": outcome,
        })

        seq = cycle_frames or [grab_bgr(env.emu)]
        for cf in seq:
            writer.write(compose(cf, st))
            frames_written += 1
        if step % args.snapshot_every == 0 or done or trunc:
            snap = os.path.join(out_dir_full, "frames", f"{track}_step{step:04d}.png")
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            cv2.imwrite(snap, compose(seq[-1], st))

    env.close()
    writer.release()

    # the identity the decomposition line claims: R == flow terms - step cost - deaths
    recomputed = flow_term + dist_term + goal_bonus + star_bonus \
        - args.step_penalty * step - args.death_penalty * deaths \
        - args.timeout_penalty * (1 if trunc else 0)
    resid = abs(recomputed - total_reward)
    priced = {}
    for w in (1.0, 2.5):
        priced[f"{w:g}"] = round(dist_term + goal_bonus + star_bonus
                                 - args.step_penalty * step + w * flow_sum_raw
                                 - args.death_penalty * deaths
                                 - args.timeout_penalty * (1 if trunc else 0), 4)

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print(f"[dash] {track}: ffmpeg not found, leaving {tmp_avi}", file=sys.stderr)
        return 3
    os.makedirs(out_dir_full, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp_avi, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
                    out_mp4], check=True)
    os.remove(tmp_avi)

    payload = {
        "track": track, "track_name": TRACK_NAMES[track],
        "model": os.path.basename(model_path), "video": os.path.basename(out_mp4),
        "steps": step, "max_steps": args.max_steps,
        "completed": bool(trunc and step >= args.max_steps),
        "star_collected": bool(star_bonus > 0),
        "reached_goal": bool(goal_bonus > 0),
        "died": bool(done), "deaths": deaths, "in_track_falls": falls,
        "reward_recorded": round(total_reward, 4),
        "flow_weight": flow_weight, "step_penalty": args.step_penalty,
        "death_penalty": args.death_penalty, "timeout_penalty": args.timeout_penalty,
        "dist_reward_enabled": dist_reward, "dist_weight": args.dist_weight,
        "dist_term_sum": round(dist_term, 6), "goal_bonus": round(goal_bonus, 2),
        "star_bonus": round(star_bonus, 2),
        "frameskip": args.frameskip,
        "flow_sum_raw": round(flow_sum_raw, 6), "flow_term_sum": round(flow_term, 6),
        "reward_at_flow_weight": priced, "decomposition_residual": round(resid, 9),
        "final_ram_units": [round(float(v), 3) for v in units],
        "min_height_units": round(min(hist_h), 3) if hist_h else None,
        "max_speed_units": round(max(hist_speed), 3) if hist_speed else None,
        "frames_written": frames_written, "emulator_cycles": cycles,
        "seconds": round(frames_written / args.fps, 2),
        "wall_seconds": round(time.time() - t0, 1),
        "width": W, "height": H, "fps": args.fps,
        "supervisor": args.supervisor, "state_model": bool(sm),
        "policy_device": device,
        "pi_argmax_mismatches": pi_mismatch,
    }
    if pi_mismatch:
        print(f"[dash] {track}: WARNING predict() != argmax(pi) on {pi_mismatch} "
              f"decisions; the probability bars do not describe the executed action",
              file=sys.stderr, flush=True)
    with open(os.path.join(out_dir_full, f"dashboard_{track}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    stationary = sum(1 for r in trace if abs(r[2]) < 0.5)
    last_fall = max((r[0] for r in trace if r[9]), default=None)
    stats = {
        "columns": ["step", "action", "speed", "lateral", "height", "vvel",
                    "flow_mean", "reward", "cum_reward", "fell"],
        "units": "track units (20.12 / 4096); flow_mean in px per decision",
        "track": track, "steps": len(trace),
        "stationary_decisions_speed_lt_0p5": stationary,
        "last_fall_step": last_fall,
        "decisions_after_last_fall": (None if last_fall is None
                                      else len(trace) - last_fall),
        "zero_flow_decisions": sum(1 for r in trace if r[6] <= 0.0),
        "rows": trace,
    }
    with open(os.path.join(out_dir_full, f"dashboard_{track}_trace.json"), "w") as f:
        json.dump(stats, f)
    payload["stationary_decisions"] = stationary
    payload["last_fall_step"] = last_fall
    payload["decisions_after_last_fall"] = stats["decisions_after_last_fall"]
    payload["zero_flow_decisions"] = stats["zero_flow_decisions"]
    with open(os.path.join(out_dir_full, f"dashboard_{track}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[dash] {track}: R={total_reward:+.2f} steps={step}/{args.max_steps} "
          f"completed={payload['completed']} frames={frames_written} "
          f"({payload['seconds']}s) resid={resid:.1e} -> {out_mp4}", flush=True)
    print("RESULT\t" + json.dumps(payload), flush=True)
    return 0


def physics_rows(env, names, kill_plane, scale, ram, total_reward, flow_sum):
    """Panel content read off the env and src/ram_state.py, so the frame cannot
    drift from the instrument that actually scored the run. Eight rows: the
    SciML panel fits exactly this many at 24 px."""
    u = Sm64DsState.units(ram) if ram is not None else np.zeros(len(names))
    if getattr(env, "dist_reward_enabled", False):
        last_d = f"{env.last_dist:.1f}" if env.last_dist is not None else "n/a"
        row7 = (f"dist to goal: {last_d}", f"weight x{env.dist_weight:g}")
        row8 = ("terminal rewards", f"star +{env.star_reward:g}, goal +{env.goal_reward:g}")
    else:
        row7 = ("reward terms", f"flow x{env.flow_weight:g}, step -{env.step_penalty:g}, "
                                f"death -{env.death_penalty:g}")
        row8 = ("flow sum at x1.0", f"{flow_sum:+.2f}")

    return [
        ("RAM struct 20.12 fixed point", "0x0217F74C..764, /4096"),
        (f"{names[1]} (track units)", f"{u[1]:+.2f}   cap 100.0"),
        (f"{names[3]}", f"{u[3]:+.2f}"),
        (f"{names[4]}", f"{u[4]:+.2f}   kill {kill_plane / scale:.1f}"),
        (f"{names[5]}", f"{u[5]:+.2f}"),
        ("decisions per second", f"{60.0 / env.frameskip:.0f} (frameskip "
                                 f"{env.frameskip})"),
        row7,
        row8,
    ]


def find_ffmpeg():
    """winget installs ffmpeg outside PATH on this machine, so search for it.
    The package layout is Packages/<Gyan.FFmpeg_...>/<ffmpeg-x.y-full_build>/bin/,
    which a fixed-depth glob misses, hence the bounded walk."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    bases = [os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
             r"C:\ffmpeg", os.path.expandvars(r"%USERPROFILE%\scoop\apps")]
    for base in bases:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if root.count(os.sep) - base.count(os.sep) > 4:
                del _dirs[:]
                continue
            for f in files:
                if f.lower() == "ffmpeg.exe" and os.path.exists(os.path.join(root, f)):
                    return os.path.join(root, f)
    return None


# ------------------------------------------------------------- orchestration
def _job_cmd(args, track):
    cmd = [sys.executable, os.path.abspath(__file__),
           "--model", args.model, "--rom", args.rom, "--out-dir", args.out_dir,
           "--fps", str(args.fps), "--max-steps", str(args.max_steps),
           "--flow-weight", str(args.flow_weight), "--step-penalty", str(args.step_penalty),
           "--death-penalty", str(args.death_penalty),
           "--timeout-penalty", str(args.timeout_penalty),
           "--frameskip", str(args.frameskip), "--device", args.device,
           "--torch-threads", str(args.torch_threads),
           "--dist-weight", str(args.dist_weight),
           "--goal-reward", str(args.goal_reward),
           "--star-reward", str(args.star_reward),
           "--snapshot-every", str(args.snapshot_every),
           "--dist-mode", args.dist_mode,
           "--supervisor", args.supervisor_label]
    if args.route:
        cmd += ["--route", args.route]
    if args.dist_reward is True:
        cmd.append("--dist-reward")
    elif args.dist_reward is False:
        cmd.append("--no-dist-reward")
    if args.use_ram_death is True:
        cmd.append("--use-ram-death")
    elif args.use_ram_death is False:
        cmd.append("--no-ram-death")
    if args.no_state_model:
        cmd.append("--no-state-model")
    cmd += ["--single", track]
    return cmd


def _parse_stream(lines, results):
    for ln in lines:
        print(ln, flush=True)
        if ln.startswith("RESULT\t"):
            try:
                results.append(json.loads(ln[len("RESULT\t"):]))
            except json.JSONDecodeError:
                print("[dash] unparseable RESULT line", file=sys.stderr, flush=True)


def _run_python(args, out_dir_full, results):
    for track in args.track_list:
        print(f"\n=== [{args.supervisor_label}] recording {track} ===", flush=True)
        p = subprocess.Popen(_job_cmd(args, track), cwd=ROOT_DIR,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        try:
            out, _ = p.communicate(timeout=args.run_budget)
            code = p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            code = "killed:over-budget"
        _parse_stream((out or "").splitlines(), results)
        if code != 0:
            print(f"[dash] {track} worker exit={code}", file=sys.stderr, flush=True)


def _run_rust(args, out_dir_full, results):
    jobfile = os.path.join(ROOT_DIR, "out", "dashboard_jobs.tsv")
    os.makedirs(os.path.dirname(jobfile), exist_ok=True)
    with open(jobfile, "w", encoding="utf-8") as f:
        for track in args.track_list:
            cmd = _job_cmd(args, track)
            f.write("\t".join([track, str(args.run_budget), cmd[0]] + cmd[1:]) + "\n")
    print(f"\n=== [{args.supervisor_label}] {len(args.track_list)} jobs, "
          f"budget {args.run_budget}s each ===", flush=True)
    cmd = [RUST_SUP, jobfile, os.path.join(ROOT_DIR, "out", "supervisor_log.json")]
    if args.parallel > 1:
        cmd += ["--parallel", str(args.parallel)]
    p = subprocess.Popen(cmd,
                         cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    assert p.stdout is not None
    _parse_stream(p.stdout, results)
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/curriculum_flow25_r3_best.zip")
    ap.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    ap.add_argument("--tracks", default=",".join(DEFAULT_TRACKS))
    ap.add_argument("--out-dir", default="videos/dashboard")
    ap.add_argument("--fps", type=int, default=60, help="1 frame per emulated cycle")
    ap.add_argument("--max-steps", type=int, default=1350)
    ap.add_argument("--flow-weight", type=float, default=1.0,
                    help="1.0 = the instrument src/eval.py uses; 2.5 = training weight")
    ap.add_argument("--step-penalty", type=float, default=0.02)
    ap.add_argument("--death-penalty", type=float, default=100.0)
    ap.add_argument("--timeout-penalty", type=float, default=0.0)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--device", default="cpu",
                    help="torch device for the policy. 'cpu' is the instrument that "
                         "reproduces the published benchmark; 'auto' picks the GPU, "
                         "whose arithmetic flips ds3 (see tools/probe_device_"
                         "divergence.py)")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="intra-op threads per worker; the default (all cores) makes "
                         "parallel workers slower than sequential ones")
    ap.add_argument("--snapshot-every", type=int, default=300)
    ap.add_argument("--dist-reward", dest="dist_reward", action="store_true", default=None,
                    help="Enable 3D distance potential reward shaping")
    ap.add_argument("--no-dist-reward", dest="dist_reward", action="store_false",
                    help="Force disable 3D distance potential reward shaping")
    ap.add_argument("--dist-weight", type=float, default=0.05,
                    help="Weight for 3D potential distance shaping (default: 0.05)")
    ap.add_argument("--goal-reward", type=float, default=200.0,
                    help="Terminal bonus for reaching goal coordinates")
    ap.add_argument("--star-reward", type=float, default=500.0,
                    help="Terminal bonus for star collection")
    ap.add_argument("--use-ram-death", dest="use_ram_death", action="store_true", default=None,
                    help="Enable instantaneous RAM-based kill-plane detection")
    ap.add_argument("--no-ram-death", dest="use_ram_death", action="store_false",
                    help="Force disable instantaneous RAM-based kill-plane detection")
    ap.add_argument("--run-budget", type=float, default=1800.0,
                    help="wall-clock seconds allowed per track before it is killed")
    ap.add_argument("--parallel", type=int, default=1,
                    help="jobs the Rust supervisor may run at once (each is its own "
                         "process, so this is safe; two emulators in ONE process is not)")
    ap.add_argument("--supervisor", choices=["auto", "rust", "python"], default="auto")
    ap.add_argument("--no-state-model", action="store_true")
    ap.add_argument("--dist-mode", default="progress", choices=["progress", "potential", "route"],
                    help="distance-reward instrument (route needs --route)")
    ap.add_argument("--route", default=None, help="route JSON for --dist-mode route")
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gif", default=None, help="inline README preview for one track")
    ap.add_argument("--gif-track", default=None,
                    help="which recorded track the preview comes from (default: last)")
    ap.add_argument("--gif-len", type=float, default=10.0)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-width", type=int, default=960)
    args = ap.parse_args()

    args.track_list = [t.strip() for t in args.tracks.split(",") if t.strip()]
    for t in args.track_list:
        if t not in TRACKS:
            ap.error(f"unknown track {t!r} (have {TRACKS})")

    out_dir_full = (args.out_dir if os.path.isabs(args.out_dir)
                    else os.path.join(ROOT_DIR, args.out_dir))

    if args.single is not None:
        os.makedirs(out_dir_full, exist_ok=True)
        sys.exit(_record_single(args, out_dir_full))

    use = args.supervisor
    if use == "auto":
        use = "rust" if os.path.exists(RUST_SUP) else "python"
    args.supervisor_label = use
    results = []
    if use == "rust":
        _run_rust(args, out_dir_full, results)
    else:
        _run_python(args, out_dir_full, results)

    os.makedirs(out_dir_full, exist_ok=True)
    summary = {"supervisor": use, "supervisor_binary": RUST_SUP if use == "rust" else None,
               "parallel": args.parallel if use == "rust" else 1,
               "model": os.path.basename(args.model), "flow_weight": args.flow_weight,
               "max_steps": args.max_steps, "frameskip": args.frameskip,
               "tracks": {r["track"]: r for r in results},
               "completed": sorted(r["track"] for r in results if r["completed"]),
               "died": sorted(r["track"] for r in results if r["died"])}
    with open(os.path.join(out_dir_full, "dashboard_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[dash] {use}: {len(results)}/{len(args.track_list)} tracks reported; "
          f"completed {summary['completed'] or 'none'}", flush=True)

    if args.gif and results:
        want = args.gif_track or results[-1]["track"]
        payload = next((r for r in results if r["track"] == want), None)
        if payload is None:
            print(f"[dash] --gif-track {want} not in this batch "
                  f"({[r['track'] for r in results]})", file=sys.stderr, flush=True)
        else:
            last = os.path.join(out_dir_full, f"dashboard_{want}.mp4")
            if build_gif(args, last, payload):
                summary["gif"] = {"path": os.path.basename(args.gif), "track": want}
                json.dump(summary, open(os.path.join(out_dir_full,
                                "dashboard_summary.json"), "w", encoding="utf-8"),
                          indent=2)
    return 0


def build_gif(args, mp4, payload):
    ffmpeg = find_ffmpeg()
    if ffmpeg is None or not os.path.exists(mp4):
        print("[dash] gif skipped (no ffmpeg or no mp4)", file=sys.stderr, flush=True)
        return False
    clip_seconds = payload["frames_written"] / args.fps
    gstart = max(0.0, clip_seconds - args.gif_len)
    filt = f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos"
    out_dir = os.path.dirname(mp4)
    pal = os.path.join(out_dir, "gif_palette.png")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(gstart),
                    "-t", str(args.gif_len), "-i", mp4,
                    "-vf", filt + ",palettegen=stats_mode=diff", pal], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(gstart),
                    "-t", str(args.gif_len), "-i", mp4, "-i", pal,
                    "-lavfi", f"{filt}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    args.gif], check=True)
    print(f"[dash] wrote {args.gif} ({os.path.getsize(args.gif) / 1e6:.1f} MB, "
          f"{args.gif_len}s from t={gstart:.2f} of {clip_seconds:.2f}s)", flush=True)
    return True


if __name__ == "__main__":
    sys.exit(main())
