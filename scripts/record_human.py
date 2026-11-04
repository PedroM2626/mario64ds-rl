"""Interactive human recorder for Super Mario 64 DS (Tick Tock Clock by default).

Inspired by the smg-rl human recorder. Two input modes, both applied through
Mario64DSEnv.step so observations and rewards come from the training code:

  --input free (default)  every emulated frame (60 Hz) applies whatever buttons
      are held, any combination. Each frame's button mask is stored, so the run
      replays exactly (scripts/prepare_human_route.py checks it). Decisions
      whose 4 frames held one constant combination that an action of the set
      expresses exactly are "exact" and can be used as replay-buffer demos
      (src/demos.py); the rest still count for the route and checkpoints.
  --input agent  input sampled once per decision and mapped to the action set
      before it is applied (every decision is exact, but presses closer than
      4 frames merge: Up and B starting on the same frame do not jump).

Controls (only while the recorder window has focus; otherwise the game pauses):
    Arrows or WASD   D-pad              Space or K   B (jump)
    Shift or J       Y (run / punch)    Ctrl or C    R (crouch; crouch + jump = backflip)
    E or U           A (punch)          Q            L (camera behind Mario)
    X                X (camera)
    Enter            start               F5           restart attempt (discarded)
    F8               save the current attempt as a partial demo and stop
    F9 / Esc         stop the session without saving the current attempt

A death discards the attempt and reloads the start state. The RAM star event
(TTC) or the slide finish (src/env.py: end area of TRACK_GOALS) promotes the
attempt to completed/. Every --checkpoint-every
decisions a savestate is written, which prepare_human_route.py turns into a
start-state curriculum.

Usage:
    python scripts/record_human.py                 # ds8, free input, compound12 labels
    python scripts/record_human.py --input agent --action-set legacy6
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.demos import sha256_file  # noqa: E402
from src.env import ACTION_SETS, FREE_BUTTONS, buttons_to_mask, mask_to_buttons  # noqa: E402
from src.envs import rom_path, savestate_path  # noqa: E402

WINDOW = "SM64DS RL - Human recorder"
FPS = 60
SCALE = 3
RECORDER_VERSION = 2

VK = {"left": (0x25, 0x41), "up": (0x26, 0x57), "right": (0x27, 0x44), "down": (0x28, 0x53),
      "B": (0x20, 0x4B), "Y": (0x10, 0x4A), "R": (0x11, 0x43), "A": (0x45, 0x55),
      "L": (0x51,), "X": (0x58,)}
VK_ENTER, VK_F5, VK_F8, VK_F9, VK_ESC = 0x0D, 0x74, 0x77, 0x78, 0x1B


def repo_rel(path):
    """Repo-relative POSIX path when inside the repo, else absolute (other drive)."""
    try:
        rel = os.path.relpath(path, ROOT_DIR)
    except ValueError:
        return os.path.abspath(path)
    return os.path.abspath(path) if rel.startswith("..") else rel.replace(os.sep, "/")


def keys_to_action(pressed, action_buttons):
    """Best action of the set for the held buttons -> (index, exact).

    Opposite directions cancel. Among actions whose buttons are all held, the
    one with the most buttons wins (lowest index on ties), so an unrepresentable
    combination degrades to its largest representable part, never to an extra
    button the human did not press.
    """
    held = set(pressed)
    for a, b in (("left", "right"), ("up", "down")):
        if a in held and b in held:
            held -= {a, b}
    best, best_n = 0, -1
    for i, buttons in enumerate(action_buttons):
        if set(buttons) <= held and len(buttons) > best_n:
            best, best_n = i, len(buttons)
    return best, set(action_buttons[best]) == set(pressed)


def label_window(frame_masks, action_buttons):
    """Action label of one decision window -> (index, exact).

    Exact only when every frame held the same buttons and one action holds
    exactly those: then the agent taking that action reproduces the window.
    """
    action, exact = keys_to_action(mask_to_buttons(frame_masks[-1]), action_buttons)
    return action, exact and len(set(frame_masks)) == 1


def find_ffmpeg():
    try:
        from scripts.record_dashboard import find_ffmpeg as _find
        return _find()
    except Exception:
        return shutil.which("ffmpeg")


class VideoWriter:
    """60 Hz MP4 of the raw top screen through an ffmpeg pipe on a worker thread."""

    def __init__(self, path, w=256, h=192):
        exe = find_ffmpeg()
        self.path, self.frames, self.error = path, 0, None
        self.proc = None
        if exe is None:
            print("[rec] ffmpeg not found: recording without gameplay.mp4", flush=True)
            return
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0", "-an",
               "-vf", "scale=iw*2:ih*2:flags=neighbor", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.q = queue.Queue(maxsize=120)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        try:
            while (item := self.q.get()) is not None:
                self.proc.stdin.write(item)
        except Exception as exc:  # noqa: BLE001 - surfaced by close()
            self.error = exc
        finally:
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    def write(self, rgb):
        if self.proc is None:
            return
        self.q.put(np.ascontiguousarray(rgb).tobytes())  # blocks (pauses the game) if ffmpeg lags
        self.frames += 1

    def close(self):
        if self.proc is None:
            return False
        self.q.put(None)
        self.worker.join(timeout=120)
        code = self.proc.wait(timeout=120)
        if code or self.error is not None:
            raise RuntimeError(f"ffmpeg failed: {self.proc.stderr.read().decode(errors='replace')}")
        return True

    def abort(self):
        if self.proc is None:
            return
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.proc.kill()
        self.proc.wait()


class Attempt:
    def __init__(self, root, number, env):
        self.path = os.path.join(root, f"attempt-{number:06d}")
        os.makedirs(os.path.join(self.path, "checkpoints"))
        open(os.path.join(self.path, ".recorder-owned"), "w").close()
        self.video = VideoWriter(os.path.join(self.path, "gameplay.mp4"))
        self.obs = [env._get_obs()[..., 0].copy()]
        self.pos = [env.ram.get_pos_3d().copy()]
        self.actions, self.frame_masks, self.exact, self.rewards = [], [], [], []
        self.dist_reward, self.flow_reward, self.dist_to_goal = [], [], []
        self.checkpoints = []

    def record(self, action, frame_masks, exact, obs, reward, info, pos):
        self.actions.append(action)
        self.frame_masks.extend(frame_masks)
        self.exact.append(exact)
        self.rewards.append(reward)
        self.dist_reward.append(info.get("dist_reward") or 0.0)
        self.flow_reward.append(info.get("flow_reward") or 0.0)
        self.dist_to_goal.append(np.nan if info.get("dist_to_goal") is None else info["dist_to_goal"])
        self.obs.append(obs[..., 0].copy())
        self.pos.append(pos.copy())

    def checkpoint(self, env):
        step = len(self.actions)
        name = f"step-{step:06d}.dst"
        p = os.path.join(self.path, "checkpoints", name)
        env.emu.savestate.save_file(p)
        self.checkpoints.append({"file": f"checkpoints/{name}", "step": step,
                                 "pos": [round(float(v), 3) for v in self.pos[-1]]})

    def discard(self):
        self.video.abort()
        # Delete only a folder this recorder created.
        if (os.path.basename(self.path).startswith("attempt-")
                and os.path.isfile(os.path.join(self.path, ".recorder-owned"))):
            shutil.rmtree(self.path)

    def promote(self, dest_root, state, env, input_mode, source_state, source_hash, rom_hash, info):
        has_video = self.video.close()
        np.savez_compressed(
            os.path.join(self.path, "demo.npz"),
            obs=np.stack(self.obs).astype(np.uint8), actions=np.asarray(self.actions, np.int16),
            frame_masks=np.asarray(self.frame_masks, np.uint16), exact=np.asarray(self.exact, bool),
            rewards=np.asarray(self.rewards, np.float32), positions=np.stack(self.pos).astype(np.float64),
            dist_reward=np.asarray(self.dist_reward, np.float32),
            flow_reward=np.asarray(self.flow_reward, np.float32),
            dist_to_goal=np.asarray(self.dist_to_goal, np.float32))
        with open(os.path.join(self.path, "trajectory.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["decision", "emulated_frame", "x", "y", "z", "action", "action_name",
                        "frame_buttons", "exact", "reward"])
            k = env.frameskip
            for t, p in enumerate(self.pos):
                a = self.actions[t - 1] if t else ""
                frames = " | ".join("+".join(mask_to_buttons(m)) or "-"
                                    for m in self.frame_masks[(t - 1) * k:t * k]) if t else ""
                w.writerow([t, t * k, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", a,
                            env.action_names[a] if t else "", frames,
                            self.exact[t - 1] if t else "", f"{self.rewards[t - 1]:.5f}" if t else ""])
        plot_trajectory(np.stack(self.pos), os.path.join(self.path, "trajectory.png"), state)
        files = ["demo.npz", "trajectory.csv", "trajectory.png"] + (["gameplay.mp4"] if has_video else [])
        files += [c["file"] for c in self.checkpoints]
        hist = np.bincount(np.asarray(self.actions, int), minlength=len(env.action_names))
        manifest = {
            "recorder_version": RECORDER_VERSION, "state": state,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "source_state": repo_rel(source_state),
            "source_state_sha256": source_hash, "rom_sha256": rom_hash,
            "track_id": env.track_id, "input_mode": input_mode, "action_set": env.action_set,
            "action_names": list(env.action_names), "frameskip": env.frameskip,
            "decisions": len(self.actions), "emulated_frames": len(self.actions) * env.frameskip,
            "star_collected": bool(info.get("star_collected")),
            "reached_goal": bool(info.get("reached_goal")), "died": bool(info.get("died")),
            "final_pos": [float(v) for v in self.pos[-1]],
            "final_dist_to_goal": info.get("dist_to_goal"),
            "max_height": float(np.max(np.stack(self.pos)[:, 1])),
            "exact_fraction": float(np.mean(self.exact)) if self.exact else None,
            "action_histogram": {env.action_names[i]: int(c) for i, c in enumerate(hist)},
            "frame_mask_order": list(FREE_BUTTONS),
            "buttons_used": {b: int(sum(1 for m in self.frame_masks if m >> i & 1))
                             for i, b in enumerate(FREE_BUTTONS)},
            "env_config": {"flow_weight": env.flow_weight, "dist_reward_mode": env.dist_reward_mode,
                           "dist_weight": env.dist_weight, "step_penalty": env.step_penalty,
                           "star_reward": env.star_reward, "death_penalty": env.death_penalty},
            "checkpoints": self.checkpoints, "video_frames": self.video.frames,
            "files_sha256": {f: sha256_file(os.path.join(self.path, f)) for f in files},
            "note": ("Human demonstration. frame_masks (one per emulated frame, frame_mask_order) "
                     "replay the run exactly; actions are action_set labels per decision and are "
                     "agent-reproducible only where exact is true."),
        }
        with open(os.path.join(self.path, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.remove(os.path.join(self.path, ".recorder-owned"))
        os.makedirs(dest_root, exist_ok=True)
        target = os.path.join(dest_root, os.path.basename(self.path))
        os.rename(self.path, target)
        return target


def plot_trajectory(pos, out, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.5))
    sc = a.scatter(pos[:, 0], pos[:, 2], c=pos[:, 1], s=4, cmap="viridis")
    a.plot(pos[[0, -1], 0], pos[[0, -1], 2], "o", color="#df9e25")
    a.annotate("start", pos[0, [0, 2]])
    a.annotate(label, pos[-1, [0, 2]])
    a.set(xlabel="World X", ylabel="World Z", title="Top-down path (colour = height Y)")
    a.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=a, label="Y")
    b.plot(pos[:, 1])
    b.set(xlabel="decision", ylabel="height Y", title="Height over the attempt")
    for ax in (a, b):
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


class Keyboard:
    def __init__(self):
        self.u = ctypes.windll.user32
        self.u.FindWindowW.restype = ctypes.c_void_p
        self.u.GetForegroundWindow.restype = ctypes.c_void_p
        self.prev = {}

    def down(self, vk):
        return bool(self.u.GetAsyncKeyState(vk) & 0x8000)

    def pressed_edge(self, vk):
        now, before = self.down(vk), self.prev.get(vk, False)
        self.prev[vk] = now
        return now and not before

    def buttons(self):
        return tuple(b for b, vks in VK.items() if any(self.down(v) for v in vks))

    def focused(self):
        h = self.u.FindWindowW(None, WINDOW)
        return bool(h) and self.u.GetForegroundWindow() == h


def show(rgb, lines, names=()):
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22 * (len(lines) + 1) + 6), (0, 0, 0), -1)
    for i, line in enumerate(list(lines) + [f"keys: {' + '.join(names) if names else 'none'}"]):
        cv2.putText(img, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.imshow(WINDOW, img)
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="ds8", help="Start savestate (track id or path; default ds8)")
    ap.add_argument("--input", default="free", choices=["free", "agent"],
                    help="free: any buttons, read every frame (default); agent: one action per decision")
    ap.add_argument("--action-set", default="compound12", choices=sorted(ACTION_SETS),
                    help="Action set used to label (free) or restrict (agent) the input")
    ap.add_argument("--frameskip", type=int, default=4, help="Frames per decision (training uses 4)")
    ap.add_argument("--checkpoint-every", type=int, default=30,
                    help="Write a curriculum savestate every N decisions (default 30 = 2 s)")
    ap.add_argument("--max-decisions", type=int, default=9000,
                    help="Per-attempt limit (default 9000 = 10 min); reaching it saves a partial demo")
    ap.add_argument("--max-minutes", type=int, default=120)
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT_DIR, "runs", "human-demonstrations"))
    args = ap.parse_args()
    if os.name != "nt":
        raise SystemExit("The recorder reads the keyboard through the Win32 API and needs Windows.")

    from src.env import Mario64DSEnv
    source_state = savestate_path(args.state)
    rom = rom_path()
    for p in (rom, source_state):
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p} (see data/README.md and scripts/manage_states.py)")
    source_hash, rom_hash = sha256_file(source_state), sha256_file(rom)

    run = os.path.join(args.out, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    pending = os.path.join(run, "pending")
    os.makedirs(pending)
    status = {"run": run, "state": "starting", "source_state": args.state, "input_mode": args.input,
              "action_set": args.action_set,
              "attempts_started": 0, "deaths_discarded": 0, "restarts_discarded": 0, "saved": []}

    def write_status():
        with open(os.path.join(run, "status.json"), "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)

    write_status()
    print(json.dumps({"run": run, "window": WINDOW}), flush=True)

    env = Mario64DSEnv(rom_path=rom, state_path=source_state, max_steps=args.max_decisions,
                       frameskip=args.frameskip, flow_weight=0.0, action_set=args.action_set,
                       dist_reward_mode="progress")
    kb = Keyboard()
    attempt = None
    clock = {"next": time.perf_counter()}
    overlay = {"lines": [], "names": ()}

    def on_cycle():
        rgb = env.get_screen_rgb()
        if attempt is not None:
            attempt.video.write(rgb)
        show(rgb, overlay["lines"], overlay["names"])
        now = time.perf_counter()
        if now < clock["next"]:
            time.sleep(clock["next"] - now)
        clock["next"] = max(clock["next"] + 1.0 / FPS, time.perf_counter())  # no catch-up bursts

    def live_mask():
        # Sampled by env.step before every emulated frame in --input free.
        overlay["names"] = kb.buttons()
        return buttons_to_mask(overlay["names"])

    def idle(lines, names=()):
        show(env.get_screen_rgb(), lines, names)
        time.sleep(0.03)
        clock["next"] = time.perf_counter()

    env.on_cycle = on_cycle  # src/env.py calls it after every emulated frame
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    env.reset()
    deadline = time.monotonic() + 60 * args.max_minutes
    started, release_required, info = False, True, {}
    try:
        while time.monotonic() < deadline:
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            n = status["attempts_started"]
            if not kb.focused():
                idle([f"Attempt {n} | Click this window to play. Game is paused."])
                continue
            if kb.down(VK_F9) or kb.down(VK_ESC):
                break
            if not started:
                idle(["Enter: start. Arrows/WASD move  Space/K jump  Shift/J run  Ctrl/C crouch",
                      "E/U punch  Q camera  X camera | F5 restart  F8 save partial  F9/Esc stop"])
                started = kb.down(VK_ENTER)
                continue
            if attempt is not None and kb.pressed_edge(VK_F5):
                attempt.discard()
                attempt = None
                status["restarts_discarded"] += 1
                env.reset()
                release_required = True
                continue
            if attempt is not None and kb.pressed_edge(VK_F8):
                target = attempt.promote(os.path.join(run, "partial"), "partial", env, args.input,
                                         source_state, source_hash, rom_hash, info)
                attempt = None
                status["saved"].append(target)
                print(json.dumps({"partial_recording": target}), flush=True)
                break
            names = kb.buttons()
            if attempt is None:
                if release_required and names:
                    idle([f"Attempt {n + 1} | Release all controls to begin."], names)
                    continue
                if status["attempts_started"] >= args.max_attempts:
                    status["state"] = "attempt_limit_reached"
                    break
                release_required = False
                status["attempts_started"] += 1
                attempt = Attempt(pending, status["attempts_started"], env)
                status["state"] = "recording"
                write_status()
            overlay["lines"] = [f"Attempt {status['attempts_started']} | decision {len(attempt.actions)} "
                                f"| {args.input} input",
                                "Recording. Death resets. F5 restart  F8 save partial  F9 stop"]
            if args.input == "free":
                obs, reward, terminated, truncated, info = env.step(live_mask)
                action, exact = label_window(env.last_frame_masks, env.action_buttons)
            else:
                overlay["names"] = names
                action, exact = keys_to_action(names, env.action_buttons)
                obs, reward, terminated, truncated, info = env.step(action)
            pos = info["pos_3d"] if info.get("pos_3d") is not None else env.ram.get_pos_3d()
            attempt.record(action, list(env.last_frame_masks), exact, obs, float(reward), info, pos)
            if info.get("star_collected") or info.get("reached_goal"):  # star (TTC) or slide finish
                target = attempt.promote(os.path.join(run, "completed"), "completed", env, args.input,
                                         source_state, source_hash, rom_hash, info)
                attempt = None
                status.update(state="completed")
                status["saved"].append(target)
                print(json.dumps({"validated_recording": target, "state": "completed"}), flush=True)
                idle(["Goal reached! Recording saved."])
                time.sleep(3)
                break
            if terminated:  # death (no other terminal exists on ds8)
                attempt.discard()
                attempt = None
                status["deaths_discarded"] += 1
                write_status()
                env.reset()
                release_required = True
                continue
            if truncated:
                target = attempt.promote(os.path.join(run, "partial"), "partial", env, args.input,
                                         source_state, source_hash, rom_hash, info)
                attempt = None
                status["saved"].append(target)
                print(json.dumps({"partial_recording": target, "reason": "max_decisions"}), flush=True)
                break
            if len(attempt.actions) % args.checkpoint_every == 0:
                attempt.checkpoint(env)
        else:
            status["state"] = "time_limit_reached"
        if status["state"] == "recording":
            status["state"] = "stopped_by_user" if not status["saved"] else "saved_partial"
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        if attempt is not None:
            attempt.discard()
        env.close()
        write_status()
    print(json.dumps({"state": status["state"], "saved": status["saved"]}), flush=True)


if __name__ == "__main__":
    main()
