import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import logging
import os

from src.ram_state import Sm64DsState, TRACK_GOALS, COURSE_TO_DEFAULT_TRACK

logger = logging.getLogger(__name__)

# Canonical action order. Keep display labels next to the semantics so the
# dashboard, collectors and trainers cannot drift apart.
ACTION_NAMES = ("noop", "left", "right", "up (accelerate)", "down (brake)", "B (jump)")
ACTION_BUTTONS = ((), ("left",), ("right",), ("up",), ("down",), ("B",))

# "compound12" keeps the legacy six in the same indices and appends held
# combinations. Measured on ds8 (40 frames): Up held then Up+B travels 148.8
# units vs 62.2 for the legacy Up -> B sequence, and Up+Y (run) walks 190 vs
# 135. Every checkpoint trained before it is legacy6 and stays loadable.
ACTION_SETS = {
    "legacy6": (ACTION_NAMES, ACTION_BUTTONS),
    "compound12": (
        ACTION_NAMES + ("up+left", "up+right", "up+B (running jump)", "up+left+B",
                        "up+right+B", "up+Y (run)"),
        ACTION_BUTTONS + (("up", "left"), ("up", "right"), ("up", "B"), ("up", "left", "B"),
                          ("up", "right", "B"), ("up", "Y")),
    ),
    # The ten actions of the coin-only agent that finished three slides in a video (None, Up, Up-R, R,
    # Down-R, Down, Down-L, L, Up-L, A). The DS jump button is B, so "A" is "B (jump)" here.
    "ref10": (
        ("none", "up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left", "jump"),
        ((), ("up",), ("up", "right"), ("right",), ("down", "right"), ("down",), ("down", "left"),
         ("left",), ("up", "left"), ("B",)),
    ),
}

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask
except ImportError:
    logger.warning("py-desmume is not installed or failed to load; "
                   "Mario64DSEnv can only be used with a mocked emulator.")
    DeSmuME = None
    class Keys:
        KEY_A = 1; KEY_B = 2; KEY_X = 4; KEY_Y = 8; KEY_UP = 16; KEY_DOWN = 32; KEY_LEFT = 64; KEY_RIGHT = 128
        KEY_START = 256; KEY_SELECT = 512; KEY_L = 1024; KEY_R = 2048
    def keymask(k):
        return k

BUTTON_KEYS = {"left": Keys.KEY_LEFT, "right": Keys.KEY_RIGHT, "up": Keys.KEY_UP,
               "down": Keys.KEY_DOWN, "A": Keys.KEY_A, "B": Keys.KEY_B, "X": Keys.KEY_X,
               "Y": Keys.KEY_Y, "L": Keys.KEY_L, "R": Keys.KEY_R}
# Bit order of a per-frame button mask. Measured on ds8: B jumps, R crouches
# (R then B = backflip), A/Y punch (Y also runs while moving), L recentres the
# camera. Start/Select are left out: the pause menu can exit the course.
FREE_BUTTONS = ("left", "right", "up", "down", "A", "B", "X", "Y", "L", "R")


def buttons_to_mask(buttons) -> int:
    return sum(1 << FREE_BUTTONS.index(b) for b in buttons)


def mask_to_buttons(mask: int) -> tuple:
    return tuple(b for i, b in enumerate(FREE_BUTTONS) if mask >> i & 1)

class _NoBonus(Exception):
    """Internal: skip the exploration bonus for this step."""


class Mario64DSEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, rom_path, state_path, render_mode=None, max_steps=1350, frameskip=4,
                 death_penalty=100.0, timeout_penalty=0.0, step_penalty=0.02,
                 coin_reward_enabled=False, flow_weight=1.0, verbose=False,
                 strict_errors=False, dist_reward_enabled=True, dist_weight=0.05,
                 goal_reward=200.0, star_reward=500.0, use_ram_death=True,
                 dist_reward_mode="progress",
                 track_id=None, action_set="legacy6", route_path=None,
                 route_total_reward=500.0, waypoint_reward=1.0, start_states=None, full_start_prob=0.25,
                 death_rule="position", explore_bonus=0.0, explore_cell=400.0,
                 ram_coin_reward=0.0, black_screen_death=True, goal_override=None,
                 respawn_jump=0.0, fall_speed=0.0, fall_steps=15):
        super(Mario64DSEnv, self).__init__()
        if action_set not in ACTION_SETS:
            raise ValueError(f"unknown action_set {action_set!r}; choose from {sorted(ACTION_SETS)}")
        self.action_set = action_set
        self.action_names, self.action_buttons = ACTION_SETS[action_set]

        self.rom_path = rom_path
        self.state_path = state_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frameskip = frameskip
        # verbose is opt-in: training loops can run millions of steps and the
        # old unconditional prints were both noisy and a pipe-pressure risk.
        self.verbose = bool(verbose)
        # strict_errors turns a display/memory read failure into an exception
        # instead of a silent black frame that the env would later call "death".
        self.strict_errors = bool(strict_errors)
        self._obs_error_logged = False
        # Reward convention (see README "Suicide Paradox"):
        # timeout = survival -> no penalty; abyss death -> -100.
        # Keep death_penalty > timeout_penalty, otherwise the agent learns to commit suicide.
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        # Anti-camping: fixed temporal cost per step.
        self.step_penalty = step_penalty
        self.flow_weight = flow_weight
        # Coin reward HSV: DISABLED by default.
        self.coin_reward_enabled = coin_reward_enabled
        
        # 3D Proximity & Goal Completion configuration
        self.dist_reward_enabled = bool(dist_reward_enabled)
        self.dist_weight = float(dist_weight)
        self.goal_reward = float(goal_reward)
        self.star_reward = float(star_reward)
        self.use_ram_death = bool(use_ram_death)
        # "position": Y below the course's kill_y (a real fall into the void).
        # "struct": legacy rule of the runs trained before 2026-11-04, which also
        # counted the 0x0217F760 fast-fall clamp, i.e. any long drop, as death.
        if death_rule not in ("position", "struct"):
            raise ValueError(f"death_rule must be 'position' or 'struct', got {death_rule!r}")
        self.death_rule = death_rule
        # Count-based exploration bonus (Strehl & Littman 2008; Bellemare et al.
        # 2016): explore_bonus / sqrt(N) for the 3D cell Mario is in, where N
        # counts this env's visits to the cell across episodes, so a dead end
        # stops paying once it has been seen. 0 disables it.
        self.explore_bonus = float(explore_bonus)
        self.explore_cell = float(explore_cell)
        self.visit_counts = {}
        self.last_explore_bonus = 0.0
        # Reward per coin collected, read from the course coin counter in RAM
        # (src/ram_state.COIN_COUNT_ADDR). Unlike the HSV detector above, it
        # cannot be triggered by yellow floors or walls. 0 disables it.
        self.ram_coin_reward = float(ram_coin_reward)
        # The black-screen fallback can fire in dark parts of a course (the
        # ds3 exit is a dark shaft); False leaves death to the position rule.
        self.black_screen_death = bool(black_screen_death)
        # The game itself puts Mario back at the start when he falls off the ds3 slide (the coin count
        # resets) well above kill_y: a displacement larger than this in one decision is such a respawn
        # and counts as a death (0 = off; free fall moves ~150 units per decision).
        self.respawn_jump = float(respawn_jump)
        # Off the slide Mario falls ~150 units per decision for ~100 decisions before the respawn. This
        # ends the episode once he has descended faster than fall_speed for fall_steps decisions in a row
        # (the ds3 human run never does it for more than 9 at 140; 0 = off).
        self.fall_speed, self.fall_steps = float(fall_speed), int(fall_steps)
        self._fall_run = 0
        self._last_pos = None
        # Optional intermediate goal {"pos": [x, y, z], "radius": r}: entering
        # that sphere ends the episode as reached_goal with goal_reward (e.g. a
        # spot partway down a course), in addition to the course's own finish.
        self.goal_override = None
        if goal_override is not None:
            self.goal_override = {"pos": np.asarray(goal_override["pos"], dtype=np.float64),
                                  "radius": float(goal_override["radius"])}
        self.coins = None
        self.last_ram_coin_reward = 0.0
        self.dist_reward_mode = str(dist_reward_mode).lower()
        self.track_id = track_id or self._infer_track_id()
        self.goal_spec = TRACK_GOALS.get(self.track_id)

        # "route": ratcheted arc length along a recorded human path (src/route.py)
        # replaces the straight-line distance term; star/goal detection is unchanged.
        self.route = None
        if self.dist_reward_mode in ("route", "waypoint"):
            if not route_path:
                raise ValueError("dist_reward_mode='route' requires route_path")
            from src.route import Route
            self.route = Route.load(route_path)
        self.route_total_reward = float(route_total_reward)
        self.waypoint_reward = float(waypoint_reward)
        self.route_best_s = None

        # Start-state curriculum: reset() loads the canonical state_path with
        # probability full_start_prob, otherwise a uniform pick of start_states
        # (e.g. checkpoints saved along a human run, as in Salimans & Chen 2018).
        self.start_states = list(start_states or [])
        self.full_start_prob = float(full_start_prob)
        self.current_start = state_path
        
        # Determine paths relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # HSV Tracking is handled in _get_obs()
        self.coins_template = None

        # Initialize Emulator
        try:
            self.emu = DeSmuME()
            self.emu.open(self.rom_path)
            self.emu.savestate.load_file(self.state_path)
            self.has_emulator = True
        except Exception as e:
            logger.error("failed to initialize emulator: %s", e)
            self.has_emulator = False
            raise

        # Exact-state observation via RAM (see src/ram_state.py)
        self.ram = Sm64DsState(self.emu)
        if not self.ram.verify():
            logger.warning("Mario physics RAM struct not at expected address; "
                           "state-based controllers will be unreliable.")

        # Fallback to RAM course_id if track_id was not matched from path
        if self.goal_spec is None and self.has_emulator:
            try:
                cid = self.ram.get_course_id()
                tid = COURSE_TO_DEFAULT_TRACK.get(cid)
                if tid:
                    self.track_id = tid
                    self.goal_spec = TRACK_GOALS.get(tid)
            except Exception:
                pass

        # legacy6: 0 Noop, 1 Left, 2 Right, 3 Up, 4 Down, 5 Jump(B); compound12 appends combos
        self.action_space = spaces.Discrete(len(self.action_names))
        
        # Observation space: 84x84 grayscale
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.prev_gray = None
        self.episode_steps = 0
        self.prev_dist = None
        self.last_dist = None
        self.last_dist_reward = 0.0
        self.reached_goal = False
        self.star_collected = False
        self.died = False
        self.start_star_count = 0
        self.target_pos = None

        # Reward-source instrumentation, exposed for the dashboard recorder
        self.last_flow = None
        self.last_flow_mean = 0.0
        self.last_flow_reward = 0.0
        self.last_coin_reward = 0.0

    def _infer_track_id(self) -> str | None:
        if not self.state_path:
            return None
        base = os.path.basename(self.state_path).lower()
        for k in ("ds9", "ds8", "ds6", "ds5", "ds4", "ds3", "ds2", "ds1"):
            if k in base:
                return k
        return None

    def _get_obs(self):
        if not self.has_emulator:
            return np.zeros((84, 84, 1), dtype=np.uint8)
            
        try:
            frame = self.emu.display_buffer_as_rgbx()
            if isinstance(frame, (bytes, bytearray)):
                frame = np.frombuffer(frame, dtype=np.uint8).reshape((384, 256, 4))
            else:
                frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
            # FIX (color bug): py-desmume display buffer is BGR(X), not RGBX
            # as previously assumed — cv2.imwrite showed Mario as red and video
            # (imageio, RGB) showed blue. With RGB2HSV on BGR data, hue rotates
            # (H -> 120-H) and yellow mask [15-40] detected cyan features (H~90 -> 30)
            # rather than coins (H~25 -> 95): coin reward was broken historically.
            top_screen = frame[:192, :, [2, 1, 0]]  # BGR -> RGB
            
            gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
            self.last_top_screen_gray = gray.copy()
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

            # Coin Tracking via HSV — only if enabled (disabled by default due
            # to checkered floor / flower wall detection).
            if not self.coin_reward_enabled:
                self.last_coin_reward = 0.0
                return np.expand_dims(resized, axis=-1)

            hsv = cv2.cvtColor(top_screen, cv2.COLOR_RGB2HSV)
            # Yellow/Gold coins in Mario 64 DS
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            coin_reward = 0.0
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 5:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        mean_x = int(M["m10"] / M["m00"])
                        # Center of screen is 128
                        dist = abs(mean_x - 128.0)
                        coin_reward = max(0, (128.0 - dist) / 128.0 * 0.5)
                        
            self.last_coin_reward = coin_reward

            return np.expand_dims(resized, axis=-1)
        except Exception as exc:
            # A display-buffer failure used to silently produce a black frame,
            # which the death detector then interpreted as a terminal state.
            if self.strict_errors:
                raise
            if not self._obs_error_logged:
                logger.exception("frame read failed; returning black observation "
                                 "(further failures will be silent): %s", exc)
                self._obs_error_logged = True
            self.last_coin_reward = 0.0
            return np.zeros((84, 84, 1), dtype=np.uint8)

    def _get_info(self):
        info = {
            "steps": self.episode_steps,
            "success": bool(self.reached_goal or self.star_collected),
            "reached_goal": bool(self.reached_goal),
            "star_collected": bool(self.star_collected),
            "died": bool(self.died),
            "dist_to_goal": self.last_dist,
            "dist_reward": self.last_dist_reward,
            "flow_reward": self.last_flow_reward,
            "coin_reward": self.last_coin_reward,
            "route_progress": self.route_best_s,
            "explore_bonus": self.last_explore_bonus,
            "explore_cells": len(self.visit_counts),
            "coins": self.coins,
            "coin_reward": self.last_ram_coin_reward,
            "start_state": os.path.basename(self.current_start) if self.current_start else None,
        }
        try:
            info["ram"] = self.ram.poll()
            info["pos_3d"] = self.ram.get_pos_3d()
            info["star_count"] = self.ram.get_star_count()
            info["course_id"] = self.ram.get_course_id()
        except Exception as exc:
            if self.strict_errors:
                raise
            logger.debug("RAM poll failed: %s", exc)
            info["ram"] = None
            info["pos_3d"] = None
            info["star_count"] = None
            info["course_id"] = None
        return info

    def _emit_cycle(self):
        """Fired after every emulated frame so a recorder can capture true 60 Hz
        footage instead of one image per action. Set env.on_cycle = callable."""
        cb = getattr(self, "on_cycle", None)
        if cb is not None:
            cb()

    def step(self, action):
        if not self.has_emulator:
            self.episode_steps += 1
            return self._get_obs(), 1.0, self.episode_steps > 100, False, self._get_info()

        # Map actions. An agent action holds its buttons for the whole window.
        # A human recording instead supplies one FREE_BUTTONS mask per frame:
        # a sequence of `frameskip` masks (replay) or a callable sampled before
        # every frame (live input). Both run the same reward code below.
        if callable(action):
            frame_mask = lambda f: int(action())  # noqa: E731
        elif np.ndim(action) == 1:
            masks = [int(m) for m in action]
            if len(masks) != self.frameskip:
                raise ValueError(f"expected {self.frameskip} per-frame masks, got {len(masks)}")
            frame_mask = masks.__getitem__
        else:
            held_mask = buttons_to_mask(self.action_buttons[int(action)])
            frame_mask = lambda f: held_mask  # noqa: E731

        held = 0
        self.last_frame_masks = []
        for f in range(self.frameskip):
            m = frame_mask(f)
            for i, b in enumerate(FREE_BUTTONS):
                bit = 1 << i
                if m & bit and not held & bit:
                    self.emu.input.keypad_add_key(keymask(BUTTON_KEYS[b]))
                elif held & bit and not m & bit:
                    self.emu.input.keypad_rm_key(keymask(BUTTON_KEYS[b]))
            held = m
            self.last_frame_masks.append(m)
            self.emu.cycle()
            self._emit_cycle()

        for b in mask_to_buttons(held):
            self.emu.input.keypad_rm_key(keymask(BUTTON_KEYS[b]))

        obs = self._get_obs()
        self.episode_steps += 1
        
        # Time cost per step (anti-camping, see __init__)
        reward = -self.step_penalty
        
        done = False 
        truncated = False
        
        if self.episode_steps >= self.max_steps:
            truncated = True
            # Timeout = survival (no explicit finish line detected).
            reward -= self.timeout_penalty
            if self.verbose:
                logger.info("timeout at step %d", self.episode_steps)

        # Death Check: RAM kill plane / kill_y first, fallback to black screen
        is_dead = False
        death_source = None
        if self.use_ram_death and self.has_emulator:
            try:
                curr_ram = self.ram.poll()
                pos_3d = self.ram.get_pos_3d()
                kill_y = self.goal_spec.get("kill_y") if self.goal_spec else None
                dead = (Sm64DsState.is_dead(state=curr_ram, pos_3d=pos_3d, kill_y=kill_y)
                        if self.death_rule == "struct"
                        else Sm64DsState.fell_out_of_course(pos_3d, kill_y))
                if dead:
                    is_dead = True
                    death_source = "RAM"
            except Exception as exc:
                if self.strict_errors:
                    raise
                logger.debug("RAM death check failed: %s", exc)

        if (self.respawn_jump > 0 or self.fall_speed > 0) and self.has_emulator:
            pos_now = self.ram.get_pos_3d()
            if self._last_pos is not None and pos_now is not None:
                if self.fall_speed > 0:
                    self._fall_run = self._fall_run + 1 if self._last_pos[1] - pos_now[1] > self.fall_speed else 0
                    if not is_dead and self._fall_run >= self.fall_steps:
                        is_dead = True
                        death_source = "fall"
                if (self.respawn_jump > 0 and not is_dead
                        and float(np.linalg.norm(pos_now - self._last_pos)) > self.respawn_jump):
                    is_dead = True
                    death_source = "respawn"
            self._last_pos = None if pos_now is None else np.array(pos_now, dtype=np.float64)

        obs_2d = np.squeeze(obs)
        if not is_dead and self.black_screen_death:
            if np.mean(obs_2d < 10) > 0.95:
                is_dead = True
                death_source = "black_screen"

        if is_dead:
            done = True
            self.died = True
            reward -= self.death_penalty
            if self.verbose:
                logger.info("death detected (%s, penalty=-%g) at step %d",
                            death_source, self.death_penalty, self.episode_steps)

        # 3D Distance Potential Reward and Completion Detection
        self.last_dist_reward = 0.0
        if not done and not truncated and self.has_emulator and self.goal_spec is not None and self.target_pos is not None:
            try:
                curr_pos = self.ram.get_pos_3d()
                curr_dist = float(np.linalg.norm(curr_pos - self.target_pos))
                self.last_dist = curr_dist
                
                # 3D Distance Reward: High-Water Mark ("progress") vs Bidirectional ("potential")
                # vs ratcheted arc length along a recorded route ("route")
                if self.dist_reward_enabled:
                    if self.dist_reward_mode in ("route", "waypoint"):
                        s = (self.route.locate(curr_pos) if self.route_best_s is None
                             else self.route.progress(curr_pos, self.route_best_s))
                        if s is not None:
                            if self.route_best_s is not None and s > self.route_best_s:
                                if self.dist_reward_mode == "waypoint":
                                    # a fixed reward per waypoint of the route newly passed (ratcheted)
                                    dist_reward = self.route.passed(self.route_best_s, s) * self.waypoint_reward
                                else:
                                    dist_reward = ((s - self.route_best_s) / self.route.length
                                                   * self.route_total_reward)
                                self.last_dist_reward = float(dist_reward)
                                reward += dist_reward
                            if self.route_best_s is None or s > self.route_best_s:
                                self.route_best_s = s
                    elif self.dist_reward_mode == "progress":
                        if self.best_dist is not None and curr_dist < self.best_dist:
                            progress = self.best_dist - curr_dist
                            dist_reward = progress * self.dist_weight
                            self.last_dist_reward = float(dist_reward)
                            reward += dist_reward
                            self.best_dist = curr_dist
                    else:  # "potential"
                        if self.prev_dist is not None:
                            progress = self.prev_dist - curr_dist
                            dist_reward = progress * self.dist_weight
                            self.last_dist_reward = float(dist_reward)
                            reward += dist_reward
                self.prev_dist = curr_dist
                
                # Check Completion Condition
                goal_type = self.goal_spec.get("goal_type", "position")
                success_dist = self.goal_spec.get("success_dist", 250.0)
                
                if self.goal_override is not None and np.linalg.norm(
                        curr_pos - self.goal_override["pos"]) <= self.goal_override["radius"]:
                    done = True
                    self.reached_goal = True
                    reward += self.goal_reward
                elif goal_type == "star":
                    curr_stars = self.ram.get_star_count()
                    if curr_dist <= success_dist or (curr_stars > self.start_star_count and curr_dist <= 500.0):
                        done = True
                        self.star_collected = True
                        reward += self.star_reward
                        if self.verbose:
                            logger.info("Star collected! (reward=+%g) at step %d",
                                        self.star_reward, self.episode_steps)
                else:  # "position": near target_pos, or anywhere in the measured end area
                    if curr_dist <= success_dist or Sm64DsState.in_finish_region(
                            curr_pos, self.target_pos, self.goal_spec.get("finish_region")):
                        done = True
                        self.reached_goal = True
                        reward += self.goal_reward
                        if self.verbose:
                            logger.info("Goal reached! (dist=%.1f <= %.1f, reward=+%g) at step %d",
                                        curr_dist, success_dist, self.goal_reward, self.episode_steps)
            except Exception as exc:
                if self.strict_errors:
                    raise
                logger.debug("3D distance reward computation failed: %s", exc)

        self.last_ram_coin_reward = 0.0
        if self.has_emulator:
            try:
                coins = self.ram.get_coins()
                if self.ram_coin_reward and self.coins is not None and coins > self.coins:
                    self.last_ram_coin_reward = self.ram_coin_reward * (coins - self.coins)
                    reward += self.last_ram_coin_reward
                self.coins = coins
            except Exception as exc:
                if self.strict_errors:
                    raise
                logger.debug("coin counter read failed: %s", exc)

        self.last_explore_bonus = 0.0
        if self.explore_bonus > 0 and not done and self.has_emulator:
            try:
                # No bonus in free fall: a fall through the void crosses a new
                # cell every few decisions and would otherwise be paid as
                # exploration (measured: training episodes shrank from 790 to
                # 301 decisions in 120k steps when falls were paid).
                if Sm64DsState.is_dead(state=self.ram.poll()):
                    raise _NoBonus
                cell = tuple(np.floor(self.ram.get_pos_3d() / self.explore_cell).astype(int))
                n = self.visit_counts.get(cell, 0) + 1
                self.visit_counts[cell] = n
                self.last_explore_bonus = self.explore_bonus / np.sqrt(n)
                reward += self.last_explore_bonus
            except _NoBonus:
                pass
            except Exception as exc:
                if self.strict_errors:
                    raise
                logger.debug("exploration bonus failed: %s", exc)

        # Coin Tracking / Guidance via HSV
        if not done and hasattr(self, 'last_coin_reward'):
            reward += self.last_coin_reward

        # Optical Flow (Fast 32x32)
        # Zeroed before the gate below so a consumer always reads the term that
        # was actually added to this step's reward (skipped on terminal steps).
        self.last_flow, self.last_flow_mean, self.last_flow_reward = None, 0.0, 0.0
        if not done and not truncated:
            curr_gray_small = cv2.resize(obs_2d, (32, 32), interpolation=cv2.INTER_AREA)
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, curr_gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_y = flow[..., 1]
                # Positive flow_y means pixels are moving down -> Mario is moving forward.
                flow_reward = np.clip(np.mean(flow_y), 0, None) * self.flow_weight
                self.last_flow = flow
                self.last_flow_mean = float(np.mean(flow_y))
                self.last_flow_reward = float(flow_reward)
                reward += flow_reward
            self.prev_gray = curr_gray_small

        info = self._get_info()
        
        if self.render_mode == 'human':
            self.render()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_start = self.state_path
        if self.start_states and self.np_random.random() >= self.full_start_prob:
            self.current_start = self.start_states[int(self.np_random.integers(len(self.start_states)))]
        if self.has_emulator:
            self.emu.savestate.load_file(self.current_start)

        self.episode_steps = 0
        self.prev_gray = None

        # Re-resolve track & goal if not already configured
        if self.goal_spec is None:
            self.track_id = self._infer_track_id()
            self.goal_spec = TRACK_GOALS.get(self.track_id)
            if self.goal_spec is None and self.has_emulator:
                try:
                    cid = self.ram.get_course_id()
                    tid = COURSE_TO_DEFAULT_TRACK.get(cid)
                    if tid:
                        self.track_id = tid
                        self.goal_spec = TRACK_GOALS.get(tid)
                except Exception:
                    pass

        self.prev_dist = None
        self.best_dist = None
        self.last_dist = None
        self.last_dist_reward = 0.0
        self.last_flow_reward = 0.0
        self.route_best_s = None
        self.coins = None  # re-read below: a new start may hold fewer coins than the last episode
        self.last_ram_coin_reward = 0.0
        self.reached_goal = False
        self.star_collected = False
        self.died = False
        self._last_pos = None
        self._fall_run = 0

        if self.has_emulator and self.goal_spec is not None:
            try:
                current_pos = self.ram.get_pos_3d()
                self._last_pos = np.array(current_pos, dtype=np.float64)
                self.target_pos = self.goal_spec["target_pos"]
                self.prev_dist = float(np.linalg.norm(current_pos - self.target_pos))
                self.best_dist = self.prev_dist
                self.last_dist = self.prev_dist
                self.start_star_count = self.ram.get_star_count()
                # A checkpoint start begins at its own place on the route, so
                # the route already covered before it is never paid.
                if self.route is not None:
                    self.route_best_s = self.route.locate(current_pos)
            except Exception as exc:
                if self.strict_errors:
                    raise
                logger.debug("Goal initialization in reset failed: %s", exc)
                self.start_star_count = 0
                self.target_pos = None
        else:
            self.start_star_count = 0
            self.target_pos = None

        if self.has_emulator:
            try:
                self.coins = self.ram.get_coins()
            except Exception as exc:
                if self.strict_errors:
                    raise
                logger.debug("coin counter read failed: %s", exc)

        obs = self._get_obs()
        return obs, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            obs = self._get_obs()
            if obs is not None:
                display_img = cv2.resize(obs, (336, 336), interpolation=cv2.INTER_NEAREST)
                try:
                    cv2.imshow("Mario 64 DS RL", display_img)
                    cv2.waitKey(1)
                except cv2.error as exc:
                    # Headless containers (and CI) do not have a GUI backend.
                    logger.warning("cv2.imshow unavailable in this environment: %s", exc)
                    self.render_mode = None
        elif self.render_mode == 'rgb_array':
            return self.get_screen_rgb()

    def get_screen_rgb(self):
        """Real top screen of the game in RGB (256x192), for video recording."""
        if not self.has_emulator:
            return np.zeros((192, 256, 3), dtype=np.uint8)
        frame = self.emu.display_buffer_as_rgbx()
        if isinstance(frame, (bytes, bytearray)):
            frame = np.frombuffer(frame, dtype=np.uint8).reshape((384, 256, 4))
        else:
            frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
        # Buffer is BGR(X) (see fix in _get_obs): convert to RGB
        return frame[:192, :, [2, 1, 0]]

    def close(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if self.has_emulator:
            self.emu.destroy()
