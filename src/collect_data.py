"""Collect real emulator transitions into a world-model replay buffer.

This is the *data-collection* stage of the model-based pipeline: the emulator is
the only source of ground truth, and we deliberately keep its use small (tens of
thousands of steps) because the world model lets the *policy* train afterwards
without touching the (slow) emulator at all.

Behavior policies used to generate transitions:
* ``random``  — diverse exploration, including failures (teaches the terminal /
  death dynamics and the abyss penalty).
* ``ppo``     — replays an already-trained SB3 PPO agent (e.g. the benchmark
  ``curriculum_flow25_r3_best.zip``) so the buffer also contains *successful*
  full-descent trajectories. This is exactly the reference's principle of
  grounding the learned dynamics in genuine transitions.
* ``ppo_noise`` — perturbed-expert exploration: the PPO policy *sampled*
  (stochastic), with each action replaced by a random one with probability
  ``--noise-eps``. This is the highest-value failure data for the value/Q
  heads: deaths happen at racing speed along the expert line (the exact
  distribution a planner flies through), unlike random-policy deaths, which
  are slow meanders that leave the value landscape blind where the agent
  actually races. No expert actions are supervised anywhere -- the controller
  is still expert-free; the benchmark PPO only positions/drives the emulator
  during data collection.
* ``explore`` — checkpoint-seeded random exploration. One deterministic PPO
  descent saves an intermediate savestate every ``--ckpt-every`` steps; a
  *random* policy then plays short episodes from every checkpoint. This is the
  near-cliff data the world model was missing: from the track start a random
  policy only ever dies at the first cliff (deaths stop at ~step 600 of 1350),
  so the model never saw a lethal edge past the early sections. Note the PPO
  agent is used only to *position* the emulator (no expert actions enter the
  buffer) — the recorded transitions are random-policy ones, most ending in a
  fall, evenly spread along the whole descent.

Run as a module, e.g.:

    python -m src.collect_data --states ds1,ds2,ds3 --episodes-per-state 8 \
        --behavior random,ppo --ppo-model models/curriculum_flow25_r3_best.zip \
        --out data/wm_buffer.pkl
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np

from src.env import Mario64DSEnv
from src.wm_replay import EpisodeBuffer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stack_from_deque(dq: deque) -> np.ndarray:
    """(1,4,84,84) CHW float in [0,255] -> matches SB3 VecFrameStack order."""
    return np.stack(list(dq), axis=0)[np.newaxis, :]


def _rollout(env, policy, frameskip_actions) -> dict:
    obs, _ = env.reset()
    first = obs[:, :, 0]
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)
    frames, actions, rewards, continues = [], [], [], []
    done = False
    steps = 0
    max_steps = env.max_steps
    while not done and steps < max_steps:
        action = frameskip_actions(steps, _stack_from_deque(dq))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        frame = obs[:, :, 0]
        dq.append(frame)
        frames.append(frame)
        actions.append(int(action))
        rewards.append(float(reward))
        # continue = 1 unless a *terminal* death occurred at this step
        continues.append(0.0 if terminated else 1.0)
        done = bool(terminated or truncated)
        steps += 1
    return dict(frames=np.array(frames), actions=np.array(actions),
                rewards=np.array(rewards), continues=np.array(continues))


def _ppo_stack_policy(ppo, deterministic):
    def policy_fn(_s, _o):
        a, _ = ppo.predict(_o, deterministic=deterministic)
        return int(a[0])
    return policy_fn


def _save_checkpoints(env, ppo, state, ckpt_every, max_steps, out_dir, verbose):
    """One deterministic PPO descent over ``state``, saving an emulator savestate
    every ``ckpt_every`` steps. Returns the list of savestate paths (positions
    evenly spread along the descent; the *transitions* played from them later are
    random-policy ones)."""
    obs, _ = env.reset()
    first = obs[:, :, 0]
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)
    steps, done = 0, False
    ckpts = []
    policy = _ppo_stack_policy(ppo, True)
    while not done and steps < max_steps:
        action = policy(steps, _stack_from_deque(dq))
        obs, _, terminated, truncated, _ = env.step(action)
        dq.append(obs[:, :, 0])
        steps += 1
        done = bool(terminated or truncated)
        if steps % ckpt_every == 0 and not done:
            path = os.path.join(out_dir, f"{state}_k{steps}.dsx")
            env.emu.savestate.save_file(path)
            ckpts.append(path)
            if verbose:
                print(f"[explore] {state}: checkpoint @ step {steps}", flush=True)
    if verbose:
        print(f"[explore] {state}: PPO seeding descent done ({steps} steps, "
              f"{len(ckpts)} checkpoints)", flush=True)
    return ckpts


def _explore_from_checkpoints(env, state, ckpts, episodes_per_ckpt, max_ep_steps,
                              buffer, rng, verbose):
    """Random-policy episodes from every checkpoint of one track.

    A checkpoint load that fails twice (DeSmuME occasionally refuses a savestate
    we just wrote) is skipped with a warning instead of aborting the collection.
    """
    loaded = 0
    for ck in ckpts:
        env.state_path = ck
        try:
            env.reset()
        except Exception as e:
            print(f"[explore] WARNING: {os.path.basename(ck)} failed to load "
                  f"({e}); skipping", flush=True)
            continue
        for ep in range(episodes_per_ckpt):
            def policy_fn(_s, _o, _rng=rng):
                return int(_rng.integers(6))
            prev_max = env.max_steps
            env.max_steps = max_ep_steps
            try:
                roll = _rollout(env, None, policy_fn)
            finally:
                env.max_steps = prev_max
            completed = (len(roll["continues"]) > 0 and roll["continues"][-1] > 0.5)
            buffer.add(roll["frames"], roll["actions"], roll["rewards"],
                       roll["continues"], source="explore", completed=completed)
            loaded += 1
            if verbose:
                print(f"[explore] {state}/{os.path.basename(ck)}/ep{ep}: "
                      f"len={len(roll['frames'])} R={roll['rewards'].sum():.1f} "
                      f"{'COMPLETED' if completed else 'died'}", flush=True)
    return loaded


def _probe_branches(env, ppo, state, probe_every, branch_len, prefix_len,
                    buffer, max_steps, out_dir, verbose,
                    main_policy=None, source_tag="probe", branch_policy="ppo"):
    """Counterfactual action probing along a descent.

    Every ``probe_every`` steps of the main rollout (deterministic PPO by
    default, or ``main_policy`` for perturbed rollouts) we save an emulator
    savestate and, from that exact state, try EVERY action once, continuing each
    branch for ``branch_len`` steps. All 6 branches share the same prefix (the
    last ``prefix_len`` frames), so the world model sees the SAME belief state
    with different actions and their REAL outcomes -- the action-contrast data
    that offline (s, a, return) pairs cannot provide and that imagination
    cannot substitute (measured: every head is action-blind on imagined
    latents, while the prior itself is action-sensitive).

    ``branch_policy`` selects the continuation after the probe action:
    * ``ppo``   -- the expert recovers; branches then measure speed contrast
      (on-line, a single racing-speed action is essentially always recoverable,
      and the recovery erases any line-position difference, so the V bootstrap
      at the branch end reads the same for all actions).
    * ``noop``  -- coast (no input) after the probe action; branches measure the
      RAW geometric consequence of the action: steering toward an edge falls,
      steering to the center slides on. This is the cliff-avoidance gradient.
    * ``hold``  -- keep repeating the probe action (the most adversarial raw
      continuation).

    Episode layout: frames = [prefix (rollout frames)][branch frames], with
    ``probe_at`` = index of the probe action (i.e. the branch starts at frame
    index ``probe_at``; the pre-action state is the posterior at index
    ``probe_at - 1``).
    """
    if main_policy is None:
        main_policy = _ppo_stack_policy(ppo, True)
    env.state_path = os.path.join(ROOT, "data",
                                  f"Super Mario 64 DS (USA) (Rev 1).{state}")
    obs, _ = env.reset()
    first = obs[:, :, 0]
    dq = deque([np.zeros_like(first)] * 3 + [first], maxlen=4)
    pre_frames = deque(maxlen=prefix_len)
    pre_actions = deque(maxlen=prefix_len)
    pre_rewards = deque(maxlen=prefix_len)
    pre_continues = deque(maxlen=prefix_len)
    n_probe = n_died = 0
    steps, done = 0, False
    while not done and steps < max_steps:
        action = main_policy(steps, _stack_from_deque(dq))
        obs, r, terminated, truncated, _ = env.step(action)
        frame = obs[:, :, 0]
        dq.append(frame)
        pre_frames.append(frame)
        pre_actions.append(int(action))
        pre_rewards.append(float(r))
        pre_continues.append(0.0 if terminated else 1.0)
        steps += 1
        done = bool(terminated or truncated)
        if done or steps % probe_every != 0:
            continue
        path = os.path.join(out_dir, f"{state}_p{steps}.dsx")
        env.emu.savestate.save_file(path)
        prev_gray = env.prev_gray     # keep the flow-reward baseline across reloads
        for a_probe in range(6):
            env.state_path = path
            try:
                env.reset()          # reload the exact probe state
            except Exception as e:
                print(f"[probe] WARNING: reload failed at {state} step {steps}: {e}",
                      flush=True)
                break
            env.prev_gray = prev_gray
            frames = [np.asarray(f, np.uint8) for f in pre_frames]
            actions = [int(a) for a in pre_actions]
            rewards = [float(x) for x in pre_rewards]
            continues = [float(x) for x in pre_continues]
            # the probe action itself, then PPO recovery for branch_len-1 steps
            a = a_probe
            dead = False
            for k in range(branch_len):
                obs_b, r_b, term_b, trunc_b, _ = env.step(int(a))
                frames.append(obs_b[:, :, 0])
                actions.append(int(a))
                rewards.append(float(r_b))
                continues.append(0.0 if term_b else 1.0)
                dead = bool(term_b or trunc_b)
                if dead:
                    break
                if branch_policy == "noop":
                    a = 0
                elif branch_policy == "hold":
                    a = a_probe
                else:
                    # PPO stack: last 4 frames, oldest..newest along axis 1, uint8
                    recent = frames[-4:]
                    if len(recent) < 4:  # early-episode probes: zero-pad the stack
                        recent = ([np.zeros_like(frames[-1])] * (4 - len(recent))
                                  + recent)
                    stack = np.stack(recent, axis=0)[np.newaxis, :]
                    pred, _ = ppo.predict(stack, deterministic=True)
                    a = int(pred[0])
            buffer.add(np.array(frames), np.array(actions), np.array(rewards),
                       np.array(continues), source=source_tag,
                       completed=not dead, probe_at=len(pre_frames),
                       probe_id=steps,
                       branch_cont={"noop": 0, "hold": 1, "ppo": 2}[branch_policy])
            n_probe += 1
            n_died += int(dead)
            if verbose:
                print(f"[probe] {state}@{steps} a={a_probe}: len={len(rewards)} "
                      f"R={sum(rewards):7.1f} {'died' if dead else 'ok'}", flush=True)
        # resume the main descent from the probe state, then drop the savestate
        env.state_path = path
        try:
            env.reset()
        except Exception as e:
            print(f"[probe] WARNING: resume failed at {state} step {steps}: {e}",
                  flush=True)
            break
        env.prev_gray = prev_gray
        try:
            os.remove(path)
        except OSError:
            pass
    if verbose:
        print(f"[probe] {state}: {n_probe} branches ({n_died} deaths) from "
              f"{steps} descent steps", flush=True)
    return n_probe


def collect(rom_path, states, episodes_per_state, behaviors, buffer: EpisodeBuffer,
            ppo_model=None, seed=0, max_steps=1350, flow_weight=2.5, step_penalty=0.02,
            frameskip=4, verbose=True, ppo_deterministic=False,
            ckpt_every=150, explore_eps=8, explore_max_steps=250,
            incremental_out=None, noise_eps=0.15,
            probe_every=12, branch_len=12, prefix_len=24, branch_policy="ppo"):
    rng = np.random.default_rng(seed)
    num_actions = 6
    ppo = None
    if any(b in behaviors for b in ("ppo", "explore", "ppo_noise", "probe",
                                     "probe_noise")):
        if not ppo_model:
            raise ValueError("--ppo-model required when using ppo/explore/"
                             "ppo_noise/probe/probe_noise behavior")
        from stable_baselines3 import PPO
        ppo = PPO.load(ppo_model)

    # DeSmuME cannot be (re)initialized more than once per process (access
    # violation), so we keep a SINGLE emulator instance alive and switch the
    # savestate file it loads on each reset() to move between tracks.
    existing = [s for s in states
                if os.path.exists(os.path.join(ROOT, "data",
                                               f"Super Mario 64 DS (USA) (Rev 1).{s}"))]
    if not existing:
        raise FileNotFoundError("no valid savestates found")
    env = Mario64DSEnv(rom_path, os.path.join(ROOT, "data",
                       f"Super Mario 64 DS (USA) (Rev 1).{existing[0]}"),
                       max_steps=max_steps, flow_weight=flow_weight,
                       step_penalty=step_penalty, frameskip=frameskip)
    try:
        for state in existing:
            env.state_path = os.path.join(ROOT, "data",
                                          f"Super Mario 64 DS (USA) (Rev 1).{state}")
            # checkpoint-seeded random exploration: descend with the PPO purely to
            # POSITION the emulator (no expert actions are recorded), then collect
            # random episodes around every checkpoint. Interleaved per track so a
            # savestate quirk on one track cannot lose the others' data.
            if "explore" in behaviors:
                import shutil
                ckpt_dir = os.path.join(ROOT, "data", "_explore_ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                try:
                    ckpts = _save_checkpoints(env, ppo, state, ckpt_every, max_steps,
                                              ckpt_dir, verbose)
                    _explore_from_checkpoints(env, state, ckpts, explore_eps,
                                             explore_max_steps, buffer, rng, verbose)
                except Exception as e:
                    print(f"[explore] WARNING: {state} exploration failed: {e}",
                          flush=True)
                finally:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                if incremental_out:
                    buffer.save(incremental_out)
                    print(f"[collect] incremental save -> {incremental_out}", flush=True)
            if "probe" in behaviors:
                import shutil
                probe_dir = os.path.join(ROOT, "data", "_probe_ckpt")
                os.makedirs(probe_dir, exist_ok=True)
                try:
                    _probe_branches(env, ppo, state, probe_every, branch_len,
                                   prefix_len, buffer, max_steps, probe_dir, verbose,
                                   branch_policy=branch_policy)
                except Exception as e:
                    print(f"[probe] WARNING: {state} probing failed: {e}", flush=True)
                finally:
                    shutil.rmtree(probe_dir, ignore_errors=True)
                if incremental_out:
                    buffer.save(incremental_out)
                    print(f"[collect] incremental save -> {incremental_out}", flush=True)
            if "probe_noise" in behaviors:
                import shutil
                probe_dir = os.path.join(ROOT, "data", "_probe_ckpt")
                os.makedirs(probe_dir, exist_ok=True)

                def noisy_policy(_s, _o, _rng=rng, _eps=noise_eps):
                    if _rng.random() < _eps:
                        return int(_rng.integers(6))
                    a, _ = ppo.predict(_o, deterministic=False)
                    return int(a[0])
                try:
                    _probe_branches(env, ppo, state, probe_every, branch_len,
                                   prefix_len, buffer, max_steps, probe_dir,
                                   verbose, main_policy=noisy_policy,
                                   source_tag="probe_noise",
                                   branch_policy=branch_policy)
                except Exception as e:
                    print(f"[probe] WARNING: {state} noisy probing failed: {e}",
                          flush=True)
                finally:
                    shutil.rmtree(probe_dir, ignore_errors=True)
                if incremental_out:
                    buffer.save(incremental_out)
                    print(f"[collect] incremental save -> {incremental_out}", flush=True)
            for beh in behaviors:
                if beh in ("explore", "probe", "probe_noise"):
                    continue  # handled above (checkpoint-seeded / branch probes)
                for ep in range(episodes_per_state):
                    if beh == "random":
                        def policy_fn(_s, _o, _rng=rng):
                            return int(_rng.integers(num_actions))
                    elif beh == "ppo":
                        def policy_fn(_s, _o, _ppo=ppo, _det=ppo_deterministic):
                            a, _ = _ppo.predict(_o, deterministic=_det)
                            return int(a[0])
                    elif beh == "ppo_noise":
                        def policy_fn(_s, _o, _ppo=ppo, _rng=rng, _eps=noise_eps):
                            if _rng.random() < _eps:
                                return int(_rng.integers(num_actions))
                            a, _ = _ppo.predict(_o, deterministic=False)
                            return int(a[0])
                    else:
                        raise ValueError(beh)
                    roll = _rollout(env, None, policy_fn)
                    completed = (len(roll["continues"]) > 0 and roll["continues"][-1] > 0.5)
                    buffer.add(roll["frames"], roll["actions"], roll["rewards"],
                               roll["continues"], source=beh, completed=completed)
                    if verbose:
                        print(f"[collect] {state}/{beh} ep{ep}: len={len(roll['frames'])} "
                              f"R={roll['rewards'].sum():.1f} "
                              f"{'COMPLETED' if completed else 'died'}", flush=True)
    finally:
        env.close()
    return buffer


def main():
    parser = argparse.ArgumentParser(description="Collect real transitions for the world model")
    parser.add_argument("--rom", default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--states", default="ds1,ds2,ds3")
    parser.add_argument("--episodes-per-state", type=int, default=8)
    parser.add_argument("--behavior", default="random,ppo",
                        help="comma list of {random,ppo,ppo_noise,explore,probe,"
                             "probe_noise}")
    parser.add_argument("--noise-eps", type=float, default=0.15,
                        help="ppo_noise: probability of replacing the sampled PPO "
                             "action with a random one")
    parser.add_argument("--probe-every", type=int, default=12,
                        help="probe: save a savestate + branch all 6 actions every "
                             "N steps of the PPO descent (counterfactual action data)")
    parser.add_argument("--branch-len", type=int, default=12,
                        help="probe: branch length in steps (probe action + PPO "
                             "recovery); a fall turns black within ~12 steps")
    parser.add_argument("--prefix-len", type=int, default=24,
                        help="probe: descent frames stored before each branch so the "
                             "probe state's belief is warm, matching deployment")
    parser.add_argument("--branch-policy", choices=["ppo", "noop", "hold"],
                        default="ppo",
                        help="probe: continuation after the probe action -- 'ppo' "
                             "(expert recovery; speed contrast), 'noop' (coast; raw "
                             "geometric consequence of the action, the "
                             "cliff-avoidance gradient) or 'hold' (repeat it)")
    parser.add_argument("--ppo-model", default="models/curriculum_flow25_r3_best.zip")
    parser.add_argument("--ppo-deterministic", action="store_true",
                        help="greedy PPO rollouts (reproduces the completing benchmark behavior)")
    parser.add_argument("--ckpt-every", type=int, default=150,
                        help="explore: save a seeding savestate every N steps of the "
                             "PPO descent (positions the random policy along the track)")
    parser.add_argument("--explore-eps", type=int, default=8,
                        help="explore: random episodes per checkpoint")
    parser.add_argument("--explore-max-steps", type=int, default=250,
                        help="explore: per-episode step cap (focuses data on the "
                             "cliff nearest each checkpoint; a surviving episode "
                             "also supplies clean mid-track 'alive' data)")
    parser.add_argument("--out", default="data/wm_buffer.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1350)
    parser.add_argument("--flow-weight", type=float, default=2.5)
    args = parser.parse_args()

    rom_path = os.path.join(ROOT, args.rom)
    states = [s.strip() for s in args.states.split(",") if s.strip()]
    behaviors = [b.strip() for b in args.behavior.split(",") if b.strip()]
    ppo_path = os.path.join(ROOT, args.ppo_model) if args.ppo_model else None
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    buffer = EpisodeBuffer()
    if os.path.exists(out_path):
        buffer.load(out_path)
        print(f"[collect] resumed buffer with {buffer.total_steps} steps / {len(buffer)} episodes")

    collect(rom_path, states, args.episodes_per_state, behaviors, buffer,
            ppo_model=ppo_path, seed=args.seed, max_steps=args.max_steps,
            flow_weight=args.flow_weight, ppo_deterministic=args.ppo_deterministic,
            ckpt_every=args.ckpt_every, explore_eps=args.explore_eps,
            explore_max_steps=args.explore_max_steps, incremental_out=out_path,
            noise_eps=args.noise_eps, probe_every=args.probe_every,
            branch_len=args.branch_len, prefix_len=args.prefix_len,
            branch_policy=args.branch_policy)
    buffer.save(out_path)
    print(f"[collect] saved {buffer.total_steps} steps / {len(buffer)} episodes -> {out_path}")


if __name__ == "__main__":
    main()
