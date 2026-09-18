"""Grava 1 vídeo por fase (ds1, ds2, ds3) mostrando o agente treinado jogando.

Cada fase roda em um subprocesso isolado (o DeSmuME não suporta múltiplos
emuladores no mesmo processo — access violation). Grava a tela real do jogo
em cores (256x192) via env.get_screen_rgb(), não a observação grayscale do
agente. Saída: videos/phase_1.mp4, phase_2.mp4, phase_3.mp4.

Exemplos:
    python record_phases.py --model models/grid_ppo_nature_s0_500k_best.zip
    python record_phases.py --model models/grid_ppo_nature_s0_500k_best.zip --fps 15
"""

import argparse
import os
import subprocess
import sys


PHASES = ["ds1", "ds2", "ds3"]


def main():
    parser = argparse.ArgumentParser(description="Grava vídeos do agente jogando cada fase")
    parser.add_argument("--model", type=str, required=True,
                        help="Caminho para modelo SB3 (.zip)")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS do vídeo (frameskip=4 sobre 60 Hz -> 15 = tempo real)")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--single", type=str, default=None,
                        help=argparse.SUPPRESS)  # uso interno: grava 1 fase e sai
    args = parser.parse_args()

    if args.single is None:
        # Modo orquestrador: 1 subprocesso por fase (isola o DeSmuME)
        os.makedirs(args.out_dir, exist_ok=True)
        for phase in PHASES:
            out = os.path.join(args.out_dir, f"phase_{phase}.mp4")
            print(f"\n=== Gravando {phase} -> {out} ===", flush=True)
            r = subprocess.run(
                [sys.executable, __file__, "--model", args.model, "--rom", args.rom,
                 "--out-dir", args.out_dir, "--fps", str(args.fps),
                 "--max-steps", str(args.max_steps), "--single", phase],
            )
            if r.returncode != 0:
                print(f"AVISO: gravação de {phase} falhou (exit={r.returncode})", flush=True)
        return

    # Modo single: grava UMA fase neste processo
    _record_single(args)


def _record_single(args):
    import imageio
    import numpy as np
    from stable_baselines3 import PPO

    from src.env import Mario64DSEnv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(base_dir, args.rom)
    savestate_path = os.path.join(base_dir, "data", f"Super Mario 64 DS (USA) (Rev 1).{args.single}")
    out_path = os.path.join(args.out_dir, f"phase_{args.single}.mp4")

    model = PPO.load(args.model)

    env = Mario64DSEnv(rom_path=rom_path, state_path=savestate_path)
    obs, info = env.reset()

    # FrameStack REAL (igual ao treino via FrameStackObservation): deque de 4
    # frames sucessivos, inicializado com 4 cópias do frame de reset. Repetir
    # o frame atual 4x remove a noção de movimento e muda o comportamento do
    # modelo (agente morre) — por isso o stack deslizante é obrigatório.
    from collections import deque
    stack_deque = deque([obs[:, :, 0]] * 4, maxlen=4)

    def make_stack():
        # (4, 84, 84) CHW — mesmo formato do treino (VecFrameStack+VecTransposeImage)
        return np.stack(list(stack_deque), axis=0)[np.newaxis, :]

    frames = [env.get_screen_rgb()]
    done, steps, total_reward = False, 0, 0.0
    while not done and steps < args.max_steps:
        action, _ = model.predict(make_stack(), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        stack_deque.append(obs[:, :, 0])
        total_reward += float(reward)
        done = bool(terminated or truncated)
        frames.append(env.get_screen_rgb())
        steps += 1

    env.close()

    # frameskip=4: o env roda a 15 passos/s; fps=15 = playback em tempo real
    with imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    survived = steps >= args.max_steps
    print(f"{args.single}: reward={total_reward:.2f}, steps={steps}, "
          f"{'TIMEOUT (sobreviveu)' if survived else 'MORTE'} | "
          f"{out_path} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
