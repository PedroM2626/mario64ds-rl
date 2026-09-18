"""Curriculum learning: aprende as 3 pistas progressivamente sem esquecer.

Evidência que motiva (ver README §Generalização 3 pistas):
- Fine-tune do 2-way 500k com ds2 no mix em lr=3e-4 → esquecimento
  catastrófico de ds1 (experimento A).
- 2M total em lr=3e-4 → 0/3 determinístico (experimento D): steps não
  resolvem; a interferência na CNN compartilhada é o gargalo.

Estratégia do curriculum:
1. Partir do melhor modelo (2-way 500k: ds1✓ ds3✓).
2. Fases com LR DECRESCENTE (1e-4 → 5e-5 → 2.5e-5): updates menores
   preservam as pistas já aprendidas enquanto aprendem a nova.
3. Todas as pistas no mix em TODAS as fases (todo update vê as 3).
4. Seleção do melhor checkpoint BALANCEADO: EvalCallback roda eval
   determinístico nas 3 pistas (vec env com 1 env por pista) e salva o
   checkpoint com maior recompensa MÉDIA — um especialista de pista única
   tem média ~-33 (sobrevive 1, morre 2); um modelo que sobrevive as 3 tem
   média ~+63. A média seleciona o balanceado.
5. Para cedo quando todas as 3 pistas sobrevivem.

Uso:
    python -m src.train_curriculum --run-id ppo_curriculum
    python -m src.train_curriculum --run-id ppo_curriculum \
        --start-model models/grid_ppo_nature_s0_500k_best.zip \
        --phases "200000:1e-4,200000:5e-5,200000:2.5e-5"
"""

import argparse
import csv
import os

import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

from src.env import Mario64DSEnv
from src.train_ppo import MLflowCallback

PHASES_DEFAULT = "200000:1e-4,200000:5e-5,200000:2.5e-5"


def make_env(rom_path, state_path, rank=0, seed=0, max_steps=900, frameskip=4):
    def _init():
        env = Mario64DSEnv(
            rom_path=rom_path, state_path=state_path,
            max_steps=max_steps, frameskip=frameskip,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec(rom_path, states, rank0, seed, max_steps, frameskip, n_stack=4):
    vec = SubprocVecEnv([
        make_env(rom_path, sp, rank=rank0 + i, seed=seed,
                 max_steps=max_steps, frameskip=frameskip)
        for i, sp in enumerate(states)
    ])
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecTransposeImage(vec)
    return vec


def eval_pistas(model_path, rom_path, states, seed, n_eps=3, max_steps=900):
    """Eval determinístico por pista (SubprocVecEnv sequencial — seguro)."""
    from src.eval import _load_sb3_with_legacy_patch
    model = _load_sb3_with_legacy_patch(PPO, model_path, device="auto")

    rows = []
    for s_idx, sp in enumerate(states):
        vec = build_vec(rom_path, [sp], 2000 + s_idx, seed, max_steps, 4)
        for ep in range(n_eps):
            obs = vec.reset()
            done, ep_rew, steps = False, 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rews, dones, infos = vec.step(action)
                ep_rew += float(rews[0])
                steps += 1
                done = bool(dones[0])
            rows.append({"pista": f"ds{s_idx + 1}", "episode": ep,
                         "reward": ep_rew, "steps": steps,
                         "survived": steps >= max_steps})
        vec.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Curriculum learning nas 3 pistas")
    parser.add_argument("--rom", type=str, default="data/Super Mario 64 DS (USA) (Rev 1).nds")
    parser.add_argument("--run-id", type=str, default="ppo_curriculum")
    parser.add_argument("--start-model", type=str, default="models/grid_ppo_nature_s0_500k_best.zip",
                        help="Modelo inicial (default: 2-way 500k, ds1✓ ds3✓)")
    parser.add_argument("--states", type=str, default="ds1,ds2,ds3")
    parser.add_argument("--phases", type=str, default=PHASES_DEFAULT,
                        help="Fases 'steps:lr' separadas por vírgula (LR decrescente)")
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Steps entre evals do EvalCallback (deve ser < steps da fase)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    phases = []
    for token in args.phases.split(","):
        steps_s, lr_s = token.strip().split(":")
        phases.append((int(steps_s), float(lr_s)))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_full = os.path.join(base_dir, args.rom)
    states = [
        os.path.join(base_dir, "data", f"Super Mario 64 DS (USA) (Rev 1).{s.strip()}")
        for s in args.states.split(",") if s.strip()
    ]
    for sp in states:
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Savestate não encontrado: {sp}")
    if not os.path.exists(os.path.join(base_dir, args.start_model)):
        raise FileNotFoundError(f"Modelo inicial não encontrado: {args.start_model}")

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
    mlflow.set_experiment("Mario64_NDS_RL")

    models_dir = os.path.join(base_dir, "models")
    results_csv = os.path.join(base_dir, "results_curriculum.csv")

    all_rows = []
    current_model = args.start_model
    global_best_path, global_best_mean = None, -np.inf

    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO_Nature_Curriculum")
        mlflow.log_param("start_model", args.start_model)
        mlflow.log_param("states", args.states)
        mlflow.log_param("phases", args.phases)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("n_envs", args.n_envs)

        for i, (steps, lr) in enumerate(phases, 1):
            phase_id = f"{args.run_id}_p{i}"
            print(f"\n===== FASE {i}/{len(phases)}: {steps} steps @ lr={lr} "
                  f"(resume: {current_model}) =====", flush=True)

            train_envs = build_vec(rom_full, states, 0, args.seed, 450, 4, n_stack=4)
            # Eval env com TODAS as pistas (1 env por pista): a recompensa média
            # do EvalCallback seleciona o checkpoint balanceado.
            eval_env = build_vec(rom_full, states, 1000 + i, args.seed, 450, 4, n_stack=4)

            model = PPO.load(current_model, env=train_envs, device="auto")
            # LR da fase: PPO.load restaura o schedule salvo; sobrescrevemos
            # com o LR decrescente do curriculum.
            model.learning_rate = lr
            model.lr_schedule = lambda _: lr
            print(f"Device: {model.device} | lr fase: {lr}", flush=True)

            best_model_dir = os.path.join(models_dir, phase_id)
            os.makedirs(best_model_dir, exist_ok=True)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=best_model_dir,
                log_path=os.path.join(base_dir, "tensorboard_logs", phase_id),
                eval_freq=args.eval_freq,
                n_eval_episodes=3,   # 3 eps × 3 envs = 9 eps por eval
                deterministic=True,  # sobrevivência é o que importa
                render=False,
            )

            try:
                model.learn(total_timesteps=steps,
                            callback=[eval_callback, MLflowCallback()],
                            progress_bar=True,
                            reset_num_timesteps=False)
            finally:
                train_envs.close()
                eval_env.close()

            # Eval determinístico por pista do melhor checkpoint da fase
            phase_best = os.path.join(best_model_dir, "best_model.zip")
            if not os.path.exists(phase_best):
                print(f"FASE {i}: best_model.zip não encontrado; pulando eval", flush=True)
                continue

            rows = eval_pistas(phase_best, rom_full, states, args.seed, n_eps=3,
                               max_steps=900)
            pista_means = {}
            for pista in sorted(set(r["pista"] for r in rows)):
                vv = [r for r in rows if r["pista"] == pista]
                pista_means[pista] = (np.mean([r["reward"] for r in vv]),
                                      sum(1 for r in vv if r["survived"]), len(vv))
            balanced_mean = float(np.mean([m[0] for m in pista_means.values()]))
            all_survived = all(m[1] == m[2] for m in pista_means.values())

            for pista, (mean_r, surv, n) in pista_means.items():
                print(f"  {pista}: {mean_r:.2f} · surv {surv}/{n}", flush=True)
                mlflow.log_metric(f"{pista}_reward", mean_r, step=i)
                mlflow.log_metric(f"{pista}_surv", surv / n, step=i)
            mlflow.log_metric("balanced_mean", balanced_mean, step=i)
            print(f"  MÉDIA BALANCEADA: {balanced_mean:.2f} | todas sobrevivem: {all_survived}", flush=True)

            for r in rows:
                r["phase"] = i
                r["lr"] = lr
                all_rows.append(r)

            if balanced_mean > global_best_mean:
                global_best_mean = balanced_mean
                global_best_path = phase_best

            # Próxima fase parte do melhor balanceado desta
            current_model = phase_best

            if all_survived:
                print(f"\n🎉 CONVERGIU na fase {i}: todas as 3 pistas sobrevivem!", flush=True)
                break

        # Consolidar CSV
        with open(results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "lr", "pista", "episode",
                                              "reward", "steps", "survived"])
            w.writeheader()
            w.writerows(all_rows)

        # Salvar o melhor global
        if global_best_path:
            import shutil
            final_dst = os.path.join(models_dir, f"{args.run_id}_best.zip")
            shutil.copy2(global_best_path, final_dst)
            mlflow.log_artifact(final_dst, artifact_path="models")
            print(f"\nMelhor modelo balanceado: {final_dst} (média {global_best_mean:.2f})", flush=True)

        print(f"Curriculum concluído. CSV: {results_csv}", flush=True)


if __name__ == "__main__":
    main()
