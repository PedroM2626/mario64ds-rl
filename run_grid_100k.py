"""Grade justa 100k: (PPO, Rainbow, QRDQN) x (nature, impala) x seeds 0,1,2 = 18 runs.

Roda tudo em SEQUÊNCIA (nunca em paralelo: DeSmuME dá access violation
com emuladores demais). Cada treino é um subprocesso isolado; falha num
não aborta a grade. Ao final de cada treino, roda eval determinístico
(3 eps por savestate) e anexa em results_grid_100k.csv.

Uso:
    python run_grid_100k.py
    python run_grid_100k.py --timesteps 100000 --n-episodes 3
    python run_grid_100k.py --dry-run   (só imprime comandos)
"""

import argparse
import csv
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

ALGOS = ["ppo", "rainbow", "qrdqn"]
FEATURES = ["nature", "impala"]
SEEDS = [0, 1, 2]

RESULTS_CSV = os.path.join(BASE, "results_grid_100k.csv")


def run_id(algo, feat, seed, timesteps):
    k = timesteps // 1000
    return f"grid_{algo}_{feat}_s{seed}_{k}k"


def train_cmd(algo, feat, seed, timesteps, n_envs):
    rid = run_id(algo, feat, seed, timesteps)
    if algo == "ppo":
        return [PY, "-m", "src.train_ppo", "--run-id", rid,
                "--features", feat, "--n-envs", str(n_envs),
                "--timesteps", str(timesteps), "--seed", str(seed)], rid
    if algo == "rainbow":
        return [PY, "-m", "src.train", "--run-id", rid,
                "--features", feat, "--timesteps", str(timesteps),
                "--seed", str(seed)], rid
    if algo == "qrdqn":
        return [PY, "-m", "src.train_qrdqn", "--run-id", rid,
                "--features", feat, "--n-envs", str(n_envs),
                "--timesteps", str(timesteps), "--seed", str(seed)], rid
    raise ValueError(algo)


def model_path(algo, rid):
    models = os.path.join(BASE, "models")
    if algo in ("ppo", "qrdqn"):
        # EvalCallback salva <rid>/best_model.zip; train salva <rid>_final.zip e <rid>_best.zip
        for cand in [os.path.join(models, f"{rid}_best.zip"),
                     os.path.join(models, rid, "best_model.zip"),
                     os.path.join(models, f"{rid}_final.zip")]:
            if os.path.exists(cand):
                return cand
        return os.path.join(models, f"{rid}_best.zip")  # esperado (pode não existir se falhou)
    # rainbow
    for cand in [os.path.join(models, f"{rid}_best.pth"),
                 os.path.join(models, f"{rid}_final.pth")]:
        if os.path.exists(cand):
            return cand
    return os.path.join(models, f"{rid}_best.pth")


def eval_cmd(algo, feat, model, seed, n_episodes, out_csv):
    cmd = [PY, "-m", "src.eval", "--algo", algo, "--model", model,
           "--n-episodes", str(n_episodes), "--deterministic",
           "--seed", str(seed), "--out", out_csv]
    if algo == "rainbow":
        cmd += ["--features", feat]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--n-envs", type=int, default=2,
                    help="n-envs p/ PPO/QRDQN (Rainbow usa 2 fixo: ds1+ds3)")
    ap.add_argument("--n-episodes", type=int, default=3,
                    help="eps por savestate no eval")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-csv", type=str, default=None,
                    help="CSV consolidado (default: results_grid_<timesteps//1000>k.csv)")
    ap.add_argument("--algos", type=str, default="ppo,rainbow,qrdqn",
                    help="Algos separados por vírgula (ex. 'ppo,rainbow')")
    ap.add_argument("--features-list", type=str, default="nature,impala",
                    help="Features separados por vírgula (ex. 'nature')")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="Seeds separadas por vírgula (ex. '0')")
    ap.add_argument("--skip-done", action="store_true",
                    help="Pula run_ids que já têm 6 linhas ok no results CSV")
    ap.add_argument("--only", type=str, default="",
                    help="Filtro legado por substring: ex. 'qrdqn' ou 'rainbow_nature_s1'")
    args = ap.parse_args()

    results_csv = args.results_csv or os.path.join(BASE, f"results_grid_{args.timesteps // 1000}k.csv")
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    feats = [f.strip() for f in args.features_list.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    plan = [(a, f, s) for a in algos for f in feats for s in seeds]
    if args.only:
        plan = [t for t in plan if args.only in f"{t[0]}_{t[1]}_s{t[2]}"]
    if args.skip_done:
        import csv as _csv
        # considera done quem já tem >=6 linhas (2 savestates x 3 eps)
        from collections import Counter as _C
        cnt = _C()
        if os.path.exists(results_csv):
            with open(results_csv) as fh:
                cnt = _C(r["run_id"] for r in _csv.DictReader(fh) if r["reward"] != "")
        plan = [t for t in plan
                if run_id(t[0], t[1], t[2], args.timesteps) not in
                {k for k, v in cnt.items() if v >= 6}]
    print(f"Grade: {len(plan)} runs | timesteps={args.timesteps} | n_envs={args.n_envs} | eval_eps={args.n_episodes}")
    for i, (a, f, s) in enumerate(plan, 1):
        cmd, rid = train_cmd(a, f, s, args.timesteps, args.n_envs)
        print(f"[{i}/{len(plan)}] {rid}: {' '.join(cmd)}")
        if args.dry_run:
            mp = model_path(a, rid)
            ec = eval_cmd(a, f, mp, s, args.n_episodes, f"eval_{rid}.csv")
            print(f"      eval: {' '.join(ec)}")

    if args.dry_run:
        return

    # CSV consolidado
    new_file = not os.path.exists(results_csv)
    fout = open(results_csv, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=["run_id", "algo", "features", "seed", "timesteps",
                                         "train_status", "train_seconds",
                                         "savestate", "episode", "reward", "steps", "survived",
                                         "eval_csv"])
    if new_file:
        w.writeheader()
        fout.flush()

    for i, (a, f, s) in enumerate(plan, 1):
        cmd, rid = train_cmd(a, f, s, args.timesteps, args.n_envs)
        print(f"\n===== [{i}/{len(plan)}] TRAIN {rid} =====", flush=True)
        t0 = time.time()
        status = "ok"
        try:
            r = subprocess.run(cmd, cwd=BASE)
            if r.returncode != 0:
                status = f"exit={r.returncode}"
        except Exception as e:
            status = f"error: {e}"
        dt = time.time() - t0
        print(f"TRAIN {rid}: {status} em {dt:.0f}s", flush=True)

        mp = model_path(a, rid)
        if not os.path.exists(mp):
            print(f"EVAL {rid}: SKIP (modelo não encontrado: {mp})", flush=True)
            w.writerow({"run_id": rid, "algo": a, "features": f, "seed": s,
                        "timesteps": args.timesteps, "train_status": status,
                        "train_seconds": f"{dt:.0f}", "savestate": "", "episode": "",
                        "reward": "", "steps": "", "survived": "", "eval_csv": ""})
            fout.flush()
            continue

        # Eval por savestate em processos separados (isola DeSmuME)
        for sidx in (0, 1):
            out_csv = os.path.join(BASE, f"eval_{rid}_ds{sidx}.csv")
            ec = eval_cmd(a, f, mp, s, args.n_episodes, out_csv) + ["--savestate-idx", str(sidx)]
            print(f"----- EVAL {rid} ds{sidx} -----", flush=True)
            try:
                r = subprocess.run(ec, cwd=BASE)
                if r.returncode != 0:
                    print(f"EVAL {rid} ds{sidx}: exit={r.returncode}", flush=True)
            except Exception as e:
                print(f"EVAL {rid} ds{sidx} error: {e}", flush=True)
            # anexa linhas do CSV parcial
            if os.path.exists(out_csv):
                with open(out_csv) as fh:
                    for row in csv.DictReader(fh):
                        w.writerow({"run_id": rid, "algo": a, "features": f, "seed": s,
                                    "timesteps": args.timesteps, "train_status": status,
                                    "train_seconds": f"{dt:.0f}",
                                    "savestate": row["savestate"], "episode": row["episode"],
                                    "reward": row["reward"], "steps": row["steps"],
                                    "survived": row["survived"], "eval_csv": os.path.basename(out_csv)})
                        fout.flush()
    fout.close()
    print(f"\nGrade concluída. Consolidado em {results_csv}")


if __name__ == "__main__":
    main()
