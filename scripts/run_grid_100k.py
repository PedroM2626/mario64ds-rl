"""Fair 100k benchmark grid: (PPO, Rainbow, QRDQN) x (nature, impala) x seeds 0,1,2 = 18 runs.

Executes all training configurations sequentially in isolated subprocesses to prevent
DeSmuME memory access violations. Upon training completion, runs deterministic evaluations
(3 episodes per savestate) and logs metrics to a consolidated results CSV in results/.

Usage:
    python scripts/run_grid_100k.py
    python scripts/run_grid_100k.py --timesteps 100000 --n-episodes 3
    python scripts/run_grid_100k.py --dry-run
"""

import argparse
import csv
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PY = sys.executable

ALGOS = ["ppo", "rainbow", "qrdqn"]
FEATURES = ["nature", "impala"]
SEEDS = [0, 1, 2]

RESULTS_CSV = os.path.join(BASE, "results", "results_grid_100k.csv")


def run_id(algo, feat, seed, timesteps):
    k = timesteps // 1000
    return f"grid_{algo}_{feat}_s{seed}_{k}k"


def train_cmd(algo, feat, seed, timesteps, n_envs, flow_weight=1.0, device="auto"):
    rid = run_id(algo, feat, seed, timesteps)
    common = ["--flow-weight", str(flow_weight), "--device", device]
    if algo == "ppo":
        return [PY, "-m", "src.train_ppo", "--run-id", rid,
                "--features", feat, "--n-envs", str(n_envs),
                "--timesteps", str(timesteps), "--seed", str(seed),
                *common], rid
    if algo == "rainbow":
        return [PY, "-m", "src.train", "--run-id", rid,
                "--features", feat, "--timesteps", str(timesteps),
                "--seed", str(seed), *common], rid
    if algo == "qrdqn":
        return [PY, "-m", "src.train_qrdqn", "--run-id", rid,
                "--features", feat, "--n-envs", str(n_envs),
                "--timesteps", str(timesteps), "--seed", str(seed),
                *common], rid
    raise ValueError(algo)


def model_path(algo, rid):
    models = os.path.join(BASE, "models")
    if algo in ("ppo", "qrdqn"):
        # EvalCallback saves <rid>/best_model.zip; train saves <rid>_final.zip and <rid>_best.zip
        for cand in [os.path.join(models, f"{rid}_best.zip"),
                     os.path.join(models, rid, "best_model.zip"),
                     os.path.join(models, f"{rid}_final.zip")]:
            if os.path.exists(cand):
                return cand
        return os.path.join(models, f"{rid}_best.zip")
    # rainbow
    for cand in [os.path.join(models, f"{rid}_best.pth"),
                 os.path.join(models, f"{rid}_final.pth")]:
        if os.path.exists(cand):
            return cand
    return os.path.join(models, f"{rid}_best.pth")


def eval_cmd(algo, feat, model, seed, n_episodes, out_csv,
             flow_weight=1.0, device="cpu"):
    cmd = [PY, "-m", "src.eval", "--algo", algo, "--model", model,
           "--n-episodes", str(n_episodes), "--deterministic",
           "--seed", str(seed), "--out", out_csv,
           "--flow-weight", str(flow_weight), "--device", device]
    if algo == "rainbow":
        cmd += ["--features", feat]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Fair benchmark grid across algorithms, architectures, and seeds")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--n-envs", type=int, default=2,
                    help="n-envs for PPO/QRDQN (Rainbow uses fixed 2: ds1+ds3)")
    ap.add_argument("--n-episodes", type=int, default=3,
                    help="Episodes per savestate in evaluation")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--results-csv", type=str, default=None,
                    help="Consolidated CSV path (default: results/results_grid_<timesteps//1000>k.csv)")
    ap.add_argument("--algos", type=str, default="ppo,rainbow,qrdqn",
                    help="Comma-separated algorithms (e.g. 'ppo,rainbow')")
    ap.add_argument("--features-list", type=str, default="nature,impala",
                    help="Comma-separated feature extractors (e.g. 'nature,impala')")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="Comma-separated seeds (e.g. '0,1,2')")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip run_ids that already have >=6 valid evaluation rows in results CSV")
    ap.add_argument("--only", type=str, default="",
                    help="Substring filter for run_id: e.g. 'qrdqn' or 'rainbow_nature_s1'")
    ap.add_argument("--flow-weight", type=float, default=1.0,
                    help="Training flow weight passed to ALL algorithms. The historical "
                         "100k/500k grid used 1.0; the current PPO default is 2.5, so this "
                         "flag is required for a fair or reproducible comparison.")
    ap.add_argument("--eval-flow-weight", type=float, default=1.0,
                    help="Flow weight used by src.eval (independent evaluation)")
    ap.add_argument("--savestate-idxs", type=str, default="0,2",
                    help="Comma-separated savestate indices to evaluate. The committed "
                         "100k/500k CSVs contain ds1 and ds3 (0 and 2), not ds2.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device passed to training and evaluation. cpu reproduces the "
                         "published ds3 result; auto/cuda study the divergent arm.")
    args = ap.parse_args()

    results_dir = os.path.join(BASE, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv = args.results_csv or os.path.join(results_dir, f"results_grid_{args.timesteps // 1000}k.csv")
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    feats = [f.strip() for f in args.features_list.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    savestate_idxs = [int(s) for s in args.savestate_idxs.split(",") if s.strip() != ""]
    if any(i not in (0, 1, 2) for i in savestate_idxs) or not savestate_idxs:
        raise SystemExit("--savestate-idxs must be a non-empty subset of 0,1,2")

    plan = [(a, f, s) for a in algos for f in feats for s in seeds]
    if args.only:
        plan = [t for t in plan if args.only in f"{t[0]}_{t[1]}_s{t[2]}"]
    if args.skip_done:
        import csv as _csv
        from collections import Counter as _C
        cnt = _C()
        if os.path.exists(results_csv):
            with open(results_csv) as fh:
                cnt = _C(r["run_id"] for r in _csv.DictReader(fh) if r["reward"] != "")
        plan = [t for t in plan
                if run_id(t[0], t[1], t[2], args.timesteps) not in
                {k for k, v in cnt.items() if v >= len(savestate_idxs) * args.n_episodes}]
    print(f"Grid: {len(plan)} runs | timesteps={args.timesteps} | n_envs={args.n_envs} | "
          f"eval_eps={args.n_episodes} | states={savestate_idxs} | "
          f"flow={args.flow_weight} | device={args.device}")
    for i, (a, f, s) in enumerate(plan, 1):
        cmd, rid = train_cmd(a, f, s, args.timesteps, args.n_envs,
                             flow_weight=args.flow_weight, device=args.device)
        print(f"[{i}/{len(plan)}] {rid}: {' '.join(cmd)}")
        if args.dry_run:
            for sidx in savestate_idxs:
                mp = model_path(a, rid)
                ec = eval_cmd(a, f, mp, s, args.n_episodes,
                              f"eval_{rid}_ds{sidx}.csv",
                              flow_weight=args.eval_flow_weight,
                              device=args.device) + ["--savestate-idx", str(sidx)]
                print(f"      eval ds{sidx}: {' '.join(ec)}")

    if args.dry_run:
        return

    # Consolidated CSV header. Refuse to append a new schema to a legacy file:
    # the committed 100k/500k CSVs predate the flow/device columns, and a mixed
    # header would make the artifact unparseable.
    fieldnames = ["run_id", "algo", "features", "seed", "timesteps",
                  "train_status", "train_seconds", "flow_weight",
                  "eval_flow_weight", "device",
                  "savestate", "episode", "reward", "steps", "survived",
                  "eval_csv"]
    new_file = not os.path.exists(results_csv)
    if not new_file:
        with open(results_csv, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != fieldnames:
            # Keep the historical artifact intact and write the new schema to a
            # sidecar instead of mixing headers or silently dropping provenance.
            legacy = results_csv
            results_csv = os.path.splitext(legacy)[0] + "_v2.csv"
            new_file = not os.path.exists(results_csv)
            print(f"[grid] {legacy} has a legacy header; writing "
                  f"{results_csv} instead.", flush=True)
    fout = open(results_csv, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if new_file:
        w.writeheader()
        fout.flush()

    for i, (a, f, s) in enumerate(plan, 1):
        cmd, rid = train_cmd(a, f, s, args.timesteps, args.n_envs,
                             flow_weight=args.flow_weight, device=args.device)
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
        print(f"TRAIN {rid}: {status} in {dt:.0f}s", flush=True)

        mp = model_path(a, rid)
        if not os.path.exists(mp):
            print(f"EVAL {rid}: SKIP (model checkpoint not found: {mp})", flush=True)
            w.writerow({"run_id": rid, "algo": a, "features": f, "seed": s,
                        "timesteps": args.timesteps, "train_status": status,
                        "train_seconds": f"{dt:.0f}", "flow_weight": args.flow_weight,
                        "eval_flow_weight": args.eval_flow_weight, "device": args.device,
                        "savestate": "", "episode": "", "reward": "", "steps": "",
                        "survived": "", "eval_csv": ""})
            fout.flush()
            continue

        # Evaluate per savestate in isolated subprocesses
        for sidx in savestate_idxs:
            out_csv = os.path.join(results_dir, f"eval_{rid}_ds{sidx}.csv")
            ec = eval_cmd(a, f, mp, s, args.n_episodes, out_csv,
                          flow_weight=args.eval_flow_weight,
                          device=args.device) + ["--savestate-idx", str(sidx)]
            print(f"----- EVAL {rid} ds{sidx} -----", flush=True)
            try:
                r = subprocess.run(ec, cwd=BASE)
                if r.returncode != 0:
                    print(f"EVAL {rid} ds{sidx}: exit={r.returncode}", flush=True)
            except Exception as e:
                print(f"EVAL {rid} ds{sidx} error: {e}", flush=True)
            if os.path.exists(out_csv):
                with open(out_csv) as fh:
                    for row in csv.DictReader(fh):
                        w.writerow({"run_id": rid, "algo": a, "features": f, "seed": s,
                                    "timesteps": args.timesteps, "train_status": status,
                                    "train_seconds": f"{dt:.0f}",
                                    "flow_weight": args.flow_weight,
                                    "eval_flow_weight": args.eval_flow_weight,
                                    "device": args.device,
                                    "savestate": row["savestate"], "episode": row["episode"],
                                    "reward": row["reward"], "steps": row["steps"],
                                    "survived": row["survived"], "eval_csv": os.path.basename(out_csv)})
                        fout.flush()
    fout.close()
    print(f"\nGrid completed. Consolidated results in {results_csv}")


if __name__ == "__main__":
    main()
