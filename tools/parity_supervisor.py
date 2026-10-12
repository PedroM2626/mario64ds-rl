"""Parity control for the Rust supervisor.

rust/sm64_sup.exe replaced the Python subprocess loop in
scripts/record_dashboard.py, so the port has to prove it runs the same
experiments, not just that it runs. Both arms get the identical flag set and the
identical three-track job list; the comparison is on what each worker reported
(steps, reward, the weight-free flow sum, final RAM state, falls, the
predict/argmax consistency counter), not on wall-clock.

The third check is the one the Python loop never had: a job that overruns its
budget must be killed and reported as such, because a wedged DeSmuME process
otherwise hangs the whole recording batch.

    python tools/parity_supervisor.py --steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDER = os.path.join(ROOT, "scripts", "record_dashboard.py")
RESULT_RE = re.compile(r"^RESULT\t(.*)$", re.MULTILINE)
COMPARE_FIELDS = ["steps", "reward_recorded", "flow_sum_raw", "in_track_falls",
                  "deaths", "completed", "died", "pi_argmax_mismatches",
                  "final_ram_units", "frames_written", "emulator_cycles"]


def digest(path):
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_arm(supervisor, python, steps, tracks, model, out_dir, parallel=1):
    cmd = [python, RECORDER, "--tracks", tracks, "--max-steps", str(steps),
           "--out-dir", out_dir, "--model", model, "--supervisor", supervisor,
           "--run-budget", "300", "--parallel", str(parallel)]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = round(time.time() - t0, 1)
    payload = p.stdout + p.stderr
    results = [json.loads(m) for m in RESULT_RE.findall(payload)]
    return {"cmd": cmd, "exit": p.returncode, "wall_seconds": wall,
            "results": {r["track"]: r for r in results}, "raw": payload}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--tracks", default="ds1,ds2,ds3")
    ap.add_argument("--model", default="models/curriculum_flow25_r3_best.zip")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--budget-test-steps", type=int, default=1350,
                    help="a run this long cannot finish inside --budget seconds")
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--parallel", type=int, default=3,
                    help="workers in the parallel arm (one per track)")
    args = ap.parse_args()

    rom = os.path.join(ROOT, "data", "Super Mario 64 DS (USA) (Rev 1).nds")
    battery = os.path.join(ROOT, "data", "Desmume", "Battery",
                           "Super Mario 64 DS (USA) (Rev 1).dsv")
    save_before = {"rom": digest(rom), "battery": digest(battery)}

    py = run_arm("python", args.python, args.steps, args.tracks, args.model,
                 "out/parity_python")
    rs = run_arm("rust", args.python, args.steps, args.tracks, args.model,
                 "out/parity_rust", parallel=1)
    rp = run_arm("rust", args.python, args.steps, args.tracks, args.model,
                 "out/parity_rust_parallel", parallel=args.parallel)
    save_after = {"rom": digest(rom), "battery": digest(battery)}
    save_unchanged = save_before == save_after

    rows = []
    for track in args.tracks.split(","):
        got = {"python": py["results"].get(track), "rust": rs["results"].get(track),
               "rust_parallel": rp["results"].get(track)}
        base = got["python"]
        if base is None or any(v is None for v in got.values()):
            rows.append({"track": track, "verdict": "missing-result",
                         "present": {k: v is not None for k, v in got.items()}})
            continue
        diffs = {}
        for arm, payload in got.items():
            if arm == "python":
                continue
            diffs.update({f"{arm}.{f}": (base.get(f), payload.get(f))
                          for f in COMPARE_FIELDS if base.get(f) != payload.get(f)})
        rows.append({"track": track, "verdict": "identical" if not diffs else "differs",
                     "fields_compared": len(COMPARE_FIELDS), "diffs": diffs,
                     "reward": base.get("reward_recorded")})
        print(f"[parity] {track}: {rows[-1]['verdict']} "
              f"({base['steps']} steps, R={base['reward_recorded']:+.3f})"
              + (f" diffs={list(diffs)}" if diffs else ""), flush=True)

    speedup = round(rs["wall_seconds"] / rp["wall_seconds"], 2) if rp["wall_seconds"] else None
    print(f"[parity] wall: python {py['wall_seconds']}s | rust sequential "
          f"{rs['wall_seconds']}s | rust parallel x{args.parallel} "
          f"{rp['wall_seconds']}s -> {speedup}x", flush=True)
    print(f"[parity] ROM/battery hashes unchanged by the parallel arm: {save_unchanged}",
          flush=True)

    # the kill path: the rust arm must survive an overrunning worker and still
    # report the remaining jobs
    job = os.path.join(ROOT, "out", "parity_budget_job.tsv")
    with open(job, "w", encoding="utf-8") as f:
        f.write(f"ds1\t{args.budget:g}\t{args.python}\t{RECORDER}\t--single\tds1\t"
                f"--max-steps\t{args.budget_test_steps}\t--out-dir\tout/parity_budget"
                f"\t--model\t{args.model}\t--supervisor\trust\n")
    sup = subprocess.run([os.path.join(ROOT, "rust", "sm64_sup.exe"), job,
                          os.path.join(ROOT, "out", "parity_budget_log.json")],
                         cwd=ROOT, capture_output=True, text=True)
    killed = [json.loads(m) for m in re.findall(r"^RUN\t(.*)$", sup.stdout, re.MULTILINE)]
    kill_row = next((k for k in killed if k.get("name") == "ds1"), None)
    kill_ok = bool(kill_row) and kill_row.get("status") == "killed:over-budget"
    print(f"[parity] budget kill: {kill_row and kill_row.get('status')} "
          f"after {kill_row and kill_row.get('seconds')}s "
          f"(limit {args.budget:g}s) -> {'PASS' if kill_ok else 'FAIL'}", flush=True)

    identical = all(r["verdict"] == "identical" for r in rows)
    report = {"steps_per_run": args.steps, "tracks": args.tracks, "model": args.model,
              "python_cmd": py["cmd"], "rust_cmd": rs["cmd"], "rust_parallel_cmd": rp["cmd"],
              "python_exit": py["exit"], "rust_exit": rs["exit"],
              "wall_seconds": {"python": py["wall_seconds"], "rust": rs["wall_seconds"],
                               "rust_parallel": rp["wall_seconds"],
                               "parallel_workers": args.parallel, "speedup": speedup},
              "save_files": {"before": save_before, "after": save_after,
                             "unchanged": save_unchanged},
              "compared_fields": COMPARE_FIELDS, "rows": rows,
              "kill_test": {"status": kill_row and kill_row.get("status"),
                            "seconds": kill_row and kill_row.get("seconds"),
                            "budget_s": args.budget, "pass": kill_ok,
                            "supervisor_exit": sup.returncode},
              "verdict": ("PASS" if identical and kill_ok and save_unchanged
                          and speedup and speedup > 1.0 else "FAIL")}
    out = os.path.join(ROOT, "results", "supervisor_parity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[parity] {report['verdict']}: {sum(r['verdict'] == 'identical' for r in rows)}"
          f"/{len(rows)} tracks identical, kill path "
          f"{'ok' if kill_ok else 'BROKEN'} -> {out}", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
