"""Gate: every number quoted in the README showcase block and in rust/README.md
must equal what the recorded runs actually displayed, or what the code says.

This project has already published a table cell that no run produced, so prose
figures are not trusted here. Each check re-derives its expected value from the
run artifacts (videos/dashboard/dashboard_*.json, results/supervisor_parity.json,
results/step_profile.json) or from the source itself -- not from the README.

    python tools/check_readme_numbers.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
RUST_README = os.path.join(ROOT, "rust", "README.md")
DASH_DIR = os.path.join(ROOT, "videos", "dashboard")
PARITY = os.path.join(ROOT, "results", "supervisor_parity.json")
PROFILE = os.path.join(ROOT, "results", "step_profile.json")
DIVERGENCE = os.path.join(ROOT, "results", "device_divergence.json")

CHECKS = []


def check(label, expected, got, ok=None):
    ok = (str(expected) == str(got)) if ok is None else ok
    CHECKS.append((label, expected, got, bool(ok)))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reprice(payload, weight):
    """Independent re-derivation of the episode return from the logged terms:
    step cost + weight * (weight-free flow sum) - death/timeout penalties."""
    return (-payload["step_penalty"] * payload["steps"]
            + weight * payload["flow_sum_raw"]
            - payload["death_penalty"] * payload["deaths"]
            - payload["timeout_penalty"] * (1 if payload["died"] else 0))


def table_row(readme, track):
    for ln in readme.splitlines():
        if f"`{track}`" in ln and ln.strip().startswith("|"):
            return [c.strip() for c in ln.strip().strip("|").split("|")]
    return None


def showcase(readme, payloads):
    for track, p in sorted(payloads.items()):
        row = table_row(readme, track)
        if row is None:
            check(f"README row for {track}", "present", "missing")
            continue
        joined = " | ".join(row)
        steps_cell = next((c for c in row if re.fullmatch(r"\d+ / \d+", c)), None)
        check(f"{track}: steps cell", f"{p['steps']} / {p['max_steps']}", steps_cell)
        for w in (1.0, 2.5):
            val = reprice(p, w)
            cell = next((c for c in row if re.fullmatch(r"[+-]\d+\.\d\d", c)
                         and abs(float(c) - val) < 0.006), None)
            check(f"{track}: R at flow x{w:g} cell", f"{val:+.2f}", cell)
            if w == 1.0:
                check(f"{track}: R x1.0 cell equals the run's own total",
                      f"{p['reward_recorded']:+.2f}", f"{val:+.2f}",
                      ok=abs(val - p["reward_recorded"]) < 0.006)
        check(f"{track}: clip duration cell", f"{p['seconds']:.1f} s",
              f"{p['seconds']:.1f} s", ok=f"{p['seconds']:.1f} s" in joined)
        check(f"{track}: fps cell", f"{p['fps']} fps", f"{p['fps']} fps",
              ok=f"{p['fps']} fps" in joined)
        check(f"{track}: run completed", True, p["completed"], ok=bool(p["completed"]))
        check(f"{track}: decomposition residual < 1e-4", "<1e-4",
              f"{p['decomposition_residual']:.1e}",
              ok=p["decomposition_residual"] < 1e-4)
        check(f"{track}: predict() == argmax(pi) on every decision", 0,
              p["pi_argmax_mismatches"], ok=p["pi_argmax_mismatches"] == 0)
        check(f"{track}: flow-sum re-pricing agrees with the payload",
              f"{p['reward_at_flow_weight']['2.5']:+.2f}", f"{reprice(p, 2.5):+.2f}",
              ok=abs(p["reward_at_flow_weight"]["2.5"] - reprice(p, 2.5)) < 0.006)
        check(f"{track}: policy device cell names the instrument", p["policy_device"],
              p["policy_device"], ok=p["policy_device"] in joined)
    hdr = [ln for ln in readme.splitlines() if "single recorded episode" in ln]
    check("showcase column labels the reward as one episode, not the 3-episode mean",
          "labelled", "found" if hdr else "MISSING", ok=bool(hdr))


def repro_checks(readme):
    """The ds3 finding: the README must carry the measured split, not a verdict."""
    if not os.path.exists(DIVERGENCE):
        check("out/device_divergence.json", "present",
              "MISSING (run tools/probe_device_divergence.py)")
        return
    rep = load(DIVERGENCE)["tracks"]
    for track, row in sorted(rep.items()):
        if "error" in row:
            continue
        cuda, cpu = row["cuda"], row["cpu"]
        if cuda["steps"] == cpu["steps"]:
            continue
        want = [str(cuda["steps"]), str(cpu["steps"]), f"{cpu['reward']:+.2f}"]
        seg = _track_paragraph(readme, track)
        for w in want:
            check(f"{track}: README states the {w} arm figure", "quoted",
                  "found" if w in seg else "MISSING", ok=w in seg)
        div = row["first_divergent_decision"]
        if div is not None:
            check(f"{track}: README states the first divergent decision",
                  "quoted", "found" if str(div) in seg else "MISSING", ok=str(div) in seg)
        m = row.get("margin_at_divergence")
        if m is not None:
            pat = f"{m:.4f}" if m >= 0.001 else f"{m:.6f}"
            tail = pat.lstrip("0.")
            check(f"{track}: README states the decision margin ({pat})", "quoted",
                  "found" if tail in seg else "MISSING", ok=tail in seg)


def _track_paragraph(readme, track):
    """The README text around a track id: the row plus the notes under it.

    Anchored on the device-divergence section when present, so prose added
    earlier in the README cannot push the paragraph out of the window."""
    start = max(readme.find("One checkpoint, two outcomes"), 0)
    i = readme.find(f"`{track}`", start)
    return readme[max(0, i - 2000):i + 4000] if i >= 0 else ""


def parallel_checks(readme):
    rust_readme = open(RUST_README, encoding="utf-8").read() if os.path.exists(
        RUST_README) else ""
    body = readme + rust_readme
    if not os.path.exists(PARITY):
        check("results/supervisor_parity.json", "present",
              "MISSING (run tools/parity_supervisor.py --steps 40)")
        return
    rep = load(PARITY)
    wall = rep.get("wall_seconds", {})
    sp = wall.get("speedup")
    if sp:
        check(f"prose quotes the parallel speedup {sp}x", "quoted",
              "found" if re.search(rf"{sp}\s*[x×]", body) else "MISSING",
              ok=bool(re.search(rf"{sp}\s*[x×]", body)))
    for arm in ("python", "rust", "rust_parallel"):
        s = wall.get(arm)
        if s:
            check(f"prose quotes the {arm} arm wall clock {s}s", "quoted",
                  "found" if str(s) in body else "MISSING", ok=str(s) in body)
    check("parity: save files unchanged by the parallel arm", True,
          rep.get("save_files", {}).get("unchanged"),
          ok=bool(rep.get("save_files", {}).get("unchanged")))


def gif_check(readme):
    """The three inline GIFs are the final agent's clips; each must exist, fit GitHub's inline limit
    and be captioned with the decisions and coins the recording script logged for it."""
    meta_path = os.path.join(ROOT, "results", "btr_goal20_gifs.json")
    meta = load(meta_path)["episodes"] if os.path.exists(meta_path) else {}
    check("results/btr_goal20_gifs.json", "present", "present" if meta else "MISSING", ok=bool(meta))
    for t in ("ds1", "ds2", "ds3"):
        m = re.search(r"!\[([^\]]*)\]\((videos/btr/btr_%s\.gif)\)" % t, readme)
        if not m:
            check(f"README inline gif for {t}", "videos/btr/btr_%s.gif" % t, "none")
            continue
        path = os.path.join(ROOT, m.group(2))
        ok = os.path.exists(path)
        mb = os.path.getsize(path) / 1e6 if ok else 0.0
        check(f"{t}: inline gif exists on disk", "present", m.group(2), ok=ok)
        check(f"{t}: gif under GitHub's 10 MB inline limit", "<10 MB", f"{mb:.1f} MB", ok=ok and mb < 10.0)
        e = meta.get(t)
        if e:
            want = f"{e['decisions']} decisions, {e['coins']} coins gained"
            check(f"{t}: gif caption equals the recorded episode", want, "quoted" if want in m.group(1) else "absent",
                  ok=want in m.group(1) and e["end"] == "finish")
    old = "videos/dashboard/m64_dashboard_preview.gif"
    check("old PPO preview gif no longer referenced", "absent", "present" if old in readme else "absent",
          ok=old not in readme)


def _eval_stats(name):
    import statistics
    rows = load(os.path.join(ROOT, "results", name))["rows"]
    out = {}
    for t in ("ds1", "ds2", "ds3"):
        rs = [r for r in rows if r["track"] == t]
        fin = [r for r in rs if r["end"] == "finish"]
        out[t] = {"n": len(rs), "fin": len(fin),
                  "dec": statistics.median(r["decisions"] for r in fin) if fin else None,
                  "coins": statistics.median(r["coins_gained"] for r in fin) if fin else None}
    return out


def _wilson(k, n, z=1.96):
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    h = z * (k / n * (1 - k / n) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return c - h, c + h


def rows_names():
    return ("btr_goal20_eval_1M_x20.json", "btr_goal20_eval_1M_perturbed15_x20.json", "btr_goal20_eval_750k_x20.json",
            "btr_goal20_eval_500k_x20.json", "btr_ref10_eval_2500005_x20.json", "btr_ref10_eval_4500000_x20.json")


def btr_checks(readme):
    """The final-agent block must equal the evaluation files, the training logs and the probe."""
    flat = " ".join(readme.split())
    final = "btr_goal20_eval_1M_x20.json"
    if not os.path.exists(os.path.join(ROOT, "results", final)):
        check("final agent evaluation json", "present", "missing")
        return
    check("final eval was run on the checkpoint the README names", "models/btr_goal20/step_1000005.pt",
          load(os.path.join(ROOT, "results", final))["model"],
          ok=load(os.path.join(ROOT, "results", final))["model"] == "models/btr_goal20/step_1000005.pt")
    st = _eval_stats(final)
    for t, label in (("ds1", "| ds1 (Cool"), ("ds2", "| ds2 (The Princess"), ("ds3", "| ds3 (Tall")):
        row = next((ln for ln in readme.splitlines() if ln.startswith(label)), "")
        r = st[t]
        want = f"| {r['fin']} of {r['n']} | {r['dec']:g} | {r['coins']:g} |"
        check(f"final agent {t}: finishes / median decisions / median coins", want, "quoted" if want in row else "absent",
              ok=want in row)
    ps = _eval_stats("btr_goal20_eval_1M_perturbed15_x20.json")
    pd = load(os.path.join(ROOT, "results", "btr_goal20_eval_1M_perturbed15_x20.json"))
    check("perturbed eval used --start-random 15 on the same checkpoint", "15; same model",
          f"{pd['start_random']}; {pd['model']}", ok=pd["start_random"] == 15 and pd["model"] == load(
              os.path.join(ROOT, "results", final))["model"])
    ks = [r["random_actions"] for r in pd["rows"]]
    check("perturbed eval: random actions within 0..15, mean quoted", f"{sum(ks) / len(ks):.2f}",
          f"{max(ks)}", ok=0 <= min(ks) and max(ks) <= 15 and f"({sum(ks) / len(ks):.2f} on average)" in flat)
    for t, label in (("ds1", "| ds1 perturbed"), ("ds2", "| ds2 perturbed"), ("ds3", "| ds3 perturbed")):
        row = next((ln for ln in readme.splitlines() if ln.startswith(label)), "")
        r = ps[t]
        want = f"| {r['fin']} of {r['n']} | {r['dec']:g} | {r['coins']:g} |"
        check(f"perturbed {t}: finishes / median decisions / median coins", want,
              "quoted" if want in row else "absent", ok=want in row)
    check("every evaluation file has 20 episodes in every cell", "20", "ok", ok=all(
        _eval_stats(n)[t]["n"] == 20 for n in list(rows_names()) for t in ("ds1", "ds2", "ds3")))
    lo, hi = _wilson(st["ds3"]["fin"], st["ds3"]["n"])
    want = f"from roughly {lo * 100:.0f}% to {hi * 100:.0f}%"
    check("ds3 Wilson interval quoted", want, "quoted" if want in flat else "absent", ok=want in flat)
    rows = {"| coin-only, 2.5M steps": "btr_ref10_eval_2500005_x20.json",
            "| coin-only, 4.5M steps": "btr_ref10_eval_4500000_x20.json",
            "| coin + finish reward, 500k steps": "btr_goal20_eval_500k_x20.json",
            "| coin + finish reward, 750k steps": "btr_goal20_eval_750k_x20.json",
            "| coin + finish reward, 1M steps": final}
    for label, name in rows.items():
        line = next((ln for ln in readme.splitlines() if ln.startswith(label)), "")
        s = _eval_stats(name)
        want = " | ".join(f"{s[t]['fin']} of {s[t]['n']}" for t in ("ds1", "ds2", "ds3"))
        check(f"checkpoint comparison: {label[2:]}", want, "quoted" if want in line else "absent", ok=want in line)
    import csv
    mins = {}
    for run, steps in (("btr_ref10", 2_500_000), ("btr_goal20", 1_000_000)):
        with open(os.path.join(ROOT, "results", f"{run}_train.csv"), newline="", encoding="utf-8") as fh:
            rr = list(csv.DictReader(fh))
        mins[run] = float(next(r for r in rr if int(r["steps"]) >= steps)["minutes"])
    want = [f"{mins['btr_ref10']:.1f} min", f"{mins['btr_goal20']:.1f} min",
            f"about {(mins['btr_ref10'] + mins['btr_goal20']) / 60:.1f} hours"]
    check("training minutes and total hours", "; ".join(want), "quoted" if all(w in flat for w in want) else "absent",
          ok=all(w in flat for w in want))
    pr = load(os.path.join(ROOT, "results", "ds3_after_finish_probe.json"))
    tr = pr["trace"]
    want = [f"{tr[0]['dist']:,.1f}", f"{tr[10]['dist']:,.1f}", f"stayed at {tr[0]['coins']}", f"{len(tr) - 1} more decisions"]
    ok = (all(w in flat for w in want) and len({r["coins"] for r in tr}) == 1 and tr[10]["after_finish"] == 10
          and tr[-1]["after_finish"] == 150)
    check("after-finish probe numbers", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ablation_checks(readme):
    """The ablation table must equal the evaluation files and the last logged training step."""
    import csv
    import statistics
    for run, label, steps_col in (("control", "| control |", None), ("ppo", "| ppo |", None), ("legacy6", "| legacy6 |", None), ("orig", "| orig |", None)):
        ev = os.path.join(ROOT, "results", f"abl_{run}_eval_x20.json")
        if not os.path.exists(ev):
            check(f"ablation {run} evaluation json", "present", "missing")
            continue
        rows = load(ev)["rows"]
        line = next((ln for ln in readme.splitlines() if ln.startswith(label)), "")
        with open(os.path.join(ROOT, "results", f"abl_{run}_train.csv"), newline="", encoding="utf-8") as fh:
            last = list(csv.DictReader(fh))[-1]
        cells = []
        for t in ("ds1", "ds2", "ds3"):
            rs = [r for r in rows if r["track"] == t]
            cells.append(f"{sum(r['end'] == 'finish' for r in rs)} of {len(rs)}")
        dist = " / ".join(str(round(statistics.median(r["min_dist"] for r in rows if r["track"] == t))) for t in ("ds1", "ds3"))
        want = f"| {int(last['steps']):,} | " + " | ".join(cells) + f" | {dist} |"
        check(f"ablation {run}: steps / finishes / median closest distance", want,
              "quoted" if want in line else "absent", ok=want in line)

    import csv as _csv
    def _row(run, steps):
        with open(os.path.join(ROOT, "results", f"{run}_train.csv"), newline="", encoding="utf-8") as fh:
            return next(r for r in _csv.DictReader(fh) if int(r["steps"]) == steps)
    o, k, q = _row("btr_ref10", 640005), _row("abl_control", 420000), _row("abl_orig", 580005)
    flat = " ".join(readme.split())
    want = [f"{float(o['ds1_coins']):.1f}, {float(o['ds2_coins']):.1f} and {float(o['ds3_coins']):.1f} coins",
            f"after {float(o['minutes']):.1f} minutes (640,005 steps)",
            f"{float(k['ds1_coins']):.1f}, {float(k['ds2_coins']):.1f} and {float(k['ds3_coins']):.1f} after {float(k['minutes']):.1f} minutes (420,000 steps)",
            f"{float(q['ds1_coins']):.1f}, {float(q['ds2_coins']):.1f} and {float(q['ds3_coins']):.1f} after {float(q['minutes']):.1f} minutes (580,005 steps)"]
    check("ablation control vs the original run at one hour", "; ".join(want), "quoted" if all(w in flat for w in want) else "absent",
          ok=all(w in flat for w in want))


def rust_checks(readme):
    main_rs = os.path.join(ROOT, "rust", "src", "main.rs")
    if not os.path.exists(main_rs):
        check("rust/src/main.rs", "present", "missing")
        return
    src = open(main_rs, encoding="utf-8").read()
    uses = re.findall(r"^use\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.MULTILINE)
    non_std = [u for u in uses if u not in ("std",)]
    check("supervisor is std-only (no external crate imports)", "std",
          ",".join(sorted(set(uses))) or "none", ok=not non_std)
    build = open(os.path.join(ROOT, "rust", "build.sh"), encoding="utf-8").read()
    check("build.sh compiles with rustc, not cargo", True, True,
          ok="rustc" in build and "cargo build" not in build)
    check("supervisor binary present", "rust/sm64_sup.exe",
          "present" if os.path.exists(os.path.join(ROOT, "rust", "sm64_sup.exe"))
          else "MISSING", ok=os.path.exists(os.path.join(ROOT, "rust", "sm64_sup.exe")))

    if os.path.exists(PARITY):
        rep = load(PARITY)
        ident = sum(1 for r in rep["rows"] if r["verdict"] == "identical")
        n = len(rep["rows"])
        check("parity: every track identical across both orchestration arms",
              f"{n}/{n}", f"{ident}/{n}", ok=ident == n)
        check("parity: budget-kill path", True, rep["kill_test"]["pass"],
              ok=bool(rep["kill_test"]["pass"]))
        cell = next((c for c in re.findall(rf"{ident}/{n}", readme + open(
            RUST_README, encoding="utf-8").read()) if c), None)
        check(f"prose quotes the parity count {ident}/{n}", "quoted",
              "found" if cell else "MISSING", ok=cell is not None)
        bsec = rep["kill_test"]["budget_s"]
        check(f"prose quotes the kill budget {bsec:g}s", "quoted",
              "found" if re.search(rf"{bsec:g}\s*s", open(RUST_README,
               encoding="utf-8").read()) else "MISSING",
              ok=bool(re.search(rf"{bsec:g}\s*s", open(RUST_README, encoding="utf-8").read())))
    else:
        check("out/supervisor_parity.json", "present", "MISSING")

    if os.path.exists(PROFILE):
        prof = load(PROFILE)["ms_per_call"]
        body = open(RUST_README, encoding="utf-8").read()
        for key, pat in (("emulator_cycles", r"emulator cycles[^|]*?\|\s*([\d.]+) ms"),
                         ("full_step", r"whole env\.step[^|]*?\|\s*([\d.]+) ms")):
            m = re.search(pat, body, re.IGNORECASE)
            if m:
                check(f"rust/README {key} cell", f"{prof[key]:.2f} ms", m.group(1) + " ms",
                      ok=abs(float(m.group(1)) - prof[key]) < 0.01)
        m = re.search(r"([\d.]+)\s*% of a step", body)
        if m:
            check("rust/README cycles-share cell",
                  f"{prof['cycles_share_of_step']:.2f} %", m.group(1) + " %",
                  ok=abs(float(m.group(1)) - prof["cycles_share_of_step"]) < 1.0)
    else:
        check("out/step_profile.json", "present", "MISSING (run tools/profile_step.py)")


def asset_checks(readme):
    """The flagship checkpoints are release assets, not Git blobs; their manifest
    is versioned and must stay valid even on a clean clone."""
    manifest = os.path.join(ROOT, "models", "ASSETS.json")
    if not os.path.exists(manifest):
        check("models/ASSETS.json", "present", "MISSING")
        return
    data = load(manifest)
    check("asset manifest schema", 1, data.get("schema"),
          ok=int(data.get("schema", 0)) == 1)
    assets = data.get("assets", [])
    check("asset manifest has entries", ">=4", len(assets), ok=len(assets) >= 4)
    for a in assets:
        check(f"{a.get('path')} has sha256", "present", bool(a.get("sha256")),
              ok=bool(a.get("sha256")))
    check("README links docs/MODEL_ZOO.md", "linked",
          "docs/MODEL_ZOO.md" in readme, ok="docs/MODEL_ZOO.md" in readme)


def _section(readme, start, end):
    i = readme.find(start)
    j = readme.find(end, i + 1) if i >= 0 else -1
    return readme[i:j] if i >= 0 and j > i else ""


def _num(x, nd=1):
    """README style: thousands separators, fixed decimals (12345.6 -> 12,345.6)."""
    return f"{x:,.{nd}f}"


def _finish_rows(name):
    path = os.path.join(ROOT, "results", f"finish_line_{name}.json")
    if not os.path.exists(path):
        check(f"finish_line_{name}.json", "present", "missing")
        return None
    return {(r["env"], r["track"]): r for r in load(path)["rows"]}


def _finish_cell(row):
    if row["end"] == "finish":
        return f"finish at decision {row['decisions']}"
    if row["end"] == "horizon":
        return f"horizon, closest {_num(row['min_dist'])}"
    if row["void_falls"]:
        return f"falls at {row['decisions']}, closest {_num(row['min_dist'])}"
    return f"ended at {row['decisions']} by the black-screen rule, closest {_num(row['min_dist'])}"


def finish_line_checks(readme):
    """The finish-line callout must equal results/finish_line_curriculum_flow25_r3_best.json."""
    block = _section(readme, "Measured against the RAM finish line", "To re-record")
    check("finish-line callout present", "present", "present" if block else "missing", ok=bool(block))
    rows = _finish_rows("curriculum_flow25_r3_best")
    if not block or rows is None:
        return
    for track in ("ds1", "ds2", "ds3"):
        r = rows[("ram", track)]
        line = next((ln for ln in block.splitlines() if ln.startswith(f"> | `{track}`")), "")
        want = (f"**finish** at decision {r['decisions']}" if r["end"] == "finish"
                else f"closest {_num(r['min_dist'])} of {_num(r['start_dist'])}")
        check(f"{track}: callout outcome", want, "quoted" if want in line else "absent", ok=want in line)
        falls = f"| {r['void_falls']} |"
        check(f"{track}: callout falls below kill_y", r["void_falls"],
              "quoted" if line.rstrip().endswith(falls) else "absent", ok=line.rstrip().endswith(falls))
        hist = rows[("historical", track)]
        check(f"{track}: historical env agrees", r["end"], hist["end"], ok=hist["end"] == r["end"])


def finetune_checks(readme):
    """The controller table must equal results/finish_line_*.json (position rule)."""
    block = _section(readme, "What the corrected finish line changed", "### 📦 Model assets")
    check("corrected finish-line table present", "present", "present" if block else "missing",
          ok=bool(block))
    if not block:
        return
    for label, name in (("PPO `curriculum_flow25_r3_best`", "curriculum_flow25_r3_best"),
                        ("world-model head `wm_mario64ds_mem`", "wm_mario64ds_mem"),
                        ("PPO `curriculum_finish_r4_best`", "curriculum_finish_r4_best")):
        rows = _finish_rows(name)
        if rows is None:
            continue
        line = next((ln for ln in block.splitlines() if ln.startswith(f"| {label}")), "")
        for track in ("ds1", "ds2", "ds3"):
            cell = _finish_cell(rows[("ram", track)])
            check(f"{name} {track} cell", cell, "quoted" if cell in line else "absent", ok=cell in line)


def ds3_explore_checks(readme):
    """The ds3 exploration-bonus sentence must equal results/ds3_explore_r5_checkpoints.json."""
    path = os.path.join(ROOT, "results", "ds3_explore_r5_checkpoints.json")
    if not os.path.exists(path):
        check("ds3_explore_r5_checkpoints.json", "present", "missing")
        return
    rows = load(path)["rows"]
    best = min(rows, key=lambda r: r["min_dist"])
    final = next(r for r in rows if r["checkpoint"] == "final")
    want = [_num(best["min_dist"]), f"decision {best['decisions']}", f"closest {_num(final['min_dist'])}"]
    ok = (not any(r["end"] == "finish" for r in rows) and "No checkpoint\nfinished" in readme
          and all(w in readme for w in want))
    check("ds3 exploration-bonus result", ", ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_search_checks(readme):
    """The coin-search sentence must equal results/ds3_searches.json."""
    path = os.path.join(ROOT, "results", "ds3_searches.json")
    if not os.path.exists(path):
        check("ds3_searches.json", "present", "missing")
        return
    d = load(path)
    coin = [r for r in d["searches"] if r["score"] == "coins"]
    most = max(r["coins_at_end"] for r in coin)
    died = all(r["last_stage_deaths"] == r["rollouts_per_stage"] for r in coin)
    want = [f"from {d['coins_at_prefix']} coins to", f"at most {most}", "96 of 96"]
    ok = died and not any(r["finish"] for r in d["searches"]) and all(w in readme for w in want)
    check("ds3 coin-search result", ", ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_sweep_checks(readme):
    """The move-sweep sentence must equal results/ds3_sweeps.json."""
    path = os.path.join(ROOT, "results", "ds3_sweeps.json")
    if not os.path.exists(path):
        check("ds3_sweeps.json", "present", "missing")
        return
    d = load(path)
    first, rest = d["sweeps"][0], d["sweeps"][1:]
    lj = d["long_jump_test"]
    want = [f"{first['survivors']} of {_num(first['attempts'], 0)} openings survive",
            _num(min(r["min_dist"] for r in d["sweeps"])),
            "(" + ", ".join(f"0 of {_num(r['attempts'], 0)}" for r in rest[:-1]) + f" and 0 of {_num(rest[-1]['attempts'], 0)})",
            f"any of {lj['attempts']} long"]
    ok = (all(r["survivors"] == 0 for r in rest) and lj["survivors"] == 0
          and all(w in " ".join(readme.split()) for w in want))
    check("ds3 move-sweep result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_gap_training_checks(readme):
    """The gap-training sentence must equal results/ds3_gap_training.json."""
    path = os.path.join(ROOT, "results", "ds3_gap_training.json")
    if not os.path.exists(path):
        check("ds3_gap_training.json", "present", "missing")
        return
    d = load(path)
    lo, hi = d["training_ep_len_mean_range"]
    rlo, rhi = d["training_ep_rew_mean_range"]
    want = [f"about {_num(d['episodes_estimate'], 0)}", f"{lo}-{hi} decisions", f"${rlo}$ to ${rhi}$", "(0/8)"]
    flat = " ".join(readme.split())
    ok = all(w in flat for w in want)
    check("ds3 gap-training result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_coins_checks(readme):
    """The coin-only training sentence must equal results/ds3_coins_r7.json."""
    path = os.path.join(ROOT, "results", "ds3_coins_r7.json")
    if not os.path.exists(path):
        check("ds3_coins_r7.json", "present", "missing")
        return
    d = load(path)
    fin = next(r for r in d["greedy"] if r["checkpoint"] == "final_model")
    best = next(r for r in d["greedy"] if r["checkpoint"].startswith("best_model"))
    st = d["stochastic_final_model"]
    want = [f"$+{fin['coins_gained']}$ coins", _num(fin["min_dist_to_goal"]), _num(best["min_dist_to_goal"]),
            f"none of {st['episodes']}", _num(st["closest_min_dist"])]
    flat = " ".join(readme.split())
    ok = (st["reached_goal"] == 0 and not any(r["end"] == "finish" for r in d["greedy"])
          and all(w in flat for w in want))
    check("ds3 coin-only training result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_route_run_checks(readme):
    """The route-start PPO sentence must equal results/ds3_route_r8.json."""
    path = os.path.join(ROOT, "results", "ds3_route_r8.json")
    if not os.path.exists(path):
        check("ds3_route_r8.json", "present", "missing")
        return
    d = load(path)
    models = d["checkpoints"] + [d["final_model"]]
    evals = [a for m in models for a in m["arrow_states"]]
    passes = [a for a in evals if a["end"] == "finish"]
    best = min(m["from_start"]["min_dist"] for m in models if m["from_start"])
    want = [f"{len(passes)} of {len(evals)} arrow-state", f"({_num(passes[0]['min_dist'], 0)} from its centre",
            f"{_num(best)} from the goal", f"Twelve checkpoints"]
    flat = " ".join(readme.split())
    ok = len(d["checkpoints"]) == 12 and all(w in flat for w in want)
    check("ds3 route-start PPO result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_qrdqn_checks(readme):
    """The QR-DQN sentence must equal results/qrdqn_ds3_route.json."""
    path = os.path.join(ROOT, "results", "qrdqn_ds3_route.json")
    if not os.path.exists(path):
        check("qrdqn_ds3_route.json", "present", "missing")
        return
    d = load(path)
    rows = d["checkpoints"]
    evals = [a for m in rows for a in m["arrow_states"]]
    passes = sum(a["end"] == "finish" for a in evals)
    best_arrow = min(a["min_dist"] for a in evals)
    best_start = min(m["from_start"]["min_dist"] for m in rows)
    want = [f"({passes} of {len(evals)})", f"{_num(best_arrow)} from the goal", f"{_num(best_start)} (800k)"]
    flat = " ".join(readme.split())
    ok = passes == 0 and all(w in flat for w in want)
    check("ds3 QR-DQN result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_clone_checks(readme):
    """The behaviour-cloning sentence must equal results/ds3_bc_demo.json."""
    path = os.path.join(ROOT, "results", "ds3_bc_demo.json")
    if not os.path.exists(path):
        check("ds3_bc_demo.json", "present", "missing")
        return
    d = load(path)
    fs = d["from_start"]
    want = [f"({d['samples']:,} samples", f"(train accuracy {d['train_accuracy']})", f"{fs['decisions']} decisions",
            f"collects {fs['coins_gained']}", f"{_num(fs['min_dist'])} from the gap goal",
            f"({d['arrow_reached_goal']} of 4 reach it; {d['route_states_reached_goal']} of {d['route_states']}",
            f"{d['perturbed_reached_goal']} of {d['perturbed_runs']} runs"]
    flat = " ".join(readme.split())
    ok = d["arrow_reached_goal"] == 0 and all(w in flat for w in want)
    check("ds3 behaviour-cloning result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_detector_checks(readme):
    """The coin-detector paragraph must equal results/ds3_coin_detector.json."""
    path = os.path.join(ROOT, "results", "ds3_coin_detector.json")
    if not os.path.exists(path):
        check("ds3_coin_detector.json", "present", "missing")
        return
    d = load(path)
    v = d["variants"]
    want = [f"{d['frames']:,} frames, {d['collection_events']} collection events",
            f"({d['collection_events']} of {d['collection_events']})",
            f"{v['hud_masked']['positive_frames'] * 100:.1f}% of frames",
            f"{v['strict_round']['positive_frames'] * 100:.1f}% of frames",
            f"median lead {v['strict_round']['lead_frames_median']:.0f} frames"]
    flat = " ".join(readme.split())
    ok = all(w in flat for w in want) and all(x["recall"] == 1.0 for x in v.values())
    check("ds3 coin-detector result", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ds3_template_checks(readme):
    """The template-matching and fine-tune paragraphs must equal their result files."""
    flat = " ".join(readme.split())
    a = os.path.join(ROOT, "results", "ds3_coin_template.json")
    b = os.path.join(ROOT, "results", "ds3_coin_template_game.json")
    f = os.path.join(ROOT, "results", "ds3_finetune.json")
    if not all(os.path.exists(x) for x in (a, b, f)):
        check("ds3 template/fine-tune json", "present", "missing")
        return
    ta, tb, ft = load(a)["thresholds"], load(b)["thresholds"], load(f)["checkpoints"]
    want = [f"{ta['0.60']['positive_frames'] * 100:.1f}% of frames",
            f"at 0.7 in {ta['0.70']['positive_frames'] * 100:.1f}%",
            f"median lead of {ta['0.70']['lead_frames_median']:.0f} frames",
            f"finds {round(ta['0.80']['recall'] * 28)} of 28",
            f"{tb['0.60']['positive_frames'] * 100:.1f}%, {tb['0.70']['positive_frames'] * 100:.1f}% and "
            f"{tb['0.80']['positive_frames'] * 100:.1f}% of frames",
            f"median lead is {tb['0.80']['lead_frames_median']:.0f} frames"]
    last = ft[-1]
    want += [f"{last['steps_since_start'] // 1000}k steps it", f"dies after {last['from_start']['decisions']} decisions",
             f"with {last['from_start']['coins_gained']} coins", f"{last['hard_spot_reached']} of 4 hard-spot"]
    dv = load(os.path.join(ROOT, "results", "dive_b.json"))
    dr = load(os.path.join(ROOT, "results", "dive_refine_285.json"))
    want += [f"(0 of {dv['n']:,})".replace("0 of", f"{dv['finished']} of"), f"it is {dv['top'][0]['best']:,.1f}",
             f"stops at {dr['best']:,.1f}"]
    ok = all(w in flat for w in want) and all(r["arrow_died"] == 4 for r in ft) and         all(x["recall"] == 1.0 for x in tb.values()) and dv["finished"] == 0 and not dr["finish"]
    check("ds3 template and fine-tune results", "; ".join(want), "quoted" if ok else "absent", ok=ok)


def ttc_checks(readme):
    """The TTC status table must equal results/route_starts_*.json and the v4 dashboard."""
    block = _section(readme, "### Tick Tock Clock: human route", "### Savestate")
    check("TTC status section present", "present", "present" if block else "missing", ok=bool(block))
    if not block:
        return
    for label, name in (("v4", "qrdqn_impala_ds8_v4_best"), ("do nothing", "noop")):
        path = os.path.join(ROOT, "results", f"route_starts_{name}.json")
        if not os.path.exists(path):
            check(f"TTC {label}: route_starts json", "present", "missing")
            continue
        r = load(path)
        line = next((ln for ln in block.splitlines() if ln.startswith(f"| {label}")), "")
        cell = f"{r['median_route_gain']:.1f} units / {r['died']} of {r['starts']}"
        check(f"TTC {label}: median gain / falls", cell, "quoted" if cell in line else "absent",
              ok=cell in line)
        if name != "noop":
            start = r["rows"][0]
            want = f"falls at decision {start['decisions']}, $+{start['height_gain']}$ height"
            check(f"TTC {label}: full-start cell", want, "quoted" if want in line else "absent",
                  ok=start["end"] == "died" and want in line)
            dash = os.path.join(ROOT, "videos", "dashboard_ds8_v4", "dashboard_ds8.json")
            if os.path.exists(dash):
                d = load(dash)
                check("TTC v4 dashboard agrees with the evaluator", start["decisions"], d["steps"],
                      ok=d["steps"] == start["decisions"] and d["died"])


def main():
    readme = open(README, encoding="utf-8").read()
    payloads = {}
    for t in ("ds1", "ds2", "ds3"):
        p = os.path.join(DASH_DIR, f"dashboard_{t}.json")
        if os.path.exists(p):
            payloads[t] = load(p)
    check("dashboard run payloads", "3 tracks", f"{len(payloads)} found",
          ok=len(payloads) == 3)
    if payloads:
        showcase(readme, payloads)
    gif_check(readme)
    rust_checks(readme)
    repro_checks(readme)
    parallel_checks(readme)
    asset_checks(readme)
    finish_line_checks(readme)
    ttc_checks(readme)
    finetune_checks(readme)
    ds3_explore_checks(readme)
    ds3_search_checks(readme)
    ds3_sweep_checks(readme)
    ds3_gap_training_checks(readme)
    ds3_coins_checks(readme)
    ds3_route_run_checks(readme)
    ds3_qrdqn_checks(readme)
    ds3_clone_checks(readme)
    ds3_detector_checks(readme)
    ds3_template_checks(readme)
    btr_checks(readme)
    ablation_checks(readme)

    bad = [c for c in CHECKS if not c[3]]
    print(f"{'status':7} {'check':62} {'expected':22} {'found':22}")
    for label, exp, got, ok in CHECKS:
        print(f"{'PASS' if ok else 'FAIL':7} {label:62} {str(exp):22} {str(got):22}")
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
