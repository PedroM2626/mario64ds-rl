"""Random search for the hand -> next platform transfer of Tick Tock Clock (evaluation only, no learning).

Usage: python scripts/search_ttc_jump.py <savestate.dst> <ref10|compound12> <trials> <seed> <out.json>
Each trial: 28 decisions of random actions, each held 1-8 decisions, from the savestate. Success (as reported in
the README) = best route arc length >= 51000 and height > 5800 after the 28 decisions."""
import sys, os, json, random
ROOT = r"D:\mario64ds-rl"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from scripts.train_btr import ref_env_kwargs
from src.env import Mario64DSEnv
from src.envs import rom_path

state, aset, trials, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
kw = ref_env_kwargs("ds8", action_set=aset)
env = Mario64DSEnv(rom_path=rom_path(), state_path=state, max_steps=200, track_id="ds8", **kw)
n = env.action_space.n
rng = random.Random(seed)
res = []
for k in range(trials):
    env.reset()
    best, a, left, alive, y_end = 0.0, 0, 0, True, 0.0
    seq = []
    for t in range(28):
        if left == 0:
            a, left = rng.randrange(n), rng.randint(1, 8)
        left -= 1
        seq.append(a)
        _, _, term, _, info = env.step(a)
        s = float(info.get("route_progress") or 0.0)
        best = max(best, s)
        y_end = float(env.ram.get_pos_3d()[1])
        if term:
            alive = False
            break
    res.append({"best_s": round(best, 1), "y_end": round(y_end, 1), "alive": alive, "seq": seq})
json.dump(res, open(out, "w"))
ok = sum(1 for r in res if r["best_s"] >= 51000 and r["y_end"] > 5800)
print(aset, os.path.basename(state), "trials", len(res), "success(best_s>=51000 and y>5800)", ok, "max best_s", max(r["best_s"] for r in res))
