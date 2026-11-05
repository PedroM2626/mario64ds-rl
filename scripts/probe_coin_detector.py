"""Does an image-based "coin centred on screen" signal work on ds3?

Replays a route (config/routes/ds3_search_route_v1.json) from the ds3 start, capturing every
emulated frame and the RAM coin counter (src/ram_state.COIN_COUNT_ADDR). Several
detectors (the original HSV yellow mask of src/env.py, with and without the HUD masked,
and stricter variants) are then compared with the counter:

  recall     collection events (counter increases) that had a blob in the preceding
             `--lookback` frames, and one within the central third of the screen
  false pos  fraction of positive frames with no collection within `--horizon` frames
             after them (includes real coins that were passed or collected later, so it
             is an upper bound on false positives; the overlay image shows what fires)
  hud        fraction of positive frames whose largest blob lies in the HUD strip

The outputs are an overlay image (true: the frames just before collections; suspect:
positive frames far from any collection; arrow: decisions 330-346, where the planks' coins
first appear) and a JSON with the numbers.

Usage:
    python scripts/probe_coin_detector.py --route config/routes/ds3_search_route_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HUD_ROWS = 24  # the lives and coin counters sit in the top strip of the screen

VARIANTS = {
    "original": dict(lo=(15, 100, 100), hi=(40, 255, 255), min_area=6, hud=False, round=False),
    "hud_masked": dict(lo=(15, 100, 100), hi=(40, 255, 255), min_area=6, hud=True, round=False),
    "strict": dict(lo=(20, 150, 150), hi=(32, 255, 255), min_area=12, hud=True, round=False),
    "strict_round": dict(lo=(20, 150, 150), hi=(32, 255, 255), min_area=12, hud=True, round=True),
    "hud_round": dict(lo=(15, 100, 100), hi=(40, 255, 255), min_area=6, hud=True, round=True),
}


def detect(rgb, cfg):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(cfg["lo"]), np.array(cfg["hi"]))
    if cfg["hud"]:
        mask[:HUD_ROWS] = 0
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < cfg["min_area"]:
            continue
        if cfg["round"] and not (0.5 <= w / max(h, 1) <= 2.0 and area / max(w * h, 1) >= 0.55):
            continue
        blobs.append({"area": int(area), "cx": float(cent[i][0]), "cy": float(cent[i][1]), "box": (int(x), int(y), int(w), int(h))})
    return blobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", default="config/routes/ds3_search_route_v1.json")
    ap.add_argument("--upto", type=int, default=717)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--out-image", default="results/ds3_coin_detector.png")
    ap.add_argument("--out-json", default="results/ds3_coin_detector.json")
    args = ap.parse_args()

    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    spec = json.load(open(os.path.join(ROOT_DIR, args.route), encoding="utf-8"))
    actions = spec["actions"][:args.upto]
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(spec["track"]), max_steps=100000,
                       flow_weight=0.0, dist_reward_enabled=False, track_id=spec["track"],
                       action_set=spec["action_set"], use_ram_death=False, black_screen_death=False)
    frames, coins = [], []
    env.reset()
    env.on_cycle = lambda: (frames.append(env.get_screen_rgb().copy()), coins.append(env.ram.get_coins()))
    for a in actions:
        env.step(int(a))
    env.close()
    coins = np.asarray(coins)
    events = np.flatnonzero(np.diff(coins) > 0) + 1
    print(f"{len(frames)} frames, coins {coins[0]} -> {coins[-1]}, {len(events)} collection events", flush=True)

    results, per_frame = {}, {}
    for name, cfg in VARIANTS.items():
        det = [detect(f, cfg) for f in frames]
        per_frame[name] = det
        pos = np.array([len(d) > 0 for d in det])
        central = np.array([any(abs(b["cx"] - 128) < 43 for b in d) for d in det])
        rec = rec_c = 0
        for e in events:
            lo = max(0, e - args.lookback)
            rec += bool(pos[lo:e].any())
            rec_c += bool(central[lo:e].any())
        near = np.zeros(len(frames), bool)
        for e in events:
            near[max(0, e - args.horizon):e + 1] = True
        fp = float(np.mean(~near[pos])) if pos.any() else 0.0
        in_hud = float(np.mean([min(b["cy"] for b in d) < HUD_ROWS for d in det if d])) if pos.any() else 0.0
        # lead time: consecutive positive frames (gaps of up to 3 allowed) ending at each collection
        leads = []
        for e in events:
            n, gap = 0, 0
            for f in range(e - 1, max(-1, e - 301), -1):
                if central[f]:
                    n += 1 + gap
                    gap = 0
                else:
                    gap += 1
                    if gap > 3:
                        break
            leads.append(n)
        # the red-arrow stretch (decisions 330-350): how often is a coin seen, and where
        arrow = [d for d in det[330 * 4:350 * 4]]
        arrow_pos = float(np.mean([len(d) > 0 for d in arrow]))
        arrow_right = float(np.mean([any(b["cx"] > 140 for b in d) for d in arrow]))
        results[name] = {"lead_frames_median": float(np.median(leads)), "lead_frames_p25": float(np.percentile(leads, 25)),
                         "lead_frames_p75": float(np.percentile(leads, 75)),
                         "arrow_stretch_positive": arrow_pos, "arrow_stretch_blob_right_of_centre": arrow_right,
                         "positive_frames": float(np.mean(pos)), "recall": rec / len(events),
                         "recall_central": rec_c / len(events), "false_positive_upper_bound": fp,
                         "hud_blob_share": in_hud}
        print(f"{name:13s} positive {np.mean(pos):5.1%} | recall {rec}/{len(events)} (central {rec_c}) | "
              f"positives with no collection within {args.horizon} frames {fp:5.1%} | hud {in_hud:5.1%} | "
              f"lead frames median {np.median(leads):.0f} (p25 {np.percentile(leads, 25):.0f}, p75 {np.percentile(leads, 75):.0f}) | "
              f"arrow stretch: coin seen {arrow_pos:5.1%}, right of centre {arrow_right:5.1%}", flush=True)

    # overlay: what the strict_round detector sees at chosen frames
    cfg = VARIANTS["strict_round"]
    pos = np.array([len(d) > 0 for d in per_frame["strict_round"]])
    near = np.zeros(len(frames), bool)
    for e in events:
        near[max(0, e - args.horizon):e + 1] = True
    rng = np.random.default_rng(0)
    true_idx = [int(e) - 8 for e in events[rng.choice(len(events), min(6, len(events)), replace=False)]]
    sus = np.flatnonzero(pos & ~near)
    sus_idx = [int(i) for i in rng.choice(sus, min(6, len(sus)), replace=False)] if len(sus) else []
    arrow_idx = [330 * 4, 338 * 4, 346 * 4]

    def tile(i, label):
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR).copy()
        for b in per_frame["strict_round"][i]:
            x, y, w, h = b["box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        return cv2.putText(img, f"{label} f={i} c={coins[i]}", (3, 34), 0, .38, (0, 255, 255), 1)

    tiles = [tile(i, "true") for i in true_idx] + [tile(i, "suspect") for i in sus_idx] + [tile(i, "arrow") for i in arrow_idx]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(ROOT_DIR, args.out_image), img)
    out = {"route": args.route, "frames": len(frames), "collection_events": int(len(events)),
           "lookback": args.lookback, "horizon": args.horizon, "variants": results}
    with open(os.path.join(ROOT_DIR, args.out_json), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {args.out_image}, {args.out_json}")


if __name__ == "__main__":
    main()
