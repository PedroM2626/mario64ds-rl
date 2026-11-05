"""Template matching of the coin (images/coins.png) on the ds3 route frames.

The colour detectors of scripts/probe_coin_detector.py were either positive in almost every
frame (sand and planks are close to yellow in hue) or only saw a coin when it was about to be
collected. This one matches the coin's *appearance*: the yellow coin of images/coins.png
(RGBA, with the star emblem) is cut out, scaled to a bank of sizes (a coin seen from far is
a few pixels, one at arm's length is ~50) and matched with its alpha mask
(cv2.matchTemplate, normalised correlation, on the screen below the HUD strip).

Same route replay and metrics as the colour probe: recall of the 28 collection events, the share
of positive frames, positives with no collection within 120 frames (an upper bound on false
positives), the lead time (consecutive positive central frames before a collection) and how often a
coin is seen in the red-arrow stretch (decisions 330-350). The correlation threshold is swept.

--mode game builds the templates from the game itself instead: the coins that the strict colour
detector sees in the last frames before a collection (certainly coins) are cropped, scaled to the bank
of sizes and used as templates. Only the even-numbered collection events provide templates; recall and
lead time are measured on the odd-numbered ones (held out), and false positives over all frames far
from every collection.

Usage:
    python scripts/probe_coin_template.py --cache <dir>/ds3_route_frames.npz [--mode image|game]
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

HUD_ROWS = 24
HEIGHTS = (7, 9, 12, 16, 21, 28, 37, 48)   # template heights in pixels (a 192-px-high screen)


def make_bank(path):
    im = cv2.imread(os.path.join(ROOT_DIR, path), cv2.IMREAD_UNCHANGED)
    bgr, alpha = im[:, :, :3], im[:, :, 3]
    # the image holds a blue coin (left) and a yellow one (right); take the yellow one
    half = alpha.copy()
    half[:, :im.shape[1] // 2 - 4] = 0
    ys, xs = np.nonzero(half > 128)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    bgr, alpha = bgr[crop], alpha[crop]
    bank = []
    for h in HEIGHTS:
        w = max(3, int(round(bgr.shape[1] * h / bgr.shape[0])))
        t = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = (cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        bank.append((t, np.repeat(m[:, :, None], 3, axis=2)))
    return bank


def game_bank(frames, events, train):
    """Templates cropped from the frames just before the training collections."""
    from scripts.probe_coin_detector import VARIANTS, detect as color_detect
    cfg = VARIANTS["strict_round"]
    crops = []
    for e in train:
        best = None
        for f in range(max(0, e - 8), e):
            for b in color_detect(frames[f], cfg):
                if best is None or b["area"] > best[0]["area"]:
                    best = (b, f)
        if best is None:
            continue
        b, f = best
        x, y, w, h = b["box"]
        pad = 2
        y0, y1, x0, x1 = max(0, y - pad), min(192, y + h + pad), max(0, x - pad), min(256, x + w + pad)
        crops.append(cv2.cvtColor(frames[f][y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    bank = []
    for c in crops:
        for h in HEIGHTS:
            if h > 3 * c.shape[0] or h < c.shape[0] / 3:
                continue
            w = max(3, int(round(c.shape[1] * h / c.shape[0])))
            t = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            yy, xx = np.ogrid[:h, :w]
            ell = (((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1.0).astype(np.float32)
            bank.append((t, np.repeat(ell[:, :, None], 3, axis=2)))
    return bank, crops


def detect(rgb, bank, min_score):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    blobs = []
    for t, m in bank:
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED, mask=m)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        h, w = t.shape[:2]
        res[: max(0, HUD_ROWS - h // 2), :] = 0.0       # centre of the template inside the HUD strip
        ys, xs = np.nonzero(res >= min_score)
        for y, x in zip(ys, xs):
            blobs.append({"score": float(res[y, x]), "cx": x + w / 2.0, "cy": y + h / 2.0, "area": h * w,
                          "box": (int(x), int(y), int(w), int(h))})
    # non-maximum suppression: keep the best score within overlapping boxes
    blobs.sort(key=lambda b: -b["score"])
    kept = []
    for b in blobs:
        if all(abs(b["cx"] - k["cx"]) > max(b["box"][2], k["box"][2]) / 2 or
               abs(b["cy"] - k["cy"]) > max(b["box"][3], k["box"][3]) / 2 for k in kept):
            kept.append(b)
    return kept


_BANK = None


def _init(bank, th):
    global _BANK
    _BANK = (bank, th)


def _detect_one(f):
    return detect(f, *_BANK)


def capture(route, upto, cache):
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return list(z["frames"]), z["coins"]
    from src.env import Mario64DSEnv
    from src.envs import rom_path, savestate_path
    spec = json.load(open(os.path.join(ROOT_DIR, route), encoding="utf-8"))
    env = Mario64DSEnv(rom_path=rom_path(), state_path=savestate_path(spec["track"]), max_steps=100000,
                       flow_weight=0.0, dist_reward_enabled=False, track_id=spec["track"],
                       action_set=spec["action_set"], use_ram_death=False, black_screen_death=False)
    frames, coins = [], []
    env.reset()
    env.on_cycle = lambda: (frames.append(env.get_screen_rgb().copy()), coins.append(env.ram.get_coins()))
    for a in spec["actions"][:upto]:
        env.step(int(a))
    env.close()
    coins = np.asarray(coins)
    if cache:
        np.savez_compressed(cache, frames=np.stack(frames), coins=coins)
    return frames, coins


def metrics(det, events, n_frames, lookback, horizon, all_events=None):
    pos = np.array([len(d) > 0 for d in det])
    central = np.array([any(abs(b["cx"] - 128) < 43 for b in d) for d in det])
    rec = sum(bool(pos[max(0, e - lookback):e].any()) for e in events)
    near = np.zeros(n_frames, bool)
    for e in (events if all_events is None else all_events):
        near[max(0, e - horizon):e + 1] = True
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
    arrow = det[330 * 4:350 * 4]
    return {"positive_frames": float(pos.mean()), "recall": rec / len(events),
            "false_positive_upper_bound": float(np.mean(~near[pos])) if pos.any() else 0.0,
            "lead_frames_median": float(np.median(leads)), "lead_frames_p25": float(np.percentile(leads, 25)),
            "lead_frames_p75": float(np.percentile(leads, 75)),
            "arrow_stretch_positive": float(np.mean([len(d) > 0 for d in arrow])),
            "arrow_stretch_blob_right_of_centre": float(np.mean([any(b["cx"] > 140 for b in d) for d in arrow]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", default="config/routes/ds3_search_route_v1.json")
    ap.add_argument("--upto", type=int, default=717)
    ap.add_argument("--mode", default="image", choices=["image", "game"])
    ap.add_argument("--template", default="images/coins.png")
    ap.add_argument("--cache", default=None, help="npz with the replayed frames (created if missing)")
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--out-image", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    suffix = "" if args.mode == "image" else "_game"
    args.out_image = args.out_image or f"results/ds3_coin_template{suffix}.png"
    args.out_json = args.out_json or f"results/ds3_coin_template{suffix}.json"

    frames, coins = capture(args.route, args.upto, args.cache)
    events = np.flatnonzero(np.diff(coins) > 0) + 1
    print(f"{len(frames)} frames, coins {coins[0]} -> {coins[-1]}, {len(events)} collection events", flush=True)
    eval_events = events
    if args.mode == "game":
        train, eval_events = events[0::2], events[1::2]
        bank, crops = game_bank(frames, events, train)
        print(f"game templates: {len(crops)} crops from {len(train)} training events -> {len(bank)} scaled templates; "
              f"evaluating on {len(eval_events)} held-out events", flush=True)
        if crops:
            row = [cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4), interpolation=cv2.INTER_NEAREST) for c in crops]
            hh = max(r.shape[0] for r in row)
            row = [cv2.copyMakeBorder(r, 0, hh - r.shape[0], 0, 4, cv2.BORDER_CONSTANT) for r in row]
            cv2.imwrite(os.path.join(ROOT_DIR, args.out_image.replace(".png", "_templates.png")), np.hstack(row))
    else:
        bank = make_bank(args.template)
    ths = [float(t) for t in args.thresholds.split(",")]
    from multiprocessing import Pool
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(bank, min(ths))) as pool:
        raw = pool.map(_detect_one, frames, chunksize=16)
    results = {}
    for th in ths:
        det = [[b for b in d if b["score"] >= th] for d in raw]
        m = metrics(det, eval_events, len(frames), args.lookback, args.horizon, all_events=events)
        results[f"{th:.2f}"] = m
        print(f"threshold {th:.2f}: positive {m['positive_frames']:5.1%} | recall {m['recall'] * len(eval_events):.0f}/{len(eval_events)} | "
              f"positives without collection within {args.horizon} frames {m['false_positive_upper_bound']:5.1%} | "
              f"lead median {m['lead_frames_median']:.0f} (p25 {m['lead_frames_p25']:.0f}, p75 {m['lead_frames_p75']:.0f}) | "
              f"arrow stretch: seen {m['arrow_stretch_positive']:5.1%}, right of centre {m['arrow_stretch_blob_right_of_centre']:5.1%}",
              flush=True)

    # overlay at the threshold with the best precision among those that keep recall >= 0.9
    ok = [t for t in ths if results[f"{t:.2f}"]["recall"] >= 0.9]
    th = max(ok, key=lambda t: (-results[f"{t:.2f}"]["false_positive_upper_bound"], t)) if ok else ths[0]
    det = [[b for b in d if b["score"] >= th] for d in raw]
    pos = np.array([len(d) > 0 for d in det])
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
        for b in det[i]:
            x, y, w, h = b["box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        return cv2.putText(img, f"{label} f={i} c={coins[i]} th={th}", (3, 34), 0, .36, (0, 255, 255), 1)

    tiles = [tile(i, "true") for i in true_idx] + [tile(i, "suspect") for i in sus_idx] + [tile(i, "arrow") for i in arrow_idx]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(ROOT_DIR, args.out_image),
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    out = {"mode": args.mode, "template": args.template if args.mode == "image" else "crops from the game frames",
           "evaluated_events": int(len(eval_events)), "heights": list(HEIGHTS), "route": args.route, "frames": len(frames),
           "collection_events": int(len(events)), "lookback": args.lookback, "horizon": args.horizon,
           "overlay_threshold": th, "thresholds": results}
    with open(os.path.join(ROOT_DIR, args.out_json), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {args.out_image} (threshold {th}), {args.out_json}")


if __name__ == "__main__":
    main()
