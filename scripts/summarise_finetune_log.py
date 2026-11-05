"""Turn a fine-tune watcher log (scripts/run_ds3_finetune.sh) into results/ds3_finetune.json."""
import json
import re
import sys

log, out = sys.argv[1], sys.argv[2]
rows, cur = [], None
for line in open(log, encoding="utf-8", errors="replace"):
    m = re.match(r"== step_(\d+)_steps", line)
    if m:
        cur = {"checkpoint_steps": int(m.group(1))}
        rows.append(cur)
    elif cur is not None and line.startswith("arrow-states:"):
        cur["arrow_died"] = line.count(": died")
        cur["arrow_min_dist"] = [int(x) for x in re.findall(r"d=(\d+)", line)]
    elif cur is not None and line.startswith("hard spot:"):
        cur["hard_spot_reached"] = line.count(": fini")
        cur["hard_spot_states"] = 4
    elif cur is not None and line.startswith("from start:"):
        j = json.loads(line[len("from start:"):].strip())
        cur["from_start"] = {k: j[k] for k in ("end", "decisions", "coins_gained", "min_dist")}
json.dump({"run": "ds3_finetune3", "settings": "lr 1e-5, target_kl 0.02, clip 0.1, ent 0, value warm-up 100k, 800k steps",
           "checkpoints": rows}, open(out, "w", encoding="utf-8"), indent=2)
print(len(rows), "checkpoints")
