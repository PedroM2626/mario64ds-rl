"""Verify the large assets declared in models/ASSETS.json.

A clean clone contains the code but not the ~200 MB of checkpoints used by the
README commands. This tool reports which of those files are present, whether
their size/SHA256 match the manifest, and can refresh the manifest after a new
run. It performs no training and does not need the emulator.

    python tools/verify_assets.py                 # report only
    python tools/verify_assets.py --require       # fail if a required asset is missing
    python tools/verify_assets.py --update        # rewrite size/sha256 for local files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "models", "ASSETS.json")


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--require", action="store_true",
                    help="exit non-zero if an asset marked required is missing")
    ap.add_argument("--update", action="store_true",
                    help="rewrite size/sha256 for assets that exist locally")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        data = json.load(f)
    assets = data.get("assets", [])
    if not assets:
        print(f"FAIL {args.manifest}: no assets listed", file=sys.stderr)
        return 1

    if args.update:
        updated = 0
        for a in assets:
            p = resolve(a["path"])
            if not os.path.exists(p):
                print(f"MISSING {a['path']} (not updated)")
                continue
            a["size"] = os.path.getsize(p)
            a["sha256"] = sha256_file(p)
            updated += 1
            print(f"UPDATED {a['path']} {a['size']} {a['sha256'][:12]}...")
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"manifest written: {args.manifest} ({updated} assets updated)")
        return 0

    missing, bad = [], []
    print(f"{'status':8} {'asset':54} {'size':>12}  sha256")
    for a in assets:
        p = resolve(a["path"])
        req = bool(a.get("required", False))
        if not os.path.exists(p):
            status = "MISSING"
            if req:
                missing.append(a["path"])
            print(f"{status:8} {a['path']:54} {'-':>12}  expected {str(a.get('sha256',''))[:12]}...")
            continue
        size = os.path.getsize(p)
        digest = sha256_file(p)
        size_ok = not a.get("size") or int(a["size"]) == size
        hash_ok = not a.get("sha256") or str(a["sha256"]).lower() == digest.lower()
        ok = size_ok and hash_ok
        status = "OK" if ok else "MISMATCH"
        if not ok:
            bad.append(a["path"])
        print(f"{status:8} {a['path']:54} {size:>12}  {digest[:12]}...")

    print(f"\n{len(assets) - len(missing) - len(bad)}/{len(assets)} assets verified; "
          f"{len(missing)} missing, {len(bad)} mismatched")
    if bad:
        print("Run with --update only if you intentionally replaced these files; "
              "otherwise re-download the published bytes.", file=sys.stderr)
        return 1
    if missing and args.require:
        print("Missing required assets (use --require only when they should be present).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
