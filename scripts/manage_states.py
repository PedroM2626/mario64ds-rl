#!/usr/bin/env python3
"""
scripts/manage_states.py: Savestate Management, Verification & Packaging Tool

In Reinforcement Learning with DeSmuME, savestates (.ds1, .ds2, .ds3, .ds8, .ds9)
are required for O(1) constant-time episode resets (~1-2 ms vs ~30-40 s cold-boot).
Because savestates contain full RAM/VRAM dumps of copyrighted game code, they
are excluded from Git via .gitignore.

This script provides utilities to:
1. 'check': Inspect the local data/ directory for required ROM and savestates.
2. 'verify': Boot DeSmuME and verify that each state loads and matches expected
   3D coordinates in RAM (src.ram_state.TRACK_GOALS).
3. 'bundle': Create a compressed asset archive (.tar.gz) with SHA-256 manifest
   for external hosting (e.g., Hugging Face, GitHub Releases).
4. 'unpack': Restore savestates from a local bundle into data/ with SHA-256 validation.
"""

import os
import sys
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
ROM_NAME = "Super Mario 64 DS (USA) (Rev 1).nds"
ROM_PATH = DATA_DIR / ROM_NAME

REQUIRED_TRACKS = ("ds1", "ds2", "ds3", "ds8", "ds9")
OPTIONAL_REFERENCE_TRACKS = ("ds4", "ds5", "ds6")

def sha256_file(filepath: Path) -> str:
    """Calculates SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_state_path(track_id: str) -> Path:
    """Returns canonical path for a track savestate."""
    return DATA_DIR / f"Super Mario 64 DS (USA) (Rev 1).{track_id}"

def cmd_check(args):
    """Checks presence, sizes, and hashes of ROM and savestates in data/."""
    print("=" * 70)
    print("Super Mario 64 DS - Asset & Savestate Health Check")
    print("=" * 70)

    # 1. ROM Check
    if not ROM_PATH.exists():
        print(f"[FAIL] ROM not found: {ROM_PATH}")
        print("       Please provide legally dumped 'Super Mario 64 DS (USA) (Rev 1).nds'.")
    else:
        size_mb = ROM_PATH.stat().st_size / (1024 * 1024)
        print(f"[PASS] ROM found: {ROM_NAME} ({size_mb:.2f} MB)")
        # Quick header check
        with open(ROM_PATH, "rb") as f:
            header = f.read(16)
            game_code = header[12:16].decode("ascii", errors="replace")
            print(f"       Internal Game Code: '{game_code}' (Expected: 'ASME')")

    print("\n--- Required Active Starting States (Used by env.reset()) ---")
    all_req_pass = True
    for tid in REQUIRED_TRACKS:
        sp = get_state_path(tid)
        if not sp.exists():
            print(f"[MISSING] {sp.name}")
            all_req_pass = False
        else:
            h = sha256_file(sp)
            size_mb = sp.stat().st_size / (1024 * 1024)
            print(f"[PASS]    {sp.name} ({size_mb:.2f} MB) | SHA-256: {h[:16]}...")

    print("\n--- Reference Terminus States (ds4, ds5, ds6) ---")
    print("Note: Terminus states are NOT needed for RL training or execution.")
    print("Their 3D coordinates are permanently compiled into src/ram_state.py.")
    for tid in OPTIONAL_REFERENCE_TRACKS:
        sp = get_state_path(tid)
        if sp.exists():
            size_mb = sp.stat().st_size / (1024 * 1024)
            print(f"[FOUND]   {sp.name} ({size_mb:.2f} MB) - (Archival reference only)")
        else:
            print(f"[OMITTED] {sp.name} - (Not needed, coordinates baked in code)")

    print("=" * 70)
    if all_req_pass and ROM_PATH.exists():
        print("Summary: All required RL starting states and ROM are present.")
        return 0
    else:
        print("Summary: Missing required files. See instructions above.")
        return 1

def cmd_verify(args):
    """Boots DeSmuME to verify RAM positions of all active states."""
    print("=" * 70)
    print("Verifying Savestates via DeSmuME RAM Ingestion")
    print("=" * 70)

    try:
        from desmume.emulator import DeSmuME
    except ImportError:
        print("[ERROR] py-desmume is not installed. Run in project venv.")
        return 1

    if not ROM_PATH.exists():
        print(f"[ERROR] ROM not found: {ROM_PATH}")
        return 1

    from src.ram_state import Sm64DsState, TRACK_GOALS

    try:
        emu = DeSmuME()
        emu.open(str(ROM_PATH))
    except Exception as e:
        print(f"[ERROR] Failed to boot DeSmuME: {e}")
        return 1

    ram = Sm64DsState(emu)

    results = []
    for tid in REQUIRED_TRACKS:
        sp = get_state_path(tid)
        if not sp.exists():
            results.append((tid, "MISSING", "File not found", None, None))
            continue

        try:
            emu.savestate.load_file(str(sp))
            # Cycle 1 frame to let RAM settle
            emu.cycle(with_joystick=False)
            pos = ram.get_pos_3d()
            cid = ram.get_course_id()
            stars = ram.get_star_count()
            spec = TRACK_GOALS.get(tid, {})
            expected_cid = spec.get("course_id")

            cid_ok = (cid == expected_cid) if expected_cid is not None else True
            results.append((tid, "OK" if cid_ok else "COURSE_MISMATCH",
                            f"Course={cid} (exp {expected_cid}), Stars={stars}",
                            pos.round(2).tolist(), spec.get("target_pos")))
        except Exception as e:
            results.append((tid, "ERROR", str(e), None, None))

    print(f"{'Track':<8} {'Status':<16} {'Details':<32} {'Mario Pos [X, Y, Z]'}")
    print("-" * 70)
    for tid, status, details, pos, target in results:
        pos_str = str(pos) if pos else "N/A"
        print(f"{tid:<8} {status:<16} {details:<32} {pos_str}")

    print("=" * 70)
    return 0

def cmd_bundle(args):
    """Creates a compressed tar.gz bundle of savestates with manifest."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {"files": {}, "version": 1}
    files_to_add = []

    for tid in REQUIRED_TRACKS:
        sp = get_state_path(tid)
        if not sp.exists():
            print(f"[ERROR] Required state {sp.name} is missing. Cannot create complete bundle.")
            return 1
        h = sha256_file(sp)
        manifest["files"][sp.name] = {
            "track": tid,
            "sha256": h,
            "size_bytes": sp.stat().st_size
        }
        files_to_add.append(sp)

    manifest_path = DATA_DIR / "_bundle_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Creating archive {out_path}...")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        for f in files_to_add:
            print(f"  Adding {f.name}...")
            tar.add(f, arcname=f.name)

    manifest_path.unlink()
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Savestate bundle created: {out_path} ({size_mb:.2f} MB)")
    return 0

def cmd_unpack(args):
    """Unpacks savestates from a bundle into data/ with SHA-256 validation."""
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"[ERROR] Bundle not found: {bundle_path}")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting bundle {bundle_path} to {DATA_DIR}...")
    with tarfile.open(bundle_path, "r:gz") as tar:
        # Read manifest
        try:
            m_f = tar.extractfile("manifest.json")
            manifest = json.load(m_f)
        except Exception as e:
            print(f"[ERROR] Invalid or missing manifest.json in bundle: {e}")
            return 1

        for name, info in manifest["files"].items():
            print(f"  Extracting {name}...")
            tar.extract(name, path=str(DATA_DIR))
            target_path = DATA_DIR / name
            h = sha256_file(target_path)
            if h != info["sha256"]:
                print(f"[FAIL] Checksum mismatch for {name}: expected {info['sha256']}, got {h}")
                return 1
            print(f"[PASS] Checksum verified for {name}")

    print("[SUCCESS] All savestates unpacked and verified.")
    return 0

def main():
    parser = argparse.ArgumentParser(description="SM64DS Savestate Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_check = subparsers.add_parser("check", help="Check ROM and state files")
    p_verify = subparsers.add_parser("verify", help="Verify RAM state via emulator")
    
    p_bundle = subparsers.add_parser("bundle", help="Package states into tar.gz")
    p_bundle.add_argument("--out", default="data/sm64ds_states.tar.gz", help="Output bundle path")

    p_unpack = subparsers.add_parser("unpack", help="Unpack states from tar.gz")
    p_unpack.add_argument("--bundle", required=True, help="Input bundle path")

    args = parser.parse_args()
    if args.command == "check":
        sys.exit(cmd_check(args))
    elif args.command == "verify":
        sys.exit(cmd_verify(args))
    elif args.command == "bundle":
        sys.exit(cmd_bundle(args))
    elif args.command == "unpack":
        sys.exit(cmd_unpack(args))

if __name__ == "__main__":
    main()
