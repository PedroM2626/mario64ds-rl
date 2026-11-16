"""Run another script of this repository at idle CPU priority with a limited number of torch threads.

Used for work that must not slow down other jobs on the same machine (the CPU is shared with a Mario Galaxy training).

Usage:
    python scripts/lowprio.py scripts/collect_distill_data.py --episodes 4 ...
Environment: LOWPRIO_THREADS (default 2) = torch/OpenMP threads.
"""
from __future__ import annotations

import os
import runpy
import sys

THREADS = int(os.environ.get("LOWPRIO_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

try:
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000040):   # IDLE_PRIORITY_CLASS
            raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
    else:
        os.nice(19)
except Exception as exc:  # the priority is a courtesy; the job still runs without it
    print("lowprio: could not lower the priority:", exc, file=sys.stderr)

try:
    import torch
    torch.set_num_threads(THREADS)
except Exception:
    pass

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
script = os.path.abspath(sys.argv[1])
root = os.path.dirname(os.path.dirname(script))
sys.path.insert(0, root)
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
