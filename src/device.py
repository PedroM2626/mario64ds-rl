"""Device resolution shared by trainers, evaluators and deployment scripts.

A single place decides what --device auto|cpu|cuda means, so an explicit CUDA
request never fails deep inside torch.load(map_location=...) with an opaque
message. The published benchmark is defined on CPU (see
results/device_divergence.json), so evaluators default to cpu and only use auto
when the caller asks for it.
"""

from __future__ import annotations


def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda[:N] to a concrete torch device string.

    auto picks CUDA when it is available, otherwise CPU. An explicit CUDA
    request on a CPU-only machine raises a short, actionable error instead of
    producing a stack trace from deep inside torch.
    """
    import torch

    req = str(requested or "auto").strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cpu":
        return "cpu"
    if req == "cuda" or req.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {requested!r} was requested but CUDA is not available. "
                "Use --device cpu (the instrument that reproduces the published "
                "benchmark) or --device auto."
            )
        return req
    raise SystemExit(f"unknown device {requested!r}; use auto, cpu or cuda[:N]")
