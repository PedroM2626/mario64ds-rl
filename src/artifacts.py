"""Small provenance helpers for checkpoints and evaluation artifacts.

MLflow used to receive the full 20 MB checkpoints as artifacts on every run.
That made mlruns/ grow by gigabytes while adding little information. The
helpers here log a compact JSON record (path, size, SHA256) instead, and are
also reused by the asset manifest/verification tooling.
"""

from __future__ import annotations

import hashlib
import os


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: str) -> dict:
    path = os.path.abspath(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "size": os.path.getsize(path) if os.path.exists(path) else None,
        "sha256": sha256_file(path) if os.path.exists(path) else None,
    }


def log_model_reference(path: str, artifact_path: str = "models") -> dict:
    """Log a JSON reference instead of the whole checkpoint to the active MLflow run."""
    import mlflow

    rec = file_record(path)
    mlflow.log_dict(rec, artifact_file=f"{artifact_path}/{rec['name']}.json")
    return rec
