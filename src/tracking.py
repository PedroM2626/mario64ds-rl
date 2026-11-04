"""MLflow if it is installed, otherwise a no-op stand-in with the same calls.

Training must not depend on the tracking server being installed: the metrics
also go to TensorBoard and the trainers' own CSV/JSON outputs.
"""
from __future__ import annotations

from contextlib import nullcontext

try:
    import mlflow
except ImportError:
    class _NoMlflow:
        available = False

        def start_run(self, *args, **kwargs):
            return nullcontext()

        def active_run(self):
            return None

        def __getattr__(self, name):  # set_tracking_uri, log_param, log_metric, ...
            return lambda *args, **kwargs: None

    mlflow = _NoMlflow()

__all__ = ["mlflow"]
