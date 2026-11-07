"""Ordered 3D route along a recorded human path, used as a progress measure.

The straight-line distance to the Tick Tock Clock star is ~98% height and the
ascent winds around the tower, so horizontal moves along the real path almost
never lower it (docs/REWARD_SYSTEM_AND_RAM.md, section 10.8). A route instead
measures arc length along an ordered polyline of positions that a human
actually traversed (scripts/prepare_human_route.py builds it from a recording).

Progress is searched only in a bounded window ahead of the best arc length so
far, and only inside a corridor around the polyline, so a fall onto a lower
part of the tower or a far teleport cannot claim distant credit.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROUTE_SCHEMA = 1


def resample_polyline(points, spacing: float) -> np.ndarray:
    """Points every `spacing` units of 3D path length (stationary samples drop out)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("a route needs at least two 3D points")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        raise ValueError("the recorded path has zero length")
    marks = np.arange(0.0, total, float(spacing))
    if total - marks[-1] > 1e-6:
        marks = np.append(marks, total)
    out = np.empty((len(marks), 3))
    for k in range(3):
        out[:, k] = np.interp(marks, cum, pts[:, k])
    return out


def erase_loops(waypoints, radius: float) -> np.ndarray:
    """Cut every loop where the path returns within `radius` of an earlier point.

    A human run up Tick Tock Clock contains falls followed by a re-climb of the
    same section, and laps ridden on the moving clock hands while waiting. Kept
    in the route, both would pay arc-length progress for repeating ground. The
    earliest earlier point within reach wins, so the largest loop is removed.
    """
    kept = []
    for q in np.asarray(waypoints, dtype=np.float64):
        if len(kept) > 2:
            near = np.flatnonzero(np.linalg.norm(np.asarray(kept[:-2]) - q, axis=1) < radius)
            if len(near):
                kept = kept[:near[0] + 1]
                continue
        kept.append(q)
    return np.asarray(kept)


class Route:
    def __init__(self, waypoints, corridor_radius: float = 250.0,
                 lookahead: float = 600.0, lookbehind: float = 300.0, meta=None):
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3 or len(self.waypoints) < 2:
            raise ValueError("a route needs at least two 3D waypoints")
        self.seg = np.diff(self.waypoints, axis=0)
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        if np.any(self.seg_len <= 0):
            raise ValueError("consecutive route waypoints must differ")
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.corridor_radius = float(corridor_radius)
        self.lookahead = float(lookahead)
        self.lookbehind = float(lookbehind)
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str) -> "Route":
        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
        if spec.get("schema") != ROUTE_SCHEMA:
            raise ValueError(f"unsupported route schema in {path}: {spec.get('schema')}")
        return cls(spec["waypoints"], spec.get("corridor_radius", 250.0),
                   spec.get("lookahead", 600.0), spec.get("lookbehind", 300.0), meta=spec)

    def passed(self, old_s: float, new_s: float) -> int:
        """How many waypoints lie in the arc-length interval (old_s, new_s] (waypoint k sits at cum[k])."""
        return int(np.searchsorted(self.cum, new_s, side="right") - np.searchsorted(self.cum, old_s, side="right"))

    def _project(self, pos, lo: int, hi: int):
        """(arc_length, distance) of the closest point on segments lo..hi-1."""
        a = self.waypoints[lo:hi]
        d = self.seg[lo:hi]
        t = np.clip(np.einsum("ij,ij->i", pos - a, d) / (self.seg_len[lo:hi] ** 2), 0.0, 1.0)
        closest = a + t[:, None] * d
        dist = np.linalg.norm(closest - pos, axis=1)
        i = int(np.argmin(dist))
        return float(self.cum[lo + i] + t[i] * self.seg_len[lo + i]), float(dist[i])

    def locate(self, pos) -> float | None:
        """Global arc length of pos (used once at reset), or None off-corridor."""
        s, dist = self._project(np.asarray(pos, dtype=np.float64), 0, len(self.seg))
        return s if dist <= self.corridor_radius else None

    def progress(self, pos, best_s: float) -> float | None:
        """Arc length of pos searched near best_s, or None when off-corridor."""
        # segments that overlap [best_s - lookbehind, best_s + lookahead]
        lo = max(int(np.searchsorted(self.cum, best_s - self.lookbehind, side="right")) - 1, 0)
        hi = int(np.searchsorted(self.cum, best_s + self.lookahead, side="left"))
        hi = min(max(hi, lo + 1), len(self.seg))
        s, dist = self._project(np.asarray(pos, dtype=np.float64), lo, hi)
        # a long segment can project past the window; that is still a skip
        if dist > self.corridor_radius or s > best_s + self.lookahead:
            return None
        return s


def route_checkpoints(route_path: str, base_dir: str | None = None) -> list[str]:
    """Absolute savestate paths of the checkpoints listed in a route file."""
    route = Route.load(route_path)
    base = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for cp in route.meta.get("checkpoints", []):
        p = cp["path"]
        out.append(p if os.path.isabs(p) else os.path.join(base, p))
    return out
