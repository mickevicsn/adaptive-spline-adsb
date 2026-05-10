"""ADS-B velocity confidence scaling for V-Spline observation residuals.

The original V-Spline objective treats every paired velocity row as equally
trustworthy.  ADS-B velocity fields may be stale or asynchronously aggregated
relative to position fields, so this helper computes a deterministic per-sample
scale in [min_scale, 1] from position/velocity consistency checks.  It is used
as a multiplier on the velocity residual weight, not as a replacement for the
reported velocity observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import math
import numpy as np


@dataclass(frozen=True)
class VelocityConfidenceConfig:
    """Heuristic gates for per-observation ADS-B velocity trust."""

    reported_vs_position_gate_mps: float = 55.0
    vertical_rate_gate_mps: float = 9.0
    track_angle_gate_deg: float = 35.0
    low_motion_speed_mps: float = 3.0
    duplicate_distance_m: float = 10.0
    # 4BAAD9 report action: allow demonstrably stale/asynchronous velocity rows
    # to lose all influence instead of retaining a nonzero residual floor.  The
    # caller still controls the base velocity weight, so this only affects rows
    # that fail the consistency gates below.
    min_scale: float = 0.0
    catastrophic_multiplier: float = 3.0
    # Quantized/duplicate positions are especially harmful for nodal Hermite
    # fits.  The previous hard-coded 0.25 kept stale velocity in the objective;
    # expose it as a config field and default to zero influence for those rows.
    low_motion_scale: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_velocity_confidence_scale(
    t: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    config: VelocityConfidenceConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-row velocity residual scales and diagnostics.

    The checks intentionally use only paired V-Spline arrays so the mathematical
    cores stay independent from raw ADS-B packet schemas.  Downweighting reasons
    include reported-vs-finite-difference disagreement, vertical-rate mismatch,
    track-vs-displacement angular inconsistency, and near-duplicate/low-motion
    rows where ADS-B quantization makes derivatives unreliable.
    """
    cfg = config or VelocityConfidenceConfig()
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    n = int(t.size)
    if n == 0:
        return np.zeros(0, dtype=float), {"enabled": False, "reason": "empty_input", "config": cfg.as_dict()}
    if n < 2 or y.shape != v.shape or y.shape[0] != n or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(v)):
        return np.ones(n, dtype=float), {"enabled": False, "reason": "invalid_or_too_few_samples", "config": cfg.as_dict()}
    dt = np.diff(t)
    if not np.all(dt > 0):
        return np.ones(n, dtype=float), {"enabled": False, "reason": "nonpositive_dt", "config": cfg.as_dict()}

    fd = _central_position_velocity(t, y)
    mismatch = np.linalg.norm(v - fd, axis=1)
    gate = max(float(cfg.reported_vs_position_gate_mps), 1e-9)
    mismatch_scale = np.minimum(1.0, gate / np.maximum(mismatch, gate))
    catastrophic = mismatch > float(cfg.catastrophic_multiplier) * gate
    mismatch_scale = np.where(catastrophic, float(cfg.min_scale), mismatch_scale)

    if y.shape[1] >= 3 and v.shape[1] >= 3:
        vertical_mismatch = np.abs(v[:, 2] - fd[:, 2])
        vgate = max(float(cfg.vertical_rate_gate_mps), 1e-9)
        vertical_scale = np.minimum(1.0, vgate / np.maximum(vertical_mismatch, vgate))
    else:
        vertical_mismatch = np.zeros(n, dtype=float)
        vertical_scale = np.ones(n, dtype=float)

    if y.shape[1] >= 2 and v.shape[1] >= 2:
        horiz_reported = v[:, :2]
        horiz_fd = fd[:, :2]
        reported_speed = np.linalg.norm(horiz_reported, axis=1)
        fd_speed = np.linalg.norm(horiz_fd, axis=1)
        dot = np.sum(horiz_reported * horiz_fd, axis=1)
        denom = np.maximum(reported_speed * fd_speed, 1e-9)
        cosang = np.clip(dot / denom, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosang))
        angle_gate = max(float(cfg.track_angle_gate_deg), 1e-6)
        angle_scale = np.minimum(1.0, angle_gate / np.maximum(angle_deg, angle_gate))
        angle_scale = np.where((reported_speed < cfg.low_motion_speed_mps) | (fd_speed < cfg.low_motion_speed_mps), 1.0, angle_scale)
    else:
        reported_speed = np.linalg.norm(v, axis=1)
        fd_speed = np.linalg.norm(fd, axis=1)
        angle_deg = np.zeros(n, dtype=float)
        angle_scale = np.ones(n, dtype=float)

    # Quantized/duplicate positions can make finite-difference velocities nearly
    # zero even while reported ADS-B velocity carries a stale airborne value.  Keep
    # such rows in the objective but make their velocity influence weak.
    nearest_distance = _nearest_neighbor_distance(y)
    low_motion = (fd_speed < float(cfg.low_motion_speed_mps)) | (nearest_distance < float(cfg.duplicate_distance_m))
    low_motion_scale = np.where(low_motion, float(cfg.low_motion_scale), 1.0)

    scale = mismatch_scale * vertical_scale * angle_scale * low_motion_scale
    scale = np.clip(scale, float(cfg.min_scale), 1.0)

    return scale.astype(float), {
        "enabled": True,
        "config": cfg.as_dict(),
        "min_scale": float(np.min(scale)),
        "median_scale": float(np.median(scale)),
        "mean_scale": float(np.mean(scale)),
        "downweighted_count": int(np.sum(scale < 0.999)),
        "catastrophic_mismatch_count": int(np.sum(catastrophic)),
        "reported_vs_position_mismatch_mps": _summary(mismatch),
        "vertical_rate_mismatch_mps": _summary(vertical_mismatch),
        "track_angle_mismatch_deg": _summary(angle_deg),
        "finite_difference_speed_mps": _summary(fd_speed),
        "reported_speed_mps": _summary(reported_speed),
        "near_duplicate_or_low_motion_count": int(np.sum(low_motion)),
    }


def _central_position_velocity(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = int(t.size)
    fd = np.zeros_like(y, dtype=float)
    fd[0, :] = (y[1, :] - y[0, :]) / max(float(t[1] - t[0]), 1e-9)
    fd[-1, :] = (y[-1, :] - y[-2, :]) / max(float(t[-1] - t[-2]), 1e-9)
    if n > 2:
        denom = np.maximum(t[2:] - t[:-2], 1e-9)
        fd[1:-1, :] = (y[2:, :] - y[:-2, :]) / denom[:, None]
    return fd


def _nearest_neighbor_distance(y: np.ndarray) -> np.ndarray:
    n = int(y.shape[0])
    if n == 1:
        return np.zeros(1, dtype=float)
    d = np.linalg.norm(np.diff(y, axis=0), axis=1)
    out = np.empty(n, dtype=float)
    out[0] = d[0]
    out[-1] = d[-1]
    if n > 2:
        out[1:-1] = np.minimum(d[:-1], d[1:])
    return out


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }
