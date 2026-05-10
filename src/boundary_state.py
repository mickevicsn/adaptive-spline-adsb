"""Shared boundary-state estimation for C1 piecewise V-Spline stitching."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import math

import numpy as np

from raw_keyframe_vspline_adapter import PreparedVSplineSample
from trajectory_segmentation import AcceptedBoundary, DynamicSegment


@dataclass(frozen=True)
class BoundaryStateConfig:
    """Shared boundary-state estimator configuration.

    The 4BAAD9 audit found that raw boundary anchoring was too brittle for
    heavily segmented ADS-B tracks.  The default therefore uses a weighted
    compromise biased toward robust local regression; callers can still request
    ``position_source="raw_sample"`` for legacy/debug ablations.
    """

    position_source: Literal["raw_sample", "robust_local_regression", "weighted_compromise"] = "weighted_compromise"
    position_raw_weight: float = 0.35
    position_robust_weight: float = 0.65
    window_points: int = 11
    min_side_points: int = 4
    poly_order: int = 2
    robust_iters: int = 3
    huber_k: float = 1.345
    blend_reported_velocity_weight: float = 0.10
    max_velocity_factor: float = 2.0
    max_acceleration_mps2: float = 30.0


@dataclass(frozen=True)
class SharedBoundaryState:
    boundary_id: str
    t_boundary: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    confidence: float
    method: str
    diagnostics: dict[str, Any]
    acceleration_mps2: tuple[float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def position_array(self) -> np.ndarray:
        return np.asarray(self.position_m, dtype=float)

    @property
    def velocity_array(self) -> np.ndarray:
        return np.asarray(self.velocity_mps, dtype=float)

    @property
    def acceleration_array(self) -> np.ndarray | None:
        if self.acceleration_mps2 is None:
            return None
        return np.asarray(self.acceleration_mps2, dtype=float)


def estimate_shared_boundary_states(
    all_samples: Sequence[PreparedVSplineSample],
    boundaries: Sequence[AcceptedBoundary],
    config: BoundaryStateConfig,
) -> dict[str, SharedBoundaryState]:
    """Estimate one canonical position/velocity tuple for every boundary."""
    return {
        boundary.boundary_id: estimate_shared_boundary_state(all_samples, boundary, config)
        for boundary in boundaries
    }


def estimate_shared_boundary_state(
    all_samples: Sequence[PreparedVSplineSample],
    boundary: AcceptedBoundary,
    config: BoundaryStateConfig,
) -> SharedBoundaryState:
    idx = int(boundary.sample_index)
    if idx <= 0 or idx >= len(all_samples) - 1:
        sample = all_samples[idx]
        return SharedBoundaryState(
            boundary_id=boundary.boundary_id,
            t_boundary=float(sample.t),
            position_m=tuple(float(x) for x in sample.y),
            velocity_mps=tuple(float(x) for x in sample.v),
            confidence=0.25,
            method="raw_endpoint_fallback",
            diagnostics={"reason": "boundary at global edge"},
            acceleration_mps2=None,
        )

    t0 = float(all_samples[idx].t)
    left_idx = list(range(max(0, idx - int(config.window_points) + 1), idx + 1))
    right_idx = list(range(idx, min(len(all_samples), idx + int(config.window_points))))

    left = _fit_window(all_samples, left_idx, t0, config)
    right = _fit_window(all_samples, right_idx, t0, config)

    raw_pos = np.asarray(all_samples[idx].y, dtype=float)
    raw_vel = np.asarray(all_samples[idx].v, dtype=float)
    position_source = str(config.position_source).strip().lower()
    robust_acc: np.ndarray | None = None
    acc: np.ndarray | None = None
    position_objective: dict[str, Any] | None = None

    if not left["ok"] and not right["ok"]:
        pos = raw_pos.copy()
        vel = raw_vel.copy()
        acc: np.ndarray | None = None
        confidence = 0.15
        method = "raw_boundary_sample_fallback"
        robust_pos = None
        robust_acc = None
        position_error_m = 0.0
    else:
        estimates = [r for r in (left, right) if r["ok"]]
        weights = np.asarray([1.0 / max(float(r["residual_rms_m"]), 1e-6) for r in estimates], dtype=float)
        weights /= float(np.sum(weights))
        robust_pos = np.sum([w * r["position"] for w, r in zip(weights, estimates)], axis=0)
        vel = np.sum([w * r["velocity"] for w, r in zip(weights, estimates)], axis=0)
        acc_estimates = [np.asarray(r["acceleration"], dtype=float) for r in estimates if r.get("acceleration") is not None]
        if acc_estimates:
            if len(acc_estimates) == len(estimates):
                robust_acc = np.sum([w * a for w, a in zip(weights, acc_estimates)], axis=0)
            else:
                robust_acc = np.mean(acc_estimates, axis=0)
            acc = _clip_acceleration(np.asarray(robust_acc, dtype=float), config)
        else:
            robust_acc = None
            acc = None

        if position_source == "raw_sample":
            pos = raw_pos.copy()
            method = "raw_position_with_robust_local_quadratic_velocity"
            position_objective = _boundary_position_objective(
                pos,
                raw_pos,
                np.asarray(robust_pos, dtype=float),
                raw_weight=float(config.position_raw_weight),
                robust_weight=float(config.position_robust_weight),
            )
        elif position_source == "robust_local_regression":
            pos = np.asarray(robust_pos, dtype=float)
            method = "robust_local_quadratic_position_velocity_blend"
            position_objective = _boundary_position_objective(
                pos,
                raw_pos,
                np.asarray(robust_pos, dtype=float),
                raw_weight=float(config.position_raw_weight),
                robust_weight=float(config.position_robust_weight),
            )
        elif position_source == "weighted_compromise":
            pos = _weighted_boundary_position(
                raw_position=raw_pos,
                robust_position=np.asarray(robust_pos, dtype=float),
                raw_weight=float(config.position_raw_weight),
                robust_weight=float(config.position_robust_weight),
            )
            method = "weighted_raw_and_robust_position_with_robust_local_quadratic_velocity"
            position_objective = _boundary_position_objective(
                pos,
                raw_pos,
                np.asarray(robust_pos, dtype=float),
                raw_weight=float(config.position_raw_weight),
                robust_weight=float(config.position_robust_weight),
            )
        else:
            raise ValueError(
                "BoundaryStateConfig.position_source must be "
                "'raw_sample', 'robust_local_regression', or 'weighted_compromise'"
            )
        position_error_m = float(np.linalg.norm(np.asarray(robust_pos, dtype=float) - raw_pos))

        if float(config.blend_reported_velocity_weight) > 0.0:
            reported = raw_vel
            a = min(max(float(config.blend_reported_velocity_weight), 0.0), 1.0)
            vel = (1.0 - a) * vel + a * reported

        vel = _clip_velocity(vel, all_samples, idx, config)

        lr_agreement = 0.0
        if left["ok"] and right["ok"]:
            lr_agreement = float(np.linalg.norm(left["velocity"] - right["velocity"]))
        residual = float(np.mean([float(r["residual_rms_m"]) for r in estimates]))
        confidence = float(1.0 / (1.0 + residual / 50.0 + lr_agreement / 100.0))
        confidence = min(max(confidence, 0.0), 1.0)

    return SharedBoundaryState(
        boundary_id=boundary.boundary_id,
        t_boundary=t0,
        position_m=tuple(float(x) for x in pos),
        velocity_mps=tuple(float(x) for x in vel),
        confidence=confidence,
        method=method,
        diagnostics={
            "boundary_sample_index": idx,
            "position_source": position_source,
            "raw_boundary_position_m": raw_pos.tolist(),
            "raw_boundary_velocity_mps": raw_vel.tolist(),
            "robust_boundary_position_m": None if robust_pos is None else np.asarray(robust_pos, dtype=float).tolist(),
            "robust_boundary_acceleration_mps2": None if robust_acc is None else np.asarray(robust_acc, dtype=float).tolist(),
            "selected_boundary_acceleration_mps2": None if acc is None else np.asarray(acc, dtype=float).tolist(),
            "selected_boundary_position_m": np.asarray(pos, dtype=float).tolist(),
            "selected_minus_raw_position_error_m": float(np.linalg.norm(np.asarray(pos, dtype=float) - raw_pos)),
            "selected_minus_robust_position_error_m": (
                None if robust_pos is None else float(np.linalg.norm(np.asarray(pos, dtype=float) - np.asarray(robust_pos, dtype=float)))
            ),
            "robust_minus_raw_position_error_m": float(position_error_m),
            "position_objective": position_objective,
            "left": _clean_report(left),
            "right": _clean_report(right),
            "config": asdict(config),
        },
        acceleration_mps2=None if acc is None else tuple(float(x) for x in np.asarray(acc, dtype=float)),
    )



def _weighted_boundary_position(
    *,
    raw_position: np.ndarray,
    robust_position: np.ndarray,
    raw_weight: float,
    robust_weight: float,
) -> np.ndarray:
    """Weighted least-squares compromise between raw and robust positions.

    The returned point minimizes:

        w_raw ||p - p_raw||^2 + w_robust ||p - p_robust||^2

    This keeps the segment join near the raw ADS-B boundary point without
    forcing the entire reconstruction through one noisy sample.
    """
    wr = max(float(raw_weight), 0.0)
    wb = max(float(robust_weight), 0.0)
    total = wr + wb
    if total <= 0.0:
        return np.asarray(raw_position, dtype=float).copy()
    return (wr * np.asarray(raw_position, dtype=float) + wb * np.asarray(robust_position, dtype=float)) / total


def _boundary_position_objective(
    selected_position: np.ndarray,
    raw_position: np.ndarray,
    robust_position: np.ndarray,
    *,
    raw_weight: float,
    robust_weight: float,
) -> dict[str, float]:
    selected = np.asarray(selected_position, dtype=float)
    raw = np.asarray(raw_position, dtype=float)
    robust = np.asarray(robust_position, dtype=float)
    wr = max(float(raw_weight), 0.0)
    wb = max(float(robust_weight), 0.0)
    raw_error = float(np.linalg.norm(selected - raw))
    robust_error = float(np.linalg.norm(selected - robust))
    return {
        "raw_weight": wr,
        "robust_weight": wb,
        "raw_error_m": raw_error,
        "robust_error_m": robust_error,
        "weighted_sse_m2": float(wr * raw_error * raw_error + wb * robust_error * robust_error),
    }


def _fit_window(
    samples: Sequence[PreparedVSplineSample],
    indices: Sequence[int],
    t_boundary: float,
    config: BoundaryStateConfig,
) -> dict[str, Any]:
    if len(indices) < int(config.min_side_points):
        return {"ok": False, "reason": "too_few_points", "n": len(indices)}

    t = np.asarray([samples[i].t for i in indices], dtype=float) - float(t_boundary)
    y = np.asarray([samples[i].y for i in indices], dtype=float)
    order = min(int(config.poly_order), len(indices) - 1)
    if order < 1:
        return {"ok": False, "reason": "polynomial_order_too_low", "n": len(indices)}

    scale = max(float(np.max(np.abs(t))), 1.0)
    tau = t / scale
    a = np.vander(tau, N=order + 1, increasing=True)
    weights = np.ones(len(indices), dtype=float)

    beta = np.zeros((order + 1, y.shape[1]), dtype=float)
    for _ in range(max(1, int(config.robust_iters))):
        try:
            aw = a * np.sqrt(weights)[:, None]
            yw = y * np.sqrt(weights)[:, None]
            beta = np.linalg.lstsq(aw, yw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"ok": False, "reason": "singular_local_regression", "n": len(indices)}

        residual = y - a @ beta
        residual_norm = np.linalg.norm(residual, axis=1)
        sigma = 1.4826 * np.median(np.abs(residual_norm - np.median(residual_norm)))
        sigma = max(float(sigma), 1e-6)
        threshold = float(config.huber_k) * sigma
        weights = np.where(residual_norm <= threshold, 1.0, threshold / np.maximum(residual_norm, 1e-9))

    pred = a @ beta
    residual = y - pred
    residual_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    # beta[0] is position at tau=0. beta[1] is dpos/dtau, so divide by scale.
    # beta[2] is the quadratic coefficient in tau; d2pos/dt2 = 2*beta[2]/scale^2.
    position = beta[0]
    velocity = beta[1] / scale
    acceleration = (2.0 * beta[2] / (scale * scale)) if order >= 2 else None
    cond = float(np.linalg.cond(a)) if a.size else float("inf")

    return {
        "ok": True,
        "n": len(indices),
        "indices": list(int(i) for i in indices),
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "residual_rms_m": residual_rms,
        "condition_number": cond,
        "scale_s": scale,
    }


def _clip_velocity(
    vel: np.ndarray,
    samples: Sequence[PreparedVSplineSample],
    idx: int,
    config: BoundaryStateConfig,
) -> np.ndarray:
    local = samples[max(0, idx - 3) : min(len(samples), idx + 4)]
    reported = np.asarray([s.v for s in local], dtype=float)
    speeds = np.linalg.norm(reported, axis=1)
    ref = float(np.median(speeds)) if speeds.size else float(np.linalg.norm(vel))
    max_speed = max(10.0, float(config.max_velocity_factor) * max(ref, 1e-6))
    speed = float(np.linalg.norm(vel))
    if speed > max_speed:
        return vel * (max_speed / speed)
    return vel


def _clip_acceleration(acc: np.ndarray, config: BoundaryStateConfig) -> np.ndarray:
    max_acc = max(float(config.max_acceleration_mps2), 1e-9)
    norm = float(np.linalg.norm(acc))
    if norm > max_acc:
        return np.asarray(acc, dtype=float) * (max_acc / norm)
    return np.asarray(acc, dtype=float)


def _clean_report(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in report.items():
        if key in {"position", "velocity"}:
            out[key] = np.asarray(value, dtype=float).tolist()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating,)):
            out[key] = float(value) if math.isfinite(float(value)) else None
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, float):
            out[key] = value if math.isfinite(value) else None
        else:
            out[key] = value
    return out
