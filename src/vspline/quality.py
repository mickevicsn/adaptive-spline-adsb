"""Quality metrics for local and piecewise V-Spline fits.

The module intentionally separates two ideas:

* observation-fit diagnostics: how closely the reconstruction follows the ADS-B
  samples that were used as noisy observations; and
* reference-free trajectory-model diagnostics: whether the reconstructed curve is
  a useful continuous trajectory model when no ground-truth flight path exists.

The second family is designed for ADS-B research where the important product is
not only pointwise denoising, but a differentiable, locally interpretable model
of velocity, acceleration, curvature, jerk, and event-aware continuity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import math

import numpy as np

from trajectory_segmentation import DynamicSegment
from vspline.velocity_confidence import compute_velocity_confidence_scale


@dataclass(frozen=True)
class SegmentQualityReport:
    raw_fit_metrics: dict[str, float]
    motion_metrics: dict[str, float]
    continuity_metrics: dict[str, float]
    trajectory_model_metrics: dict[str, Any]
    bad_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_segment_quality(
    segment: DynamicSegment,
    fit: Any,
    *,
    render_step_s: float = 0.25,
    bad_raw_rmse_m: float = 200.0,
    bad_jerk_rms_mps3: float = 20.0,
) -> SegmentQualityReport:
    t = np.asarray([sample.t for sample in segment.samples], dtype=float)
    y = np.asarray([sample.y for sample in segment.samples], dtype=float)
    fitted = fit.evaluate(t, deriv=0)
    raw_fit = position_error_metrics(y, fitted)
    motion = motion_metrics(fit, step_s=render_step_s)
    trajectory = trajectory_model_metrics(
        segment,
        fit,
        step_s=render_step_s,
        raw_fit_metrics=raw_fit,
    )
    continuity = {
        "endpoint_position_constraint_error_m": float(fit.diagnostics.get("hard_endpoint_constraint_max_abs_error") or 0.0),
    }
    bad: list[str] = []
    if float(raw_fit["rmse_3d_m"]) > float(bad_raw_rmse_m):
        bad.append("raw_fit_rmse_high")
    if float(motion["jerk_rms_mps3"]) > float(bad_jerk_rms_mps3):
        bad.append("jerk_rms_high")
    return SegmentQualityReport(
        raw_fit_metrics=raw_fit,
        motion_metrics=motion,
        continuity_metrics=continuity,
        trajectory_model_metrics=trajectory,
        bad_reasons=tuple(bad),
    )


def verify_component_continuity(
    fits: Sequence[tuple[DynamicSegment, Any]],
    *,
    position_tolerance_m: float = 1e-6,
    velocity_tolerance_mps: float = 1e-6,
) -> dict[str, Any]:
    """Verify exact internal C0/C1 continuity for a piecewise component.

    C0 continuity means equal reconstructed position at each shared boundary;
    C1 continuity means equal reconstructed first derivative/velocity.  We do
    not require C2 continuity, so acceleration jumps are diagnostic only.
    """
    if len(fits) < 2:
        return {
            "boundary_count": 0.0,
            "position_tolerance_m": float(position_tolerance_m),
            "velocity_tolerance_mps": float(velocity_tolerance_mps),
            "position_continuity_ok": True,
            "velocity_continuity_ok": True,
            "max_position_jump_m": 0.0,
            "max_velocity_jump_mps": 0.0,
            "max_acceleration_jump_mps2": 0.0,
            "max_jerk_jump_mps3": 0.0,
            "boundaries": [],
        }
    pos_jumps = []
    vel_jumps = []
    acc_jumps = []
    jerk_jumps = []
    boundary_reports: list[dict[str, Any]] = []
    for (left_seg, left_fit), (right_seg, right_fit) in zip(fits[:-1], fits[1:]):
        # Dynamic segments share their endpoint timestamp by construction.  Use
        # the mean to avoid introducing a tiny discontinuity from representation
        # noise if one side was serialized/deserialized.
        tb_left = float(left_seg.t1)
        tb_right = float(right_seg.t0)
        tb = 0.5 * (tb_left + tb_right)
        pos_l = left_fit.evaluate([tb], deriv=0)[0]
        pos_r = right_fit.evaluate([tb], deriv=0)[0]
        vel_l = left_fit.evaluate([tb], deriv=1)[0]
        vel_r = right_fit.evaluate([tb], deriv=1)[0]
        acc_l = left_fit.evaluate([tb], deriv=2)[0]
        acc_r = right_fit.evaluate([tb], deriv=2)[0]
        # Jerk is available for the B-spline backend; keep it optional so the
        # verifier remains robust to any future fit object with fewer derivatives.
        jerk_l = None
        jerk_r = None
        jerk_jump = None
        try:
            jerk_l_arr = left_fit.evaluate([tb], deriv=3)[0]
            jerk_r_arr = right_fit.evaluate([tb], deriv=3)[0]
            jerk_l = jerk_l_arr.tolist()
            jerk_r = jerk_r_arr.tolist()
            jerk_jump = float(np.linalg.norm(jerk_l_arr - jerk_r_arr))
        except Exception:
            pass
        if jerk_jump is not None:
            jerk_jumps.append(jerk_jump)

        pos_jump = float(np.linalg.norm(pos_l - pos_r))
        vel_jump = float(np.linalg.norm(vel_l - vel_r))
        acc_jump = float(np.linalg.norm(acc_l - acc_r))
        pos_jumps.append(pos_jump)
        vel_jumps.append(vel_jump)
        acc_jumps.append(acc_jump)
        boundary_reports.append(
            {
                "left_segment_id": left_seg.segment_id,
                "right_segment_id": right_seg.segment_id,
                "boundary_time_left_s": tb_left,
                "boundary_time_right_s": tb_right,
                "boundary_time_eval_s": float(tb),
                "left_position_m": pos_l.tolist(),
                "right_position_m": pos_r.tolist(),
                "left_velocity_mps": vel_l.tolist(),
                "right_velocity_mps": vel_r.tolist(),
                "left_acceleration_mps2": acc_l.tolist(),
                "right_acceleration_mps2": acc_r.tolist(),
                "left_jerk_mps3": jerk_l,
                "right_jerk_mps3": jerk_r,
                "position_jump_m": pos_jump,
                "velocity_jump_mps": vel_jump,
                "acceleration_jump_mps2": acc_jump,
                "jerk_jump_mps3": jerk_jump,
                "position_continuity_ok": bool(pos_jump <= float(position_tolerance_m)),
                "velocity_continuity_ok": bool(vel_jump <= float(velocity_tolerance_mps)),
            }
        )
    max_pos = float(max(pos_jumps) if pos_jumps else 0.0)
    max_vel = float(max(vel_jumps) if vel_jumps else 0.0)
    return {
        "boundary_count": float(len(pos_jumps)),
        "position_tolerance_m": float(position_tolerance_m),
        "velocity_tolerance_mps": float(velocity_tolerance_mps),
        "position_continuity_ok": bool(max_pos <= float(position_tolerance_m)),
        "velocity_continuity_ok": bool(max_vel <= float(velocity_tolerance_mps)),
        "max_position_jump_m": max_pos,
        "max_velocity_jump_mps": max_vel,
        "max_acceleration_jump_mps2": float(max(acc_jumps) if acc_jumps else 0.0),
        "max_jerk_jump_mps3": float(max(jerk_jumps) if jerk_jumps else 0.0),
        "boundaries": boundary_reports,
    }


def position_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    delta = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    e3 = np.linalg.norm(delta, axis=1)
    eh = np.linalg.norm(delta[:, :2], axis=1)
    if delta.shape[1] >= 3:
        signed_vertical = delta[:, 2]
        ev = np.abs(signed_vertical)
    else:
        signed_vertical = np.zeros(delta.shape[0], dtype=float)
        ev = np.zeros(delta.shape[0], dtype=float)
    return {
        "mse_3d_m2": float(np.mean(e3**2)),
        "rmse_3d_m": float(np.sqrt(np.mean(e3**2))),
        "rmse_horizontal_m": float(np.sqrt(np.mean(eh**2))),
        "rmse_vertical_m": float(np.sqrt(np.mean(ev**2))),
        "median_error_3d_m": float(np.median(e3)),
        "p95_error_3d_m": float(np.quantile(e3, 0.95)),
        "max_error_3d_m": float(np.max(e3)),
        "median_horizontal_error_m": float(np.median(eh)),
        "p95_horizontal_error_m": float(np.quantile(eh, 0.95)),
        "max_horizontal_error_m": float(np.max(eh)),
        "median_vertical_error_m": float(np.median(ev)),
        "p95_vertical_error_m": float(np.quantile(ev, 0.95)),
        "max_vertical_error_m": float(np.max(ev)),
        "mean_signed_vertical_error_m": float(np.mean(signed_vertical)),
    }


def motion_metrics(fit: Any, *, step_s: float) -> dict[str, float]:
    t0 = float(fit.t[0])
    t1 = float(fit.t[-1])
    step = max(float(step_s), 1e-6)
    grid = np.arange(t0, t1 + step * 0.5, step)
    if grid.size == 0 or grid[-1] < t1:
        grid = np.append(grid, t1)
    acc = fit.evaluate(grid, deriv=2)
    acc_norm = np.linalg.norm(acc, axis=1)
    if grid.size >= 2:
        dt = np.diff(grid)
        jerk = np.diff(acc, axis=0) / dt[:, None]
        jerk_norm = np.linalg.norm(jerk, axis=1)
    else:
        jerk_norm = np.zeros(1, dtype=float)
    return {
        "accel_rms_mps2": float(np.sqrt(np.mean(acc_norm**2))),
        "accel_p95_mps2": float(np.quantile(acc_norm, 0.95)),
        "accel_max_mps2": float(np.max(acc_norm)),
        "jerk_rms_mps3": float(np.sqrt(np.mean(jerk_norm**2))),
        "jerk_p95_mps3": float(np.quantile(jerk_norm, 0.95)),
        "jerk_max_mps3": float(np.max(jerk_norm)),
    }


# ---------------------------------------------------------------------------
# Reference-free trajectory-model metrics
# ---------------------------------------------------------------------------


def trajectory_model_metrics(
    segment: DynamicSegment,
    fit: Any,
    *,
    step_s: float,
    raw_fit_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a reconstruction as a continuous trajectory model without truth.

    These metrics intentionally do **not** use a hidden/reference trajectory.
    Raw ADS-B positions and velocities are treated only as noisy observations.
    The score rewards properties that are valuable for aviation trajectory
    analysis: velocity evidence consistency, finite-difference kinematic
    coherence, derivative smoothness, plausible turn/acceleration behavior,
    useful dynamic detail retention, and derivative closure of the rendered
    continuous curve.
    """
    try:
        t = np.asarray([sample.t for sample in segment.samples], dtype=float)
        y = np.asarray([sample.y for sample in segment.samples], dtype=float)
        v_obs = np.asarray([sample.v for sample in segment.samples], dtype=float)
        if t.size < 2 or y.ndim != 2 or v_obs.shape != y.shape:
            return {"enabled": False, "reason": "too_few_or_invalid_samples", "truth_data_used": False}

        raw_fit = raw_fit_metrics or position_error_metrics(y, fit.evaluate(t, deriv=0))
        regime = _trajectory_regime_bucket(segment)
        speed_floor = _trajectory_speed_floor_mps(regime)
        duration_s = max(float(t[-1] - t[0]), 0.0)

        try:
            velocity_scale, velocity_confidence = compute_velocity_confidence_scale(t, y, v_obs)
        except Exception as exc:  # pragma: no cover - defensive for foreign fit inputs
            velocity_scale = np.ones(t.shape, dtype=float)
            velocity_confidence = {"enabled": False, "reason": f"confidence_scaling_failed:{type(exc).__name__}"}

        v_model_obs = np.asarray(fit.evaluate(t, deriv=1), dtype=float)
        velocity_evidence = _velocity_evidence_metrics(v_obs, v_model_obs, velocity_scale)

        finite_difference = _finite_difference_kinematic_metrics(t, y, fit, velocity_scale)
        smoothness = _smoothness_metrics_from_fit(fit, t0=float(t[0]), t1=float(t[-1]), step_s=step_s, speed_floor_mps=speed_floor)
        plausibility = _physical_plausibility_metrics(smoothness, regime=regime)
        detail = _dynamic_detail_metrics(finite_difference)
        closure = _derivative_closure_metrics(fit, t0=float(t[0]), t1=float(t[-1]), step_s=step_s)

        component_scores: dict[str, float | None] = {
            "observation_position_score": _score_low(float(raw_fit.get("rmse_3d_m", 0.0)), soft=25.0, hard=180.0),
            "velocity_evidence_score": velocity_evidence.get("score_0_100"),
            "finite_difference_kinematics_score": finite_difference.get("score_0_100"),
            "trajectory_smoothness_score": smoothness.get("score_0_100"),
            "physical_plausibility_score": plausibility.get("score_0_100"),
            "dynamic_detail_preservation_score": detail.get("score_0_100"),
            "derivative_closure_score": closure.get("score_0_100"),
        }
        weights = {
            "observation_position_score": 0.12,
            "velocity_evidence_score": 0.18,
            "finite_difference_kinematics_score": 0.16,
            "trajectory_smoothness_score": 0.22,
            "physical_plausibility_score": 0.14,
            "dynamic_detail_preservation_score": 0.10,
            "derivative_closure_score": 0.08,
        }
        weighted_score = _weighted_score(component_scores, weights)

        return {
            "enabled": True,
            "metric_family": "reference_free_trajectory_model_metrics_v1",
            "truth_data_used": False,
            "truth_data_note": "raw ADS-B is used only as noisy observation/proxy evidence; no hidden true trajectory or external reference is used",
            "scientific_intent": (
                "reward continuous local trajectory models useful for velocity, wind-impact, pilot-input, "
                "curvature, acceleration, jerk, and event-aware analysis; this is deliberately not a pure raw-point overfit score"
            ),
            "regime_bucket": regime,
            "duration_s": duration_s,
            "n_observations": int(t.size),
            "speed_floor_mps": float(speed_floor),
            "raw_observation_fit": {
                "rmse_3d_m": float(raw_fit.get("rmse_3d_m", 0.0)),
                "p95_error_3d_m": float(raw_fit.get("p95_error_3d_m", 0.0)),
                "score_0_100": component_scores["observation_position_score"],
                "note": "ADS-B observations are noisy; this score has low weight so smoothing is not punished as if raw data were truth",
            },
            "velocity_evidence": velocity_evidence,
            "velocity_confidence_scaling": velocity_confidence,
            "finite_difference_kinematics": finite_difference,
            "smoothness": smoothness,
            "physical_plausibility": plausibility,
            "dynamic_detail_preservation": detail,
            "derivative_closure": closure,
            "component_scores_0_100": component_scores,
            "weights": weights,
            "weighted_score_0_100": weighted_score,
            "interpretation": _score_interpretation(weighted_score),
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostics should not break production renders
        return {
            "enabled": False,
            "reason": "trajectory_model_metric_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "truth_data_used": False,
        }


def aggregate_trajectory_model_metrics(
    segment_entries: Sequence[dict[str, Any]],
    *,
    piecewise_global_quality: dict[str, Any] | None = None,
    raw_times: Sequence[float] | None = None,
    render_times: Sequence[float] | None = None,
    fit_mode: str | None = None,
    reconstruction_backend: str | None = None,
) -> dict[str, Any]:
    """Aggregate segment-level reference-free trajectory-model metrics.

    The returned weighted score intentionally combines local segment quality with
    event-aware join behavior and hard-gap honesty.  This favors methods that do
    not invent smooth continuity where the data contain a real surveillance gap
    or discontinuity, which is a central claim of segmented aviation V-Splines.
    """
    rows: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
    for entry in segment_entries:
        if not isinstance(entry, dict):
            continue
        q = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
        metrics = q.get("trajectory_model_metrics") if isinstance(q.get("trajectory_model_metrics"), dict) else None
        if not metrics or not metrics.get("enabled"):
            continue
        weight = _segment_entry_weight(entry, metrics)
        rows.append((entry, weight, metrics))

    component_keys = [
        "observation_position_score",
        "velocity_evidence_score",
        "finite_difference_kinematics_score",
        "trajectory_smoothness_score",
        "physical_plausibility_score",
        "dynamic_detail_preservation_score",
        "derivative_closure_score",
    ]
    component_means = {
        key: _weighted_mean([float(m["component_scores_0_100"][key]) for _, _, m in rows if _is_number(m.get("component_scores_0_100", {}).get(key))],
                            [w for _, w, m in rows if _is_number(m.get("component_scores_0_100", {}).get(key))])
        for key in component_keys
    }
    segment_weighted_score = _weighted_mean(
        [float(m.get("weighted_score_0_100")) for _, _, m in rows if _is_number(m.get("weighted_score_0_100"))],
        [w for _, w, m in rows if _is_number(m.get("weighted_score_0_100"))],
    )

    by_regime: dict[str, dict[str, Any]] = {}
    for regime in sorted({str(m.get("regime_bucket") or "unknown") for _, _, m in rows} | {"ground", "airborne", "approach_final"}):
        selected = [(w, m) for _, w, m in rows if str(m.get("regime_bucket") or "unknown") == regime]
        by_regime[regime] = {
            "segment_count": int(len(selected)),
            "weighted_score_0_100": _weighted_mean(
                [float(m.get("weighted_score_0_100")) for w, m in selected if _is_number(m.get("weighted_score_0_100"))],
                [w for w, m in selected if _is_number(m.get("weighted_score_0_100"))],
            ),
        }

    join_score, join_details = _event_aware_join_score(piecewise_global_quality or {})
    gap_score, gap_details = _hard_gap_honesty_score(raw_times, render_times)
    scope_score = _locality_scope_score(fit_mode=fit_mode, reconstruction_backend=reconstruction_backend, gap_details=gap_details)

    # Segment quality dominates, but the global score also rewards event-aware
    # joins, not bridging hard surveillance gaps, and being a local model when
    # the research question is local flight dynamics rather than whole-track
    # state smoothing.
    method_scores = {
        "segment_trajectory_score": segment_weighted_score,
        "event_aware_join_score": join_score,
        "hard_gap_honesty_score": gap_score,
        "locality_scope_score": scope_score,
    }
    method_weights = {
        "segment_trajectory_score": 0.72,
        "event_aware_join_score": 0.12,
        "hard_gap_honesty_score": 0.10,
        "locality_scope_score": 0.06,
    }
    weighted = _weighted_score(method_scores, method_weights)

    return {
        "enabled": bool(rows),
        "metric_family": "reference_free_trajectory_model_metrics_v1",
        "truth_data_used": False,
        "truth_data_note": "all metrics are computed from noisy ADS-B observations, fitted derivatives, render timing, and declared event boundaries; no true/reference trajectory is used",
        "scientific_intent": (
            "measure whether the output is a useful aviation trajectory model for derivative analysis, "
            "wind-impact hypotheses, pilot/autopilot input signatures, and event-aware continuity"
        ),
        "segment_count": int(len(rows)),
        "segment_weighted_score_0_100": segment_weighted_score,
        "component_mean_scores_0_100": component_means,
        "by_regime": by_regime,
        "event_aware_join_score": join_details,
        "hard_gap_honesty": gap_details,
        "locality_scope": {
            "score_0_100": scope_score,
            "fit_mode": fit_mode,
            "reconstruction_backend": reconstruction_backend,
            "note": "local segmented models score higher for local wind/control analysis; whole-track smoothers remain valid baselines but are less local by design",
        },
        "method_component_scores_0_100": method_scores,
        "method_score_weights": method_weights,
        "weighted_score_0_100": weighted,
        "interpretation": _score_interpretation(weighted),
    }


def _velocity_evidence_metrics(v_obs: np.ndarray, v_model: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(v_model, dtype=float) - np.asarray(v_obs, dtype=float)
    err = np.linalg.norm(delta, axis=1)
    eh = np.linalg.norm(delta[:, :2], axis=1)
    ev = np.abs(delta[:, 2]) if delta.shape[1] >= 3 else np.zeros(delta.shape[0], dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    reliable = w > 0.05
    reliable_count = int(np.sum(reliable))
    if reliable_count < 2 or float(np.sum(w[reliable])) <= 1e-9:
        return {
            "enabled": False,
            "reason": "too_few_reliable_velocity_observations",
            "reliable_observation_count": reliable_count,
            "score_0_100": None,
        }
    rmse = _weighted_rms(err, w)
    rmse_h = _weighted_rms(eh, w)
    rmse_v = _weighted_rms(ev, w)
    p95 = _weighted_quantile(err, w, 0.95)
    score = _weighted_score(
        {
            "velocity_rmse": _score_low(rmse, soft=10.0, hard=55.0),
            "velocity_p95": _score_low(p95, soft=25.0, hard=95.0),
            "vertical_velocity_rmse": _score_low(rmse_v, soft=3.0, hard=18.0),
        },
        {"velocity_rmse": 0.55, "velocity_p95": 0.25, "vertical_velocity_rmse": 0.20},
    )
    return {
        "enabled": True,
        "reliable_observation_count": reliable_count,
        "mean_velocity_confidence_scale": float(np.mean(w)),
        "weighted_rmse_3d_mps": rmse,
        "weighted_rmse_horizontal_mps": rmse_h,
        "weighted_rmse_vertical_mps": rmse_v,
        "weighted_p95_error_3d_mps": p95,
        "score_0_100": score,
        "note": "compares fitted derivative velocity to confidence-weighted ADS-B velocity observations; velocity is observational evidence, not truth",
    }


def _finite_difference_kinematic_metrics(t: np.ndarray, y: np.ndarray, fit: Any, velocity_scale: np.ndarray) -> dict[str, Any]:
    dt = np.diff(t)
    valid = dt > 1e-6
    if not np.any(valid):
        return {"enabled": False, "reason": "no_positive_intervals", "score_0_100": None}
    median_dt = float(np.median(dt[valid])) if np.any(valid) else 0.0
    long_gap_threshold = max(30.0, 5.0 * median_dt) if median_dt > 0.0 else 30.0
    normal = valid & (dt <= long_gap_threshold)
    if not np.any(normal):
        return {
            "enabled": False,
            "reason": "only_long_gap_intervals",
            "excluded_long_gap_interval_count": int(np.sum(valid)),
            "score_0_100": None,
        }
    fd_v_all = np.diff(y, axis=0) / np.maximum(dt, 1e-6)[:, None]
    mid_all = 0.5 * (t[:-1] + t[1:])
    model_mid_all = np.asarray(fit.evaluate(mid_all, deriv=1), dtype=float)
    interval_w_all = np.sqrt(np.maximum(velocity_scale[:-1], 0.0) * np.maximum(velocity_scale[1:], 0.0))
    interval_w_all = np.maximum(interval_w_all, 0.10)  # finite differences still carry position evidence

    fd_v = fd_v_all[normal]
    model_mid = model_mid_all[normal]
    interval_w = interval_w_all[normal]
    diff = model_mid - fd_v
    err = np.linalg.norm(diff, axis=1)
    eh = np.linalg.norm(diff[:, :2], axis=1)
    ev = np.abs(diff[:, 2]) if diff.shape[1] >= 3 else np.zeros(diff.shape[0], dtype=float)

    raw_h_speed = np.linalg.norm(fd_v[:, :2], axis=1)
    model_h_speed = np.linalg.norm(model_mid[:, :2], axis=1)
    direction_mask = (raw_h_speed > 5.0) & (model_h_speed > 5.0)
    heading_errors = np.array([], dtype=float)
    if np.any(direction_mask):
        raw_xy = fd_v[direction_mask, :2]
        model_xy = model_mid[direction_mask, :2]
        dots = np.sum(raw_xy * model_xy, axis=1)
        cross = raw_xy[:, 0] * model_xy[:, 1] - raw_xy[:, 1] * model_xy[:, 0]
        heading_errors = np.abs(np.degrees(np.arctan2(cross, dots)))

    fd_rmse = _weighted_rms(err, interval_w)
    fd_h_rmse = _weighted_rms(eh, interval_w)
    fd_v_rmse = _weighted_rms(ev, interval_w)
    heading_med = float(np.median(heading_errors)) if heading_errors.size else None
    heading_p95 = float(np.quantile(heading_errors, 0.95)) if heading_errors.size else None

    score = _weighted_score(
        {
            "fd_velocity_rmse": _score_low(fd_rmse, soft=12.0, hard=70.0),
            "fd_horizontal_velocity_rmse": _score_low(fd_h_rmse, soft=10.0, hard=65.0),
            "fd_vertical_rate_rmse": _score_low(fd_v_rmse, soft=3.0, hard=20.0),
            "track_angle_median": None if heading_med is None else _score_low(heading_med, soft=5.0, hard=45.0),
        },
        {"fd_velocity_rmse": 0.38, "fd_horizontal_velocity_rmse": 0.26, "fd_vertical_rate_rmse": 0.18, "track_angle_median": 0.18},
    )

    return {
        "enabled": True,
        "interval_count": int(fd_v.shape[0]),
        "excluded_long_gap_interval_count": int(np.sum(valid & ~normal)),
        "long_gap_threshold_s": float(long_gap_threshold),
        "finite_difference_velocity_rmse_3d_mps": fd_rmse,
        "finite_difference_velocity_rmse_horizontal_mps": fd_h_rmse,
        "finite_difference_vertical_rate_rmse_mps": fd_v_rmse,
        "track_angle_median_error_deg": heading_med,
        "track_angle_p95_error_deg": heading_p95,
        "raw_fd_velocity_variability_rms_mps": _velocity_variability(fd_v, interval_w),
        "model_mid_velocity_variability_rms_mps": _velocity_variability(model_mid, interval_w),
        "score_0_100": score,
        "note": "compares fitted velocity at interval midpoints to finite-difference ADS-B position velocity; long gaps are excluded because continuity is not claimed there",
    }


def _smoothness_metrics_from_fit(fit: Any, *, t0: float, t1: float, step_s: float, speed_floor_mps: float) -> dict[str, Any]:
    step = max(float(step_s), 1e-6)
    grid = np.arange(float(t0), float(t1) + step * 0.5, step)
    if grid.size == 0 or grid[-1] < t1:
        grid = np.append(grid, t1)
    if grid.size < 2:
        return {"enabled": False, "reason": "too_short_grid", "score_0_100": None}

    pos = np.asarray(fit.evaluate(grid, deriv=0), dtype=float)
    vel = np.asarray(fit.evaluate(grid, deriv=1), dtype=float)
    acc = np.asarray(fit.evaluate(grid, deriv=2), dtype=float)
    try:
        jerk = np.asarray(fit.evaluate(grid, deriv=3), dtype=float)
        jerk_times = grid
        jerk_source = "analytic_fit_derivative"
    except Exception:
        dg = np.diff(grid)
        jerk = np.diff(acc, axis=0) / np.maximum(dg, 1e-6)[:, None]
        jerk_times = 0.5 * (grid[:-1] + grid[1:])
        jerk_source = "finite_difference_of_acceleration"

    speed_h = np.linalg.norm(vel[:, :2], axis=1)
    speed_3d = np.linalg.norm(vel, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1) if jerk.size else np.zeros(0, dtype=float)
    jerk_speed = np.interp(jerk_times, grid, np.maximum(speed_h, float(speed_floor_mps))) if jerk_norm.size else np.zeros(0, dtype=float)
    normalized_jerk = jerk_norm / np.maximum(jerk_speed, 1e-6) if jerk_norm.size else np.zeros(0, dtype=float)

    if jerk.shape[0] >= 2:
        jdt = np.diff(jerk_times)
        snap = np.diff(jerk, axis=0) / np.maximum(jdt, 1e-6)[:, None]
        snap_norm = np.linalg.norm(snap, axis=1)
    else:
        snap_norm = np.zeros(0, dtype=float)

    # Horizontal turn geometry from v and a.  This is useful for aircraft-control
    # analysis because curvature/turn-rate spikes often indicate surveillance
    # artifacts or overfitting, while smooth sustained curvature suggests a real
    # maneuver.
    vh = vel[:, :2]
    ah = acc[:, :2]
    speed_h_safe = np.maximum(np.linalg.norm(vh, axis=1), float(speed_floor_mps))
    cross = vh[:, 0] * ah[:, 1] - vh[:, 1] * ah[:, 0]
    turn_rate_rad_s = np.abs(cross) / np.maximum(speed_h_safe**2, 1e-9)
    curvature_1_m = np.abs(cross) / np.maximum(speed_h_safe**3, 1e-9)
    vertical_acc = np.abs(acc[:, 2]) if acc.shape[1] >= 3 else np.zeros(acc.shape[0], dtype=float)

    oscillation_index = _acceleration_oscillation_index(acc, duration_s=max(float(t1 - t0), 0.0))

    score = _weighted_score(
        {
            "normalized_jerk_rms": _score_low(_rms(normalized_jerk), soft=0.08, hard=0.80),
            "jerk_p95": _score_low(_q(jerk_norm, 0.95), soft=8.0, hard=55.0),
            "snap_proxy_rms": _score_low(_rms(snap_norm), soft=4.0, hard=80.0),
            "oscillation_index": _score_low(oscillation_index, soft=3.0, hard=30.0),
        },
        {"normalized_jerk_rms": 0.42, "jerk_p95": 0.26, "snap_proxy_rms": 0.20, "oscillation_index": 0.12},
    )

    return {
        "enabled": True,
        "sample_count": int(grid.size),
        "jerk_source": jerk_source,
        "speed_floor_mps": float(speed_floor_mps),
        "speed_horizontal_mps_median": _q(speed_h, 0.50),
        "speed_3d_mps_median": _q(speed_3d, 0.50),
        "accel_rms_mps2": _rms(acc_norm),
        "accel_p95_mps2": _q(acc_norm, 0.95),
        "accel_max_mps2": _max(acc_norm),
        "vertical_accel_p95_mps2": _q(vertical_acc, 0.95),
        "jerk_rms_mps3": _rms(jerk_norm),
        "jerk_p95_mps3": _q(jerk_norm, 0.95),
        "jerk_max_mps3": _max(jerk_norm),
        "jerk_per_speed_rms_1_s2": _rms(normalized_jerk),
        "jerk_per_speed_p95_1_s2": _q(normalized_jerk, 0.95),
        "snap_proxy_rms_mps4": _rms(snap_norm),
        "snap_proxy_p95_mps4": _q(snap_norm, 0.95),
        "turn_rate_p95_deg_s": None if turn_rate_rad_s.size == 0 else float(np.degrees(np.quantile(turn_rate_rad_s, 0.95))),
        "turn_rate_max_deg_s": None if turn_rate_rad_s.size == 0 else float(np.degrees(np.max(turn_rate_rad_s))),
        "curvature_p95_1_per_m": _q(curvature_1_m, 0.95),
        "acceleration_direction_changes_per_min": oscillation_index,
        "score_0_100": score,
        "note": "derivative smoothness is scored in speed-normalized units so low-speed duplicates do not dominate aviation intervals"
    }


def _physical_plausibility_metrics(smoothness: dict[str, Any], *, regime: str) -> dict[str, Any]:
    if not smoothness.get("enabled"):
        return {"enabled": False, "reason": "smoothness_metrics_disabled", "score_0_100": None}
    acc_p95 = smoothness.get("accel_p95_mps2")
    turn_p95 = smoothness.get("turn_rate_p95_deg_s")
    vertical_acc_p95 = smoothness.get("vertical_accel_p95_mps2")
    if regime == "ground":
        acc_soft, acc_hard = 3.5, 9.0
        turn_soft, turn_hard = 10.0, 35.0
        vacc_soft, vacc_hard = 1.0, 5.0
    elif regime == "approach_final":
        acc_soft, acc_hard = 5.0, 16.0
        turn_soft, turn_hard = 4.0, 14.0
        vacc_soft, vacc_hard = 2.5, 10.0
    else:
        acc_soft, acc_hard = 7.0, 22.0
        turn_soft, turn_hard = 5.0, 18.0
        vacc_soft, vacc_hard = 3.0, 12.0
    score = _weighted_score(
        {
            "acceleration_p95": _score_low(acc_p95, soft=acc_soft, hard=acc_hard),
            "turn_rate_p95": _score_low(turn_p95, soft=turn_soft, hard=turn_hard),
            "vertical_acceleration_p95": _score_low(vertical_acc_p95, soft=vacc_soft, hard=vacc_hard),
        },
        {"acceleration_p95": 0.45, "turn_rate_p95": 0.35, "vertical_acceleration_p95": 0.20},
    )
    return {
        "enabled": True,
        "regime_bucket": regime,
        "accel_p95_mps2": acc_p95,
        "turn_rate_p95_deg_s": turn_p95,
        "vertical_accel_p95_mps2": vertical_acc_p95,
        "score_0_100": score,
        "note": "broad plausibility gate; it is not an aircraft-performance certification model",
    }


def _dynamic_detail_metrics(finite_difference: dict[str, Any]) -> dict[str, Any]:
    if not finite_difference.get("enabled"):
        return {"enabled": False, "reason": "finite_difference_metrics_disabled", "score_0_100": None}
    raw_var = finite_difference.get("raw_fd_velocity_variability_rms_mps")
    model_var = finite_difference.get("model_mid_velocity_variability_rms_mps")
    if not _is_number(raw_var) or float(raw_var) <= 1e-6 or not _is_number(model_var):
        return {
            "enabled": False,
            "reason": "insufficient_velocity_variability",
            "raw_fd_velocity_variability_rms_mps": raw_var,
            "model_mid_velocity_variability_rms_mps": model_var,
            "score_0_100": None,
        }
    ratio = float(model_var) / max(float(raw_var), 1e-6)
    score = _score_band(ratio, ideal_low=0.35, ideal_high=1.20, bad_low=0.08, bad_high=2.50)
    return {
        "enabled": True,
        "velocity_detail_retention_ratio": ratio,
        "raw_fd_velocity_variability_rms_mps": raw_var,
        "model_mid_velocity_variability_rms_mps": model_var,
        "score_0_100": score,
        "note": "ratio near 1 preserves local velocity structure; very low suggests over-smoothing, very high suggests noise chasing",
    }


def _derivative_closure_metrics(fit: Any, *, t0: float, t1: float, step_s: float) -> dict[str, Any]:
    step = max(float(step_s), 1e-6)
    grid = np.arange(float(t0), float(t1) + step * 0.5, step)
    if grid.size == 0 or grid[-1] < t1:
        grid = np.append(grid, t1)
    if grid.size < 3:
        return {"enabled": False, "reason": "too_short_grid", "score_0_100": None}
    pos = np.asarray(fit.evaluate(grid, deriv=0), dtype=float)
    vel = np.asarray(fit.evaluate(grid, deriv=1), dtype=float)
    acc = np.asarray(fit.evaluate(grid, deriv=2), dtype=float)
    dt = np.diff(grid)
    pos_fd = np.diff(pos, axis=0) / np.maximum(dt, 1e-6)[:, None]
    vel_mid = 0.5 * (vel[:-1] + vel[1:])
    vel_fd = np.diff(vel, axis=0) / np.maximum(dt, 1e-6)[:, None]
    acc_mid = 0.5 * (acc[:-1] + acc[1:])
    pos_vel_err = np.linalg.norm(pos_fd - vel_mid, axis=1)
    vel_acc_err = np.linalg.norm(vel_fd - acc_mid, axis=1)
    pos_vel_rms = _rms(pos_vel_err)
    vel_acc_rms = _rms(vel_acc_err)
    score = _weighted_score(
        {
            "position_velocity_closure": _score_low(pos_vel_rms, soft=0.25, hard=5.0),
            "velocity_acceleration_closure": _score_low(vel_acc_rms, soft=0.10, hard=3.0),
        },
        {"position_velocity_closure": 0.55, "velocity_acceleration_closure": 0.45},
    )
    return {
        "enabled": True,
        "position_velocity_closure_rms_mps": pos_vel_rms,
        "velocity_acceleration_closure_rms_mps2": vel_acc_rms,
        "score_0_100": score,
        "note": "checks that rendered position, velocity, and acceleration behave as one differentiable trajectory rather than disconnected channels",
    }


def _event_aware_join_score(piecewise_global_quality: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    event = piecewise_global_quality.get("event_aware_continuity") if isinstance(piecewise_global_quality, dict) else None
    if not isinstance(event, dict) or not event.get("enabled"):
        return 100.0, {
            "enabled": False,
            "reason": "no_event_aware_join_report",
            "score_0_100": 100.0,
            "note": "single-segment or non-segmented method has no ordinary segment joins to score",
        }
    count = int(event.get("normal_join_count") or 0)
    if count <= 0:
        return 100.0, {
            "enabled": True,
            "normal_join_count": 0,
            "score_0_100": 100.0,
            "note": "no ordinary joins; event boundaries are intentionally excluded from normal continuity scoring",
        }
    acc_jump = float(event.get("normal_max_acceleration_jump_mps2") or 0.0)
    jerk_jump = float(event.get("normal_max_jerk_jump_mps3") or 0.0)
    score = _weighted_score(
        {
            "normal_acceleration_jump": _score_low(acc_jump, soft=1.0, hard=10.0),
            "normal_jerk_jump": _score_low(jerk_jump, soft=6.0, hard=80.0),
        },
        {"normal_acceleration_jump": 0.58, "normal_jerk_jump": 0.42},
    )
    return score, {
        "enabled": True,
        "normal_join_count": count,
        "event_boundary_count": int(event.get("event_boundary_count") or 0),
        "normal_max_acceleration_jump_mps2": acc_jump,
        "normal_max_jerk_jump_mps3": jerk_jump,
        "score_0_100": score,
        "note": "scores only normal segment joins; hard gaps, go-arounds, and true discontinuities are reported but not treated as continuity failures",
    }


def _hard_gap_honesty_score(raw_times: Sequence[float] | None, render_times: Sequence[float] | None) -> tuple[float | None, dict[str, Any]]:
    if raw_times is None or render_times is None:
        return None, {"enabled": False, "reason": "times_not_provided", "score_0_100": None}
    raw = np.asarray([float(x) for x in raw_times if _is_number(x)], dtype=float)
    render = np.asarray([float(x) for x in render_times if _is_number(x)], dtype=float)
    if raw.size < 2 or render.size == 0:
        return None, {"enabled": False, "reason": "insufficient_times", "score_0_100": None}
    raw.sort()
    render.sort()
    dt = np.diff(raw)
    positive = dt > 1e-6
    if not np.any(positive):
        return None, {"enabled": False, "reason": "no_positive_raw_intervals", "score_0_100": None}
    median_dt = float(np.median(dt[positive]))
    threshold = max(30.0, 5.0 * median_dt)
    gap_indices = np.where(positive & (dt > threshold))[0]
    if gap_indices.size == 0:
        return 100.0, {
            "enabled": True,
            "hard_gap_count": 0,
            "hard_gap_threshold_s": threshold,
            "bridged_gap_count": 0,
            "bridge_frame_count": 0,
            "score_0_100": 100.0,
            "note": "no hard surveillance gaps detected from raw timing",
        }
    bridged = 0
    bridge_frames = 0
    gap_reports = []
    for idx in gap_indices:
        left = float(raw[idx])
        right = float(raw[idx + 1])
        # A method is considered to bridge the hard gap if it renders interior
        # samples well away from the two observed endpoints.
        margin = min(max(median_dt, 0.25), max((right - left) * 0.25, 0.25))
        inside = (render > left + margin) & (render < right - margin)
        count = int(np.sum(inside))
        if count > 0:
            bridged += 1
            bridge_frames += count
        gap_reports.append({"t_left_s": left, "t_right_s": right, "gap_s": right - left, "interior_render_frame_count": count})
    bridge_fraction = bridged / max(int(gap_indices.size), 1)
    score = _score_low(bridge_fraction, soft=0.0, hard=1.0)
    return score, {
        "enabled": True,
        "hard_gap_count": int(gap_indices.size),
        "hard_gap_threshold_s": threshold,
        "bridged_gap_count": int(bridged),
        "bridge_frame_count": int(bridge_frames),
        "bridge_fraction": float(bridge_fraction),
        "score_0_100": score,
        "gaps": gap_reports,
        "note": "penalizes methods that render a continuous trajectory through raw timing gaps where continuity is not claimed",
    }


def _locality_scope_score(*, fit_mode: str | None, reconstruction_backend: str | None, gap_details: dict[str, Any]) -> float:
    text = f"{fit_mode or ''} {reconstruction_backend or ''}".lower()
    if "kalman" in text and "whole" in text:
        base = 25.0
    elif "whole_track" in text or "global" in text:
        base = 40.0
    elif "segmented" in text or "local" in text or "piecewise" in text:
        base = 100.0
    else:
        base = 75.0
    if gap_details.get("enabled") and int(gap_details.get("hard_gap_count") or 0) > 0:
        gap_score = gap_details.get("score_0_100")
        if _is_number(gap_score):
            base = min(base, 0.65 * base + 0.35 * float(gap_score))
    return float(np.clip(base, 0.0, 100.0))


def _trajectory_regime_bucket(segment: DynamicSegment) -> str:
    label = str(getattr(segment, "regime_label", "") or "").lower()
    features = getattr(segment, "features", {}) or {}
    speed = 0.0
    if isinstance(features, dict):
        try:
            speed = float(features.get("median_horizontal_speed_mps", 0.0) or 0.0)
        except Exception:
            speed = 0.0
    if "ground" in label or speed < 20.0:
        return "ground"
    if "descent" in label or "approach" in label or "final" in label:
        return "approach_final"
    return "airborne"


def _trajectory_speed_floor_mps(regime: str) -> float:
    if regime == "ground":
        return 5.0
    if regime == "approach_final":
        return 45.0
    if regime == "airborne":
        return 35.0
    return 30.0


def _velocity_variability(v: np.ndarray, weights: np.ndarray) -> float | None:
    arr = np.asarray(v, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.all(np.isfinite(arr), axis=1) & np.isfinite(w) & (w > 0)
    if np.sum(mask) < 2:
        return None
    arr = arr[mask]
    w = w[mask]
    mean = np.sum(arr * w[:, None], axis=0) / max(float(np.sum(w)), 1e-12)
    dev = np.linalg.norm(arr - mean, axis=1)
    return _weighted_rms(dev, w)


def _acceleration_oscillation_index(acc: np.ndarray, *, duration_s: float) -> float | None:
    arr = np.asarray(acc, dtype=float)
    if arr.shape[0] < 3 or duration_s <= 0:
        return 0.0
    a0 = arr[:-1]
    a1 = arr[1:]
    n0 = np.linalg.norm(a0, axis=1)
    n1 = np.linalg.norm(a1, axis=1)
    valid = (n0 > 1e-6) & (n1 > 1e-6)
    if not np.any(valid):
        return 0.0
    cos = np.sum(a0[valid] * a1[valid], axis=1) / np.maximum(n0[valid] * n1[valid], 1e-9)
    changes = int(np.sum(cos < -0.25))
    return float(changes / max(duration_s, 1e-6) * 60.0)


def _segment_entry_weight(entry: dict[str, Any], metrics: dict[str, Any]) -> float:
    try:
        if entry.get("t0") is not None and entry.get("t1") is not None:
            duration = max(float(entry.get("t1")) - float(entry.get("t0")), 0.0)
            if duration > 0:
                return duration
    except Exception:
        pass
    try:
        duration = float(metrics.get("duration_s") or 0.0)
        if duration > 0:
            return duration
    except Exception:
        pass
    try:
        return max(float(entry.get("n_observations") or metrics.get("n_observations") or 1.0), 1.0)
    except Exception:
        return 1.0


def _score_low(value: float | None, *, soft: float, hard: float) -> float | None:
    if not _is_number(value):
        return None
    v = float(value)
    if v <= soft:
        return 100.0
    if v >= hard:
        return 0.0
    return float(100.0 * (hard - v) / max(hard - soft, 1e-12))


def _score_band(value: float | None, *, ideal_low: float, ideal_high: float, bad_low: float, bad_high: float) -> float | None:
    if not _is_number(value):
        return None
    v = float(value)
    if ideal_low <= v <= ideal_high:
        return 100.0
    if v < ideal_low:
        if v <= bad_low:
            return 0.0
        return float(100.0 * (v - bad_low) / max(ideal_low - bad_low, 1e-12))
    if v >= bad_high:
        return 0.0
    return float(100.0 * (bad_high - v) / max(bad_high - ideal_high, 1e-12))


def _weighted_score(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    total = 0.0
    denom = 0.0
    for key, weight in weights.items():
        value = scores.get(key)
        if not _is_number(value):
            continue
        w = max(float(weight), 0.0)
        total += w * float(value)
        denom += w
    if denom <= 0.0:
        return None
    return float(np.clip(total / denom, 0.0, 100.0))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    if len(values) == 0 or len(weights) == 0:
        return None
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return None
    return float(np.sum(v[mask] * w[mask]) / max(float(np.sum(w[mask])), 1e-12))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return None
    return float(np.sqrt(np.sum(w[mask] * v[mask] * v[mask]) / max(float(np.sum(w[mask])), 1e-12)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return None
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / max(float(np.sum(w)), 1e-12)
    return float(v[min(int(np.searchsorted(cdf, float(q), side="left")), v.size - 1)])


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _rms(values: np.ndarray | Sequence[float] | None) -> float | None:
    if values is None:
        return None
    arr = _finite(np.asarray(values, dtype=float))
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr * arr)))


def _q(values: np.ndarray | Sequence[float] | None, q: float) -> float | None:
    if values is None:
        return None
    arr = _finite(np.asarray(values, dtype=float))
    if arr.size == 0:
        return None
    return float(np.quantile(arr, float(q)))


def _max(values: np.ndarray | Sequence[float] | None) -> float | None:
    if values is None:
        return None
    arr = _finite(np.asarray(values, dtype=float))
    if arr.size == 0:
        return None
    return float(np.max(arr))


def _is_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _score_interpretation(score: float | None) -> str:
    if score is None:
        return "not_available"
    if score >= 85.0:
        return "excellent_trajectory_model_for_reference_free_derivative_analysis"
    if score >= 70.0:
        return "good_trajectory_model_with_minor_derivative_or_observation_tradeoffs"
    if score >= 50.0:
        return "usable_but_some_velocity_smoothness_or_event_behavior_needs_review"
    return "weak_trajectory_model_for_derivative_analysis_under_these_observations"
