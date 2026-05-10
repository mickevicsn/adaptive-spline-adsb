"""Kalman/RTS smoothing helpers used only for segmentation features.

The final V-Spline fit still uses the prepared raw paired observations.  This
module only builds a less noisy signal for detecting sustained energy-state
changes and for assigning segment regime labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from raw_keyframe_vspline_adapter import PreparedVSplineSample


@dataclass(frozen=True)
class KalmanSegmentationConfig:
    """3D constant-velocity Kalman/RTS settings for segmentation signals."""

    enabled: bool = True
    meas_std_xy_m: float = 25.0
    meas_std_z_m: float = 40.0
    accel_std_xy_mps2: float = 8.0
    accel_std_z_mps2: float = 4.0
    gate_sigma: float = 4.5
    init_vel_points: int = 5
    min_observations: int = 4
    prefer_reported_velocity: bool = True
    reported_velocity_smoothing_window: int = 5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def smooth_samples_for_segmentation(
    samples: Sequence[PreparedVSplineSample],
    config: KalmanSegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(position_signal, velocity_signal, diagnostics)``.

    Positions/velocities are local ENU meters and meters/second.  The smoother is
    deliberately fitted independently inside a hard-gap connected component, so
    it does not invent continuity across real surveillance holes.
    """
    y_raw = np.asarray([s.y for s in samples], dtype=float)
    v_raw = np.asarray([s.v for s in samples], dtype=float)
    t = np.asarray([s.t for s in samples], dtype=float)
    n = len(samples)

    if not config.enabled or n < int(config.min_observations):
        return y_raw, v_raw, {
            "enabled": bool(config.enabled),
            "used": False,
            "reason": "disabled" if not config.enabled else "too_few_observations",
            "n_observations": int(n),
        }

    try:
        x_smooth, _ = _kalman_rts_constant_velocity_3d(
            t=t,
            y_meas=y_raw,
            meas_std_xy_m=float(config.meas_std_xy_m),
            meas_std_z_m=float(config.meas_std_z_m),
            accel_std_xy_mps2=float(config.accel_std_xy_mps2),
            accel_std_z_mps2=float(config.accel_std_z_mps2),
            gate_sigma=float(config.gate_sigma),
            init_vel_points=int(config.init_vel_points),
        )
        y_signal = x_smooth[:, :3]
        if bool(config.prefer_reported_velocity):
            v_signal = _rolling_median_2d(v_raw, int(config.reported_velocity_smoothing_window))
        else:
            v_signal = x_smooth[:, 3:6]
        pos_delta = np.linalg.norm(y_signal - y_raw, axis=1)
        vel_delta = np.linalg.norm(v_signal - v_raw, axis=1)
        return y_signal, v_signal, {
            "enabled": True,
            "used": True,
            "model": "constant_velocity_3d_rts",
            "n_observations": int(n),
            "meas_std_xy_m": float(config.meas_std_xy_m),
            "meas_std_z_m": float(config.meas_std_z_m),
            "accel_std_xy_mps2": float(config.accel_std_xy_mps2),
            "accel_std_z_mps2": float(config.accel_std_z_mps2),
            "gate_sigma": float(config.gate_sigma),
            "prefer_reported_velocity": bool(config.prefer_reported_velocity),
            "reported_velocity_smoothing_window": int(config.reported_velocity_smoothing_window),
            "median_position_adjustment_m": float(np.median(pos_delta)) if pos_delta.size else 0.0,
            "p95_position_adjustment_m": float(np.quantile(pos_delta, 0.95)) if pos_delta.size else 0.0,
            "median_velocity_adjustment_mps": float(np.median(vel_delta)) if vel_delta.size else 0.0,
            "p95_velocity_adjustment_mps": float(np.quantile(vel_delta, 0.95)) if vel_delta.size else 0.0,
        }
    except Exception as exc:
        return y_raw, v_raw, {
            "enabled": True,
            "used": False,
            "reason": f"fallback_to_raw_after_{type(exc).__name__}",
            "error": str(exc),
            "n_observations": int(n),
        }


def _rolling_median_2d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or int(window) <= 1:
        return values.astype(float, copy=True)
    half = max(1, int(window) // 2)
    out = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        a = max(0, i - half)
        b = min(values.shape[0], i + half + 1)
        out[i] = np.nanmedian(values[a:b], axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _cv_mats_3d(dt: float, accel_var_xy: float, accel_var_z: float) -> tuple[np.ndarray, np.ndarray]:
    dt = max(1e-3, float(dt))
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    F = np.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    q_xy = accel_var_xy * np.array(
        [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]],
        dtype=float,
    )
    q_z = accel_var_z * np.array(
        [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]],
        dtype=float,
    )

    Q = np.zeros((6, 6), dtype=float)
    Q[np.ix_([0, 3], [0, 3])] = q_xy
    Q[np.ix_([1, 4], [1, 4])] = q_xy
    Q[np.ix_([2, 5], [2, 5])] = q_z
    return F, Q


def _estimate_initial_velocity(y: np.ndarray, t: np.ndarray, n: int) -> np.ndarray:
    m = min(max(2, int(n)), len(t))
    if m < 2:
        return np.zeros(3, dtype=float)
    tt = t[:m] - t[0]
    A = np.column_stack([np.ones(m), tt])
    out = np.zeros(3, dtype=float)
    for dim in range(3):
        out[dim] = np.linalg.lstsq(A, y[:m, dim], rcond=None)[0][1]
    return out


def _kalman_rts_constant_velocity_3d(
    *,
    t: np.ndarray,
    y_meas: np.ndarray,
    meas_std_xy_m: float,
    meas_std_z_m: float,
    accel_std_xy_mps2: float,
    accel_std_z_mps2: float,
    gate_sigma: float,
    init_vel_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).reshape(-1)
    y_meas = np.asarray(y_meas, dtype=float)
    n = len(t)
    if y_meas.shape != (n, 3):
        raise ValueError(f"Expected y_meas shape {(n, 3)}, got {y_meas.shape}")
    if n < 2:
        x = np.zeros((n, 6), dtype=float)
        if n:
            x[0, :3] = y_meas[0]
        P = np.zeros((n, 6, 6), dtype=float)
        return x, P

    meas_var_xy = float(meas_std_xy_m) ** 2
    meas_var_z = float(meas_std_z_m) ** 2
    accel_var_xy = float(accel_std_xy_mps2) ** 2
    accel_var_z = float(accel_std_z_mps2) ** 2

    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    R_base = np.diag([meas_var_xy, meas_var_xy, meas_var_z]).astype(float)
    I = np.eye(6, dtype=float)

    v0 = _estimate_initial_velocity(y_meas, t, init_vel_points)
    x0 = np.concatenate([y_meas[0], v0]).astype(float)
    P0 = np.diag(
        [
            meas_var_xy * 2.0,
            meas_var_xy * 2.0,
            meas_var_z * 2.0,
            120.0**2,
            120.0**2,
            40.0**2,
        ]
    ).astype(float)

    x_pred = np.zeros((n, 6), dtype=float)
    P_pred = np.zeros((n, 6, 6), dtype=float)
    x_filt = np.zeros((n, 6), dtype=float)
    P_filt = np.zeros((n, 6, 6), dtype=float)
    F_list = [np.eye(6, dtype=float) for _ in range(n)]

    for k in range(n):
        if k == 0:
            xp = x0.copy()
            Pp = P0.copy()
        else:
            F, Q = _cv_mats_3d(t[k] - t[k - 1], accel_var_xy, accel_var_z)
            xp = F @ x_filt[k - 1]
            Pp = F @ P_filt[k - 1] @ F.T + Q
            F_list[k] = F

        z = y_meas[k]
        innov = z - (H @ xp)
        S0 = H @ Pp @ H.T + R_base
        d2 = float(innov.T @ np.linalg.pinv(S0) @ innov)
        gate2 = float(gate_sigma) ** 2
        scale = max(1.0, d2 / gate2) if gate2 > 0 else 1.0
        R = R_base * scale
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.pinv(S)
        xf = xp + K @ innov
        IKH = I - K @ H
        Pf = IKH @ Pp @ IKH.T + K @ R @ K.T

        x_pred[k] = xp
        P_pred[k] = Pp
        x_filt[k] = xf
        P_filt[k] = Pf

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(n - 2, -1, -1):
        F_next = F_list[k + 1]
        Ck = P_filt[k] @ F_next.T @ np.linalg.pinv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + Ck @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + Ck @ (P_smooth[k + 1] - P_pred[k + 1]) @ Ck.T

    return x_smooth, P_smooth
