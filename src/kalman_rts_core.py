"""Production Kalman filter + RTS smoother reconstruction backend.

The spline backends fit local curves after dynamic segmentation.  This module is
intentionally different: it fits one fixed-interval linear-Gaussian smoother to
all prepared paired ADS-B observations for a flight/method preset.  The output is
adapted to the same ``evaluate(t, deriv=...)`` contract used by the spline cores
so downstream rendering and evaluation can compare methods fairly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import math

import numpy as np


StateInterpolation = Literal["quintic_state_bridge", "constant_acceleration_left"]


@dataclass(frozen=True)
class KalmanRTSConfig:
    """Configuration for the 3D constant-acceleration Kalman/RTS backend.

    The dynamic model state is ``[x, y, z, vx, vy, vz, ax, ay, az]`` in local
    ENU meters, meters/second, and meters/second^2.  The process model assumes
    piecewise constant acceleration driven by white jerk.  Position and velocity
    ADS-B observations are both used as measurements.
    """

    position_std_xy_m: float = 25.0
    position_std_z_m: float = 40.0
    velocity_std_xy_mps: float = 8.0
    velocity_std_z_mps: float = 4.0
    jerk_std_xy_mps3: float = 1.2
    jerk_std_z_mps3: float = 0.7

    initial_position_std_xy_m: float = 60.0
    initial_position_std_z_m: float = 90.0
    initial_velocity_std_xy_mps: float = 80.0
    initial_velocity_std_z_mps: float = 30.0
    initial_acceleration_std_xy_mps2: float = 8.0
    initial_acceleration_std_z_mps2: float = 4.0
    init_window_points: int = 7

    use_velocity_observations: bool = True
    robust_measurement_scaling: bool = True
    gate_sigma: float = 4.5
    max_measurement_scale: float = 100.0
    min_dt_s: float = 1e-3
    covariance_ridge: float = 1e-9
    interpolation: StateInterpolation = "quintic_state_bridge"

    # Diagnostic label only; the filter is not a V-Spline and does not segment.
    backend_name: str = "global_constant_acceleration_kalman_rts"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KalmanRTSInput:
    """Prepared paired observations consumed by the Kalman/RTS core."""

    t: np.ndarray
    y: np.ndarray
    v: np.ndarray
    dim_names: tuple[str, ...] = ("x", "y", "z")


@dataclass
class KalmanRTSFit:
    """Fitted fixed-interval Kalman/RTS reconstruction."""

    t: np.ndarray
    state_filtered: np.ndarray
    covariance_filtered: np.ndarray
    state_predicted: np.ndarray
    covariance_predicted: np.ndarray
    state_smoothed: np.ndarray
    covariance_smoothed: np.ndarray
    lambda_intervals: np.ndarray
    config: KalmanRTSConfig
    diagnostics: dict[str, Any]
    dim_names: tuple[str, ...] = ("x", "y", "z")

    @property
    def n_observations(self) -> int:
        return int(self.t.size)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def positions(self) -> np.ndarray:
        return self.state_smoothed[:, 0:3]

    @property
    def velocities(self) -> np.ndarray:
        return self.state_smoothed[:, 3:6]

    @property
    def accelerations(self) -> np.ndarray:
        return self.state_smoothed[:, 6:9]

    def evaluate(self, t_eval: np.ndarray | list[float] | float, deriv: int = 0) -> np.ndarray:
        """Evaluate smoothed position, velocity, acceleration, or jerk.

        ``deriv`` follows the same convention as the spline cores:
        0 -> position, 1 -> velocity, 2 -> acceleration, 3 -> jerk.  The default
        interpolation is a local quintic state bridge through neighbouring RTS
        position/velocity/acceleration states.  It is a rendering adapter, not a
        segmentation or fitting step.
        """
        if deriv not in {0, 1, 2, 3}:
            raise ValueError("deriv must be one of 0, 1, 2, 3")
        q = np.asarray(t_eval, dtype=float).reshape(-1)
        if not np.all(np.isfinite(q)):
            raise ValueError("t_eval must contain only finite numeric values")
        if self.t.size < 2:
            raise ValueError("fit must contain at least two observations")
        if self.config.interpolation == "constant_acceleration_left":
            return _evaluate_constant_acceleration_left(q, self.t, self.state_smoothed, deriv=deriv)
        return _evaluate_quintic_state_bridge(q, self.t, self.state_smoothed, deriv=deriv)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "state_order": ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"],
            "state_filtered": self.state_filtered.tolist(),
            "state_smoothed": self.state_smoothed.tolist(),
            "lambda_intervals": self.lambda_intervals.tolist(),
            "config": asdict(self.config),
            "diagnostics": self.diagnostics,
            "dim_names": list(self.dim_names),
        }


def default_kalman_rts_config_for_preset(preset: str) -> KalmanRTSConfig:
    """Return comparable Kalman/RTS settings for the three output presets.

    Accurate trusts ADS-B observations more and allows a more agile white-jerk
    process; smooth increases observation noise and lowers process freedom;
    balanced sits between those two choices.
    """
    key = str(preset).strip().lower()
    if key == "accurate":
        return KalmanRTSConfig(
            position_std_xy_m=14.0,
            position_std_z_m=22.0,
            velocity_std_xy_mps=4.5,
            velocity_std_z_mps=2.2,
            jerk_std_xy_mps3=2.4,
            jerk_std_z_mps3=1.2,
            initial_position_std_xy_m=35.0,
            initial_position_std_z_m=55.0,
            gate_sigma=5.0,
            max_measurement_scale=70.0,
        )
    if key == "balanced":
        return KalmanRTSConfig()
    if key == "smooth":
        return KalmanRTSConfig(
            position_std_xy_m=55.0,
            position_std_z_m=85.0,
            velocity_std_xy_mps=16.0,
            velocity_std_z_mps=7.5,
            jerk_std_xy_mps3=0.35,
            jerk_std_z_mps3=0.20,
            initial_position_std_xy_m=100.0,
            initial_position_std_z_m=150.0,
            initial_velocity_std_xy_mps=120.0,
            initial_velocity_std_z_mps=50.0,
            gate_sigma=4.0,
            max_measurement_scale=150.0,
        )
    raise ValueError("Kalman-RTS preset must be one of: balanced, accurate, smooth")


def validate_kalman_input(core_input: KalmanRTSInput) -> KalmanRTSInput:
    t = np.asarray(core_input.t, dtype=float).reshape(-1)
    y = np.asarray(core_input.y, dtype=float)
    v = np.asarray(core_input.v, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if y.shape != v.shape:
        raise ValueError(f"y and v must have identical shape; got {y.shape} and {v.shape}")
    if y.shape != (t.size, 3):
        raise ValueError(f"KalmanRTSInput requires shape {(t.size, 3)}, got y={y.shape}")
    if t.size < 2:
        raise ValueError("at least two paired observations are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(v)):
        raise ValueError("t, y, and v must contain only finite values")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    dim_names = core_input.dim_names if len(core_input.dim_names) == 3 else ("x", "y", "z")
    return KalmanRTSInput(t=t, y=y, v=v, dim_names=tuple(dim_names))


def fit_kalman_rts_component(
    core_input: KalmanRTSInput,
    config: KalmanRTSConfig | None = None,
    *,
    component_id: str = "kalman_rts_full_track",
) -> KalmanRTSFit:
    """Fit a whole-track constant-acceleration Kalman filter and RTS smoother."""
    cfg = config or KalmanRTSConfig()
    inp = validate_kalman_input(core_input)
    _validate_config(cfg)
    t, y, v = inp.t, inp.y, inp.v
    n = int(t.size)

    H, R_base, measurement_labels = _measurement_model(cfg)
    z = np.column_stack([y, v]) if bool(cfg.use_velocity_observations) else y.copy()
    state_dim = 9
    I = np.eye(state_dim, dtype=float)

    x0 = _initial_state(t, y, v, cfg)
    P0 = np.diag(
        [
            float(cfg.initial_position_std_xy_m) ** 2,
            float(cfg.initial_position_std_xy_m) ** 2,
            float(cfg.initial_position_std_z_m) ** 2,
            float(cfg.initial_velocity_std_xy_mps) ** 2,
            float(cfg.initial_velocity_std_xy_mps) ** 2,
            float(cfg.initial_velocity_std_z_mps) ** 2,
            float(cfg.initial_acceleration_std_xy_mps2) ** 2,
            float(cfg.initial_acceleration_std_xy_mps2) ** 2,
            float(cfg.initial_acceleration_std_z_mps2) ** 2,
        ]
    ).astype(float)

    x_pred = np.zeros((n, state_dim), dtype=float)
    P_pred = np.zeros((n, state_dim, state_dim), dtype=float)
    x_filt = np.zeros_like(x_pred)
    P_filt = np.zeros_like(P_pred)
    F_list = [np.eye(state_dim, dtype=float) for _ in range(n)]
    Q_list = [np.zeros((state_dim, state_dim), dtype=float) for _ in range(n)]

    nis_values: list[float] = []
    innovation_norms: list[float] = []
    measurement_scales: list[float] = []
    downweighted_count = 0

    for k in range(n):
        if k == 0:
            xp = x0.copy()
            Pp = P0.copy()
        else:
            F, Q = _ca_mats_3d(
                max(float(t[k] - t[k - 1]), float(cfg.min_dt_s)),
                float(cfg.jerk_std_xy_mps3) ** 2,
                float(cfg.jerk_std_z_mps3) ** 2,
            )
            F_list[k] = F
            Q_list[k] = Q
            xp = F @ x_filt[k - 1]
            Pp = _symmetrize(F @ P_filt[k - 1] @ F.T + Q)

        zk = z[k]
        innov = zk - (H @ xp)
        S_base = _symmetrize(H @ Pp @ H.T + R_base)
        S_base_inv = np.linalg.pinv(S_base, rcond=1e-12)
        nis = float(innov.T @ S_base_inv @ innov)
        gate2 = max(1e-12, float(cfg.gate_sigma) ** 2 * float(len(zk)))
        scale = 1.0
        if bool(cfg.robust_measurement_scaling):
            scale = min(float(cfg.max_measurement_scale), max(1.0, nis / gate2))
        if scale > 1.0 + 1e-12:
            downweighted_count += 1
        R = R_base * scale
        S = _symmetrize(H @ Pp @ H.T + R)
        K = Pp @ H.T @ np.linalg.pinv(S, rcond=1e-12)
        xf = xp + K @ innov
        IKH = I - K @ H
        Pf = _symmetrize(IKH @ Pp @ IKH.T + K @ R @ K.T)
        if float(cfg.covariance_ridge) > 0:
            Pf = _symmetrize(Pf + float(cfg.covariance_ridge) * I)

        x_pred[k] = xp
        P_pred[k] = Pp
        x_filt[k] = xf
        P_filt[k] = Pf
        nis_values.append(nis)
        innovation_norms.append(float(np.linalg.norm(innov)))
        measurement_scales.append(scale)

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    smoother_gain_norms: list[float] = []
    for k in range(n - 2, -1, -1):
        F_next = F_list[k + 1]
        Ck = P_filt[k] @ F_next.T @ np.linalg.pinv(P_pred[k + 1], rcond=1e-12)
        x_smooth[k] = x_filt[k] + Ck @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = _symmetrize(P_filt[k] + Ck @ (P_smooth[k + 1] - P_pred[k + 1]) @ Ck.T)
        smoother_gain_norms.append(float(np.linalg.norm(Ck, ord="fro")))

    lambda_intervals = _process_weight_intervals(t, cfg)
    diagnostics = _diagnostics(
        component_id=component_id,
        inp=inp,
        cfg=cfg,
        x_filt=x_filt,
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        lambda_intervals=lambda_intervals,
        measurement_labels=measurement_labels,
        nis_values=np.asarray(nis_values, dtype=float),
        innovation_norms=np.asarray(innovation_norms, dtype=float),
        measurement_scales=np.asarray(measurement_scales, dtype=float),
        downweighted_count=downweighted_count,
        smoother_gain_norms=np.asarray(smoother_gain_norms, dtype=float),
    )

    return KalmanRTSFit(
        t=t,
        state_filtered=x_filt,
        covariance_filtered=P_filt,
        state_predicted=x_pred,
        covariance_predicted=P_pred,
        state_smoothed=x_smooth,
        covariance_smoothed=P_smooth,
        lambda_intervals=lambda_intervals,
        config=cfg,
        diagnostics=diagnostics,
        dim_names=inp.dim_names,
    )


def _validate_config(config: KalmanRTSConfig) -> None:
    positive_fields = [
        "position_std_xy_m",
        "position_std_z_m",
        "velocity_std_xy_mps",
        "velocity_std_z_mps",
        "jerk_std_xy_mps3",
        "jerk_std_z_mps3",
        "initial_position_std_xy_m",
        "initial_position_std_z_m",
        "initial_velocity_std_xy_mps",
        "initial_velocity_std_z_mps",
        "initial_acceleration_std_xy_mps2",
        "initial_acceleration_std_z_mps2",
        "gate_sigma",
        "max_measurement_scale",
        "min_dt_s",
    ]
    for name in positive_fields:
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if int(config.init_window_points) < 2:
        raise ValueError("init_window_points must be at least 2")
    if config.interpolation not in {"quintic_state_bridge", "constant_acceleration_left"}:
        raise ValueError("unsupported Kalman RTS interpolation mode")


def _measurement_model(config: KalmanRTSConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    H_pos = np.zeros((3, 9), dtype=float)
    H_pos[0, 0] = H_pos[1, 1] = H_pos[2, 2] = 1.0
    R_pos = np.diag(
        [
            float(config.position_std_xy_m) ** 2,
            float(config.position_std_xy_m) ** 2,
            float(config.position_std_z_m) ** 2,
        ]
    )
    labels = ["position_x", "position_y", "position_z"]
    if not bool(config.use_velocity_observations):
        return H_pos, R_pos, labels
    H_vel = np.zeros((3, 9), dtype=float)
    H_vel[0, 3] = H_vel[1, 4] = H_vel[2, 5] = 1.0
    R_vel = np.diag(
        [
            float(config.velocity_std_xy_mps) ** 2,
            float(config.velocity_std_xy_mps) ** 2,
            float(config.velocity_std_z_mps) ** 2,
        ]
    )
    H = np.vstack([H_pos, H_vel])
    R = np.zeros((6, 6), dtype=float)
    R[:3, :3] = R_pos
    R[3:, 3:] = R_vel
    labels.extend(["velocity_x", "velocity_y", "velocity_z"])
    return H, R, labels


def _ca_mats_1d(dt: float, jerk_var: float) -> tuple[np.ndarray, np.ndarray]:
    dt = max(1e-9, float(dt))
    F = np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    dt5 = dt4 * dt
    Q = float(jerk_var) * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=float,
    )
    return F, Q


def _ca_mats_3d(dt: float, jerk_var_xy: float, jerk_var_z: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(9, dtype=float)
    Q = np.zeros((9, 9), dtype=float)
    for axis, jerk_var in enumerate((jerk_var_xy, jerk_var_xy, jerk_var_z)):
        Fa, Qa = _ca_mats_1d(dt, jerk_var)
        idx = [axis, axis + 3, axis + 6]
        F[np.ix_(idx, idx)] = Fa
        Q[np.ix_(idx, idx)] = Qa
    return F, Q


def _initial_state(t: np.ndarray, y: np.ndarray, v: np.ndarray, config: KalmanRTSConfig) -> np.ndarray:
    m = min(max(2, int(config.init_window_points)), len(t))
    tt = t[:m] - t[0]
    A = np.column_stack([np.ones(m), tt, 0.5 * tt * tt])
    p0 = y[0].astype(float)
    v0 = v[0].astype(float)
    a0 = np.zeros(3, dtype=float)
    # Least-squares acceleration from the first few position rows is robust
    # enough for initialization; reported velocity remains the first velocity
    # state so the filter starts in the same observation frame as the splines.
    if m >= 3:
        for dim in range(3):
            try:
                coef = np.linalg.lstsq(A, y[:m, dim], rcond=None)[0]
                a0[dim] = float(coef[2])
            except Exception:
                a0[dim] = 0.0
    elif len(t) >= 2:
        dt = max(float(t[1] - t[0]), float(config.min_dt_s))
        a0 = (v[1] - v[0]) / dt
    return np.concatenate([p0, v0, a0]).astype(float)


def _process_weight_intervals(t: np.ndarray, config: KalmanRTSConfig) -> np.ndarray:
    h = np.diff(np.asarray(t, dtype=float))
    jerk_var_mean = (2.0 * float(config.jerk_std_xy_mps3) ** 2 + float(config.jerk_std_z_mps3) ** 2) / 3.0
    return h / max(jerk_var_mean, 1e-12)


def _diagnostics(
    *,
    component_id: str,
    inp: KalmanRTSInput,
    cfg: KalmanRTSConfig,
    x_filt: np.ndarray,
    x_smooth: np.ndarray,
    P_smooth: np.ndarray,
    lambda_intervals: np.ndarray,
    measurement_labels: list[str],
    nis_values: np.ndarray,
    innovation_norms: np.ndarray,
    measurement_scales: np.ndarray,
    downweighted_count: int,
    smoother_gain_norms: np.ndarray,
) -> dict[str, Any]:
    y_hat = x_smooth[:, :3]
    v_hat = x_smooth[:, 3:6]
    a_hat = x_smooth[:, 6:9]
    pos_delta = y_hat - inp.y
    vel_delta = v_hat - inp.v
    e3 = np.linalg.norm(pos_delta, axis=1)
    ev = np.linalg.norm(vel_delta, axis=1)
    acc_norm = np.linalg.norm(a_hat, axis=1)
    if inp.t.size >= 2:
        dt = np.diff(inp.t)
        jerk = np.diff(a_hat, axis=0) / np.maximum(dt[:, None], 1e-9)
        jerk_norm = np.linalg.norm(jerk, axis=1)
    else:
        jerk_norm = np.zeros(1, dtype=float)
    cov_trace = np.trace(P_smooth, axis1=1, axis2=2)

    return {
        "method": "global_kalman_rts_fixed_interval_smoother",
        "backend": "kalman_rts",
        "component_id": component_id,
        "basis": "none_state_space_constant_acceleration",
        "objective": "linear_gaussian_position_velocity_measurements_with_white_jerk_process_model_and_rts_fixed_interval_smoothing",
        "segmentation_used": False,
        "n_observations": int(inp.t.size),
        "n_basis": int(9 * inp.t.size),
        "state_dimension": 9,
        "state_order": ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"],
        "measurement_labels": measurement_labels,
        "filter": {
            "model": "constant_acceleration_white_jerk_3d",
            "update_count": int(inp.t.size),
            "accepted_update_count": int(inp.t.size),
            "rejected_update_count": 0,
            "downweighted_update_count": int(downweighted_count),
            "robust_measurement_scaling": bool(cfg.robust_measurement_scaling),
            "gate_sigma": float(cfg.gate_sigma),
            "max_measurement_scale": float(cfg.max_measurement_scale),
            "nis_rms": _rms(nis_values),
            "nis_p95": _q(nis_values, 0.95),
            "nis_max": _max(nis_values),
            "innovation_norm_rms": _rms(innovation_norms),
            "innovation_norm_p95": _q(innovation_norms, 0.95),
            "innovation_norm_max": _max(innovation_norms),
            "measurement_scale_p50": _q(measurement_scales, 0.50),
            "measurement_scale_p95": _q(measurement_scales, 0.95),
            "measurement_scale_max": _max(measurement_scales),
        },
        "smoother": {
            "algorithm": "rauch_tung_striebel_fixed_interval_backward_pass",
            "smoother_gain_frobenius_rms": _rms(smoother_gain_norms),
            "smoother_gain_frobenius_p95": _q(smoother_gain_norms, 0.95),
            "smoothed_covariance_trace_median": _q(cov_trace, 0.50),
            "smoothed_covariance_trace_p95": _q(cov_trace, 0.95),
        },
        "process_weight_intervals": {
            "count": int(lambda_intervals.size),
            "min": float(np.min(lambda_intervals)) if lambda_intervals.size else None,
            "max": float(np.max(lambda_intervals)) if lambda_intervals.size else None,
            "meaning": "dt divided by mean white-jerk variance; stored only for output-schema comparability with spline lambda_intervals",
        },
        "hard_position_anchors": {
            "raw_anchor_count": 0,
            "deduped_anchor_count": 0,
            "max_anchor_error_m": 0.0,
            "p95_anchor_error_m": 0.0,
            "anchors": [],
        },
        "n_anchors": 0,
        "max_anchor_error_m": 0.0,
        "hard_velocity_constraints": {"count": 0, "max_error_mps": 0.0, "p95_error_mps": 0.0, "constraints": []},
        "boundary_velocity_priors": {"count": 0, "applied_in_core": False, "priors": []},
        "boundary_acceleration_priors": {"count": 0, "applied_in_core": False, "priors": []},
        "position_residual_rmse_3d_m": float(np.sqrt(np.mean(e3 * e3))),
        "position_residual_median_3d_m": float(np.median(e3)),
        "position_residual_p95_3d_m": float(np.quantile(e3, 0.95)),
        "position_residual_max_3d_m": float(np.max(e3)),
        "position_residual_rms_by_dim": np.sqrt(np.mean(pos_delta * pos_delta, axis=0)).tolist(),
        "velocity_residual_rmse_3d_mps": float(np.sqrt(np.mean(ev * ev))),
        "velocity_residual_median_3d_mps": float(np.median(ev)),
        "velocity_residual_p95_3d_mps": float(np.quantile(ev, 0.95)),
        "velocity_residual_rms_by_dim": np.sqrt(np.mean(vel_delta * vel_delta, axis=0)).tolist(),
        "accel_rms_mps2": _rms(acc_norm),
        "accel_p95_mps2": _q(acc_norm, 0.95),
        "accel_max_mps2": _max(acc_norm),
        "jerk_rms_mps3": _rms(jerk_norm),
        "jerk_p95_mps3": _q(jerk_norm, 0.95),
        "jerk_max_mps3": _max(jerk_norm),
        "hard_endpoint_constraint_max_abs_error": 0.0,
        "config": asdict(cfg),
    }


def _evaluate_constant_acceleration_left(q: np.ndarray, t: np.ndarray, state: np.ndarray, *, deriv: int) -> np.ndarray:
    out = np.zeros((q.size, 3), dtype=float)
    idx = np.searchsorted(t, q, side="right") - 1
    idx = np.clip(idx, 0, t.size - 1)
    dt = q - t[idx]
    p = state[idx, :3]
    v = state[idx, 3:6]
    a = state[idx, 6:9]
    if deriv == 0:
        out = p + v * dt[:, None] + 0.5 * a * (dt[:, None] ** 2)
    elif deriv == 1:
        out = v + a * dt[:, None]
    elif deriv == 2:
        out = a.copy()
    else:
        out.fill(0.0)
    return out


def _evaluate_quintic_state_bridge(q: np.ndarray, t: np.ndarray, state: np.ndarray, *, deriv: int) -> np.ndarray:
    out = np.zeros((q.size, 3), dtype=float)
    for i, tq in enumerate(q):
        if tq <= t[0]:
            out[i] = _eval_extrapolated_state(float(tq - t[0]), state[0], deriv)
            continue
        if tq >= t[-1]:
            out[i] = _eval_extrapolated_state(float(tq - t[-1]), state[-1], deriv)
            continue
        k = int(np.searchsorted(t, tq, side="right") - 1)
        k = max(0, min(k, t.size - 2))
        h = float(t[k + 1] - t[k])
        if h <= 0:
            raise ValueError("fit times must be strictly increasing")
        u = float((tq - t[k]) / h)
        out[i] = _eval_quintic_interval(state[k], state[k + 1], h, u, deriv)
    return out


def _eval_extrapolated_state(dt: float, s: np.ndarray, deriv: int) -> np.ndarray:
    p = np.asarray(s[:3], dtype=float)
    v = np.asarray(s[3:6], dtype=float)
    a = np.asarray(s[6:9], dtype=float)
    if deriv == 0:
        return p + v * dt + 0.5 * a * dt * dt
    if deriv == 1:
        return v + a * dt
    if deriv == 2:
        return a.copy()
    return np.zeros(3, dtype=float)


def _eval_quintic_interval(s0: np.ndarray, s1: np.ndarray, h: float, u: float, deriv: int) -> np.ndarray:
    p0, v0, a0 = s0[:3], s0[3:6], s0[6:9]
    p1, v1, a1 = s1[:3], s1[3:6], s1[6:9]
    c0 = p0
    c1 = v0 * h
    c2 = 0.5 * a0 * h * h
    b0 = p1 - (c0 + c1 + c2)
    b1 = v1 * h - (c1 + 2.0 * c2)
    b2 = a1 * h * h - 2.0 * c2
    c3 = 10.0 * b0 - 4.0 * b1 + 0.5 * b2
    c4 = -15.0 * b0 + 7.0 * b1 - b2
    c5 = 6.0 * b0 - 3.0 * b1 + 0.5 * b2
    if deriv == 0:
        return c0 + c1 * u + c2 * u**2 + c3 * u**3 + c4 * u**4 + c5 * u**5
    if deriv == 1:
        return (c1 + 2.0 * c2 * u + 3.0 * c3 * u**2 + 4.0 * c4 * u**3 + 5.0 * c5 * u**4) / h
    if deriv == 2:
        return (2.0 * c2 + 6.0 * c3 * u + 12.0 * c4 * u**2 + 20.0 * c5 * u**3) / (h * h)
    return (6.0 * c3 + 24.0 * c4 * u + 60.0 * c5 * u**2) / (h**3)


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _rms(values: np.ndarray) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr * arr)))


def _q(values: np.ndarray, q: float) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def _max(values: np.ndarray) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.max(arr))
