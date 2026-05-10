from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import math
import warnings

import numpy as np

try:  # Keep the core usable even when loguru is not installed.
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


PenaltyMode = Literal["constant", "adaptive"]


@dataclass(frozen=True)
class VSplineCoreConfig:
    """Configuration for the paper-oriented V-Spline core.

    Parameters
    ----------
    velocity_weight:
        Global positive weight gamma on velocity residuals.
    penalty_mode:
        "constant" uses lambda_i = smoothing_lambda on every interval.
        "adaptive" uses lambda_i = adaptive_eta * dt_i / ||mean_velocity_i||^2.
    smoothing_lambda:
        Constant interval acceleration-penalty weight for penalty_mode="constant".
    adaptive_eta:
        Positive global adaptive penalty parameter for penalty_mode="adaptive".
    adaptive_speed_floor_mps:
        Optional practical floor for the adaptive denominator. Leave as None
        for the paper-literal rule. Setting this is a numerical modification.
    optimize:
        If True, optimize velocity_weight and the smoothing/adaptive parameter
        by analytic leave-one-out CV before fitting.
    optimize_velocity_weight:
        Whether the optimizer may change velocity_weight.
    log10_velocity_weight_bounds:
        Log10 bounds used when optimizing velocity_weight.
    log10_lambda_bounds:
        Log10 bounds used when optimizing smoothing_lambda or adaptive_eta.
    min_loocv_denominator:
        Small safeguard for CV denominators close to zero.
    hard_endpoint_constraints:
        If True, force endpoint Hermite states to equal the raw endpoint data.
    hard_endpoint_positions:
        If True, force p_0=y_0 and p_{n-1}=y_{n-1}.
    hard_endpoint_velocities:
        If True, force v_0=v_obs_0 and v_{n-1}=v_obs_{n-1}.
    """

    velocity_weight: float = 1.0
    penalty_mode: PenaltyMode = "constant"
    smoothing_lambda: float = 1.0
    adaptive_eta: float = 1.0
    adaptive_speed_floor_mps: float | None = None

    optimize: bool = False
    optimize_velocity_weight: bool = True
    log10_velocity_weight_bounds: tuple[float, float] = (-4.0, 4.0)
    log10_lambda_bounds: tuple[float, float] = (-8.0, 8.0)
    min_loocv_denominator: float = 1e-8

    # Expensive diagnostics.  Dense condition numbers and analytic LOOCV require
    # matrix inversions/decompositions and can dominate large segmented runs.
    compute_loocv_score: bool = False
    condition_number_max_size: int | None = 1024

    # Hard endpoint constraints for segment stitching / exact boundary states.
    hard_endpoint_constraints: bool = True
    hard_endpoint_positions: bool = True
    # 4BAAD9 report action: hard endpoint velocities amplify asynchronous ADS-B
    # velocity noise at local joins.  Keep endpoint positions hard by default,
    # but make endpoint velocities soft unless a caller explicitly opts in.
    hard_endpoint_velocities: bool = False


@dataclass(frozen=True)
class VSplineCoreInput:
    """Strict input consumed by the V-Spline core.

    t:
        Shape (n,). Strictly increasing observation times in seconds.
    y:
        Shape (n, d). Position observations in metric coordinates.
    v:
        Shape (n, d). Velocity observations in the same coordinate frame,
        in units of y per second.
    dim_names:
        Optional names for the coordinate dimensions.
    """

    t: np.ndarray
    y: np.ndarray
    v: np.ndarray
    dim_names: tuple[str, ...] = ("x", "y", "z")
    # Optional per-observation trust scale for ADS-B velocity rows.  This keeps
    # velocity observations in the Hermite V-Spline objective while making stale
    # or asynchronous velocity reports less able to create high jerk.
    velocity_weight_scale: np.ndarray | list[float] | tuple[float, ...] | None = None


@dataclass(frozen=True)
class VSplineEndpointState:
    """Optional external endpoint state override for C1 segment stitching."""

    position: np.ndarray | list[float] | tuple[float, ...] | None = None
    velocity: np.ndarray | list[float] | tuple[float, ...] | None = None


@dataclass(frozen=True)
class VSplineEndpointConstraints:
    """Endpoint hard-constraint values for local segment fitting.

    When ``start`` or ``end`` is omitted, raw first/last segment observations are
    used, preserving the legacy behavior.  When a state is supplied, its values
    replace the raw endpoint values for the selected hard constraints.
    """

    start: VSplineEndpointState | None = None
    end: VSplineEndpointState | None = None
    hard_start_position: bool = True
    hard_start_velocity: bool = True
    hard_end_position: bool = True
    hard_end_velocity: bool = True


@dataclass
class VSplineCoreFit:
    """Fitted V-Spline in nodal Hermite coordinates."""

    t: np.ndarray
    theta_position: np.ndarray
    theta_velocity: np.ndarray
    lambda_intervals: np.ndarray
    config: VSplineCoreConfig
    diagnostics: dict[str, Any]
    dim_names: tuple[str, ...] = ("x", "y", "z")

    @property
    def n_observations(self) -> int:
        return int(self.t.size)

    @property
    def dimension(self) -> int:
        return int(self.theta_position.shape[1])

    def evaluate(self, t_eval: np.ndarray | list[float] | float, deriv: int = 0) -> np.ndarray:
        """Evaluate the fitted curve or derivative.

        Parameters
        ----------
        t_eval:
            Times in the same seconds scale used at input.
        deriv:
            0 -> position, 1 -> velocity, 2 -> acceleration, 3 -> jerk.

        Returns
        -------
        ndarray, shape (len(t_eval), d)
            Evaluated values. Outside the observation range, the paper spline
            is linearly extrapolated; acceleration and jerk are zero outside
            the range.
        """
        return evaluate_hermite(
            np.asarray(t_eval, dtype=float),
            self.t,
            self.theta_position,
            self.theta_velocity,
            deriv=deriv,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "theta_position": self.theta_position.tolist(),
            "theta_velocity": self.theta_velocity.tolist(),
            "lambda_intervals": self.lambda_intervals.tolist(),
            "config": asdict(self.config),
            "diagnostics": self.diagnostics,
            "dim_names": list(self.dim_names),
        }


def validate_core_input(core_input: VSplineCoreInput) -> VSplineCoreInput:
    """Return a normalized, validated copy of the core input."""
    t = np.asarray(core_input.t, dtype=float).reshape(-1)
    y = np.asarray(core_input.y, dtype=float)
    v = np.asarray(core_input.v, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be a one-dimensional array")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)

    if y.shape != v.shape:
        raise ValueError(f"y and v must have the same shape; got {y.shape} and {v.shape}")
    if y.shape[0] != t.size:
        raise ValueError(f"t length must match y/v rows; got len(t)={t.size}, y rows={y.shape[0]}")
    if t.size < 2:
        raise ValueError("at least two paired observations are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(v)):
        raise ValueError("t, y, and v must contain only finite numeric values")

    dt = np.diff(t)
    if not np.all(dt > 0.0):
        raise ValueError("t must be strictly increasing with no duplicates")

    velocity_weight_scale = None
    if core_input.velocity_weight_scale is not None:
        velocity_weight_scale = np.asarray(core_input.velocity_weight_scale, dtype=float).reshape(-1)
        if velocity_weight_scale.shape != (t.size,):
            raise ValueError(f"velocity_weight_scale must have shape ({t.size},); got {velocity_weight_scale.shape}")
        if not np.all(np.isfinite(velocity_weight_scale)):
            raise ValueError("velocity_weight_scale must contain only finite values")
        velocity_weight_scale = np.clip(velocity_weight_scale, 0.0, 1.0)

    dim_names = core_input.dim_names
    if len(dim_names) != y.shape[1]:
        dim_names = tuple(f"x{j}" for j in range(y.shape[1]))

    return VSplineCoreInput(t=t, y=y, v=v, dim_names=tuple(dim_names), velocity_weight_scale=velocity_weight_scale)


def compute_interval_weights(
    t: np.ndarray,
    y: np.ndarray,
    config: VSplineCoreConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute lambda_i for every interval."""
    if config.velocity_weight <= 0:
        raise ValueError("velocity_weight must be positive")

    h = np.diff(t)
    if config.penalty_mode == "constant":
        if config.smoothing_lambda <= 0:
            raise ValueError("smoothing_lambda must be positive in constant mode")
        lam = np.full(h.shape, float(config.smoothing_lambda), dtype=float)
        return lam, {
            "penalty_mode": "constant",
            "smoothing_lambda": float(config.smoothing_lambda),
            "adaptive_speed_floor_mps": None,
            "speed_floor_applied_count": 0,
        }

    if config.penalty_mode != "adaptive":
        raise ValueError(f"unknown penalty_mode={config.penalty_mode!r}")
    if config.adaptive_eta <= 0:
        raise ValueError("adaptive_eta must be positive in adaptive mode")

    mean_velocity = np.diff(y, axis=0) / h[:, None]
    speed_sq_raw = np.sum(mean_velocity * mean_velocity, axis=1)
    speed_sq = speed_sq_raw.copy()

    floor = config.adaptive_speed_floor_mps
    floor_count = 0
    if floor is not None:
        if floor < 0:
            raise ValueError("adaptive_speed_floor_mps must be non-negative or None")
        floor_sq = float(floor) ** 2
        floor_count = int(np.sum(speed_sq < floor_sq))
        speed_sq = np.maximum(speed_sq, floor_sq)

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = float(config.adaptive_eta) * h / speed_sq

    if not np.all(np.isfinite(lam)):
        raise FloatingPointError(
            "adaptive interval weights are not finite. This usually means an interval "
            "has near-zero mean velocity. Set adaptive_speed_floor_mps only if you "
            "accept that practical modification."
        )

    return lam, {
        "penalty_mode": "adaptive",
        "adaptive_eta": float(config.adaptive_eta),
        "adaptive_speed_floor_mps": floor,
        "speed_floor_applied_count": floor_count,
        "min_interval_mean_speed_mps": float(np.sqrt(np.min(speed_sq_raw))),
        "max_interval_mean_speed_mps": float(np.sqrt(np.max(speed_sq_raw))),
    }


def local_acceleration_penalty_block(h: float) -> np.ndarray:
    """Analytic 4x4 Hermite block for integral (f''(t))^2 dt on one interval.

    The local coordinate order is [p_i, v_i, p_{i+1}, v_{i+1}].
    """
    if h <= 0:
        raise ValueError("interval length h must be positive")
    h2 = h * h
    h3 = h2 * h
    return np.array(
        [
            [12.0 / h3, 6.0 / h2, -12.0 / h3, 6.0 / h2],
            [6.0 / h2, 4.0 / h, -6.0 / h2, 2.0 / h],
            [-12.0 / h3, -6.0 / h2, 12.0 / h3, -6.0 / h2],
            [6.0 / h2, 2.0 / h, -6.0 / h2, 4.0 / h],
        ],
        dtype=float,
    )


def assemble_acceleration_penalty(t: np.ndarray, lambda_intervals: np.ndarray) -> np.ndarray:
    """Assemble the global acceleration-penalty matrix Omega."""
    n = t.size
    if lambda_intervals.shape != (n - 1,):
        raise ValueError(f"lambda_intervals must have shape ({n - 1},); got {lambda_intervals.shape}")

    omega = np.zeros((2 * n, 2 * n), dtype=float)
    for i, h in enumerate(np.diff(t)):
        block = float(lambda_intervals[i]) * local_acceleration_penalty_block(float(h))
        idx = np.array([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3], dtype=int)
        omega[np.ix_(idx, idx)] += block

    # Symmetrize to remove harmless round-off in downstream diagnostics.
    return 0.5 * (omega + omega.T)


def assemble_normal_matrix(
    t: np.ndarray,
    lambda_intervals: np.ndarray,
    velocity_weight: float,
    velocity_weight_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble K = D + n*Omega and return K, D_diag, Omega.

    This follows the paper's scaled normal equation:

        (B'B + gamma C'C + n Omega_lambda) theta
            = B'y + gamma C'v.

    In nodal coordinates B and C are selectors, so D has 1 on position rows
    and gamma on velocity rows.  Omega returned by this function is the
    unscaled acceleration-penalty matrix; K uses n*Omega.
    """
    n = t.size
    d_diag = np.empty(2 * n, dtype=float)
    d_diag[0::2] = 1.0
    if velocity_weight_scale is None:
        scale = np.ones(n, dtype=float)
    else:
        scale = np.clip(np.asarray(velocity_weight_scale, dtype=float).reshape(-1), 0.0, 1.0)
        if scale.shape != (n,):
            raise ValueError(f"velocity_weight_scale must have shape ({n},); got {scale.shape}")
    d_diag[1::2] = float(velocity_weight) * scale

    omega = assemble_acceleration_penalty(t, lambda_intervals)
    k_mat = float(n) * omega + np.diag(d_diag)
    return k_mat, d_diag, omega


def _stack_observations(y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Stack observations as [y_0, v_0, y_1, v_1, ...]."""
    z = np.empty((2 * y.shape[0], y.shape[1]), dtype=float)
    z[0::2, :] = y
    z[1::2, :] = v
    return z


def build_endpoint_fixed_constraints(
    y: np.ndarray,
    v: np.ndarray,
    config: VSplineCoreConfig,
    endpoint_constraints: VSplineEndpointConstraints | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build hard endpoint constraints in stacked Hermite coordinates.

    Legacy behavior fixes endpoint values to raw first/last observations.  When
    ``endpoint_constraints`` supplies start/end states, those external values are
    used instead.  This is the hook that lets adjacent dynamic segments share one
    exact boundary position and velocity, producing C1 continuity inside a
    connected component.
    """
    n, d = y.shape
    m = 2 * n
    fixed_mask = np.zeros((m, d), dtype=bool)
    fixed_values = np.zeros((m, d), dtype=float)
    constrained_rows: list[dict[str, Any]] = []

    # Base ``hard_endpoint_constraints=False`` disables legacy raw endpoint
    # constraints.  It must not silently disable external shared-boundary
    # constraints, because that would break piecewise C0/C1 continuity.
    if not config.hard_endpoint_constraints and endpoint_constraints is None:
        return fixed_mask, fixed_values, {
            "enabled": False,
            "fixed_row_count": 0,
            "fixed_scalar_count": 0,
            "rows": [],
        }

    ec = endpoint_constraints or VSplineEndpointConstraints()

    def _state_value(
        state: VSplineEndpointState | None,
        attr: str,
        fallback: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if state is not None:
            value = getattr(state, attr)
            if value is not None:
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.size != d:
                    raise ValueError(f"endpoint {attr} must have dimension {d}; got {arr.size}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"endpoint {attr} contains non-finite values")
                return arr, "external"
        return np.asarray(fallback, dtype=float).reshape(d), "raw_observation"

    def fix_row(row: int, value: np.ndarray, label: str, source: str) -> None:
        fixed_mask[row, :] = True
        fixed_values[row, :] = np.asarray(value, dtype=float).reshape(d)
        constrained_rows.append(
            {
                "row": int(row),
                "label": label,
                "source": source,
                "dimensions_fixed": int(d),
            }
        )

    start_position, start_position_source = _state_value(ec.start, "position", y[0, :])
    end_position, end_position_source = _state_value(ec.end, "position", y[-1, :])
    start_velocity, start_velocity_source = _state_value(ec.start, "velocity", v[0, :])
    end_velocity, end_velocity_source = _state_value(ec.end, "velocity", v[-1, :])

    # External shared-boundary states are stronger than the local/base endpoint
    # policy.  This matters for piecewise reconstruction: adjacent segment fits
    # must receive exactly the same boundary position AND velocity, even if a
    # caller has disabled legacy raw endpoint velocity/position constraints for
    # global endpoints.  Raw fallback endpoint constraints still obey the base
    # config flags, preserving legacy behavior for non-piecewise fits.
    force_start_position = start_position_source == "external"
    force_end_position = end_position_source == "external"
    force_start_velocity = start_velocity_source == "external"
    force_end_velocity = end_velocity_source == "external"

    if ec.hard_start_position and (config.hard_endpoint_positions or force_start_position):
        fix_row(0, start_position, "start_position", start_position_source)
    if ec.hard_end_position and (config.hard_endpoint_positions or force_end_position):
        fix_row(2 * n - 2, end_position, "end_position", end_position_source)

    if ec.hard_start_velocity and (config.hard_endpoint_velocities or force_start_velocity):
        fix_row(1, start_velocity, "start_velocity", start_velocity_source)
    if ec.hard_end_velocity and (config.hard_endpoint_velocities or force_end_velocity):
        fix_row(2 * n - 1, end_velocity, "end_velocity", end_velocity_source)

    return fixed_mask, fixed_values, {
        "enabled": True,
        "hard_endpoint_positions": bool(config.hard_endpoint_positions),
        "hard_endpoint_velocities": bool(config.hard_endpoint_velocities),
        "external_endpoint_constraints": endpoint_constraints is not None,
        "external_constraints_override_base_endpoint_flags": bool(endpoint_constraints is not None),
        "fixed_row_count": int(np.any(fixed_mask, axis=1).sum()),
        "fixed_scalar_count": int(fixed_mask.sum()),
        "rows": constrained_rows,
    }


def solve_with_fixed_entries(
    k_mat: np.ndarray,
    rhs: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
) -> np.ndarray:
    """Solve K theta = rhs subject to selected fixed theta entries.

    For each dimension independently, partition theta into free and fixed
    entries:

        K_FF theta_F + K_FC theta_C = rhs_F

    with theta_C known, so:

        theta_F = K_FF^{-1} (rhs_F - K_FC theta_C).
    """
    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (2n, d)")
    if k_mat.shape != (rhs.shape[0], rhs.shape[0]):
        raise ValueError(f"k_mat must have shape {(rhs.shape[0], rhs.shape[0])}; got {k_mat.shape}")
    if fixed_mask.shape != rhs.shape:
        raise ValueError(f"fixed_mask must have shape {rhs.shape}; got {fixed_mask.shape}")
    if fixed_values.shape != rhs.shape:
        raise ValueError(f"fixed_values must have shape {rhs.shape}; got {fixed_values.shape}")

    if not np.any(fixed_mask):
        return np.linalg.solve(k_mat, rhs)

    m, d = rhs.shape
    theta = np.empty((m, d), dtype=float)

    for j in range(d):
        fixed = fixed_mask[:, j]
        free = ~fixed

        theta[fixed, j] = fixed_values[fixed, j]

        if not np.any(free):
            # All entries are fixed for this dimension.
            continue

        if not np.any(fixed):
            theta[:, j] = np.linalg.solve(k_mat, rhs[:, j])
            continue

        k_ff = k_mat[np.ix_(free, free)]
        k_fc = k_mat[np.ix_(free, fixed)]
        rhs_free = rhs[free, j] - k_fc @ theta[fixed, j]
        theta[free, j] = np.linalg.solve(k_ff, rhs_free)

    return theta


def constrained_normal_equation_relative_residual(
    k_mat: np.ndarray,
    theta: np.ndarray,
    rhs: np.ndarray,
    fixed_mask: np.ndarray,
) -> float:
    """Relative residual of the free normal equations.

    With hard constraints, K theta - rhs is generally nonzero at fixed rows;
    those entries are Lagrange multiplier information, not a solve error. This
    diagnostic therefore checks only free rows.
    """
    residual_full = k_mat @ theta - rhs

    if not np.any(fixed_mask):
        rhs_norm = max(float(np.linalg.norm(rhs)), 1e-30)
        return float(np.linalg.norm(residual_full) / rhs_norm)

    numerator_sq = 0.0
    denominator_sq = 0.0
    for j in range(rhs.shape[1]):
        free = ~fixed_mask[:, j]
        if not np.any(free):
            continue
        numerator_sq += float(np.sum(residual_full[free, j] ** 2))
        denominator_sq += float(np.sum(rhs[free, j] ** 2))

    return float(math.sqrt(numerator_sq) / max(math.sqrt(denominator_sq), 1e-30))


def analytic_loocv_score_from_solution(
    *,
    k_mat: np.ndarray,
    d_diag: np.ndarray,
    residual: np.ndarray,
    velocity_weight: float,
    min_denominator: float,
    fixed_mask: np.ndarray | None = None,
) -> float:
    """Compute scalar leave-one-out CV using smoother diagonals.

    Unconstrained case:
        H = K^{-1} D.

    Constrained case:
        Fixed endpoint rows are excluded from CV. For free rows, the smoother
        diagonal is computed from the free/free block:

            H_FF = K_FF^{-1} D_FF.

    This avoids the pathological denominator 1 - H_ii = 0 for rows that are
    hard-fixed to the raw endpoint data.
    """
    if fixed_mask is None or not np.any(fixed_mask):
        try:
            inv_k = np.linalg.inv(k_mat)
        except np.linalg.LinAlgError:
            return float("inf")

        h_diag = np.diag(inv_k) * d_diag
        den = 1.0 - h_diag

        bad = np.abs(den) < min_denominator
        if np.any(bad):
            den = den.copy()
            den[bad] = np.sign(den[bad]) * min_denominator
            den[den == 0.0] = min_denominator

        scaled = residual / den[:, None]
        pos = scaled[0::2, :]
        vel = scaled[1::2, :]
        n, d = pos.shape
        score = (np.sum(pos * pos) + float(velocity_weight) * np.sum(vel * vel)) / float(n * d)
        return float(score)

    if fixed_mask.shape != residual.shape:
        raise ValueError(f"fixed_mask must have shape {residual.shape}; got {fixed_mask.shape}")

    total = 0.0
    normalizer = 0

    for j in range(residual.shape[1]):
        fixed = fixed_mask[:, j]
        free = ~fixed
        if not np.any(free):
            continue

        try:
            k_ff_inv = np.linalg.inv(k_mat[np.ix_(free, free)])
        except np.linalg.LinAlgError:
            return float("inf")

        d_free = d_diag[free]
        h_diag_free = np.diag(k_ff_inv) * d_free
        den = 1.0 - h_diag_free

        bad = np.abs(den) < min_denominator
        if np.any(bad):
            den = den.copy()
            den[bad] = np.sign(den[bad]) * min_denominator
            den[den == 0.0] = min_denominator

        scaled = residual[free, j] / den
        total += float(np.sum(d_free * scaled * scaled))

        # Same broad scale as the unconstrained score: one denominator unit per
        # free position observation per dimension. Velocity rows contribute to
        # the numerator with gamma, as in the fitting objective.
        normalizer += int(np.sum(free[0::2]))

    if normalizer <= 0:
        return float("inf")

    return float(total / normalizer)



def _maybe_condition_number(k_mat: np.ndarray, max_size: int | None) -> float | None:
    if max_size is not None and k_mat.shape[0] > int(max_size):
        return None
    try:
        return float(np.linalg.cond(k_mat))
    except Exception:
        return None


def solve_fixed_parameters(
    t: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    config: VSplineCoreConfig,
    *,
    compute_loocv: bool = True,
    dim_names: tuple[str, ...] | None = None,
    endpoint_constraints: VSplineEndpointConstraints | None = None,
    velocity_weight_scale: np.ndarray | None = None,
) -> VSplineCoreFit:
    """Fit the V-Spline for fixed tuning parameters."""
    lambda_intervals, lambda_report = compute_interval_weights(t, y, config)
    if velocity_weight_scale is None:
        velocity_weight_scale = np.ones(t.size, dtype=float)
    else:
        velocity_weight_scale = np.clip(np.asarray(velocity_weight_scale, dtype=float).reshape(-1), 0.0, 1.0)
        if velocity_weight_scale.shape != (t.size,):
            raise ValueError(f"velocity_weight_scale must have shape ({t.size},); got {velocity_weight_scale.shape}")
    k_mat, d_diag, omega = assemble_normal_matrix(t, lambda_intervals, config.velocity_weight, velocity_weight_scale)

    z_obs = _stack_observations(y, v)
    rhs = d_diag[:, None] * z_obs

    fixed_mask, fixed_values, constraint_report = build_endpoint_fixed_constraints(
        y,
        v,
        config,
        endpoint_constraints=endpoint_constraints,
    )
    theta = solve_with_fixed_entries(
        k_mat=k_mat,
        rhs=rhs,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )

    fitted = theta
    residual = fitted - z_obs

    solve_relative_residual = constrained_normal_equation_relative_residual(
        k_mat=k_mat,
        theta=theta,
        rhs=rhs,
        fixed_mask=fixed_mask,
    )

    velocity_residual = theta[1::2, :] - v
    objective_data = float(
        np.sum((theta[0::2, :] - y) ** 2)
        + config.velocity_weight * np.sum(velocity_weight_scale[:, None] * (velocity_residual ** 2))
    )
    objective_penalty = float(t.size) * float(np.sum(theta * (omega @ theta)))
    objective_total = objective_data + objective_penalty

    constraint_error = 0.0
    if np.any(fixed_mask):
        constraint_error = float(np.max(np.abs(theta[fixed_mask] - fixed_values[fixed_mask])))

    diagnostics: dict[str, Any] = {
        "method": "paper_oriented_v_spline_nodal_hermite_with_optional_hard_endpoints",
        "n_observations": int(t.size),
        "dimension": int(y.shape[1]),
        "dof_per_dimension": int(2 * t.size),
        "n_basis": int(2 * t.size),
        "knot_rule": "one_knot_at_every_observation_time",
        "basis": "nodal_cubic_hermite",
        "objective": "position_residuals_plus_global_velocity_residuals_plus_n_scaled_integrated_squared_acceleration",
        "min_dt_s": float(np.min(np.diff(t))),
        "max_dt_s": float(np.max(np.diff(t))),
        "min_lambda_interval": float(np.min(lambda_intervals)),
        "max_lambda_interval": float(np.max(lambda_intervals)),
        "velocity_weight": float(config.velocity_weight),
        "velocity_weight_scaling": {
            "enabled": bool(np.any(velocity_weight_scale < 0.999)),
            "min_scale": float(np.min(velocity_weight_scale)) if velocity_weight_scale.size else None,
            "median_scale": float(np.median(velocity_weight_scale)) if velocity_weight_scale.size else None,
            "mean_scale": float(np.mean(velocity_weight_scale)) if velocity_weight_scale.size else None,
            "downweighted_count": int(np.sum(velocity_weight_scale < 0.999)),
        },
        "hard_endpoint_constraints": constraint_report,
        "hard_endpoint_constraint_max_abs_error": constraint_error,
        "lambda_report": lambda_report,
        "normal_matrix_shape": list(k_mat.shape),
        "symmetry_defect": float(np.linalg.norm(k_mat - k_mat.T) / max(np.linalg.norm(k_mat), 1e-30)),
        "condition_number": _maybe_condition_number(k_mat, config.condition_number_max_size),
        "condition_number_max_size": config.condition_number_max_size,
        "solve_relative_residual_free_rows": solve_relative_residual,
        "objective_data": objective_data,
        "objective_penalty": objective_penalty,
        "objective_total": objective_total,
        "position_residual_rms_by_dim": np.sqrt(np.mean((theta[0::2, :] - y) ** 2, axis=0)).tolist(),
        "velocity_residual_rms_by_dim": np.sqrt(np.mean((theta[1::2, :] - v) ** 2, axis=0)).tolist(),
        "endpoint_position_error_start_by_dim": (theta[0, :] - y[0, :]).tolist(),
        "endpoint_velocity_error_start_by_dim": (theta[1, :] - v[0, :]).tolist(),
        "endpoint_position_error_end_by_dim": (theta[-2, :] - y[-1, :]).tolist(),
        "endpoint_velocity_error_end_by_dim": (theta[-1, :] - v[-1, :]).tolist(),
    }

    if compute_loocv:
        diagnostics["loocv_score"] = analytic_loocv_score_from_solution(
            k_mat=k_mat,
            d_diag=d_diag,
            residual=residual,
            velocity_weight=config.velocity_weight,
            min_denominator=config.min_loocv_denominator,
            fixed_mask=fixed_mask,
        )

    if dim_names is None:
        dim_names = tuple(f"x{j}" for j in range(y.shape[1]))

    return VSplineCoreFit(
        t=t.copy(),
        theta_position=theta[0::2, :].copy(),
        theta_velocity=theta[1::2, :].copy(),
        lambda_intervals=lambda_intervals.copy(),
        config=config,
        diagnostics=diagnostics,
        dim_names=tuple(dim_names),
    )


def _core_input_log_packet(core_input: VSplineCoreInput, config: VSplineCoreConfig) -> dict[str, Any]:
    """Small log packet: core input/config only, no raw arrays."""
    t = np.asarray(core_input.t, dtype=float).reshape(-1)
    y = np.asarray(core_input.y, dtype=float)
    v = np.asarray(core_input.v, dtype=float)
    return {
        "n_observations": int(t.size),
        "dimension": int(y.shape[1] if y.ndim > 1 else 1),
        "dim_names": list(core_input.dim_names),
        "t_start_s": float(t[0]) if t.size else None,
        "t_end_s": float(t[-1]) if t.size else None,
        "t_span_s": float(t[-1] - t[0]) if t.size >= 2 else None,
        "penalty_mode": config.penalty_mode,
        "velocity_weight": float(config.velocity_weight),
        "smoothing_lambda": float(config.smoothing_lambda),
        "adaptive_eta": float(config.adaptive_eta),
        "adaptive_speed_floor_mps": config.adaptive_speed_floor_mps,
        "optimize": bool(config.optimize),
        "optimize_velocity_weight": bool(config.optimize_velocity_weight),
        "hard_endpoint_constraints": bool(config.hard_endpoint_constraints),
        "hard_endpoint_positions": bool(config.hard_endpoint_positions),
        "hard_endpoint_velocities": bool(config.hard_endpoint_velocities),
        "compute_loocv_score": bool(config.compute_loocv_score),
        "condition_number_max_size": config.condition_number_max_size,
        "velocity_weight_scale_supplied": bool(core_input.velocity_weight_scale is not None),
    }


def _core_output_log_packet(fit: VSplineCoreFit) -> dict[str, Any]:
    """Small log packet: core output/diagnostic summary only, no coefficient arrays."""
    d = fit.diagnostics
    return {
        "n_observations": fit.n_observations,
        "dimension": fit.dimension,
        "dof_per_dimension": d.get("dof_per_dimension"),
        "selected_penalty_mode": fit.config.penalty_mode,
        "selected_velocity_weight": fit.config.velocity_weight,
        "selected_smoothing_lambda": fit.config.smoothing_lambda,
        "selected_adaptive_eta": fit.config.adaptive_eta,
        "hard_endpoint_constraints": d.get("hard_endpoint_constraints"),
        "hard_endpoint_constraint_max_abs_error": d.get("hard_endpoint_constraint_max_abs_error"),
        "min_lambda_interval": d.get("min_lambda_interval"),
        "max_lambda_interval": d.get("max_lambda_interval"),
        "condition_number": d.get("condition_number"),
        "solve_relative_residual_free_rows": d.get("solve_relative_residual_free_rows"),
        "loocv_score": d.get("loocv_score"),
        "position_residual_rms_by_dim": d.get("position_residual_rms_by_dim"),
        "velocity_residual_rms_by_dim": d.get("velocity_residual_rms_by_dim"),
    }


def fit_v_spline_core(
    core_input: VSplineCoreInput,
    config: VSplineCoreConfig | None = None,
    endpoint_constraints: VSplineEndpointConstraints | None = None,
) -> VSplineCoreFit:
    """Fit a 1D/2D/3D paper-oriented V-Spline.

    Logging policy for the mathematical core is intentionally narrow: report
    only compact input parameters and compact output parameters. Do not log
    raw arrays, coefficient arrays, or aviation/preprocessing details here.
    """
    cfg = config or VSplineCoreConfig()
    checked = validate_core_input(core_input)
    logger.info("V-Spline core input params: {}", _core_input_log_packet(checked, cfg))

    if cfg.optimize:
        cfg = optimize_hyperparameters(checked, cfg)

    fit = solve_fixed_parameters(
        checked.t,
        checked.y,
        checked.v,
        cfg,
        compute_loocv=bool(cfg.compute_loocv_score),
        dim_names=checked.dim_names,
        endpoint_constraints=endpoint_constraints,
        velocity_weight_scale=checked.velocity_weight_scale,
    )
    logger.info("V-Spline core output params: {}", _core_output_log_packet(fit))
    return fit


def optimize_hyperparameters(core_input: VSplineCoreInput, config: VSplineCoreConfig) -> VSplineCoreConfig:
    """Optimize positive tuning parameters by analytic LOOCV.

    This is intentionally conservative: it optimizes in log10 space and returns
    a config with optimize=False so the actual fit is a single fixed-parameter
    solve after parameter selection.
    """
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError("scipy is required for optimize=True") from exc

    checked = validate_core_input(core_input)

    log_gamma0 = math.log10(max(config.velocity_weight, 1e-300))
    if config.penalty_mode == "constant":
        log_lambda0 = math.log10(max(config.smoothing_lambda, 1e-300))
    else:
        log_lambda0 = math.log10(max(config.adaptive_eta, 1e-300))

    x0 = [log_lambda0]
    bounds = [config.log10_lambda_bounds]
    if config.optimize_velocity_weight:
        x0.insert(0, log_gamma0)
        bounds.insert(0, config.log10_velocity_weight_bounds)

    def unpack(x: np.ndarray) -> VSplineCoreConfig:
        if config.optimize_velocity_weight:
            gamma = 10.0 ** float(x[0])
            lam_or_eta = 10.0 ** float(x[1])
        else:
            gamma = config.velocity_weight
            lam_or_eta = 10.0 ** float(x[0])

        if config.penalty_mode == "constant":
            return replace(
                config,
                velocity_weight=gamma,
                smoothing_lambda=lam_or_eta,
                optimize=False,
            )

        return replace(
            config,
            velocity_weight=gamma,
            adaptive_eta=lam_or_eta,
            optimize=False,
        )

    def objective(x: np.ndarray) -> float:
        trial = unpack(x)
        try:
            fit = solve_fixed_parameters(
                checked.t,
                checked.y,
                checked.v,
                trial,
                compute_loocv=True,
                dim_names=checked.dim_names,
            )
            score = float(fit.diagnostics.get("loocv_score", float("inf")))
            if not np.isfinite(score):
                return 1e300
            return score
        except Exception:
            return 1e300

    res = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100},
    )

    if not res.success:
        warnings.warn(f"V-Spline LOOCV optimization did not fully converge: {res.message}", RuntimeWarning)

    best = unpack(np.asarray(res.x, dtype=float))
    return replace(best, optimize=False)


def evaluate_hermite(
    t_eval: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    *,
    deriv: int = 0,
) -> np.ndarray:
    """Evaluate a nodal cubic Hermite spline with linear exterior behavior."""
    if deriv not in (0, 1, 2, 3):
        raise ValueError("deriv must be 0, 1, 2, or 3")

    te = np.asarray(t_eval, dtype=float).reshape(-1)
    out = np.empty((te.size, p.shape[1]), dtype=float)

    # Left exterior: linear with first nodal velocity.
    left = te < t[0]
    if np.any(left):
        if deriv == 0:
            out[left, :] = p[0, :] + (te[left] - t[0])[:, None] * v[0, :]
        elif deriv == 1:
            out[left, :] = v[0, :]
        else:
            out[left, :] = 0.0

    # Right exterior: linear with last nodal velocity.
    right = te > t[-1]
    if np.any(right):
        if deriv == 0:
            out[right, :] = p[-1, :] + (te[right] - t[-1])[:, None] * v[-1, :]
        elif deriv == 1:
            out[right, :] = v[-1, :]
        else:
            out[right, :] = 0.0

    inside = ~(left | right)
    if np.any(inside):
        x = te[inside]
        # side="right" maps exact knots to the interval ending at that knot,
        # except the first knot; then clip to valid interval indices.
        idx = np.searchsorted(t, x, side="right") - 1
        idx = np.clip(idx, 0, t.size - 2)

        t0 = t[idx]
        t1 = t[idx + 1]
        h = t1 - t0
        s = (x - t0) / h

        p0 = p[idx, :]
        p1 = p[idx + 1, :]
        v0 = v[idx, :]
        v1 = v[idx + 1, :]

        if deriv == 0:
            h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
            h10 = s**3 - 2.0 * s**2 + s
            h01 = -2.0 * s**3 + 3.0 * s**2
            h11 = s**3 - s**2
            out[inside, :] = (
                h00[:, None] * p0
                + (h * h10)[:, None] * v0
                + h01[:, None] * p1
                + (h * h11)[:, None] * v1
            )
        elif deriv == 1:
            h00 = 6.0 * s**2 - 6.0 * s
            h10 = 3.0 * s**2 - 4.0 * s + 1.0
            h01 = -6.0 * s**2 + 6.0 * s
            h11 = 3.0 * s**2 - 2.0 * s
            out[inside, :] = (
                (h00 / h)[:, None] * p0
                + h10[:, None] * v0
                + (h01 / h)[:, None] * p1
                + h11[:, None] * v1
            )
        elif deriv == 2:
            h00 = 12.0 * s - 6.0
            h10 = 6.0 * s - 4.0
            h01 = -12.0 * s + 6.0
            h11 = 6.0 * s - 2.0
            out[inside, :] = (
                (h00 / (h * h))[:, None] * p0
                + (h10 / h)[:, None] * v0
                + (h01 / (h * h))[:, None] * p1
                + (h11 / h)[:, None] * v1
            )
        else:
            h00 = np.full_like(s, 12.0)
            h10 = np.full_like(s, 6.0)
            h01 = np.full_like(s, -12.0)
            h11 = np.full_like(s, 6.0)
            out[inside, :] = (
                (h00 / (h * h * h))[:, None] * p0
                + (h10 / (h * h))[:, None] * v0
                + (h01 / (h * h * h))[:, None] * p1
                + (h11 / (h * h))[:, None] * v1
            )

    return out


def objective_value(fit: VSplineCoreFit, y: np.ndarray, v_obs: np.ndarray) -> float:
    """Compute the fixed-parameter objective for an existing fit."""
    y = np.asarray(y, dtype=float)
    v_obs = np.asarray(v_obs, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if v_obs.ndim == 1:
        v_obs = v_obs.reshape(-1, 1)
    if y.shape != fit.theta_position.shape or v_obs.shape != fit.theta_velocity.shape:
        raise ValueError(
            "y and v_obs must match fit theta shapes; "
            f"got y={y.shape}, v_obs={v_obs.shape}, "
            f"theta_position={fit.theta_position.shape}, theta_velocity={fit.theta_velocity.shape}"
        )

    data = np.sum((fit.theta_position - y) ** 2) + fit.config.velocity_weight * np.sum(
        (fit.theta_velocity - v_obs) ** 2
    )

    theta = np.empty((2 * fit.n_observations, fit.dimension), dtype=float)
    theta[0::2, :] = fit.theta_position
    theta[1::2, :] = fit.theta_velocity
    omega = assemble_acceleration_penalty(fit.t, fit.lambda_intervals)
    penalty = fit.n_observations * np.sum(theta * (omega @ theta))
    return float(data + penalty)


if __name__ == "__main__":  # Small smoke test.
    t = np.linspace(0.0, 10.0, 11)
    y = np.column_stack([np.sin(t), np.cos(t)])
    v = np.column_stack([np.cos(t), -np.sin(t)])
    inp = VSplineCoreInput(t=t, y=y, v=v, dim_names=("x", "y"))
    cfg = VSplineCoreConfig(smoothing_lambda=0.1, velocity_weight=1.0)
    fit = fit_v_spline_core(inp, cfg)

    assert np.allclose(fit.theta_position[0], y[0])
    assert np.allclose(fit.theta_velocity[0], v[0])
    assert np.allclose(fit.theta_position[-1], y[-1])
    assert np.allclose(fit.theta_velocity[-1], v[-1])
    print("Smoke test passed")
