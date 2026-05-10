"""Component-level B-spline trajectory solver with V-Spline-like penalties.

This module fits local clamped B-spline curves with V-Spline-style position,
velocity, and derivative penalties.  The production pipeline fits one local
curve per dynamic segment.  Legacy variants can couple neighbouring segments
through hard raw boundary anchors; aviation-adapted variants use robust soft
boundary-state priors, velocity confidence scaling, and higher-order derivative
penalties while remaining segmented/local.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

import math
import warnings

import numpy as np

from .velocity_confidence import compute_velocity_confidence_scale

try:  # scipy is expected in the reconstruction environment.
    from scipy import sparse
    from scipy.interpolate import BSpline
    from scipy.sparse.linalg import spsolve
except Exception as exc:  # pragma: no cover - fail loudly at construction time.
    sparse = None  # type: ignore[assignment]
    BSpline = None  # type: ignore[assignment]
    spsolve = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

try:  # Keep the core usable even when loguru is not installed.
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

PenaltyMode = Literal["constant", "adaptive"]


def compute_interval_weights(t: np.ndarray, y: np.ndarray, config: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute V-Spline interval acceleration weights for constant/adaptive modes."""
    if float(config.velocity_weight) <= 0:
        raise ValueError("velocity_weight must be positive")

    h = np.diff(np.asarray(t, dtype=float))
    if np.any(h <= 0):
        raise ValueError("times must be strictly increasing")

    if config.penalty_mode == "constant":
        if float(config.smoothing_lambda) <= 0:
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
    if float(config.adaptive_eta) <= 0:
        raise ValueError("adaptive_eta must be positive in adaptive mode")

    y_arr = np.asarray(y, dtype=float)
    mean_velocity = np.diff(y_arr, axis=0) / h[:, None]
    speed_sq_raw = np.sum(mean_velocity * mean_velocity, axis=1)
    speed_sq = speed_sq_raw.copy()

    floor = config.adaptive_speed_floor_mps
    floor_count = 0
    if floor is not None:
        if float(floor) < 0:
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


BoundaryAnchorSource = Literal[
    "raw_segment_boundary_sample",
    "component_endpoint_sample",
    "shared_boundary_state",
    "manual",
]


@dataclass(frozen=True)
class BSplineCoreConfig:
    """Configuration for the component-level B-spline V-Spline backend.

    The defaults are conservative for ADS-B: cubic splines give C2 continuity at
    simple knots, position anchors are hard only at selected boundary samples,
    velocity is soft, and the V-Spline adaptive acceleration penalty is retained
    as the primary smoothness term.
    """

    degree: int = 3
    knot_spacing_s: float = 5.0
    min_knot_spacing_s: float = 1.0
    max_basis_count: int | None = 900
    # Avoid high-flexibility fits on short/noisy components.  For example, 27
    # observations with cubic B-splines should not get 11 basis functions.
    min_observations_per_basis: float = 8.0
    add_knots_at_anchors: bool = True

    position_weight: float = 1.0
    velocity_weight: float = 0.03
    # ADS-B velocity rows can be stale, mismatched to position rows, or simply
    # wrong.  Gate velocity observations against local position-derived velocity
    # before they are allowed to shape curvature.
    velocity_outlier_policy: Literal["none", "position_difference_gate"] = "position_difference_gate"
    velocity_outlier_gate_mps: float = 80.0
    min_velocity_weight_scale: float = 0.0
    # Optional caller-supplied velocity confidence scaling is multiplied with
    # the internal ADS-B consistency gate.  This keeps reported velocity in the
    # objective while preventing stale/asynchronous velocity rows from dominating
    # curvature.
    use_velocity_confidence_scaling: bool = True

    # Soft boundary position priors are used by aviation-adapted segmented
    # variants when hard raw boundary anchoring is disabled.  They target the
    # robust shared boundary state rather than a single ADS-B sample.
    boundary_position_prior_weight: float = 0.0
    scale_boundary_position_prior_by_confidence: bool = True

    boundary_velocity_prior_weight: float = 0.5
    scale_boundary_velocity_prior_by_confidence: bool = True
    # Optional soft C2 stitching term for independent local segment solves.  The
    # acceleration priors are usually estimated by the shared-boundary state
    # model from samples on both sides of a join.  They are deliberately soft:
    # ADS-B does not observe acceleration directly, and exact C2 constraints can
    # over-constrain short segments.
    boundary_acceleration_prior_weight: float = 0.0
    scale_boundary_acceleration_prior_by_confidence: bool = True

    penalty_mode: PenaltyMode = "adaptive"
    smoothing_lambda: float = 1.0
    adaptive_eta: float = 1e5
    adaptive_speed_floor_mps: float | None = 1.0
    acceleration_penalty_multiplier: float = 1.0
    jerk_penalty_weight: float = 0.0
    snap_penalty_weight: float = 0.0

    hard_boundary_positions: bool = True
    # Component endpoints are position anchors by default, but ADS-B-reported
    # endpoint velocity is no longer trusted as an exact derivative constraint.
    # Flight 4BAAD9 exposed a large quintic end-of-track jerk artifact caused by
    # satisfying stale/noisy endpoint velocity while also fitting dense local
    # position samples.
    hard_component_endpoint_positions: bool = False
    hard_component_endpoint_velocities: bool = False
    component_endpoint_velocity_prior_weight: float = 0.0
    component_endpoint_acceleration_prior_weight: float = 0.0

    # Optional derivative damping near true component start/end times.  The
    # pipeline supplies the guarded absolute times in BSplineCoreInput.metadata
    # so internal overlap-save segment joins do not get confused with true track
    # endpoints unless explicitly requested by the caller.
    endpoint_guard_window_s: float = 0.0
    endpoint_jerk_penalty_multiplier: float = 1.0
    endpoint_snap_penalty_multiplier: float = 1.0

    robust_position_loss: Literal["none", "huber"] = "huber"
    huber_delta_m: float = 80.0
    robust_iterations: int = 2
    min_position_weight_scale: float = 0.05

    # 4BAAD9 report action: the old 1e-9 ridge was too small to be useful as a
    # numerical guard on short/high-transition local solves.  Keep it modest so
    # the fit remains data-driven, but make the default visible in diagnostics.
    solver_ridge: float = 1e-7
    quadrature_order: Literal[2, 3] = 3
    # 0 disabled Hessian condition reporting, which made the B-spline path much
    # less observable than the Hermite path.  Compute condition numbers for
    # practical local systems by default; very large systems still skip the
    # expensive dense diagnostic.
    condition_number_max_basis: int = 256

    # Explicit solver label stored in diagnostics and debug artifacts.
    backend_name: str = "component_global_b_spline_v_spline_penalty"


@dataclass(frozen=True)
class BSplineAnchor:
    """Exact position interpolation constraint."""

    anchor_id: str
    t: float
    position: np.ndarray | Sequence[float]
    source: BoundaryAnchorSource = "manual"
    sample_index: int | None = None
    metadata: dict[str, Any] | None = None

    def position_array(self, dim: int) -> np.ndarray:
        arr = np.asarray(self.position, dtype=float).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"anchor {self.anchor_id!r} position has dim {arr.size}; expected {dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"anchor {self.anchor_id!r} contains non-finite position values")
        return arr

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "t": float(self.t),
            "position_m": np.asarray(self.position, dtype=float).reshape(-1).tolist(),
            "source": self.source,
            "sample_index": self.sample_index,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class BSplinePositionPrior:
    """Soft boundary/external position prior for robust segmented joins."""

    prior_id: str
    t: float
    position: np.ndarray | Sequence[float]
    weight: float | None = None
    confidence: float | None = None
    source: str = "boundary_state"
    metadata: dict[str, Any] | None = None

    def position_array(self, dim: int) -> np.ndarray:
        arr = np.asarray(self.position, dtype=float).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"position prior {self.prior_id!r} has dim {arr.size}; expected {dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"position prior {self.prior_id!r} contains non-finite values")
        return arr

    def as_dict(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "t": float(self.t),
            "position_m": np.asarray(self.position, dtype=float).reshape(-1).tolist(),
            "weight": self.weight,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class BSplineVelocityPrior:
    """Soft boundary or external velocity prior."""

    prior_id: str
    t: float
    velocity: np.ndarray | Sequence[float]
    weight: float | None = None
    confidence: float | None = None
    source: str = "boundary_state"
    metadata: dict[str, Any] | None = None

    def velocity_array(self, dim: int) -> np.ndarray:
        arr = np.asarray(self.velocity, dtype=float).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"velocity prior {self.prior_id!r} has dim {arr.size}; expected {dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"velocity prior {self.prior_id!r} contains non-finite values")
        return arr

    def as_dict(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "t": float(self.t),
            "velocity_mps": np.asarray(self.velocity, dtype=float).reshape(-1).tolist(),
            "weight": self.weight,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class BSplineAccelerationPrior:
    """Soft boundary acceleration prior used to reduce C2 jumps at joins."""

    prior_id: str
    t: float
    acceleration: np.ndarray | Sequence[float]
    weight: float | None = None
    confidence: float | None = None
    source: str = "boundary_state"
    metadata: dict[str, Any] | None = None

    def acceleration_array(self, dim: int) -> np.ndarray:
        arr = np.asarray(self.acceleration, dtype=float).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"acceleration prior {self.prior_id!r} has dim {arr.size}; expected {dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"acceleration prior {self.prior_id!r} contains non-finite values")
        return arr

    def as_dict(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "t": float(self.t),
            "acceleration_mps2": np.asarray(self.acceleration, dtype=float).reshape(-1).tolist(),
            "weight": self.weight,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class BSplineVelocityConstraint:
    """Exact velocity equality constraint, usually at a local segment endpoint."""

    constraint_id: str
    t: float
    velocity: np.ndarray | Sequence[float]
    source: str = "raw_segment_endpoint_sample"
    sample_index: int | None = None
    metadata: dict[str, Any] | None = None

    def velocity_array(self, dim: int) -> np.ndarray:
        arr = np.asarray(self.velocity, dtype=float).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"velocity constraint {self.constraint_id!r} has dim {arr.size}; expected {dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"velocity constraint {self.constraint_id!r} contains non-finite values")
        return arr

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "t": float(self.t),
            "velocity_mps": np.asarray(self.velocity, dtype=float).reshape(-1).tolist(),
            "source": self.source,
            "sample_index": self.sample_index,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class BSplineCoreInput:
    """Input for one hard-gap connected component."""

    t: np.ndarray
    y: np.ndarray
    v: np.ndarray
    dim_names: tuple[str, ...] = ("x", "y", "z")
    anchors: tuple[BSplineAnchor, ...] = ()
    position_priors: tuple[BSplinePositionPrior, ...] = ()
    velocity_priors: tuple[BSplineVelocityPrior, ...] = ()
    acceleration_priors: tuple[BSplineAccelerationPrior, ...] = ()
    velocity_constraints: tuple[BSplineVelocityConstraint, ...] = ()
    velocity_weight_scale: np.ndarray | Sequence[float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class BSplineCoreFit:
    """Fitted component-level B-spline trajectory."""

    t: np.ndarray
    t_origin: float
    knots: np.ndarray
    coefficients: np.ndarray
    lambda_intervals: np.ndarray
    config: BSplineCoreConfig
    diagnostics: dict[str, Any]
    dim_names: tuple[str, ...] = ("x", "y", "z")

    @property
    def n_observations(self) -> int:
        return int(self.t.size)

    @property
    def dimension(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def degree(self) -> int:
        return int(self.config.degree)

    @property
    def basis(self) -> str:
        return f"clamped_degree_{self.degree}_b_spline"

    def evaluate(self, t_eval: np.ndarray | list[float] | float, deriv: int = 0) -> np.ndarray:
        """Evaluate position, velocity, acceleration, or jerk.

        ``t_eval`` uses the same absolute seconds scale as input observations.
        Outside the component range, position/velocity are linearly extrapolated
        from endpoint states and higher derivatives are zero.  The pipeline only
        renders inside the component, but this behavior prevents accidental NaNs
        in diagnostics.
        """
        if deriv not in (0, 1, 2, 3):
            raise ValueError("deriv must be 0, 1, 2, or 3")
        te_abs = np.asarray(t_eval, dtype=float).reshape(-1)
        tau = te_abs - float(self.t_origin)
        t0 = float(self.t[0] - self.t_origin)
        t1 = float(self.t[-1] - self.t_origin)
        out = np.empty((te_abs.size, self.dimension), dtype=float)

        left = tau < t0
        right = tau > t1
        inside = ~(left | right)

        if np.any(inside):
            out[inside, :] = _evaluate_basis(
                tau[inside],
                self.knots,
                self.degree,
                deriv=deriv,
            ) @ self.coefficients

        if np.any(left) or np.any(right):
            p0 = (_evaluate_basis([t0], self.knots, self.degree, deriv=0) @ self.coefficients)[0]
            v0 = (_evaluate_basis([t0], self.knots, self.degree, deriv=1) @ self.coefficients)[0]
            p1 = (_evaluate_basis([t1], self.knots, self.degree, deriv=0) @ self.coefficients)[0]
            v1 = (_evaluate_basis([t1], self.knots, self.degree, deriv=1) @ self.coefficients)[0]
            if np.any(left):
                if deriv == 0:
                    out[left, :] = p0 + (tau[left] - t0)[:, None] * v0
                elif deriv == 1:
                    out[left, :] = v0
                else:
                    out[left, :] = 0.0
            if np.any(right):
                if deriv == 0:
                    out[right, :] = p1 + (tau[right] - t1)[:, None] * v1
                elif deriv == 1:
                    out[right, :] = v1
                else:
                    out[right, :] = 0.0
        return out

    def segment_view(
        self,
        *,
        segment_id: str,
        t_segment: np.ndarray | Sequence[float],
        extra_diagnostics: dict[str, Any] | None = None,
    ) -> "BSplineSegmentFitView":
        return BSplineSegmentFitView(
            base_fit=self,
            segment_id=segment_id,
            t=np.asarray(t_segment, dtype=float).reshape(-1),
            diagnostics={
                "method": "component_global_b_spline_segment_view",
                "basis": self.basis,
                "parent_component_fit": self.diagnostics.get("component_id"),
                "parent_diagnostics_summary": {
                    "n_basis": self.diagnostics.get("n_basis"),
                    "n_anchors": self.diagnostics.get("n_anchors"),
                    "max_anchor_error_m": self.diagnostics.get("max_anchor_error_m"),
                    "position_residual_rmse_3d_m": self.diagnostics.get("position_residual_rmse_3d_m"),
                    "velocity_residual_rmse_3d_mps": self.diagnostics.get("velocity_residual_rmse_3d_mps"),
                    "accel_rms_mps2": self.diagnostics.get("accel_rms_mps2"),
                    "jerk_rms_mps3": self.diagnostics.get("jerk_rms_mps3"),
                },
                **(extra_diagnostics or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t.tolist(),
            "t_origin": float(self.t_origin),
            "knots": self.knots.tolist(),
            "degree": int(self.degree),
            "coefficients": self.coefficients.tolist(),
            "lambda_intervals": self.lambda_intervals.tolist(),
            "config": asdict(self.config),
            "diagnostics": self.diagnostics,
            "dim_names": list(self.dim_names),
        }


@dataclass
class BSplineSegmentFitView:
    """A segment-window view of a component-global B-spline fit."""

    base_fit: BSplineCoreFit
    segment_id: str
    t: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def config(self) -> BSplineCoreConfig:
        return self.base_fit.config

    @property
    def dim_names(self) -> tuple[str, ...]:
        return self.base_fit.dim_names

    @property
    def n_observations(self) -> int:
        return int(self.t.size)

    @property
    def dimension(self) -> int:
        return self.base_fit.dimension

    @property
    def lambda_intervals(self) -> np.ndarray:
        # Report only intervals whose endpoints lie in this view.
        if self.base_fit.lambda_intervals.size == 0 or self.base_fit.t.size < 2 or self.t.size < 2:
            return np.zeros(0, dtype=float)
        t0 = float(np.min(self.t))
        t1 = float(np.max(self.t))
        left = self.base_fit.t[:-1]
        right = self.base_fit.t[1:]
        mask = (left >= t0 - 1e-9) & (right <= t1 + 1e-9)
        return self.base_fit.lambda_intervals[mask]

    def evaluate(self, t_eval: np.ndarray | list[float] | float, deriv: int = 0) -> np.ndarray:
        return self.base_fit.evaluate(t_eval, deriv=deriv)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "parent_basis": self.base_fit.basis,
            "t": self.t.tolist(),
            "diagnostics": self.diagnostics,
        }


def validate_b_spline_input(core_input: BSplineCoreInput) -> BSplineCoreInput:
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
        raise ValueError(f"y and v must have the same shape; got {y.shape} and {v.shape}")
    if y.shape[0] != t.size:
        raise ValueError(f"t length must match y/v rows; got len(t)={t.size}, y rows={y.shape[0]}")
    if t.size < 2:
        raise ValueError("at least two paired observations are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(v)):
        raise ValueError("t, y, and v must contain only finite numeric values")
    if not np.all(np.diff(t) > 0.0):
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
    return replace(core_input, t=t, y=y, v=v, velocity_weight_scale=velocity_weight_scale, dim_names=tuple(dim_names))


def fit_b_spline_component(
    core_input: BSplineCoreInput,
    config: BSplineCoreConfig | None = None,
    *,
    component_id: str | None = None,
) -> BSplineCoreFit:
    """Fit one component-global constrained B-spline trajectory."""
    if _SCIPY_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError("scipy is required for the B-spline backend") from _SCIPY_IMPORT_ERROR
    cfg = config or BSplineCoreConfig()
    checked = validate_b_spline_input(core_input)
    if cfg.degree < 3:
        raise ValueError("degree must be at least 3 for acceleration-continuous cubic-or-better fitting")
    if cfg.knot_spacing_s <= 0 or cfg.min_knot_spacing_s <= 0:
        raise ValueError("knot spacing values must be positive")
    if cfg.min_observations_per_basis <= 0:
        raise ValueError("min_observations_per_basis must be positive")
    if cfg.position_weight <= 0:
        raise ValueError("position_weight must be positive")
    if (
        cfg.velocity_weight < 0
        or cfg.boundary_velocity_prior_weight < 0
        or cfg.component_endpoint_velocity_prior_weight < 0
    ):
        raise ValueError("velocity weights must be non-negative")
    if cfg.boundary_position_prior_weight < 0:
        raise ValueError("position prior weights must be non-negative")
    if cfg.boundary_acceleration_prior_weight < 0 or cfg.component_endpoint_acceleration_prior_weight < 0:
        raise ValueError("acceleration prior weights must be non-negative")
    if cfg.velocity_outlier_gate_mps <= 0:
        raise ValueError("velocity_outlier_gate_mps must be positive")
    if cfg.min_velocity_weight_scale < 0 or cfg.min_velocity_weight_scale > 1:
        raise ValueError("min_velocity_weight_scale must be in [0, 1]")
    if cfg.acceleration_penalty_multiplier < 0 or cfg.jerk_penalty_weight < 0 or cfg.snap_penalty_weight < 0:
        raise ValueError("penalty weights must be non-negative")
    if cfg.endpoint_guard_window_s < 0:
        raise ValueError("endpoint_guard_window_s must be non-negative")
    if cfg.endpoint_jerk_penalty_multiplier < 0 or cfg.endpoint_snap_penalty_multiplier < 0:
        raise ValueError("endpoint penalty multipliers must be non-negative")

    t_abs = checked.t
    y = checked.y
    v = checked.v
    n, dim = y.shape
    t_origin = float(t_abs[0])
    tau = t_abs - t_origin

    raw_anchors = _defaulted_anchors(checked, cfg)
    anchors, anchor_dedupe_report = _dedupe_anchors(raw_anchors, dim=dim)
    position_priors = _validated_position_priors(checked.position_priors, dim=dim)
    velocity_priors = _validated_velocity_priors(checked.velocity_priors, dim=dim)
    acceleration_priors = _validated_acceleration_priors(checked.acceleration_priors, dim=dim)
    velocity_constraints = _validated_velocity_constraints(checked.velocity_constraints, dim=dim)
    extra_knots = []
    if cfg.add_knots_at_anchors:
        extra_knots.extend(float(a.t) - t_origin for a in anchors)
        extra_knots.extend(float(p.t) - t_origin for p in position_priors)
        extra_knots.extend(float(c.t) - t_origin for c in velocity_constraints)
        extra_knots.extend(float(p.t) - t_origin for p in acceleration_priors)
    knots, knot_report = build_knot_vector(tau, cfg, extra_internal_knots=extra_knots)
    degree = int(cfg.degree)
    n_basis = int(len(knots) - degree - 1)

    lambda_cfg = _lambda_proxy_config(cfg)
    lambda_intervals, lambda_report = compute_interval_weights(tau, y, lambda_cfg)
    lambda_intervals = np.asarray(lambda_intervals, dtype=float) * float(cfg.acceleration_penalty_multiplier)

    pos_weight_scale = np.ones(n, dtype=float)
    velocity_weight_scale, velocity_gate_report = _velocity_observation_weight_scale(
        tau=tau,
        y=y,
        v=v,
        config=cfg,
        caller_scale=checked.velocity_weight_scale,
    )
    irls_reports: list[dict[str, Any]] = []
    coeff = np.zeros((n_basis, dim), dtype=float)
    solve_report: dict[str, Any] = {}

    n_iters = 1
    if cfg.robust_position_loss == "huber":
        n_iters = max(1, int(cfg.robust_iterations) + 1)

    endpoint_guard_times_tau = _endpoint_guard_times_tau(checked.metadata, t_origin=t_origin, tau=tau)

    for iteration in range(n_iters):
        coeff, solve_report = _solve_b_spline_once(
            tau=tau,
            y=y,
            v=v,
            anchors=anchors,
            position_priors=position_priors,
            velocity_priors=velocity_priors,
            acceleration_priors=acceleration_priors,
            velocity_constraints=velocity_constraints,
            knots=knots,
            config=cfg,
            lambda_intervals=lambda_intervals,
            position_weight_scale=pos_weight_scale,
            velocity_weight_scale=velocity_weight_scale,
            endpoint_guard_times_tau=endpoint_guard_times_tau,
            t_origin=t_origin,
        )
        fitted_y = _evaluate_basis(tau, knots, degree, deriv=0) @ coeff
        residual_norm = np.linalg.norm(fitted_y - y, axis=1)
        if cfg.robust_position_loss != "huber" or iteration >= n_iters - 1:
            break
        delta = max(float(cfg.huber_delta_m), 1e-9)
        new_scale = np.minimum(1.0, delta / np.maximum(residual_norm, delta))
        new_scale = np.maximum(new_scale, float(cfg.min_position_weight_scale))
        irls_reports.append(
            {
                "iteration": iteration + 1,
                "max_position_residual_m": float(np.max(residual_norm)),
                "median_position_residual_m": float(np.median(residual_norm)),
                "min_weight_scale": float(np.min(new_scale)),
                "downweighted_count": int(np.sum(new_scale < 0.999)),
            }
        )
        pos_weight_scale = new_scale

    fit = BSplineCoreFit(
        t=t_abs.copy(),
        t_origin=t_origin,
        knots=knots.copy(),
        coefficients=coeff.copy(),
        lambda_intervals=lambda_intervals.copy(),
        config=cfg,
        diagnostics={},
        dim_names=checked.dim_names,
    )

    diagnostics = _fit_diagnostics(
        fit=fit,
        core_input=checked,
        anchors=anchors,
        position_priors=position_priors,
        raw_anchor_count=len(raw_anchors),
        anchor_dedupe_report=anchor_dedupe_report,
        velocity_priors=velocity_priors,
        acceleration_priors=acceleration_priors,
        velocity_constraints=velocity_constraints,
        velocity_gate_report=velocity_gate_report,
        knot_report=knot_report,
        lambda_report=lambda_report,
        solve_report=solve_report,
        irls_reports=irls_reports,
        position_weight_scale=pos_weight_scale,
        component_id=component_id,
    )
    fit.diagnostics = diagnostics
    logger.info(
        "B-spline component fit: {}",
        {
            "component_id": component_id,
            "n_observations": n,
            "n_basis": n_basis,
            "n_anchors": len(anchors),
            "degree": degree,
            "max_anchor_error_m": diagnostics.get("max_anchor_error_m"),
            "position_rmse_3d_m": diagnostics.get("position_residual_rmse_3d_m"),
            "accel_rms_mps2": diagnostics.get("accel_rms_mps2"),
        },
    )
    return fit


def _lambda_proxy_config(config: BSplineCoreConfig) -> Any:
    # compute_interval_weights needs only these fields and velocity_weight > 0.
    @dataclass(frozen=True)
    class _Proxy:
        velocity_weight: float
        penalty_mode: PenaltyMode
        smoothing_lambda: float
        adaptive_eta: float
        adaptive_speed_floor_mps: float | None

    return _Proxy(
        velocity_weight=max(float(config.velocity_weight), 1e-12),
        penalty_mode=config.penalty_mode,
        smoothing_lambda=float(config.smoothing_lambda),
        adaptive_eta=float(config.adaptive_eta),
        adaptive_speed_floor_mps=config.adaptive_speed_floor_mps,
    )


def _defaulted_anchors(core_input: BSplineCoreInput, config: BSplineCoreConfig) -> list[BSplineAnchor]:
    anchors = list(core_input.anchors)
    if config.hard_component_endpoint_positions:
        anchors.append(
            BSplineAnchor(
                anchor_id="component_start",
                t=float(core_input.t[0]),
                position=np.asarray(core_input.y[0], dtype=float),
                source="component_endpoint_sample",
                sample_index=0,
            )
        )
        anchors.append(
            BSplineAnchor(
                anchor_id="component_end",
                t=float(core_input.t[-1]),
                position=np.asarray(core_input.y[-1], dtype=float),
                source="component_endpoint_sample",
                sample_index=int(core_input.t.size - 1),
            )
        )
    if not config.hard_boundary_positions:
        anchors = [a for a in anchors if a.source == "component_endpoint_sample"]
    return anchors


def _dedupe_anchors(anchors: Sequence[BSplineAnchor], *, dim: int, tol_s: float = 1e-8) -> tuple[list[BSplineAnchor], dict[str, Any]]:
    if not anchors:
        return [], {"input_count": 0, "deduped_count": 0, "duplicates_removed": 0, "conflicts": []}
    ordered = sorted(anchors, key=lambda a: (float(a.t), str(a.anchor_id)))
    out: list[BSplineAnchor] = []
    conflicts: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    for anchor in ordered:
        pos = anchor.position_array(dim)
        if out and abs(float(anchor.t) - float(out[-1].t)) <= tol_s:
            prev = out[-1]
            prev_pos = prev.position_array(dim)
            err = float(np.linalg.norm(pos - prev_pos))
            duplicate_ids.append(str(anchor.anchor_id))
            if err > 1e-6:
                conflicts.append(
                    {
                        "kept_anchor_id": prev.anchor_id,
                        "dropped_anchor_id": anchor.anchor_id,
                        "t": float(anchor.t),
                        "position_disagreement_m": err,
                    }
                )
            continue
        out.append(anchor)
    return out, {
        "input_count": len(anchors),
        "deduped_count": len(out),
        "duplicates_removed": len(anchors) - len(out),
        "duplicate_anchor_ids_removed": duplicate_ids,
        "conflicts": conflicts,
    }


def _validated_position_priors(priors: Sequence[BSplinePositionPrior], *, dim: int) -> list[BSplinePositionPrior]:
    out: list[BSplinePositionPrior] = []
    for p in priors:
        p.position_array(dim)
        out.append(p)
    return out


def _validated_velocity_priors(priors: Sequence[BSplineVelocityPrior], *, dim: int) -> list[BSplineVelocityPrior]:
    out: list[BSplineVelocityPrior] = []
    for prior in priors:
        _ = prior.velocity_array(dim)
        out.append(prior)
    return out


def _validated_acceleration_priors(priors: Sequence[BSplineAccelerationPrior], *, dim: int) -> list[BSplineAccelerationPrior]:
    out: list[BSplineAccelerationPrior] = []
    for prior in priors:
        _ = prior.acceleration_array(dim)
        out.append(prior)
    return out



def _validated_velocity_constraints(constraints: Sequence[BSplineVelocityConstraint], *, dim: int) -> list[BSplineVelocityConstraint]:
    out: list[BSplineVelocityConstraint] = []
    for constraint in constraints:
        _ = constraint.velocity_array(dim)
        out.append(constraint)
    return out


def build_knot_vector(
    t_local: np.ndarray,
    config: BSplineCoreConfig,
    *,
    extra_internal_knots: Sequence[float] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a clamped, simple-knot B-spline knot vector in local seconds."""
    t = np.asarray(t_local, dtype=float).reshape(-1)
    if t.size < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("t_local must be strictly increasing")
    degree = int(config.degree)
    t0 = float(t[0])
    t1 = float(t[-1])
    span = t1 - t0
    if span <= 0:
        raise ValueError("component duration must be positive")

    anchor_knots = np.asarray([x for x in extra_internal_knots if t0 + 1e-9 < float(x) < t1 - 1e-9], dtype=float)
    anchor_unique = _unique_sorted_with_tol(anchor_knots) if anchor_knots.size else np.zeros(0, dtype=float)

    absolute_max_basis = config.max_basis_count
    obs_max_basis = max(degree + 1, int(math.floor(float(t.size) / max(float(config.min_observations_per_basis), 1e-9))))
    if absolute_max_basis is None:
        effective_max_basis = obs_max_basis
    else:
        effective_max_basis = min(int(absolute_max_basis), obs_max_basis)
    # Hard anchor times are informative support locations.  Do not let the
    # observations-per-basis guard drop all anchor knots and then force a small
    # basis to satisfy too many equality constraints.
    effective_max_basis = max(effective_max_basis, degree + 1 + int(anchor_unique.size))

    spacing = max(float(config.knot_spacing_s), float(config.min_knot_spacing_s))
    max_internal = max(0, int(effective_max_basis) - degree - 1)
    requested_internal = max(0, int(math.floor(span / spacing)) - 1)
    if requested_internal > max_internal:
        spacing = span / float(max_internal + 1) if max_internal > 0 else span

    uniform = np.arange(t0 + spacing, t1 - 1e-9, spacing, dtype=float)
    internal = _unique_sorted_with_tol(np.concatenate([uniform, anchor_unique]) if anchor_unique.size else uniform)

    if internal.size + degree + 1 > int(effective_max_basis):
        if anchor_unique.size >= max_internal:
            internal = anchor_unique
        else:
            remaining = max_internal - anchor_unique.size
            if remaining > 0:
                uniform_limited = np.linspace(t0, t1, remaining + 2, dtype=float)[1:-1]
                internal = _unique_sorted_with_tol(np.concatenate([anchor_unique, uniform_limited]))
            else:
                internal = anchor_unique

    knots = np.concatenate([
        np.full(degree + 1, t0, dtype=float),
        internal.astype(float),
        np.full(degree + 1, t1, dtype=float),
    ])
    n_basis = int(knots.size - degree - 1)
    return knots, {
        "degree": degree,
        "t_local_start_s": t0,
        "t_local_end_s": t1,
        "duration_s": span,
        "observation_count": int(t.size),
        "requested_knot_spacing_s": float(config.knot_spacing_s),
        "effective_knot_spacing_s": float(spacing),
        "internal_knot_count": int(internal.size),
        "anchor_knot_count": int(anchor_unique.size),
        "n_basis": n_basis,
        "max_basis_count": config.max_basis_count,
        "min_observations_per_basis": float(config.min_observations_per_basis),
        "observation_limited_max_basis": int(obs_max_basis),
        "effective_max_basis_count": int(effective_max_basis),
    }

def _unique_sorted_with_tol(values: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)
    vals = np.sort(np.asarray(values, dtype=float).reshape(-1))
    out = [float(vals[0])]
    for value in vals[1:]:
        if abs(float(value) - out[-1]) > tol:
            out.append(float(value))
    return np.asarray(out, dtype=float)


def _evaluate_basis(
    x: np.ndarray | Sequence[float],
    knots: np.ndarray,
    degree: int,
    *,
    deriv: int = 0,
) -> np.ndarray:
    """Evaluate all B-spline basis functions or derivatives at local times."""
    if BSpline is None:  # pragma: no cover
        raise ImportError("scipy is required for B-spline basis evaluation") from _SCIPY_IMPORT_ERROR
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    n_basis = int(len(knots) - degree - 1)
    if deriv > degree:
        return np.zeros((x_arr.size, n_basis), dtype=float)
    coeff = np.eye(n_basis, dtype=float)
    spl = BSpline(knots, coeff, degree, axis=0, extrapolate=False)
    if deriv:
        spl = spl.derivative(deriv)
    mat = np.asarray(spl(x_arr), dtype=float)
    # scipy returns NaN outside the knot domain with extrapolate=False.  The
    # caller handles extrapolation, but zeroing here keeps KKT assembly safe if a
    # time is exactly on a numerically ambiguous edge.
    mat[~np.isfinite(mat)] = 0.0
    return mat



def _velocity_observation_weight_scale(
    *,
    tau: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    config: BSplineCoreConfig,
    caller_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-sample velocity weights for soft ADS-B velocity observations.

    The trajectory fit should not let a stale or mismatched ADS-B velocity row
    force curvature.  This gate compares reported velocity to a simple local
    position-derived velocity and applies a Huber-like scale.  Position-derived
    velocity is too noisy to use as truth, but it is good enough to identify
    catastrophic velocity rows that would otherwise create B-spline acceleration
    spikes, especially in short rough components.
    """
    n = int(tau.size)
    caller = None if caller_scale is None else np.clip(np.asarray(caller_scale, dtype=float).reshape(-1), 0.0, 1.0)
    scale = np.ones(n, dtype=float)
    if n < 2 or float(config.velocity_weight) <= 0:
        if caller is not None and caller.shape == (n,):
            scale = caller.copy()
        return scale, {
            "enabled": bool(caller is not None),
            "policy": str(config.velocity_outlier_policy),
            "reason": "too_few_samples_or_zero_velocity_weight",
            "caller_scale_applied": bool(caller is not None),
        }
    confidence_scale = np.ones(n, dtype=float)
    confidence_report = {"enabled": False, "reason": "disabled"}
    if bool(config.use_velocity_confidence_scaling):
        confidence_scale, confidence_report = compute_velocity_confidence_scale(tau, y, v)
    if config.velocity_outlier_policy == "none":
        scale = confidence_scale
        if caller is not None and caller.shape == (n,):
            scale = scale * caller
        return np.clip(scale, 0.0, 1.0), {
            "enabled": True,
            "policy": str(config.velocity_outlier_policy),
            "reason": "position_difference_gate_disabled",
            "velocity_confidence": confidence_report,
            "caller_scale_applied": bool(caller is not None),
            "min_scale": float(np.min(scale)),
            "median_scale": float(np.median(scale)),
            "downweighted_count": int(np.sum(scale < 0.999)),
        }
    if config.velocity_outlier_policy != "position_difference_gate":
        raise ValueError(f"unknown velocity_outlier_policy: {config.velocity_outlier_policy!r}")

    v_pos = np.zeros_like(v, dtype=float)
    dt = np.diff(tau)
    if not np.all(dt > 0):
        return scale, {"enabled": False, "policy": str(config.velocity_outlier_policy), "reason": "nonpositive_dt"}
    v_pos[0, :] = (y[1, :] - y[0, :]) / max(float(tau[1] - tau[0]), 1e-9)
    v_pos[-1, :] = (y[-1, :] - y[-2, :]) / max(float(tau[-1] - tau[-2]), 1e-9)
    if n > 2:
        denom = np.maximum(tau[2:] - tau[:-2], 1e-9)
        v_pos[1:-1, :] = (y[2:, :] - y[:-2, :]) / denom[:, None]

    mismatch = np.linalg.norm(np.asarray(v, dtype=float) - v_pos, axis=1)
    gate = max(float(config.velocity_outlier_gate_mps), 1e-9)
    huber_scale = np.minimum(1.0, gate / np.maximum(mismatch, gate))
    bad = mismatch > gate
    catastrophic = mismatch > 3.0 * gate
    # Catastrophic ADS-B velocity mismatches should contribute no curvature
    # pressure.  Keeping even a tiny floor on a 30 km/s bogus velocity can still
    # dominate the least-squares system.
    huber_scale = np.where(catastrophic, 0.0, huber_scale)
    floor = float(config.min_velocity_weight_scale)
    scale = np.where(catastrophic, 0.0, np.maximum(huber_scale, floor))
    scale = scale * confidence_scale
    if caller is not None and caller.shape == (n,):
        scale = scale * caller
    scale = np.clip(scale, 0.0, 1.0)
    return scale, {
        "enabled": True,
        "policy": str(config.velocity_outlier_policy),
        "gate_mps": float(gate),
        "min_scale": float(np.min(scale)),
        "median_scale": float(np.median(scale)),
        "mean_scale": float(np.mean(scale)),
        "downweighted_count": int(np.sum(scale < 0.999)),
        "outlier_count": int(np.sum(bad)),
        "catastrophic_outlier_count": int(np.sum(catastrophic)),
        "mismatch_mps_min": float(np.min(mismatch)),
        "mismatch_mps_median": float(np.median(mismatch)),
        "mismatch_mps_p95": float(np.percentile(mismatch, 95.0)),
        "mismatch_mps_max": float(np.max(mismatch)),
        "velocity_confidence": confidence_report,
        "caller_scale_applied": bool(caller is not None),
    }

def _endpoint_guard_times_tau(metadata: dict[str, Any] | None, *, t_origin: float, tau: np.ndarray) -> tuple[float, ...]:
    """Return component endpoint guard times in local seconds.

    The core intentionally does not infer true component endpoints from every
    local segment fit.  The pipeline marks only genuine hard-gap component
    start/end times in metadata so overlap-save internal segment endpoints are
    not over-damped accidentally.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    raw = meta.get("component_endpoint_guard_times_s") or meta.get("endpoint_guard_times_s") or ()
    if isinstance(raw, (int, float)):
        raw_values = [raw]
    elif isinstance(raw, (list, tuple)):
        raw_values = list(raw)
    else:
        raw_values = []
    out: list[float] = []
    if tau.size == 0:
        return ()
    lo = float(np.min(tau)) - 1e-6
    hi = float(np.max(tau)) + 1e-6
    for value in raw_values:
        try:
            tt = float(value) - float(t_origin)
        except Exception:
            continue
        if math.isfinite(tt) and lo <= tt <= hi:
            out.append(float(tt))
    # Stable ordering and de-duplication keep diagnostics deterministic.
    deduped: list[float] = []
    for tt in sorted(out):
        if not deduped or abs(tt - deduped[-1]) > 1e-6:
            deduped.append(tt)
    return tuple(deduped)


def _apply_endpoint_guard_interval_multiplier(
    *,
    tau: np.ndarray,
    interval_weights: np.ndarray,
    guard_times_tau: Sequence[float],
    window_s: float,
    multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boost derivative penalty weights around true component endpoints."""
    weights = np.asarray(interval_weights, dtype=float).copy()
    report = {
        "enabled": False,
        "window_s": float(window_s),
        "multiplier": float(multiplier),
        "guard_time_count": int(len(tuple(guard_times_tau))),
        "affected_interval_count": 0,
        "max_effective_multiplier": 1.0,
    }
    if weights.size == 0 or tau.size < 2 or float(window_s) <= 0.0 or float(multiplier) <= 1.0 or not guard_times_tau:
        return weights, report
    mid = 0.5 * (np.asarray(tau[:-1], dtype=float) + np.asarray(tau[1:], dtype=float))
    factors = np.ones(weights.shape, dtype=float)
    window = max(float(window_s), 1e-9)
    for guard_t in guard_times_tau:
        d = np.abs(mid - float(guard_t))
        inside = d <= window
        if not np.any(inside):
            continue
        # Smooth quadratic taper: strongest at the endpoint, no step at the
        # outside edge of the guard window.
        local = np.ones(weights.shape, dtype=float)
        local[inside] = 1.0 + (float(multiplier) - 1.0) * np.square(1.0 - d[inside] / window)
        factors = np.maximum(factors, local)
    weights *= factors
    affected = factors > 1.0000001
    report.update(
        {
            "enabled": bool(np.any(affected)),
            "guard_times_tau_s": [float(v) for v in guard_times_tau],
            "affected_interval_count": int(np.sum(affected)),
            "max_effective_multiplier": float(np.max(factors)) if factors.size else 1.0,
        }
    )
    return weights, report


def _solve_b_spline_once(
    *,
    tau: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    anchors: Sequence[BSplineAnchor],
    position_priors: Sequence[BSplinePositionPrior],
    velocity_priors: Sequence[BSplineVelocityPrior],
    acceleration_priors: Sequence[BSplineAccelerationPrior],
    velocity_constraints: Sequence[BSplineVelocityConstraint],
    knots: np.ndarray,
    config: BSplineCoreConfig,
    lambda_intervals: np.ndarray,
    position_weight_scale: np.ndarray,
    velocity_weight_scale: np.ndarray,
    endpoint_guard_times_tau: Sequence[float],
    t_origin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    degree = int(config.degree)
    n_basis = int(len(knots) - degree - 1)
    n, dim = y.shape

    rows: list[Any] = []
    targets: list[np.ndarray] = []
    row_groups: list[dict[str, Any]] = []

    # Position observations.
    b0 = _evaluate_basis(tau, knots, degree, deriv=0)
    w_pos = np.sqrt(float(config.position_weight) * np.asarray(position_weight_scale, dtype=float))
    rows.append(sparse.diags(w_pos, format="csr") @ sparse.csr_matrix(b0))
    targets.append(w_pos[:, None] * y)
    row_groups.append({"name": "position_observations", "rows": int(n), "weight": float(config.position_weight)})

    position_prior_rows = 0
    position_prior_weight_values: list[float] = []
    if position_priors and float(config.boundary_position_prior_weight) > 0:
        prior_times = np.asarray([float(p.t) - t_origin for p in position_priors], dtype=float)
        b0p = _evaluate_basis(prior_times, knots, degree, deriv=0)
        weights = []
        target = []
        for prior in position_priors:
            base_w = float(config.boundary_position_prior_weight if prior.weight is None else prior.weight)
            conf = 1.0 if prior.confidence is None else min(max(float(prior.confidence), 0.0), 1.0)
            if config.scale_boundary_position_prior_by_confidence:
                base_w *= 0.25 + 0.75 * conf
            weights.append(math.sqrt(max(base_w, 0.0)))
            position_prior_weight_values.append(base_w)
            target.append(prior.position_array(dim))
        wp0 = np.asarray(weights, dtype=float)
        keep = wp0 > 0.0
        if np.any(keep):
            rows.append(sparse.diags(wp0[keep], format="csr") @ sparse.csr_matrix(b0p[keep, :]))
            targets.append(wp0[keep, None] * np.asarray(target, dtype=float)[keep, :])
        position_prior_rows = int(np.sum(keep))
        row_groups.append(
            {
                "name": "boundary_position_priors",
                "rows": int(position_prior_rows),
                "base_weight": float(config.boundary_position_prior_weight),
                "min_effective_weight": float(min(position_prior_weight_values)) if position_prior_weight_values else None,
                "max_effective_weight": float(max(position_prior_weight_values)) if position_prior_weight_values else None,
            }
        )

    # Velocity observations remain soft.  A zero velocity weight is allowed.
    if float(config.velocity_weight) > 0:
        b1 = _evaluate_basis(tau, knots, degree, deriv=1)
        scale = np.asarray(velocity_weight_scale, dtype=float).reshape(-1)
        if scale.shape != (n,):
            raise ValueError(f"velocity_weight_scale must have shape ({n},); got {scale.shape}")
        w_vel = np.sqrt(float(config.velocity_weight) * np.maximum(scale, 0.0))
        keep = w_vel > 0.0
        if np.any(keep):
            rows.append(sparse.diags(w_vel[keep], format="csr") @ sparse.csr_matrix(b1[keep, :]))
            targets.append(w_vel[keep, None] * v[keep, :])
        row_groups.append(
            {
                "name": "velocity_observations",
                "rows": int(np.sum(keep)),
                "raw_rows": int(n),
                "weight": float(config.velocity_weight),
                "min_weight_scale": float(np.min(scale)) if scale.size else None,
                "median_weight_scale": float(np.median(scale)) if scale.size else None,
                "downweighted_count": int(np.sum(scale < 0.999)),
            }
        )

    # Boundary velocity priors.  These are soft, not hard, because noisy ADS-B or
    # one-sided regression velocity can otherwise force curvature artifacts.
    prior_rows = 0
    prior_weight_values: list[float] = []
    if velocity_priors and float(config.boundary_velocity_prior_weight) > 0:
        prior_times = np.asarray([float(p.t) - t_origin for p in velocity_priors], dtype=float)
        b1p = _evaluate_basis(prior_times, knots, degree, deriv=1)
        weights = []
        target = []
        for prior in velocity_priors:
            base_w = float(config.boundary_velocity_prior_weight if prior.weight is None else prior.weight)
            conf = 1.0 if prior.confidence is None else min(max(float(prior.confidence), 0.0), 1.0)
            if config.scale_boundary_velocity_prior_by_confidence:
                # Do not erase low-confidence priors completely; make them weak.
                base_w *= 0.25 + 0.75 * conf
            weights.append(math.sqrt(max(base_w, 0.0)))
            prior_weight_values.append(base_w)
            target.append(prior.velocity_array(dim))
        wp = np.asarray(weights, dtype=float)
        rows.append(sparse.diags(wp, format="csr") @ sparse.csr_matrix(b1p))
        targets.append(wp[:, None] * np.asarray(target, dtype=float))
        prior_rows = len(velocity_priors)
        row_groups.append(
            {
                "name": "boundary_velocity_priors",
                "rows": int(prior_rows),
                "base_weight": float(config.boundary_velocity_prior_weight),
                "min_effective_weight": float(min(prior_weight_values)) if prior_weight_values else None,
                "max_effective_weight": float(max(prior_weight_values)) if prior_weight_values else None,
            }
        )

    # Boundary acceleration priors are the local-coupling term that makes two
    # independently solved segments agree in curvature when the data support it.
    # They are soft for robustness: an exact acceleration equality at both ends
    # of a short cubic segment can make the position fit worse than the raw data.
    acceleration_prior_rows = 0
    acceleration_prior_weight_values: list[float] = []
    if acceleration_priors and float(config.boundary_acceleration_prior_weight) > 0 and degree >= 2:
        prior_times = np.asarray([float(p.t) - t_origin for p in acceleration_priors], dtype=float)
        b2p = _evaluate_basis(prior_times, knots, degree, deriv=2)
        weights = []
        target = []
        for prior in acceleration_priors:
            base_w = float(config.boundary_acceleration_prior_weight if prior.weight is None else prior.weight)
            conf = 1.0 if prior.confidence is None else min(max(float(prior.confidence), 0.0), 1.0)
            if config.scale_boundary_acceleration_prior_by_confidence:
                base_w *= 0.25 + 0.75 * conf
            weights.append(math.sqrt(max(base_w, 0.0)))
            acceleration_prior_weight_values.append(base_w)
            target.append(prior.acceleration_array(dim))
        wa = np.asarray(weights, dtype=float)
        keep = wa > 0.0
        if np.any(keep):
            rows.append(sparse.diags(wa[keep], format="csr") @ sparse.csr_matrix(b2p[keep, :]))
            targets.append(wa[keep, None] * np.asarray(target, dtype=float)[keep, :])
        acceleration_prior_rows = int(np.sum(keep))
        row_groups.append(
            {
                "name": "boundary_acceleration_priors",
                "rows": int(acceleration_prior_rows),
                "base_weight": float(config.boundary_acceleration_prior_weight),
                "min_effective_weight": float(min(acceleration_prior_weight_values)) if acceleration_prior_weight_values else None,
                "max_effective_weight": float(max(acceleration_prior_weight_values)) if acceleration_prior_weight_values else None,
            }
        )

    # V-Spline-like integrated squared acceleration penalty via quadrature.
    b2q, w2 = _quadrature_design(tau, knots, degree, deriv=2, interval_weights=lambda_intervals, scale_by_n=n, order=config.quadrature_order)
    if b2q.shape[0] > 0 and np.any(w2 > 0):
        sw = np.sqrt(np.maximum(w2, 0.0))
        rows.append(sparse.diags(sw, format="csr") @ sparse.csr_matrix(b2q))
        targets.append(np.zeros((b2q.shape[0], dim), dtype=float))
        row_groups.append({"name": "integrated_squared_acceleration_penalty", "rows": int(b2q.shape[0])})

    # Optional jerk penalty.  This is intentionally separate from the V-Spline
    # acceleration penalty so it can be tuned on derivative quality diagnostics.
    if float(config.jerk_penalty_weight) > 0 and degree >= 3:
        jerk_weights = np.full(max(tau.size - 1, 0), float(config.jerk_penalty_weight), dtype=float)
        jerk_weights, jerk_guard_report = _apply_endpoint_guard_interval_multiplier(
            tau=tau,
            interval_weights=jerk_weights,
            guard_times_tau=endpoint_guard_times_tau,
            window_s=float(config.endpoint_guard_window_s),
            multiplier=float(config.endpoint_jerk_penalty_multiplier),
        )
        b3q, w3 = _quadrature_design(tau, knots, degree, deriv=3, interval_weights=jerk_weights, scale_by_n=n, order=config.quadrature_order)
        if b3q.shape[0] > 0 and np.any(w3 > 0):
            sw = np.sqrt(np.maximum(w3, 0.0))
            rows.append(sparse.diags(sw, format="csr") @ sparse.csr_matrix(b3q))
            targets.append(np.zeros((b3q.shape[0], dim), dtype=float))
            row_groups.append(
                {
                    "name": "integrated_squared_jerk_penalty",
                    "rows": int(b3q.shape[0]),
                    "weight": float(config.jerk_penalty_weight),
                    "endpoint_guard": jerk_guard_report,
                }
            )

    if float(config.snap_penalty_weight) > 0 and degree >= 4:
        snap_weights = np.full(max(tau.size - 1, 0), float(config.snap_penalty_weight), dtype=float)
        snap_weights, snap_guard_report = _apply_endpoint_guard_interval_multiplier(
            tau=tau,
            interval_weights=snap_weights,
            guard_times_tau=endpoint_guard_times_tau,
            window_s=float(config.endpoint_guard_window_s),
            multiplier=float(config.endpoint_snap_penalty_multiplier),
        )
        b4q, w4 = _quadrature_design(tau, knots, degree, deriv=4, interval_weights=snap_weights, scale_by_n=n, order=config.quadrature_order)
        if b4q.shape[0] > 0 and np.any(w4 > 0):
            sw = np.sqrt(np.maximum(w4, 0.0))
            rows.append(sparse.diags(sw, format="csr") @ sparse.csr_matrix(b4q))
            targets.append(np.zeros((b4q.shape[0], dim), dtype=float))
            row_groups.append(
                {
                    "name": "integrated_squared_snap_penalty",
                    "rows": int(b4q.shape[0]),
                    "weight": float(config.snap_penalty_weight),
                    "endpoint_guard": snap_guard_report,
                }
            )

    if not rows:
        raise RuntimeError("empty B-spline least-squares system")
    a_mat = sparse.vstack(rows, format="csr")
    rhs_targets = np.vstack(targets)

    # Hard endpoint/anchor constraints are equality constraints E c = d.
    # Position constraints use B(t); velocity constraints use B'(t).
    e_rows: list[Any] = []
    d_blocks: list[np.ndarray] = []
    if anchors:
        anchor_times = np.asarray([float(a.t) - t_origin for a in anchors], dtype=float)
        e_rows.append(sparse.csr_matrix(_evaluate_basis(anchor_times, knots, degree, deriv=0)))
        d_blocks.append(np.vstack([a.position_array(dim) for a in anchors]))
    if velocity_constraints:
        velocity_times = np.asarray([float(c.t) - t_origin for c in velocity_constraints], dtype=float)
        e_rows.append(sparse.csr_matrix(_evaluate_basis(velocity_times, knots, degree, deriv=1)))
        d_blocks.append(np.vstack([c.velocity_array(dim) for c in velocity_constraints]))

    if e_rows:
        e_mat = sparse.vstack(e_rows, format="csr")
        d_mat = np.vstack(d_blocks)
    else:
        e_mat = sparse.csr_matrix((0, n_basis), dtype=float)
        d_mat = np.zeros((0, dim), dtype=float)

    coeff, linear_report = _solve_sparse_kkt(
        a_mat=a_mat,
        target=rhs_targets,
        e_mat=e_mat,
        d_mat=d_mat,
        ridge=float(config.solver_ridge),
        condition_number_max_basis=int(config.condition_number_max_basis),
    )
    linear_report.update(
        {
            "row_groups": row_groups,
            "position_prior_row_count": int(position_prior_rows),
            "prior_row_count": int(prior_rows),
            "acceleration_prior_row_count": int(acceleration_prior_rows),
            "least_squares_rows": int(a_mat.shape[0]),
            "least_squares_cols": int(a_mat.shape[1]),
            "constraint_rows": int(e_mat.shape[0]),
            "hard_position_constraint_rows": int(len(anchors)),
            "hard_velocity_constraint_rows": int(len(velocity_constraints)),
        }
    )
    return coeff, linear_report


def _quadrature_design(
    tau: np.ndarray,
    knots: np.ndarray,
    degree: int,
    *,
    deriv: int,
    interval_weights: np.ndarray,
    scale_by_n: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tau.size < 2 or interval_weights.size == 0:
        n_basis = int(len(knots) - degree - 1)
        return np.zeros((0, n_basis), dtype=float), np.zeros(0, dtype=float)
    if interval_weights.shape != (tau.size - 1,):
        raise ValueError(f"interval_weights must have shape ({tau.size - 1},); got {interval_weights.shape}")
    if order == 2:
        nodes = np.asarray([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=float)
        weights = np.asarray([1.0, 1.0], dtype=float)
    else:
        nodes = np.asarray([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)], dtype=float)
        weights = np.asarray([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    q_times: list[float] = []
    q_weights: list[float] = []
    for i, h in enumerate(np.diff(tau)):
        if h <= 0:
            continue
        lam = max(float(interval_weights[i]), 0.0)
        if lam <= 0:
            continue
        mid = 0.5 * (float(tau[i]) + float(tau[i + 1]))
        half = 0.5 * float(h)
        for node, w in zip(nodes, weights):
            q_times.append(mid + half * float(node))
            q_weights.append(float(scale_by_n) * lam * half * float(w))
    if not q_times:
        n_basis = int(len(knots) - degree - 1)
        return np.zeros((0, n_basis), dtype=float), np.zeros(0, dtype=float)
    return _evaluate_basis(np.asarray(q_times, dtype=float), knots, degree, deriv=deriv), np.asarray(q_weights, dtype=float)


def _solve_sparse_kkt(
    *,
    a_mat: Any,
    target: np.ndarray,
    e_mat: Any,
    d_mat: np.ndarray,
    ridge: float,
    condition_number_max_basis: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_basis = int(a_mat.shape[1])
    h_mat = (a_mat.T @ a_mat).tocsc()
    if ridge > 0:
        h_mat = h_mat + float(ridge) * sparse.eye(n_basis, format="csc")
    rhs = a_mat.T @ target

    constrained = int(e_mat.shape[0]) > 0
    if constrained:
        zero = sparse.csc_matrix((e_mat.shape[0], e_mat.shape[0]), dtype=float)
        kkt = sparse.bmat([[h_mat, e_mat.T], [e_mat, zero]], format="csc")
        kkt_rhs = np.vstack([rhs, d_mat])
    else:
        kkt = h_mat
        kkt_rhs = rhs

    try:
        sol = spsolve(kkt, kkt_rhs)
        method = "sparse_kkt_spsolve" if constrained else "sparse_normal_spsolve"
    except Exception as exc:
        warnings.warn(f"sparse B-spline solve failed ({exc}); falling back to dense least-squares", RuntimeWarning)
        sol = np.linalg.lstsq(kkt.toarray(), kkt_rhs, rcond=None)[0]
        method = "dense_lstsq_fallback"

    sol = np.asarray(sol, dtype=float)
    if sol.ndim == 1:
        sol = sol.reshape(-1, 1)
    coeff = sol[:n_basis, :]

    normal_res = h_mat @ coeff - rhs
    if constrained:
        normal_res = normal_res + e_mat.T @ sol[n_basis:, :]
        constraint_res = e_mat @ coeff - d_mat
    else:
        constraint_res = np.zeros((0, coeff.shape[1]), dtype=float)

    cond_h: float | None = None
    if n_basis <= int(condition_number_max_basis):
        try:
            cond_h = float(np.linalg.cond(h_mat.toarray()))
        except Exception:
            cond_h = None

    return coeff, {
        "method": method,
        "n_basis": int(n_basis),
        "kkt_shape": list(kkt.shape),
        "ridge": float(ridge),
        "condition_number_hessian": cond_h,
        "normal_relative_residual": float(np.linalg.norm(normal_res) / max(np.linalg.norm(rhs), 1e-30)),
        "constraint_max_abs_error": float(np.max(np.abs(constraint_res))) if constraint_res.size else 0.0,
    }


def _fit_diagnostics(
    *,
    fit: BSplineCoreFit,
    core_input: BSplineCoreInput,
    anchors: Sequence[BSplineAnchor],
    position_priors: Sequence[BSplinePositionPrior],
    raw_anchor_count: int,
    anchor_dedupe_report: dict[str, Any],
    velocity_priors: Sequence[BSplineVelocityPrior],
    acceleration_priors: Sequence[BSplineAccelerationPrior],
    velocity_constraints: Sequence[BSplineVelocityConstraint],
    velocity_gate_report: dict[str, Any],
    knot_report: dict[str, Any],
    lambda_report: dict[str, Any],
    solve_report: dict[str, Any],
    irls_reports: Sequence[dict[str, Any]],
    position_weight_scale: np.ndarray,
    component_id: str | None,
) -> dict[str, Any]:
    y_hat = fit.evaluate(core_input.t, deriv=0)
    v_hat = fit.evaluate(core_input.t, deriv=1)
    a_grid, j_grid = _diagnostic_derivative_grid(fit)
    pos_delta = y_hat - core_input.y
    vel_delta = v_hat - core_input.v
    e3 = np.linalg.norm(pos_delta, axis=1)
    ev = np.linalg.norm(vel_delta, axis=1)

    anchor_reports: list[dict[str, Any]] = []
    anchor_errors = []
    for anchor in anchors:
        pred = fit.evaluate([float(anchor.t)], deriv=0)[0]
        raw = anchor.position_array(fit.dimension)
        err_vec = pred - raw
        err = float(np.linalg.norm(err_vec))
        anchor_errors.append(err)
        anchor_reports.append(
            {
                **anchor.as_dict(),
                "fitted_position_m": pred.tolist(),
                "error_vector_m": err_vec.tolist(),
                "error_norm_m": err,
            }
        )

    position_prior_reports: list[dict[str, Any]] = []
    position_prior_errors = []
    for prior in position_priors:
        pred = fit.evaluate([float(prior.t)], deriv=0)[0]
        raw = prior.position_array(fit.dimension)
        err_vec = pred - raw
        err = float(np.linalg.norm(err_vec))
        position_prior_errors.append(err)
        position_prior_reports.append(
            {
                **prior.as_dict(),
                "fitted_position_m": pred.tolist(),
                "error_vector_m": err_vec.tolist(),
                "error_norm_m": err,
            }
        )

    prior_reports: list[dict[str, Any]] = []
    prior_errors = []
    for prior in velocity_priors:
        pred = fit.evaluate([float(prior.t)], deriv=1)[0]
        raw = prior.velocity_array(fit.dimension)
        err_vec = pred - raw
        err = float(np.linalg.norm(err_vec))
        prior_errors.append(err)
        prior_reports.append(
            {
                **prior.as_dict(),
                "fitted_velocity_mps": pred.tolist(),
                "error_vector_mps": err_vec.tolist(),
                "error_norm_mps": err,
            }
        )

    acceleration_prior_reports: list[dict[str, Any]] = []
    acceleration_prior_errors = []
    for prior in acceleration_priors:
        pred = fit.evaluate([float(prior.t)], deriv=2)[0]
        raw = prior.acceleration_array(fit.dimension)
        err_vec = pred - raw
        err = float(np.linalg.norm(err_vec))
        acceleration_prior_errors.append(err)
        acceleration_prior_reports.append(
            {
                **prior.as_dict(),
                "fitted_acceleration_mps2": pred.tolist(),
                "error_vector_mps2": err_vec.tolist(),
                "error_norm_mps2": err,
            }
        )

    velocity_constraint_reports: list[dict[str, Any]] = []
    velocity_constraint_errors = []
    for constraint in velocity_constraints:
        pred = fit.evaluate([float(constraint.t)], deriv=1)[0]
        raw = constraint.velocity_array(fit.dimension)
        err_vec = pred - raw
        err = float(np.linalg.norm(err_vec))
        velocity_constraint_errors.append(err)
        velocity_constraint_reports.append(
            {
                **constraint.as_dict(),
                "fitted_velocity_mps": pred.tolist(),
                "error_vector_mps": err_vec.tolist(),
                "error_norm_mps": err,
            }
        )

    acc_norm = np.linalg.norm(a_grid, axis=1) if a_grid.size else np.zeros(1, dtype=float)
    jerk_norm = np.linalg.norm(j_grid, axis=1) if j_grid.size else np.zeros(1, dtype=float)
    diagnostics = {
        "method": fit.config.backend_name,
        "component_id": component_id,
        "basis": fit.basis,
        "degree": int(fit.degree),
        "objective": "position_residuals_plus_soft_velocity_residuals_plus_soft_boundary_acceleration_priors_plus_v_spline_adaptive_integrated_squared_acceleration_plus_optional_jerk_subject_to_hard_position_and_optional_hard_velocity_constraints",
        "n_observations": int(core_input.t.size),
        "dimension": int(core_input.y.shape[1]),
        "n_basis": int(fit.coefficients.shape[0]),
        "dof_per_dimension": int(fit.coefficients.shape[0]),
        "time_origin_s": float(fit.t_origin),
        "t_start_s": float(core_input.t[0]),
        "t_end_s": float(core_input.t[-1]),
        "duration_s": float(core_input.t[-1] - core_input.t[0]),
        "input_metadata": core_input.metadata or {},
        "min_dt_s": float(np.min(np.diff(core_input.t))),
        "max_dt_s": float(np.max(np.diff(core_input.t))),
        "knot_report": knot_report,
        "lambda_report": lambda_report,
        "min_lambda_interval": float(np.min(fit.lambda_intervals)) if fit.lambda_intervals.size else None,
        "max_lambda_interval": float(np.max(fit.lambda_intervals)) if fit.lambda_intervals.size else None,
        "solver": solve_report,
        "hard_position_anchors": {
            "raw_anchor_count": int(raw_anchor_count),
            "deduped_anchor_count": int(len(anchors)),
            "anchor_dedupe_report": anchor_dedupe_report,
            "max_anchor_error_m": float(max(anchor_errors) if anchor_errors else 0.0),
            "p95_anchor_error_m": float(np.quantile(anchor_errors, 0.95)) if anchor_errors else 0.0,
            "anchors": anchor_reports,
        },
        "n_anchors": int(len(anchors)),
        "max_anchor_error_m": float(max(anchor_errors) if anchor_errors else 0.0),
        "boundary_position_priors": {
            "count": int(len(position_priors)),
            "max_error_m": float(max(position_prior_errors) if position_prior_errors else 0.0),
            "p95_error_m": float(np.quantile(position_prior_errors, 0.95)) if position_prior_errors else 0.0,
            "priors": position_prior_reports,
        },
        "boundary_velocity_priors": {
            "count": int(len(velocity_priors)),
            "max_error_mps": float(max(prior_errors) if prior_errors else 0.0),
            "p95_error_mps": float(np.quantile(prior_errors, 0.95)) if prior_errors else 0.0,
            "priors": prior_reports,
        },
        "boundary_acceleration_priors": {
            "count": int(len(acceleration_priors)),
            "max_error_mps2": float(max(acceleration_prior_errors) if acceleration_prior_errors else 0.0),
            "p95_error_mps2": float(np.quantile(acceleration_prior_errors, 0.95)) if acceleration_prior_errors else 0.0,
            "priors": acceleration_prior_reports,
        },
        "hard_velocity_constraints": {
            "count": int(len(velocity_constraints)),
            "max_error_mps": float(max(velocity_constraint_errors) if velocity_constraint_errors else 0.0),
            "p95_error_mps": float(np.quantile(velocity_constraint_errors, 0.95)) if velocity_constraint_errors else 0.0,
            "constraints": velocity_constraint_reports,
        },
        "position_residual_rmse_3d_m": float(np.sqrt(np.mean(e3 * e3))),
        "position_residual_median_3d_m": float(np.median(e3)),
        "position_residual_p95_3d_m": float(np.quantile(e3, 0.95)),
        "position_residual_max_3d_m": float(np.max(e3)),
        "position_residual_rms_by_dim": np.sqrt(np.mean(pos_delta * pos_delta, axis=0)).tolist(),
        "velocity_residual_rmse_3d_mps": float(np.sqrt(np.mean(ev * ev))),
        "velocity_residual_median_3d_mps": float(np.median(ev)),
        "velocity_residual_p95_3d_mps": float(np.quantile(ev, 0.95)),
        "velocity_residual_rms_by_dim": np.sqrt(np.mean(vel_delta * vel_delta, axis=0)).tolist(),
        "accel_rms_mps2": float(np.sqrt(np.mean(acc_norm * acc_norm))),
        "accel_p95_mps2": float(np.quantile(acc_norm, 0.95)),
        "accel_max_mps2": float(np.max(acc_norm)),
        "jerk_rms_mps3": float(np.sqrt(np.mean(jerk_norm * jerk_norm))),
        "jerk_p95_mps3": float(np.quantile(jerk_norm, 0.95)),
        "jerk_max_mps3": float(np.max(jerk_norm)),
        "robust_position_loss": {
            "mode": fit.config.robust_position_loss,
            "huber_delta_m": float(fit.config.huber_delta_m),
            "iteration_reports": list(irls_reports),
            "min_final_weight_scale": float(np.min(position_weight_scale)),
            "downweighted_count": int(np.sum(position_weight_scale < 0.999)),
        },
        "config": asdict(fit.config),
    }
    return diagnostics


def _diagnostic_derivative_grid(fit: BSplineCoreFit) -> tuple[np.ndarray, np.ndarray]:
    span = float(fit.t[-1] - fit.t[0])
    if span <= 0:
        return np.zeros((0, fit.dimension)), np.zeros((0, fit.dimension))
    step = max(min(span / 200.0, 1.0), 0.25)
    grid = np.arange(float(fit.t[0]), float(fit.t[-1]) + step * 0.5, step)
    if grid.size == 0 or grid[-1] < fit.t[-1]:
        grid = np.append(grid, fit.t[-1])
    acc = fit.evaluate(grid, deriv=2)
    jerk = fit.evaluate(grid, deriv=3)
    return acc, jerk
