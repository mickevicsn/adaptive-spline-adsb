"""Local per-segment tuning for B-spline V-Spline reconstruction.

The policy module chooses a good first guess from the aircraft energy regime.
This module searches a small, deterministic candidate set around that first
guess and scores each candidate on position fidelity plus derivative quality.
It deliberately tunes *inside one segment* only; segment coupling is handled by
shared boundary states, constraints, and/or soft priors in the pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

import math

from trajectory_segmentation import DynamicSegment
from .bspline_core import BSplineCoreConfig
from .quality import SegmentQualityReport
from .segment_policy import SelectedBSplineSegmentParams


@dataclass(frozen=True)
class LocalSegmentTuningConfig:
    """Small local tuner for each independent B-spline/V-Spline segment.

    The objective is not a blind "lowest residual" fit, because that would
    choose the densest, least-smoothed spline for noisy ADS-B.  Instead it uses
    a balanced score:

    * raw position RMSE / p95 residuals,
    * acceleration and jerk relative to regime-specific targets,
    * a small complexity penalty for too many basis functions.

    The final segment still receives shared boundary states from the pipeline,
    so continuity is protected while parameters are tuned locally without
    forcing every boundary back to one raw ADS-B row.
    """

    enabled: bool = True
    objective: Literal["balanced", "position", "smooth"] = "balanced"
    max_candidates: int = 14

    # Candidate clamps.  The recipes are multipliers around the energy-state
    # policy output, then these bounds keep the solver in sane ADS-B territory.
    min_adaptive_eta: float = 500.0
    max_adaptive_eta: float = 1_000_000.0
    min_smoothing_lambda: float = 0.1
    max_smoothing_lambda: float = 500.0
    min_velocity_weight: float = 0.0
    max_velocity_weight: float = 0.12
    min_knot_spacing_s: float = 1.0
    max_knot_spacing_s: float = 12.0
    min_observations_per_basis: float = 2.5
    max_observations_per_basis: float = 16.0
    min_jerk_penalty_weight: float = 0.0
    max_jerk_penalty_weight: float = 0.05
    min_acceleration_prior_weight: float = 0.0
    max_acceleration_prior_weight: float = 0.5
    min_acceleration_multiplier: float = 0.35
    max_acceleration_multiplier: float = 3.0

    # Score weights.  Position is in metres.  Motion terms are normalized by
    # regime targets and converted into a metre-equivalent penalty by
    # motion_cost_scale_m.
    p95_position_weight: float = 0.20
    max_position_weight: float = 0.02
    motion_cost_scale_m: float = 8.0
    complexity_cost_scale_m: float = 4.0
    # 4BAAD9 report action: candidate selection should not optimize only the
    # interior residual.  Short transition/energy-change segments are expensive
    # because every boundary becomes a join artifact risk, so add a small risk
    # penalty that favors fewer bases, lower jerk, and stronger derivative
    # smoothing in those windows.
    join_artifact_cost_scale_m: float = 6.0

    # First-pass join harmonization.  The initial prefit uses soft boundary
    # velocity priors only; after the piecewise curve is combined, the pipeline
    # estimates one shared velocity per join and refits with that velocity as a
    # hard equality constraint.
    join_velocity_harmonization: bool = True
    prefit_boundary_velocity_prior_weight: float = 0.025
    harmonized_fit_velocity_weight: float = 1.0
    harmonized_boundary_state_weight: float = 0.75
    harmonized_reported_velocity_weight: float = 0.10
    harmonized_position_slope_weight: float = 0.35

    # Keep candidate reports finite and not enormous in JSON.
    include_all_candidate_reports: bool = True

    # Quality-triggered adaptive resegmentation.  After a segment has been
    # fitted and locally tuned, a high residual usually means the original
    # segment still contains two different motion regimes or an unmodelled
    # manoeuvre.  In that case the pipeline re-runs segmentation on just that
    # segment, adds one feasible internal boundary, then refits the component
    # with the same shared-boundary C0/C1 machinery.
    adaptive_resegmentation_enabled: bool = True
    adaptive_resegmentation_max_passes: int = 2
    adaptive_resegmentation_max_new_boundaries_per_pass: int = 8
    adaptive_resegmentation_bad_rmse_m: float = 120.0
    adaptive_resegmentation_bad_p95_m: float = 240.0
    adaptive_resegmentation_bad_max_m: float = 600.0
    # Axis-aware vertical-shape triggers.  4BAAD9 exposed a failure mode where
    # the whole-segment 3D RMSE looked acceptable, but a short level-off /
    # altitude-reversal window had a visibly bad Z overshoot.  These fields let
    # adaptive resegmentation split that local vertical lobe instead of waiting
    # for a huge aggregate 3D residual.
    adaptive_resegmentation_bad_vertical_rmse_m: float = 10.0
    adaptive_resegmentation_bad_vertical_p95_m: float = 18.0
    adaptive_resegmentation_bad_vertical_max_m: float = 30.0
    adaptive_resegmentation_bad_vertical_window_m: float = 10.0
    adaptive_resegmentation_vertical_run_min_points: int = 3
    adaptive_resegmentation_vertical_run_min_duration_s: float = 3.0
    # Adaptive refinement is allowed to create short transition windows.  Initial
    # segmentation stays conservative, but post-fit residual evidence can justify
    # a smaller split around a go-around/level-off kink.
    adaptive_resegmentation_min_points: int = 8
    adaptive_resegmentation_min_duration_s: float = 8.0
    adaptive_resegmentation_min_boundary_spacing_s: float = 8.0
    adaptive_resegmentation_use_feature_segmentation: bool = True
    adaptive_resegmentation_residual_window_points: int = 5
    adaptive_resegmentation_max_segments_per_component: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateScore:
    score: float
    components: dict[str, float]
    profile: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_bspline_param_candidates(
    selected: SelectedBSplineSegmentParams,
    base_config: BSplineCoreConfig,
    tuning: LocalSegmentTuningConfig,
) -> list[SelectedBSplineSegmentParams]:
    """Return a compact deterministic candidate set around ``selected``."""
    base_recipes = [
        ("policy_base", 1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
        ("more_position_detail", 0.45, 1.00, 0.70, 0.70, 0.75, 0.70),
        ("position_detail_velocity_light", 0.55, 0.45, 0.75, 0.80, 0.70, 0.75),
        ("position_detail_velocity_heavy", 0.70, 1.75, 0.85, 0.80, 0.90, 0.80),
        ("balanced_low_velocity", 1.00, 0.35, 1.00, 1.00, 1.00, 1.00),
        ("balanced_high_velocity", 1.00, 2.00, 1.00, 1.00, 1.00, 1.00),
        ("smoother", 1.80, 0.70, 1.25, 1.35, 1.50, 1.30),
        ("very_smooth_velocity_light", 3.20, 0.30, 1.60, 1.80, 2.50, 1.80),
        ("rough_air_guard", 4.00, 0.20, 1.80, 2.10, 3.00, 2.20),
        ("transition_flexible", 0.75, 1.25, 0.65, 0.65, 0.60, 0.60),
        ("transition_flexible_low_velocity", 0.85, 0.35, 0.70, 0.75, 0.80, 0.70),
        ("smooth_derivatives", 1.40, 0.70, 1.15, 1.20, 2.20, 1.50),
        ("low_jerk_position", 0.80, 1.00, 0.90, 0.90, 2.00, 0.90),
        ("high_smoothing_same_knots", 2.50, 0.70, 1.00, 1.20, 2.00, 1.60),
        ("low_smoothing_same_knots", 0.35, 1.00, 1.00, 0.90, 0.70, 0.70),
        ("wide_knots_low_velocity", 2.00, 0.30, 1.45, 1.60, 2.00, 1.50),
    ]
    targeted_recipes = _targeted_short_transition_recipes(selected)
    # Put the report-driven sparse/low-velocity recipes early enough to survive
    # the balanced preset's default max_candidates=10 pruning.
    recipes = base_recipes[:5] + targeted_recipes + base_recipes[5:]
    if tuning.objective == "position":
        recipes = [recipes[0], recipes[1], recipes[2], recipes[3], *targeted_recipes, base_recipes[9], base_recipes[10], base_recipes[14], base_recipes[12]]
    elif tuning.objective == "smooth":
        recipes = [recipes[0], *targeted_recipes, base_recipes[6], base_recipes[7], base_recipes[8], base_recipes[11], base_recipes[13], base_recipes[15], base_recipes[4]]

    out: list[SelectedBSplineSegmentParams] = []
    seen: set[tuple[Any, ...]] = set()
    limit = max(1, int(tuning.max_candidates))
    for name, eta_mul, vel_mul, knot_mul, obs_mul, jerk_mul, acc_mul in recipes:
        if len(out) >= limit:
            break
        cand = _candidate_from_recipe(
            selected=selected,
            base_config=base_config,
            tuning=tuning,
            name=name,
            eta_mul=eta_mul,
            vel_mul=vel_mul,
            knot_mul=knot_mul,
            obs_mul=obs_mul,
            jerk_mul=jerk_mul,
            acc_mul=acc_mul,
        )
        key = _candidate_key(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    if not out:
        out.append(selected)
    return out


def score_bspline_candidate(
    segment: DynamicSegment,
    fit: Any,
    quality: SegmentQualityReport,
    tuning: LocalSegmentTuningConfig,
) -> CandidateScore:
    """Return a scalar score; lower is better."""
    raw = quality.raw_fit_metrics
    motion = quality.motion_metrics
    profile = _regime_motion_profile(segment.regime_label)

    objective_motion_scale = {
        "position": 0.45,
        "balanced": 1.0,
        "smooth": 2.0,
    }[str(tuning.objective)]
    objective_position_scale = {
        "position": 0.85,
        "balanced": 1.0,
        "smooth": 1.20,
    }[str(tuning.objective)]

    rmse = float(raw.get("rmse_3d_m", 0.0))
    p95 = float(raw.get("p95_error_3d_m", rmse))
    max_err = float(raw.get("max_error_3d_m", p95))
    position_cost = objective_position_scale * (
        rmse
        + float(tuning.p95_position_weight) * p95
        + float(tuning.max_position_weight) * max_err
    )

    accel_rms = float(motion.get("accel_rms_mps2", 0.0))
    jerk_rms = float(motion.get("jerk_rms_mps3", 0.0))
    accel_ratio = accel_rms / max(float(profile["accel_target_mps2"]), 1e-6)
    jerk_ratio = jerk_rms / max(float(profile["jerk_target_mps3"]), 1e-6)
    motion_cost = (
        float(tuning.motion_cost_scale_m)
        * objective_motion_scale
        * float(profile["smoothness_weight"])
        * (accel_ratio * accel_ratio + 0.6 * jerk_ratio * jerk_ratio)
    )

    n_obs = max(int(getattr(fit, "n_observations", 0) or 0), 1)
    n_basis = int(fit.diagnostics.get("n_basis") or getattr(fit, "coefficients", []).__len__() or 0)
    complexity_cost = float(tuning.complexity_cost_scale_m) * (float(n_basis) / float(n_obs)) ** 2
    duration_s = float(segment.features.get("duration_s", max(float(segment.t1) - float(segment.t0), 0.0)) or 0.0)
    short_segment_factor = max(0.0, min(1.0, (75.0 - duration_s) / 75.0))
    regime = str(segment.regime_label).lower()
    transition_like = any(token in regime for token in ("transition", "turn", "energy_gain", "energy_loss", "climb", "descent"))
    join_artifact_cost = 0.0
    if transition_like and short_segment_factor > 0.0:
        density = float(n_basis) / float(n_obs)
        join_artifact_cost = (
            float(tuning.join_artifact_cost_scale_m)
            * short_segment_factor
            * (density * density + 0.10 * jerk_ratio * jerk_ratio)
        )
    score = float(position_cost + motion_cost + complexity_cost + join_artifact_cost)
    return CandidateScore(
        score=score,
        components={
            "position_cost_m": float(position_cost),
            "motion_cost_m_equivalent": float(motion_cost),
            "complexity_cost_m_equivalent": float(complexity_cost),
            "join_artifact_cost_m_equivalent": float(join_artifact_cost),
            "short_segment_factor": float(short_segment_factor),
            "rmse_3d_m": rmse,
            "p95_error_3d_m": p95,
            "max_error_3d_m": max_err,
            "accel_rms_mps2": accel_rms,
            "jerk_rms_mps3": jerk_rms,
            "n_basis": float(n_basis),
            "n_observations": float(n_obs),
        },
        profile=profile,
    )


def _targeted_short_transition_recipes(
    selected: SelectedBSplineSegmentParams,
) -> list[tuple[str, float, float, float, float, float, float]]:
    """Report-driven recipes for short transition/energy-change windows.

    The generic grid used to include several ways to make a segment more
    flexible but too few ways to make a short, transition-like segment *less*
    join-sensitive.  These recipes deliberately reduce velocity trust, widen
    knots, require more observations per basis, and strengthen derivative
    penalties.
    """
    regime = str(selected.regime_label).lower()
    features = selected.features or {}
    duration_s = float(features.get("duration_s", 0.0) or 0.0)
    n_obs = float(features.get("n_observations", 0.0) or 0.0)
    transition_like = any(token in regime for token in ("transition", "turn", "energy_gain", "energy_loss", "climb", "descent"))
    short_or_sparse = (duration_s > 0.0 and duration_s <= 90.0) or (n_obs > 0.0 and n_obs <= 80.0)
    if not (transition_like and short_or_sparse):
        return []
    return [
        ("short_transition_sparse_low_velocity", 1.60, 0.20, 1.35, 1.50, 2.40, 1.60),
        ("short_transition_strong_derivative_guard", 2.40, 0.15, 1.60, 1.80, 3.20, 2.00),
        ("energy_change_sparse_low_velocity", 1.40, 0.25, 1.25, 1.45, 2.00, 1.40),
    ]


def _candidate_from_recipe(
    *,
    selected: SelectedBSplineSegmentParams,
    base_config: BSplineCoreConfig,
    tuning: LocalSegmentTuningConfig,
    name: str,
    eta_mul: float,
    vel_mul: float,
    knot_mul: float,
    obs_mul: float,
    jerk_mul: float,
    acc_mul: float,
) -> SelectedBSplineSegmentParams:
    adaptive_eta = _clip(float(selected.adaptive_eta) * float(eta_mul), tuning.min_adaptive_eta, tuning.max_adaptive_eta)
    smoothing_lambda = _clip(float(selected.smoothing_lambda) * float(eta_mul), tuning.min_smoothing_lambda, tuning.max_smoothing_lambda)
    velocity_weight = _clip(float(selected.velocity_weight) * float(vel_mul), tuning.min_velocity_weight, tuning.max_velocity_weight)
    knot_spacing = _clip(float(selected.knot_spacing_s) * float(knot_mul), max(tuning.min_knot_spacing_s, float(base_config.min_knot_spacing_s)), tuning.max_knot_spacing_s)
    min_obs = _clip(float(selected.min_observations_per_basis) * float(obs_mul), tuning.min_observations_per_basis, tuning.max_observations_per_basis)
    jerk = _clip(float(selected.jerk_penalty_weight) * float(jerk_mul), tuning.min_jerk_penalty_weight, tuning.max_jerk_penalty_weight)
    acc_prior = _clip(float(selected.boundary_acceleration_prior_weight) * float(acc_mul), tuning.min_acceleration_prior_weight, tuning.max_acceleration_prior_weight)
    acc_mult = _clip(float(selected.acceleration_penalty_multiplier) * float(acc_mul), tuning.min_acceleration_multiplier, tuning.max_acceleration_multiplier)
    return replace(
        selected,
        smoothing_lambda=smoothing_lambda,
        adaptive_eta=adaptive_eta,
        velocity_weight=velocity_weight,
        knot_spacing_s=knot_spacing,
        min_observations_per_basis=min_obs,
        jerk_penalty_weight=jerk,
        boundary_acceleration_prior_weight=acc_prior,
        acceleration_penalty_multiplier=acc_mult,
        reason=f"local_tuning_candidate:{name};base={selected.reason}",
    )


def _candidate_key(candidate: SelectedBSplineSegmentParams) -> tuple[Any, ...]:
    return (
        candidate.penalty_mode,
        round(float(candidate.smoothing_lambda), 6),
        round(float(candidate.adaptive_eta), 6),
        round(float(candidate.velocity_weight), 6),
        round(float(candidate.knot_spacing_s), 6),
        round(float(candidate.min_observations_per_basis), 6),
        round(float(candidate.jerk_penalty_weight), 8),
        round(float(candidate.boundary_acceleration_prior_weight), 8),
        round(float(candidate.acceleration_penalty_multiplier), 6),
    )


def _regime_motion_profile(regime_label: str) -> dict[str, float]:
    regime = str(regime_label).lower()
    if "rough" in regime or "noisy" in regime or "surveillance" in regime:
        return {"accel_target_mps2": 3.5, "jerk_target_mps3": 2.5, "smoothness_weight": 1.75}
    if "turn" in regime or "transition" in regime:
        return {"accel_target_mps2": 7.0, "jerk_target_mps3": 6.0, "smoothness_weight": 0.65}
    if "energy_constant" in regime:
        return {"accel_target_mps2": 3.0, "jerk_target_mps3": 2.0, "smoothness_weight": 1.35}
    if "energy_gain" in regime or "energy_loss" in regime or "climb" in regime or "descent" in regime:
        return {"accel_target_mps2": 5.0, "jerk_target_mps3": 4.0, "smoothness_weight": 1.0}
    if "ground" in regime:
        return {"accel_target_mps2": 2.0, "jerk_target_mps3": 1.5, "smoothness_weight": 1.2}
    return {"accel_target_mps2": 4.5, "jerk_target_mps3": 3.5, "smoothness_weight": 1.0}


def _clip(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(float(value)):
        return float(lo)
    return float(min(max(float(value), float(lo)), float(hi)))
