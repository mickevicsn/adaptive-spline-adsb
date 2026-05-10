"""Energy/regime based local policies for B-spline V-Spline segments.

The segmentation layer decides *where* the aircraft state changes.  This module
keeps the fitting policy separate and decides *how much freedom* each local
B-spline segment gets: knot density, derivative smoothing, velocity trust, and
soft acceleration-prior strength.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from trajectory_segmentation import DynamicSegment
from .bspline_core import BSplineCoreConfig


@dataclass(frozen=True)
class LocalSegmentPolicyConfig:
    """Rule-based first guess for each local B-spline V-Spline segment.

    The final segment parameters may be refined by the local segment tuner, but
    the tuner always starts from these regime-aware values.  That keeps the
    search small and interpretable: steady energy states are smoother, turns and
    transitions get more local freedom, and noisy segments trust raw velocity
    less while penalizing derivatives more strongly.
    """

    enable_rule_based_params: bool = True
    ground_speed_threshold_mps: float = 25.0

    # V-Spline smoothing and velocity-trust controls.  In adaptive mode,
    # adaptive_eta is the main scale for the acceleration penalty; larger values
    # produce smoother motion within that regime.
    steady_adaptive_eta: float = 100_000.0
    steady_velocity_weight: float = 0.03
    # 4BAAD9 report action: transition/energy-change windows were getting too
    # much local freedom and too much velocity trust exactly where joins are
    # most expensive.  Start these regimes smoother and sparser.
    transition_adaptive_eta: float = 18_000.0
    transition_velocity_weight: float = 0.025
    energy_change_adaptive_eta: float = 45_000.0
    energy_change_velocity_weight: float = 0.025
    energy_constant_adaptive_eta: float = 150_000.0
    energy_constant_velocity_weight: float = 0.02
    noisy_adaptive_eta: float = 250_000.0
    noisy_velocity_weight: float = 0.01
    ground_smoothing_lambda: float = 100.0
    ground_velocity_weight: float = 0.01

    # B-spline controls.  These decide local model capacity and soft C2 pressure.
    bspline_steady_knot_spacing_s: float = 5.0
    bspline_steady_min_observations_per_basis: float = 7.0
    bspline_steady_jerk_penalty_weight: float = 0.002
    bspline_steady_acceleration_prior_weight: float = 0.08
    bspline_steady_acceleration_multiplier: float = 1.0

    bspline_transition_knot_spacing_s: float = 4.0
    bspline_transition_min_observations_per_basis: float = 5.5
    bspline_transition_jerk_penalty_weight: float = 0.004
    bspline_transition_acceleration_prior_weight: float = 0.08
    bspline_transition_acceleration_multiplier: float = 0.90

    bspline_energy_change_knot_spacing_s: float = 4.75
    bspline_energy_change_min_observations_per_basis: float = 6.5
    bspline_energy_change_jerk_penalty_weight: float = 0.004
    bspline_energy_change_acceleration_prior_weight: float = 0.10
    bspline_energy_change_acceleration_multiplier: float = 1.00

    bspline_energy_constant_knot_spacing_s: float = 7.0
    bspline_energy_constant_min_observations_per_basis: float = 10.0
    bspline_energy_constant_jerk_penalty_weight: float = 0.006
    bspline_energy_constant_acceleration_prior_weight: float = 0.12
    bspline_energy_constant_acceleration_multiplier: float = 1.25

    bspline_noisy_knot_spacing_s: float = 8.0
    bspline_noisy_min_observations_per_basis: float = 12.0
    bspline_noisy_jerk_penalty_weight: float = 0.012
    bspline_noisy_acceleration_prior_weight: float = 0.16
    bspline_noisy_acceleration_multiplier: float = 1.75

    bspline_ground_knot_spacing_s: float = 4.0
    bspline_ground_min_observations_per_basis: float = 4.0
    bspline_ground_jerk_penalty_weight: float = 0.004
    bspline_ground_acceleration_prior_weight: float = 0.04
    bspline_ground_acceleration_multiplier: float = 1.0


@dataclass(frozen=True)
class SelectedBSplineSegmentParams:
    """Concrete local parameters selected for one dynamic segment."""

    penalty_mode: Literal["constant", "adaptive"]
    smoothing_lambda: float
    adaptive_eta: float
    velocity_weight: float
    knot_spacing_s: float
    min_observations_per_basis: float
    jerk_penalty_weight: float
    boundary_acceleration_prior_weight: float
    acceleration_penalty_multiplier: float
    reason: str
    regime_label: str
    features: dict[str, float]

    def to_core_config(self, base: BSplineCoreConfig) -> BSplineCoreConfig:
        """Apply this policy to the global B-spline core defaults.

        Most local policies are free to override the base preset.  The
        aviation quintic accurate/balanced variants are different: their base
        config carries endpoint-artifact guards discovered on 4BAAD9.  Preserve
        those floors so local candidate recipes cannot silently turn the quintic
        backend back into a dense low-smoothing endpoint-sensitive fit.
        """
        backend_name = str(base.backend_name).lower()
        endpoint_tuned_quintic = "aviation_v_spline_quintic" in backend_name and not backend_name.endswith("_smooth")

        velocity_weight = float(self.velocity_weight)
        min_observations_per_basis = float(self.min_observations_per_basis)
        jerk_penalty_weight = float(self.jerk_penalty_weight)
        boundary_acceleration_prior_weight = float(self.boundary_acceleration_prior_weight)
        if endpoint_tuned_quintic:
            velocity_weight = min(velocity_weight, float(base.velocity_weight))
            min_observations_per_basis = max(min_observations_per_basis, float(base.min_observations_per_basis))
            jerk_penalty_weight = max(jerk_penalty_weight, float(base.jerk_penalty_weight))
            boundary_acceleration_prior_weight = max(
                boundary_acceleration_prior_weight,
                float(base.boundary_acceleration_prior_weight),
            )

        return replace(
            base,
            penalty_mode=self.penalty_mode,
            smoothing_lambda=float(self.smoothing_lambda),
            adaptive_eta=float(self.adaptive_eta),
            velocity_weight=velocity_weight,
            knot_spacing_s=float(self.knot_spacing_s),
            min_observations_per_basis=min_observations_per_basis,
            jerk_penalty_weight=jerk_penalty_weight,
            boundary_acceleration_prior_weight=boundary_acceleration_prior_weight,
            acceleration_penalty_multiplier=float(self.acceleration_penalty_multiplier),
            # Preserve the backend family policy.  Legacy cubic baseline can still
            # request hard raw boundary anchors, while overlap/join-smooth/quintic
            # aviation variants disable hard anchors and use robust soft position
            # priors instead.
            hard_boundary_positions=bool(base.hard_boundary_positions),
            hard_component_endpoint_positions=bool(base.hard_component_endpoint_positions),
            hard_component_endpoint_velocities=bool(base.hard_component_endpoint_velocities),
            component_endpoint_velocity_prior_weight=float(base.component_endpoint_velocity_prior_weight),
            component_endpoint_acceleration_prior_weight=float(base.component_endpoint_acceleration_prior_weight),
            endpoint_guard_window_s=float(base.endpoint_guard_window_s),
            endpoint_jerk_penalty_multiplier=float(base.endpoint_jerk_penalty_multiplier),
            endpoint_snap_penalty_multiplier=float(base.endpoint_snap_penalty_multiplier),
            # Internal join velocities are hard shared constraints; adding a
            # separate soft velocity prior at the same boundary double-counts it.
            boundary_velocity_prior_weight=0.0,
            backend_name=f"{base.backend_name}:local_segment_energy_state_policy",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _regime_flags(segment: DynamicSegment, policy: LocalSegmentPolicyConfig) -> dict[str, Any]:
    regime = str(segment.regime_label)
    features = segment.features
    mismatch = float(features.get("median_velocity_mismatch_mps", 0.0))
    speed = float(features.get("median_horizontal_speed_mps", 0.0))
    high_mismatch = mismatch > max(10.0, 0.25 * max(speed, 1.0))
    ground = regime == "ground_slow" or speed < float(policy.ground_speed_threshold_mps)
    transition = regime == "turn_or_transition" or regime.endswith("_turn") or "turn_" in regime
    noisy = (
        regime.endswith("_noisy")
        or regime == "noisy_airborne"
        or "rough_air" in regime
        or "surveillance_noisy" in regime
        or high_mismatch
    )
    energy_change = "energy_gain" in regime or "energy_loss" in regime or "energy_exchange" in regime
    energy_constant = regime == "energy_constant" or regime.startswith("energy_constant__")
    return {
        "regime": regime,
        "features": features,
        "mismatch": mismatch,
        "speed": speed,
        "high_mismatch": high_mismatch,
        "ground": ground,
        "transition": transition,
        "noisy": noisy,
        "energy_change": energy_change,
        "energy_constant": energy_constant,
    }


def select_local_bspline_params(
    segment: DynamicSegment,
    base_config: BSplineCoreConfig,
    policy: LocalSegmentPolicyConfig,
) -> SelectedBSplineSegmentParams:
    """Choose deterministic local B-spline parameters from segment energy state."""
    flags = _regime_flags(segment, policy)
    regime = flags["regime"]
    features = flags["features"]
    high_mismatch = bool(flags["high_mismatch"])

    if not policy.enable_rule_based_params:
        return SelectedBSplineSegmentParams(
            penalty_mode=base_config.penalty_mode,
            smoothing_lambda=float(base_config.smoothing_lambda),
            adaptive_eta=float(base_config.adaptive_eta),
            velocity_weight=float(base_config.velocity_weight),
            knot_spacing_s=float(base_config.knot_spacing_s),
            min_observations_per_basis=float(base_config.min_observations_per_basis),
            jerk_penalty_weight=float(base_config.jerk_penalty_weight),
            boundary_acceleration_prior_weight=float(base_config.boundary_acceleration_prior_weight),
            acceleration_penalty_multiplier=float(base_config.acceleration_penalty_multiplier),
            reason="rule_based_policy_disabled_use_base_config",
            regime_label=regime,
            features=features,
        )

    if flags["ground"]:
        return SelectedBSplineSegmentParams(
            penalty_mode="constant",
            smoothing_lambda=float(policy.ground_smoothing_lambda),
            adaptive_eta=float(base_config.adaptive_eta),
            velocity_weight=float(policy.ground_velocity_weight),
            knot_spacing_s=float(policy.bspline_ground_knot_spacing_s),
            min_observations_per_basis=float(policy.bspline_ground_min_observations_per_basis),
            jerk_penalty_weight=float(policy.bspline_ground_jerk_penalty_weight),
            boundary_acceleration_prior_weight=float(policy.bspline_ground_acceleration_prior_weight),
            acceleration_penalty_multiplier=float(policy.bspline_ground_acceleration_multiplier),
            reason="ground_or_very_slow_segment_constant_penalty_low_velocity_weight",
            regime_label=regime,
            features=features,
        )

    if flags["transition"]:
        return SelectedBSplineSegmentParams(
            penalty_mode="adaptive",
            smoothing_lambda=float(base_config.smoothing_lambda),
            adaptive_eta=float(policy.transition_adaptive_eta),
            velocity_weight=float(policy.noisy_velocity_weight if high_mismatch else policy.transition_velocity_weight),
            knot_spacing_s=float(policy.bspline_transition_knot_spacing_s),
            min_observations_per_basis=float(policy.bspline_transition_min_observations_per_basis),
            jerk_penalty_weight=float(policy.bspline_transition_jerk_penalty_weight),
            boundary_acceleration_prior_weight=float(policy.bspline_transition_acceleration_prior_weight),
            acceleration_penalty_multiplier=float(policy.bspline_transition_acceleration_multiplier),
            reason="turn_or_transition_more_knots_shared_boundary_velocity" + ("_velocity_downweighted" if high_mismatch else ""),
            regime_label=regime,
            features=features,
        )

    if flags["noisy"]:
        return SelectedBSplineSegmentParams(
            penalty_mode="adaptive",
            smoothing_lambda=float(base_config.smoothing_lambda),
            adaptive_eta=float(policy.noisy_adaptive_eta),
            velocity_weight=float(policy.noisy_velocity_weight),
            knot_spacing_s=float(policy.bspline_noisy_knot_spacing_s),
            min_observations_per_basis=float(policy.bspline_noisy_min_observations_per_basis),
            jerk_penalty_weight=float(policy.bspline_noisy_jerk_penalty_weight),
            boundary_acceleration_prior_weight=float(policy.bspline_noisy_acceleration_prior_weight),
            acceleration_penalty_multiplier=float(policy.bspline_noisy_acceleration_multiplier),
            reason="noisy_segment_fewer_knots_low_velocity_weight_stronger_derivative_smoothing",
            regime_label=regime,
            features=features,
        )

    if flags["energy_change"]:
        return SelectedBSplineSegmentParams(
            penalty_mode="adaptive",
            smoothing_lambda=float(base_config.smoothing_lambda),
            adaptive_eta=float(policy.energy_change_adaptive_eta),
            velocity_weight=float(policy.noisy_velocity_weight if high_mismatch else policy.energy_change_velocity_weight),
            knot_spacing_s=float(policy.bspline_energy_change_knot_spacing_s),
            min_observations_per_basis=float(policy.bspline_energy_change_min_observations_per_basis),
            jerk_penalty_weight=float(policy.bspline_energy_change_jerk_penalty_weight),
            boundary_acceleration_prior_weight=float(policy.bspline_energy_change_acceleration_prior_weight),
            acceleration_penalty_multiplier=float(policy.bspline_energy_change_acceleration_multiplier),
            reason="sustained_energy_change_balanced_position_and_motion" + ("_velocity_downweighted" if high_mismatch else ""),
            regime_label=regime,
            features=features,
        )

    if flags["energy_constant"]:
        return SelectedBSplineSegmentParams(
            penalty_mode="adaptive",
            smoothing_lambda=float(base_config.smoothing_lambda),
            adaptive_eta=float(policy.energy_constant_adaptive_eta),
            velocity_weight=float(policy.noisy_velocity_weight if high_mismatch else policy.energy_constant_velocity_weight),
            knot_spacing_s=float(policy.bspline_energy_constant_knot_spacing_s),
            min_observations_per_basis=float(policy.bspline_energy_constant_min_observations_per_basis),
            jerk_penalty_weight=float(policy.bspline_energy_constant_jerk_penalty_weight),
            boundary_acceleration_prior_weight=float(policy.bspline_energy_constant_acceleration_prior_weight),
            acceleration_penalty_multiplier=float(policy.bspline_energy_constant_acceleration_multiplier),
            reason="energy_constant_segment_smoothest_airborne_policy" + ("_velocity_downweighted" if high_mismatch else ""),
            regime_label=regime,
            features=features,
        )

    return SelectedBSplineSegmentParams(
        penalty_mode="adaptive",
        smoothing_lambda=float(base_config.smoothing_lambda),
        adaptive_eta=float(policy.steady_adaptive_eta),
        velocity_weight=float(policy.steady_velocity_weight),
        knot_spacing_s=float(policy.bspline_steady_knot_spacing_s),
        min_observations_per_basis=float(policy.bspline_steady_min_observations_per_basis),
        jerk_penalty_weight=float(policy.bspline_steady_jerk_penalty_weight),
        boundary_acceleration_prior_weight=float(policy.bspline_steady_acceleration_prior_weight),
        acceleration_penalty_multiplier=float(policy.bspline_steady_acceleration_multiplier),
        reason="steady_airborne_default",
        regime_label=regime,
        features=features,
    )
