"""Dynamic ADS-B trajectory segmentation for local V-Spline fitting.

This module keeps segmentation outside the raw-keyframe adapter and outside the
mathematical V-Spline core.  It works with prepared paired samples and returns
hard-gap connected components plus motion-regime segments inside each component.

The implementation is deterministic.  If ``ruptures`` is installed, PELT is used
on robust-standardized low-dimensional motion features.  If it is not installed,
the same feature matrix is used with a robust threshold candidate generator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Sequence

import math

import numpy as np

from raw_keyframe_vspline_adapter import PreparedVSplineSample
from segmentation_kalman import KalmanSegmentationConfig, smooth_samples_for_segmentation
STANDARD_GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class DynamicSegmentationConfig:
    """Configuration for hard-gap and motion-regime segmentation."""

    enabled: bool = True
    hard_gap_s: float | None = 30.0
    relative_gap_factor: float = 5.0
    min_segment_points: int = 8
    # 4BAAD9 report action: local spline backends were paying too much join
    # cost.  Default to coarser segments; callers can still lower these for a
    # diagnostic over-segmentation ablation.
    min_segment_duration_s: float = 15.0
    min_boundary_spacing_s: float = 12.0
    pelt_model: Literal["l2", "rank", "rbf"] = "l2"
    pelt_jump: int = 1
    pelt_penalty_scale: float = 3.0
    use_pelt_if_available: bool = True
    candidate_score_z: float = 5.0
    max_segments_per_component: int = 48
    prefer_under_segmentation: bool = True

    # Regime-state segmentation is now the primary segmentation objective.  It
    # remains conservative: boundaries require sustained state changes, not
    # one-sample heading/acceleration spikes.
    enable_motion_spike_boundaries: bool = False
    enable_pelt_boundaries: bool = False
    pelt_energy_features_only: bool = True

    # Regime-state segmentation.  The previous default deliberately ignored
    # horizontal turns so segment colors represented only energy changes.  For
    # reconstruction/debugging it is more useful to label sustained kinematic
    # regimes: climb/descent/speed-change + sustained turn + rough/noisy air.
    # These are still run-length/hysteresis based; isolated spikes do not create
    # accepted boundaries.
    segment_horizontal_turns: bool = True
    turn_rate_deadband_degps: float = 0.75
    turn_rate_deadband_scale: float = 0.20
    lateral_accel_deadband_mps2: float = 0.5
    lateral_accel_deadband_scale: float = 0.25
    turn_rate_medium_threshold_degps: float = 3.0
    turn_rate_rapid_threshold_degps: float = 7.0
    turn_min_heading_change_deg: float = 12.0

    enable_rough_air_segmentation: bool = False
    rough_air_score_threshold: float = 4.5
    rough_air_velocity_mismatch_weight: float = 0.45
    rough_air_residual_weight: float = 0.45
    rough_air_vertical_accel_weight: float = 0.35
    rough_air_speed_accel_weight: float = 0.25

    # Energy-state segmentation.  These settings make boundaries depend on
    # sustained changes in aircraft energy behavior, not isolated noisy spikes.
    enable_energy_state_segmentation: bool = True
    energy_state_min_points: int = 8
    energy_state_min_duration_s: float = 15.0
    energy_ground_speed_threshold_mps: float = 25.0
    energy_rate_deadband_mps: float = 1.0
    energy_rate_deadband_scale: float = 0.20
    vertical_rate_deadband_mps: float = 1.0
    vertical_rate_deadband_scale: float = 0.20
    speed_accel_deadband_mps2: float = 0.20
    speed_accel_deadband_scale: float = 0.20

    # Magnitude buckets for the viewer/debugger.  These make sustained slow,
    # medium, and rapid climbs/descents separate states, while a horizontal turn
    # with approximately constant total energy remains one state.
    energy_rate_medium_threshold_mps: float = 2.0
    energy_rate_rapid_threshold_mps: float = 6.0
    vertical_rate_medium_threshold_mps: float = 2.0
    vertical_rate_rapid_threshold_mps: float = 6.0
    speed_accel_medium_threshold_mps2: float = 0.25
    speed_accel_rapid_threshold_mps2: float = 0.8

    energy_boundary_score: float = 25.0
    protect_energy_boundaries: bool = True
    energy_smoothing_window_points: int = 7

    # Boundary placement.  A state change tells us a split is needed; it does
    # not mean the first sample of the new state is the best mathematical join.
    # Search a small transition band and prefer a real paired sample with low
    # position/velocity inconsistency and low local residual.  This keeps hard
    # boundary anchors away from obvious surveillance-noise spikes.
    optimize_boundary_sample: bool = True
    boundary_search_padding_points: int = 3
    # Cap optimized boundary relocation around the detected state transition.
    # Flight 4BAAD9 exposed a failure mode where the optimizer searched the
    # whole unstable transition band and moved a join 68 samples backward; that
    # creates artificial segment endpoints rather than cleaner joins.  ``None``
    # or a negative value restores the unrestricted diagnostic behavior.
    max_boundary_shift_points: int | None = 12
    boundary_quality_velocity_mismatch_weight: float = 0.45
    boundary_quality_residual_weight: float = 0.45
    boundary_quality_acceleration_weight: float = 0.20

    # Go-around / missed-approach detection.  This is intentionally separate
    # from total-energy-rate segmentation: during go-around initiation the
    # aircraft can briefly trade speed/altitude or pause near zero vertical
    # speed before the positive total-energy-rate phase becomes obvious.
    # Detect it as a sustained descent -> sustained climb vertical-mode
    # reversal around a local altitude minimum.
    enable_go_around_detection: bool = True
    go_around_min_points: int = 4
    go_around_min_duration_s: float = 6.0
    go_around_max_transition_gap_s: float = 25.0
    go_around_max_transition_gap_points: int = 8
    go_around_min_altitude_reversal_m: float = 10.0
    go_around_boundary_score: float = 60.0

    # Generic vertical-mode reversal / level-off detection.  Go-around detection
    # only catches the operationally important descent -> climb case.  A local
    # aviation spline also benefits from splitting other large vertical lobes
    # (climb -> level/descent, descent -> level) because a single cubic segment
    # otherwise smears the top/bottom of the maneuver and creates exactly the
    # endpoint/interior artifacts seen in altitude plots.  This detector is
    # conservative: it requires sustained support and a real altitude excursion,
    # so small energy changes still stay in one segment.
    enable_vertical_reversal_segmentation: bool = True
    vertical_reversal_min_points: int = 4
    vertical_reversal_min_duration_s: float = 6.0
    vertical_reversal_max_transition_gap_s: float = 25.0
    vertical_reversal_max_transition_gap_points: int = 8
    vertical_reversal_min_altitude_excursion_m: float = 25.0
    vertical_reversal_boundary_score: float = 55.0

    # Altitude-lobe segmentation is a second, more direct guard against the
    # failure mode where total-energy labels call a long interval
    # ``energy_constant`` even though the aircraft clearly climbs to a local
    # maximum and then descends, or descends to a local minimum and then climbs.
    # This is intentionally based on altitude-shape prominence rather than
    # energy-rate labels, so speed/height exchange does not hide go-around or
    # descent state changes inside one large spline fit.
    enable_altitude_lobe_segmentation: bool = True
    altitude_lobe_min_points: int = 8
    altitude_lobe_min_duration_s: float = 8.0
    altitude_lobe_min_prominence_m: float = 35.0
    altitude_lobe_min_side_prominence_m: float = 12.0
    altitude_lobe_gradient_gate_mps: float = 0.35
    altitude_lobe_boundary_score: float = 70.0

    # Feature source for segmentation.  ``kalman_rts`` uses a lightweight
    # 3D constant-velocity Kalman filter + RTS smoother only for segmentation
    # features and segment labels.  The final V-Spline still fits raw paired
    # observations, so this does not hide raw data from the reconstruction.
    segmentation_feature_source: Literal["raw", "kalman_rts"] = "kalman_rts"
    kalman_segmentation_config: KalmanSegmentationConfig = field(default_factory=KalmanSegmentationConfig)


@dataclass(frozen=True)
class HardGapComponent:
    """A connected component where C1 continuity may be enforced internally."""

    component_id: str
    start_sample_index: int
    end_sample_index: int  # inclusive index into the full prepared sample list
    samples: tuple[PreparedVSplineSample, ...]
    hard_gap_before_s: float | None = None
    hard_gap_after_s: float | None = None

    @property
    def t0(self) -> float:
        return float(self.samples[0].t)

    @property
    def t1(self) -> float:
        return float(self.samples[-1].t)

    @property
    def n_observations(self) -> int:
        return len(self.samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "start_sample_index": self.start_sample_index,
            "end_sample_index": self.end_sample_index,
            "n_observations": self.n_observations,
            "t0": self.t0,
            "t1": self.t1,
            "hard_gap_before_s": self.hard_gap_before_s,
            "hard_gap_after_s": self.hard_gap_after_s,
        }


@dataclass(frozen=True)
class SegmentationBoundaryCandidate:
    sample_index: int  # global prepared-sample index used as shared boundary sample
    t_boundary: float
    score: float
    reasons: tuple[str, ...]
    feature_snapshot: dict[str, float]


@dataclass(frozen=True)
class AcceptedBoundary:
    boundary_id: str
    component_id: str
    sample_index: int  # global prepared-sample index
    local_sample_index: int
    t_boundary: float
    reasons: tuple[str, ...]
    score: float
    is_hard_gap: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicSegment:
    """A motion-regime segment, inclusive of its shared boundary samples."""

    segment_id: str
    component_id: str
    start_sample_index: int
    end_sample_index: int  # inclusive global prepared-sample index
    samples: tuple[PreparedVSplineSample, ...]
    t0: float
    t1: float
    features: dict[str, float]
    regime_label: str
    start_boundary_id: str | None = None
    end_boundary_id: str | None = None

    @property
    def n_observations(self) -> int:
        return len(self.samples)

    @property
    def dt_min_s(self) -> float | None:
        if len(self.samples) < 2:
            return None
        return float(np.min(np.diff([s.t for s in self.samples])))

    @property
    def dt_max_s(self) -> float | None:
        if len(self.samples) < 2:
            return None
        return float(np.max(np.diff([s.t for s in self.samples])))

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "component_id": self.component_id,
            "start_sample_index": self.start_sample_index,
            "end_sample_index": self.end_sample_index,
            "n_observations": self.n_observations,
            "t0": self.t0,
            "t1": self.t1,
            "dt_min_s": self.dt_min_s,
            "dt_max_s": self.dt_max_s,
            "regime_label": self.regime_label,
            "features": self.features,
            "start_boundary_id": self.start_boundary_id,
            "end_boundary_id": self.end_boundary_id,
        }


@dataclass(frozen=True)
class SegmentedComponent:
    component: HardGapComponent
    boundaries: tuple[AcceptedBoundary, ...]
    segments: tuple[DynamicSegment, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def split_into_hard_gap_components(
    samples: Sequence[PreparedVSplineSample],
    config: DynamicSegmentationConfig,
) -> tuple[list[HardGapComponent], dict[str, Any]]:
    """Split prepared samples into disconnected components by hard time gaps."""
    if not samples:
        raise ValueError("Cannot split an empty sample list")

    t = np.asarray([s.t for s in samples], dtype=float)
    dt = np.diff(t)
    if dt.size:
        positive_dt = dt[dt > 0]
        med = float(np.median(positive_dt)) if positive_dt.size else 0.0
        robust_gap = med * float(config.relative_gap_factor)
    else:
        robust_gap = math.inf

    hard_gap = config.hard_gap_s
    if hard_gap is None:
        threshold = robust_gap if math.isfinite(robust_gap) and robust_gap > 0 else math.inf
    else:
        threshold = max(float(hard_gap), robust_gap) if math.isfinite(robust_gap) else float(hard_gap)

    split_after_local = [int(i) for i, gap in enumerate(dt) if gap > threshold]

    components: list[HardGapComponent] = []
    start = 0
    for comp_idx, split_after in enumerate(split_after_local + [len(samples) - 1], start=1):
        end = int(split_after)
        if end < start:
            continue
        group = tuple(samples[start : end + 1])
        if len(group) >= 2:
            before = None if start == 0 else float(t[start] - t[start - 1])
            after = None if end >= len(samples) - 1 else float(t[end + 1] - t[end])
            components.append(
                HardGapComponent(
                    component_id=f"comp_{len(components) + 1:04d}",
                    start_sample_index=start,
                    end_sample_index=end,
                    samples=group,
                    hard_gap_before_s=before,
                    hard_gap_after_s=after,
                )
            )
        start = end + 1

    if not components:
        raise ValueError("No usable hard-gap components with at least 2 samples")

    return components, {
        "hard_gap_threshold_s": None if threshold == math.inf else float(threshold),
        "requested_hard_gap_s": config.hard_gap_s,
        "relative_gap_factor": float(config.relative_gap_factor),
        "component_count": len(components),
        "component_lengths": [c.n_observations for c in components],
        "split_after_local_indices": split_after_local,
    }


def segment_component(
    component: HardGapComponent,
    config: DynamicSegmentationConfig,
) -> SegmentedComponent:
    """Return dynamic motion-regime segments for one hard-gap component."""
    n = component.n_observations
    if not config.enabled or n < max(2 * config.min_segment_points - 1, 4):
        segment = _make_dynamic_segment(
            component,
            component_features=None,
            start_local=0,
            end_local=n - 1,
            segment_number=1,
            start_boundary_id=None,
            end_boundary_id=None,
        )
        return SegmentedComponent(
            component=component,
            boundaries=(),
            segments=(segment,),
            diagnostics={
                "enabled": bool(config.enabled),
                "reason": "component too small for dynamic segmentation" if config.enabled else "disabled",
                "candidate_count": 0,
                "accepted_boundary_count": 0,
                "segment_count": 1,
            },
        )

    features = compute_feature_matrix(component.samples, config)
    candidates = propose_boundary_candidates(component, features, config)
    accepted = _cleanup_candidates(component, candidates, config)

    if len(accepted) + 1 > int(config.max_segments_per_component):
        accepted = _limit_boundaries_by_score(component, accepted, config)

    local_breaks = [b.local_sample_index for b in accepted]
    local_starts = [0] + local_breaks
    local_ends = local_breaks + [n - 1]

    segments: list[DynamicSegment] = []
    for idx, (start_local, end_local) in enumerate(zip(local_starts, local_ends), start=1):
        start_boundary_id = accepted[idx - 2].boundary_id if idx > 1 and accepted else None
        end_boundary_id = accepted[idx - 1].boundary_id if idx <= len(accepted) else None
        segments.append(
            _make_dynamic_segment(
                component,
                config=config,
                component_features=features,
                start_local=start_local,
                end_local=end_local,
                segment_number=idx,
                start_boundary_id=start_boundary_id,
                end_boundary_id=end_boundary_id,
            )
        )

    return SegmentedComponent(
        component=component,
        boundaries=tuple(accepted),
        segments=tuple(segments),
        diagnostics={
            "enabled": bool(config.enabled),
            "feature_names": features["feature_names"],
            "candidate_count": len(candidates),
            "accepted_boundary_count": len(accepted),
            "segment_count": len(segments),
            "pelt_available": bool(features.get("pelt_available", False)),
            "pelt_used": bool(features.get("pelt_used", False)),
            "candidate_preview": [asdict(c) for c in candidates[:20]],
        },
    )


def segment_prepared_samples(
    samples: Sequence[PreparedVSplineSample],
    config: DynamicSegmentationConfig,
) -> tuple[list[SegmentedComponent], dict[str, Any]]:
    components, hard_diag = split_into_hard_gap_components(samples, config)
    segmented = [segment_component(component, config) for component in components]
    return segmented, {
        "config": asdict(config),
        "hard_gap": hard_diag,
        "component_count": len(segmented),
        "segment_count": int(sum(len(c.segments) for c in segmented)),
        "boundary_count": int(sum(len(c.boundaries) for c in segmented)),
    }


def build_segmented_component_from_boundaries(
    component: HardGapComponent,
    boundaries: Sequence[AcceptedBoundary],
    config: DynamicSegmentationConfig,
    *,
    diagnostics_extra: dict[str, Any] | None = None,
) -> SegmentedComponent:
    """Build a ``SegmentedComponent`` from externally supplied boundaries.

    This is used by quality-triggered adaptive resegmentation: after a local
    fit proves that one initial segment is still too heterogeneous, the fitting
    layer can add a new feasible boundary and ask the segmentation layer to
    rebuild the segment objects consistently.  Boundaries are treated as shared
    samples, so each internal boundary sample is included in both neighbouring
    segments.
    """
    ordered = sorted(
        list(boundaries),
        key=lambda b: int(b.local_sample_index),
    )
    n = component.n_observations
    local_breaks = [int(b.local_sample_index) for b in ordered]
    local_starts = [0] + local_breaks
    local_ends = local_breaks + [n - 1]

    component_features: dict[str, Any] | None = None
    try:
        component_features = compute_feature_matrix(component.samples, config)
    except Exception:
        component_features = None

    segments: list[DynamicSegment] = []
    for idx, (start_local, end_local) in enumerate(zip(local_starts, local_ends), start=1):
        start_boundary_id = ordered[idx - 2].boundary_id if idx > 1 and ordered else None
        end_boundary_id = ordered[idx - 1].boundary_id if idx <= len(ordered) else None
        segments.append(
            _make_dynamic_segment(
                component,
                config=config,
                component_features=component_features,
                start_local=start_local,
                end_local=end_local,
                segment_number=idx,
                start_boundary_id=start_boundary_id,
                end_boundary_id=end_boundary_id,
            )
        )

    diagnostics = {
        "enabled": bool(config.enabled),
        "boundary_source": "external_supplied_boundaries",
        "accepted_boundary_count": len(ordered),
        "segment_count": len(segments),
    }
    if diagnostics_extra:
        diagnostics.update(diagnostics_extra)
    return SegmentedComponent(
        component=component,
        boundaries=tuple(ordered),
        segments=tuple(segments),
        diagnostics=diagnostics,
    )


def compute_feature_matrix(
    samples: Sequence[PreparedVSplineSample],
    config: DynamicSegmentationConfig | None = None,
) -> dict[str, Any]:
    t = np.asarray([s.t for s in samples], dtype=float)
    y_raw = np.asarray([s.y for s in samples], dtype=float)
    v_raw = np.asarray([s.v for s in samples], dtype=float)
    signal_source = str(getattr(config, "segmentation_feature_source", "raw") if config is not None else "raw")
    kalman_diag: dict[str, Any] = {"enabled": False, "used": False, "reason": "raw_feature_source"}
    if signal_source == "kalman_rts" and config is not None:
        y, v, kalman_diag = smooth_samples_for_segmentation(samples, config.kalman_segmentation_config)
    else:
        y, v = y_raw, v_raw
    n = len(samples)

    dt = np.diff(t)
    dt_safe = np.maximum(dt, 1e-6)
    pos_vel = np.diff(y, axis=0) / dt_safe[:, None]

    speed_h = np.linalg.norm(v[:, :2], axis=1)
    vertical_rate = v[:, 2]

    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    signed_heading_rate_interval = np.diff(heading) / dt_safe
    signed_heading_rate = _interval_to_sample_feature(signed_heading_rate_interval, n)
    heading_rate = np.abs(signed_heading_rate)
    lateral_acceleration = speed_h * signed_heading_rate
    abs_lateral_acceleration = np.abs(lateral_acceleration)

    # Signed kinematic / energy features.  These are deliberately first-class:
    # segmentation is supposed to find intervals where aircraft energy behavior
    # stays stable long enough, not only where the geometry bends sharply.
    speed_accel = _sample_gradient(speed_h, t)
    acceleration_interval = np.linalg.norm(np.diff(v, axis=0), axis=1) / dt_safe
    accel = _interval_to_sample_feature(acceleration_interval, n)

    vertical_accel_interval = np.diff(vertical_rate) / dt_safe
    vertical_accel = _interval_to_sample_feature(vertical_accel_interval, n)

    specific_energy_height = y[:, 2] + (speed_h**2) / (2.0 * STANDARD_GRAVITY_MPS2)
    specific_energy_rate = _sample_gradient(specific_energy_height, t)
    abs_specific_energy_rate = np.abs(specific_energy_rate)

    flight_path_angle = np.arctan2(vertical_rate, np.maximum(speed_h, 1e-6))

    pos_vel_sample = (
        np.vstack([pos_vel[0:1], 0.5 * (pos_vel[:-1] + pos_vel[1:]), pos_vel[-1:]])
        if n > 2
        else np.vstack([pos_vel, pos_vel[-1:]])
    )
    vel_mismatch = np.linalg.norm(v - pos_vel_sample, axis=1)

    # Cheap local residual proxy: distance from the straight chord over a small window.
    residual_proxy = _local_chord_residual(t, y, window=5)

    raw = np.column_stack(
        [
            _rolling_median(heading_rate, 3),
            _rolling_median(signed_heading_rate, 3),
            _rolling_median(lateral_acceleration, 3),
            _rolling_median(abs_lateral_acceleration, 3),
            _rolling_median(speed_h, 3),
            _rolling_median(vertical_rate, 3),
            _rolling_median(np.abs(vertical_rate), 3),
            _rolling_median(speed_accel, 3),
            _rolling_median(accel, 3),
            _rolling_median(vertical_accel, 3),
            specific_energy_height,
            _rolling_median(specific_energy_rate, 3),
            _rolling_median(abs_specific_energy_rate, 3),
            vel_mismatch,
            residual_proxy,
            flight_path_angle,
        ]
    )
    feature_names = [
        "heading_rate_radps",
        "signed_heading_rate_radps",
        "lateral_acceleration_mps2",
        "abs_lateral_acceleration_mps2",
        "horizontal_speed_mps",
        "vertical_rate_mps",
        "abs_vertical_rate_mps",
        "speed_acceleration_mps2",
        "acceleration_mps2",
        "vertical_acceleration_mps2",
        "specific_energy_height_m",
        "specific_energy_rate_mps",
        "abs_specific_energy_rate_mps",
        "velocity_mismatch_mps",
        "local_residual_proxy_m",
        "flight_path_angle_rad",
    ]
    z = _robust_standardize(raw)
    return {
        "t": t,
        "y": y,
        "v": v,
        "raw": raw,
        "z": z,
        "feature_names": feature_names,
        "feature_source": signal_source,
        "kalman_segmentation": kalman_diag,
        "pelt_available": False,
        "pelt_used": False,
    }

def propose_boundary_candidates(
    component: HardGapComponent,
    features: dict[str, Any],
    config: DynamicSegmentationConfig,
) -> list[SegmentationBoundaryCandidate]:
    """Generate boundaries from independently-detected flight primitives.

    The segmentation pipeline is now two-stage:

    1.  Build an *etalon* set of raw ADS-B motion primitives inside the hard-gap
        component.  Energy/vertical mode, horizontal turn mode, and optional
        rough/noisy-air mode are detected independently, with hysteresis and
        run-length cleanup.  This prevents a long level turn from being hidden
        inside an ``energy_constant`` interval and prevents a climb/descent lobe
        from being hidden by speed/height energy exchange.
    2.  Convert primitive transitions into candidate shared samples.  Cleanup is
        handled later by a dynamic feasibility selector, so all climb, descent,
        speed-change, and turn transitions get a chance to become real V-Spline
        segment joins.
    """
    z = np.asarray(features["z"], dtype=float)
    raw = np.asarray(features["raw"], dtype=float)
    names = list(features["feature_names"])
    name_to_idx = {name: i for i, name in enumerate(names)}
    n = component.n_observations

    def cols(*wanted: str) -> list[int]:
        return [name_to_idx[w] for w in wanted if w in name_to_idx]

    scored: dict[int, SegmentationBoundaryCandidate] = {}

    etalon_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    etalon_diag: dict[str, Any] = {"enabled": False}
    if config.enable_energy_state_segmentation:
        etalon_candidates, etalon_diag = _propose_etalon_regime_candidates(component, raw, z, names, config)
        if etalon_diag:
            component_labels = etalon_diag.pop("_regime_state_labels", None)
            component_label_parts = etalon_diag.pop("_regime_label_parts", None)
            component_rough_score = etalon_diag.pop("_rough_air_score", None)
            if component_labels is not None:
                features["regime_state_labels"] = component_labels
            if component_label_parts is not None:
                features["regime_label_parts"] = component_label_parts
            if component_rough_score is not None:
                features["rough_air_score"] = component_rough_score
        for local_idx, candidate_score, reasons in etalon_candidates:
            _add_candidate(scored, component, int(local_idx), float(candidate_score), reasons, raw, z, names)
    features["energy_state_diagnostics"] = etalon_diag or {"enabled": False}

    # Optional generic spike detector.  It is intentionally secondary: etalon
    # primitive transitions are the primary boundaries, while spikes only add a
    # candidate if the caller explicitly enables them for diagnostics.
    if config.enable_motion_spike_boundaries:
        spike_cols = cols(
            "speed_acceleration_mps2",
            "vertical_acceleration_mps2",
            "specific_energy_rate_mps",
            "velocity_mismatch_mps",
            "local_residual_proxy_m",
        )
        if config.segment_horizontal_turns:
            spike_cols.extend(cols("heading_rate_radps", "abs_lateral_acceleration_mps2"))
        score = np.linalg.norm(z[:, spike_cols], axis=1) if spike_cols else np.zeros(n, dtype=float)
        for local_idx in np.where(score >= float(config.candidate_score_z))[0]:
            _add_candidate(scored, component, int(local_idx), float(score[local_idx]), ("robust_feature_spike",), raw, z, names)

    # PELT remains available as an ablation source, but it does not drive the
    # default segmentation because PELT is weak at explaining *what* happened at
    # a boundary.  The etalon primitive labels above are interpretable and path
    # general.
    if config.use_pelt_if_available and config.enable_pelt_boundaries:
        score_cols = cols("specific_energy_rate_mps", "velocity_mismatch_mps", "local_residual_proxy_m")
        score = np.linalg.norm(z[:, score_cols], axis=1) if score_cols else np.zeros(n, dtype=float)
        try:
            import ruptures as rpt  # type: ignore

            min_size = max(2, int(config.min_segment_points) - 1)
            jump = max(1, int(config.pelt_jump))
            if config.pelt_energy_features_only:
                signal_cols = cols(
                    "vertical_rate_mps",
                    "speed_acceleration_mps2",
                    "specific_energy_rate_mps",
                    "abs_specific_energy_rate_mps",
                )
            else:
                signal_cols = cols(
                    "horizontal_speed_mps",
                    "vertical_rate_mps",
                    "speed_acceleration_mps2",
                    "specific_energy_rate_mps",
                    "abs_specific_energy_rate_mps",
                    "velocity_mismatch_mps",
                    "local_residual_proxy_m",
                )
                if config.segment_horizontal_turns:
                    signal_cols.extend(cols("heading_rate_radps", "abs_lateral_acceleration_mps2"))
            if not signal_cols:
                signal_cols = list(range(z.shape[1]))
            signal = z[:, signal_cols]
            penalty = float(config.pelt_penalty_scale) * math.log(max(n, 2)) * signal.shape[1]
            algo = rpt.Pelt(model=str(config.pelt_model), min_size=min_size, jump=jump).fit(signal)
            bkps = algo.predict(pen=penalty)
            features["pelt_available"] = True
            features["pelt_used"] = True
            for bkp in bkps:
                if bkp >= n:
                    continue
                local_idx = int(bkp)
                local_score = float(score[min(max(local_idx, 0), n - 1)])
                _add_candidate(scored, component, local_idx, max(local_score, 1.0), ("pelt",), raw, z, names)
        except Exception:
            features["pelt_available"] = False
            features["pelt_used"] = False

    return sorted(scored.values(), key=lambda c: (c.sample_index, -c.score))


def _propose_etalon_regime_candidates(
    component: HardGapComponent,
    raw: np.ndarray,
    z: np.ndarray,
    names: list[str],
    config: DynamicSegmentationConfig,
) -> tuple[list[tuple[int, float, tuple[str, ...]]], dict[str, Any]]:
    """Build raw-ADS-B etalon primitive segments and transition candidates.

    Energy/vertical, turn, and rough-air labels are produced independently first.
    Candidate boundaries are then the union of independent primitive transitions
    plus conservative altitude-lobe/vertical-reversal guards.  This makes the
    segmentation explain actual flight maneuvers instead of only large numeric
    feature spikes.
    """
    name_to_idx = {name: i for i, name in enumerate(names)}
    t = np.asarray([s.t for s in component.samples], dtype=float)
    n = len(t)
    if n < 3:
        return [], {"enabled": True, "reason": "too_few_samples", "candidate_count": 0}

    def col(name: str, default: float = 0.0) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.full(n, float(default), dtype=float)
        return np.asarray(raw[:, idx], dtype=float)

    def zcol(name: str) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.zeros(n, dtype=float)
        return np.asarray(z[:, idx], dtype=float)

    speed = col("horizontal_speed_mps")
    vertical_rate = col("vertical_rate_mps")
    speed_accel = col("speed_acceleration_mps2")
    energy_rate = col("specific_energy_rate_mps")
    signed_heading_rate = col("signed_heading_rate_radps")
    lateral_accel = col("lateral_acceleration_mps2")

    energy_deadband = max(
        float(config.energy_rate_deadband_mps),
        float(config.energy_rate_deadband_scale) * _robust_scale(energy_rate),
    )
    vertical_deadband = max(
        float(config.vertical_rate_deadband_mps),
        float(config.vertical_rate_deadband_scale) * _robust_scale(vertical_rate),
    )
    accel_deadband = max(
        float(config.speed_accel_deadband_mps2),
        float(config.speed_accel_deadband_scale) * _robust_scale(speed_accel),
    )
    turn_floor = math.radians(float(config.turn_rate_deadband_degps))
    turn_adaptive = float(config.turn_rate_deadband_scale) * _robust_scale(signed_heading_rate)
    # Do not let one busy hold pattern raise the adaptive turn threshold so high
    # that lower-rate but sustained turns disappear.
    turn_cap = max(turn_floor, math.radians(float(config.turn_rate_medium_threshold_degps)))
    turn_rate_deadband_radps = max(turn_floor, min(turn_cap, turn_adaptive if turn_adaptive > 0.0 else turn_floor))
    lateral_floor = float(config.lateral_accel_deadband_mps2)
    lateral_adaptive = float(config.lateral_accel_deadband_scale) * _robust_scale(lateral_accel)
    lateral_cap = max(lateral_floor, 2.5 * lateral_floor)
    lateral_deadband = max(lateral_floor, min(lateral_cap, lateral_adaptive if lateral_adaptive > 0.0 else lateral_floor))

    rough_score = _rough_air_score(
        z_velocity_mismatch=zcol("velocity_mismatch_mps"),
        z_residual=zcol("local_residual_proxy_m"),
        z_vertical_accel=zcol("vertical_acceleration_mps2"),
        z_speed_accel=zcol("speed_acceleration_mps2"),
        config=config,
    )

    regime_labels, label_parts = _regime_state_labels(
        speed=speed,
        vertical_rate=vertical_rate,
        speed_accel=speed_accel,
        energy_rate=energy_rate,
        signed_heading_rate=signed_heading_rate,
        lateral_accel=lateral_accel,
        rough_score=rough_score,
        t=t,
        config=config,
        energy_deadband=energy_deadband,
        vertical_deadband=vertical_deadband,
        accel_deadband=accel_deadband,
        turn_rate_deadband_radps=turn_rate_deadband_radps,
        lateral_deadband=lateral_deadband,
    )
    energy_labels = np.asarray(label_parts["energy"], dtype=object)
    turn_labels = np.asarray(label_parts["turn"], dtype=object)
    rough_labels = np.asarray(label_parts["rough_air"], dtype=object)

    energy_runs = _label_runs(energy_labels, t)
    turn_runs = _label_runs(turn_labels, t)
    rough_runs = _label_runs(rough_labels, t)
    composite_runs = _label_runs(regime_labels, t)

    candidates: list[tuple[int, float, tuple[str, ...]]] = []

    # Independent primitive transitions.  These are the etalon boundaries: every
    # sustained climb/descent/speed-change and every sustained horizontal turn is
    # represented even if another primitive stays constant.
    candidates.extend(
        _transition_candidates_from_runs(
            component=component,
            runs=energy_runs,
            raw=raw,
            z=z,
            names=names,
            config=config,
            base_score=80.0,
            detector_name="energy_etalon",
        )
    )
    if bool(config.segment_horizontal_turns):
        candidates.extend(
            _transition_candidates_from_runs(
                component=component,
                runs=turn_runs,
                raw=raw,
                z=z,
                names=names,
                config=config,
                base_score=76.0,
                detector_name="turn_etalon",
            )
        )
    if bool(config.enable_rough_air_segmentation):
        candidates.extend(
            _transition_candidates_from_runs(
                component=component,
                runs=rough_runs,
                raw=raw,
                z=z,
                names=names,
                config=config,
                base_score=66.0,
                detector_name="rough_air_etalon",
            )
        )

    altitude_m = col("specific_energy_height_m") - (speed**2) / (2.0 * STANDARD_GRAVITY_MPS2)

    go_around_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    go_around_diag: dict[str, Any] = {"enabled": False}
    if bool(config.enable_go_around_detection):
        go_around_candidates, go_around_diag = _propose_go_around_candidates(
            component=component,
            t=t,
            altitude_m=altitude_m,
            vertical_rate=vertical_rate,
            speed_accel=speed_accel,
            energy_rate=energy_rate,
            vertical_deadband=vertical_deadband,
            config=config,
        )
        candidates.extend(go_around_candidates)

    vertical_reversal_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    vertical_reversal_diag: dict[str, Any] = {"enabled": False}
    if bool(config.enable_vertical_reversal_segmentation):
        vertical_reversal_candidates, vertical_reversal_diag = _propose_vertical_reversal_candidates(
            component=component,
            t=t,
            altitude_m=altitude_m,
            vertical_rate=vertical_rate,
            energy_rate=energy_rate,
            vertical_deadband=vertical_deadband,
            config=config,
        )
        candidates.extend(vertical_reversal_candidates)

    altitude_lobe_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    altitude_lobe_diag: dict[str, Any] = {"enabled": False}
    if bool(config.enable_altitude_lobe_segmentation):
        altitude_lobe_candidates, altitude_lobe_diag = _propose_altitude_lobe_candidates(
            component=component,
            t=t,
            altitude_m=altitude_m,
            vertical_rate=vertical_rate,
            vertical_deadband=vertical_deadband,
            config=config,
        )
        candidates.extend(altitude_lobe_candidates)

    return candidates, {
        "enabled": True,
        "detector": "independent_energy_turn_etalon",
        "candidate_count": len(candidates),
        "energy_run_count": len(energy_runs),
        "turn_run_count": len(turn_runs),
        "rough_air_run_count": len(rough_runs),
        "composite_run_count": len(composite_runs),
        "energy_rate_deadband_mps": float(energy_deadband),
        "vertical_rate_deadband_mps": float(vertical_deadband),
        "speed_accel_deadband_mps2": float(accel_deadband),
        "turn_rate_deadband_degps": float(math.degrees(turn_rate_deadband_radps)),
        "lateral_accel_deadband_mps2": float(lateral_deadband),
        "rough_air_score_threshold": float(config.rough_air_score_threshold),
        "labels_preview": regime_labels[: min(n, 80)].tolist(),
        "label_parts_preview": {k: v[: min(n, 80)].tolist() for k, v in label_parts.items()},
        "energy_etalon_runs": energy_runs[:120],
        "turn_etalon_runs": turn_runs[:120],
        "rough_air_runs": rough_runs[:80],
        "composite_runs": composite_runs[:160],
        "go_around": go_around_diag,
        "vertical_reversal": vertical_reversal_diag,
        "altitude_lobe": altitude_lobe_diag,
        "_regime_state_labels": regime_labels,
        "_regime_label_parts": label_parts,
        "_rough_air_score": rough_score,
    }


def _transition_candidates_from_runs(
    *,
    component: HardGapComponent,
    runs: list[dict[str, Any]],
    raw: np.ndarray,
    z: np.ndarray,
    names: list[str],
    config: DynamicSegmentationConfig,
    base_score: float,
    detector_name: str,
) -> list[tuple[int, float, tuple[str, ...]]]:
    out: list[tuple[int, float, tuple[str, ...]]] = []
    if len(runs) < 2:
        return out
    n = component.n_observations
    name_to_idx = {name: i for i, name in enumerate(names)}

    def abs_z(name: str, idx: int) -> float:
        col = name_to_idx.get(name)
        if col is None or z.size == 0:
            return 0.0
        return abs(float(z[min(max(idx, 0), z.shape[0] - 1), col]))

    for left, right in zip(runs[:-1], runs[1:]):
        if str(left.get("label")) == str(right.get("label")):
            continue
        raw_local_idx = int(right.get("start_idx", 0))
        local_idx, placement = _choose_boundary_sample_for_transition(
            left=left,
            right=right,
            raw=raw,
            z=z,
            names=names,
            config=config,
        )
        local_idx = int(min(max(local_idx, 1), n - 2))
        if local_idx <= 0 or local_idx >= n - 1:
            continue

        label_from = str(left.get("label", "unknown"))
        label_to = str(right.get("label", "unknown"))
        score = float(base_score)
        score += 2.0 * _primitive_transition_weight(detector_name, label_from, label_to)
        score += 0.7 * abs_z("specific_energy_rate_mps", local_idx)
        score += 0.6 * abs_z("vertical_rate_mps", local_idx)
        score += 0.45 * abs_z("speed_acceleration_mps2", local_idx)
        score += 0.75 * abs_z("heading_rate_radps", local_idx)
        score += 0.45 * abs_z("abs_lateral_acceleration_mps2", local_idx)
        score += 0.35 * abs_z("local_residual_proxy_m", local_idx)

        reasons = [
            f"{detector_name}_transition",
            "regime_state_transition",
            f"regime_from_{label_from}",
            f"regime_to_{label_to}",
        ]
        if detector_name.startswith("energy"):
            reasons.append("energy_state_transition")
            if _contains_any(label_from, label_to, tokens=("climb",)):
                reasons.append("climb_state_transition")
            if _contains_any(label_from, label_to, tokens=("descent",)):
                reasons.append("descent_state_transition")
            if _contains_any(label_from, label_to, tokens=("accel", "decel")):
                reasons.append("speed_change_transition")
        elif detector_name.startswith("turn"):
            reasons.append("turn_state_transition")
        elif detector_name.startswith("rough"):
            reasons.append("rough_air_state_transition")

        if local_idx != raw_local_idx:
            reasons.append("optimized_boundary_sample")
            reasons.append(f"boundary_shift_{local_idx - raw_local_idx:+d}_samples")
        if placement:
            reason = placement.get("reason")
            if isinstance(reason, str) and reason:
                reasons.append(f"placement_{reason}")
        out.append((local_idx, score, tuple(reasons)))
    return out


def _primitive_transition_weight(detector_name: str, label_from: str, label_to: str) -> float:
    labels = (str(label_from), str(label_to))
    if detector_name.startswith("turn"):
        if any("turn_" in label for label in labels):
            return 12.0
        return 4.0
    if detector_name.startswith("rough"):
        if any("rough_air" in label for label in labels):
            return 10.0
        return 3.0
    weight = 4.0
    if _contains_any(*labels, tokens=("climb", "descent")):
        weight += 10.0
    if _contains_any(*labels, tokens=("energy_gain", "energy_loss")):
        weight += 6.0
    if _contains_any(*labels, tokens=("accel", "decel")):
        weight += 4.0
    if _contains_any(*labels, tokens=("rapid",)):
        weight += 3.0
    return weight


def _contains_any(*labels: str, tokens: tuple[str, ...]) -> bool:
    joined = " ".join(str(label) for label in labels)
    return any(token in joined for token in tokens)

def _propose_energy_state_candidates(
    component: HardGapComponent,
    raw: np.ndarray,
    z: np.ndarray,
    names: list[str],
    config: DynamicSegmentationConfig,
) -> tuple[list[tuple[int, float, tuple[str, ...]]], dict[str, Any]]:
    """Find sustained regime transitions.

    The detector is intentionally state-based rather than spike-based.  Every
    sample gets a robust composite label: vertical/energy mode, optional turn
    mode, and optional rough/noisy-air mode.  A boundary is proposed only when a
    stable run changes into another stable run with enough support on both sides.
    """
    name_to_idx = {name: i for i, name in enumerate(names)}
    t = np.asarray([s.t for s in component.samples], dtype=float)
    n = len(t)
    if n < 3:
        return [], {"enabled": True, "reason": "too_few_samples", "candidate_count": 0}

    def col(name: str, default: float = 0.0) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.full(n, float(default), dtype=float)
        return np.asarray(raw[:, idx], dtype=float)

    def zcol(name: str) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.zeros(n, dtype=float)
        return np.asarray(z[:, idx], dtype=float)

    speed = col("horizontal_speed_mps")
    vertical_rate = col("vertical_rate_mps")
    speed_accel = col("speed_acceleration_mps2")
    energy_rate = col("specific_energy_rate_mps")
    signed_heading_rate = col("signed_heading_rate_radps")
    lateral_accel = col("lateral_acceleration_mps2")

    # Component-local robust deadbands.  Absolute floors prevent tiny numerical
    # noise from becoming state changes; scale terms adapt to noisier tracks.
    energy_deadband = max(
        float(config.energy_rate_deadband_mps),
        float(config.energy_rate_deadband_scale) * _robust_scale(energy_rate),
    )
    vertical_deadband = max(
        float(config.vertical_rate_deadband_mps),
        float(config.vertical_rate_deadband_scale) * _robust_scale(vertical_rate),
    )
    accel_deadband = max(
        float(config.speed_accel_deadband_mps2),
        float(config.speed_accel_deadband_scale) * _robust_scale(speed_accel),
    )
    turn_rate_deadband_radps = max(
        math.radians(float(config.turn_rate_deadband_degps)),
        float(config.turn_rate_deadband_scale) * _robust_scale(signed_heading_rate),
    )
    lateral_deadband = max(
        float(config.lateral_accel_deadband_mps2),
        float(config.lateral_accel_deadband_scale) * _robust_scale(lateral_accel),
    )

    rough_score = _rough_air_score(
        z_velocity_mismatch=zcol("velocity_mismatch_mps"),
        z_residual=zcol("local_residual_proxy_m"),
        z_vertical_accel=zcol("vertical_acceleration_mps2"),
        z_speed_accel=zcol("speed_acceleration_mps2"),
        config=config,
    )

    labels, label_parts = _regime_state_labels(
        speed=speed,
        vertical_rate=vertical_rate,
        speed_accel=speed_accel,
        energy_rate=energy_rate,
        signed_heading_rate=signed_heading_rate,
        lateral_accel=lateral_accel,
        rough_score=rough_score,
        t=t,
        config=config,
        energy_deadband=energy_deadband,
        vertical_deadband=vertical_deadband,
        accel_deadband=accel_deadband,
        turn_rate_deadband_radps=turn_rate_deadband_radps,
        lateral_deadband=lateral_deadband,
    )
    features_diag_target = None  # keep a visible marker for static readers; values are stored below

    runs = _label_runs(labels, t)
    min_points = max(2, int(config.energy_state_min_points))
    min_duration = max(0.0, float(config.energy_state_min_duration_s))
    stable_runs = [
        run
        for run in runs
        if run["n_points"] >= min_points and run["duration_s"] >= min_duration
    ]

    candidates: list[tuple[int, float, tuple[str, ...]]] = []
    for left, right in zip(stable_runs[:-1], stable_runs[1:]):
        if left["label"] == right["label"]:
            continue
        raw_local_idx = int(right["start_idx"])
        local_idx, placement = _choose_boundary_sample_for_transition(
            left=left,
            right=right,
            raw=raw,
            z=z,
            names=names,
            config=config,
        )
        if local_idx <= 0 or local_idx >= n - 1:
            continue

        local_score = float(config.energy_boundary_score)
        local_score += float(abs(zcol("specific_energy_rate_mps")[local_idx]))
        local_score += 0.5 * float(abs(zcol("speed_acceleration_mps2")[local_idx]))
        local_score += 0.5 * float(abs(zcol("vertical_rate_mps")[local_idx]))
        local_score += 0.5 * float(abs(zcol("heading_rate_radps")[local_idx]))
        local_score += 0.5 * float(abs(zcol("local_residual_proxy_m")[local_idx]))
        local_score += 0.5 * float(rough_score[local_idx])

        reasons = [
            "regime_state_transition",
            f"regime_from_{left['label']}",
            f"regime_to_{right['label']}",
        ]
        if local_idx != raw_local_idx:
            reasons.append("optimized_boundary_sample")
            reasons.append(f"boundary_shift_{local_idx - raw_local_idx:+d}_samples")
        if "turn_" in str(left["label"]) or "turn_" in str(right["label"]):
            reasons.append("turn_state_transition")
        if "rough_air" in str(left["label"]) or "rough_air" in str(right["label"]):
            reasons.append("rough_air_state_transition")
        if any(token in str(left["label"]) or token in str(right["label"]) for token in ("climb", "descent", "energy_gain", "energy_loss", "accel", "decel")):
            reasons.append("energy_state_transition")
        candidates.append((local_idx, local_score, tuple(reasons)))

    altitude_m = raw[:, name_to_idx["specific_energy_height_m"]] - (speed**2) / (2.0 * STANDARD_GRAVITY_MPS2)

    go_around_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    go_around_diag: dict[str, Any] = {"enabled": False}
    if bool(config.enable_go_around_detection):
        go_around_candidates, go_around_diag = _propose_go_around_candidates(
            component=component,
            t=t,
            altitude_m=altitude_m,
            vertical_rate=vertical_rate,
            speed_accel=speed_accel,
            energy_rate=energy_rate,
            vertical_deadband=vertical_deadband,
            config=config,
        )
        candidates.extend(go_around_candidates)

    vertical_reversal_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    vertical_reversal_diag: dict[str, Any] = {"enabled": False}
    if bool(config.enable_vertical_reversal_segmentation):
        vertical_reversal_candidates, vertical_reversal_diag = _propose_vertical_reversal_candidates(
            component=component,
            t=t,
            altitude_m=altitude_m,
            vertical_rate=vertical_rate,
            energy_rate=energy_rate,
            vertical_deadband=vertical_deadband,
            config=config,
        )
        candidates.extend(vertical_reversal_candidates)

    altitude_lobe_candidates: list[tuple[int, float, tuple[str, ...]]] = []
    altitude_lobe_diag: dict[str, Any] = {"enabled": False}
    if bool(config.enable_altitude_lobe_segmentation):
        altitude_lobe_candidates, altitude_lobe_diag = _propose_altitude_lobe_candidates(
            component=component,
            t=t,
            altitude_m=altitude_m,
            vertical_rate=vertical_rate,
            vertical_deadband=vertical_deadband,
            config=config,
        )
        candidates.extend(altitude_lobe_candidates)

    return candidates, {
        "enabled": True,
        "candidate_count": len(candidates),
        "raw_run_count": len(runs),
        "stable_run_count": len(stable_runs),
        "energy_rate_deadband_mps": float(energy_deadband),
        "vertical_rate_deadband_mps": float(vertical_deadband),
        "speed_accel_deadband_mps2": float(accel_deadband),
        "turn_rate_deadband_degps": float(math.degrees(turn_rate_deadband_radps)),
        "lateral_accel_deadband_mps2": float(lateral_deadband),
        "rough_air_score_threshold": float(config.rough_air_score_threshold),
        "labels_preview": labels[: min(n, 80)].tolist(),
        "label_parts_preview": {k: v[: min(n, 80)].tolist() for k, v in label_parts.items()},
        "stable_runs": stable_runs[:60],
        "go_around": go_around_diag,
        "vertical_reversal": vertical_reversal_diag,
        "altitude_lobe": altitude_lobe_diag,
        # Private/transient arrays used by ``_make_dynamic_segment``.  The caller
        # removes these keys before storing diagnostics so large numpy arrays do
        # not leak into JSON reports.
        "_regime_state_labels": labels,
        "_regime_label_parts": label_parts,
        "_rough_air_score": rough_score,
    }


def _choose_boundary_sample_for_transition(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    raw: np.ndarray,
    z: np.ndarray,
    names: list[str],
    config: DynamicSegmentationConfig,
) -> tuple[int, dict[str, Any]]:
    """Choose the least-damaging real sample for a segment join.

    Run-length state detection identifies that a boundary is needed, but using
    the first sample of the new run can pin the curve to a local noise spike.
    This helper searches the transition band between stable runs and minimizes a
    small local quality objective.  It never invents an off-sample boundary, so
    hard endpoint constraints remain tied to real ADS-B paired samples.
    """
    n = int(raw.shape[0])
    raw_idx = int(right.get("start_idx", 0))
    raw_idx = int(min(max(raw_idx, 1), max(n - 2, 1)))
    if not bool(config.optimize_boundary_sample) or n < 3:
        return raw_idx, {"enabled": False, "selected_idx": raw_idx, "reason": "disabled_or_too_few_samples"}

    name_to_idx = {name: i for i, name in enumerate(names)}

    def abs_z(name: str, idx: int) -> float:
        col = name_to_idx.get(name)
        if col is None:
            return 0.0
        return abs(float(z[idx, col]))

    pad = max(0, int(config.boundary_search_padding_points))
    left_end = int(left.get("end_idx", raw_idx - 1))
    right_start = int(right.get("start_idx", raw_idx))
    raw_window_lo = max(1, min(left_end, right_start) - pad)
    raw_window_hi = min(n - 2, max(left_end, right_start) + pad)

    shift_cfg = getattr(config, "max_boundary_shift_points", None)
    max_shift = None if shift_cfg is None or int(shift_cfg) < 0 else int(shift_cfg)
    lo = raw_window_lo
    hi = raw_window_hi
    search_was_clipped = False
    if max_shift is not None:
        clipped_lo = max(lo, raw_idx - max_shift)
        clipped_hi = min(hi, raw_idx + max_shift)
        search_was_clipped = clipped_lo != lo or clipped_hi != hi
        lo, hi = clipped_lo, clipped_hi

    if hi < lo:
        return raw_idx, {
            "enabled": True,
            "selected_idx": raw_idx,
            "reason": "empty_search_window_after_shift_cap",
            "raw_idx": int(raw_idx),
            "unclipped_search_window": [int(raw_window_lo), int(raw_window_hi)],
            "max_shift_points": None if max_shift is None else int(max_shift),
        }

    scores: list[tuple[float, int]] = []
    for idx in range(lo, hi + 1):
        score = 0.0
        score += float(config.boundary_quality_velocity_mismatch_weight) * abs_z("velocity_mismatch_mps", idx)
        score += float(config.boundary_quality_residual_weight) * abs_z("local_residual_proxy_m", idx)
        score += float(config.boundary_quality_acceleration_weight) * (
            abs_z("acceleration_mps2", idx)
            + 0.5 * abs_z("vertical_acceleration_mps2", idx)
            + 0.5 * abs_z("speed_acceleration_mps2", idx)
        )
        # Tiny tie-breaker: do not move far from the state transition unless the
        # local quality difference is real.
        score += 0.05 * abs(idx - raw_idx)
        scores.append((float(score), int(idx)))
    best_score, best_idx = min(scores, key=lambda item: (item[0], abs(item[1] - raw_idx), item[1]))
    return int(best_idx), {
        "enabled": True,
        "raw_idx": int(raw_idx),
        "selected_idx": int(best_idx),
        "shift_samples": int(best_idx - raw_idx),
        "search_window": [int(lo), int(hi)],
        "unclipped_search_window": [int(raw_window_lo), int(raw_window_hi)],
        "transition_span_points": int(max(0, raw_window_hi - raw_window_lo)),
        "max_shift_points": None if max_shift is None else int(max_shift),
        "search_was_clipped": bool(search_was_clipped),
        "selected_quality_score": float(best_score),
    }




def _propose_altitude_lobe_candidates(
    *,
    component: HardGapComponent,
    t: np.ndarray,
    altitude_m: np.ndarray,
    vertical_rate: np.ndarray,
    vertical_deadband: float,
    config: DynamicSegmentationConfig,
) -> tuple[list[tuple[int, float, tuple[str, ...]]], dict[str, Any]]:
    """Detect large altitude lobes hidden inside energy-constant states.

    Energy-state segmentation is useful for aviation, but it can miss a
    speed/height exchange: the aircraft climbs while decelerating, then descends
    while accelerating, so total specific energy remains roughly constant.  For
    segmented splines that is still two different local trajectory models.  This
    detector therefore looks directly at altitude prominence and finite-difference
    vertical motion.  It proposes boundaries at prominent local maxima/minima or,
    when there is a long level shelf between opposite vertical modes, at the start
    of the later mode so climb/plateau/descent are not collapsed into one fit.
    """
    n = len(t)
    if n < 5:
        return [], {"enabled": True, "reason": "too_few_samples", "candidate_count": 0}

    t = np.asarray(t, dtype=float)
    altitude = _rolling_median(np.asarray(altitude_m, dtype=float), max(3, int(config.energy_smoothing_window_points)))
    reported_vz = np.asarray(vertical_rate, dtype=float)
    altitude_vz = _sample_gradient(altitude, t)
    # Prefer the geometric altitude derivative for lobe detection, but retain a
    # small reported-vz contribution when it is finite.  This keeps stale ADS-B
    # vertical-rate fields from suppressing an obvious altitude lobe.
    blended_vz = np.where(np.isfinite(reported_vz), 0.75 * altitude_vz + 0.25 * reported_vz, altitude_vz)
    blended_vz = _rolling_median(blended_vz, max(3, int(config.energy_smoothing_window_points)))

    gate = max(float(config.altitude_lobe_gradient_gate_mps), 0.5 * float(vertical_deadband), 0.05)
    modes: list[str] = []
    for vz in blended_vz:
        if vz >= gate:
            modes.append("climb")
        elif vz <= -gate:
            modes.append("descent")
        else:
            modes.append("level")

    runs = _label_runs(np.asarray(modes, dtype=object), t)
    min_points = max(2, int(config.altitude_lobe_min_points))
    min_duration = max(0.0, float(config.altitude_lobe_min_duration_s))
    min_prom = max(0.0, float(config.altitude_lobe_min_prominence_m))
    min_side_prom = max(0.0, float(config.altitude_lobe_min_side_prominence_m))

    stable = [
        r
        for r in runs
        if r["label"] in {"climb", "descent", "level"}
        and r["n_points"] >= min_points
        and r["duration_s"] >= min_duration
    ]

    def _run_slice(run: dict[str, Any]) -> slice:
        return slice(int(run["start_idx"]), int(run["end_idx"]) + 1)

    def _median_alt(run: dict[str, Any]) -> float:
        return float(np.nanmedian(altitude[_run_slice(run)]))

    out: list[tuple[int, float, tuple[str, ...]]] = []
    inspected_pairs = 0
    for i, left in enumerate(stable[:-1]):
        left_label = str(left["label"])
        if left_label not in {"climb", "descent"}:
            continue
        for j in range(i + 1, len(stable)):
            right = stable[j]
            right_label = str(right["label"])
            if right_label == left_label:
                break
            if right_label == "level":
                # Look through a sustained shelf.  The later climb/descent is the
                # true proof that this is a lobe instead of a simple level-off.
                continue
            if {left_label, right_label} != {"climb", "descent"}:
                continue
            inspected_pairs += 1

            left_med = _median_alt(left)
            right_med = _median_alt(right)
            start = int(left["start_idx"])
            end = int(right["end_idx"])
            if end <= start:
                continue
            between_start = int(left["end_idx"])
            between_end = int(right["start_idx"])
            if between_end < between_start:
                between_start = start
                between_end = end
            search_slice = altitude[between_start : between_end + 1]
            if search_slice.size == 0:
                continue

            intervening = stable[i + 1 : j]
            level_duration = float(sum(float(r["duration_s"]) for r in intervening if str(r["label"]) == "level"))

            if left_label == "climb" and right_label == "descent":
                extrema_offset = int(np.nanargmax(search_slice))
                extrema_idx = int(between_start + extrema_offset)
                extrema_alt = float(altitude[extrema_idx])
                left_prom = extrema_alt - left_med
                right_prom = extrema_alt - right_med
                event_shape = "local_altitude_maximum"
                direction = "altitude_lobe_climb_to_descent"
                if level_duration >= min_duration and int(right["start_idx"]) > int(left["end_idx"]):
                    # A long shelf after the climb should not be glued to the
                    # following descent.  Put the strongest boundary at descent
                    # onset while the extrema still supplies the evidence.
                    extrema_idx = int(right["start_idx"])
            elif left_label == "descent" and right_label == "climb":
                extrema_offset = int(np.nanargmin(search_slice))
                extrema_idx = int(between_start + extrema_offset)
                extrema_alt = float(altitude[extrema_idx])
                left_prom = left_med - extrema_alt
                right_prom = right_med - extrema_alt
                event_shape = "local_altitude_minimum"
                direction = "altitude_lobe_descent_to_climb"
                if level_duration >= min_duration and int(right["start_idx"]) > int(left["end_idx"]):
                    extrema_idx = int(right["start_idx"])
            else:
                continue

            extrema_idx = int(min(max(extrema_idx, 1), n - 2))
            total_prom = max(0.0, float(left_prom)) + max(0.0, float(right_prom))
            if total_prom < min_prom or min(float(left_prom), float(right_prom)) < min_side_prom:
                continue

            # Require that the proposed boundary leaves usable support on both
            # sides.  This mirrors cleanup but avoids emitting obvious edge lobes.
            if extrema_idx < min_points - 1 or (n - extrema_idx) < min_points:
                continue
            if (t[extrema_idx] - t[0]) < min_duration or (t[-1] - t[extrema_idx]) < min_duration:
                continue

            score = float(config.altitude_lobe_boundary_score) + total_prom
            score += 2.0 * abs(float(np.nanmedian(blended_vz[_run_slice(left)])))
            score += 2.0 * abs(float(np.nanmedian(blended_vz[_run_slice(right)])))
            out.append(
                (
                    extrema_idx,
                    float(score),
                    (
                        "altitude_lobe_transition",
                        direction,
                        event_shape,
                        "large_vertical_excursion",
                    ),
                )
            )
            break

    return out, {
        "enabled": True,
        "candidate_count": len(out),
        "vertical_rate_gate_mps": float(gate),
        "altitude_lobe_runs": runs[:40],
        "stable_lobe_run_count": len(stable),
        "inspected_lobe_pairs": inspected_pairs,
        "min_prominence_m": float(min_prom),
        "min_side_prominence_m": float(min_side_prom),
    }


def _propose_vertical_reversal_candidates(
    *,
    component: HardGapComponent,
    t: np.ndarray,
    altitude_m: np.ndarray,
    vertical_rate: np.ndarray,
    energy_rate: np.ndarray,
    vertical_deadband: float,
    config: DynamicSegmentationConfig,
) -> tuple[list[tuple[int, float, tuple[str, ...]]], dict[str, Any]]:
    """Detect large sustained vertical-mode changes beyond go-arounds.

    The dedicated go-around detector handles descent -> climb around a local
    minimum.  This detector catches the equally important spline-fitting cases
    visible in altitude plots: a sustained climb followed by level/descent
    around a local maximum, or a sustained descent followed by level/climb
    around a local minimum.  It uses a blended reported/finite-difference
    vertical-rate signal so stale ADS-B vertical rates or Kalman-smoothed
    segmentation features do not hide a clear altitude lobe.
    """
    n = len(t)
    if n < 5:
        return [], {"enabled": True, "reason": "too_few_samples", "candidate_count": 0}

    t = np.asarray(t, dtype=float)
    altitude_m = _rolling_median(np.asarray(altitude_m, dtype=float), max(3, int(config.energy_smoothing_window_points)))
    reported_vz = np.asarray(vertical_rate, dtype=float)
    altitude_vz = _sample_gradient(altitude_m, t)
    # Use both sources when possible.  A real altitude lobe in the finite
    # difference signal should still be segmentable even when reported vertical
    # rate is asynchronous/stale or the feature source has been Kalman-smoothed.
    blended_vz = np.where(
        np.isfinite(reported_vz),
        0.5 * reported_vz + 0.5 * altitude_vz,
        altitude_vz,
    )
    blended_vz = _rolling_median(blended_vz, max(3, int(config.energy_smoothing_window_points)))
    energy_rate = _rolling_median(np.asarray(energy_rate, dtype=float), max(3, int(config.energy_smoothing_window_points)))

    vz_gate = max(float(vertical_deadband), 0.75)
    mode: list[str] = []
    for vz in blended_vz:
        if vz <= -vz_gate:
            mode.append("descent")
        elif vz >= vz_gate:
            mode.append("climb")
        else:
            mode.append("level")

    runs = _label_runs(np.asarray(mode, dtype=object), t)
    min_points = max(2, int(config.vertical_reversal_min_points))
    min_duration = max(0.0, float(config.vertical_reversal_min_duration_s))
    max_gap_s = max(0.0, float(config.vertical_reversal_max_transition_gap_s))
    max_gap_points = max(0, int(config.vertical_reversal_max_transition_gap_points))
    min_excursion_m = max(0.0, float(config.vertical_reversal_min_altitude_excursion_m))

    stable = [
        run
        for run in runs
        if run["label"] in {"climb", "descent", "level"}
        and run["n_points"] >= min_points
        and run["duration_s"] >= min_duration
    ]

    out: list[tuple[int, float, tuple[str, ...]]] = []
    checked_pairs = 0
    for i, left in enumerate(stable[:-1]):
        if left["label"] not in {"climb", "descent"}:
            continue
        for right in stable[i + 1 : min(len(stable), i + 4)]:
            if right["label"] == left["label"]:
                break
            if right["label"] not in {"level", "climb", "descent"}:
                continue
            transition_points = max(0, int(right["start_idx"] - left["end_idx"] - 1))
            transition_s = float(t[int(right["start_idx"])] - t[int(left["end_idx"])])
            if transition_points > max_gap_points or transition_s > max_gap_s:
                continue

            a = int(left["end_idx"])
            b = int(right["start_idx"])
            if b <= a:
                continue
            checked_pairs += 1
            transition_slice = altitude_m[a : b + 1]
            if left["label"] == "climb":
                local_idx = int(a + int(np.argmax(transition_slice)))
                before_excursion = float(altitude_m[local_idx] - np.nanmedian(altitude_m[left["start_idx"] : left["end_idx"] + 1]))
                after_excursion = float(altitude_m[local_idx] - np.nanmedian(altitude_m[right["start_idx"] : right["end_idx"] + 1]))
                event_shape = "local_altitude_maximum"
                direction_reason = f"vertical_mode_reversal_climb_to_{right['label']}"
            else:
                local_idx = int(a + int(np.argmin(transition_slice)))
                before_excursion = float(np.nanmedian(altitude_m[left["start_idx"] : left["end_idx"] + 1]) - altitude_m[local_idx])
                after_excursion = float(np.nanmedian(altitude_m[right["start_idx"] : right["end_idx"] + 1]) - altitude_m[local_idx])
                event_shape = "local_altitude_minimum"
                direction_reason = f"vertical_mode_reversal_descent_to_{right['label']}"

            local_idx = int(min(max(local_idx, 1), n - 2))
            # A genuine reversal has altitude evidence on at least one side and
            # a non-trivial difference to the other side.  Requiring both sides
            # strongly would miss climb -> level-off cases near the end of an
            # available component, so use a total excursion plus a minimum peak
            # side criterion.
            total_excursion = max(0.0, before_excursion) + max(0.0, after_excursion)
            if total_excursion < min_excursion_m or max(before_excursion, after_excursion) < 0.5 * min_excursion_m:
                continue

            strength = total_excursion
            strength += 2.0 * abs(float(np.nanmedian(blended_vz[left["start_idx"] : left["end_idx"] + 1])))
            strength += 2.0 * abs(float(np.nanmedian(blended_vz[right["start_idx"] : right["end_idx"] + 1])))
            strength += abs(float(np.nanmedian(energy_rate[left["start_idx"] : right["end_idx"] + 1])))
            score = float(config.vertical_reversal_boundary_score) + float(strength)
            out.append(
                (
                    local_idx,
                    score,
                    (
                        "vertical_reversal_transition",
                        direction_reason,
                        event_shape,
                    ),
                )
            )
            break

    return out, {
        "enabled": True,
        "candidate_count": len(out),
        "vertical_rate_gate_mps": float(vz_gate),
        "vertical_mode_runs": runs[:40],
        "checked_vertical_pairs": checked_pairs,
        "min_altitude_excursion_m": float(min_excursion_m),
    }


def _propose_go_around_candidates(
    *,
    component: HardGapComponent,
    t: np.ndarray,
    altitude_m: np.ndarray,
    vertical_rate: np.ndarray,
    speed_accel: np.ndarray,
    energy_rate: np.ndarray,
    vertical_deadband: float,
    config: DynamicSegmentationConfig,
) -> tuple[list[tuple[int, float, tuple[str, ...]]], dict[str, Any]]:
    """Detect sustained approach/descent -> go-around climb reversals.

    A go-around is not always obvious from total specific-energy-rate alone at
    the exact initiation point.  The first visible sign in ADS-B is often a
    vertical-mode reversal: a sustained descent/approach segment, a short
    flattening or local altitude minimum, then a sustained climb.  This detector
    uses smoothed vertical rate and altitude only to propose a boundary; the
    final reconstruction still uses V-Spline and shared C1 boundary state.
    """
    n = len(t)
    if n < 5:
        return [], {"enabled": True, "reason": "too_few_samples", "candidate_count": 0}

    t = np.asarray(t, dtype=float)
    altitude_m = _rolling_median(np.asarray(altitude_m, dtype=float), max(3, int(config.energy_smoothing_window_points)))
    vertical_rate = _rolling_median(np.asarray(vertical_rate, dtype=float), max(3, int(config.energy_smoothing_window_points)))
    speed_accel = _rolling_median(np.asarray(speed_accel, dtype=float), max(3, int(config.energy_smoothing_window_points)))
    energy_rate = _rolling_median(np.asarray(energy_rate, dtype=float), max(3, int(config.energy_smoothing_window_points)))

    vz_gate = max(float(vertical_deadband), 0.75)
    mode: list[str] = []
    for vz in vertical_rate:
        if vz <= -vz_gate:
            mode.append("descent")
        elif vz >= vz_gate:
            mode.append("climb")
        else:
            mode.append("level")

    runs = _label_runs(np.asarray(mode, dtype=object), t)
    min_points = max(2, int(config.go_around_min_points))
    min_duration = max(0.0, float(config.go_around_min_duration_s))
    max_gap_s = max(0.0, float(config.go_around_max_transition_gap_s))
    max_gap_points = max(0, int(config.go_around_max_transition_gap_points))
    min_reversal_m = max(0.0, float(config.go_around_min_altitude_reversal_m))

    out: list[tuple[int, float, tuple[str, ...]]] = []
    checked_pairs = 0
    for i, left in enumerate(runs):
        if left["label"] != "descent":
            continue
        if left["n_points"] < min_points or left["duration_s"] < min_duration:
            continue

        # Find the next sustained climb run, allowing a short level/transition
        # run between descent and climb.  This catches flare/level-off before
        # the missed-approach climb becomes established.
        for j in range(i + 1, min(len(runs), i + 4)):
            right = runs[j]
            if right["label"] == "descent":
                break
            if right["label"] != "climb":
                continue
            if right["n_points"] < min_points or right["duration_s"] < min_duration:
                continue
            transition_points = max(0, int(right["start_idx"] - left["end_idx"] - 1))
            transition_s = float(t[int(right["start_idx"])] - t[int(left["end_idx"])])
            if transition_points > max_gap_points or transition_s > max_gap_s:
                continue

            a = int(left["end_idx"])
            b = int(right["start_idx"])
            if b <= a:
                continue
            checked_pairs += 1
            local_min_rel = int(np.argmin(altitude_m[a : b + 1]))
            local_idx = int(a + local_min_rel)
            local_idx = int(min(max(local_idx, 1), n - 2))

            descent_drop = float(np.nanmedian(altitude_m[max(0, left["start_idx"]): left["end_idx"] + 1]) - altitude_m[local_idx])
            climb_gain = float(np.nanmedian(altitude_m[right["start_idx"]: min(n, right["end_idx"] + 1)]) - altitude_m[local_idx])
            if max(descent_drop, climb_gain) < min_reversal_m:
                continue

            strength = max(0.0, descent_drop) + max(0.0, climb_gain)
            strength += 2.0 * abs(float(np.nanmedian(vertical_rate[right["start_idx"]: right["end_idx"] + 1])))
            strength += max(0.0, float(np.nanmedian(energy_rate[right["start_idx"]: right["end_idx"] + 1])))
            strength += max(0.0, float(np.nanmedian(speed_accel[right["start_idx"]: right["end_idx"] + 1])))
            score = float(config.go_around_boundary_score) + float(strength)
            out.append(
                (
                    local_idx,
                    score,
                    (
                        "go_around_transition",
                        "vertical_mode_reversal_descent_to_climb",
                        "local_altitude_minimum",
                    ),
                )
            )
            break

    return out, {
        "enabled": True,
        "candidate_count": len(out),
        "vertical_rate_gate_mps": float(vz_gate),
        "vertical_mode_runs": runs[:40],
        "checked_descent_climb_pairs": checked_pairs,
    }

def _rough_air_score(
    *,
    z_velocity_mismatch: np.ndarray,
    z_residual: np.ndarray,
    z_vertical_accel: np.ndarray,
    z_speed_accel: np.ndarray,
    config: DynamicSegmentationConfig,
) -> np.ndarray:
    """Dimensionless rough/noisy-air score used only for segmentation labels.

    The score intentionally excludes heading-rate/lateral acceleration so a clean
    coordinated turn is not mislabeled as turbulence.  High local residuals,
    reported-vs-position velocity mismatch, and vertical/speed acceleration
    irregularity are better proxies for rough air or bad surveillance data.
    """
    parts = [
        float(config.rough_air_velocity_mismatch_weight) * np.asarray(z_velocity_mismatch, dtype=float),
        float(config.rough_air_residual_weight) * np.asarray(z_residual, dtype=float),
        float(config.rough_air_vertical_accel_weight) * np.asarray(z_vertical_accel, dtype=float),
        float(config.rough_air_speed_accel_weight) * np.asarray(z_speed_accel, dtype=float),
    ]
    score = np.sqrt(np.sum([np.square(x) for x in parts], axis=0))
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _regime_state_labels(
    *,
    speed: np.ndarray,
    vertical_rate: np.ndarray,
    speed_accel: np.ndarray,
    energy_rate: np.ndarray,
    signed_heading_rate: np.ndarray,
    lateral_accel: np.ndarray,
    rough_score: np.ndarray,
    t: np.ndarray,
    config: DynamicSegmentationConfig,
    energy_deadband: float,
    vertical_deadband: float,
    accel_deadband: float,
    turn_rate_deadband_radps: float,
    lateral_deadband: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return composite etalon regime labels and independent label parts.

    The labels are intentionally built from independent primitives:
    vertical/energy mode, turn mode, and optional rough-air mode.  This is the
    key difference from the old energy-only state machine: a horizontal turn no
    longer disappears into ``energy_constant``, and a climb/descent is labeled
    as a climb/descent even when speed is being traded against altitude.
    """
    smooth_window = max(1, int(config.energy_smoothing_window_points))
    t = np.asarray(t, dtype=float)

    speed_s = _rolling_median(np.asarray(speed, dtype=float), smooth_window)
    vz_s = _rolling_median(np.asarray(vertical_rate, dtype=float), smooth_window)
    ax_s = _rolling_median(np.asarray(speed_accel, dtype=float), smooth_window)
    erate_s = _rolling_median(np.asarray(energy_rate, dtype=float), smooth_window)
    hr_s = _rolling_median(np.asarray(signed_heading_rate, dtype=float), smooth_window)
    lat_s = _rolling_median(np.asarray(lateral_accel, dtype=float), smooth_window)
    rough_s = _rolling_median(np.asarray(rough_score, dtype=float), smooth_window)

    energy = _independent_energy_labels(
        speed=speed_s,
        vertical_rate=vz_s,
        speed_accel=ax_s,
        energy_rate=erate_s,
        ground_speed_threshold=float(config.energy_ground_speed_threshold_mps),
        energy_deadband=float(energy_deadband),
        vertical_deadband=float(vertical_deadband),
        accel_deadband=float(accel_deadband),
        energy_rate_medium_threshold=float(config.energy_rate_medium_threshold_mps),
        energy_rate_rapid_threshold=float(config.energy_rate_rapid_threshold_mps),
        vertical_rate_medium_threshold=float(config.vertical_rate_medium_threshold_mps),
        vertical_rate_rapid_threshold=float(config.vertical_rate_rapid_threshold_mps),
        speed_accel_medium_threshold=float(config.speed_accel_medium_threshold_mps2),
        speed_accel_rapid_threshold=float(config.speed_accel_rapid_threshold_mps2),
    )
    energy = _stabilize_labels(
        energy,
        t,
        min_points=max(2, int(config.energy_state_min_points)),
        min_duration=max(0.0, float(config.energy_state_min_duration_s)),
        protected_tokens=("climb", "descent", "accel", "decel", "energy_gain", "energy_loss"),
        protected_min_points=max(2, min(int(config.energy_state_min_points), int(config.min_segment_points))),
        protected_min_duration=max(0.0, min(float(config.energy_state_min_duration_s), float(config.min_segment_duration_s))),
    )

    turn = _independent_turn_labels(
        signed_heading_rate=hr_s,
        lateral_accel=lat_s,
        speed=speed_s,
        t=t,
        config=config,
        turn_rate_deadband_radps=float(turn_rate_deadband_radps),
        lateral_deadband=float(lateral_deadband),
    )

    rough = np.asarray(
        ["rough_air" if bool(config.enable_rough_air_segmentation) and float(r) >= float(config.rough_air_score_threshold) else "smooth_air" for r in rough_s],
        dtype=object,
    )
    rough = _stabilize_labels(
        rough,
        t,
        min_points=max(2, min(int(config.energy_state_min_points), int(config.min_segment_points))),
        min_duration=max(0.0, min(float(config.energy_state_min_duration_s), float(config.min_segment_duration_s))),
        protected_tokens=("rough_air",),
        protected_min_points=max(2, min(4, int(config.min_segment_points))),
        protected_min_duration=max(0.0, min(6.0, float(config.min_segment_duration_s))),
    )

    labels: list[str] = []
    for energy_label, turn_label, rough_label in zip(energy.astype(str), turn.astype(str), rough.astype(str)):
        if rough_label == "rough_air":
            parts = ["rough_air", energy_label]
        else:
            parts = [energy_label]
        if turn_label != "straight":
            parts.append(turn_label)
        labels.append("__".join(parts))

    labels_arr = _stabilize_labels(
        np.asarray(labels, dtype=object),
        t,
        min_points=max(2, min(int(config.energy_state_min_points), int(config.min_segment_points))),
        min_duration=max(0.0, min(float(config.energy_state_min_duration_s), float(config.min_segment_duration_s))),
        protected_tokens=("climb", "descent", "turn_", "rough_air", "accel", "decel"),
        protected_min_points=max(2, min(4, int(config.min_segment_points))),
        protected_min_duration=max(0.0, min(6.0, float(config.min_segment_duration_s))),
    )

    return labels_arr.astype(object), {
        "energy": energy.astype(object),
        "turn": turn.astype(object),
        "rough_air": rough.astype(object),
    }


def _independent_energy_labels(
    *,
    speed: np.ndarray,
    vertical_rate: np.ndarray,
    speed_accel: np.ndarray,
    energy_rate: np.ndarray,
    ground_speed_threshold: float,
    energy_deadband: float,
    vertical_deadband: float,
    accel_deadband: float,
    energy_rate_medium_threshold: float,
    energy_rate_rapid_threshold: float,
    vertical_rate_medium_threshold: float,
    vertical_rate_rapid_threshold: float,
    speed_accel_medium_threshold: float,
    speed_accel_rapid_threshold: float,
) -> np.ndarray:
    labels: list[str] = []
    for spd, vz, ax, edot in zip(speed, vertical_rate, speed_accel, energy_rate):
        if float(spd) < float(ground_speed_threshold):
            labels.append("ground_slow")
            continue

        # Segment identity is the maneuver kind.  Numeric strength and
        # total-energy rate remain in segment features, but they no longer split
        # a long climb/descent into noisy gain/loss/constant shards.
        if vz > vertical_deadband:
            labels.append("climb")
        elif vz < -vertical_deadband:
            labels.append("descent")
        elif ax > accel_deadband:
            labels.append("level_accel")
        elif ax < -accel_deadband:
            labels.append("level_decel")
        elif edot > energy_deadband:
            labels.append("energy_gain")
        elif edot < -energy_deadband:
            labels.append("energy_loss")
        else:
            labels.append("energy_constant")
    return np.asarray(labels, dtype=object)


def _independent_turn_labels(
    *,
    signed_heading_rate: np.ndarray,
    lateral_accel: np.ndarray,
    speed: np.ndarray,
    t: np.ndarray,
    config: DynamicSegmentationConfig,
    turn_rate_deadband_radps: float,
    lateral_deadband: float,
) -> np.ndarray:
    n = len(t)
    if n == 0 or not bool(config.segment_horizontal_turns):
        return np.asarray(["straight"] * n, dtype=object)

    hr = np.asarray(signed_heading_rate, dtype=float).reshape(-1)
    lat = np.asarray(lateral_accel, dtype=float).reshape(-1)
    speed = np.asarray(speed, dtype=float).reshape(-1)
    labels: list[str] = []
    for hri, lati, spdi in zip(hr, lat, speed):
        if float(spdi) < float(config.energy_ground_speed_threshold_mps):
            labels.append("straight")
            continue
        active = abs(float(hri)) >= float(turn_rate_deadband_radps) or abs(float(lati)) >= float(lateral_deadband)
        if not active:
            labels.append("straight")
            continue
        direction = "left" if float(hri) >= 0.0 else "right"
        band = _magnitude_band(
            math.degrees(abs(float(hri))),
            float(config.turn_rate_medium_threshold_degps),
            float(config.turn_rate_rapid_threshold_degps),
        )
        labels.append(f"turn_{direction}_{band}")

    arr = np.asarray(labels, dtype=object)
    # Also catch sustained low-rate heading changes.  Holds/traffic-vector arcs
    # can average below the instantaneous deadband while still accumulating a
    # large heading change that must be its own spline segment.
    arr = _mark_cumulative_turns(arr, hr, speed, t, config)
    # Bridge short straight gaps inside same-direction turns.  ADS-B track angle
    # quantization often creates brief false-straight islands in a real turn.
    arr = _bridge_turn_gaps(
        arr,
        t,
        max_gap_points=max(2, min(12, int(config.min_segment_points))),
        max_gap_s=max(4.0, min(18.0, float(config.min_segment_duration_s))),
    )
    arr = _drop_weak_turn_runs(arr, hr, t, config)
    arr = _stabilize_labels(
        arr,
        t,
        min_points=max(2, min(int(config.energy_state_min_points), int(config.min_segment_points))),
        min_duration=max(0.0, min(float(config.energy_state_min_duration_s), float(config.min_segment_duration_s))),
        protected_tokens=("turn_",),
        protected_min_points=max(2, min(4, int(config.min_segment_points))),
        protected_min_duration=max(0.0, min(4.0, float(config.min_segment_duration_s))),
    )
    arr = _drop_weak_turn_runs(arr, hr, t, config)
    return arr.astype(object)



def _mark_cumulative_turns(
    labels: np.ndarray,
    signed_heading_rate: np.ndarray,
    speed: np.ndarray,
    t: np.ndarray,
    config: DynamicSegmentationConfig,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=object).copy()
    hr = np.asarray(signed_heading_rate, dtype=float).reshape(-1)
    speed = np.asarray(speed, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = len(labels)
    if n < 3:
        return labels
    dt = np.diff(t)
    if dt.size == 0:
        return labels
    # Search roughly one procedure-turn/vectoring arc.  This is independent of
    # sample rate because the data are time-indexed.
    window_s = max(20.0, min(60.0, 3.0 * float(config.min_segment_duration_s)))
    min_heading_rad = math.radians(max(12.0, float(config.turn_min_heading_change_deg)))
    for i in range(n):
        if str(labels[i]).startswith("turn_"):
            continue
        if float(speed[min(i, len(speed) - 1)]) < float(config.energy_ground_speed_threshold_mps):
            continue
        t0 = float(t[i]) - 0.5 * window_s
        t1 = float(t[i]) + 0.5 * window_s
        lo = int(np.searchsorted(t, t0, side="left"))
        hi = int(np.searchsorted(t, t1, side="right")) - 1
        lo = max(0, min(lo, n - 2))
        hi = max(lo, min(hi, n - 1))
        if hi <= lo:
            continue
        # Integrate over intervals lo..hi-1 using sample-rate heading rates.
        hseg = 0.5 * (hr[lo:hi] + hr[lo + 1: hi + 1]) * dt[lo:hi]
        heading_change = float(np.nansum(hseg))
        if abs(heading_change) < min_heading_rad:
            continue
        direction = "left" if heading_change >= 0.0 else "right"
        mean_rate_degps = math.degrees(abs(heading_change)) / max(float(t[hi] - t[lo]), 1e-6)
        band = _magnitude_band(
            mean_rate_degps,
            float(config.turn_rate_medium_threshold_degps),
            float(config.turn_rate_rapid_threshold_degps),
        )
        labels[i] = f"turn_{direction}_{band}"
    return labels

def _bridge_turn_gaps(labels: np.ndarray, t: np.ndarray, *, max_gap_points: int, max_gap_s: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=object).copy()
    t = np.asarray(t, dtype=float)
    changed = True
    while changed:
        changed = False
        runs = _label_runs(labels, t)
        for idx in range(1, len(runs) - 1):
            run = runs[idx]
            if str(run["label"]) != "straight":
                continue
            left = str(runs[idx - 1]["label"])
            right = str(runs[idx + 1]["label"])
            if not (left.startswith("turn_") and right.startswith("turn_")):
                continue
            if _turn_direction(left) != _turn_direction(right):
                continue
            if int(run["n_points"]) > int(max_gap_points) or float(run["duration_s"]) > float(max_gap_s):
                continue
            labels[int(run["start_idx"]): int(run["end_idx"]) + 1] = left
            changed = True
            break
    return labels


def _drop_weak_turn_runs(labels: np.ndarray, signed_heading_rate: np.ndarray, t: np.ndarray, config: DynamicSegmentationConfig) -> np.ndarray:
    labels = np.asarray(labels, dtype=object).copy()
    hr = np.asarray(signed_heading_rate, dtype=float)
    t = np.asarray(t, dtype=float)
    if labels.size == 0:
        return labels
    min_heading_rad = math.radians(max(0.0, float(config.turn_min_heading_change_deg)))
    # A short high-rate heading arc is still a real turn.  Do not require the
    # general energy-segment minimum here; final boundary cleanup enforces spline
    # support after neighbouring runs are bridged/merged.
    min_duration = max(0.0, min(4.0, float(config.min_segment_duration_s)))
    dt = np.diff(t)
    for run in _label_runs(labels, t):
        label = str(run["label"])
        if not label.startswith("turn_"):
            continue
        start = int(run["start_idx"])
        end = int(run["end_idx"])
        if end <= start or dt.size == 0:
            heading_change = 0.0
        else:
            lo = max(0, start)
            hi = min(end, len(dt) - 1)
            heading_change = abs(float(np.nansum(0.5 * (hr[lo: hi + 1] + hr[lo + 1: hi + 2]) * dt[lo: hi + 1])))
        if heading_change < min_heading_rad or float(run["duration_s"]) < min_duration:
            labels[start: end + 1] = "straight"
    return labels


def _turn_direction(label: str) -> str:
    text = str(label)
    if "turn_left" in text:
        return "left"
    if "turn_right" in text:
        return "right"
    return "none"


def _stabilize_labels(
    labels: np.ndarray,
    t: np.ndarray,
    *,
    min_points: int,
    min_duration: float,
    protected_tokens: tuple[str, ...] = (),
    protected_min_points: int | None = None,
    protected_min_duration: float | None = None,
) -> np.ndarray:
    """Merge unsupported categorical runs without hiding protected maneuvers."""
    labels = np.asarray(labels, dtype=object).copy()
    t = np.asarray(t, dtype=float)
    if labels.size == 0:
        return labels
    protected_min_points = min_points if protected_min_points is None else int(protected_min_points)
    protected_min_duration = min_duration if protected_min_duration is None else float(protected_min_duration)

    for _ in range(12):
        runs = _label_runs(labels, t)
        changed = False
        if len(runs) <= 1:
            break
        for idx, run in enumerate(runs):
            label = str(run["label"])
            is_protected = any(token in label for token in protected_tokens)
            req_points = protected_min_points if is_protected else int(min_points)
            req_duration = protected_min_duration if is_protected else float(min_duration)
            if int(run["n_points"]) >= req_points and float(run["duration_s"]) >= req_duration:
                continue

            left = runs[idx - 1] if idx > 0 else None
            right = runs[idx + 1] if idx + 1 < len(runs) else None
            replacement: str | None = None
            if left is not None and right is not None and str(left["label"]) == str(right["label"]):
                replacement = str(left["label"])
            elif left is not None and right is not None:
                # Prefer the stronger neighbouring primitive; this avoids small
                # chattering islands while keeping long climb/turn states intact.
                left_strength = float(left["duration_s"]) + 0.25 * float(left["n_points"])
                right_strength = float(right["duration_s"]) + 0.25 * float(right["n_points"])
                replacement = str(left["label"] if left_strength >= right_strength else right["label"])
            elif left is not None:
                replacement = str(left["label"])
            elif right is not None:
                replacement = str(right["label"])

            if replacement is not None and replacement != label:
                labels[int(run["start_idx"]): int(run["end_idx"]) + 1] = replacement
                changed = True
                break
        if not changed:
            break
    return labels

def _suppress_weak_turn_runs(labels: np.ndarray, signed_heading_rate: np.ndarray, t: np.ndarray, config: DynamicSegmentationConfig) -> np.ndarray:
    labels = np.asarray(labels, dtype=object).copy()
    hr = np.asarray(signed_heading_rate, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(labels)
    if n == 0:
        return labels
    min_heading_rad = math.radians(max(0.0, float(config.turn_min_heading_change_deg)))
    dt = np.diff(t)
    start = 0
    while start < n:
        current = str(labels[start])
        end = start
        while end + 1 < n and str(labels[end + 1]) == current:
            end += 1
        if "turn_" in current:
            if end > start and dt.size:
                a = max(0, start)
                b = min(end, len(dt) - 1)
                heading_change = abs(float(np.nansum(0.5 * (hr[a : b + 1] + hr[a + 1 : b + 2]) * dt[a : b + 1])))
            else:
                heading_change = 0.0
            if heading_change < min_heading_rad:
                stripped = _strip_turn_from_label(current)
                labels[start : end + 1] = [stripped for _ in range(end - start + 1)]
        start = end + 1
    return labels


def _strip_turn_from_label(label: str) -> str:
    parts = [p for p in str(label).split("__") if not p.startswith("turn_")]
    return "__".join(parts) if parts else "energy_unknown"


def _energy_state_labels(
    *,
    speed: np.ndarray,
    vertical_rate: np.ndarray,
    speed_accel: np.ndarray,
    energy_rate: np.ndarray,
    ground_speed_threshold: float,
    energy_deadband: float,
    vertical_deadband: float,
    accel_deadband: float,
    smoothing_window: int,
    energy_rate_medium_threshold: float = 2.0,
    energy_rate_rapid_threshold: float = 6.0,
    vertical_rate_medium_threshold: float = 2.0,
    vertical_rate_rapid_threshold: float = 6.0,
    speed_accel_medium_threshold: float = 0.25,
    speed_accel_rapid_threshold: float = 0.8,
) -> np.ndarray:
    speed = _rolling_median(np.asarray(speed, dtype=float), smoothing_window)
    vertical_rate = _rolling_median(np.asarray(vertical_rate, dtype=float), smoothing_window)
    speed_accel = _rolling_median(np.asarray(speed_accel, dtype=float), smoothing_window)
    energy_rate = _rolling_median(np.asarray(energy_rate, dtype=float), smoothing_window)

    labels: list[str] = []
    for spd, vz, ax, edot in zip(speed, vertical_rate, speed_accel, energy_rate):
        if spd < ground_speed_threshold:
            labels.append("ground_slow")
            continue

        e_band = _magnitude_band(abs(float(edot)), energy_rate_medium_threshold, energy_rate_rapid_threshold)
        vz_band = _magnitude_band(abs(float(vz)), vertical_rate_medium_threshold, vertical_rate_rapid_threshold)
        ax_band = _magnitude_band(abs(float(ax)), speed_accel_medium_threshold, speed_accel_rapid_threshold)

        if edot > energy_deadband:
            if vz > vertical_deadband:
                labels.append(f"climb_{vz_band}_energy_gain")
            elif ax > accel_deadband:
                labels.append(f"level_accel_{ax_band}_energy_gain")
            else:
                labels.append(f"energy_gain_{e_band}")
        elif edot < -energy_deadband:
            if vz < -vertical_deadband:
                labels.append(f"descent_{vz_band}_energy_loss")
            elif ax < -accel_deadband:
                labels.append(f"level_decel_{ax_band}_energy_loss")
            else:
                labels.append(f"energy_loss_{e_band}")
        else:
            # Constant total energy is a single state even when the aircraft is
            # turning horizontally.  Only call it exchange if altitude and speed
            # are being traded strongly enough to persist beyond noise.
            strong_vz = abs(float(vz)) >= max(vertical_rate_medium_threshold, vertical_deadband)
            strong_ax = abs(float(ax)) >= max(speed_accel_medium_threshold, accel_deadband)
            if strong_vz and strong_ax and ((vz > 0 and ax < 0) or (vz < 0 and ax > 0)):
                direction = "climb" if vz > 0 else "descent"
                labels.append(f"energy_exchange_{direction}_{vz_band}")
            else:
                labels.append("energy_constant")
    return np.asarray(labels, dtype=object)


def _magnitude_band(value: float, medium_threshold: float, rapid_threshold: float) -> str:
    value = abs(float(value))
    medium = max(0.0, float(medium_threshold))
    rapid = max(medium, float(rapid_threshold))
    if value >= rapid:
        return "rapid"
    if value >= medium:
        return "medium"
    return "slow"


def _label_runs(labels: np.ndarray, t: np.ndarray) -> list[dict[str, Any]]:
    if len(labels) == 0:
        return []
    runs: list[dict[str, Any]] = []
    start = 0
    current = str(labels[0])
    for i in range(1, len(labels)):
        if str(labels[i]) != current:
            runs.append(_make_label_run(current, start, i - 1, t))
            start = i
            current = str(labels[i])
    runs.append(_make_label_run(current, start, len(labels) - 1, t))
    return runs


def _make_label_run(label: str, start: int, end: int, t: np.ndarray) -> dict[str, Any]:
    return {
        "label": str(label),
        "start_idx": int(start),
        "end_idx": int(end),
        "t0": float(t[start]),
        "t1": float(t[end]),
        "duration_s": float(t[end] - t[start]) if end >= start else 0.0,
        "n_points": int(end - start + 1),
    }


def _cleanup_candidates(
    component: HardGapComponent,
    candidates: Sequence[SegmentationBoundaryCandidate],
    config: DynamicSegmentationConfig,
) -> list[AcceptedBoundary]:
    """Select a feasible boundary set with dynamic programming.

    Candidate generation now deliberately proposes every sustained etalon
    primitive transition.  A greedy NMS can drop weak-but-important turn starts
    when they sit near a stronger altitude candidate, so cleanup is formulated as
    a path-selection problem: maximize transition value while respecting minimum
    segment support, duration, spacing, and the segment-count cap.
    """
    if not candidates:
        return []

    t = np.asarray([s.t for s in component.samples], dtype=float)
    n = component.n_observations
    min_pts = max(2, int(config.min_segment_points))
    min_dur = max(0.0, float(config.min_segment_duration_s))
    spacing = max(0.0, float(config.min_boundary_spacing_s))
    max_boundaries = max(0, int(config.max_segments_per_component) - 1)
    if max_boundaries <= 0:
        return []

    # Aggregate candidates that landed on the same local sample.
    by_local: dict[int, SegmentationBoundaryCandidate] = {}
    for c in candidates:
        local = int(c.sample_index - component.start_sample_index)
        if local <= 0 or local >= n - 1:
            continue
        if (local + 1) < min_pts or (n - local) < min_pts:
            continue
        if (t[local] - t[0]) < min_dur or (t[-1] - t[local]) < min_dur:
            continue
        previous = by_local.get(local)
        if previous is None:
            by_local[local] = c
        else:
            by_local[local] = SegmentationBoundaryCandidate(
                sample_index=int(c.sample_index),
                t_boundary=float(c.t_boundary),
                score=max(float(previous.score), float(c.score)),
                reasons=tuple(sorted(set(previous.reasons + c.reasons))),
                feature_snapshot=previous.feature_snapshot,
            )

    ordered_locals = sorted(by_local)
    if not ordered_locals:
        return []

    # State: (last_local_index, selected_count) -> (score, path_locals)
    # last_local_index uses -1 to denote the component start before any boundary.
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(-1, 0): (0.0, ())}
    for local in ordered_locals:
        candidate = by_local[local]
        next_states = dict(states)
        for (last_local, count), (score_so_far, path) in states.items():
            if count >= max_boundaries:
                continue
            start_local = 0 if last_local < 0 else last_local
            if not _segment_span_is_feasible(t, start_local, local, min_pts=min_pts, min_dur=min_dur):
                continue
            if last_local >= 0 and (float(t[local]) - float(t[last_local])) < spacing:
                continue
            # Require the remaining tail to still be feasible.  This lets the DP
            # keep early boundaries only if a complete segmentation remains possible.
            if not _segment_span_is_feasible(t, local, n - 1, min_pts=min_pts, min_dur=min_dur):
                continue
            value = _candidate_selection_value(candidate)
            key = (local, count + 1)
            trial = (score_so_far + value, (*path, local))
            prev = next_states.get(key)
            if prev is None or trial[0] > prev[0]:
                next_states[key] = trial
        states = next_states

    best_score = -math.inf
    best_path: tuple[int, ...] = ()
    for (last_local, count), (score, path) in states.items():
        if count == 0:
            continue
        if last_local >= 0:
            if not _segment_span_is_feasible(t, last_local, n - 1, min_pts=min_pts, min_dur=min_dur):
                continue
        # Small complexity penalty prevents adding marginal duplicate boundaries
        # when the candidate value is essentially tied.
        adjusted = score - 0.01 * count
        if adjusted > best_score:
            best_score = adjusted
            best_path = path

    accepted: list[AcceptedBoundary] = []
    for local in sorted(best_path):
        c = by_local[int(local)]
        accepted.append(
            AcceptedBoundary(
                boundary_id=f"{component.component_id}_bnd_{len(accepted) + 1:04d}",
                component_id=component.component_id,
                sample_index=int(c.sample_index),
                local_sample_index=int(local),
                t_boundary=float(c.t_boundary),
                reasons=tuple(sorted(set(c.reasons + ("dynamic_boundary_selection",)))),
                score=float(c.score),
            )
        )
    return accepted


def _segment_span_is_feasible(
    t: np.ndarray,
    start: int,
    end: int,
    *,
    min_pts: int,
    min_dur: float,
) -> bool:
    start = int(start)
    end = int(end)
    if end < start or start < 0 or end >= len(t):
        return False
    if (end - start + 1) < int(min_pts):
        return False
    if (float(t[end]) - float(t[start])) < float(min_dur):
        return False
    return True


def _candidate_selection_value(candidate: SegmentationBoundaryCandidate) -> float:
    reasons = " ".join(str(r) for r in candidate.reasons)
    value = float(candidate.score)
    # A turn start/end is a hard semantic requirement for local splines: a
    # long heading-change arc should never be hidden inside energy_constant.
    if "turn_state_transition" in reasons or "turn_etalon_transition" in reasons:
        value += 160.0
    if "altitude_lobe_transition" in reasons or "vertical_reversal_transition" in reasons or "go_around_transition" in reasons:
        value += 75.0
    if "climb_state_transition" in reasons or "descent_state_transition" in reasons:
        value += 50.0
    if "energy_etalon_transition" in reasons or "energy_state_transition" in reasons:
        value += 12.0
    if "rough_air_state_transition" in reasons:
        value += 30.0
    if "pelt" in reasons:
        value -= 10.0
    return value

def _limit_boundaries_by_score(
    component: HardGapComponent,
    boundaries: Sequence[AcceptedBoundary],
    config: DynamicSegmentationConfig,
) -> list[AcceptedBoundary]:
    """Keep the strongest feasible boundaries when a component is over-split.

    The previous cap kept the earliest boundaries after sorting in time.  On
    long descents this could discard stronger go-around/turn/energy boundaries
    later in the track.  This version keeps high-score boundaries while checking
    that the final ordered set still leaves enough samples and duration on every
    segment.
    """
    max_boundaries = max(0, int(config.max_segments_per_component) - 1)
    if len(boundaries) <= max_boundaries:
        return list(boundaries)

    t = np.asarray([s.t for s in component.samples], dtype=float)
    min_pts = max(2, int(config.min_segment_points))
    min_dur = max(0.0, float(config.min_segment_duration_s))
    spacing = max(0.0, float(config.min_boundary_spacing_s))

    chosen: list[AcceptedBoundary] = []
    for candidate in sorted(boundaries, key=lambda b: (-float(b.score), int(b.local_sample_index))):
        if len(chosen) >= max_boundaries:
            break
        trial = sorted([*chosen, candidate], key=lambda b: int(b.local_sample_index))
        if not _boundary_set_is_feasible(t, trial, min_pts=min_pts, min_dur=min_dur, spacing=spacing):
            continue
        chosen = trial

    # If greedy score selection could not fill the budget because of support
    # constraints, the chosen subset is still the safest answer.  Renumber IDs so
    # downstream reports remain stable and ordered.
    renumbered: list[AcceptedBoundary] = []
    for idx, boundary in enumerate(sorted(chosen, key=lambda b: int(b.local_sample_index)), start=1):
        renumbered.append(
            AcceptedBoundary(
                boundary_id=f"{component.component_id}_bnd_{idx:04d}",
                component_id=boundary.component_id,
                sample_index=boundary.sample_index,
                local_sample_index=boundary.local_sample_index,
                t_boundary=boundary.t_boundary,
                reasons=tuple([*boundary.reasons, "max_segment_cap_score_selected"]),
                score=boundary.score,
                is_hard_gap=boundary.is_hard_gap,
            )
        )
    return renumbered


def _boundary_set_is_feasible(
    t: np.ndarray,
    boundaries: Sequence[AcceptedBoundary],
    *,
    min_pts: int,
    min_dur: float,
    spacing: float,
) -> bool:
    ordered = sorted(boundaries, key=lambda b: int(b.local_sample_index))
    locals_ = [int(b.local_sample_index) for b in ordered]
    if any(b <= 0 or b >= len(t) - 1 for b in locals_):
        return False
    if any((t[b] - t[a]) < spacing for a, b in zip(locals_[:-1], locals_[1:])):
        return False
    starts = [0] + locals_
    ends = locals_ + [len(t) - 1]
    for start, end in zip(starts, ends):
        if (end - start + 1) < min_pts:
            return False
        if (float(t[end]) - float(t[start])) < min_dur:
            return False
    return True


def _segment_state_features_from_component(
    component_features: dict[str, Any],
    start_local: int,
    end_local: int,
) -> dict[str, Any]:
    """Summarize component-context regime labels over a segment span.

    Segment-local robust normalization is useful for numeric features, but it
    can erase rough/noisy-air context after a boundary has already isolated that
    interval.  This helper projects the full-component labels onto each segment
    so the label, smoothing policy, and diagnostics reflect the state transition
    that created the segment.
    """
    out: dict[str, Any] = {}
    if start_local < 0 or end_local < start_local:
        return out
    sl = slice(int(start_local), int(end_local) + 1)

    labels = np.asarray(component_features.get("regime_state_labels", []), dtype=object).reshape(-1)
    if labels.size > int(end_local):
        dominant, fraction = _dominant_label(labels[sl])
        out["dominant_regime_state"] = dominant
        out["dominant_regime_state_fraction"] = float(fraction)

    label_parts_raw = component_features.get("regime_label_parts") or {}
    if isinstance(label_parts_raw, dict):
        part_specs = (
            ("energy", "dominant_energy_state", "dominant_energy_state_fraction"),
            ("turn", "dominant_turn_state", "dominant_turn_state_fraction"),
            ("rough_air", "dominant_rough_air_state", "dominant_rough_air_state_fraction"),
        )
        for source_key, value_key, fraction_key in part_specs:
            part_arr = np.asarray(label_parts_raw.get(source_key, []), dtype=object).reshape(-1)
            if part_arr.size > int(end_local):
                dominant, fraction = _dominant_label(part_arr[sl])
                out[value_key] = dominant
                out[fraction_key] = float(fraction)

    rough = np.asarray(component_features.get("rough_air_score", []), dtype=float).reshape(-1)
    if rough.size > int(end_local):
        seg = np.nan_to_num(rough[sl], nan=0.0, posinf=0.0, neginf=0.0)
        if seg.size:
            out["median_rough_air_score"] = float(np.median(seg))
            out["p95_rough_air_score"] = float(np.quantile(seg, 0.95))
    return out


def _make_dynamic_segment(
    component: HardGapComponent,
    *,
    config: DynamicSegmentationConfig | None = None,
    component_features: dict[str, Any] | None = None,
    start_local: int,
    end_local: int,
    segment_number: int,
    start_boundary_id: str | None,
    end_boundary_id: str | None,
) -> DynamicSegment:
    samples = tuple(component.samples[start_local : end_local + 1])
    features = _segment_features(samples, config)
    if component_features is not None:
        features.update(_segment_state_features_from_component(component_features, start_local, end_local))
    regime = classify_regime(features)
    return DynamicSegment(
        segment_id=f"{component.component_id}_seg_{segment_number:04d}",
        component_id=component.component_id,
        start_sample_index=component.start_sample_index + int(start_local),
        end_sample_index=component.start_sample_index + int(end_local),
        samples=samples,
        t0=float(samples[0].t),
        t1=float(samples[-1].t),
        features=features,
        regime_label=regime,
        start_boundary_id=start_boundary_id,
        end_boundary_id=end_boundary_id,
    )


def classify_regime(features: dict[str, float]) -> str:
    """Classify a viewer/debug segment by dominant composite regime state."""
    speed = float(features.get("median_horizontal_speed_mps", 0.0))
    mismatch = float(features.get("median_velocity_mismatch_mps", 0.0))
    regime_state = str(
        features.get(
            "dominant_regime_state",
            features.get("dominant_energy_state", "energy_unknown"),
        )
    )

    if speed < 25.0 or regime_state == "ground_slow":
        return "ground_slow"

    if mismatch > max(20.0, 0.45 * max(speed, 1.0)) and "rough_air" not in regime_state:
        return f"{regime_state}__surveillance_noisy" if regime_state != "energy_unknown" else "noisy_airborne"

    return regime_state

def _segment_features(
    samples: Sequence[PreparedVSplineSample],
    config: DynamicSegmentationConfig | None = None,
) -> dict[str, float]:
    cfg = config or DynamicSegmentationConfig()
    t = np.asarray([s.t for s in samples], dtype=float)
    y_raw = np.asarray([s.y for s in samples], dtype=float)
    v_raw = np.asarray([s.v for s in samples], dtype=float)
    signal_source = str(getattr(cfg, "segmentation_feature_source", "raw"))
    kalman_used = False
    if signal_source == "kalman_rts":
        y, v, kalman_diag = smooth_samples_for_segmentation(samples, cfg.kalman_segmentation_config)
        kalman_used = bool(kalman_diag.get("used", False))
    else:
        y, v = y_raw, v_raw

    dt = np.diff(t)
    dt_safe = np.maximum(dt, 1e-6)
    speed_h = np.linalg.norm(v[:, :2], axis=1)
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    signed_heading_rate_interval = np.diff(heading) / dt_safe if dt.size else np.zeros(1)
    signed_heading_rate = _interval_to_sample_feature(signed_heading_rate_interval, len(samples)) if len(samples) > 1 else np.zeros(1)
    heading_rate = np.abs(signed_heading_rate)
    lateral_accel = speed_h * signed_heading_rate[: len(speed_h)]
    vertical_acc = np.diff(v[:, 2]) / dt_safe if dt.size else np.zeros(1)
    speed_accel = _sample_gradient(speed_h, t)
    specific_energy_height = y[:, 2] + (speed_h**2) / (2.0 * STANDARD_GRAVITY_MPS2)
    specific_energy_rate = _sample_gradient(specific_energy_height, t)
    pos_vel = np.diff(y, axis=0) / dt_safe[:, None] if dt.size else np.zeros((1, 3))
    pos_vel_sample = np.vstack([pos_vel[0:1], 0.5 * (pos_vel[:-1] + pos_vel[1:]), pos_vel[-1:]]) if len(samples) > 2 else np.vstack([pos_vel, pos_vel[-1:]])
    mismatch = np.linalg.norm(v - pos_vel_sample[: len(samples)], axis=1)
    residual_proxy = _local_chord_residual(t, y, window=5)
    total_heading_change = float(abs(heading[-1] - heading[0])) if heading.size else 0.0
    duration = float(t[-1] - t[0]) if t.size >= 2 else 0.0

    erate_dead = max(float(cfg.energy_rate_deadband_mps), float(cfg.energy_rate_deadband_scale) * _robust_scale(specific_energy_rate))
    vz_dead = max(float(cfg.vertical_rate_deadband_mps), float(cfg.vertical_rate_deadband_scale) * _robust_scale(v[:, 2]))
    ax_dead = max(float(cfg.speed_accel_deadband_mps2), float(cfg.speed_accel_deadband_scale) * _robust_scale(speed_accel))
    turn_dead = max(math.radians(float(cfg.turn_rate_deadband_degps)), float(cfg.turn_rate_deadband_scale) * _robust_scale(signed_heading_rate))
    lat_dead = max(float(cfg.lateral_accel_deadband_mps2), float(cfg.lateral_accel_deadband_scale) * _robust_scale(lateral_accel))

    mini_raw = np.column_stack([mismatch, residual_proxy, _interval_to_sample_feature(vertical_acc, len(samples)), speed_accel])
    mini_z = _robust_standardize(mini_raw)
    rough_score = _rough_air_score(
        z_velocity_mismatch=mini_z[:, 0],
        z_residual=mini_z[:, 1],
        z_vertical_accel=mini_z[:, 2],
        z_speed_accel=mini_z[:, 3],
        config=cfg,
    )

    regime_labels, label_parts = _regime_state_labels(
        speed=speed_h,
        vertical_rate=v[:, 2],
        speed_accel=speed_accel,
        energy_rate=specific_energy_rate,
        signed_heading_rate=signed_heading_rate,
        lateral_accel=lateral_accel,
        rough_score=rough_score,
        t=t,
        config=cfg,
        energy_deadband=erate_dead,
        vertical_deadband=vz_dead,
        accel_deadband=ax_dead,
        turn_rate_deadband_radps=turn_dead,
        lateral_deadband=lat_dead,
    )
    energy_labels = np.asarray(label_parts["energy"], dtype=object)
    turn_labels = np.asarray(label_parts["turn"], dtype=object)
    rough_labels = np.asarray(label_parts["rough_air"], dtype=object)
    dominant_regime_state, dominant_regime_fraction = _dominant_label(regime_labels)
    dominant_energy_state, dominant_energy_fraction = _dominant_label(energy_labels)
    dominant_turn_state, dominant_turn_fraction = _dominant_label(turn_labels)
    dominant_rough_state, dominant_rough_fraction = _dominant_label(rough_labels)
    rough_p95 = float(np.quantile(rough_score, 0.95)) if rough_score.size else 0.0
    if bool(cfg.enable_rough_air_segmentation) and rough_p95 >= float(cfg.rough_air_score_threshold) and "rough_air" not in str(dominant_regime_state):
        dominant_regime_state = f"{dominant_energy_state}__rough_air_mixed"

    return {
        "duration_s": duration,
        "n_observations": float(len(samples)),
        "segmentation_feature_source": 1.0 if signal_source == "kalman_rts" else 0.0,
        "kalman_segmentation_used": 1.0 if kalman_used else 0.0,
        "median_dt_s": float(np.median(dt)) if dt.size else 0.0,
        "median_horizontal_speed_mps": float(np.median(speed_h)) if speed_h.size else 0.0,
        "p95_horizontal_speed_mps": float(np.quantile(speed_h, 0.95)) if speed_h.size else 0.0,
        "median_vertical_rate_mps": float(np.median(v[:, 2])) if v.size else 0.0,
        "median_abs_vertical_rate_mps": float(np.median(np.abs(v[:, 2]))) if v.size else 0.0,
        "p95_heading_rate_degps": float(np.degrees(np.quantile(heading_rate, 0.95))) if heading_rate.size else 0.0,
        "median_signed_heading_rate_degps": float(np.degrees(np.median(signed_heading_rate))) if signed_heading_rate.size else 0.0,
        "p95_abs_lateral_acceleration_mps2": float(np.quantile(np.abs(lateral_accel), 0.95)) if lateral_accel.size else 0.0,
        "total_heading_change_deg": float(np.degrees(total_heading_change)),
        "p95_vertical_acceleration_mps2": float(np.quantile(np.abs(vertical_acc), 0.95)) if vertical_acc.size else 0.0,
        "median_speed_acceleration_mps2": float(np.median(speed_accel)) if speed_accel.size else 0.0,
        "p95_abs_speed_acceleration_mps2": float(np.quantile(np.abs(speed_accel), 0.95)) if speed_accel.size else 0.0,
        "median_specific_energy_height_m": float(np.median(specific_energy_height)) if specific_energy_height.size else 0.0,
        "specific_energy_height_change_m": float(specific_energy_height[-1] - specific_energy_height[0]) if specific_energy_height.size else 0.0,
        "median_specific_energy_rate_mps": float(np.median(specific_energy_rate)) if specific_energy_rate.size else 0.0,
        "p95_abs_specific_energy_rate_mps": float(np.quantile(np.abs(specific_energy_rate), 0.95)) if specific_energy_rate.size else 0.0,
        "dominant_regime_state": dominant_regime_state,
        "dominant_regime_state_fraction": float(dominant_regime_fraction),
        "dominant_energy_state": dominant_energy_state,
        "dominant_energy_state_fraction": float(dominant_energy_fraction),
        "dominant_turn_state": dominant_turn_state,
        "dominant_turn_state_fraction": float(dominant_turn_fraction),
        "dominant_rough_air_state": dominant_rough_state,
        "dominant_rough_air_state_fraction": float(dominant_rough_fraction),
        "energy_rate_deadband_mps": float(erate_dead),
        "vertical_rate_deadband_mps": float(vz_dead),
        "speed_accel_deadband_mps2": float(ax_dead),
        "turn_rate_deadband_degps": float(np.degrees(turn_dead)),
        "lateral_accel_deadband_mps2": float(lat_dead),
        "median_rough_air_score": float(np.median(rough_score)) if rough_score.size else 0.0,
        "p95_rough_air_score": rough_p95,
        "median_velocity_mismatch_mps": float(np.median(mismatch)) if mismatch.size else 0.0,
    }


def _add_candidate(
    out: dict[int, SegmentationBoundaryCandidate],
    component: HardGapComponent,
    local_idx: int,
    score: float,
    reasons: tuple[str, ...],
    raw: np.ndarray,
    z: np.ndarray,
    names: list[str],
) -> None:
    local_idx = int(min(max(local_idx, 1), component.n_observations - 2))
    global_idx = component.start_sample_index + local_idx
    snapshot = {f"{name}": float(raw[local_idx, i]) for i, name in enumerate(names)}
    snapshot.update({f"z_{name}": float(z[local_idx, i]) for i, name in enumerate(names)})
    previous = out.get(global_idx)
    if previous is None:
        out[global_idx] = SegmentationBoundaryCandidate(
            sample_index=global_idx,
            t_boundary=float(component.samples[local_idx].t),
            score=float(score),
            reasons=tuple(sorted(set(reasons))),
            feature_snapshot=snapshot,
        )
    else:
        out[global_idx] = SegmentationBoundaryCandidate(
            sample_index=global_idx,
            t_boundary=previous.t_boundary,
            score=max(float(previous.score), float(score)),
            reasons=tuple(sorted(set(previous.reasons + reasons))),
            feature_snapshot=previous.feature_snapshot,
        )



def _sample_gradient(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=float)
    if n == 2:
        dt = max(float(t[1] - t[0]), 1e-6)
        g = (values[1] - values[0]) / dt
        return np.asarray([g, g], dtype=float)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            a, b = 0, 1
        elif i == n - 1:
            a, b = n - 2, n - 1
        else:
            a, b = i - 1, i + 1
        dt = max(float(t[b] - t[a]), 1e-6)
        out[i] = (values[b] - values[a]) / dt
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    if n == 0 or int(window) <= 1:
        return values.astype(float, copy=True)
    half = max(1, int(window) // 2)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = float(np.nanmedian(values[a:b]))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    scale = 1.4826 * mad
    if scale <= 1e-9:
        scale = float(np.nanstd(values))
    return float(scale if math.isfinite(scale) and scale > 1e-9 else 0.0)


def _dominant_label(labels: np.ndarray) -> tuple[str, float]:
    labels = np.asarray(labels, dtype=object).reshape(-1)
    if labels.size == 0:
        return "energy_unknown", 0.0
    unique, counts = np.unique(labels.astype(str), return_counts=True)
    idx = int(np.argmax(counts))
    return str(unique[idx]), float(counts[idx] / max(labels.size, 1))


def _interval_to_sample_feature(interval_values: np.ndarray, n: int) -> np.ndarray:
    interval_values = np.asarray(interval_values, dtype=float).reshape(-1)
    if n <= 1:
        return np.zeros(n, dtype=float)
    if n == 2:
        return np.asarray([interval_values[0], interval_values[0]], dtype=float)
    return np.concatenate(
        [
            interval_values[0:1],
            0.5 * (interval_values[:-1] + interval_values[1:]),
            interval_values[-1:],
        ]
    )


def _robust_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    std = np.nanstd(x, axis=0)
    scale = 1.4826 * mad
    # If most values are identical, MAD is zero.  Fall back to standard
    # deviation, then to 1.0 so isolated legitimate regime changes do not
    # explode to absurd z scores.
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    z = (x - med) / scale
    z = np.clip(z, -50.0, 50.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _local_chord_residual(t: np.ndarray, y: np.ndarray, *, window: int) -> np.ndarray:
    n = len(t)
    if n < 3:
        return np.zeros(n, dtype=float)
    half = max(1, int(window) // 2)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n - 1, i + half)
        if b <= a or t[b] == t[a]:
            continue
        alpha = (t[i] - t[a]) / (t[b] - t[a])
        interp = (1.0 - alpha) * y[a] + alpha * y[b]
        out[i] = float(np.linalg.norm(y[i] - interp))
    return out
