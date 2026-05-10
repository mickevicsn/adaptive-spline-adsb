"""
End-to-end ADS-B track-output pipeline.

This script combines the existing project modules into one output builder that
writes the viewer folder layout:

    track_output/
      flights.json
      flights/<flightId>/flight.json
      flights/<flightId>/methods/raw_adsb.json
      flights/<flightId>/methods/v_spline_bspline_<preset>.json
      flights/<flightId>/methods/v_spline_hermite_<preset>.json
      flights/<flightId>/methods/minimal/*.json
      flights/<flightId>/debug/

The default production reconstruction set is intentionally compact: Kalman/RTS,
endpoint-guarded aviation quintic V-Splines, and the strongest cubic overlap
B-Spline comparator.  Legacy piecewise, global-component, join-smoothed, and
Hermite backends remain selectable diagnostics.  Kalman/RTS uses the same
prepared paired observations and post-processing JSON contract, but deliberately
skips dynamic spline segmentation because it is a state-space smoother rather
than a spline.

Default no-argument behavior
----------------------------
The intended entrypoint is the root-level ``main.py`` file, outside ``src/``.
It defines the database path, output folder, log folder, and ICAO/track list,
then calls this pipeline with an explicit configuration.

The pipeline can still be imported and called directly from tests or notebooks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import json
import math
import os
import shutil
import csv
import time

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from raw_adsb import (
    RawAdsbNormalizeConfig,
    RawAdsbNormalizer,
    write_json_packet,
)
from raw_keyframe_vspline_adapter import (
    RawKeyframeVSplineAdapter,
    RawKeyframeVSplineAdapterConfig,
    RawKeyframeVSplinePreparation,
)
from sql_loader import SqlAdsbLoadConfig, SqlAdsbLoader
from track_rules import (
    TrackRuleConfig,
    load_track_rule_registry,
)
from geo_utils import FT_TO_M, KNOT_TO_MPS, ecef_to_geodetic, geodetic_to_ecef
from kalman_rts_core import (
    KalmanRTSConfig,
    KalmanRTSInput,
    default_kalman_rts_config_for_preset,
    fit_kalman_rts_component,
)
from vspline.bspline_core import (
    BSplineAnchor,
    BSplineAccelerationPrior,
    BSplineCoreConfig,
    BSplineCoreInput,
    BSplinePositionPrior,
    BSplineVelocityConstraint,
    BSplineVelocityPrior,
    fit_b_spline_component,
)
from vspline.hermite_core import (
    VSplineCoreConfig as HermiteCoreConfig,
    VSplineCoreInput as HermiteCoreInput,
    VSplineEndpointConstraints as HermiteEndpointConstraints,
    VSplineEndpointState as HermiteEndpointState,
    fit_v_spline_core as fit_hermite_v_spline_core,
)
from trajectory_segmentation import (
    AcceptedBoundary,
    DynamicSegmentationConfig,
    DynamicSegment,
    HardGapComponent,
    SegmentedComponent,
    build_segmented_component_from_boundaries,
    segment_component,
    segment_prepared_samples,
)
from boundary_state import (
    BoundaryStateConfig,
    SharedBoundaryState,
    estimate_shared_boundary_states,
)
from vspline.segment_policy import LocalSegmentPolicyConfig, select_local_bspline_params
from vspline.local_tuning import (
    LocalSegmentTuningConfig,
    generate_bspline_param_candidates,
    score_bspline_candidate,
)
from vspline.quality import (
    aggregate_trajectory_model_metrics,
    evaluate_segment_quality,
    position_error_metrics,
    verify_component_continuity,
)
from vspline.velocity_confidence import compute_velocity_confidence_scale


SCHEMA_VERSION = "1.5"
MANIFEST_SCHEMA_VERSION = 1
RAW_ADSB_METHOD_ID = "raw_adsb"
BSPLINE_V_SPLINE_METHOD_ID = "v_spline_bspline"
HERMITE_V_SPLINE_METHOD_ID = "v_spline_hermite"
KALMAN_RTS_METHOD_ID = "kalman_rts"
RECONSTRUCTION_PRESETS = ("balanced", "accurate", "smooth")
V_SPLINE_PRESETS = RECONSTRUCTION_PRESETS
KALMAN_RTS_PRESETS = RECONSTRUCTION_PRESETS
# Keep this alias for older helper names; new payloads pass an explicit method id.
V_SPLINE_METHOD_ID = BSPLINE_V_SPLINE_METHOD_ID
# The latest 4BAAD9 runs show the endpoint-guarded quintic aviation spline as
# the strongest V-Spline family.  Keep only representative methods in the
# default manifest; legacy/global/Hermite backends remain available by explicit
# backend selection or opt-in flags in main.py.
DEFAULT_METHOD_ID = "aviation_v_spline_quintic_balanced"
MINIMAL_METHODS_DIRNAME = "minimal"
MINIMAL_PAYLOAD_VERSION = 2

GLOBAL_BSPLINE_BACKENDS = {"bspline_component_global"}
BSPLINE_BACKENDS = {
    "bspline_piecewise",
    "bspline_overlap",
    "bspline_join_smooth",
    "quintic_bspline",
    "quintic_kalman_boundary",
    *GLOBAL_BSPLINE_BACKENDS,
}
HERMITE_BACKENDS = {"hermite_piecewise", "hermite_stable"}
OVERLAP_SAVE_BACKENDS = {"bspline_overlap", "quintic_bspline", "quintic_kalman_boundary"}
SOFT_BOUNDARY_BACKENDS = {
    "bspline_overlap",
    "bspline_join_smooth",
    "hermite_stable",
    "quintic_bspline",
    "quintic_kalman_boundary",
}
KALMAN_BOUNDARY_BACKENDS = {"quintic_kalman_boundary"}


@dataclass(frozen=True)
class VSplineOutputSpec:
    method_id: str
    label: str
    file_stem: str
    backend: str
    preset: Literal["balanced", "accurate", "smooth"]


@dataclass(frozen=True)
class KalmanRTSOutputSpec:
    method_id: str
    label: str
    file_stem: str
    preset: Literal["balanced", "accurate", "smooth"]


@dataclass(frozen=True)
class SyntheticGapOutputSpec:
    """Diagnostic reconstruction run fitted after deleting raw data windows."""

    method_id: str
    label: str
    file_stem: str
    base_method_id: str
    base_family: Literal["kalman_rts", "v_spline"]
    base_spec: KalmanRTSOutputSpec | VSplineOutputSpec



def _default_hermite_config() -> HermiteCoreConfig:
    return HermiteCoreConfig(
        velocity_weight=0.015,
        penalty_mode="adaptive",
        adaptive_eta=120_000.0,
        adaptive_speed_floor_mps=1.0,
        smoothing_lambda=1.0,
        optimize=False,
        compute_loocv_score=False,
        condition_number_max_size=1024,
        hard_endpoint_constraints=True,
        hard_endpoint_positions=True,
        hard_endpoint_velocities=False,
    )


def _default_bspline_config_for_preset(preset: str) -> BSplineCoreConfig:
    by_preset = {
        "accurate": dict(
            knot_spacing_s=2.0,                       # was 2.5
            min_observations_per_basis=2.75,           # was 3.5
            velocity_weight=0.065,                     # was 0.050
            adaptive_eta=25_000.0,                     # was 38_000
            huber_delta_m=40.0,                        # was 50
            jerk_penalty_weight=0.00030,               # was 0.0007
            boundary_acceleration_prior_weight=0.018,  # was 0.030
        ),
        "balanced": dict(
            knot_spacing_s=3.5,                       # was 4.25
            min_observations_per_basis=4.75,           # was 6.0
            velocity_weight=0.050,                     # was 0.038
            adaptive_eta=52_000.0,                     # was 75_000
            huber_delta_m=55.0,                        # was 68
            jerk_penalty_weight=0.00080,               # was 0.0014
            boundary_acceleration_prior_weight=0.035,  # was 0.060
        ),
        "smooth": dict(
            knot_spacing_s=4.75,                      # was 6.0
            min_observations_per_basis=6.5,            # was 8.5
            velocity_weight=0.033,                     # was 0.022
            adaptive_eta=125_000.0,                    # was 190_000
            huber_delta_m=78.0,                        # was 100
            jerk_penalty_weight=0.00240,               # was 0.0042
            boundary_acceleration_prior_weight=0.060,  # was 0.105
        ),
    }[str(preset)]
    return BSplineCoreConfig(
        degree=3,
        # Report action: prevent overly dense short-segment bases and enable
        # Hessian conditioning diagnostics on normal local systems.
        min_knot_spacing_s=1.0,
        max_basis_count=900,
        position_weight=2.0,
        penalty_mode="adaptive",
        smoothing_lambda=1.0,
        adaptive_speed_floor_mps=1.0,
        velocity_outlier_policy="position_difference_gate",
        velocity_outlier_gate_mps=50.0,
        min_velocity_weight_scale=0.0,
        hard_boundary_positions=True,
        hard_component_endpoint_positions=True,
        hard_component_endpoint_velocities=False,
        component_endpoint_velocity_prior_weight=0.0,
        component_endpoint_acceleration_prior_weight=0.0,
        robust_position_loss="huber",
        robust_iterations=2,
        solver_ridge=1e-7,
        condition_number_max_basis=256,
        **by_preset,
    )


def _default_hermite_config_for_preset(preset: str) -> HermiteCoreConfig:
    base = _default_hermite_config()
    by_preset = {
        "accurate": dict(velocity_weight=0.02, adaptive_eta=60_000.0),
        "balanced": dict(velocity_weight=0.015, adaptive_eta=120_000.0),
        "smooth": dict(velocity_weight=0.008, adaptive_eta=280_000.0),
    }[str(preset)]
    return replace(base, penalty_mode="adaptive", smoothing_lambda=1.0, **by_preset)


def _default_policy_config_for_preset(preset: str) -> LocalSegmentPolicyConfig:
    if preset == "accurate":
        return LocalSegmentPolicyConfig(
            steady_adaptive_eta=60_000.0,
            transition_adaptive_eta=14_000.0,
            energy_change_adaptive_eta=30_000.0,
            energy_constant_adaptive_eta=90_000.0,
            noisy_adaptive_eta=180_000.0,
            steady_velocity_weight=0.04,
            transition_velocity_weight=0.035,
            energy_change_velocity_weight=0.032,
            energy_constant_velocity_weight=0.025,
            noisy_velocity_weight=0.01,
        )
    if preset == "smooth":
        return LocalSegmentPolicyConfig(
            steady_adaptive_eta=180_000.0,
            transition_adaptive_eta=35_000.0,
            energy_change_adaptive_eta=95_000.0,
            energy_constant_adaptive_eta=260_000.0,
            noisy_adaptive_eta=400_000.0,
            steady_velocity_weight=0.02,
            transition_velocity_weight=0.020,
            energy_change_velocity_weight=0.015,
            energy_constant_velocity_weight=0.010,
            noisy_velocity_weight=0.006,
        )
    return LocalSegmentPolicyConfig()


def _default_tuning_config_for_preset(preset: str) -> LocalSegmentTuningConfig:
    objective = {"accurate": "position", "balanced": "balanced", "smooth": "smooth"}[str(preset)]
    default_candidates = {"accurate": 14, "balanced": 14, "smooth": 10}[str(preset)]
    preset_key = str(preset)
    default_rmse = {"accurate": 90.0, "balanced": 120.0, "smooth": 160.0}[preset_key]
    default_p95 = {"accurate": 180.0, "balanced": 240.0, "smooth": 320.0}[preset_key]
    default_max = {"accurate": 450.0, "balanced": 600.0, "smooth": 800.0}[preset_key]
    default_vertical_rmse = {"accurate": 6.0, "balanced": 8.0, "smooth": 12.0}[preset_key]
    default_vertical_p95 = {"accurate": 12.0, "balanced": 16.0, "smooth": 24.0}[preset_key]
    default_vertical_max = {"accurate": 22.0, "balanced": 28.0, "smooth": 45.0}[preset_key]
    default_vertical_window = {"accurate": 7.5, "balanced": 10.0, "smooth": 16.0}[preset_key]
    return LocalSegmentTuningConfig(
        enabled=True,
        objective=objective,  # type: ignore[arg-type]
        max_candidates=default_candidates,
        include_all_candidate_reports=True,
        join_velocity_harmonization=True,
        adaptive_resegmentation_enabled=True,
        adaptive_resegmentation_max_passes=2,
        adaptive_resegmentation_bad_rmse_m=default_rmse,
        adaptive_resegmentation_bad_p95_m=default_p95,
        adaptive_resegmentation_bad_max_m=default_max,
        adaptive_resegmentation_bad_vertical_rmse_m=default_vertical_rmse,
        adaptive_resegmentation_bad_vertical_p95_m=default_vertical_p95,
        adaptive_resegmentation_bad_vertical_max_m=default_vertical_max,
        adaptive_resegmentation_bad_vertical_window_m=default_vertical_window,
        adaptive_resegmentation_vertical_run_min_points=3,
        adaptive_resegmentation_vertical_run_min_duration_s=3.0,
        adaptive_resegmentation_min_points=8,
        adaptive_resegmentation_min_duration_s=8.0,
        adaptive_resegmentation_min_boundary_spacing_s=8.0,
        adaptive_resegmentation_max_segments_per_component=18,
    )


def _overlap_guard_duration_s(backend: str, preset: str) -> float:
    if backend not in OVERLAP_SAVE_BACKENDS:
        return 0.0
    return {"accurate": 10.0, "balanced": 16.0, "smooth": 28.0}.get(str(preset), 16.0)


def _is_boundary_true_discontinuity(boundary: AcceptedBoundary | dict[str, Any] | None) -> bool:
    if boundary is None:
        return False
    if isinstance(boundary, dict):
        reasons = tuple(str(x).lower() for x in boundary.get("reasons", ()) or ())
        is_hard_gap = bool(boundary.get("is_hard_gap", False))
    else:
        reasons = tuple(str(x).lower() for x in boundary.reasons)
        is_hard_gap = bool(boundary.is_hard_gap)
    text = " ".join(reasons)
    return bool(
        is_hard_gap
        or "hard_gap" in text
        or "go_around" in text
        or "missed_approach" in text
        or "surveillance_discontinuity" in text
        or "track_discontinuity" in text
        or "true_discontinuity" in text
    )


def _boundary_event_bucket(boundary: AcceptedBoundary | dict[str, Any] | None) -> str:
    if boundary is None:
        return "normal_segment_join"
    if isinstance(boundary, dict):
        reasons = tuple(str(x).lower() for x in boundary.get("reasons", ()) or ())
        is_hard_gap = bool(boundary.get("is_hard_gap", False))
    else:
        reasons = tuple(str(x).lower() for x in boundary.reasons)
        is_hard_gap = bool(boundary.is_hard_gap)
    text = " ".join(reasons)
    if is_hard_gap or "hard_gap" in text:
        return "hard_gap"
    if "go_around" in text or "missed_approach" in text:
        return "go_around"
    if "surveillance" in text or "track_discontinuity" in text or "true_discontinuity" in text:
        return "surveillance_or_track_discontinuity"
    return "normal_segment_join"


def _segment_regime_bucket(segment: DynamicSegment | dict[str, Any] | None) -> str:
    if segment is None:
        return "unknown"
    label = str(segment.get("regime_label") if isinstance(segment, dict) else segment.regime_label).lower()
    features = segment.get("features", {}) if isinstance(segment, dict) else segment.features
    speed = float(features.get("median_horizontal_speed_mps", 0.0) or 0.0) if isinstance(features, dict) else 0.0
    if "ground" in label or speed < 20.0:
        return "ground"
    if "descent" in label or "approach" in label or "final" in label:
        return "approach_final"
    return "airborne"


def _regime_speed_floor_mps(segment: DynamicSegment, preset: str) -> float:
    bucket = _segment_regime_bucket(segment)
    if bucket == "ground":
        return {"accurate": 3.0, "balanced": 5.0, "smooth": 8.0}.get(str(preset), 5.0)
    if bucket == "approach_final":
        return {"accurate": 35.0, "balanced": 45.0, "smooth": 55.0}.get(str(preset), 45.0)
    if bucket == "airborne":
        return {"accurate": 25.0, "balanced": 35.0, "smooth": 45.0}.get(str(preset), 35.0)
    return {"accurate": 20.0, "balanced": 30.0, "smooth": 40.0}.get(str(preset), 30.0)


def _event_aware_component_continuity(
    segmented_component: SegmentedComponent,
    continuity: dict[str, Any],
) -> dict[str, Any]:
    """Summarize join derivative jumps by event/regime bucket.

    Jerk/continuity across hard gaps, go-arounds, and true surveillance/track
    discontinuities is reported but excluded from normal-continuity aggregates.
    """
    boundary_by_join: dict[tuple[str | None, str | None], AcceptedBoundary] = {}
    for left, right, boundary in zip(segmented_component.segments[:-1], segmented_component.segments[1:], segmented_component.boundaries):
        boundary_by_join[(left.segment_id, right.segment_id)] = boundary

    rows = []
    for item in continuity.get("boundaries", []) or []:
        left_id = item.get("left_segment_id")
        right_id = item.get("right_segment_id")
        boundary = boundary_by_join.get((left_id, right_id))
        left_segment = next((s for s in segmented_component.segments if s.segment_id == left_id), None)
        bucket = _boundary_event_bucket(boundary)
        regime_bucket = _segment_regime_bucket(left_segment)
        rows.append({
            **item,
            "boundary_id": None if boundary is None else boundary.boundary_id,
            "event_bucket": bucket,
            "regime_bucket": regime_bucket,
            "excluded_from_normal_continuity_score": bool(bucket != "normal_segment_join"),
        })

    def _summary(filter_fn) -> dict[str, Any]:
        selected = [r for r in rows if filter_fn(r)]
        def max_key(key: str) -> float:
            vals = [float(r.get(key) or 0.0) for r in selected if r.get(key) is not None]
            return float(max(vals) if vals else 0.0)
        return {
            "count": int(len(selected)),
            "max_position_jump_m": max_key("position_jump_m"),
            "max_velocity_jump_mps": max_key("velocity_jump_mps"),
            "max_acceleration_jump_mps2": max_key("acceleration_jump_mps2"),
            "max_jerk_jump_mps3": max_key("jerk_jump_mps3"),
        }

    by_event = {name: _summary(lambda r, n=name: r.get("event_bucket") == n) for name in sorted({r.get("event_bucket") for r in rows} | {"normal_segment_join", "hard_gap", "go_around"})}
    by_regime = {name: _summary(lambda r, n=name: r.get("regime_bucket") == n) for name in sorted({r.get("regime_bucket") for r in rows} | {"ground", "airborne", "approach_final"})}
    normal = _summary(lambda r: not bool(r.get("excluded_from_normal_continuity_score")))
    return {
        "enabled": True,
        "normal_interior_samples_note": "raw-fit and motion metrics remain per-segment; join derivative scores below exclude event boundaries where continuity is not claimed",
        "normal_segment_joins": normal,
        "by_event": by_event,
        "by_regime": by_regime,
        "rows": rows,
    }


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem paths used by the pipeline."""

    database_path: Path = Path(os.environ.get("ADSB_SQLITE_PATH", "adsb_raw.sqlite"))
    output_dir: Path = Path(os.environ.get("TRACK_OUTPUT_DIR", "track_output"))
    rules_path: Path | None = (
        Path(os.environ["TRACK_RULES_PATH"])
        if os.environ.get("TRACK_RULES_PATH")
        else None
    )


@dataclass(frozen=True)
class TrackOutputPipelineConfig:
    """Code-level pipeline configuration."""

    paths: PipelinePaths = field(default_factory=PipelinePaths)
    log_dir: Path = Path("logs")
    icao_list: tuple[str, ...] | None = None
    clean_output_dir: bool = True
    raw_keyframe_time_quantization_s: float = 1.0
    v_spline_time_step_s: float = 0.25
    v_spline_output_frequency_hz: float = 4.0
    include_raw_events_inline: bool = False
    write_debug_artifacts: bool = True
    dynamic_segmentation_config: DynamicSegmentationConfig = field(default_factory=DynamicSegmentationConfig)
    boundary_state_config: BoundaryStateConfig = field(default_factory=BoundaryStateConfig)
    local_segment_policy_config: LocalSegmentPolicyConfig = field(default_factory=LocalSegmentPolicyConfig)
    local_segment_tuning_config: LocalSegmentTuningConfig = field(default_factory=LocalSegmentTuningConfig)
    show_progress: bool = True

    # Production reconstruction can emit multiple comparable methods.
    # Kalman/RTS is a state-space smoother: it uses the same prepared ADS-B
    # observations and output schema but skips dynamic spline segmentation.
    kalman_rts_output_enabled: bool = True
    kalman_rts_output_presets: tuple[Literal["balanced", "accurate", "smooth"], ...] = (
        "balanced",
        "accurate",
        "smooth",
    )
    kalman_rts_config: KalmanRTSConfig = field(default_factory=KalmanRTSConfig)
    kalman_rts_config_by_preset: dict[str, KalmanRTSConfig] = field(default_factory=dict)

    # Production V-Spline methods.  Defaults intentionally keep only the two
    # informative V-Spline families from the latest 4BAAD9 runs: endpoint-
    # guarded quintic aviation splines and the best cubic overlap comparator.
    # Legacy piecewise, join-smoothed, global, and Hermite diagnostics are still
    # selectable through v_spline_output_backends.
    v_spline_output_backends: tuple[str, ...] = (
        "quintic_bspline",
        "bspline_overlap",
    )
    use_kalman_boundary_prior: bool = False
    event_aware_evaluation_enabled: bool = True
    holdout_evaluation_fraction: float = 0.15

    # Synthetic gap-holdout benchmark.  These extra methods are diagnostic and
    # are excluded from the normal evaluator leaderboard; the evaluator ranks
    # them separately by error at the deliberately deleted raw ADS-B samples.
    synthetic_gap_holdout_enabled: bool = True
    synthetic_gap_holdout_methods: tuple[str, ...] = (
        "kalman_rts_balanced",
        "aviation_v_spline_quintic_balanced",
        "v_spline_bspline_overlap_smooth",
    )
    synthetic_gap_holdout_gap_count: int = 4
    synthetic_gap_holdout_fraction: float = 0.05
    synthetic_gap_holdout_min_gap_s: float = 8.0
    synthetic_gap_holdout_max_gap_s: float = 16.0
    synthetic_gap_holdout_guard_s: float = 45.0
    synthetic_gap_holdout_seed: int = 17
    v_spline_output_presets: tuple[Literal["balanced", "accurate", "smooth"], ...] = (
        "balanced",
        "accurate",
        "smooth",
    )
    bspline_config: BSplineCoreConfig = field(
        default_factory=lambda: BSplineCoreConfig(
            degree=3,
            knot_spacing_s=5.0,
            position_weight=1.0,
            velocity_weight=0.03,
            penalty_mode="adaptive",
            adaptive_eta=1e5,
            adaptive_speed_floor_mps=1.0,
            velocity_outlier_gate_mps=50.0,
            hard_component_endpoint_positions=True,
            hard_component_endpoint_velocities=False,
            component_endpoint_velocity_prior_weight=0.0,
            component_endpoint_acceleration_prior_weight=0.0,
            boundary_velocity_prior_weight=0.5,
            jerk_penalty_weight=0.0,
            robust_position_loss="huber",
            robust_iterations=2,
            solver_ridge=1e-7,
            condition_number_max_basis=256,
        )
    )
    hermite_config: HermiteCoreConfig = field(default_factory=_default_hermite_config)
    bspline_config_by_preset: dict[str, BSplineCoreConfig] = field(default_factory=dict)
    hermite_config_by_preset: dict[str, HermiteCoreConfig] = field(default_factory=dict)
    local_segment_policy_config_by_preset: dict[str, LocalSegmentPolicyConfig] = field(default_factory=dict)
    local_segment_tuning_config_by_preset: dict[str, LocalSegmentTuningConfig] = field(default_factory=dict)

    adapter_config: RawKeyframeVSplineAdapterConfig = field(
        default_factory=lambda: RawKeyframeVSplineAdapterConfig(
            max_gap_s=None,
            min_segment_observations=2,
            fail_on_short_segment=True,
            duplicate_time_tolerance_s=0.0,
        )
    )


@dataclass
class BuiltFlight:
    """Paths and metadata for one generated flight."""

    manifest_entry: dict[str, Any]
    flight_json: dict[str, Any]
    raw_adsb_path: Path
    # Backwards-compatible aliases point at the B-spline output when present.
    v_spline_path: Path | None
    raw_adsb_minimal_path: Path
    v_spline_minimal_path: Path | None
    v_spline_paths: dict[str, Path] = field(default_factory=dict)
    v_spline_minimal_paths: dict[str, Path] = field(default_factory=dict)
    reconstruction_paths: dict[str, Path] = field(default_factory=dict)
    reconstruction_minimal_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class FlightDebugContext:
    """Per-flight debug artifact writer.

    The normal viewer JSON remains compact enough for interactive use.  This
    context writes academic/reproducibility artifacts under
    ``flights/<flight_id>/debug``: step timings, rule/config snapshots,
    segmentation, boundary states, quality tables, and a flight-local log.
    """

    flight_id: str
    icao: str
    debug_dir: Path
    enabled: bool = True
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def step(self, name: str, **details: Any) -> "FlightPipelineStep":
        return FlightPipelineStep(self, name, details)

    def write_json(self, filename: str, payload: Any) -> Path | None:
        if not self.enabled:
            return None
        path = self.debug_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_packet(_clean_json(payload), path)
        self.artifacts.append({"type": "json", "file": str(path.name), "path": str(path)})
        return path

    def write_csv(self, filename: str, rows: list[dict[str, Any]]) -> Path | None:
        if not self.enabled:
            return None
        path = self.debug_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_scalar(row.get(key)) for key in keys})
        self.artifacts.append({"type": "csv", "file": str(path.name), "path": str(path), "row_count": len(rows)})
        return path

    def flush_manifest(self) -> None:
        self.write_json(
            "debug_manifest.json",
            {
                "flight_id": self.flight_id,
                "icao": self.icao,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "steps": self.steps,
                "artifacts": self.artifacts,
            },
        )


class FlightPipelineStep:
    """Context manager that records one durable pipeline step."""

    def __init__(self, debug: FlightDebugContext, name: str, details: dict[str, Any]) -> None:
        self.debug = debug
        self.name = name
        self.details = details
        self.record: dict[str, Any] = {}
        self._t0 = 0.0

    def __enter__(self) -> dict[str, Any]:
        self._t0 = time.perf_counter()
        self.record = {
            "name": self.name,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "details": _clean_json(self.details),
        }
        if self.debug.enabled:
            self.debug.steps.append(self.record)
        logger.info("Flight {} step started: {}", self.debug.flight_id, self.name)
        return self.record

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.record["duration_s"] = float(time.perf_counter() - self._t0)
        self.record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if exc is None:
            self.record["status"] = "ok"
            logger.info(
                "Flight {} step completed: {} ({:.3f}s)",
                self.debug.flight_id,
                self.name,
                self.record["duration_s"],
            )
        else:
            self.record["status"] = "failed"
            self.record["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Flight {} step failed: {}", self.debug.flight_id, self.name)
            self.debug.flush_manifest()
        return False


class TrackOutputPipeline:
    """Build the complete track_output folder from SQLite rows and track rules."""

    def __init__(self, config: TrackOutputPipelineConfig | None = None) -> None:
        self.config = config or TrackOutputPipelineConfig()
        self._configure_file_logging()

    def _configure_file_logging(self) -> None:
        """Attach one run-level log file sink when loguru is available."""
        log_dir = self.config.log_dir
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(logger, "add"):
                if hasattr(logger, "configure"):
                    logger.configure(extra={"flight_id": "-"})
                run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                logger.add(
                    log_dir / f"track_output_pipeline_{run_ts}.log",
                    level="DEBUG",
                    rotation="25 MB",
                    retention=10,
                    enqueue=False,
                    backtrace=False,
                    diagnose=False,
                    format=(
                        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                        "flight={extra[flight_id]} | {name}:{function}:{line} - {message}"
                    ),
                )
        except Exception:
            pass

    def _select_rules(self, registry: Any) -> tuple[list[TrackRuleConfig], list[dict[str, Any]]]:
        """Select rules and require every requested ICAO to be in flight_rules.json.

        The pipeline no longer creates inferred broad rules.  A curated
        ``flight_rules.json`` entry is mandatory because time windows, field
        elevation, CRC policy, and event filters are part of the experiment
        definition.  This keeps academic results reproducible.
        """
        requested = self.config.icao_list
        if requested is None:
            rules = list(registry.rules)
            if not rules:
                raise ValueError("flight_rules.json contains no rules")
            return rules, []

        selected: list[TrackRuleConfig] = []
        warnings: list[dict[str, Any]] = []
        missing: list[str] = []
        seen: set[str] = set()

        for item in requested:
            key = str(item).strip().upper()
            if not key:
                continue
            if key in seen:
                warnings.append({"track_id": key, "icao": key, "warning": "duplicate requested ICAO ignored"})
                continue
            seen.add(key)
            try:
                selected.append(registry.get(key))
            except KeyError:
                missing.append(key)

        if missing:
            source = getattr(registry, "source_path", None) or "flight_rules.json"
            raise KeyError(
                "Missing required flight_rules.json entries for ICAO/track id(s): "
                f"{', '.join(missing)}. Rule file: {source}"
            )
        if not selected:
            raise ValueError("No non-empty ICAO/track ids were requested")
        return selected, warnings

    def run(self) -> dict[str, Any]:
        cfg = self.config
        out_dir = cfg.paths.output_dir

        if cfg.clean_output_dir and out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "flights").mkdir(parents=True, exist_ok=True)

        registry = load_track_rule_registry(cfg.paths.rules_path)
        selected_rules, selection_warnings = self._select_rules(registry)
        loader = self._make_raw_loader()

        built: list[BuiltFlight] = []
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = list(selection_warnings)

        for rule in _progress_iter(
            selected_rules,
            enabled=cfg.show_progress,
            desc="Flights",
            unit="flight",
            leave=True,
        ):
            try:
                built.append(self._build_one_flight(rule, loader))
            except Exception as exc:
                errors.append(
                    {
                        "track_id": rule.track_id,
                        "icao": rule.icao,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.exception("Failed to build flight {}", rule.track_id)

        flights_json = {
            "schemaVersion": MANIFEST_SCHEMA_VERSION,
            "defaultMethod": DEFAULT_METHOD_ID,
            "flights": [b.manifest_entry for b in built],
        }
        write_json_packet(flights_json, out_dir / "flights.json")

        summary = {
            "output_dir": str(out_dir),
            "requested_icao_count": len(cfg.icao_list) if cfg.icao_list is not None else None,
            "selected_rule_count": len(selected_rules),
            "flight_count": len(built),
            "warning_count": len(warnings),
            "warnings": warnings,
            "error_count": len(errors),
            "errors": errors,
            "flights_json": str(out_dir / "flights.json"),
        }
        logger.info("Track-output pipeline completed: {}", summary)
        return summary

    def _make_raw_loader(self) -> SqlAdsbLoader:
        cfg = self.config
        db_path = Path(cfg.paths.database_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"SQLite ADS-B database not found: {db_path}. "
                "Set ADSB_SQLITE_PATH or PipelinePaths.database_path."
            )
        return SqlAdsbLoader(
            db_path,
            SqlAdsbLoadConfig(
                include_crc_false=True,
                derive_vertical_rate_columns=True,
            ),
        )

    def _method_config_for_spec(self, spec: VSplineOutputSpec) -> TrackOutputPipelineConfig:
        """Return a config view with preset- and variant-specific V-Spline settings."""
        cfg = self.config
        preset = str(spec.preset)
        backend = str(spec.backend)
        bspline_config = cfg.bspline_config_by_preset.get(preset) or _default_bspline_config_for_preset(preset)
        hermite_config = cfg.hermite_config_by_preset.get(preset) or _default_hermite_config_for_preset(preset)
        policy_config = cfg.local_segment_policy_config_by_preset.get(preset) or _default_policy_config_for_preset(preset)
        tuning_config = cfg.local_segment_tuning_config_by_preset.get(preset) or _default_tuning_config_for_preset(preset)
        boundary_config = cfg.boundary_state_config
        use_kalman_boundary_prior = bool(cfg.use_kalman_boundary_prior)
        if backend == "bspline_piecewise":
            bspline_config = replace(bspline_config, backend_name=f"v_spline_bspline_{preset}")
        elif backend == "hermite_piecewise":
            bspline_config = replace(bspline_config, backend_name=f"v_spline_hermite_boundary_policy_{preset}")

        # Aviation-adapted variants avoid single-sample hard raw boundary anchoring.
        # Flight 4BAAD9 showed that purely soft position priors still allow 10-25 m
        # C0 mismatches at ordinary joins.  These variants therefore use a robust
        # shared boundary *position* as the hard C0 anchor while keeping velocity and
        # acceleration as de-trusted soft/harmonized evidence.
        if backend in SOFT_BOUNDARY_BACKENDS:
            boundary_config = replace(
                boundary_config,
                position_source="weighted_compromise",
                # Report action: aviation variants now lean materially farther
                # toward robust boundary states; this is the 0.35/0.65 sweep
                # point recommended before trying robust-only joins.
                position_raw_weight=0.35,
                position_robust_weight=0.65,
                blend_reported_velocity_weight=0.0,
            )

        if backend == "bspline_overlap":
            bspline_config = replace(
                bspline_config,
                hard_boundary_positions=True,
                boundary_position_prior_weight=0.0,
                boundary_velocity_prior_weight=0.15,
                backend_name=f"v_spline_bspline_overlap_{preset}",
            )
        elif backend == "bspline_join_smooth":
            acc_w = {"accurate": 0.22, "balanced": 0.45, "smooth": 0.90}[preset]
            jerk_w = {"accurate": 0.015, "balanced": 0.035, "smooth": 0.080}[preset]
            bspline_config = replace(
                bspline_config,
                hard_boundary_positions=True,
                boundary_position_prior_weight=0.0,
                boundary_acceleration_prior_weight=acc_w,
                jerk_penalty_weight=jerk_w,
                backend_name=f"v_spline_bspline_join_smooth_{preset}",
            )
        elif backend == "hermite_stable":
            hermite_config = replace(
                hermite_config,
                velocity_weight=max(float(hermite_config.velocity_weight) * 0.25, 1e-12),
                adaptive_speed_floor_mps={"accurate": 25.0, "balanced": 35.0, "smooth": 45.0}[preset],
                hard_endpoint_velocities=False,
                optimize=False,
            )
            bspline_config = replace(
                bspline_config,
                hard_boundary_positions=True,
                boundary_position_prior_weight=0.0,
                backend_name=f"v_spline_hermite_stable_boundary_policy_{preset}",
            )
        elif backend == "bspline_component_global":
            # Segmentation ablation / join-risk guardrail: fit each hard-gap
            # component as one cubic V-Spline so dynamic-regime joins cannot
            # introduce endpoint artifacts.  Keep local tuning, but disable
            # quality-triggered re-segmentation so this backend remains a true
            # one-component spline comparator.
            bspline_config = replace(
                bspline_config,
                degree=3,
                hard_boundary_positions=True,
                hard_component_endpoint_positions=False,
                boundary_position_prior_weight=0.0,
                boundary_velocity_prior_weight=0.0,
                boundary_acceleration_prior_weight=0.0,
                jerk_penalty_weight=max(float(bspline_config.jerk_penalty_weight), {"accurate": 0.001, "balanced": 0.002, "smooth": 0.006}[preset]),
                backend_name=f"aviation_v_spline_bspline_global_{preset}",
            )
            tuning_config = replace(
                tuning_config,
                enabled=False,
                max_candidates=1,
                join_velocity_harmonization=False,
                adaptive_resegmentation_enabled=False,
            )
        elif backend in {"quintic_bspline", "quintic_kalman_boundary"}:
            # 4BAAD9 endpoint-artifact fix: quintic accurate/balanced were
            # internally well joined but developed a large true component-end
            # jerk artifact.  Use position-only component endpoints, lower stale
            # ADS-B velocity trust, lighter C2 pressure, and an endpoint-only
            # jerk/snap guard.  Smooth keeps its stronger derivative profile.
            endpoint_guarded = preset in {"accurate", "balanced"}
            acc_w = {"accurate": 0.18, "balanced": 0.20, "smooth": 1.05}[preset]
            jerk_w = {"accurate": 0.020, "balanced": 0.025, "smooth": 0.110}[preset]
            snap_w = {"accurate": 0.0005, "balanced": 0.001, "smooth": 0.008}[preset]
            velocity_w = {"accurate": 0.020, "balanced": 0.015, "smooth": float(bspline_config.velocity_weight)}[preset]
            min_obs_floor = {"accurate": 5.0, "balanced": 5.0, "smooth": 9.0}[preset]
            bspline_config = replace(
                bspline_config,
                degree=5,
                position_weight=(1.6 if endpoint_guarded else float(bspline_config.position_weight)),
                velocity_weight=velocity_w,
                velocity_outlier_gate_mps=50.0,
                adaptive_speed_floor_mps=(20.0 if endpoint_guarded else bspline_config.adaptive_speed_floor_mps),
                hard_boundary_positions=True,
                hard_component_endpoint_positions=True,
                hard_component_endpoint_velocities=False,
                component_endpoint_velocity_prior_weight=0.0,
                component_endpoint_acceleration_prior_weight=0.0,
                boundary_position_prior_weight=0.0,
                boundary_velocity_prior_weight=0.05,
                boundary_acceleration_prior_weight=acc_w,
                jerk_penalty_weight=jerk_w,
                snap_penalty_weight=snap_w,
                endpoint_guard_window_s=(8.0 if endpoint_guarded else 0.0),
                endpoint_jerk_penalty_multiplier=(4.0 if endpoint_guarded else 1.0),
                endpoint_snap_penalty_multiplier=(4.0 if endpoint_guarded else 1.0),
                min_observations_per_basis=max(float(bspline_config.min_observations_per_basis), min_obs_floor),
                backend_name=(
                    f"aviation_v_spline_quintic_kalman_boundary_{preset}"
                    if backend == "quintic_kalman_boundary"
                    else f"aviation_v_spline_quintic_{preset}"
                ),
            )
            if endpoint_guarded:
                tuning_config = replace(
                    tuning_config,
                    max_velocity_weight=min(float(tuning_config.max_velocity_weight), 0.035),
                    min_observations_per_basis=max(float(tuning_config.min_observations_per_basis), min_obs_floor),
                    min_jerk_penalty_weight=max(float(tuning_config.min_jerk_penalty_weight), jerk_w),
                    min_acceleration_prior_weight=max(float(tuning_config.min_acceleration_prior_weight), acc_w),
                    harmonized_reported_velocity_weight=0.0,
                    prefit_boundary_velocity_prior_weight=min(float(tuning_config.prefit_boundary_velocity_prior_weight), 0.01),
                    join_artifact_cost_scale_m=max(float(tuning_config.join_artifact_cost_scale_m), 10.0),
                )
            if backend == "quintic_kalman_boundary":
                use_kalman_boundary_prior = True

        return replace(
            cfg,
            bspline_config=bspline_config,
            hermite_config=hermite_config,
            boundary_state_config=boundary_config,
            local_segment_policy_config=policy_config,
            local_segment_tuning_config=tuning_config,
            use_kalman_boundary_prior=use_kalman_boundary_prior,
        )

    def _method_config_for_kalman_spec(self, spec: KalmanRTSOutputSpec) -> TrackOutputPipelineConfig:
        """Return a config view with preset-specific Kalman/RTS settings."""
        cfg = self.config
        preset = str(spec.preset)
        kalman_config = cfg.kalman_rts_config_by_preset.get(preset) or default_kalman_rts_config_for_preset(preset)
        return replace(cfg, kalman_rts_config=kalman_config)


    def _build_one_flight(self, rule: TrackRuleConfig, loader: SqlAdsbLoader) -> BuiltFlight:
        """Execute the production pipeline for one curated flight rule."""
        cfg = self.config
        flight_id = rule.track_id
        icao = rule.icao.upper()

        flight_dir = cfg.paths.output_dir / "flights" / flight_id
        methods_dir = flight_dir / "methods"
        minimal_methods_dir = methods_dir / MINIMAL_METHODS_DIRNAME
        debug_dir = flight_dir / "debug"
        methods_dir.mkdir(parents=True, exist_ok=True)
        minimal_methods_dir.mkdir(parents=True, exist_ok=True)

        debug = FlightDebugContext(
            flight_id=flight_id,
            icao=icao,
            debug_dir=debug_dir,
            enabled=bool(cfg.write_debug_artifacts),
        )
        debug.write_json("flight_rule.json", rule.to_dict())
        debug.write_json("config_snapshot.json", self._debug_config_snapshot(rule))

        flight_log_sink_id: int | None = None
        try:
            if debug.enabled and hasattr(logger, "add"):
                flight_log_sink_id = logger.add(
                    debug_dir / "flight.log",
                    level="DEBUG",
                    filter=lambda record, fid=flight_id: record.get("extra", {}).get("flight_id") == fid,
                    enqueue=False,
                    backtrace=False,
                    diagnose=False,
                    format=(
                        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                        "{name}:{function}:{line} - {message}"
                    ),
                )
        except Exception:
            flight_log_sink_id = None

        context = logger.contextualize(flight_id=flight_id) if hasattr(logger, "contextualize") else nullcontext()
        try:
            with context:
                logger.info("Building flight {} ({})", flight_id, icao)

                with debug.step("01_load_sql_rows", database_path=str(cfg.paths.database_path)) as step:
                    load_result = loader.load_rule(
                        rule,
                        apply_rule_time_window=False,
                        apply_rule_crc=False,
                    )
                    step["output"] = {
                        "row_count": int(len(load_result.dataframe)),
                        "loader_report": load_result.report,
                    }
                debug.write_json("raw_loader_report.json", load_result.report)

                with debug.step("02_normalize_raw_adsb", input_rows=int(len(load_result.dataframe))) as step:
                    normalizer_cfg = RawAdsbNormalizeConfig(
                        apply_rule_filters=True,
                        keep_unpaired_timestamps=True,
                        parse_vertical_rate_from_decoded_json=True,
                        keyframe_time_quantization_s=rule.keyframe_time_quantization_s
                        or cfg.raw_keyframe_time_quantization_s,
                        baro_z_reference="field",
                        derive_velocity_delta_acceleration=True,
                    )
                    normalized = RawAdsbNormalizer(normalizer_cfg).normalize(load_result.dataframe, rule=rule)
                    step["output"] = {
                        "keyframe_count": len(normalized.keyframes),
                        "event_count": int(len(normalized.events)),
                        "normalizer_report": normalized.report,
                    }
                debug.write_json(
                    "normalization_report.json",
                    {
                        "normalizer_config": asdict(normalizer_cfg),
                        "normalizer_report": normalized.report,
                        "event_count": int(len(normalized.events)),
                        "keyframe_count": len(normalized.keyframes),
                    },
                )

                with debug.step("03_choose_origin_and_project_to_local_frame", keyframe_count=len(normalized.keyframes)) as step:
                    origin = self._choose_origin(normalized.keyframes, rule)
                    raw_keyframes = self._add_local_xy_to_keyframes(normalized.keyframes, origin, rule)
                    step["output"] = {"origin": origin, "local_keyframe_count": len(raw_keyframes)}

                with debug.step("04_write_raw_payload", keyframe_count=len(raw_keyframes)):
                    raw_payload = self._build_raw_adsb_payload(
                        rule=rule,
                        origin=origin,
                        keyframes=raw_keyframes,
                        normalized_report=normalized.report,
                        loader_report=load_result.report,
                        events=normalized.events,
                    )
                    raw_path = methods_dir / "raw_adsb.json"
                    write_json_packet(raw_payload, raw_path)

                    raw_minimal_path = minimal_methods_dir / "raw_adsb.json"
                    write_json_packet(
                        self._build_minimal_method_payload(
                            raw_payload,
                            detailed_file=f"flights/{flight_id}/methods/raw_adsb.json",
                        ),
                        raw_minimal_path,
                    )

                with debug.step("05_prepare_paired_vspline_samples", raw_keyframe_count=len(raw_keyframes)) as step:
                    adapter = RawKeyframeVSplineAdapter(raw_path, cfg.adapter_config)
                    adapter.payload = raw_payload
                    prepared = adapter.prepare()
                    step["output"] = {
                        "paired_sample_count": len(prepared.samples),
                        "prepared_segment_count": len(prepared.segments),
                        "adapter_diagnostics": prepared.diagnostics,
                    }
                debug.write_json(
                    "prepared_samples_report.json",
                    {
                        "paired_sample_count": len(prepared.samples),
                        "prepared_segment_count": len(prepared.segments),
                        "adapter_diagnostics": prepared.diagnostics,
                    },
                )

                with debug.step("06_dynamic_energy_segmentation", paired_sample_count=len(prepared.samples)) as step:
                    segmented_components, segmentation_diagnostics = segment_prepared_samples(
                        prepared.samples,
                        cfg.dynamic_segmentation_config,
                    )
                    step["output"] = {
                        "component_count": len(segmented_components),
                        "segment_count": int(sum(len(c.segments) for c in segmented_components)),
                        "boundary_count": int(sum(len(c.boundaries) for c in segmented_components)),
                    }
                debug.write_json(
                    "segmentation.json",
                    self._debug_segmentation_payload(segmented_components, segmentation_diagnostics),
                )

                component_contexts: list[tuple[SegmentedComponent, dict[str, Any]]] = []
                with debug.step("07_estimate_shared_boundary_states", component_count=len(segmented_components)) as step:
                    for segmented_component in _progress_iter(
                        segmented_components,
                        enabled=cfg.show_progress,
                        desc=f"Boundary states {flight_id}",
                        unit="component",
                        leave=False,
                    ):
                        shared_states = estimate_shared_boundary_states(
                            prepared.samples,
                            segmented_component.boundaries,
                            cfg.boundary_state_config,
                        )
                        component_contexts.append((segmented_component, shared_states))
                    step["output"] = {
                        "component_count": len(component_contexts),
                        "boundary_state_count": int(sum(len(states) for _, states in component_contexts)),
                    }
                debug.write_json("boundary_states.json", self._debug_boundary_state_payload(component_contexts))

                v_spline_paths: dict[str, Path] = {}
                v_spline_minimal_paths: dict[str, Path] = {}
                v_spline_payloads: dict[str, dict[str, Any]] = {}
                reconstruction_paths: dict[str, Path] = {}
                reconstruction_minimal_paths: dict[str, Path] = {}
                reconstruction_payloads: dict[str, dict[str, Any]] = {}

                kalman_specs = _kalman_rts_output_specs(cfg)
                with debug.step("08_fit_kalman_rts_methods", method_count=len(kalman_specs)) as step:
                    for spec in _progress_iter(
                        kalman_specs,
                        enabled=cfg.show_progress,
                        desc=f"Kalman-RTS methods {flight_id}",
                        unit="method",
                        leave=False,
                    ):
                        method_cfg = self._method_config_for_kalman_spec(spec)
                        component_result = self._fit_kalman_rts_components(
                            flight_id=flight_id,
                            prepared_samples=prepared.samples,
                            method_config=method_cfg,
                            method_id=spec.method_id,
                        )
                        k_payload = self._build_kalman_rts_payload(
                            rule=rule,
                            origin=origin,
                            raw_keyframes=raw_keyframes,
                            prepared=prepared,
                            fits=component_result["fits"],
                            piecewise_report=component_result["piecewise_report"],
                            segment_metadata=component_result["segment_metadata"],
                            method_id=spec.method_id,
                            method_label=spec.label,
                            preset=spec.preset,
                            method_config=method_cfg,
                        )
                        reconstruction_payloads[spec.method_id] = k_payload

                        k_path = methods_dir / f"{spec.file_stem}.json"
                        write_json_packet(k_payload, k_path)
                        reconstruction_paths[spec.method_id] = k_path

                        k_minimal_path = minimal_methods_dir / f"{spec.file_stem}.json"
                        write_json_packet(
                            self._build_minimal_method_payload(
                                k_payload,
                                detailed_file=f"flights/{flight_id}/methods/{spec.file_stem}.json",
                            ),
                            k_minimal_path,
                        )
                        reconstruction_minimal_paths[spec.method_id] = k_minimal_path

                    step["output"] = {
                        "method_count": len(kalman_specs),
                        "methods": [spec.method_id for spec in kalman_specs],
                        "segmentation_applied": False,
                    }

                output_specs = _v_spline_output_specs(cfg)
                synthetic_gap_specs = _synthetic_gap_output_specs(cfg, kalman_specs, output_specs)
                with debug.step("09_fit_v_spline_methods", method_count=len(output_specs)) as step:
                    for spec in _progress_iter(
                        output_specs,
                        enabled=cfg.show_progress,
                        desc=f"V-Spline methods {flight_id}",
                        unit="method",
                        leave=False,
                    ):
                        method_cfg = self._method_config_for_spec(spec)
                        fits: list[tuple[str, Any, Any]] = []
                        piecewise_components: list[dict[str, Any]] = []
                        segment_metadata: dict[str, dict[str, Any]] = {}

                        method_component_contexts = component_contexts
                        if spec.backend in GLOBAL_BSPLINE_BACKENDS:
                            method_component_contexts = [
                                (
                                    build_segmented_component_from_boundaries(
                                        segmented_component.component,
                                        (),
                                        method_cfg.dynamic_segmentation_config,
                                        diagnostics_extra={
                                            "boundary_source": "component_global_backend_no_dynamic_boundaries",
                                            "accepted_boundary_count": 0,
                                            "segment_count": 1,
                                            "original_dynamic_segment_count": len(segmented_component.segments),
                                            "original_dynamic_boundary_count": len(segmented_component.boundaries),
                                            "original_dynamic_boundary_ids": [b.boundary_id for b in segmented_component.boundaries],
                                            "note": "dynamic segmentation is still reported for diagnosis, but this backend fits each hard-gap component as one V-Spline segment",
                                        },
                                    ),
                                    {},
                                )
                                for segmented_component, _shared_states in component_contexts
                            ]

                        for segmented_component, shared_states in _progress_iter(
                            method_component_contexts,
                            enabled=cfg.show_progress,
                            desc=f"{spec.method_id} components {flight_id}",
                            unit="component",
                            leave=False,
                        ):
                            component_result = self._fit_component_with_local_b_spline(
                                flight_id=flight_id,
                                prepared_samples=prepared.samples,
                                segmented_component=segmented_component,
                                shared_states=shared_states,
                                method_config=method_cfg,
                                backend=spec.backend,
                            )
                            fits.extend(component_result["fits"])
                            for sid, meta in component_result["segment_metadata"].items():
                                segment_metadata[sid] = meta
                            piecewise_components.append(component_result["piecewise_component"])

                        global_continuity = _piecewise_global_continuity(piecewise_components, fits)
                        if spec.backend in GLOBAL_BSPLINE_BACKENDS:
                            segmentation_role = (
                                "hard_gap_components_are_fit_as_one_global_cubic_b_spline_vspline_segment; "
                                "dynamic_regime_segmentation_is_reported_as_a_diagnostic_ablation_not_used_as_fit_boundaries"
                            )
                            join_note = "no dynamic-regime internal joins are fitted, so segmentation cannot create C0/C1 join artifacts inside a hard-gap component"
                            segmentation_enabled_for_method = False
                        elif spec.backend in HERMITE_BACKENDS:
                            segmentation_role = "dynamic_energy_state_segments_are_independent_local_hermite_vspline_fits_with_per_segment_quality_tuning_and_quality_triggered_resegmentation"
                            join_note = "robust shared boundary positions/velocities are used for stable Hermite endpoints; velocity observations are confidence-weighted; acceleration priors are diagnostics only for the paper Hermite core"
                            segmentation_enabled_for_method = bool(cfg.dynamic_segmentation_config.enabled)
                        else:
                            segmentation_role = "dynamic_energy_state_segments_are_independent_local_b_spline_vspline_fits_with_per_segment_quality_tuning_and_quality_triggered_resegmentation"
                            join_note = "ordinary joins use robust hard C0 boundary positions, harmonized/de-trusted velocity evidence, and soft acceleration/jerk priors where configured"
                            segmentation_enabled_for_method = bool(cfg.dynamic_segmentation_config.enabled)

                        piecewise_report = {
                            "enabled": segmentation_enabled_for_method,
                            "mode": spec.backend,
                            "method_id": spec.method_id,
                            "segmentation_role": segmentation_role,
                            "join_continuity_note": join_note,
                            "segmentation_diagnostics": segmentation_diagnostics,
                            "connected_components": piecewise_components,
                            "global_quality": global_continuity,
                        }

                        v_payload = self._build_v_spline_payload(
                            rule=rule,
                            origin=origin,
                            raw_keyframes=raw_keyframes,
                            prepared=prepared,
                            fits=fits,
                            piecewise_report=piecewise_report,
                            segment_metadata=segment_metadata,
                            method_id=spec.method_id,
                            method_label=spec.label,
                            backend=spec.backend,
                            preset=spec.preset,
                            method_config=method_cfg,
                        )
                        v_spline_payloads[spec.method_id] = v_payload
                        reconstruction_payloads[spec.method_id] = v_payload

                        v_path = methods_dir / f"{spec.file_stem}.json"
                        write_json_packet(v_payload, v_path)
                        v_spline_paths[spec.method_id] = v_path
                        reconstruction_paths[spec.method_id] = v_path

                        v_minimal_path = minimal_methods_dir / f"{spec.file_stem}.json"
                        write_json_packet(
                            self._build_minimal_method_payload(
                                v_payload,
                                detailed_file=f"flights/{flight_id}/methods/{spec.file_stem}.json",
                            ),
                            v_minimal_path,
                        )
                        v_spline_minimal_paths[spec.method_id] = v_minimal_path
                        reconstruction_minimal_paths[spec.method_id] = v_minimal_path

                    step["output"] = {
                        "method_count": len(v_spline_payloads),
                        "methods": list(v_spline_payloads.keys()),
                    }

                with debug.step("10_fit_synthetic_gap_holdout_methods", method_count=len(synthetic_gap_specs)) as step:
                    gap_plan = self._build_synthetic_gap_holdout_plan(prepared.samples)
                    gap_plan_debug = {k: v for k, v in gap_plan.items() if k != "training_samples"}
                    debug.write_json("synthetic_gap_holdout.json", gap_plan_debug)
                    gap_payloads: dict[str, dict[str, Any]] = {}
                    if synthetic_gap_specs and bool(gap_plan.get("enabled")):
                        gap_prepared = self._prepared_with_synthetic_gap_training_samples(prepared, gap_plan)
                        gap_raw_keyframes = self._filter_raw_keyframes_for_synthetic_gap(raw_keyframes, gap_plan)
                        gap_segmented_components, gap_segmentation_diagnostics = segment_prepared_samples(
                            list(gap_plan.get("training_samples") or []),
                            cfg.dynamic_segmentation_config,
                        )
                        gap_component_contexts: list[tuple[SegmentedComponent, dict[str, Any]]] = []
                        for segmented_component in gap_segmented_components:
                            shared_states = estimate_shared_boundary_states(
                                list(gap_plan.get("training_samples") or []),
                                segmented_component.boundaries,
                                cfg.boundary_state_config,
                            )
                            gap_component_contexts.append((segmented_component, shared_states))

                        for gap_spec in _progress_iter(
                            synthetic_gap_specs,
                            enabled=cfg.show_progress,
                            desc=f"Synthetic gap methods {flight_id}",
                            unit="method",
                            leave=False,
                        ):
                            if gap_spec.base_family == "kalman_rts":
                                base_spec = gap_spec.base_spec
                                assert isinstance(base_spec, KalmanRTSOutputSpec)
                                method_cfg = self._method_config_for_kalman_spec(base_spec)
                                component_result = self._fit_kalman_rts_components(
                                    flight_id=flight_id,
                                    prepared_samples=list(gap_plan.get("training_samples") or []),
                                    method_config=method_cfg,
                                    method_id=gap_spec.method_id,
                                )
                                payload = self._build_kalman_rts_payload(
                                    rule=rule,
                                    origin=origin,
                                    raw_keyframes=gap_raw_keyframes,
                                    prepared=gap_prepared,
                                    fits=component_result["fits"],
                                    piecewise_report=component_result["piecewise_report"],
                                    segment_metadata=component_result["segment_metadata"],
                                    method_id=gap_spec.method_id,
                                    method_label=gap_spec.label,
                                    preset=base_spec.preset,
                                    method_config=method_cfg,
                                )
                            else:
                                base_spec = gap_spec.base_spec
                                assert isinstance(base_spec, VSplineOutputSpec)
                                method_cfg = self._method_config_for_spec(base_spec)
                                fits: list[tuple[str, Any, Any]] = []
                                piecewise_components: list[dict[str, Any]] = []
                                segment_metadata: dict[str, dict[str, Any]] = {}
                                method_component_contexts = gap_component_contexts
                                if base_spec.backend in GLOBAL_BSPLINE_BACKENDS:
                                    method_component_contexts = [
                                        (
                                            build_segmented_component_from_boundaries(
                                                segmented_component.component,
                                                (),
                                                method_cfg.dynamic_segmentation_config,
                                                diagnostics_extra={
                                                    "boundary_source": "synthetic_gap_component_global_backend_no_dynamic_boundaries",
                                                    "accepted_boundary_count": 0,
                                                    "segment_count": 1,
                                                    "original_dynamic_segment_count": len(segmented_component.segments),
                                                    "original_dynamic_boundary_count": len(segmented_component.boundaries),
                                                    "original_dynamic_boundary_ids": [b.boundary_id for b in segmented_component.boundaries],
                                                    "note": "synthetic-gap diagnostic fit; dynamic segmentation is reported but not used by component-global backend",
                                                },
                                            ),
                                            {},
                                        )
                                        for segmented_component, _shared_states in gap_component_contexts
                                    ]

                                for segmented_component, shared_states in method_component_contexts:
                                    component_result = self._fit_component_with_local_b_spline(
                                        flight_id=flight_id,
                                        prepared_samples=list(gap_plan.get("training_samples") or []),
                                        segmented_component=segmented_component,
                                        shared_states=shared_states,
                                        method_config=method_cfg,
                                        backend=base_spec.backend,
                                    )
                                    fits.extend(component_result["fits"])
                                    for sid, meta in component_result["segment_metadata"].items():
                                        segment_metadata[sid] = meta
                                    piecewise_components.append(component_result["piecewise_component"])

                                global_continuity = _piecewise_global_continuity(piecewise_components, fits)
                                piecewise_report = {
                                    "enabled": bool(method_cfg.dynamic_segmentation_config.enabled) and base_spec.backend not in GLOBAL_BSPLINE_BACKENDS,
                                    "mode": base_spec.backend,
                                    "method_id": gap_spec.method_id,
                                    "base_method_id": gap_spec.base_method_id,
                                    "diagnostic_role": "synthetic_gap_holdout_reconstruction",
                                    "segmentation_role": "synthetic-gap diagnostic refit from training samples with contiguous raw windows deleted",
                                    "join_continuity_note": "same backend join policy as the base method; score deleted-gap interpolation in evaluate_reconstructions.py",
                                    "segmentation_diagnostics": gap_segmentation_diagnostics,
                                    "connected_components": piecewise_components,
                                    "global_quality": global_continuity,
                                }
                                payload = self._build_v_spline_payload(
                                    rule=rule,
                                    origin=origin,
                                    raw_keyframes=gap_raw_keyframes,
                                    prepared=gap_prepared,
                                    fits=fits,
                                    piecewise_report=piecewise_report,
                                    segment_metadata=segment_metadata,
                                    method_id=gap_spec.method_id,
                                    method_label=gap_spec.label,
                                    backend=base_spec.backend,
                                    preset=base_spec.preset,
                                    method_config=method_cfg,
                                )

                            payload = self._attach_synthetic_gap_metadata(
                                payload,
                                plan=gap_plan,
                                base_method_id=gap_spec.base_method_id,
                            )
                            gap_payloads[gap_spec.method_id] = payload
                            reconstruction_payloads[gap_spec.method_id] = payload

                            gap_path = methods_dir / f"{gap_spec.file_stem}.json"
                            write_json_packet(payload, gap_path)
                            reconstruction_paths[gap_spec.method_id] = gap_path
                            gap_minimal_path = minimal_methods_dir / f"{gap_spec.file_stem}.json"
                            write_json_packet(
                                self._build_minimal_method_payload(
                                    payload,
                                    detailed_file=f"flights/{flight_id}/methods/{gap_spec.file_stem}.json",
                                ),
                                gap_minimal_path,
                            )
                            reconstruction_minimal_paths[gap_spec.method_id] = gap_minimal_path

                    step["output"] = {
                        "enabled": bool(cfg.synthetic_gap_holdout_enabled),
                        "requested_base_methods": list(cfg.synthetic_gap_holdout_methods),
                        "method_count": len(gap_payloads),
                        "methods": list(gap_payloads.keys()),
                        "plan": gap_plan_debug,
                        "note": "diagnostic methods are excluded from the normal leaderboard and ranked by deleted-point error by evaluate_reconstructions.py",
                    }

                with debug.step("11_write_academic_debug_pack", method_count=len(reconstruction_payloads)):
                    debug.write_json("reconstruction_quality.json", self._debug_reconstruction_quality(reconstruction_payloads))
                    debug.write_csv("segment_metrics.csv", self._debug_segment_rows(reconstruction_payloads))
                    debug.write_csv("join_metrics.csv", self._debug_join_rows(reconstruction_payloads))
                    debug.write_csv("trajectory_model_metrics.csv", self._debug_trajectory_model_rows(reconstruction_payloads))

                with debug.step("12_write_flight_manifest"):
                    start_time = _iso_utc(float(raw_keyframes[0]["t"])) if raw_keyframes else None
                    end_time = _iso_utc(float(raw_keyframes[-1]["t"])) if raw_keyframes else None
                    callsign = _best_callsign(load_result.dataframe)
                    label_date = start_time[:10] if start_time else "unknown-date"
                    label = f"{flight_id} · {label_date}"

                    flight_json = {
                        "schemaVersion": MANIFEST_SCHEMA_VERSION,
                        "flightId": flight_id,
                        "icao": icao,
                        "callsign": callsign,
                        "startTimeUtc": start_time,
                        "endTimeUtc": end_time,
                        "origin": "",
                        "destination": "",
                        "aircraftRegistration": "",
                        "aircraftType": "",
                        "notes": "",
                    }
                    write_json_packet(flight_json, flight_dir / "flight.json")

                    methods = [
                        {
                            "methodId": RAW_ADSB_METHOD_ID,
                            "label": "Raw ADS-B",
                            "file": f"flights/{flight_id}/methods/{MINIMAL_METHODS_DIRNAME}/raw_adsb.json",
                            "detailedFile": f"flights/{flight_id}/methods/raw_adsb.json",
                        }
                    ]
                    emitted_synthetic_gap_specs = [s for s in synthetic_gap_specs if s.method_id in reconstruction_paths]
                    for spec in [*kalman_specs, *output_specs, *emitted_synthetic_gap_specs]:
                        methods.append(
                            {
                                "methodId": spec.method_id,
                                "label": spec.label,
                                "file": f"flights/{flight_id}/methods/{MINIMAL_METHODS_DIRNAME}/{spec.file_stem}.json",
                                "detailedFile": f"flights/{flight_id}/methods/{spec.file_stem}.json",
                            }
                        )

                    default_method_id = (
                        DEFAULT_METHOD_ID
                        if DEFAULT_METHOD_ID in reconstruction_paths
                        else (next(iter(reconstruction_paths.keys())) if reconstruction_paths else RAW_ADSB_METHOD_ID)
                    )

                    manifest_entry = {
                        "flightId": flight_id,
                        "icao": icao,
                        "callsign": callsign,
                        "label": label,
                        "startTimeUtc": start_time,
                        "endTimeUtc": end_time,
                        "origin": "",
                        "destination": "",
                        "flightMetadataFile": f"flights/{flight_id}/flight.json",
                        "defaultMethod": default_method_id,
                        "methods": methods,
                        "debugDirectory": f"flights/{flight_id}/debug" if debug.enabled else None,
                    }

                default_vspline_id = DEFAULT_METHOD_ID if DEFAULT_METHOD_ID in v_spline_paths else (output_specs[0].method_id if output_specs else None)
                built = BuiltFlight(
                    manifest_entry=manifest_entry,
                    flight_json=flight_json,
                    raw_adsb_path=raw_path,
                    v_spline_path=v_spline_paths.get(default_vspline_id) if default_vspline_id else None,
                    raw_adsb_minimal_path=raw_minimal_path,
                    v_spline_minimal_path=v_spline_minimal_paths.get(default_vspline_id) if default_vspline_id else None,
                    v_spline_paths=v_spline_paths,
                    v_spline_minimal_paths=v_spline_minimal_paths,
                    reconstruction_paths=reconstruction_paths,
                    reconstruction_minimal_paths=reconstruction_minimal_paths,
                )
                debug.write_json(
                    "flight_summary.json",
                    {
                        "manifest_entry": manifest_entry,
                        "raw_adsb_path": str(raw_path),
                        "v_spline_paths": {k: str(v) for k, v in v_spline_paths.items()},
                        "reconstruction_paths": {k: str(v) for k, v in reconstruction_paths.items()},
                        "raw_keyframe_count": len(raw_keyframes),
                        "paired_sample_count": len(prepared.samples),
                    },
                )
                debug.flush_manifest()
                logger.info("Completed flight {}", flight_id)
                return built
        finally:
            if flight_log_sink_id is not None and hasattr(logger, "remove"):
                try:
                    logger.remove(flight_log_sink_id)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Synthetic deleted-gap benchmark helpers
    # ------------------------------------------------------------------

    def _build_synthetic_gap_holdout_plan(self, samples: list[Any]) -> dict[str, Any]:
        """Select deterministic contiguous ADS-B windows to delete for testing.

        This is deliberately a *gap* holdout, not a random point holdout.  The
        reconstruction is fitted without these contiguous windows and the
        evaluator later scores predictions at the deleted samples.
        """
        cfg = self.config
        samples = sorted(list(samples), key=lambda sample: float(sample.t))
        n = len(samples)
        if not bool(cfg.synthetic_gap_holdout_enabled):
            return {"enabled": False, "reason": "synthetic_gap_holdout_disabled"}
        if n < 24:
            return {"enabled": False, "reason": "too_few_prepared_samples", "sample_count": int(n)}

        t = np.asarray([float(s.t) for s in samples], dtype=float)
        dt = np.diff(t)
        finite_dt = dt[np.isfinite(dt) & (dt > 0)]
        if finite_dt.size == 0:
            return {"enabled": False, "reason": "non_increasing_or_degenerate_time_grid", "sample_count": int(n)}
        typical_dt = float(np.median(finite_dt))
        if not math.isfinite(typical_dt) or typical_dt <= 0:
            typical_dt = 1.0

        requested_gap_count = max(0, int(cfg.synthetic_gap_holdout_gap_count))
        if requested_gap_count <= 0:
            return {"enabled": False, "reason": "zero_requested_gap_count"}
        fraction = max(0.0, min(0.50, float(cfg.synthetic_gap_holdout_fraction)))
        if fraction <= 0.0:
            return {"enabled": False, "reason": "zero_requested_holdout_fraction"}

        min_gap_s = max(typical_dt * 2.0, float(cfg.synthetic_gap_holdout_min_gap_s))
        max_gap_s = max(min_gap_s, float(cfg.synthetic_gap_holdout_max_gap_s))
        min_gap_points = max(2, int(round(min_gap_s / typical_dt)))
        max_gap_points = max(min_gap_points, int(round(max_gap_s / typical_dt)))
        target_holdout_points = max(min_gap_points, int(round(n * fraction)))
        gap_count = min(requested_gap_count, max(1, target_holdout_points // min_gap_points))
        if gap_count <= 0:
            return {"enabled": False, "reason": "no_gap_count_after_fraction_limit"}
        gap_points = int(max(min_gap_points, min(max_gap_points, max(2, target_holdout_points // gap_count))))

        guard_points = max(3, int(round(max(float(cfg.synthetic_gap_holdout_guard_s), max_gap_s * 1.5) / typical_dt)))
        start_min = guard_points
        start_max = n - guard_points - gap_points - 1
        if start_max <= start_min:
            guard_points = max(3, int(round(max_gap_s / typical_dt)))
            start_min = guard_points
            start_max = n - guard_points - gap_points - 1
        if start_max <= start_min:
            return {
                "enabled": False,
                "reason": "track_too_short_for_requested_gap_windows_and_guards",
                "sample_count": int(n),
                "gap_points": int(gap_points),
                "guard_points": int(guard_points),
            }

        candidate_starts = [int(round(x)) for x in np.linspace(start_min, start_max, gap_count * 4 + 2)[1:-1]]
        # Use the seed only to rotate the deterministic candidate list; this
        # keeps runs reproducible while avoiding repeated deletion at exactly the
        # same relative positions if the caller changes the seed.
        if candidate_starts:
            shift = int(cfg.synthetic_gap_holdout_seed) % len(candidate_starts)
            candidate_starts = candidate_starts[shift:] + candidate_starts[:shift]

        deleted_indices: set[int] = set()
        windows: list[dict[str, Any]] = []
        max_internal_dt = max(typical_dt * 3.5, min_gap_s * 0.5)
        for start in candidate_starts:
            if len(windows) >= gap_count:
                break
            end = int(start + gap_points - 1)
            if start <= 0 or end >= n - 1:
                continue
            proposed = set(range(start, end + 1))
            if deleted_indices & set(range(start - gap_points, end + gap_points + 1)):
                continue
            block_dt = np.diff(t[start : end + 1])
            if block_dt.size and np.any(block_dt > max_internal_dt):
                continue
            deleted_indices.update(proposed)
            gap_id = f"synthetic_gap_{len(windows) + 1:02d}"
            windows.append(
                {
                    "gap_id": gap_id,
                    "start_sample_index": int(start),
                    "end_sample_index": int(end),
                    "deleted_sample_count": int(end - start + 1),
                    "deleted_t0": float(t[start]),
                    "deleted_t1": float(t[end]),
                    "training_t_before": float(t[start - 1]),
                    "training_t_after": float(t[end + 1]),
                    "training_gap_s": float(t[end + 1] - t[start - 1]),
                    "deleted_duration_s": float(t[end] - t[start]) if end > start else 0.0,
                }
            )

        if not windows or not deleted_indices:
            return {"enabled": False, "reason": "no_non_overlapping_gap_windows_selected", "sample_count": int(n)}

        gap_by_index: dict[int, str] = {}
        for window in windows:
            for idx in range(int(window["start_sample_index"]), int(window["end_sample_index"]) + 1):
                gap_by_index[idx] = str(window["gap_id"])

        training_samples = [s for i, s in enumerate(samples) if i not in deleted_indices]
        deleted_samples = []
        for i in sorted(deleted_indices):
            sample = samples[i]
            y = tuple(float(v) for v in sample.y)
            v = tuple(float(vv) for vv in sample.v)
            deleted_samples.append(
                {
                    "gap_id": gap_by_index.get(i),
                    "sample_index": int(i),
                    "raw_index": int(sample.raw_index) if sample.raw_index is not None else None,
                    "keyframe_id": str(sample.keyframe_id),
                    "t": float(sample.t),
                    "position": {"x_m": y[0], "y_m": y[1], "z_m": y[2]},
                    "velocity": {"east_mps": v[0], "north_mps": v[1], "up_mps": v[2]},
                }
            )

        return {
            "enabled": True,
            "methodology": "deterministic_contiguous_raw_sample_deletion_then_refit_and_score_at_deleted_points",
            "sample_count_before": int(n),
            "training_sample_count": int(len(training_samples)),
            "deleted_sample_count": int(len(deleted_samples)),
            "gap_count": int(len(windows)),
            "typical_dt_s": float(typical_dt),
            "requested_fraction": float(fraction),
            "requested_gap_count": int(requested_gap_count),
            "gap_points": int(gap_points),
            "guard_points": int(guard_points),
            "windows": windows,
            "deleted_samples": deleted_samples,
            "training_samples": training_samples,
            "deleted_keyframe_ids": [str(s["keyframe_id"]) for s in deleted_samples],
        }

    def _filter_raw_keyframes_for_synthetic_gap(self, raw_keyframes: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
        deleted_ids = {str(x) for x in plan.get("deleted_keyframe_ids", [])}
        if not deleted_ids:
            return list(raw_keyframes)
        return [kf for kf in raw_keyframes if str(kf.get("id")) not in deleted_ids]

    def _prepared_with_synthetic_gap_training_samples(
        self,
        prepared: RawKeyframeVSplinePreparation,
        plan: dict[str, Any],
    ) -> RawKeyframeVSplinePreparation:
        diagnostics = dict(prepared.diagnostics)
        diagnostics["synthetic_gap_holdout"] = {
            k: v
            for k, v in plan.items()
            if k not in {"training_samples"}
        }
        return replace(
            prepared,
            samples=list(plan.get("training_samples") or []),
            segments=[],
            diagnostics=diagnostics,
        )

    def _attach_synthetic_gap_metadata(
        self,
        payload: dict[str, Any],
        *,
        plan: dict[str, Any],
        base_method_id: str,
    ) -> dict[str, Any]:
        holdout_meta = {
            k: v
            for k, v in plan.items()
            if k not in {"training_samples"}
        }
        holdout_meta["base_method_id"] = base_method_id
        holdout_meta["diagnostic_role"] = "synthetic_gap_holdout_reconstruction"
        payload["synthetic_gap_holdout"] = _clean_json(holdout_meta)
        quality = payload.setdefault("quality", {})
        if isinstance(quality, dict):
            quality["diagnostic_role"] = "synthetic_gap_holdout_reconstruction"
            quality["synthetic_gap_holdout"] = _clean_json(holdout_meta)
            quality["training_raw_keyframe_count"] = int(quality.get("raw_keyframe_count") or 0)
            quality["deleted_gap_sample_count"] = int(holdout_meta.get("deleted_sample_count") or 0)
        return payload

    # ------------------------------------------------------------------
    # Component fit backends
    # ------------------------------------------------------------------

    def _fit_kalman_rts_components(
        self,
        *,
        flight_id: str,
        prepared_samples: list[Any],
        method_config: TrackOutputPipelineConfig | None = None,
        method_id: str = KALMAN_RTS_METHOD_ID,
    ) -> dict[str, Any]:
        """Fit one whole-flight Kalman/RTS smoother without any segmentation.

        The fit uses the already prepared paired ADS-B observations exactly as
        the V-Spline methods do.  Unlike the spline backends, this method does
        not split by dynamic regime, hard gaps, boundary states, local tuning,
        join harmonization, or adaptive resegmentation.  A single synthetic
        segment wrapper is created only to reuse the shared renderer/evaluator
        contract; it is explicitly marked as a non-segmentation adapter.
        """
        cfg = method_config or self.config
        samples = list(prepared_samples)
        if len(samples) < 2:
            raise ValueError("Kalman/RTS reconstruction requires at least two paired samples")

        # The adapter normally returns strictly time-sorted samples.  Sorting here
        # keeps this backend robust in direct tests without changing the selected
        # observation set.
        samples.sort(key=lambda sample: float(sample.t))
        t = np.asarray([sample.t for sample in samples], dtype=float)
        y = np.asarray([sample.y for sample in samples], dtype=float)
        v = np.asarray([sample.v for sample in samples], dtype=float)

        component_id = f"{flight_id}_kalman_rts_whole_track"
        segment = DynamicSegment(
            segment_id=f"{method_id}_whole_track",
            component_id=component_id,
            start_sample_index=int(getattr(samples[0], "raw_index", 0) or 0),
            end_sample_index=int(getattr(samples[-1], "raw_index", len(samples) - 1) or len(samples) - 1),
            samples=tuple(samples),
            t0=float(t[0]),
            t1=float(t[-1]),
            features={
                "dynamic_segmentation_applied": 0.0,
                "hard_gap_splitting_applied": 0.0,
                "whole_track_state_smoother": 1.0,
            },
            regime_label="kalman_rts_whole_track_no_segmentation",
            start_boundary_id=None,
            end_boundary_id=None,
        )
        fit = fit_kalman_rts_component(
            KalmanRTSInput(t=t, y=y, v=v, dim_names=("x", "y", "z")),
            cfg.kalman_rts_config,
            component_id=component_id,
        )
        quality = evaluate_segment_quality(segment, fit, render_step_s=cfg.v_spline_time_step_s)
        fits: list[tuple[str, DynamicSegment, Any]] = [(segment.segment_id, segment, fit)]

        meta = {
            **segment.as_dict(),
            "component_solver": "kalman_rts_fixed_interval_smoother",
            "fit_mode": "one_whole_track_state_space_smoother_no_segmentation",
            "dynamic_segmentation_applied": False,
            "hard_gap_splitting_applied": False,
            "selected_core_config": asdict(cfg.kalman_rts_config),
            "local_tuning": {"enabled": False, "reason": "kalman_rts_is_not_a_spline"},
            "adaptive_resegmentation": {"enabled": False, "reason": "kalman_rts_is_not_a_spline"},
            "join_velocity_harmonization": {"enabled": False, "reason": "single_global_state_smoother"},
            "boundary_velocity_overrides_applied": [],
            "quality": quality.as_dict(),
            "diagnostics": fit.diagnostics,
        }
        continuity = verify_component_continuity([(segment, fit)])
        component_report = {
            "component_id": component_id,
            "method_id": method_id,
            "t0": float(t[0]),
            "t1": float(t[-1]),
            "start_sample_index": int(segment.start_sample_index),
            "end_sample_index": int(segment.end_sample_index),
            "sample_count": len(samples),
            "segmentation": {
                "enabled": False,
                "reason": "kalman_rts_skips_all_spline_segmentation_and_fits_the_whole_prepared_track",
                "dynamic_segmentation_applied": False,
                "hard_gap_splitting_applied": False,
                "segment_wrapper_role": "renderer_contract_only_not_a_model_segment",
                "boundary_count": 0,
            },
            "boundary_states": {},
            "join_velocity_harmonization": {"enabled": False, "reports": []},
            "adaptive_resegmentation": {"enabled": False, "reason": "not_a_spline"},
            "boundaries": [],
            "segments": [meta],
            "continuity": continuity,
        }

        piecewise_report = {
            "enabled": False,
            "mode": "kalman_rts_whole_track",
            "method_id": method_id,
            "dynamic_segmentation_applied": False,
            "hard_gap_splitting_applied": False,
            "segmentation_role": "not_used; single whole-track state-space smoother",
            "join_continuity_note": "no spline joins are created because the method fits one global fixed-interval state-space smoother",
            "segmentation_diagnostics": {
                "enabled": False,
                "used_by_method": False,
                "component_count": 1,
                "segment_count": 0,
                "boundary_count": 0,
                "dynamic_segmentation_applied": False,
                "hard_gap_splitting_applied": False,
            },
            "connected_components": [component_report],
            "global_quality": _piecewise_global_continuity([component_report], fits),
        }
        return {
            "fits": fits,
            "segment_metadata": {segment.segment_id: meta},
            "piecewise_report": piecewise_report,
        }

    def _augment_boundary_states_with_kalman_rts(
        self,
        *,
        segmented_component: SegmentedComponent,
        shared_states: dict[str, Any],
        method_config: TrackOutputPipelineConfig,
    ) -> dict[str, Any]:
        """Experimental Kalman/RTS boundary-state helper for V-Spline only.

        The returned states are used only as boundary priors/constraints for local
        segmented V-Spline interiors.  The rendered trajectory remains the spline
        fit; Kalman/RTS is not used as the reconstruction method.
        """
        if not segmented_component.boundaries:
            return shared_states
        samples = list(segmented_component.component.samples)
        if len(samples) < 3:
            return shared_states
        try:
            t = np.asarray([s.t for s in samples], dtype=float)
            y = np.asarray([s.y for s in samples], dtype=float)
            v = np.asarray([s.v for s in samples], dtype=float)
            fit = fit_kalman_rts_component(
                KalmanRTSInput(t=t, y=y, v=v, dim_names=("x", "y", "z")),
                method_config.kalman_rts_config,
                component_id=f"{segmented_component.component.component_id}/boundary_helper",
            )
        except Exception as exc:
            logger.warning("Kalman boundary helper failed for {}: {}", segmented_component.component.component_id, exc)
            return shared_states

        out = dict(shared_states)
        for boundary in segmented_component.boundaries:
            base = shared_states.get(boundary.boundary_id)
            if base is None:
                continue
            tb = float(boundary.t_boundary)
            try:
                k_pos = np.asarray(fit.evaluate([tb], deriv=0)[0], dtype=float)
                k_vel = np.asarray(fit.evaluate([tb], deriv=1)[0], dtype=float)
                k_acc = np.asarray(fit.evaluate([tb], deriv=2)[0], dtype=float)
            except Exception:
                continue
            base_pos = np.asarray(base.position_array, dtype=float)
            base_vel = np.asarray(base.velocity_array, dtype=float)
            base_acc = base.acceleration_array
            confidence = min(1.0, max(0.0, float(base.confidence)) + 0.15)
            # Keep the helper weak: robust V-Spline boundary state remains half of
            # the blend so the method is still a spline-owned local objective.
            pos = 0.5 * base_pos + 0.5 * k_pos
            vel = 0.5 * base_vel + 0.5 * k_vel
            if base_acc is not None and np.all(np.isfinite(base_acc)):
                acc = 0.5 * np.asarray(base_acc, dtype=float) + 0.5 * k_acc
            else:
                acc = k_acc
            out[boundary.boundary_id] = SharedBoundaryState(
                boundary_id=boundary.boundary_id,
                t_boundary=tb,
                position_m=tuple(float(x) for x in pos),
                velocity_mps=tuple(float(x) for x in vel),
                acceleration_mps2=tuple(float(x) for x in acc),
                confidence=confidence,
                method=f"kalman_rts_boundary_prior_blended_with_{base.method}",
                diagnostics={
                    "role": "experimental_boundary_state_helper_only",
                    "base_shared_state": base.as_dict(),
                    "kalman_boundary_position_m": k_pos.tolist(),
                    "kalman_boundary_velocity_mps": k_vel.tolist(),
                    "kalman_boundary_acceleration_mps2": k_acc.tolist(),
                    "blend_weights": {"robust_boundary_state": 0.5, "kalman_rts_helper": 0.5},
                    "kalman_fit_diagnostics_summary": {
                        "method": fit.diagnostics.get("method"),
                        "n_observations": fit.diagnostics.get("n_observations"),
                        "objective_total": fit.diagnostics.get("objective_total"),
                    },
                },
            )
        return out

    def _fit_component_with_local_b_spline(
        self,
        *,
        flight_id: str,
        prepared_samples: list[Any],
        segmented_component: SegmentedComponent,
        shared_states: dict[str, Any],
        refinement_depth: int = 0,
        refinement_history: list[dict[str, Any]] | None = None,
        method_config: TrackOutputPipelineConfig | None = None,
        backend: str = "bspline_piecewise",
    ) -> dict[str, Any]:
        """Independent local V-Spline segment fits with coupled joins.

        This backend is still *V-Spline reconstruction with a B-spline basis*:
        position and velocity observations are fitted with a V-Spline-like
        integrated acceleration penalty.  Segments are not merged into one global
        curve.  Instead, joints are smoothed in two passes:

        1. fit each segment with the selected boundary policy: legacy methods may
           use hard raw position anchors, while aviation variants use robust
           boundary-state targets and soft/free internal endpoint velocities;
        2. combine neighbouring endpoint derivatives into one harmonized shared
           join velocity, then refit each segment with that velocity as a hard
           equality constraint.

        Aviation variants do not force every join through one raw ADS-B row.
        They report event-aware continuity so hard gaps and true discontinuities
        are not scored as ordinary spline joins.
        """
        cfg = method_config or self.config
        backend = str(backend)
        if backend not in BSPLINE_BACKENDS and backend not in HERMITE_BACKENDS:
            raise ValueError(f"Unsupported V-Spline backend: {backend!r}")
        if bool(cfg.use_kalman_boundary_prior) and backend in KALMAN_BOUNDARY_BACKENDS:
            shared_states = self._augment_boundary_states_with_kalman_rts(
                segmented_component=segmented_component,
                shared_states=shared_states,
                method_config=cfg,
            )
        tuning_cfg = cfg.local_segment_tuning_config
        component = segmented_component.component
        segments = list(segmented_component.segments)
        fits: list[tuple[str, DynamicSegment, Any]] = []
        component_fit_pairs: list[tuple[DynamicSegment, Any]] = []
        component_segments_meta: list[dict[str, Any]] = []
        segment_metadata: dict[str, dict[str, Any]] = {}
        join_velocity_reports: list[dict[str, Any]] = []
        refinement_history = list(refinement_history or [])
        final_segment_results: list[dict[str, Any]] = []

        selected_policy_params: dict[str, Any] = {
            segment.segment_id: select_local_bspline_params(
                segment,
                cfg.bspline_config,
                cfg.local_segment_policy_config,
            )
            for segment in segments
        }

        def _state_for(boundary_id: str | None) -> Any | None:
            return shared_states.get(boundary_id) if boundary_id else None

        def _endpoint_position_target(sample: Any, state: Any | None = None) -> tuple[np.ndarray, str, dict[str, Any]]:
            raw = np.asarray(sample.y, dtype=float)
            if state is not None and backend in SOFT_BOUNDARY_BACKENDS:
                pos = np.asarray(state.position_array, dtype=float)
                return pos, f"shared_boundary_state_position:{state.method}", {
                    "position_constraint": "robust_weighted_boundary_state",
                    "raw_boundary_position_m": raw.tolist(),
                    "selected_minus_raw_position_error_m": float(np.linalg.norm(pos - raw)),
                    "boundary_state_confidence": float(state.confidence),
                }
            return raw, "raw_segment_boundary_sample", {"position_constraint": "reported_raw_boundary_sample"}

        def _anchor_for_endpoint(segment: DynamicSegment, role: str, sample: Any, sample_index: int, boundary_id: str | None) -> BSplineAnchor:
            state = _state_for(boundary_id)
            position, source_label, extra = _endpoint_position_target(sample, state)
            return BSplineAnchor(
                anchor_id=f"{segment.segment_id}:{role}:{boundary_id or 'component_endpoint'}:position",
                t=float(state.t_boundary if state is not None else sample.t),
                position=position,
                source="shared_boundary_state" if state is not None and backend in SOFT_BOUNDARY_BACKENDS else "raw_segment_boundary_sample",
                sample_index=int(sample_index),
                metadata={
                    "role": f"segment_{role}",
                    "segment_id": segment.segment_id,
                    "boundary_id": boundary_id,
                    "boundary_state_method": None if state is None else state.method,
                    "position_source": source_label,
                    **extra,
                },
            )

        def _position_prior_for_endpoint(
            segment: DynamicSegment,
            role: str,
            sample: Any,
            sample_index: int,
            boundary_id: str | None,
            selected_config: BSplineCoreConfig,
        ) -> BSplinePositionPrior | None:
            if float(selected_config.boundary_position_prior_weight) <= 0.0:
                return None
            state = _state_for(boundary_id)
            if state is None:
                # Component endpoints get a weak soft prior to their raw sample only
                # when hard anchors have been disabled by the aviation variant.
                if bool(selected_config.hard_boundary_positions):
                    return None
                return BSplinePositionPrior(
                    prior_id=f"{segment.segment_id}:{role}:component_endpoint:position",
                    t=float(sample.t),
                    position=np.asarray(sample.y, dtype=float),
                    weight=float(selected_config.boundary_position_prior_weight),
                    confidence=0.5,
                    source="component_endpoint_raw_position_soft_prior",
                    metadata={"role": f"segment_{role}", "segment_id": segment.segment_id, "sample_index": int(sample_index)},
                )
            position, source_label, extra = _endpoint_position_target(sample, state)
            return BSplinePositionPrior(
                prior_id=f"{segment.segment_id}:{role}:{boundary_id}:position",
                t=float(state.t_boundary),
                position=position,
                weight=float(selected_config.boundary_position_prior_weight),
                confidence=float(state.confidence),
                source=source_label,
                metadata={
                    "role": f"segment_{role}",
                    "segment_id": segment.segment_id,
                    "boundary_id": boundary_id,
                    "sample_index": int(sample_index),
                    **extra,
                },
            )

        def _velocity_for_endpoint(sample: Any, boundary_id: str | None, overrides: dict[str, np.ndarray] | None) -> tuple[np.ndarray, str, dict[str, Any]]:
            state = _state_for(boundary_id)
            if boundary_id and overrides and boundary_id in overrides:
                return np.asarray(overrides[boundary_id], dtype=float), "harmonized_post_join_velocity", {"boundary_id": boundary_id}
            if state is not None:
                return np.asarray(state.velocity_array, dtype=float), "shared_boundary_state", {
                    "boundary_id": boundary_id,
                    "confidence": float(state.confidence),
                    "boundary_state_method": state.method,
                }
            return np.asarray(sample.v, dtype=float), "raw_segment_endpoint_sample", {}

        def _velocity_constraint_for_endpoint(
            segment: DynamicSegment,
            role: str,
            sample: Any,
            sample_index: int,
            boundary_id: str | None,
            overrides: dict[str, np.ndarray] | None,
        ) -> BSplineVelocityConstraint:
            velocity, source, extra = _velocity_for_endpoint(sample, boundary_id, overrides)
            return BSplineVelocityConstraint(
                constraint_id=f"{segment.segment_id}:{role}:{boundary_id or 'component_endpoint'}:velocity",
                t=float(_state_for(boundary_id).t_boundary if _state_for(boundary_id) is not None else sample.t),
                velocity=velocity,
                source=source,
                sample_index=int(sample_index),
                metadata={
                    "role": f"segment_{role}",
                    "segment_id": segment.segment_id,
                    "boundary_id": boundary_id,
                    **extra,
                },
            )

        def _velocity_prior_for_endpoint(
            segment: DynamicSegment,
            role: str,
            sample: Any,
            sample_index: int,
            boundary_id: str | None,
        ) -> BSplineVelocityPrior | None:
            state = _state_for(boundary_id)
            if state is None:
                return None
            return BSplineVelocityPrior(
                prior_id=f"{segment.segment_id}:{role}:{boundary_id}:prefit_velocity_prior",
                t=float(state.t_boundary),
                velocity=np.asarray(state.velocity_array, dtype=float),
                weight=float(tuning_cfg.prefit_boundary_velocity_prior_weight),
                confidence=float(state.confidence),
                source=f"prefit_shared_boundary_state:{state.method}",
                metadata={
                    "role": f"segment_{role}",
                    "segment_id": segment.segment_id,
                    "boundary_id": boundary_id,
                    "purpose": "soft_prior_before_post_join_velocity_harmonization",
                    "sample_index": int(sample_index),
                },
            )

        def _acceleration_prior_for_endpoint(
            segment: DynamicSegment,
            role: str,
            boundary_id: str | None,
            selected_config: BSplineCoreConfig,
        ) -> BSplineAccelerationPrior | None:
            if not boundary_id or float(selected_config.boundary_acceleration_prior_weight) <= 0.0:
                return None
            state = _state_for(boundary_id)
            if state is None:
                return None
            acc = state.acceleration_array
            if acc is None:
                return None
            return BSplineAccelerationPrior(
                prior_id=f"{segment.segment_id}:{role}:{boundary_id}:acceleration",
                t=float(state.t_boundary),
                acceleration=acc,
                weight=float(selected_config.boundary_acceleration_prior_weight),
                confidence=float(state.confidence),
                source=f"shared_boundary_state:{state.method}",
                metadata={
                    "role": f"segment_{role}",
                    "segment_id": segment.segment_id,
                    "boundary_id": boundary_id,
                },
            )

        boundary_by_id = {b.boundary_id: b for b in segmented_component.boundaries}
        component_sample_by_global = {int(component.start_sample_index) + i: sample for i, sample in enumerate(component.samples)}

        def _expanded_fit_samples(segment: DynamicSegment) -> tuple[list[Any], dict[str, Any]]:
            guard_s = _overlap_guard_duration_s(backend, str(getattr(cfg.bspline_config, "backend_name", "")))
            # The public helper expects a preset string; backend_name may not contain it
            # during tests, so infer from the configured preset-like eta only if needed.
            preset_hint = "balanced"
            name = str(getattr(cfg.bspline_config, "backend_name", ""))
            for candidate in ("accurate", "balanced", "smooth"):
                if candidate in name:
                    preset_hint = candidate
                    break
            guard_s = _overlap_guard_duration_s(backend, preset_hint)
            if guard_s <= 0.0:
                return list(segment.samples), {
                    "enabled": False,
                    "guard_duration_s": 0.0,
                    "fit_start_sample_index": int(segment.start_sample_index),
                    "fit_end_sample_index": int(segment.end_sample_index),
                    "render_start_sample_index": int(segment.start_sample_index),
                    "render_end_sample_index": int(segment.end_sample_index),
                    "borrowed_before_count": 0,
                    "borrowed_after_count": 0,
                }

            start_idx = int(segment.start_sample_index)
            end_idx = int(segment.end_sample_index)
            start_t = float(segment.t0)
            end_t = float(segment.t1)
            lower_stop = int(component.start_sample_index)
            upper_stop = int(component.end_sample_index)
            stop_boundaries: list[dict[str, Any]] = []
            for boundary in segmented_component.boundaries:
                bidx = int(boundary.sample_index)
                if not _is_boundary_true_discontinuity(boundary):
                    continue
                stop_boundaries.append(boundary.as_dict())
                if bidx <= start_idx:
                    lower_stop = max(lower_stop, bidx)
                if bidx >= end_idx:
                    upper_stop = min(upper_stop, bidx)

            lo = start_idx
            while lo > lower_stop:
                prev = component_sample_by_global.get(lo - 1)
                if prev is None or (start_t - float(prev.t)) > guard_s:
                    break
                lo -= 1
            hi = end_idx
            while hi < upper_stop:
                nxt = component_sample_by_global.get(hi + 1)
                if nxt is None or (float(nxt.t) - end_t) > guard_s:
                    break
                hi += 1
            samples = [component_sample_by_global[i] for i in range(lo, hi + 1) if i in component_sample_by_global]
            return samples, {
                "enabled": True,
                "guard_duration_s": float(guard_s),
                "fit_start_sample_index": int(lo),
                "fit_end_sample_index": int(hi),
                "render_start_sample_index": start_idx,
                "render_end_sample_index": end_idx,
                "borrowed_before_count": int(max(0, start_idx - lo)),
                "borrowed_after_count": int(max(0, hi - end_idx)),
                "true_discontinuity_guard_stops": stop_boundaries,
                "note": "fit uses expanded overlap-save guard observations; render/evaluation uses only original segment interior",
            }

        def _velocity_scale(t: np.ndarray, y: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            scale, report = compute_velocity_confidence_scale(t, y, v)
            return np.asarray(scale, dtype=float), report

        def _apply_regime_speed_floor(config: BSplineCoreConfig | HermiteCoreConfig, segment: DynamicSegment) -> Any:
            # Infer preset from backend name where available; otherwise balanced.
            name = str(getattr(cfg.bspline_config, "backend_name", ""))
            preset_hint = next((p for p in ("accurate", "balanced", "smooth") if p in name), "balanced")
            endpoint_tuned_quintic = "aviation_v_spline_quintic" in name and preset_hint in {"accurate", "balanced"}
            if endpoint_tuned_quintic and getattr(config, "adaptive_speed_floor_mps", None) is not None:
                # The quintic endpoint-artifact preset deliberately sets a lower
                # explicit floor (20 m/s).  Do not replace it with the generic
                # approach/airborne regime floor, which previously made the
                # final segment too flexible in the endpoint guard zone.
                return config
            floor = _regime_speed_floor_mps(segment, preset_hint)
            return replace(config, adaptive_speed_floor_mps=floor)

        def _build_core_input(
            segment: DynamicSegment,
            selected_config: BSplineCoreConfig,
            *,
            boundary_velocity_overrides: dict[str, np.ndarray] | None,
            hard_internal_velocities: bool,
            include_acceleration_priors: bool,
        ) -> BSplineCoreInput:
            fit_samples, overlap_report = _expanded_fit_samples(segment)
            original_samples = list(segment.samples)
            t = np.asarray([s.t for s in fit_samples], dtype=float)
            y = np.asarray([s.y for s in fit_samples], dtype=float)
            v = np.asarray([s.v for s in fit_samples], dtype=float)
            if t.size < 2:
                raise ValueError(f"segment {segment.segment_id} has fewer than two observations")

            start_sample = original_samples[0]
            end_sample = original_samples[-1]
            anchors = (
                _anchor_for_endpoint(segment, "start", start_sample, segment.start_sample_index, segment.start_boundary_id),
                _anchor_for_endpoint(segment, "end", end_sample, segment.end_sample_index, segment.end_boundary_id),
            )
            position_priors = tuple(
                prior
                for prior in (
                    _position_prior_for_endpoint(segment, "start", start_sample, segment.start_sample_index, segment.start_boundary_id, selected_config),
                    _position_prior_for_endpoint(segment, "end", end_sample, segment.end_sample_index, segment.end_boundary_id, selected_config),
                )
                if prior is not None
            )
            velocity_weight_scale, velocity_confidence_report = _velocity_scale(t, y, v)

            velocity_constraints: list[BSplineVelocityConstraint] = []
            velocity_priors: list[BSplineVelocityPrior] = []
            for role, sample, sample_index, boundary_id in (
                ("start", start_sample, segment.start_sample_index, segment.start_boundary_id),
                ("end", end_sample, segment.end_sample_index, segment.end_boundary_id),
            ):
                # True component endpoints are not joins.  Do not force ADS-B
                # endpoint velocity as an exact derivative unless a backend asks
                # for legacy behavior; this was the main quintic end-of-track
                # jerk failure mode on 4BAAD9.  Internal boundary velocities are
                # soft in the prefit and hard after harmonization.
                if boundary_id is None:
                    if bool(getattr(selected_config, "hard_component_endpoint_velocities", False)):
                        velocity_constraints.append(
                            _velocity_constraint_for_endpoint(
                                segment,
                                role,
                                sample,
                                sample_index,
                                boundary_id,
                                boundary_velocity_overrides,
                            )
                        )
                    elif float(getattr(selected_config, "component_endpoint_velocity_prior_weight", 0.0)) > 0.0:
                        velocity_priors.append(
                            BSplineVelocityPrior(
                                prior_id=f"{segment.segment_id}:{role}:component_endpoint:velocity",
                                t=float(sample.t),
                                velocity=np.asarray(sample.v, dtype=float),
                                weight=float(selected_config.component_endpoint_velocity_prior_weight),
                                confidence=0.25,
                                source="component_endpoint_raw_velocity_soft_prior",
                                metadata={
                                    "role": f"segment_{role}",
                                    "segment_id": segment.segment_id,
                                    "sample_index": int(sample_index),
                                    "purpose": "de_trusted_component_endpoint_velocity",
                                },
                            )
                        )
                elif hard_internal_velocities:
                    velocity_constraints.append(
                        _velocity_constraint_for_endpoint(
                            segment,
                            role,
                            sample,
                            sample_index,
                            boundary_id,
                            boundary_velocity_overrides,
                        )
                    )
                else:
                    prior = _velocity_prior_for_endpoint(segment, role, sample, sample_index, boundary_id)
                    if prior is not None:
                        velocity_priors.append(prior)

            acceleration_priors = ()
            if include_acceleration_priors:
                acceleration_priors = tuple(
                    prior
                    for prior in (
                        _acceleration_prior_for_endpoint(segment, "start", segment.start_boundary_id, selected_config),
                        _acceleration_prior_for_endpoint(segment, "end", segment.end_boundary_id, selected_config),
                    )
                    if prior is not None
                )

            component_endpoint_guard_times = []
            if segment.start_boundary_id is None:
                component_endpoint_guard_times.append(float(start_sample.t))
            if segment.end_boundary_id is None:
                component_endpoint_guard_times.append(float(end_sample.t))

            return BSplineCoreInput(
                t=t,
                y=y,
                v=v,
                dim_names=("x", "y", "z"),
                anchors=anchors,
                position_priors=position_priors,
                velocity_priors=tuple(velocity_priors),
                velocity_constraints=tuple(velocity_constraints),
                acceleration_priors=tuple(acceleration_priors),
                velocity_weight_scale=velocity_weight_scale,
                metadata={
                    "component": component.as_dict(),
                    "segment": segment.as_dict(),
                    "overlap_save": overlap_report,
                    "velocity_confidence_scaling": velocity_confidence_report,
                    "render_window": {"t0": float(segment.t0), "t1": float(segment.t1)},
                    "component_endpoint_guard_times_s": component_endpoint_guard_times,
                },
            )

        def _hermite_config_from_selected_params(selected_params: Any) -> HermiteCoreConfig:
            # The B-spline policy represents the effective acceleration penalty as
            # lambda/eta multiplied by acceleration_penalty_multiplier.  The paper
            # Hermite core has one interval-penalty scale, so fold that multiplier
            # into the selected lambda/eta to keep candidate strength comparable.
            multiplier = float(getattr(selected_params, "acceleration_penalty_multiplier", 1.0) or 1.0)
            velocity_ratio = float(cfg.hermite_config.velocity_weight) / max(float(cfg.bspline_config.velocity_weight), 1e-12)
            return replace(
                cfg.hermite_config,
                penalty_mode=selected_params.penalty_mode,
                smoothing_lambda=float(selected_params.smoothing_lambda) * multiplier,
                adaptive_eta=float(selected_params.adaptive_eta) * multiplier,
                velocity_weight=max(float(selected_params.velocity_weight) * velocity_ratio, 1e-12),
                hard_endpoint_constraints=True,
                hard_endpoint_positions=True,
                hard_endpoint_velocities=bool(cfg.hermite_config.hard_endpoint_velocities),
            )

        def _build_hermite_core_input(
            segment: DynamicSegment,
            selected_bspline_config: BSplineCoreConfig,
            *,
            boundary_velocity_overrides: dict[str, np.ndarray] | None,
            hard_internal_velocities: bool,
            include_acceleration_priors: bool,
        ) -> tuple[HermiteCoreInput, HermiteEndpointConstraints, tuple[BSplineAnchor, ...], tuple[BSplineVelocityConstraint, ...], tuple[BSplineVelocityPrior, ...], tuple[BSplineAccelerationPrior, ...]]:
            samples = list(segment.samples)
            t = np.asarray([s.t for s in samples], dtype=float)
            y = np.asarray([s.y for s in samples], dtype=float)
            v = np.asarray([s.v for s in samples], dtype=float)
            velocity_weight_scale, velocity_confidence_report = _velocity_scale(t, y, v)
            if t.size < 2:
                raise ValueError(f"segment {segment.segment_id} has fewer than two observations")

            start_sample = samples[0]
            end_sample = samples[-1]
            start_anchor = _anchor_for_endpoint(segment, "start", start_sample, segment.start_sample_index, segment.start_boundary_id)
            end_anchor = _anchor_for_endpoint(segment, "end", end_sample, segment.end_sample_index, segment.end_boundary_id)

            velocity_constraints: list[BSplineVelocityConstraint] = []
            velocity_priors: list[BSplineVelocityPrior] = []

            def _endpoint_state(role: str, sample: Any, sample_index: int, boundary_id: str | None, anchor: BSplineAnchor) -> tuple[HermiteEndpointState, bool]:
                hard_velocity = bool(cfg.hermite_config.hard_endpoint_velocities) and (boundary_id is None or hard_internal_velocities)
                velocity, _source, _extra = _velocity_for_endpoint(sample, boundary_id, boundary_velocity_overrides)
                if hard_velocity:
                    velocity_constraints.append(
                        _velocity_constraint_for_endpoint(
                            segment,
                            role,
                            sample,
                            sample_index,
                            boundary_id,
                            boundary_velocity_overrides,
                        )
                    )
                else:
                    prior = _velocity_prior_for_endpoint(segment, role, sample, sample_index, boundary_id)
                    if prior is not None:
                        velocity_priors.append(prior)
                return (
                    HermiteEndpointState(
                        position=anchor.position_array(y.shape[1]),
                        velocity=np.asarray(velocity, dtype=float),
                    ),
                    hard_velocity,
                )

            start_state, hard_start_velocity = _endpoint_state(
                "start",
                start_sample,
                segment.start_sample_index,
                segment.start_boundary_id,
                start_anchor,
            )
            end_state, hard_end_velocity = _endpoint_state(
                "end",
                end_sample,
                segment.end_sample_index,
                segment.end_boundary_id,
                end_anchor,
            )

            acceleration_priors = ()
            if include_acceleration_priors:
                acceleration_priors = tuple(
                    prior
                    for prior in (
                        _acceleration_prior_for_endpoint(segment, "start", segment.start_boundary_id, selected_bspline_config),
                        _acceleration_prior_for_endpoint(segment, "end", segment.end_boundary_id, selected_bspline_config),
                    )
                    if prior is not None
                )

            endpoint_constraints = HermiteEndpointConstraints(
                start=start_state,
                end=end_state,
                hard_start_position=True,
                hard_end_position=True,
                hard_start_velocity=hard_start_velocity,
                hard_end_velocity=hard_end_velocity,
            )
            return (
                HermiteCoreInput(t=t, y=y, v=v, dim_names=("x", "y", "z"), velocity_weight_scale=velocity_weight_scale),
                endpoint_constraints,
                (start_anchor, end_anchor),
                tuple(velocity_constraints),
                tuple(velocity_priors),
                tuple(acceleration_priors),
            )

        def _attach_hermite_segment_diagnostics(
            *,
            fit: Any,
            core_input: HermiteCoreInput,
            anchors: tuple[BSplineAnchor, ...],
            velocity_constraints: tuple[BSplineVelocityConstraint, ...],
            velocity_priors: tuple[BSplineVelocityPrior, ...],
            acceleration_priors: tuple[BSplineAccelerationPrior, ...],
            selected_params: Any,
            selected_config: HermiteCoreConfig,
            component_id: str,
        ) -> None:
            y_hat = fit.evaluate(core_input.t, deriv=0)
            v_hat = fit.evaluate(core_input.t, deriv=1)
            pos_delta = y_hat - core_input.y
            vel_delta = v_hat - core_input.v
            e3 = np.linalg.norm(pos_delta, axis=1)
            ev = np.linalg.norm(vel_delta, axis=1)

            anchor_reports: list[dict[str, Any]] = []
            anchor_errors: list[float] = []
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

            velocity_constraint_reports: list[dict[str, Any]] = []
            velocity_constraint_errors: list[float] = []
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

            velocity_prior_reports: list[dict[str, Any]] = []
            velocity_prior_errors: list[float] = []
            for prior in velocity_priors:
                pred = fit.evaluate([float(prior.t)], deriv=1)[0]
                raw = prior.velocity_array(fit.dimension)
                err_vec = pred - raw
                err = float(np.linalg.norm(err_vec))
                velocity_prior_errors.append(err)
                velocity_prior_reports.append(
                    {
                        **prior.as_dict(),
                        "applied_in_core": False,
                        "fitted_velocity_mps": pred.tolist(),
                        "error_vector_mps": err_vec.tolist(),
                        "error_norm_mps": err,
                    }
                )

            acceleration_prior_reports: list[dict[str, Any]] = []
            acceleration_prior_errors: list[float] = []
            for prior in acceleration_priors:
                pred = fit.evaluate([float(prior.t)], deriv=2)[0]
                raw = prior.acceleration_array(fit.dimension)
                err_vec = pred - raw
                err = float(np.linalg.norm(err_vec))
                acceleration_prior_errors.append(err)
                acceleration_prior_reports.append(
                    {
                        **prior.as_dict(),
                        "applied_in_core": False,
                        "fitted_acceleration_mps2": pred.tolist(),
                        "error_vector_mps2": err_vec.tolist(),
                        "error_norm_mps2": err,
                    }
                )

            span = float(fit.t[-1] - fit.t[0])
            if span > 0:
                step = max(min(span / 200.0, 1.0), 0.25)
                grid = np.arange(float(fit.t[0]), float(fit.t[-1]) + step * 0.5, step)
                if grid.size == 0 or grid[-1] < fit.t[-1]:
                    grid = np.append(grid, fit.t[-1])
                acc_norm = np.linalg.norm(fit.evaluate(grid, deriv=2), axis=1)
                jerk_norm = np.linalg.norm(fit.evaluate(grid, deriv=3), axis=1)
            else:
                acc_norm = np.zeros(1, dtype=float)
                jerk_norm = np.zeros(1, dtype=float)

            fit.diagnostics.update(
                {
                    "method": "paper_oriented_or_stable_v_spline_nodal_hermite_piecewise_segment",
                    "backend": backend,
                    "component_id": component_id,
                    "basis": "nodal_cubic_hermite",
                    "objective": "position_residuals_plus_global_velocity_residuals_plus_n_scaled_integrated_squared_acceleration_subject_to_endpoint_constraints",
                    "n_basis": int(2 * fit.n_observations),
                    "dof_per_dimension": int(2 * fit.n_observations),
                    "selected_policy_params": selected_params.as_dict(),
                    "effective_acceleration_penalty_multiplier_folded_into_lambda_or_eta": float(
                        getattr(selected_params, "acceleration_penalty_multiplier", 1.0) or 1.0
                    ),
                    "solver": {
                        "method": "dense_constrained_normal_equations",
                        "condition_number_hessian": fit.diagnostics.get("condition_number"),
                        "normal_relative_residual": fit.diagnostics.get("solve_relative_residual_free_rows"),
                        "constraint_max_abs_error": fit.diagnostics.get("hard_endpoint_constraint_max_abs_error"),
                    },
                    "hard_position_anchors": {
                        "raw_anchor_count": int(len(anchors)),
                        "deduped_anchor_count": int(len(anchors)),
                        "max_anchor_error_m": float(max(anchor_errors) if anchor_errors else 0.0),
                        "p95_anchor_error_m": float(np.quantile(anchor_errors, 0.95)) if anchor_errors else 0.0,
                        "anchors": anchor_reports,
                    },
                    "n_anchors": int(len(anchors)),
                    "max_anchor_error_m": float(max(anchor_errors) if anchor_errors else 0.0),
                    "boundary_velocity_priors": {
                        "count": int(len(velocity_priors)),
                        "applied_in_core": False,
                        "reason": "paper Hermite V-Spline core has only position/velocity observations and acceleration penalty; prefit priors are diagnostics only",
                        "max_error_mps": float(max(velocity_prior_errors) if velocity_prior_errors else 0.0),
                        "p95_error_mps": float(np.quantile(velocity_prior_errors, 0.95)) if velocity_prior_errors else 0.0,
                        "priors": velocity_prior_reports,
                    },
                    "boundary_acceleration_priors": {
                        "count": int(len(acceleration_priors)),
                        "applied_in_core": False,
                        "reason": "not part of the paper Hermite V-Spline objective",
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
                    "config": asdict(selected_config),
                }
            )

        def _fit_one(
            segment: DynamicSegment,
            selected_params: Any,
            *,
            boundary_velocity_overrides: dict[str, np.ndarray] | None,
            hard_internal_velocities: bool,
            include_acceleration_priors: bool,
            phase: str,
        ) -> tuple[Any, Any, Any]:
            selected_bspline_config = _apply_regime_speed_floor(selected_params.to_core_config(cfg.bspline_config), segment)
            if backend in BSPLINE_BACKENDS:
                selected_config = selected_bspline_config
                if not hard_internal_velocities and float(tuning_cfg.prefit_boundary_velocity_prior_weight) > 0.0:
                    selected_config = replace(
                        selected_config,
                        boundary_velocity_prior_weight=float(tuning_cfg.prefit_boundary_velocity_prior_weight),
                    )
                core_input = _build_core_input(
                    segment,
                    selected_config,
                    boundary_velocity_overrides=boundary_velocity_overrides,
                    hard_internal_velocities=hard_internal_velocities,
                    include_acceleration_priors=include_acceleration_priors,
                )
                fit = fit_b_spline_component(
                    core_input,
                    selected_config,
                    component_id=f"{component.component_id}/{segment.segment_id}/{phase}",
                )
            elif backend in HERMITE_BACKENDS:
                selected_config = _apply_regime_speed_floor(_hermite_config_from_selected_params(selected_params), segment)
                (
                    core_input,
                    endpoint_constraints,
                    anchors,
                    velocity_constraints,
                    velocity_priors,
                    acceleration_priors,
                ) = _build_hermite_core_input(
                    segment,
                    selected_bspline_config,
                    boundary_velocity_overrides=boundary_velocity_overrides,
                    hard_internal_velocities=hard_internal_velocities,
                    include_acceleration_priors=include_acceleration_priors,
                )
                fit = fit_hermite_v_spline_core(
                    core_input,
                    selected_config,
                    endpoint_constraints=endpoint_constraints,
                )
                _attach_hermite_segment_diagnostics(
                    fit=fit,
                    core_input=core_input,
                    anchors=anchors,
                    velocity_constraints=velocity_constraints,
                    velocity_priors=velocity_priors,
                    acceleration_priors=acceleration_priors,
                    selected_params=selected_params,
                    selected_config=selected_config,
                    component_id=f"{component.component_id}/{segment.segment_id}/{phase}",
                )
            else:
                raise ValueError(f"Unsupported V-Spline backend: {backend!r}")

            quality = evaluate_segment_quality(
                segment,
                fit,
                render_step_s=cfg.v_spline_time_step_s,
            )
            return fit, quality, selected_config

        def _segment_holdout_report(segment: DynamicSegment, selected_config: Any) -> dict[str, Any]:
            """Deterministic local holdout check for raw-fidelity diagnostics.

            This is deliberately local and cheap: it withholds a small fraction
            of interior observations inside the already selected segment and
            refits the same spline family on the remaining observations.  The
            production reconstruction above is unchanged; these numbers are a
            diagnostic guard against variants that merely interpolate noisy raw
            ADS-B samples.
            """
            fraction = float(cfg.holdout_evaluation_fraction)
            if fraction <= 0.0:
                return {"enabled": False, "reason": "disabled"}
            fraction = min(max(fraction, 0.01), 0.40)
            samples = list(segment.samples)
            n = len(samples)
            min_train = max(8, int(getattr(selected_config, "degree", 3)) + 4)
            if n < min_train + 3:
                return {"enabled": False, "reason": "too_few_samples", "n_observations": int(n), "min_train": int(min_train)}
            interior = list(range(1, n - 1))
            target_count = max(1, int(round(fraction * n)))
            target_count = min(target_count, max(1, n - min_train), len(interior))
            stride = max(2, int(round(len(interior) / max(target_count, 1))))
            holdout_local = interior[::stride][:target_count]
            if not holdout_local:
                return {"enabled": False, "reason": "no_holdout_indices"}
            train_samples = [s for i, s in enumerate(samples) if i not in set(holdout_local)]
            holdout_samples = [samples[i] for i in holdout_local]
            try:
                t_train = np.asarray([s.t for s in train_samples], dtype=float)
                y_train = np.asarray([s.y for s in train_samples], dtype=float)
                v_train = np.asarray([s.v for s in train_samples], dtype=float)
                t_hold = np.asarray([s.t for s in holdout_samples], dtype=float)
                y_hold = np.asarray([s.y for s in holdout_samples], dtype=float)
                velocity_scale, velocity_report = compute_velocity_confidence_scale(t_train, y_train, v_train)
                if backend in BSPLINE_BACKENDS:
                    hold_cfg = replace(
                        selected_config,
                        hard_boundary_positions=False,
                        hard_component_endpoint_positions=False,
                        boundary_position_prior_weight=0.0,
                        boundary_velocity_prior_weight=0.0,
                        boundary_acceleration_prior_weight=0.0,
                        robust_iterations=min(int(getattr(selected_config, "robust_iterations", 1)), 1),
                    )
                    hold_fit = fit_b_spline_component(
                        BSplineCoreInput(
                            t=t_train,
                            y=y_train,
                            v=v_train,
                            dim_names=("x", "y", "z"),
                            velocity_weight_scale=velocity_scale,
                            metadata={
                                "diagnostic_role": "segment_observation_holdout",
                                "source_segment_id": segment.segment_id,
                            },
                        ),
                        hold_cfg,
                        component_id=f"{component.component_id}/{segment.segment_id}/holdout",
                    )
                elif backend in HERMITE_BACKENDS:
                    hold_fit = fit_hermite_v_spline_core(
                        HermiteCoreInput(t=t_train, y=y_train, v=v_train, dim_names=("x", "y", "z"), velocity_weight_scale=velocity_scale),
                        replace(selected_config, optimize=False),
                        endpoint_constraints=None,
                    )
                else:  # pragma: no cover
                    return {"enabled": False, "reason": f"unsupported_backend:{backend}"}
                y_pred = hold_fit.evaluate(t_hold, deriv=0)
                metrics = position_error_metrics(y_hold, y_pred)
                return {
                    "enabled": True,
                    "fraction_requested": float(fraction),
                    "n_observations": int(n),
                    "n_train": int(len(train_samples)),
                    "n_holdout": int(len(holdout_samples)),
                    "holdout_local_indices": [int(i) for i in holdout_local],
                    "holdout_global_sample_indices": [int(segment.start_sample_index + i) for i in holdout_local],
                    "metrics": metrics,
                    "velocity_confidence_scaling": velocity_report,
                }
            except Exception as exc:
                return {
                    "enabled": False,
                    "reason": "holdout_refit_failed",
                    "error": str(exc),
                    "n_observations": int(n),
                    "n_holdout_requested": int(len(holdout_local)),
                }

        def _position_slope_at_join(left_segment: DynamicSegment, right_segment: DynamicSegment) -> np.ndarray | None:
            left_samples = list(left_segment.samples)
            right_samples = list(right_segment.samples)
            options: list[np.ndarray] = []
            if len(left_samples) >= 2:
                s0, s1 = left_samples[-2], left_samples[-1]
                dt = float(s1.t - s0.t)
                if dt > 0:
                    options.append((np.asarray(s1.y, dtype=float) - np.asarray(s0.y, dtype=float)) / dt)
            if len(right_samples) >= 2:
                s0, s1 = right_samples[0], right_samples[1]
                dt = float(s1.t - s0.t)
                if dt > 0:
                    options.append((np.asarray(s1.y, dtype=float) - np.asarray(s0.y, dtype=float)) / dt)
            if len(left_samples) >= 2 and len(right_samples) >= 2:
                s0, s1 = left_samples[-2], right_samples[1]
                dt = float(s1.t - s0.t)
                if dt > 0:
                    options.append((np.asarray(s1.y, dtype=float) - np.asarray(s0.y, dtype=float)) / dt)
            if not options:
                return None
            return np.median(np.vstack(options), axis=0)

        def _clip_join_velocity(v: np.ndarray, left_segment: DynamicSegment, right_segment: DynamicSegment) -> tuple[np.ndarray, dict[str, Any]]:
            samples = list(left_segment.samples[-4:]) + list(right_segment.samples[:4])
            reported = np.asarray([s.v for s in samples], dtype=float)
            speeds = np.linalg.norm(reported, axis=1) if reported.size else np.zeros(0, dtype=float)
            ref = float(np.median(speeds)) if speeds.size else float(np.linalg.norm(v))
            max_speed = max(30.0, float(cfg.boundary_state_config.max_velocity_factor) * max(ref, 1e-6))
            speed = float(np.linalg.norm(v))
            clipped = False
            out = np.asarray(v, dtype=float)
            if speed > max_speed:
                out = out * (max_speed / speed)
                clipped = True
            return out, {"speed_mps_before_clip": speed, "speed_limit_mps": max_speed, "clipped": bool(clipped)}

        def _estimate_harmonized_join_velocities(
            prefit_pairs: list[tuple[DynamicSegment, Any, Any]],
        ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
            overrides: dict[str, np.ndarray] = {}
            reports: list[dict[str, Any]] = []
            if len(prefit_pairs) < 2:
                return overrides, reports
            for (left_segment, left_fit, left_quality), (right_segment, right_fit, right_quality) in zip(prefit_pairs[:-1], prefit_pairs[1:]):
                boundary_id = left_segment.end_boundary_id
                if not boundary_id or boundary_id != right_segment.start_boundary_id:
                    continue
                state = shared_states.get(boundary_id)
                if state is None:
                    continue
                tb = float(state.t_boundary)
                left_v = np.asarray(left_fit.evaluate([tb], deriv=1)[0], dtype=float)
                right_v = np.asarray(right_fit.evaluate([tb], deriv=1)[0], dtype=float)
                state_v = np.asarray(state.velocity_array, dtype=float)
                raw_v = np.asarray(left_segment.samples[-1].v, dtype=float)
                slope_v = _position_slope_at_join(left_segment, right_segment)

                left_rmse = float(left_quality.raw_fit_metrics.get("rmse_3d_m", 0.0))
                right_rmse = float(right_quality.raw_fit_metrics.get("rmse_3d_m", 0.0))
                left_w = float(tuning_cfg.harmonized_fit_velocity_weight) / (1.0 + left_rmse / 80.0)
                right_w = float(tuning_cfg.harmonized_fit_velocity_weight) / (1.0 + right_rmse / 80.0)
                weighted: list[tuple[str, float, np.ndarray]] = [
                    ("left_prefit_endpoint_velocity", left_w, left_v),
                    ("right_prefit_endpoint_velocity", right_w, right_v),
                    ("initial_shared_boundary_state_velocity", float(tuning_cfg.harmonized_boundary_state_weight), state_v),
                    ("reported_boundary_velocity", float(tuning_cfg.harmonized_reported_velocity_weight), raw_v),
                ]
                if slope_v is not None:
                    weighted.append(("raw_position_slope_velocity", float(tuning_cfg.harmonized_position_slope_weight), np.asarray(slope_v, dtype=float)))
                kept = [(name, w, vec) for name, w, vec in weighted if w > 0 and np.all(np.isfinite(vec))]
                if not kept:
                    continue
                total_w = float(sum(w for _, w, _ in kept))
                velocity = sum(w * vec for _, w, vec in kept) / max(total_w, 1e-12)
                velocity, clip_report = _clip_join_velocity(np.asarray(velocity, dtype=float), left_segment, right_segment)
                overrides[boundary_id] = velocity
                reports.append(
                    {
                        "boundary_id": boundary_id,
                        "t_boundary": tb,
                        "left_segment_id": left_segment.segment_id,
                        "right_segment_id": right_segment.segment_id,
                        "harmonized_velocity_mps": velocity.tolist(),
                        "initial_shared_velocity_mps": state_v.tolist(),
                        "reported_boundary_velocity_mps": raw_v.tolist(),
                        "left_prefit_velocity_mps": left_v.tolist(),
                        "right_prefit_velocity_mps": right_v.tolist(),
                        "raw_position_slope_velocity_mps": None if slope_v is None else np.asarray(slope_v, dtype=float).tolist(),
                        "left_prefit_rmse_3d_m": left_rmse,
                        "right_prefit_rmse_3d_m": right_rmse,
                        "weights": {name: float(w) for name, w, _ in kept},
                        "clip": clip_report,
                    }
                )
            return overrides, reports

        boundary_velocity_overrides: dict[str, np.ndarray] = {}
        prefit_pairs: list[tuple[DynamicSegment, Any, Any]] = []
        if (
            bool(tuning_cfg.join_velocity_harmonization)
            and bool(segmented_component.boundaries)
        ):
            for segment in _progress_iter(
                segments,
                enabled=cfg.show_progress,
                desc=f"{backend} prefit {flight_id}/{component.component_id}",
                unit="segment",
                leave=False,
            ):
                selected_params = selected_policy_params[segment.segment_id]
                fit, quality, _selected_config = _fit_one(
                    segment,
                    selected_params,
                    boundary_velocity_overrides=None,
                    hard_internal_velocities=False,
                    include_acceleration_priors=False,
                    phase="prefit_free_join_velocity",
                )
                prefit_pairs.append((segment, fit, quality))
            boundary_velocity_overrides, join_velocity_reports = _estimate_harmonized_join_velocities(prefit_pairs)

        for segment in _progress_iter(
            segments,
            enabled=cfg.show_progress,
            desc=f"{backend} tuned segments {flight_id}/{component.component_id}",
            unit="segment",
            leave=False,
        ):
            selected_policy = selected_policy_params[segment.segment_id]
            candidate_reports: list[dict[str, Any]] = []
            best: tuple[float, Any, Any, Any, Any, Any] | None = None
            candidates = [selected_policy]
            if bool(tuning_cfg.enabled):
                candidates = generate_bspline_param_candidates(selected_policy, cfg.bspline_config, tuning_cfg)

            for candidate in candidates:
                try:
                    fit, quality, selected_config = _fit_one(
                        segment,
                        candidate,
                        boundary_velocity_overrides=boundary_velocity_overrides,
                        hard_internal_velocities=True,
                        include_acceleration_priors=True,
                        phase="final_tuned",
                    )
                    score = score_bspline_candidate(segment, fit, quality, tuning_cfg)
                    report = {
                        "candidate_params": candidate.as_dict(),
                        "score": score.as_dict(),
                        "quality": quality.as_dict(),
                        "diagnostics_summary": {
                            "n_basis": fit.diagnostics.get("n_basis"),
                            "max_anchor_error_m": fit.diagnostics.get("max_anchor_error_m"),
                            "hard_velocity_constraint_max_error_mps": fit.diagnostics.get("hard_velocity_constraints", {}).get("max_error_mps"),
                            "condition_number_hessian": fit.diagnostics.get("solver", {}).get("condition_number_hessian"),
                        },
                    }
                    if bool(tuning_cfg.include_all_candidate_reports):
                        candidate_reports.append(report)
                    if best is None or float(score.score) < float(best[0]):
                        best = (float(score.score), candidate, fit, quality, selected_config, score)
                except Exception as exc:
                    if bool(tuning_cfg.include_all_candidate_reports):
                        candidate_reports.append(
                            {
                                "candidate_params": candidate.as_dict(),
                                "failed": True,
                                "error": str(exc),
                            }
                        )
                    continue

            if best is None:
                # Let the base policy failure surface with the original traceback.
                fit, quality, selected_config = _fit_one(
                    segment,
                    selected_policy,
                    boundary_velocity_overrides=boundary_velocity_overrides,
                    hard_internal_velocities=True,
                    include_acceleration_priors=True,
                    phase="final_policy_fallback",
                )
                selected_params = selected_policy
                selected_score = score_bspline_candidate(segment, fit, quality, tuning_cfg)
            else:
                _, selected_params, fit, quality, selected_config, selected_score = best

            final_segment_results.append(
                {
                    "segment": segment,
                    "fit": fit,
                    "quality": quality,
                    "selected_params": selected_params,
                    "selected_config": selected_config,
                    "selected_score": selected_score,
                }
            )
            fits.append((segment.segment_id, segment, fit))
            component_fit_pairs.append((segment, fit))
            holdout_report = _segment_holdout_report(segment, selected_config)

            meta = {
                **segment.as_dict(),
                "component_solver": (
                    "local_segment_paper_or_stable_hermite_v_spline_tuned_with_post_join_velocity_harmonization"
                    if backend in HERMITE_BACKENDS
                    else "local_segment_b_spline_or_quintic_v_spline_tuned_with_post_join_velocity_harmonization"
                ),
                "selected_params": selected_params.as_dict(),
                "selected_core_config": asdict(selected_config),
                "local_tuning": {
                    "enabled": bool(tuning_cfg.enabled),
                    "objective": str(tuning_cfg.objective),
                    "selected_score": selected_score.as_dict(),
                    "candidate_count": len(candidates),
                    "successful_candidate_count": int(sum(1 for c in candidate_reports if not c.get("failed"))),
                    "candidates": candidate_reports,
                },
                "endpoint_constraints": {
                    "mode": (
                        "robust_boundary_position_with_soft_or_hard_velocity_policy_acceleration_priors_diagnostic_only"
                        if backend in HERMITE_BACKENDS
                        else "soft_or_hard_boundary_position_policy_with_harmonized_shared_boundary_velocity_plus_soft_shared_boundary_acceleration"
                    ),
                    "source": (
                        "legacy piecewise methods may hard-anchor raw samples; aviation variants hard-anchor robust shared boundary positions instead of forcing one noisy raw ADS-B boundary row"
                    ),
                    "hard_position_anchor_count": int(fit.diagnostics.get("n_anchors") or 0),
                    "hard_velocity_constraint_count": int(fit.diagnostics.get("hard_velocity_constraints", {}).get("count") or 0),
                    "soft_acceleration_prior_count": (
                        0
                        if backend in HERMITE_BACKENDS
                        else int(fit.diagnostics.get("boundary_acceleration_priors", {}).get("count") or 0)
                    ),
                    "diagnostic_acceleration_prior_count": (
                        int(fit.diagnostics.get("boundary_acceleration_priors", {}).get("count") or 0)
                        if backend in HERMITE_BACKENDS
                        else 0
                    ),
                    "start_boundary_id": segment.start_boundary_id,
                    "end_boundary_id": segment.end_boundary_id,
                },
                "quality": quality.as_dict(),
                "holdout_evaluation": holdout_report,
                "diagnostics": fit.diagnostics,
            }
            segment_metadata[segment.segment_id] = meta
            component_segments_meta.append(meta)

        refined_component, refinement_report = self._refine_bad_b_spline_segments(
            segmented_component=segmented_component,
            final_segment_results=final_segment_results,
            refinement_depth=refinement_depth,
            method_config=cfg,
        )
        if refined_component is not None:
            refinement_history.append(refinement_report)
            refined_shared_states = estimate_shared_boundary_states(
                prepared_samples,
                refined_component.boundaries,
                cfg.boundary_state_config,
            )
            return self._fit_component_with_local_b_spline(
                flight_id=flight_id,
                prepared_samples=prepared_samples,
                segmented_component=refined_component,
                shared_states=refined_shared_states,
                refinement_depth=refinement_depth + 1,
                refinement_history=refinement_history,
                method_config=cfg,
                backend=backend,
            )

        exact_join_claimed = (
            (backend in BSPLINE_BACKENDS and bool(cfg.bspline_config.hard_boundary_positions))
            or (
                backend == "hermite_piecewise"
                and bool(cfg.hermite_config.hard_endpoint_positions)
                and bool(cfg.hermite_config.hard_endpoint_velocities)
            )
        )
        continuity = verify_component_continuity(
            component_fit_pairs,
            position_tolerance_m=(1e-6 if exact_join_claimed else 25.0),
            velocity_tolerance_mps=(1e-6 if exact_join_claimed else 5.0),
        )
        event_aware_continuity = _event_aware_component_continuity(segmented_component, continuity)
        if exact_join_claimed and not continuity.get("position_continuity_ok", True):
            raise RuntimeError(
                f"Local V-Spline ({backend}) C0 continuity failed for {flight_id}/{component.component_id}: "
                f"max_position_jump_m={continuity.get('max_position_jump_m')}"
            )
        if exact_join_claimed and not continuity.get("velocity_continuity_ok", True):
            raise RuntimeError(
                f"Local V-Spline ({backend}) C1 continuity failed for {flight_id}/{component.component_id}: "
                f"max_velocity_jump_mps={continuity.get('max_velocity_jump_mps')}"
            )

        boundary_report_by_id = {str(item.get("boundary_id")): item for item in join_velocity_reports}
        if backend in HERMITE_BACKENDS:
            solver_summary = {
                "backend": backend,
                "basis": "nodal_cubic_hermite_per_segment",
                "objective": (
                    "paper_position_velocity_residuals_plus_n_scaled_integrated_squared_acceleration_"
                    "with_per_segment_quality_tuning_and_adsb_velocity_confidence_scaling"
                ),
                "continuity": (
                    "legacy Hermite claims exact C0/C1 with hard endpoint states; stable Hermite uses robust boundary states and soft/noisy velocity handling; acceleration priors are diagnostic only"
                ),
                "local_segment_tuning_config": tuning_cfg.as_dict(),
            }
        else:
            solver_summary = {
                "backend": backend,
                "basis": (
                    f"clamped_degree_{cfg.bspline_config.degree}_b_spline_per_hard_gap_component"
                    if backend in GLOBAL_BSPLINE_BACKENDS
                    else f"clamped_degree_{cfg.bspline_config.degree}_b_spline_per_segment"
                ),
                "objective": (
                    "global_component_position_velocity_residuals_plus_v_spline_acceleration_penalty_no_dynamic_regime_joins"
                    if backend in GLOBAL_BSPLINE_BACKENDS
                    else "local_segment_position_velocity_residuals_plus_confidence_weighted_velocity_residuals_plus_v_spline_like_acceleration_or_higher_order_penalty_with_per_segment_quality_tuning"
                ),
                "continuity": "B-spline aviation variants now hard-anchor robust shared boundary positions for exact C0 at ordinary joins; velocity is harmonized/de-trusted and acceleration/jerk remain soft event-aware priors",
                "local_segment_tuning_config": tuning_cfg.as_dict(),
            }
        piecewise_component = {
            **component.as_dict(),
            "segmentation": segmented_component.diagnostics,
            "solver": solver_summary,
            "join_velocity_harmonization": {
                "enabled": bool(tuning_cfg.join_velocity_harmonization),
                "report_count": len(join_velocity_reports),
                "reports": join_velocity_reports,
            },
            "adaptive_resegmentation": {
                "enabled": bool(tuning_cfg.adaptive_resegmentation_enabled),
                "final_refinement_depth": int(refinement_depth),
                "history": refinement_history,
            },
            "boundaries": [
                {
                    **boundary.as_dict(),
                    "shared_state": shared_states[boundary.boundary_id].as_dict(),
                    "join_constraint": (
                        "position_from_robust_shared_boundary_state_for_stable_variant_or_raw_for_legacy; velocity_soft_or_harmonized; acceleration_prior_diagnostic_only"
                        if backend in HERMITE_BACKENDS
                        else "position_hard_anchor_from_robust_shared_boundary_state_for_aviation_variants_or_raw_anchor_for_legacy; velocity_harmonized_when_available; acceleration_soft_prior_when_estimable"
                    ),
                    "harmonized_join_velocity": boundary_report_by_id.get(str(boundary.boundary_id)),
                    "event_bucket": _boundary_event_bucket(boundary),
                }
                for boundary in segmented_component.boundaries
            ],
            "segments": component_segments_meta,
            "continuity": continuity,
            "event_aware_continuity": event_aware_continuity,
        }
        return {
            "fits": fits,
            "segment_metadata": segment_metadata,
            "piecewise_component": piecewise_component,
        }

    def _refine_bad_b_spline_segments(
        self,
        *,
        segmented_component: SegmentedComponent,
        final_segment_results: list[dict[str, Any]],
        refinement_depth: int,
        method_config: TrackOutputPipelineConfig | None = None,
    ) -> tuple[SegmentedComponent | None, dict[str, Any]]:
        """Add internal boundaries to locally tuned segments that still fit badly.

        This is intentionally post-fit: the first answer is the best locally
        tuned fit under the current segmentation.  Only if that answer still has
        large position error do we split the offending segment and refit the
        whole component so all joins continue to share exact C0/C1 constraints.
        """
        cfg = method_config or self.config
        tuning = cfg.local_segment_tuning_config
        component = segmented_component.component
        report: dict[str, Any] = {
            "enabled": bool(tuning.adaptive_resegmentation_enabled),
            "refinement_depth": int(refinement_depth),
            "component_id": component.component_id,
            "initial_segment_count": len(segmented_component.segments),
            "initial_boundary_count": len(segmented_component.boundaries),
            "bad_segments": [],
            "new_boundaries": [],
            "reason": None,
        }
        if not bool(tuning.adaptive_resegmentation_enabled):
            report["reason"] = "disabled"
            return None, report
        if int(refinement_depth) >= int(tuning.adaptive_resegmentation_max_passes):
            report["reason"] = "max_passes_reached"
            return None, report
        configured_max_segments = int(cfg.dynamic_segmentation_config.max_segments_per_component)
        adaptive_max_segments_raw = getattr(tuning, "adaptive_resegmentation_max_segments_per_component", None)
        adaptive_max_segments = (
            configured_max_segments
            if adaptive_max_segments_raw is None
            else max(configured_max_segments, int(adaptive_max_segments_raw))
        )
        report["max_segments_per_component"] = int(adaptive_max_segments)
        if len(segmented_component.segments) + 1 > int(adaptive_max_segments):
            report["reason"] = "max_segments_per_component_reached"
            return None, report

        bad_results: list[dict[str, Any]] = []
        for item in final_segment_results:
            quality = item.get("quality")
            segment = item.get("segment")
            fit = item.get("fit")
            if quality is None or segment is None or fit is None:
                continue
            raw = getattr(quality, "raw_fit_metrics", {}) or {}
            rmse = float(raw.get("rmse_3d_m", 0.0))
            p95 = float(raw.get("p95_error_3d_m", rmse))
            max_err = float(raw.get("max_error_3d_m", p95))
            residual_profile = self._adaptive_resegmentation_residual_profile(
                segment=segment,
                fit=fit,
                tuning=tuning,
            )
            vertical_rmse = float(raw.get("rmse_vertical_m", residual_profile.get("rmse_vertical_m", 0.0)) or 0.0)
            vertical_p95 = float(raw.get("p95_vertical_error_m", residual_profile.get("p95_vertical_error_m", vertical_rmse)) or vertical_rmse)
            vertical_max = float(raw.get("max_vertical_error_m", residual_profile.get("max_vertical_error_m", vertical_p95)) or vertical_p95)
            vertical_window = float(residual_profile.get("max_vertical_window_error_m", 0.0) or 0.0)
            vertical_run_points = int(residual_profile.get("vertical_window_run_points", 0) or 0)
            vertical_run_duration = float(residual_profile.get("vertical_window_run_duration_s", 0.0) or 0.0)

            reasons: list[str] = []
            normalized_priorities: list[float] = []

            def add_reason(metric_value: float, threshold: float, reason: str) -> None:
                if math.isfinite(float(threshold)) and float(threshold) > 0.0:
                    normalized_priorities.append(float(metric_value) / float(threshold))
                    if float(metric_value) > float(threshold):
                        reasons.append(reason)

            add_reason(rmse, float(tuning.adaptive_resegmentation_bad_rmse_m), "rmse_high")
            add_reason(p95, float(tuning.adaptive_resegmentation_bad_p95_m), "p95_high")
            add_reason(max_err, float(tuning.adaptive_resegmentation_bad_max_m), "max_error_high")
            add_reason(vertical_rmse, float(getattr(tuning, "adaptive_resegmentation_bad_vertical_rmse_m", math.inf)), "vertical_rmse_high")
            add_reason(vertical_p95, float(getattr(tuning, "adaptive_resegmentation_bad_vertical_p95_m", math.inf)), "vertical_p95_high")
            add_reason(vertical_max, float(getattr(tuning, "adaptive_resegmentation_bad_vertical_max_m", math.inf)), "vertical_max_error_high")

            vertical_window_threshold = float(getattr(tuning, "adaptive_resegmentation_bad_vertical_window_m", math.inf))
            if math.isfinite(vertical_window_threshold) and vertical_window_threshold > 0.0:
                normalized_priorities.append(vertical_window / vertical_window_threshold)
                min_run_points = int(getattr(tuning, "adaptive_resegmentation_vertical_run_min_points", 1))
                min_run_duration = float(getattr(tuning, "adaptive_resegmentation_vertical_run_min_duration_s", 0.0))
                if (
                    vertical_window > vertical_window_threshold
                    and vertical_run_points >= max(1, min_run_points)
                    and vertical_run_duration >= max(0.0, min_run_duration)
                ):
                    reasons.append("vertical_window_error_high")

            if reasons:
                priority = float(max(normalized_priorities) if normalized_priorities else 1.0)
                item["adaptive_resegmentation_quality"] = {
                    **residual_profile,
                    "rmse_3d_m": rmse,
                    "p95_error_3d_m": p95,
                    "max_error_3d_m": max_err,
                    "rmse_vertical_m": vertical_rmse,
                    "p95_vertical_error_m": vertical_p95,
                    "max_vertical_error_m": vertical_max,
                    "reasons": list(dict.fromkeys(reasons)),
                    "priority": priority,
                }
                bad_results.append(item)
                report["bad_segments"].append(
                    {
                        "segment_id": segment.segment_id,
                        "t0": float(segment.t0),
                        "t1": float(segment.t1),
                        "n_observations": int(segment.n_observations),
                        "rmse_3d_m": rmse,
                        "p95_error_3d_m": p95,
                        "max_error_3d_m": max_err,
                        "rmse_vertical_m": vertical_rmse,
                        "p95_vertical_error_m": vertical_p95,
                        "max_vertical_error_m": vertical_max,
                        "max_vertical_window_error_m": vertical_window,
                        "vertical_window_run_points": vertical_run_points,
                        "vertical_window_run_duration_s": vertical_run_duration,
                        "reasons": list(dict.fromkeys(reasons)),
                        "priority": priority,
                    }
                )

        if not bad_results:
            report["reason"] = "all_segments_within_quality_thresholds"
            return None, report

        existing = list(segmented_component.boundaries)
        proposed = list(existing)
        max_new = max(1, int(tuning.adaptive_resegmentation_max_new_boundaries_per_pass))
        for item in sorted(
            bad_results,
            key=lambda x: -float((x.get("adaptive_resegmentation_quality") or {}).get("priority", getattr(x.get("quality"), "raw_fit_metrics", {}).get("p95_error_3d_m", 0.0))),
        ):
            if len(report["new_boundaries"]) >= max_new:
                break
            if len(proposed) + 2 > int(adaptive_max_segments):
                break
            boundary = self._choose_quality_refinement_boundary(
                component=component,
                segment=item["segment"],
                fit=item["fit"],
                existing_boundaries=proposed,
                method_config=cfg,
                quality_context=item.get("adaptive_resegmentation_quality"),
            )
            if boundary is None:
                continue
            proposed.append(boundary)
            report["new_boundaries"].append(
                {
                    "sample_index": int(boundary.sample_index),
                    "local_sample_index": int(boundary.local_sample_index),
                    "t_boundary": float(boundary.t_boundary),
                    "score": float(boundary.score),
                    "reasons": list(boundary.reasons),
                }
            )

        if len(proposed) == len(existing):
            report["reason"] = "bad_segments_found_but_no_feasible_internal_boundary"
            return None, report

        renumbered: list[AcceptedBoundary] = []
        for idx, boundary in enumerate(sorted(proposed, key=lambda b: int(b.local_sample_index)), start=1):
            reasons = list(boundary.reasons)
            if boundary.boundary_id not in {b.boundary_id for b in existing}:
                reasons.append("adaptive_quality_resegmentation")
            renumbered.append(
                AcceptedBoundary(
                    boundary_id=f"{component.component_id}_bnd_{idx:04d}",
                    component_id=component.component_id,
                    sample_index=int(boundary.sample_index),
                    local_sample_index=int(boundary.local_sample_index),
                    t_boundary=float(boundary.t_boundary),
                    reasons=tuple(dict.fromkeys(reasons)),
                    score=float(boundary.score),
                    is_hard_gap=bool(boundary.is_hard_gap),
                )
            )

        report["reason"] = "quality_triggered_resegmentation_applied"
        report["final_boundary_count"] = len(renumbered)
        report["final_segment_count"] = len(renumbered) + 1
        refined = build_segmented_component_from_boundaries(
            component,
            renumbered,
            cfg.dynamic_segmentation_config,
            diagnostics_extra={"adaptive_resegmentation_last_pass": report},
        )
        return refined, report

    def _choose_quality_refinement_boundary(
        self,
        *,
        component: HardGapComponent,
        segment: DynamicSegment,
        fit: Any,
        existing_boundaries: list[AcceptedBoundary],
        method_config: TrackOutputPipelineConfig | None = None,
        quality_context: dict[str, Any] | None = None,
    ) -> AcceptedBoundary | None:
        cfg = method_config or self.config
        tuning = cfg.local_segment_tuning_config
        min_points = max(3, int(tuning.adaptive_resegmentation_min_points))
        min_duration = max(0.0, float(tuning.adaptive_resegmentation_min_duration_s))
        min_spacing = max(0.0, float(tuning.adaptive_resegmentation_min_boundary_spacing_s))
        reasons_from_quality = tuple(str(r) for r in ((quality_context or {}).get("reasons") or ()))
        prefer_vertical = any("vertical" in r for r in reasons_from_quality)

        def make_boundary(sample_index: int, score: float, reasons: tuple[str, ...]) -> AcceptedBoundary | None:
            local = int(sample_index) - int(component.start_sample_index)
            if not self._boundary_set_is_feasible_for_component(
                component,
                [*existing_boundaries],
                candidate_local=local,
                min_points=min_points,
                min_duration_s=min_duration,
                min_spacing_s=min_spacing,
            ):
                return None
            sample = component.samples[local]
            return AcceptedBoundary(
                boundary_id=f"{component.component_id}_quality_refine_{sample_index}",
                component_id=component.component_id,
                sample_index=int(sample_index),
                local_sample_index=int(local),
                t_boundary=float(sample.t),
                reasons=reasons,
                score=float(score),
                is_hard_gap=False,
            )

        # First try the real segmenter again, but only within the offending
        # segment and with more permissive thresholds.  This keeps refinement
        # grounded in motion/energy-state changes when such a change is visible.
        if bool(tuning.adaptive_resegmentation_use_feature_segmentation) and not prefer_vertical:
            try:
                local_component = HardGapComponent(
                    component_id=component.component_id,
                    start_sample_index=int(segment.start_sample_index),
                    end_sample_index=int(segment.end_sample_index),
                    samples=tuple(segment.samples),
                    hard_gap_before_s=None,
                    hard_gap_after_s=None,
                )
                refine_cfg = replace(
                    cfg.dynamic_segmentation_config,
                    enabled=True,
                    min_segment_points=min_points,
                    min_segment_duration_s=min_duration,
                    min_boundary_spacing_s=min_spacing,
                    energy_state_min_points=max(3, min_points - 1),
                    energy_state_min_duration_s=max(0.0, min_duration * 0.75),
                    enable_motion_spike_boundaries=True,
                    candidate_score_z=min(3.0, float(cfg.dynamic_segmentation_config.candidate_score_z)),
                    prefer_under_segmentation=False,
                    max_segments_per_component=2,
                )
                sub = segment_component(local_component, refine_cfg)
                for b in sorted(sub.boundaries, key=lambda x: (-float(x.score), int(x.sample_index))):
                    boundary = make_boundary(
                        int(b.sample_index),
                        max(float(b.score), 100.0),
                        tuple([*b.reasons, "adaptive_resegmentation_feature_split"]),
                    )
                    if boundary is not None:
                        return boundary
            except Exception:
                pass

        # Fallback: split at the largest raw-position residual inside the bad
        # segment, excluding endpoints and infeasible low-support positions.
        samples = list(segment.samples)
        if len(samples) < 2 * min_points - 1:
            return None
        t = np.asarray([s.t for s in samples], dtype=float)
        y = np.asarray([s.y for s in samples], dtype=float)
        try:
            y_hat = fit.evaluate(t, deriv=0)
        except Exception:
            return None
        delta = np.asarray(y_hat, dtype=float) - y
        residual = np.linalg.norm(delta, axis=1)
        window = max(1, int(tuning.adaptive_resegmentation_residual_window_points))
        if window > 1 and residual.size >= window:
            residual_3d_score = _rolling_median_1d(residual, window)
        else:
            residual_3d_score = residual

        prefer_vertical_residual = bool(prefer_vertical) and delta.ndim == 2 and delta.shape[1] >= 3
        if prefer_vertical_residual:
            vertical_abs = np.abs(delta[:, 2])
            vertical_score = _rolling_median_1d(vertical_abs, window) if window > 1 and vertical_abs.size >= window else vertical_abs
            residual_score = vertical_score + 0.05 * residual_3d_score
            fallback_reasons = ("adaptive_resegmentation_vertical_residual_peak",)
        else:
            residual_score = residual_3d_score
            fallback_reasons = ("adaptive_resegmentation_residual_peak",)

        candidates: list[tuple[float, int]] = []
        for local_in_segment in range(1, len(samples) - 1):
            global_sample_index = int(segment.start_sample_index) + int(local_in_segment)
            component_local = global_sample_index - int(component.start_sample_index)
            if not self._boundary_set_is_feasible_for_component(
                component,
                [*existing_boundaries],
                candidate_local=component_local,
                min_points=min_points,
                min_duration_s=min_duration,
                min_spacing_s=min_spacing,
            ):
                continue
            candidates.append((float(residual_score[local_in_segment]), global_sample_index))
        if not candidates:
            return None
        score, sample_index = max(candidates, key=lambda x: (x[0], -abs(x[1] - int(segment.start_sample_index))))
        return make_boundary(
            sample_index,
            max(100.0, float(score)),
            fallback_reasons,
        )

    @staticmethod
    def _adaptive_resegmentation_residual_profile(
        *,
        segment: DynamicSegment,
        fit: Any,
        tuning: LocalSegmentTuningConfig,
    ) -> dict[str, Any]:
        samples = list(segment.samples)
        if not samples:
            return {}
        t = np.asarray([s.t for s in samples], dtype=float)
        y = np.asarray([s.y for s in samples], dtype=float)
        try:
            y_hat = np.asarray(fit.evaluate(t, deriv=0), dtype=float)
        except Exception:
            return {"enabled": False, "reason": "fit_evaluate_failed"}
        if y_hat.shape != y.shape or y.ndim != 2:
            return {"enabled": False, "reason": "invalid_residual_shape"}

        delta = y_hat - y
        e3 = np.linalg.norm(delta, axis=1)
        vertical = np.abs(delta[:, 2]) if delta.shape[1] >= 3 else np.zeros(delta.shape[0], dtype=float)
        window = max(1, int(getattr(tuning, "adaptive_resegmentation_residual_window_points", 5)))
        vertical_window = _rolling_median_1d(vertical, window) if window > 1 and vertical.size >= window else vertical
        residual_window = _rolling_median_1d(e3, window) if window > 1 and e3.size >= window else e3

        def safe_q(values: np.ndarray, quantile: float) -> float:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.quantile(arr, quantile)) if arr.size else 0.0

        max_vertical_idx = int(np.nanargmax(vertical)) if vertical.size else 0
        max_vertical_window_idx = int(np.nanargmax(vertical_window)) if vertical_window.size else 0
        threshold = float(getattr(tuning, "adaptive_resegmentation_bad_vertical_window_m", math.inf))
        run = _longest_threshold_run(t, vertical_window, threshold) if math.isfinite(threshold) else {}

        return {
            "enabled": True,
            "residual_window_points": int(window),
            "rmse_3d_m": float(math.sqrt(float(np.mean(e3**2)))) if e3.size else 0.0,
            "p95_error_3d_m": safe_q(e3, 0.95),
            "max_error_3d_m": float(np.nanmax(e3)) if e3.size else 0.0,
            "rmse_vertical_m": float(math.sqrt(float(np.mean(vertical**2)))) if vertical.size else 0.0,
            "p95_vertical_error_m": safe_q(vertical, 0.95),
            "max_vertical_error_m": float(np.nanmax(vertical)) if vertical.size else 0.0,
            "max_vertical_error_local_in_segment": int(max_vertical_idx),
            "max_vertical_error_sample_index": int(segment.start_sample_index) + int(max_vertical_idx),
            "max_vertical_error_t": float(t[max_vertical_idx]) if t.size else None,
            "max_vertical_window_error_m": float(np.nanmax(vertical_window)) if vertical_window.size else 0.0,
            "max_vertical_window_local_in_segment": int(max_vertical_window_idx),
            "max_vertical_window_sample_index": int(segment.start_sample_index) + int(max_vertical_window_idx),
            "max_vertical_window_t": float(t[max_vertical_window_idx]) if t.size else None,
            "max_3d_window_error_m": float(np.nanmax(residual_window)) if residual_window.size else 0.0,
            "vertical_window_run_points": int(run.get("n_points", 0) or 0),
            "vertical_window_run_duration_s": float(run.get("duration_s", 0.0) or 0.0),
            "vertical_window_run_start_t": run.get("start_t"),
            "vertical_window_run_end_t": run.get("end_t"),
        }

    @staticmethod
    def _boundary_set_is_feasible_for_component(
        component: HardGapComponent,
        boundaries: list[AcceptedBoundary],
        *,
        candidate_local: int,
        min_points: int,
        min_duration_s: float,
        min_spacing_s: float,
    ) -> bool:
        t = np.asarray([s.t for s in component.samples], dtype=float)
        n = len(t)
        if candidate_local <= 0 or candidate_local >= n - 1:
            return False
        locals_existing = [int(b.local_sample_index) for b in boundaries]
        if candidate_local in locals_existing:
            return False
        locals_all = sorted([*locals_existing, int(candidate_local)])
        if any(b <= 0 or b >= n - 1 for b in locals_all):
            return False
        if any((float(t[b]) - float(t[a])) < float(min_spacing_s) for a, b in zip(locals_all[:-1], locals_all[1:])):
            return False
        starts = [0] + locals_all
        ends = locals_all + [n - 1]
        for start, end in zip(starts, ends):
            if (int(end) - int(start) + 1) < int(min_points):
                return False
            if (float(t[end]) - float(t[start])) < float(min_duration_s):
                return False
        return True

    # ------------------------------------------------------------------
    # Per-flight debug helpers
    # ------------------------------------------------------------------

    def _debug_config_snapshot(self, rule: TrackRuleConfig) -> dict[str, Any]:
        cfg = self.config
        return {
            "purpose": "reproducibility snapshot for one flight build",
            "rule": rule.to_dict(),
            "paths": asdict(cfg.paths),
            "log_dir": cfg.log_dir,
            "icao_list": cfg.icao_list,
            "v_spline_time_step_s": cfg.v_spline_time_step_s,
            "v_spline_output_frequency_hz": cfg.v_spline_output_frequency_hz,
            "dynamic_segmentation_config": asdict(cfg.dynamic_segmentation_config),
            "boundary_state_config": asdict(cfg.boundary_state_config),
            "local_segment_policy_config": asdict(cfg.local_segment_policy_config),
            "local_segment_tuning_config": cfg.local_segment_tuning_config.as_dict(),
            "v_spline_output_backends": list(cfg.v_spline_output_backends),
            "v_spline_output_presets": list(cfg.v_spline_output_presets),
            "use_kalman_boundary_prior": bool(cfg.use_kalman_boundary_prior),
            "event_aware_evaluation_enabled": bool(cfg.event_aware_evaluation_enabled),
            "holdout_evaluation_fraction": float(cfg.holdout_evaluation_fraction),
            "synthetic_gap_holdout_enabled": bool(cfg.synthetic_gap_holdout_enabled),
            "synthetic_gap_holdout_methods": list(cfg.synthetic_gap_holdout_methods),
            "synthetic_gap_holdout_gap_count": int(cfg.synthetic_gap_holdout_gap_count),
            "synthetic_gap_holdout_fraction": float(cfg.synthetic_gap_holdout_fraction),
            "synthetic_gap_holdout_min_gap_s": float(cfg.synthetic_gap_holdout_min_gap_s),
            "synthetic_gap_holdout_max_gap_s": float(cfg.synthetic_gap_holdout_max_gap_s),
            "synthetic_gap_holdout_guard_s": float(cfg.synthetic_gap_holdout_guard_s),
            "synthetic_gap_holdout_seed": int(cfg.synthetic_gap_holdout_seed),
            "kalman_rts_output_enabled": bool(cfg.kalman_rts_output_enabled),
            "kalman_rts_output_presets": list(cfg.kalman_rts_output_presets),
            "bspline_config": asdict(cfg.bspline_config),
            "hermite_config": asdict(cfg.hermite_config),
            "kalman_rts_config": asdict(cfg.kalman_rts_config),
            "bspline_config_by_preset": {k: asdict(v) for k, v in cfg.bspline_config_by_preset.items()},
            "hermite_config_by_preset": {k: asdict(v) for k, v in cfg.hermite_config_by_preset.items()},
            "kalman_rts_config_by_preset": {k: asdict(v) for k, v in cfg.kalman_rts_config_by_preset.items()},
            "local_segment_policy_config_by_preset": {k: asdict(v) for k, v in cfg.local_segment_policy_config_by_preset.items()},
            "local_segment_tuning_config_by_preset": {k: v.as_dict() for k, v in cfg.local_segment_tuning_config_by_preset.items()},
            "adapter_config": asdict(cfg.adapter_config),
        }

    def _debug_segmentation_payload(
        self,
        segmented_components: list[SegmentedComponent],
        segmentation_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "segmentation_diagnostics": segmentation_diagnostics,
            "component_count": len(segmented_components),
            "segment_count": int(sum(len(c.segments) for c in segmented_components)),
            "boundary_count": int(sum(len(c.boundaries) for c in segmented_components)),
            "components": [
                {
                    **component.component.as_dict(),
                    "diagnostics": component.diagnostics,
                    "boundaries": [b.as_dict() for b in component.boundaries],
                    "segments": [s.as_dict() for s in component.segments],
                }
                for component in segmented_components
            ],
        }

    def _debug_boundary_state_payload(
        self,
        component_contexts: list[tuple[SegmentedComponent, dict[str, Any]]],
    ) -> dict[str, Any]:
        components = []
        for segmented_component, states in component_contexts:
            components.append(
                {
                    "component_id": segmented_component.component.component_id,
                    "boundary_count": len(segmented_component.boundaries),
                    "state_count": len(states),
                    "states": {
                        str(boundary_id): state.as_dict() if hasattr(state, "as_dict") else _clean_json(state)
                        for boundary_id, state in states.items()
                    },
                }
            )
        return {"component_count": len(components), "components": components}

    def _debug_reconstruction_quality(self, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
        methods: dict[str, Any] = {}
        for method_id, payload in payloads.items():
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
            piecewise = quality.get("piecewise") if isinstance(quality.get("piecewise"), dict) else {}
            global_quality = piecewise.get("global_quality") if isinstance(piecewise.get("global_quality"), dict) else {}
            trajectory_eval = quality.get("trajectory_model_evaluation") if isinstance(quality.get("trajectory_model_evaluation"), dict) else {}
            methods[method_id] = {
                "basis": quality.get("basis"),
                "objective": quality.get("objective"),
                "fit_mode": quality.get("fit_mode"),
                "paired_sample_count": quality.get("paired_sample_count"),
                "render_keyframe_count": quality.get("render_keyframe_count"),
                "segment_count": quality.get("segment_count"),
                "global_quality": global_quality,
                "trajectory_model_evaluation": trajectory_eval,
                "trajectory_model_weighted_score_0_100": trajectory_eval.get("weighted_score_0_100"),
                "component_count": len(piecewise.get("connected_components") or []),
                "continuity": {
                    "position_continuity_ok": global_quality.get("position_continuity_ok"),
                    "velocity_continuity_ok": global_quality.get("velocity_continuity_ok"),
                    "max_internal_position_jump_m": global_quality.get("max_internal_position_jump_m"),
                    "max_internal_velocity_jump_mps": global_quality.get("max_internal_velocity_jump_mps"),
                    "max_internal_acceleration_jump_mps2": global_quality.get("max_internal_acceleration_jump_mps2"),
                },
                "adaptive_resegmentation": [
                    comp.get("adaptive_resegmentation")
                    for comp in piecewise.get("connected_components") or []
                    if isinstance(comp, dict) and comp.get("adaptive_resegmentation") is not None
                ],
            }
        return {"method_count": len(methods), "methods": methods}

    def _debug_segment_rows(self, payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for method_id, payload in payloads.items():
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
            for seg in quality.get("segments") or []:
                if not isinstance(seg, dict):
                    continue
                q = seg.get("quality") if isinstance(seg.get("quality"), dict) else {}
                raw = q.get("raw_fit_metrics") if isinstance(q.get("raw_fit_metrics"), dict) else {}
                motion = q.get("motion_metrics") if isinstance(q.get("motion_metrics"), dict) else {}
                continuity = q.get("continuity_metrics") if isinstance(q.get("continuity_metrics"), dict) else {}
                trajectory = q.get("trajectory_model_metrics") if isinstance(q.get("trajectory_model_metrics"), dict) else {}
                trajectory_scores = trajectory.get("component_scores_0_100") if isinstance(trajectory.get("component_scores_0_100"), dict) else {}
                velocity_evidence = trajectory.get("velocity_evidence") if isinstance(trajectory.get("velocity_evidence"), dict) else {}
                fd_kinematics = trajectory.get("finite_difference_kinematics") if isinstance(trajectory.get("finite_difference_kinematics"), dict) else {}
                smoothness_model = trajectory.get("smoothness") if isinstance(trajectory.get("smoothness"), dict) else {}
                plausibility = trajectory.get("physical_plausibility") if isinstance(trajectory.get("physical_plausibility"), dict) else {}
                detail = trajectory.get("dynamic_detail_preservation") if isinstance(trajectory.get("dynamic_detail_preservation"), dict) else {}
                closure = trajectory.get("derivative_closure") if isinstance(trajectory.get("derivative_closure"), dict) else {}
                selected = seg.get("selected_params") if isinstance(seg.get("selected_params"), dict) else {}
                selected_core = seg.get("selected_core_config") if isinstance(seg.get("selected_core_config"), dict) else {}
                local_tuning = seg.get("local_tuning") if isinstance(seg.get("local_tuning"), dict) else {}
                score = local_tuning.get("selected_score") if isinstance(local_tuning.get("selected_score"), dict) else {}
                score_components = score.get("components") if isinstance(score.get("components"), dict) else {}
                diagnostics = seg.get("diagnostics") if isinstance(seg.get("diagnostics"), dict) else {}
                solver = diagnostics.get("solver") if isinstance(diagnostics.get("solver"), dict) else {}
                rows.append(
                    {
                        "method_id": method_id,
                        "segment_id": seg.get("segment_id"),
                        "component_id": seg.get("component_id"),
                        "regime_label": seg.get("regime_label"),
                        "t0": seg.get("t0"),
                        "t1": seg.get("t1"),
                        "duration_s": (float(seg["t1"]) - float(seg["t0"])) if seg.get("t0") is not None and seg.get("t1") is not None else None,
                        "n_observations": seg.get("n_observations"),
                        "start_sample_index": seg.get("start_sample_index"),
                        "end_sample_index": seg.get("end_sample_index"),
                        "start_boundary_id": seg.get("start_boundary_id"),
                        "end_boundary_id": seg.get("end_boundary_id"),
                        "rmse_3d_m": raw.get("rmse_3d_m"),
                        "rmse_horizontal_m": raw.get("rmse_horizontal_m"),
                        "rmse_vertical_m": raw.get("rmse_vertical_m"),
                        "median_error_3d_m": raw.get("median_error_3d_m"),
                        "p95_error_3d_m": raw.get("p95_error_3d_m"),
                        "max_error_3d_m": raw.get("max_error_3d_m"),
                        "median_horizontal_error_m": raw.get("median_horizontal_error_m"),
                        "p95_horizontal_error_m": raw.get("p95_horizontal_error_m"),
                        "max_horizontal_error_m": raw.get("max_horizontal_error_m"),
                        "median_vertical_error_m": raw.get("median_vertical_error_m"),
                        "p95_vertical_error_m": raw.get("p95_vertical_error_m"),
                        "max_vertical_error_m": raw.get("max_vertical_error_m"),
                        "accel_rms_mps2": motion.get("accel_rms_mps2"),
                        "accel_p95_mps2": motion.get("accel_p95_mps2"),
                        "accel_max_mps2": motion.get("accel_max_mps2"),
                        "jerk_rms_mps3": motion.get("jerk_rms_mps3"),
                        "jerk_p95_mps3": motion.get("jerk_p95_mps3"),
                        "jerk_max_mps3": motion.get("jerk_max_mps3"),
                        "endpoint_constraint_error_m": continuity.get("endpoint_position_constraint_error_m"),
                        "trajectory_model_score_0_100": trajectory.get("weighted_score_0_100"),
                        "trajectory_model_interpretation": trajectory.get("interpretation"),
                        "trajectory_truth_data_used": trajectory.get("truth_data_used"),
                        "trajectory_regime_bucket": trajectory.get("regime_bucket"),
                        "trajectory_position_score_0_100": trajectory_scores.get("observation_position_score"),
                        "trajectory_velocity_evidence_score_0_100": trajectory_scores.get("velocity_evidence_score"),
                        "trajectory_fd_kinematics_score_0_100": trajectory_scores.get("finite_difference_kinematics_score"),
                        "trajectory_smoothness_score_0_100": trajectory_scores.get("trajectory_smoothness_score"),
                        "trajectory_plausibility_score_0_100": trajectory_scores.get("physical_plausibility_score"),
                        "trajectory_detail_score_0_100": trajectory_scores.get("dynamic_detail_preservation_score"),
                        "trajectory_derivative_closure_score_0_100": trajectory_scores.get("derivative_closure_score"),
                        "velocity_evidence_weighted_rmse_3d_mps": velocity_evidence.get("weighted_rmse_3d_mps"),
                        "velocity_evidence_weighted_p95_3d_mps": velocity_evidence.get("weighted_p95_error_3d_mps"),
                        "fd_velocity_rmse_3d_mps": fd_kinematics.get("finite_difference_velocity_rmse_3d_mps"),
                        "fd_track_angle_median_error_deg": fd_kinematics.get("track_angle_median_error_deg"),
                        "jerk_per_speed_rms_1_s2": smoothness_model.get("jerk_per_speed_rms_1_s2"),
                        "snap_proxy_rms_mps4": smoothness_model.get("snap_proxy_rms_mps4"),
                        "turn_rate_p95_deg_s": smoothness_model.get("turn_rate_p95_deg_s"),
                        "curvature_p95_1_per_m": smoothness_model.get("curvature_p95_1_per_m"),
                        "physical_accel_p95_mps2": plausibility.get("accel_p95_mps2"),
                        "velocity_detail_retention_ratio": detail.get("velocity_detail_retention_ratio"),
                        "position_velocity_closure_rms_mps": closure.get("position_velocity_closure_rms_mps"),
                        "velocity_acceleration_closure_rms_mps2": closure.get("velocity_acceleration_closure_rms_mps2"),
                        "selected_score": score.get("score"),
                        "position_cost_m": score_components.get("position_cost_m"),
                        "motion_cost_m_equivalent": score_components.get("motion_cost_m_equivalent"),
                        "complexity_cost_m_equivalent": score_components.get("complexity_cost_m_equivalent"),
                        "selected_reason": selected.get("reason"),
                        "adaptive_eta": selected.get("adaptive_eta"),
                        "smoothing_lambda": selected.get("smoothing_lambda"),
                        "velocity_weight": selected.get("velocity_weight"),
                        "knot_spacing_s": selected.get("knot_spacing_s"),
                        "min_observations_per_basis": selected.get("min_observations_per_basis"),
                        "jerk_penalty_weight": selected.get("jerk_penalty_weight"),
                        "boundary_acceleration_prior_weight": selected.get("boundary_acceleration_prior_weight"),
                        "effective_velocity_weight": selected_core.get("velocity_weight"),
                        "effective_velocity_outlier_gate_mps": selected_core.get("velocity_outlier_gate_mps"),
                        "effective_adaptive_speed_floor_mps": selected_core.get("adaptive_speed_floor_mps"),
                        "effective_min_observations_per_basis": selected_core.get("min_observations_per_basis"),
                        "effective_jerk_penalty_weight": selected_core.get("jerk_penalty_weight"),
                        "effective_snap_penalty_weight": selected_core.get("snap_penalty_weight"),
                        "effective_boundary_velocity_prior_weight": selected_core.get("boundary_velocity_prior_weight"),
                        "effective_boundary_acceleration_prior_weight": selected_core.get("boundary_acceleration_prior_weight"),
                        "effective_hard_component_endpoint_velocities": selected_core.get("hard_component_endpoint_velocities"),
                        "effective_endpoint_guard_window_s": selected_core.get("endpoint_guard_window_s"),
                        "effective_endpoint_jerk_penalty_multiplier": selected_core.get("endpoint_jerk_penalty_multiplier"),
                        "effective_endpoint_snap_penalty_multiplier": selected_core.get("endpoint_snap_penalty_multiplier"),
                        "candidate_count": local_tuning.get("candidate_count"),
                        "successful_candidate_count": local_tuning.get("successful_candidate_count"),
                        "n_basis": diagnostics.get("n_basis"),
                        "condition_number_hessian": solver.get("condition_number_hessian"),
                    }
                )
        return rows

    def _debug_join_rows(self, payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for method_id, payload in payloads.items():
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
            piecewise = quality.get("piecewise") if isinstance(quality.get("piecewise"), dict) else {}
            for comp in piecewise.get("connected_components") or []:
                if not isinstance(comp, dict):
                    continue
                component_id = comp.get("component_id")
                continuity = comp.get("continuity") if isinstance(comp.get("continuity"), dict) else {}
                continuity_boundaries = continuity.get("boundaries") if isinstance(continuity.get("boundaries"), list) else []
                join_reports_by_boundary: dict[str, dict[str, Any]] = {}
                for boundary in comp.get("boundaries") or []:
                    if isinstance(boundary, dict) and isinstance(boundary.get("harmonized_join_velocity"), dict):
                        join_reports_by_boundary[str(boundary.get("boundary_id"))] = boundary["harmonized_join_velocity"]
                for idx, boundary in enumerate(comp.get("boundaries") or []):
                    if not isinstance(boundary, dict):
                        continue
                    boundary_id = str(boundary.get("boundary_id"))
                    shared = boundary.get("shared_state") if isinstance(boundary.get("shared_state"), dict) else {}
                    join = join_reports_by_boundary.get(boundary_id, {})
                    cont = continuity_boundaries[idx] if idx < len(continuity_boundaries) and isinstance(continuity_boundaries[idx], dict) else {}
                    hv = join.get("harmonized_velocity_mps") if isinstance(join.get("harmonized_velocity_mps"), list) else [None, None, None]
                    rows.append(
                        {
                            "method_id": method_id,
                            "component_id": component_id,
                            "boundary_id": boundary_id,
                            "sample_index": boundary.get("sample_index"),
                            "local_sample_index": boundary.get("local_sample_index"),
                            "t_boundary": boundary.get("t_boundary"),
                            "reasons": list(boundary.get("reasons") or []),
                            "position_jump_m": cont.get("position_jump_m"),
                            "velocity_jump_mps": cont.get("velocity_jump_mps"),
                            "acceleration_jump_mps2": cont.get("acceleration_jump_mps2"),
                            "jerk_jump_mps3": cont.get("jerk_jump_mps3"),
                            "position_continuity_ok": cont.get("position_continuity_ok"),
                            "velocity_continuity_ok": cont.get("velocity_continuity_ok"),
                            "shared_state_confidence": shared.get("confidence"),
                            "shared_state_method": shared.get("method"),
                            "harmonized_velocity_east_mps": hv[0] if len(hv) > 0 else None,
                            "harmonized_velocity_north_mps": hv[1] if len(hv) > 1 else None,
                            "harmonized_velocity_up_mps": hv[2] if len(hv) > 2 else None,
                            "left_prefit_rmse_3d_m": join.get("left_prefit_rmse_3d_m"),
                            "right_prefit_rmse_3d_m": join.get("right_prefit_rmse_3d_m"),
                            "velocity_clip": join.get("clip"),
                        }
                    )
        return rows

    def _debug_trajectory_model_rows(self, payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Return method/segment rows for reference-free trajectory-model scoring."""
        rows: list[dict[str, Any]] = []
        for method_id, payload in payloads.items():
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
            evaluation = quality.get("trajectory_model_evaluation") if isinstance(quality.get("trajectory_model_evaluation"), dict) else {}
            method_scores = evaluation.get("method_component_scores_0_100") if isinstance(evaluation.get("method_component_scores_0_100"), dict) else {}
            mean_scores = evaluation.get("component_mean_scores_0_100") if isinstance(evaluation.get("component_mean_scores_0_100"), dict) else {}
            gap = evaluation.get("hard_gap_honesty") if isinstance(evaluation.get("hard_gap_honesty"), dict) else {}
            join = evaluation.get("event_aware_join_score") if isinstance(evaluation.get("event_aware_join_score"), dict) else {}
            locality = evaluation.get("locality_scope") if isinstance(evaluation.get("locality_scope"), dict) else {}
            rows.append(
                {
                    "row_type": "method_summary",
                    "method_id": method_id,
                    "preset": quality.get("preset"),
                    "backend": quality.get("reconstruction_backend"),
                    "fit_mode": quality.get("fit_mode"),
                    "truth_data_used": evaluation.get("truth_data_used"),
                    "trajectory_model_weighted_score_0_100": evaluation.get("weighted_score_0_100"),
                    "trajectory_model_interpretation": evaluation.get("interpretation"),
                    "segment_trajectory_score_0_100": method_scores.get("segment_trajectory_score"),
                    "event_aware_join_score_0_100": method_scores.get("event_aware_join_score"),
                    "hard_gap_honesty_score_0_100": method_scores.get("hard_gap_honesty_score"),
                    "locality_scope_score_0_100": method_scores.get("locality_scope_score"),
                    "mean_position_score_0_100": mean_scores.get("observation_position_score"),
                    "mean_velocity_evidence_score_0_100": mean_scores.get("velocity_evidence_score"),
                    "mean_fd_kinematics_score_0_100": mean_scores.get("finite_difference_kinematics_score"),
                    "mean_smoothness_score_0_100": mean_scores.get("trajectory_smoothness_score"),
                    "mean_plausibility_score_0_100": mean_scores.get("physical_plausibility_score"),
                    "mean_detail_score_0_100": mean_scores.get("dynamic_detail_preservation_score"),
                    "mean_derivative_closure_score_0_100": mean_scores.get("derivative_closure_score"),
                    "hard_gap_count": gap.get("hard_gap_count"),
                    "bridged_gap_count": gap.get("bridged_gap_count"),
                    "bridge_frame_count": gap.get("bridge_frame_count"),
                    "normal_join_count": join.get("normal_join_count"),
                    "normal_max_acceleration_jump_mps2": join.get("normal_max_acceleration_jump_mps2"),
                    "normal_max_jerk_jump_mps3": join.get("normal_max_jerk_jump_mps3"),
                    "locality_note": locality.get("note"),
                }
            )
            for seg in quality.get("segments") or []:
                if not isinstance(seg, dict):
                    continue
                q = seg.get("quality") if isinstance(seg.get("quality"), dict) else {}
                metrics = q.get("trajectory_model_metrics") if isinstance(q.get("trajectory_model_metrics"), dict) else {}
                if not metrics:
                    continue
                scores = metrics.get("component_scores_0_100") if isinstance(metrics.get("component_scores_0_100"), dict) else {}
                vel = metrics.get("velocity_evidence") if isinstance(metrics.get("velocity_evidence"), dict) else {}
                fd = metrics.get("finite_difference_kinematics") if isinstance(metrics.get("finite_difference_kinematics"), dict) else {}
                smooth = metrics.get("smoothness") if isinstance(metrics.get("smoothness"), dict) else {}
                detail = metrics.get("dynamic_detail_preservation") if isinstance(metrics.get("dynamic_detail_preservation"), dict) else {}
                rows.append(
                    {
                        "row_type": "segment",
                        "method_id": method_id,
                        "preset": quality.get("preset"),
                        "backend": quality.get("reconstruction_backend"),
                        "segment_id": seg.get("segment_id"),
                        "component_id": seg.get("component_id"),
                        "regime_label": seg.get("regime_label"),
                        "trajectory_regime_bucket": metrics.get("regime_bucket"),
                        "t0": seg.get("t0"),
                        "t1": seg.get("t1"),
                        "n_observations": seg.get("n_observations"),
                        "truth_data_used": metrics.get("truth_data_used"),
                        "trajectory_model_score_0_100": metrics.get("weighted_score_0_100"),
                        "position_score_0_100": scores.get("observation_position_score"),
                        "velocity_evidence_score_0_100": scores.get("velocity_evidence_score"),
                        "fd_kinematics_score_0_100": scores.get("finite_difference_kinematics_score"),
                        "smoothness_score_0_100": scores.get("trajectory_smoothness_score"),
                        "plausibility_score_0_100": scores.get("physical_plausibility_score"),
                        "detail_score_0_100": scores.get("dynamic_detail_preservation_score"),
                        "derivative_closure_score_0_100": scores.get("derivative_closure_score"),
                        "velocity_weighted_rmse_3d_mps": vel.get("weighted_rmse_3d_mps"),
                        "fd_velocity_rmse_3d_mps": fd.get("finite_difference_velocity_rmse_3d_mps"),
                        "track_angle_median_error_deg": fd.get("track_angle_median_error_deg"),
                        "jerk_per_speed_rms_1_s2": smooth.get("jerk_per_speed_rms_1_s2"),
                        "snap_proxy_rms_mps4": smooth.get("snap_proxy_rms_mps4"),
                        "turn_rate_p95_deg_s": smooth.get("turn_rate_p95_deg_s"),
                        "velocity_detail_retention_ratio": detail.get("velocity_detail_retention_ratio"),
                    }
                )
        return rows

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _build_minimal_method_payload(self, payload: dict[str, Any], *, detailed_file: str) -> dict[str, Any]:
        """Return the compact method JSON consumed by the viewer.

        The detailed method JSON remains the trace/debug artifact.  This payload
        keeps only the fields read by ``adsb_viewer`` to build raw points,
        animation samples, track lines, and vectors.
        """
        method = payload.get("method") if isinstance(payload.get("method"), dict) else {}
        method_id = str(method.get("id") or payload.get("method_id") or "")
        raw_keyframes = [
            self._minimal_keyframe(kf, include_link_fields=True)
            for kf in payload.get("raw_keyframes") or []
            if isinstance(kf, dict)
        ]
        render_keyframes = [
            self._minimal_keyframe(kf, include_link_fields=True)
            for kf in payload.get("render_keyframes") or []
            if isinstance(kf, dict)
        ]

        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        minimal_quality = {
            "method": quality.get("method") or method_id or None,
            "raw_keyframe_count": len(raw_keyframes),
            "render_keyframe_count": len(render_keyframes),
        }
        for key in (
            "paired_keyframe_count",
            "paired_sample_count",
            "segment_count",
            "fit_mode",
            "fit_time_start",
            "fit_time_end",
            "time_step_s",
            "output_frequency_hz",
            "tuning",
        ):
            if key in quality:
                minimal_quality[key] = quality.get(key)

        out: dict[str, Any] = {
            "schema_version": payload.get("schema_version"),
            "minimal_payload_version": MINIMAL_PAYLOAD_VERSION,
            "detail_level": "viewer_minimal",
            "detailed_file": detailed_file,
            "track_id": payload.get("track_id"),
            "icao": payload.get("icao"),
            "method": {
                "id": method_id or None,
                "label": method.get("label") or method_id or None,
            },
            "raw_keyframes": raw_keyframes,
            "render_keyframes": render_keyframes,
            "display": self._minimal_display(payload.get("display")),
            "quality": minimal_quality,
        }

        piecewise_ref = self._minimal_piecewise_reference(payload, render_keyframes)
        if piecewise_ref:
            # Keep segment plotting metadata lightweight and viewer-friendly.
            # Heavy diagnostics, feature matrices, and fitting internals stay in
            # the detailed method JSON referenced by ``detailed_file``.
            out["piecewise"] = piecewise_ref.get("piecewise")
            out["segments"] = piecewise_ref.get("segments", [])
            out["segment_boundaries"] = piecewise_ref.get("segment_boundaries", [])

        return _clean_json(out)

    def _minimal_keyframe(self, kf: dict[str, Any], *, include_link_fields: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": kf.get("id"),
            "t": kf.get("t"),
            "position": self._minimal_position(kf.get("position")),
            "velocity": self._minimal_velocity(kf.get("velocity")),
            "acceleration": self._minimal_acceleration(kf.get("acceleration")),
        }
        if include_link_fields:
            for key in (
                "source_keyframe_id",
                "segment_id",
                "row_ids",
                "event_kind",
            ):
                if key in kf:
                    out[key] = kf.get(key)
        return {k: v for k, v in out.items() if v is not None}

    @staticmethod
    def _minimal_position(position: Any) -> dict[str, Any] | None:
        if not isinstance(position, dict):
            return None
        selected = _select_present(
            position,
            (
                "source",
                "frame",
                "lat",
                "lon",
                "x_m",
                "y_m",
                "z_m",
                "altitude_ft_msl",
                "altitude_m_msl",
                "height_above_field_m",
                "height_above_reference_m",
                "z_reference",
                "vertical_reference_ft_msl",
                "interpolated",
            ),
        )
        return selected or None

    @staticmethod
    def _minimal_velocity(velocity: Any) -> dict[str, Any] | None:
        if not isinstance(velocity, dict):
            return None
        selected = _select_present(
            velocity,
            (
                "source",
                "frame",
                "dimension",
                "east_mps",
                "north_mps",
                "up_mps",
                "vertical_rate_mps",
                "ground_speed_mps",
                "ground_speed_kt",
                "track",
                "track_deg",
                "track_unit",
                "vertical_component_available",
            ),
        )
        return selected or None

    @staticmethod
    def _minimal_acceleration(acceleration: Any) -> dict[str, Any] | None:
        if not isinstance(acceleration, dict):
            return None
        selected = _select_present(
            acceleration,
            (
                "source",
                "frame",
                "dimension",
                "east_mps2",
                "north_mps2",
                "up_mps2",
                "vertical_mps2",
                "horizontal_mps2",
                "total_mps2",
                "magnitude_mps2",
                "vertical_component_available",
            ),
        )
        return selected or None

    @staticmethod
    def _minimal_display(display: Any) -> dict[str, Any]:
        if not isinstance(display, dict):
            return {}
        # Display metadata is already compact and controls whether the viewer
        # uses raw connection lines or reconstructed path lines.
        return _deepcopy_json(display)

    def _minimal_piecewise_reference(
        self,
        payload: dict[str, Any],
        render_keyframes: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Build lightweight segment references for viewer-side plotting.

        The detailed V-Spline payload contains full per-segment diagnostics.
        The minimal payload only needs enough information to slice the already
        sampled ``render_keyframes`` into segment polylines and optionally draw
        internal shared-boundary markers.
        """
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        piecewise = quality.get("piecewise") if isinstance(quality.get("piecewise"), dict) else None
        segments_src = quality.get("segments") if isinstance(quality.get("segments"), list) else []

        if not piecewise and not segments_src:
            return None

        render_ranges = self._render_keyframe_ranges_by_segment(render_keyframes)
        segment_refs = self._minimal_segment_refs(segments_src, render_ranges)
        boundary_refs = self._minimal_segment_boundary_refs(piecewise, segment_refs)
        component_refs = self._minimal_component_refs(piecewise)
        global_quality = piecewise.get("global_quality") if isinstance(piecewise, dict) else {}

        return _clean_json(
            {
                "piecewise": {
                    "enabled": bool(piecewise.get("enabled", bool(segment_refs))) if isinstance(piecewise, dict) else bool(segment_refs),
                    "mode": piecewise.get("mode") if isinstance(piecewise, dict) else None,
                    "component_count": global_quality.get("connected_component_count") if isinstance(global_quality, dict) else None,
                    "segment_count": len(segment_refs),
                    "boundary_count": len(boundary_refs),
                    "components": component_refs,
                },
                "segments": segment_refs,
                "segment_boundaries": boundary_refs,
            }
        )

    @staticmethod
    def _render_keyframe_ranges_by_segment(render_keyframes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        ranges: dict[str, dict[str, Any]] = {}
        for idx, kf in enumerate(render_keyframes):
            if not isinstance(kf, dict):
                continue
            segment_id = kf.get("segment_id")
            if segment_id is None:
                continue
            sid = str(segment_id)
            item = ranges.setdefault(
                sid,
                {
                    "render_keyframe_start_index": idx,
                    "render_keyframe_end_index": idx,
                    "render_keyframe_start_id": kf.get("id"),
                    "render_keyframe_end_id": kf.get("id"),
                    "render_time_start": kf.get("t"),
                    "render_time_end": kf.get("t"),
                },
            )
            item["render_keyframe_end_index"] = idx
            item["render_keyframe_end_id"] = kf.get("id")
            item["render_time_end"] = kf.get("t")
        return ranges

    @staticmethod
    def _minimal_segment_refs(
        segments_src: list[Any],
        render_ranges: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        seen: set[str] = set()

        for seg in segments_src:
            if not isinstance(seg, dict):
                continue
            segment_id = seg.get("segment_id")
            if segment_id is None:
                continue
            sid = str(segment_id)
            seen.add(sid)
            features = seg.get("features") if isinstance(seg.get("features"), dict) else {}
            ref = {
                "segment_id": sid,
                "component_id": seg.get("component_id"),
                "regime_label": seg.get("regime_label"),
                "energy_state": features.get("dominant_energy_state"),
                "energy_state_fraction": features.get("dominant_energy_state_fraction"),
                "specific_energy_height_change_m": features.get("specific_energy_height_change_m"),
                "median_specific_energy_rate_mps": features.get("median_specific_energy_rate_mps"),
                "t0": seg.get("t0"),
                "t1": seg.get("t1"),
                "start_sample_index": seg.get("start_sample_index"),
                "end_sample_index": seg.get("end_sample_index"),
                "n_observations": seg.get("n_observations"),
                "start_boundary_id": seg.get("start_boundary_id"),
                "end_boundary_id": seg.get("end_boundary_id"),
            }
            ref.update(render_ranges.get(sid, {}))
            refs.append({k: v for k, v in ref.items() if v is not None})

        # Be robust for older detailed payloads where sampled render frames have
        # segment IDs but quality.segments was not populated.
        for sid, rr in render_ranges.items():
            if sid in seen:
                continue
            refs.append({"segment_id": sid, **rr})

        refs.sort(key=lambda item: (float(item.get("t0", item.get("render_time_start", 0.0)) or 0.0), str(item.get("segment_id", ""))))
        return refs

    @staticmethod
    def _minimal_component_refs(piecewise: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(piecewise, dict):
            return []
        components = piecewise.get("connected_components")
        if not isinstance(components, list):
            return []

        out: list[dict[str, Any]] = []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            segment_ids = []
            for seg in comp.get("segments") or []:
                if isinstance(seg, dict) and seg.get("segment_id") is not None:
                    segment_ids.append(str(seg.get("segment_id")))
            out.append(
                _clean_json(
                    {
                        "component_id": comp.get("component_id"),
                        "t0": comp.get("t0"),
                        "t1": comp.get("t1"),
                        "start_sample_index": comp.get("start_sample_index"),
                        "end_sample_index": comp.get("end_sample_index"),
                        "hard_gap_before_s": comp.get("hard_gap_before_s"),
                        "hard_gap_after_s": comp.get("hard_gap_after_s"),
                        "segment_ids": segment_ids,
                    }
                )
            )
        return out

    @staticmethod
    def _minimal_segment_boundary_refs(
        piecewise: dict[str, Any] | None,
        segment_refs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(piecewise, dict):
            return []
        components = piecewise.get("connected_components")
        if not isinstance(components, list):
            return []

        boundary_to_left_segment: dict[str, str] = {}
        boundary_to_right_segment: dict[str, str] = {}
        for seg in segment_refs:
            sid = seg.get("segment_id")
            if sid is None:
                continue
            start_boundary_id = seg.get("start_boundary_id")
            end_boundary_id = seg.get("end_boundary_id")
            if start_boundary_id:
                boundary_to_right_segment[str(start_boundary_id)] = str(sid)
            if end_boundary_id:
                boundary_to_left_segment[str(end_boundary_id)] = str(sid)

        out: list[dict[str, Any]] = []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            component_id = comp.get("component_id")
            for boundary in comp.get("boundaries") or []:
                if not isinstance(boundary, dict):
                    continue
                bid = boundary.get("boundary_id")
                if bid is None:
                    continue
                bid_s = str(bid)
                shared_state = boundary.get("shared_state") if isinstance(boundary.get("shared_state"), dict) else {}
                out.append(
                    _clean_json(
                        {
                            "boundary_id": bid_s,
                            "component_id": component_id,
                            "t_boundary": boundary.get("t_boundary"),
                            "sample_index": boundary.get("sample_index"),
                            "local_sample_index": boundary.get("local_sample_index"),
                            "left_segment_id": boundary_to_left_segment.get(bid_s),
                            "right_segment_id": boundary_to_right_segment.get(bid_s),
                            "reasons": boundary.get("reasons"),
                            "score": boundary.get("score"),
                            "is_hard_gap": boundary.get("is_hard_gap"),
                            "position_m": shared_state.get("position_m"),
                            "velocity_mps": shared_state.get("velocity_mps"),
                            "confidence": shared_state.get("confidence"),
                        }
                    )
                )
        out.sort(key=lambda item: (float(item.get("t_boundary", 0.0) or 0.0), str(item.get("boundary_id", ""))))
        return out


    def _build_raw_adsb_payload(
        self,
        *,
        rule: TrackRuleConfig,
        origin: dict[str, Any],
        keyframes: list[dict[str, Any]],
        normalized_report: dict[str, Any],
        loader_report: dict[str, Any],
        events: pd.DataFrame,
    ) -> dict[str, Any]:
        render_keyframes = [self._raw_render_keyframe(kf) for kf in keyframes]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "track_id": rule.track_id,
            "icao": rule.icao.upper(),
            "method": {"id": RAW_ADSB_METHOD_ID, "label": "Raw ADS-B"},
            "reference": self._reference(rule, origin, method_id=RAW_ADSB_METHOD_ID),
            "raw_keyframes": keyframes,
            "render_keyframes": render_keyframes,
            "display": {
                "render_animation": {
                    "mode": "render_keyframe",
                    "position_anchor": "observed_same_t",
                    "position_interpolation": "none",
                    "vector_value_interpolation": "none",
                    "reconstructed": False,
                    "resampled": False,
                },
                "vectors": {
                    "show_velocity": True,
                    "show_acceleration": True,
                    "velocity_component": "3d_when_available",
                    "acceleration_component": "3d_when_available",
                },
            },
            "quality": {
                "method": RAW_ADSB_METHOD_ID,
                "raw_keyframe_count": len(keyframes),
                "render_keyframe_count": len(render_keyframes),
                "paired_keyframe_count": sum(1 for k in keyframes if k.get("paired_for_vspline") is True),
                "loader_report": loader_report,
                "normalizer_report": normalized_report,
            },
        }

        if self.config.include_raw_events_inline:
            payload["raw_events"] = _dataframe_records(events)
        else:
            payload["raw_event_count"] = int(len(events))
        return payload

    def _build_v_spline_payload(
        self,
        *,
        rule: TrackRuleConfig,
        origin: dict[str, Any],
        raw_keyframes: list[dict[str, Any]],
        prepared: Any,
        fits: list[tuple[str, Any, Any]],
        piecewise_report: dict[str, Any] | None = None,
        segment_metadata: dict[str, dict[str, Any]] | None = None,
        method_id: str = V_SPLINE_METHOD_ID,
        method_label: str = "V-Spline",
        backend: str | None = None,
        preset: str | None = None,
        method_config: TrackOutputPipelineConfig | None = None,
    ) -> dict[str, Any]:
        cfg = method_config or self.config
        backend = backend or "bspline_piecewise"
        if backend not in BSPLINE_BACKENDS and backend not in HERMITE_BACKENDS:
            raise ValueError(f"Unsupported production backend: {backend!r}")
        render_keyframes = self._build_v_spline_render_keyframes(
            fits,
            origin,
            rule,
            method_id=method_id,
            method_config=cfg,
        )
        if backend in HERMITE_BACKENDS:
            basis = "nodal_cubic_hermite_per_segment"
            objective = (
                "paper_v_spline_position_velocity_residuals_plus_confidence_weighted_velocity_residuals_"
                "plus_n_scaled_integrated_squared_acceleration_with_regime_specific_adaptive_speed_floors"
            )
            fit_mode = (
                "dynamic_piecewise_stable_hermite_v_spline_soft_velocity_endpoints"
                if backend == "hermite_stable"
                else "dynamic_piecewise_paper_hermite_v_spline_per_segment_tuned_exact_c1_post_join_harmonized"
            )
            base_config = asdict(cfg.hermite_config)
            continuity_breaks = (
                "hard_gap_and_true_event_boundaries_only; stable Hermite reports event-aware join continuity and intentionally does not claim hard continuity across discontinuities"
                if backend == "hermite_stable"
                else "hard_gap_components_only; internal_dynamic_segment_joins_are_exact_C0_C1_after_post_join_velocity_harmonization"
            )
        else:
            basis = (
                f"clamped_degree_{cfg.bspline_config.degree}_b_spline_per_hard_gap_component"
                if backend in GLOBAL_BSPLINE_BACKENDS
                else f"clamped_degree_{cfg.bspline_config.degree}_b_spline_per_segment"
            )
            if backend in GLOBAL_BSPLINE_BACKENDS:
                objective = (
                    "global_component_b_spline_v_spline_position_plus_confidence_weighted_velocity_residuals_"
                    "and_v_spline_acceleration_penalty_no_dynamic_regime_join_boundaries"
                )
                fit_mode = "one_cubic_b_spline_v_spline_per_hard_gap_component_segmentation_ablation"
            elif backend in {"quintic_bspline", "quintic_kalman_boundary"}:
                objective = (
                    "aviation_segmented_quintic_v_spline_position_plus_confidence_weighted_velocity_residuals_"
                    "with_integrated_squared_acceleration_jerk_and_snap_penalties_plus_robust_hard_position_and_soft_velocity_acceleration_boundary_priors"
                )
                fit_mode = "segmented_local_quintic_aviation_v_spline_overlap_save_with_robust_hard_C0_and_soft_higher_order_join_priors"
            elif backend == "bspline_overlap":
                objective = (
                    "overlap_save_segmented_b_spline_v_spline_position_plus_confidence_weighted_velocity_residuals_"
                    "and_v_spline_acceleration_penalty_with_robust_hard_boundary_position_anchors"
                )
                fit_mode = "segmented_local_b_spline_overlap_save_render_trusted_interiors_only_robust_hard_C0_joins"
            elif backend == "bspline_join_smooth":
                objective = (
                    "join_smoothed_segmented_b_spline_v_spline_position_plus_confidence_weighted_velocity_residuals_"
                    "with_robust_hard_C0_joins_stronger_soft_acceleration_priors_and_integrated_squared_jerk_penalty"
                )
                fit_mode = "segmented_local_b_spline_with_robust_hard_C0_and_stronger_C2_C3_like_soft_join_behavior"
            else:
                objective = (
                    "local_segment_position_velocity_residuals_plus_v_spline_like_integrated_squared_acceleration_"
                    "plus_optional_jerk_plus_soft_shared_boundary_acceleration_priors_subject_to_shared_boundary_position_velocity_policy"
                )
                fit_mode = "dynamic_piecewise_local_b_spline_per_segment_tuned_exact_c1_post_join_harmonized_soft_c2_join"
            base_config = asdict(cfg.bspline_config)
            if backend in GLOBAL_BSPLINE_BACKENDS:
                continuity_breaks = "hard_gap_components_only; no dynamic-regime internal joins are present in this backend"
            else:
                continuity_breaks = (
                    "hard_gap_and_true_event_boundaries_only; overlap-save/higher-order aviation variants render trusted segment interiors and enforce robust hard C0 at ordinary joins"
                    if backend in SOFT_BOUNDARY_BACKENDS or backend in OVERLAP_SAVE_BACKENDS
                    else "hard_gap_components_only; internal_dynamic_segment_joins_are_exact_C0_C1_after_post_join_velocity_harmonization_with_soft_C2_priors"
                )

        holdout_reports = [
            meta.get("holdout_evaluation")
            for meta in (segment_metadata or {}).values()
            if isinstance(meta.get("holdout_evaluation"), dict)
        ]
        holdout_enabled = [r for r in holdout_reports if r.get("enabled") and isinstance(r.get("metrics"), dict)]
        holdout_eval = {
            "enabled": bool(holdout_enabled),
            "fraction_requested": float(cfg.holdout_evaluation_fraction),
            "segment_count": int(len(holdout_reports)),
            "evaluated_segment_count": int(len(holdout_enabled)),
            "holdout_sample_count": int(sum(int(r.get("n_holdout") or 0) for r in holdout_enabled)),
            "rmse_3d_m_mean_by_segment": (
                float(np.mean([float(r["metrics"].get("rmse_3d_m", 0.0)) for r in holdout_enabled]))
                if holdout_enabled else None
            ),
            "p95_error_3d_m_max_by_segment": (
                float(max(float(r["metrics"].get("p95_error_3d_m", 0.0)) for r in holdout_enabled))
                if holdout_enabled else None
            ),
            "note": "each segment diagnostic withholds deterministic interior observations, refits the same local spline family, and scores predictions at held-out raw ADS-B points",
        }

        quality = {
            "method": method_id,
            "reconstruction_backend": backend,
            "basis": basis,
            "objective": objective,
            "time_step_s": cfg.v_spline_time_step_s,
            "output_frequency_hz": cfg.v_spline_output_frequency_hz,
            "raw_keyframe_count": len(raw_keyframes),
            "paired_sample_count": len(prepared.samples),
            "render_keyframe_count": len(render_keyframes),
            "fit_mode": fit_mode,
            "segment_count": len(fits),
            "fit_time_start": render_keyframes[0]["t"] if render_keyframes else None,
            "fit_time_end": render_keyframes[-1]["t"] if render_keyframes else None,
            "adapter_diagnostics": prepared.diagnostics,
            "preset": preset,
            "base_core_config": base_config,
            "bspline_config": asdict(cfg.bspline_config) if backend in BSPLINE_BACKENDS else None,
            "hermite_config": asdict(cfg.hermite_config) if backend in HERMITE_BACKENDS else None,
            "core_config": asdict(fits[0][2].config) if len(fits) == 1 else base_config,
            "piecewise": piecewise_report,
            "holdout_evaluation": holdout_eval,
            "segments": [
                {
                    **((segment_metadata or {}).get(segment_id, {})),
                    "segment_id": segment_id,
                    "n_observations": int(segment.n_observations),
                    "t0": float(segment.t0),
                    "t1": float(segment.t1),
                    "dt_min_s": segment.dt_min_s,
                    "dt_max_s": segment.dt_max_s,
                    "lambda_interval_min": float(np.min(fit.lambda_intervals)) if fit.lambda_intervals.size else None,
                    "lambda_interval_max": float(np.max(fit.lambda_intervals)) if fit.lambda_intervals.size else None,
                    "selected_core_config": asdict(fit.config),
                    "diagnostics": fit.diagnostics,
                }
                for segment_id, segment, fit in fits
            ],
        }
        quality["trajectory_model_evaluation"] = aggregate_trajectory_model_metrics(
            quality["segments"],
            piecewise_global_quality=(piecewise_report or {}).get("global_quality") if isinstance(piecewise_report, dict) else None,
            raw_times=[kf.get("t") for kf in raw_keyframes if isinstance(kf, dict)],
            render_times=[kf.get("t") for kf in render_keyframes if isinstance(kf, dict)],
            fit_mode=fit_mode,
            reconstruction_backend=backend,
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": rule.track_id,
            "icao": rule.icao.upper(),
            "method": {"id": method_id, "label": method_label},
            "reference": self._reference(rule, origin, method_id=method_id, method_config=cfg),
            "raw_keyframes": raw_keyframes,
            "render_keyframes": render_keyframes,
            "display": {
                "v_spline_animation": {
                    "enabled": True,
                    "method": method_id,
                    "basis": basis,
                    "objective": quality["objective"],
                    "reconstructed": True,
                    "resampled": True,
                    "time_step_s": cfg.v_spline_time_step_s,
                    "output_frequency_hz": cfg.v_spline_output_frequency_hz,
                    "raw_points_connected": False,
                    "reconstructed_path_as_line": True,
                    "raw_layer": "raw_keyframes",
                    "path_layer": "render_keyframes",
                    "continuity_breaks": continuity_breaks,
                },
                "render_animation": {
                    "mode": "render_keyframe",
                    "position_anchor": "v_spline_reconstructed",
                    "position_interpolation": "none_already_resampled",
                    "vector_value_interpolation": "none_already_resampled",
                    "reconstructed": True,
                    "resampled": True,
                    "time_step_s": cfg.v_spline_time_step_s,
                },
                "vectors": {
                    "show_velocity": True,
                    "show_acceleration": True,
                    "velocity_component": "3d",
                    "acceleration_component": "3d",
                },
            },
            "quality": quality,
        }

    def _build_kalman_rts_payload(
        self,
        *,
        rule: TrackRuleConfig,
        origin: dict[str, Any],
        raw_keyframes: list[dict[str, Any]],
        prepared: Any,
        fits: list[tuple[str, Any, Any]],
        piecewise_report: dict[str, Any] | None = None,
        segment_metadata: dict[str, dict[str, Any]] | None = None,
        method_id: str = KALMAN_RTS_METHOD_ID,
        method_label: str = "Kalman-RTS",
        preset: str | None = None,
        method_config: TrackOutputPipelineConfig | None = None,
    ) -> dict[str, Any]:
        cfg = method_config or self.config
        render_keyframes = self._build_v_spline_render_keyframes(
            fits,
            origin,
            rule,
            method_id=method_id,
            method_config=cfg,
            position_source="kalman_rts_smoothed_state",
            velocity_source="kalman_rts_smoothed_velocity",
            acceleration_source="kalman_rts_smoothed_acceleration",
        )
        basis = "none_state_space_constant_acceleration"
        objective = (
            "linear_gaussian_position_velocity_measurements_with_white_jerk_process_model_"
            "and_rauch_tung_striebel_fixed_interval_smoothing"
        )
        quality = {
            "method": method_id,
            "reconstruction_backend": "kalman_rts",
            "basis": basis,
            "objective": objective,
            "time_step_s": cfg.v_spline_time_step_s,
            "output_frequency_hz": cfg.v_spline_output_frequency_hz,
            "raw_keyframe_count": len(raw_keyframes),
            "paired_sample_count": len(prepared.samples),
            "render_keyframe_count": len(render_keyframes),
            "fit_mode": "whole_track_kalman_filter_plus_rts_smoother_no_segmentation",
            "segment_count": 0,
            "state_smoother_count": len(fits),
            "fit_time_start": render_keyframes[0]["t"] if render_keyframes else None,
            "fit_time_end": render_keyframes[-1]["t"] if render_keyframes else None,
            "adapter_diagnostics": prepared.diagnostics,
            "preset": preset,
            "base_core_config": asdict(cfg.kalman_rts_config),
            "kalman_rts_config": asdict(cfg.kalman_rts_config),
            "core_config": asdict(fits[0][2].config) if len(fits) == 1 else asdict(cfg.kalman_rts_config),
            "segmentation_applied": False,
            "piecewise": piecewise_report,
            "segments": [
                {
                    **((segment_metadata or {}).get(segment_id, {})),
                    "segment_id": segment_id,
                    "n_observations": int(segment.n_observations),
                    "t0": float(segment.t0),
                    "t1": float(segment.t1),
                    "dt_min_s": segment.dt_min_s,
                    "dt_max_s": segment.dt_max_s,
                    "lambda_interval_min": float(np.min(fit.lambda_intervals)) if fit.lambda_intervals.size else None,
                    "lambda_interval_max": float(np.max(fit.lambda_intervals)) if fit.lambda_intervals.size else None,
                    "selected_core_config": asdict(fit.config),
                    "diagnostics": fit.diagnostics,
                }
                for segment_id, segment, fit in fits
            ],
        }
        quality["trajectory_model_evaluation"] = aggregate_trajectory_model_metrics(
            quality["segments"],
            piecewise_global_quality=(piecewise_report or {}).get("global_quality") if isinstance(piecewise_report, dict) else None,
            raw_times=[kf.get("t") for kf in raw_keyframes if isinstance(kf, dict)],
            render_times=[kf.get("t") for kf in render_keyframes if isinstance(kf, dict)],
            fit_mode=quality["fit_mode"],
            reconstruction_backend="kalman_rts",
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "track_id": rule.track_id,
            "icao": rule.icao.upper(),
            "method": {"id": method_id, "label": method_label},
            "reference": self._reference(rule, origin, method_id=method_id, method_config=cfg),
            "raw_keyframes": raw_keyframes,
            "render_keyframes": render_keyframes,
            "display": {
                "v_spline_animation": {
                    "enabled": False,
                    "method": method_id,
                    "basis": basis,
                    "objective": objective,
                    "reconstructed": True,
                    "resampled": True,
                    "time_step_s": cfg.v_spline_time_step_s,
                    "output_frequency_hz": cfg.v_spline_output_frequency_hz,
                    "raw_points_connected": False,
                    "reconstructed_path_as_line": True,
                    "raw_layer": "raw_keyframes",
                    "path_layer": "render_keyframes",
                    "continuity_breaks": "none_from_segmentation; one_whole_track_state_smoother",
                    "note": "Kalman-RTS is not a V-Spline; this block is retained only for viewer compatibility.",
                },
                "render_animation": {
                    "mode": "render_keyframe",
                    "position_anchor": "kalman_rts_smoothed_state",
                    "position_interpolation": "none_already_resampled",
                    "vector_value_interpolation": "none_already_resampled",
                    "reconstructed": True,
                    "resampled": True,
                    "time_step_s": cfg.v_spline_time_step_s,
                },
                "vectors": {
                    "show_velocity": True,
                    "show_acceleration": True,
                    "velocity_component": "3d",
                    "acceleration_component": "3d",
                },
            },
            "quality": quality,
        }

    # ------------------------------------------------------------------
    # Reference, local coordinates, and render helpers
    # ------------------------------------------------------------------

    def _reference(
        self,
        rule: TrackRuleConfig,
        origin: dict[str, Any],
        *,
        method_id: str,
        method_config: TrackOutputPipelineConfig | None = None,
    ) -> dict[str, Any]:
        cfg = method_config or self.config
        field_ft = rule.field_elevation.elevation_ft_msl
        ref: dict[str, Any] = {
            "coordinate_system": "WGS84",
            "local_frame": "horizontal_enu_plus_barometric_z",
            "origin": {
                "lat": origin["lat"],
                "lon": origin["lon"],
                "altitude_reference": "field_elevation",
            },
            "field_elevation_ft_msl": field_ft,
            "track_unit": "deg",
            "speed_unit": "kt",
            "time_unit": "unix_seconds",
            "keyframe_time_quantization_s": rule.keyframe_time_quantization_s,
            "horizontal_projection": {
                "type": "wgs84_ecef_to_local_enu",
                "altitude_used_for_horizontal_projection_m": 0.0,
                "x_axis": "east_m",
                "y_axis": "north_m",
            },
            "vertical_observation": {
                "source": "altitude_ft_msl",
                "z_unit": "m",
                "z_reference": "field",
                "active_vertical_reference_ft_msl": field_ft,
            },
            "velocity_observation": {
                "source": "reported_adsb_ground_speed_track_plus_adsb_vertical_rate",
                "frame": "local_enu",
                "dimension": "3d",
                "ground_speed_unit": "kt",
                "component_unit": "m/s",
                "track_unit": "deg",
                "track_convention": "clockwise_from_true_north",
                "east_mps": "speed_mps * sin(track)",
                "north_mps": "speed_mps * cos(track)",
                "up_mps": "ADS-B vertical_rate_fpm converted to m/s when present",
            },
            "acceleration_observation": {
                "source": "derived_from_consecutive_adsb_velocity_delta",
                "used_as_vspline_observation": False,
            },
            "keyframe_aggregation": {
                "mode": "same_t_bucket",
                "time_quantization_s": rule.keyframe_time_quantization_s,
                "position": "mean of valid position observations within t",
                "velocity": "mean of velocity vectors computed from reported GS+track and ADS-B vertical rate within t",
                "acceleration": "derived from consecutive velocity keyframes for display only",
            },
        }
        if method_id.startswith(KALMAN_RTS_METHOD_ID):
            ref["kalman_rts"] = {
                "backend": "whole_track_constant_acceleration_kalman_rts",
                "state_space_model": "3d_constant_acceleration_with_white_jerk_process_noise",
                "state_order": ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"],
                "coordinate_state": "local_enu_position_velocity_acceleration_for_the_full_prepared_flight",
                "measurements": "prepared paired ADS-B position and velocity observations",
                "objective": "linear Gaussian filtering followed by one RTS fixed-interval backward smoothing pass over the full prepared flight",
                "segmentation": "none; no dynamic segmentation, hard-gap splitting, boundary-state estimation, local tuning, or join harmonization is applied to this method",
                "render_interpolation": cfg.kalman_rts_config.interpolation,
                "config": asdict(cfg.kalman_rts_config),
            }
        elif method_id.startswith(HERMITE_V_SPLINE_METHOD_ID):
            ref["v_spline"] = {
                "backend": "paper_oriented_nodal_hermite_v_spline",
                "basis": "nodal_cubic_hermite_per_segment",
                "coordinate_state": "nodal_position_velocity_theta_in_local_enu_plus_barometric_z_per_dynamic_segment",
                "objective": (
                    "paper_v_spline_position_velocity_residuals_plus_n_scaled_integrated_squared_acceleration_"
                    "with_adaptive_or_constant_interval_penalty"
                ),
                "hard_constraints": "reported raw boundary positions plus harmonized shared join velocities",
                "continuity": "C0/C1 within and across dynamic segments; second derivative is piecewise continuous and diagnostic across joins",
                "time_normalization": "absolute_seconds_observation_knots; interval_lengths_enter_penalty",
            }
        elif method_id.startswith(BSPLINE_V_SPLINE_METHOD_ID) or method_id == V_SPLINE_METHOD_ID:
            ref["v_spline"] = {
                "backend": "local_segment_b_spline_v_spline_penalty",
                "basis": f"clamped_degree_{cfg.bspline_config.degree}_b_spline_per_segment",
                "coordinate_state": "control_points_c_j_in_local_enu_plus_barometric_z_per_dynamic_segment",
                "objective": (
                    "local_segment_position_velocity_residuals_plus_v_spline_like_integrated_squared_acceleration_"
                    "plus_optional_jerk"
                ),
                "hard_constraints": "reported raw boundary positions plus harmonized shared join velocities",
                "continuity": "C2 inside each segment; exact C0/C1 at dynamic segment joins; acceleration continuity is a soft diagnostic prior",
                "time_normalization": "segment_local_seconds_knots_t_minus_t_segment_start",
            }
        return ref

    def _choose_origin(self, keyframes: list[dict[str, Any]], rule: TrackRuleConfig) -> dict[str, Any]:
        if rule.origin_lat_deg is not None and rule.origin_lon_deg is not None:
            return {"lat": float(rule.origin_lat_deg), "lon": float(rule.origin_lon_deg)}

        for kf in keyframes:
            pos = kf.get("position") if isinstance(kf.get("position"), dict) else None
            if not pos:
                continue
            lat = _num(pos.get("lat") or pos.get("lat_deg"))
            lon = _num(pos.get("lon") or pos.get("lon_deg"))
            if lat is not None and lon is not None:
                return {"lat": lat, "lon": lon}

        raise ValueError(f"Cannot choose local origin for {rule.track_id}: no keyframe position")

    def _add_local_xy_to_keyframes(
        self,
        keyframes: list[dict[str, Any]],
        origin: dict[str, Any],
        rule: TrackRuleConfig,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for kf in keyframes:
            item = _deepcopy_json(kf)
            pos = item.get("position") if isinstance(item.get("position"), dict) else None
            if pos is not None:
                lat = _num(pos.get("lat") or pos.get("lat_deg"))
                lon = _num(pos.get("lon") or pos.get("lon_deg"))
                if lat is not None and lon is not None:
                    east, north = geodetic_to_local_xy(lat, lon, origin["lat"], origin["lon"])
                    pos["x_m"] = east
                    pos["y_m"] = north
                if pos.get("z_m") is None:
                    alt_ft = _num(pos.get("altitude_ft_msl"))
                    if alt_ft is not None:
                        pos["z_m"] = (alt_ft - rule.field_elevation.elevation_ft_msl) * FT_TO_M
                pos["frame"] = "horizontal_enu_plus_barometric_z"
            out.append(item)
        return out

    def _raw_render_keyframe(self, kf: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(kf.get("id")),
            "t": kf.get("t"),
            "source_keyframe_id": kf.get("id"),
            "position": kf.get("position"),
            "velocity": kf.get("velocity"),
            "acceleration": kf.get("acceleration"),
            "quality": {"status": "observed", "method": RAW_ADSB_METHOD_ID},
        }

    def _build_v_spline_render_keyframes(
        self,
        fits: list[tuple[str, Any, Any]],
        origin: dict[str, Any],
        rule: TrackRuleConfig,
        *,
        method_id: str = V_SPLINE_METHOD_ID,
        method_config: TrackOutputPipelineConfig | None = None,
        position_source: str = "v_spline_reconstructed",
        velocity_source: str = "v_spline_first_derivative",
        acceleration_source: str = "v_spline_second_derivative",
    ) -> list[dict[str, Any]]:
        cfg = method_config or self.config
        frames: list[dict[str, Any]] = []
        next_id = 1
        last_t: float | None = None
        for segment_id, segment, fit in fits:
            times_all = np.asarray(_time_grid(segment.t0, segment.t1, cfg.v_spline_time_step_s), dtype=float)
            if last_t is not None and times_all.size and abs(float(times_all[0]) - last_t) < 1e-9:
                times_all = times_all[1:]
            if times_all.size == 0:
                continue

            # Vectorize derivative evaluation.  B-spline evaluation builds basis
            # matrices, so per-sample evaluate([t]) calls are unnecessarily slow
            # when rendering thousands of animation frames.
            pos_all = fit.evaluate(times_all, deriv=0)
            vel_all = fit.evaluate(times_all, deriv=1)
            acc_all = fit.evaluate(times_all, deriv=2)

            for t, pos, vel, acc in zip(times_all, pos_all, vel_all, acc_all):
                frame = {
                    "id": f"vsf_{next_id:06d}",
                    "t": float(t),
                    "segment_id": segment_id,
                    "position": self._render_position(pos, origin, rule, source=position_source),
                    "velocity": self._render_velocity(vel, source=velocity_source),
                    "acceleration": self._render_acceleration(acc, source=acceleration_source),
                    "quality": {"status": "reconstructed", "method": method_id},
                }
                frames.append(frame)
                next_id += 1
                last_t = float(t)
        return frames

    def _render_position(self, pos: np.ndarray, origin: dict[str, Any], rule: TrackRuleConfig, *, source: str = "v_spline_reconstructed") -> dict[str, Any]:
        x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
        lat, lon = local_xy_to_latlon(x, y, origin["lat"], origin["lon"])
        field_ft = float(rule.field_elevation.elevation_ft_msl)
        altitude_m_msl = z + field_ft * FT_TO_M
        return {
            "source": source,
            "frame": "horizontal_enu_plus_barometric_z",
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "lat": lat,
            "lon": lon,
            "altitude_ft_msl": altitude_m_msl / FT_TO_M,
            "altitude_m_msl": altitude_m_msl,
            "height_above_reference_m": z,
            "z_reference": "field",
            "vertical_reference_ft_msl": field_ft,
            "interpolated": True,
        }

    @staticmethod
    def _render_velocity(vel: np.ndarray, *, source: str = "v_spline_first_derivative") -> dict[str, Any]:
        east, north, up = (float(vel[0]), float(vel[1]), float(vel[2]))
        ground_speed = math.hypot(east, north)
        track = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0 if ground_speed > 0 else None
        return {
            "source": source,
            "frame": "local_enu",
            "dimension": "3d",
            "east_mps": east,
            "north_mps": north,
            "up_mps": up,
            "vertical_rate_mps": up,
            "ground_speed_mps": ground_speed,
            "ground_speed_kt": ground_speed / KNOT_TO_MPS,
            "track": track,
            "track_deg": track,
            "track_unit": "deg",
            "vertical_component_available": True,
        }

    @staticmethod
    def _render_acceleration(acc: np.ndarray, *, source: str = "v_spline_second_derivative") -> dict[str, Any]:
        east, north, up = (float(acc[0]), float(acc[1]), float(acc[2]))
        horizontal = math.hypot(east, north)
        total = math.sqrt(east * east + north * north + up * up)
        return {
            "source": source,
            "frame": "local_enu",
            "dimension": "3d",
            "east_mps2": east,
            "north_mps2": north,
            "up_mps2": up,
            "horizontal_mps2": horizontal,
            "total_mps2": total,
            "vertical_component_available": True,
        }


# ---------------------------------------------------------------------------
# Geometry and JSON helpers
# ---------------------------------------------------------------------------


def geodetic_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Local east/north from WGS84 lat/lon using zero-altitude ECEF tangent plane."""
    x, y, z = geodetic_to_ecef(lat, lon, 0.0)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, 0.0)
    dx, dy, dz = x - x0, y - y0, z - z0
    la = math.radians(lat0)
    lo = math.radians(lon0)
    sl, cl = math.sin(la), math.cos(la)
    so, co = math.sin(lo), math.cos(lo)
    east = -so * dx + co * dy
    north = -sl * co * dx - sl * so * dy + cl * dz
    return east, north


def local_xy_to_latlon(east: float, north: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Approximate inverse of geodetic_to_local_xy on the zero-altitude tangent plane."""
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, 0.0)
    la = math.radians(lat0)
    lo = math.radians(lon0)
    sl, cl = math.sin(la), math.cos(la)
    so, co = math.sin(lo), math.cos(lo)
    dx = -so * east - sl * co * north
    dy = co * east - sl * so * north
    dz = cl * north
    lat, lon, _ = ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)
    return lat, lon


def _time_grid(t0: float, t1: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("time step must be positive")
    n = int(math.floor((t1 - t0) / step + 1e-9))
    times = [float(t0 + i * step) for i in range(n + 1)]
    if not times or abs(times[-1] - t1) > 1e-9:
        times.append(float(t1))
    return times


def _iso_utc(unix_seconds: float) -> str:
    return datetime.fromtimestamp(float(unix_seconds), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _best_callsign(df: pd.DataFrame) -> str:
    if "callsign" not in df.columns:
        return ""
    s = df["callsign"].dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return ""
    return str(s.mode().iloc[0])


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _deepcopy_json(obj: Any) -> Any:
    # Minimal payload construction calls this for every keyframe field.
    # Avoid a JSON serialize/parse round-trip per scalar; _clean_json already
    # converts numpy/pandas values into JSON-safe Python objects.
    return _clean_json(obj)


def _select_present(payload: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: _deepcopy_json(payload[key]) for key in keys if key in payload and payload.get(key) is not None}


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _csv_scalar(value: Any) -> Any:
    value = _clean_json(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value




def _kalman_rts_output_specs(config: TrackOutputPipelineConfig) -> list[KalmanRTSOutputSpec]:
    """Return requested Kalman/RTS output specs.

    The default emits three files: ``kalman_rts_balanced.json``,
    ``kalman_rts_accurate.json``, and ``kalman_rts_smooth.json``.  This is kept
    separate from ``_v_spline_output_specs`` because Kalman/RTS is a state-space
    smoother, not a spline backend.
    """
    if not bool(config.kalman_rts_output_enabled):
        return []
    preset_requested = tuple(config.kalman_rts_output_presets) or KALMAN_RTS_PRESETS

    def normalize_preset(item: str) -> Literal["balanced", "accurate", "smooth"]:
        key = str(item).strip().lower()
        if key not in KALMAN_RTS_PRESETS:
            raise ValueError("Kalman-RTS presets must be one of: balanced, accurate, smooth")
        return key  # type: ignore[return-value]

    out: list[KalmanRTSOutputSpec] = []
    seen: set[str] = set()
    for preset_item in preset_requested:
        preset = normalize_preset(str(preset_item))
        method_id = f"{KALMAN_RTS_METHOD_ID}_{preset}"
        if method_id in seen:
            continue
        out.append(
            KalmanRTSOutputSpec(
                method_id=method_id,
                label=f"Kalman-RTS ({preset})",
                file_stem=method_id,
                preset=preset,
            )
        )
        seen.add(method_id)
    return out

def _v_spline_output_specs(config: TrackOutputPipelineConfig) -> list[VSplineOutputSpec]:
    """Return all requested production V-Spline output specs.

    The default emits legacy B-spline/Hermite files plus aviation global,
    overlap, join-smoothed, stable-Hermite, and quintic variants under each of the
    balanced/accurate/smooth presets.  Backend aliases are accepted so existing
    shell invocations keep working, but method ids and filenames are always
    explicit about both backend and preset.
    """
    backend_requested = list(tuple(config.v_spline_output_backends) or ("bspline_piecewise",))
    if bool(config.use_kalman_boundary_prior) and "quintic_kalman_boundary" not in {str(b).strip().lower() for b in backend_requested}:
        backend_requested.append("quintic_kalman_boundary")
    preset_requested = tuple(config.v_spline_output_presets) or V_SPLINE_PRESETS

    def normalize_backend(item: str) -> str:
        key = str(item).strip().lower()
        if key in {"", "bspline", "bspline_vspline", "b_spline", "v_spline_bspline", "bspline_piecewise"}:
            return "bspline_piecewise"
        if key in {"hermite", "hermite_vspline", "v_spline_hermite", "paper_hermite", "hermite_piecewise"}:
            return "hermite_piecewise"
        if key in {"bspline_global", "global_bspline", "component_global", "bspline_component_global", "v_spline_bspline_global", "aviation_global"}:
            return "bspline_component_global"
        if key in {"bspline_overlap", "overlap", "overlap_save", "v_spline_bspline_overlap"}:
            return "bspline_overlap"
        if key in {"bspline_join_smooth", "join_smooth", "v_spline_bspline_join_smooth"}:
            return "bspline_join_smooth"
        if key in {"hermite_stable", "stable_hermite", "v_spline_hermite_stable"}:
            return "hermite_stable"
        if key in {"quintic", "quintic_bspline", "aviation_quintic", "aviation_v_spline_quintic"}:
            return "quintic_bspline"
        if key in {"quintic_kalman_boundary", "aviation_quintic_kalman_boundary", "aviation_v_spline_quintic_kalman_boundary"}:
            return "quintic_kalman_boundary"
        raise ValueError(
            "Unsupported V-Spline backend {!r}. Supported values include: "
            "bspline_piecewise, hermite_piecewise, bspline_component_global, bspline_overlap, "
            "bspline_join_smooth, hermite_stable, quintic_bspline, quintic_kalman_boundary.".format(item)
        )

    def normalize_preset(item: str) -> Literal["balanced", "accurate", "smooth"]:
        key = str(item).strip().lower()
        if key not in V_SPLINE_PRESETS:
            raise ValueError("V-Spline presets must be one of: balanced, accurate, smooth")
        return key  # type: ignore[return-value]

    out: list[VSplineOutputSpec] = []
    seen: set[str] = set()
    for backend_item in backend_requested:
        backend = normalize_backend(str(backend_item))
        for preset_item in preset_requested:
            preset = normalize_preset(str(preset_item))
            if backend == "bspline_piecewise":
                method_id = f"{BSPLINE_V_SPLINE_METHOD_ID}_{preset}"
                label = f"B-Spline V-Spline ({preset})"
            elif backend == "hermite_piecewise":
                method_id = f"{HERMITE_V_SPLINE_METHOD_ID}_{preset}"
                label = f"Hermite V-Spline ({preset})"
            elif backend == "bspline_component_global":
                method_id = f"aviation_v_spline_bspline_global_{preset}"
                label = f"Aviation Global B-Spline V-Spline ({preset})"
            elif backend == "bspline_overlap":
                method_id = f"v_spline_bspline_overlap_{preset}"
                label = f"B-Spline V-Spline overlap-save ({preset})"
            elif backend == "bspline_join_smooth":
                method_id = f"v_spline_bspline_join_smooth_{preset}"
                label = f"B-Spline V-Spline join-smoothed ({preset})"
            elif backend == "hermite_stable":
                method_id = f"v_spline_hermite_stable_{preset}"
                label = f"Stable Hermite V-Spline ({preset})"
            elif backend == "quintic_bspline":
                method_id = f"aviation_v_spline_quintic_{preset}"
                label = f"Aviation Quintic V-Spline ({preset})"
            elif backend == "quintic_kalman_boundary":
                method_id = f"aviation_v_spline_quintic_kalman_boundary_{preset}"
                label = f"Aviation Quintic V-Spline + Kalman boundary prior ({preset})"
            else:  # pragma: no cover - normalize_backend guards this path.
                raise ValueError(f"Unsupported V-Spline backend after normalization: {backend!r}")
            if method_id in seen:
                continue
            out.append(
                VSplineOutputSpec(
                    method_id=method_id,
                    label=label,
                    file_stem=method_id,
                    backend=backend,
                    preset=preset,
                )
            )
            seen.add(method_id)
    return out


def _synthetic_gap_output_specs(
    config: TrackOutputPipelineConfig,
    kalman_specs: Iterable[KalmanRTSOutputSpec],
    v_spline_specs: Iterable[VSplineOutputSpec],
) -> list[SyntheticGapOutputSpec]:
    """Return diagnostic holdout-gap reconstructions requested by config.

    The base method must already be part of the normal emitted method set.  This
    keeps the benchmark small and makes every synthetic-gap method directly
    comparable with a normal production method in the same run.
    """
    if not bool(config.synthetic_gap_holdout_enabled):
        return []
    requested = tuple(str(m).strip() for m in config.synthetic_gap_holdout_methods if str(m).strip())
    if not requested:
        return []

    base_by_id: dict[str, tuple[Literal["kalman_rts", "v_spline"], KalmanRTSOutputSpec | VSplineOutputSpec]] = {}
    for spec in kalman_specs:
        base_by_id[spec.method_id] = ("kalman_rts", spec)
    for spec in v_spline_specs:
        base_by_id[spec.method_id] = ("v_spline", spec)

    out: list[SyntheticGapOutputSpec] = []
    seen: set[str] = set()
    for base_method_id in requested:
        if base_method_id in seen:
            continue
        item = base_by_id.get(base_method_id)
        if item is None:
            continue
        family, base_spec = item
        method_id = f"{base_method_id}_synthetic_gap"
        out.append(
            SyntheticGapOutputSpec(
                method_id=method_id,
                label=f"Synthetic-gap holdout · {base_spec.label}",
                file_stem=method_id,
                base_method_id=base_method_id,
                base_family=family,
                base_spec=base_spec,
            )
        )
        seen.add(base_method_id)
    return out


def _piecewise_global_continuity(
    piecewise_components: list[dict[str, Any]],
    fits: list[tuple[str, Any, Any]],
) -> dict[str, Any]:
    def _continuity_reports() -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for comp in piecewise_components:
            cont = comp.get("continuity") if isinstance(comp.get("continuity"), dict) else {}
            if cont:
                reports.append(cont)
        return reports

    reports = _continuity_reports()
    event_reports = [
        comp.get("event_aware_continuity")
        for comp in piecewise_components
        if isinstance(comp.get("event_aware_continuity"), dict)
    ]

    def _max_metric(key: str) -> float:
        values = []
        for cont in reports:
            if key in cont and cont.get(key) is not None:
                values.append(float(cont.get(key)))
        return float(max(values) if values else 0.0)

    def _all_ok(flag_key: str) -> bool:
        # Treat components without internal joins as continuous.  Components
        # that do have joins carry the authoritative verifier flags.
        values = [bool(cont.get(flag_key, True)) for cont in reports]
        return bool(all(values))

    def _event_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for report in event_reports:
            rows.extend([r for r in report.get("rows", []) if isinstance(r, dict)])
        return rows

    rows = _event_rows()

    def _max_from_rows(key: str, predicate) -> float:
        values = [float(r.get(key) or 0.0) for r in rows if predicate(r) and r.get(key) is not None]
        return float(max(values) if values else 0.0)

    event_aware = {
        "enabled": bool(event_reports),
        "normal_join_count": int(sum(1 for r in rows if not bool(r.get("excluded_from_normal_continuity_score")))),
        "event_boundary_count": int(sum(1 for r in rows if bool(r.get("excluded_from_normal_continuity_score")))),
        "normal_max_position_jump_m": _max_from_rows("position_jump_m", lambda r: not bool(r.get("excluded_from_normal_continuity_score"))),
        "normal_max_velocity_jump_mps": _max_from_rows("velocity_jump_mps", lambda r: not bool(r.get("excluded_from_normal_continuity_score"))),
        "normal_max_acceleration_jump_mps2": _max_from_rows("acceleration_jump_mps2", lambda r: not bool(r.get("excluded_from_normal_continuity_score"))),
        "normal_max_jerk_jump_mps3": _max_from_rows("jerk_jump_mps3", lambda r: not bool(r.get("excluded_from_normal_continuity_score"))),
        "by_event": {
            name: {
                "count": int(sum(1 for r in rows if r.get("event_bucket") == name)),
                "max_acceleration_jump_mps2": _max_from_rows("acceleration_jump_mps2", lambda r, n=name: r.get("event_bucket") == n),
                "max_jerk_jump_mps3": _max_from_rows("jerk_jump_mps3", lambda r, n=name: r.get("event_bucket") == n),
            }
            for name in sorted({str(r.get("event_bucket")) for r in rows} | {"normal_segment_join", "hard_gap", "go_around"})
        },
    }

    return {
        "connected_component_count": len(piecewise_components),
        "segment_count": len(fits),
        "internal_boundary_count": int(sum(len(c.get("boundaries", [])) for c in piecewise_components)),
        "position_continuity_ok": _all_ok("position_continuity_ok"),
        "velocity_continuity_ok": _all_ok("velocity_continuity_ok"),
        "max_internal_position_jump_m": _max_metric("max_position_jump_m"),
        "max_internal_velocity_jump_mps": _max_metric("max_velocity_jump_mps"),
        "max_internal_acceleration_jump_mps2": _max_metric("max_acceleration_jump_mps2"),
        "acceleration_continuity_enforced": False,
        "event_aware_continuity": event_aware,
    }


def _rolling_median_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    n = arr.size
    if n == 0:
        return arr
    w = max(1, int(window))
    if w <= 1:
        return arr.copy()
    half = w // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = float(np.nanmedian(arr[a:b]))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _longest_threshold_run(t: np.ndarray, values: np.ndarray, threshold: float) -> dict[str, Any]:
    arr_t = np.asarray(t, dtype=float).reshape(-1)
    arr_v = np.asarray(values, dtype=float).reshape(-1)
    n = min(arr_t.size, arr_v.size)
    if n == 0 or not math.isfinite(float(threshold)):
        return {}
    above = np.isfinite(arr_v[:n]) & (arr_v[:n] > float(threshold))
    best: tuple[int, int] | None = None
    start: int | None = None
    for idx, flag in enumerate(above.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None
    if best is None:
        return {}
    a, b = best
    return {
        "start_idx": int(a),
        "end_idx": int(b),
        "n_points": int(b - a + 1),
        "start_t": float(arr_t[a]),
        "end_t": float(arr_t[b]),
        "duration_s": float(arr_t[b] - arr_t[a]) if b >= a else 0.0,
        "max_value": float(np.nanmax(arr_v[a : b + 1])),
    }


def _progress_iter(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    unit: str,
    leave: bool,
) -> Iterable[Any]:
    """Wrap an iterable with tqdm when available and enabled."""
    if not enabled or tqdm is None:
        return iterable
    total = len(iterable) if hasattr(iterable, "__len__") else None
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True)


def _progress_write(message: str) -> None:
    """Print a status line without breaking active tqdm bars."""
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)

def _dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return [_clean_json(row) for row in df.to_dict(orient="records")]



