"""Production SQL entrypoint for ADS-B reconstruction comparisons.

Run from the project root:

    python main.py

This entrypoint is deliberately small. SQLite is the only production data
source, and every requested ICAO/track id must exist in ``flight_rules.json``.
By default it emits the legacy B-spline/Hermite V-Spline methods, aviation
global-component/overlap/join-smoothed/quintic V-Spline variants, and whole-track Kalman/RTS
baselines for balanced/accurate/smooth presets.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from boundary_state import BoundaryStateConfig  # noqa: E402
from raw_keyframe_vspline_adapter import RawKeyframeVSplineAdapterConfig  # noqa: E402
from kalman_rts_core import KalmanRTSConfig, default_kalman_rts_config_for_preset  # noqa: E402
from segmentation_kalman import KalmanSegmentationConfig  # noqa: E402
from track_output_pipeline import PipelinePaths, TrackOutputPipeline, TrackOutputPipelineConfig, _clean_json  # noqa: E402
from trajectory_segmentation import DynamicSegmentationConfig  # noqa: E402
from vspline.bspline_core import BSplineCoreConfig  # noqa: E402
from vspline.hermite_core import VSplineCoreConfig as HermiteCoreConfig  # noqa: E402
from vspline.local_tuning import LocalSegmentTuningConfig  # noqa: E402
from vspline.segment_policy import LocalSegmentPolicyConfig  # noqa: E402


Preset = Literal["balanced", "accurate", "smooth"]
PRESETS: tuple[Preset, Preset, Preset] = ("balanced", "accurate", "smooth")

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null", "off"}:
        return None
    return float(value)


def _env_str_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optional(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _preset() -> Preset:
    preset = os.getenv("RECONSTRUCTION_PRESET", "balanced").strip().lower()
    if preset not in PRESETS:
        raise ValueError("RECONSTRUCTION_PRESET must be one of: balanced, accurate, smooth")
    return preset  # type: ignore[return-value]


def _preset_tuple() -> tuple[Preset, ...]:
    raw = os.getenv("RECONSTRUCTION_PRESETS", "balanced,accurate,smooth").strip().lower()
    if not raw:
        return PRESETS
    values: list[Preset] = []
    seen: set[str] = set()
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in PRESETS:
            raise ValueError("RECONSTRUCTION_PRESETS must contain only: balanced, accurate, smooth")
        if item not in seen:
            values.append(item)  # type: ignore[arg-type]
            seen.add(item)
    return tuple(values) or PRESETS


def _icao_list() -> tuple[str, ...] | None:
    """Optional requested subset.

    When unset, the pipeline processes every rule in flight_rules.json.  When
    set, every requested value must match a rule track_id or an unambiguous ICAO.
    """
    raw = os.getenv("ICAO_LIST", "").strip()
    if not raw:
        return None
    return tuple(item.strip().upper() for item in raw.split(",") if item.strip())

def _bspline_config(preset: Preset) -> BSplineCoreConfig:
    # Local energy-state policy and local tuning override many of these values
    # per segment.  These are sane base limits, not a global one-size-fits-all
    # smoothing recipe.
    by_preset = {
        "accurate": dict(
            knot_spacing_s=3.0,
            min_observations_per_basis=4.0,
            velocity_weight=0.04,
            adaptive_eta=50_000.0,
            huber_delta_m=60.0,
            jerk_penalty_weight=0.001,
            boundary_acceleration_prior_weight=0.04,
        ),
        "balanced": dict(
            knot_spacing_s=5.0,
            min_observations_per_basis=7.0,
            velocity_weight=0.03,
            adaptive_eta=100_000.0,
            huber_delta_m=80.0,
            jerk_penalty_weight=0.002,
            boundary_acceleration_prior_weight=0.08,
        ),
        "smooth": dict(
            knot_spacing_s=7.0,
            min_observations_per_basis=10.0,
            velocity_weight=0.015,
            adaptive_eta=250_000.0,
            huber_delta_m=120.0,
            jerk_penalty_weight=0.006,
            boundary_acceleration_prior_weight=0.14,
        ),
    }[preset]
    return BSplineCoreConfig(
        degree=3,
        # Report action: prevent very dense short-segment bases and make solver
        # health observable by default.
        min_knot_spacing_s=1.0,
        max_basis_count=900,
        position_weight=2.0,
        penalty_mode="adaptive",
        smoothing_lambda=1.0,
        adaptive_speed_floor_mps=1.0,
        velocity_outlier_policy="position_difference_gate",
        velocity_outlier_gate_mps=_env_float("VELOCITY_OUTLIER_GATE_MPS", 50.0),
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


def _hermite_config(preset: Preset) -> HermiteCoreConfig:
    # Paper-oriented nodal Hermite V-Spline core.  The local segment policy folds
    # per-regime acceleration multipliers into lambda/eta at fit time so these
    # base values remain directly comparable with the B-spline presets.
    # Report action: legacy Hermite was over-trusting ADS-B velocity and hard
    # endpoint velocity constraints.  Keep the backend available as a diagnostic
    # method, but de-trust velocity by default.
    by_preset = {
        "accurate": dict(velocity_weight=0.02, adaptive_eta=60_000.0),
        "balanced": dict(velocity_weight=0.015, adaptive_eta=120_000.0),
        "smooth": dict(velocity_weight=0.008, adaptive_eta=280_000.0),
    }[preset]
    return HermiteCoreConfig(
        penalty_mode="adaptive",
        smoothing_lambda=1.0,
        adaptive_speed_floor_mps=1.0,
        optimize=False,
        compute_loocv_score=False,
        condition_number_max_size=1024,
        hard_endpoint_constraints=True,
        hard_endpoint_positions=True,
        hard_endpoint_velocities=False,
        **by_preset,
    )


def _kalman_rts_config(preset: Preset) -> KalmanRTSConfig:
    # Whole-track state-space comparator.  Presets mirror the spline intent:
    # accurate trusts observations more, smooth trusts the white-jerk process
    # model more, balanced sits between them.
    return default_kalman_rts_config_for_preset(preset)


def _segmentation_config() -> DynamicSegmentationConfig:
    kalman = KalmanSegmentationConfig(
        enabled=_env_bool("KALMAN_SEGMENTATION", True),
        meas_std_xy_m=25.0,
        meas_std_z_m=40.0,
        accel_std_xy_mps2=8.0,
        accel_std_z_mps2=4.0,
        gate_sigma=4.5,
        init_vel_points=5,
        min_observations=4,
        prefer_reported_velocity=True,
        reported_velocity_smoothing_window=5,
    )
    return DynamicSegmentationConfig(
        enabled=_env_bool("DYNAMIC_SEGMENTATION", True),
        hard_gap_s=_env_optional_float("DYNAMIC_HARD_GAP_S", 30.0),
        relative_gap_factor=5.0,
        min_segment_points=_env_int("MIN_SEGMENT_POINTS", 8),
        # Report action: reduce join pressure on local spline backends.  These
        # defaults are intentionally coarser than the previous 20 s / 20 s / 24
        # recipe that over-segmented flight 4BAAD9.
        min_segment_duration_s=_env_float("MIN_SEGMENT_DURATION_S", 15.0),
        min_boundary_spacing_s=_env_float("MIN_BOUNDARY_SPACING_S", 12.0),
        max_segments_per_component=_env_int("MAX_SEGMENTS_PER_COMPONENT", 48),
        prefer_under_segmentation=True,
        enable_motion_spike_boundaries=False,
        enable_pelt_boundaries=False,
        segment_horizontal_turns=_env_bool("SEGMENT_HORIZONTAL_TURNS", True),
        enable_rough_air_segmentation=_env_bool("ENABLE_ROUGH_AIR_SEGMENTATION", False),
        enable_energy_state_segmentation=True,
        energy_state_min_points=_env_int("ENERGY_STATE_MIN_POINTS", 8),
        energy_state_min_duration_s=_env_float("ENERGY_STATE_MIN_DURATION_S", 15.0),
        protect_energy_boundaries=_env_bool("PROTECT_ENERGY_BOUNDARIES", True),
        energy_smoothing_window_points=_env_int("ENERGY_SMOOTHING_WINDOW_POINTS", 5),
        enable_go_around_detection=_env_bool("ENABLE_GO_AROUND_DETECTION", True),
        enable_vertical_reversal_segmentation=_env_bool("ENABLE_VERTICAL_REVERSAL_SEGMENTATION", True),
        vertical_reversal_min_points=_env_int("VERTICAL_REVERSAL_MIN_POINTS", 4),
        vertical_reversal_min_duration_s=_env_float("VERTICAL_REVERSAL_MIN_DURATION_S", 6.0),
        vertical_reversal_min_altitude_excursion_m=_env_float("VERTICAL_REVERSAL_MIN_ALTITUDE_EXCURSION_M", 25.0),
        enable_altitude_lobe_segmentation=_env_bool("ENABLE_ALTITUDE_LOBE_SEGMENTATION", True),
        altitude_lobe_min_points=_env_int("ALTITUDE_LOBE_MIN_POINTS", 8),
        altitude_lobe_min_duration_s=_env_float("ALTITUDE_LOBE_MIN_DURATION_S", 8.0),
        energy_rate_deadband_mps=_env_float("ENERGY_RATE_DEADBAND_MPS", 1.0),
        energy_rate_deadband_scale=_env_float("ENERGY_RATE_DEADBAND_SCALE", 0.20),
        vertical_rate_deadband_mps=_env_float("VERTICAL_RATE_DEADBAND_MPS", 1.0),
        vertical_rate_deadband_scale=_env_float("VERTICAL_RATE_DEADBAND_SCALE", 0.20),
        speed_accel_deadband_mps2=_env_float("SPEED_ACCEL_DEADBAND_MPS2", 0.20),
        speed_accel_deadband_scale=_env_float("SPEED_ACCEL_DEADBAND_SCALE", 0.20),
        turn_rate_deadband_degps=_env_float("TURN_RATE_DEADBAND_DEGPS", 0.75),
        turn_rate_deadband_scale=_env_float("TURN_RATE_DEADBAND_SCALE", 0.20),
        lateral_accel_deadband_mps2=_env_float("LATERAL_ACCEL_DEADBAND_MPS2", 0.5),
        lateral_accel_deadband_scale=_env_float("LATERAL_ACCEL_DEADBAND_SCALE", 0.25),
        rough_air_score_threshold=_env_float("ROUGH_AIR_SCORE_THRESHOLD", 4.5),
        altitude_lobe_min_prominence_m=_env_float("ALTITUDE_LOBE_MIN_PROMINENCE_M", 35.0),
        altitude_lobe_min_side_prominence_m=_env_float("ALTITUDE_LOBE_MIN_SIDE_PROMINENCE_M", 12.0),
        altitude_lobe_gradient_gate_mps=_env_float("ALTITUDE_LOBE_GRADIENT_GATE_MPS", 0.35),
        max_boundary_shift_points=_env_int("MAX_BOUNDARY_SHIFT_POINTS", 6),
        segmentation_feature_source="kalman_rts",
        kalman_segmentation_config=kalman,
    )


def _boundary_config() -> BoundaryStateConfig:
    return BoundaryStateConfig(
        position_source="weighted_compromise",
        # Report action: bias internal joins toward the robust local estimate
        # instead of the single raw ADS-B row.
        position_raw_weight=_env_float("BOUNDARY_POSITION_RAW_WEIGHT", 0.35),
        position_robust_weight=_env_float("BOUNDARY_POSITION_ROBUST_WEIGHT", 0.65),
        window_points=_env_int("BOUNDARY_WINDOW_POINTS", 11),
        min_side_points=_env_int("BOUNDARY_MIN_SIDE_POINTS", 4),
        poly_order=2,
        robust_iters=3,
        huber_k=1.345,
        blend_reported_velocity_weight=_env_float("BOUNDARY_REPORTED_VELOCITY_WEIGHT", 0.0),
        max_velocity_factor=_env_float("BOUNDARY_MAX_VELOCITY_FACTOR", 2.5),
    )

def _local_policy_config(preset: Preset) -> LocalSegmentPolicyConfig:
    # These policy values are intentionally few.  They decide the first guess by
    # flight regime; LocalSegmentTuningConfig then searches a small set around
    # that guess independently for each segment.
    if preset == "accurate":
        return LocalSegmentPolicyConfig(
            steady_adaptive_eta=15_000.0,           # was 24_000
            transition_adaptive_eta=3_500.0,        # was 5_500
            energy_change_adaptive_eta=7_500.0,     # was 12_000
            energy_constant_adaptive_eta=24_000.0,  # was 38_000
            noisy_adaptive_eta=48_000.0,            # was 75_000

            steady_velocity_weight=0.095,           # was 0.072
            transition_velocity_weight=0.090,       # was 0.068
            energy_change_velocity_weight=0.086,    # was 0.064
            energy_constant_velocity_weight=0.070,  # was 0.052
            noisy_velocity_weight=0.036,            # was 0.024
        )

    if preset == "smooth":
        return LocalSegmentPolicyConfig(
            steady_adaptive_eta=62_000.0,           # was 95_000
            transition_adaptive_eta=11_500.0,       # was 18_000
            energy_change_adaptive_eta=29_000.0,    # was 45_000
            energy_constant_adaptive_eta=88_000.0,  # was 135_000
            noisy_adaptive_eta=150_000.0,           # was 220_000

            steady_velocity_weight=0.052,           # was 0.036
            transition_velocity_weight=0.054,       # was 0.037
            energy_change_velocity_weight=0.048,    # was 0.033
            energy_constant_velocity_weight=0.038,  # was 0.025
            noisy_velocity_weight=0.020,            # was 0.012
        )
    return LocalSegmentPolicyConfig()


def _local_tuning_config(preset: Preset) -> LocalSegmentTuningConfig:
    objective = {"accurate": "position", "balanced": "balanced", "smooth": "smooth"}[preset]
    default_candidates = {"accurate": 14, "balanced": 14, "smooth": 10}[preset]
    default_rmse = {"accurate": 90.0, "balanced": 120.0, "smooth": 160.0}[preset]
    default_p95 = {"accurate": 180.0, "balanced": 240.0, "smooth": 320.0}[preset]
    default_max = {"accurate": 450.0, "balanced": 600.0, "smooth": 800.0}[preset]
    default_vertical_rmse = {"accurate": 6.0, "balanced": 8.0, "smooth": 12.0}[preset]
    default_vertical_p95 = {"accurate": 12.0, "balanced": 16.0, "smooth": 24.0}[preset]
    default_vertical_max = {"accurate": 22.0, "balanced": 28.0, "smooth": 45.0}[preset]
    default_vertical_window = {"accurate": 7.5, "balanced": 10.0, "smooth": 16.0}[preset]
    return LocalSegmentTuningConfig(
        enabled=_env_bool("SEGMENT_TUNING", True),
        objective=objective,  # type: ignore[arg-type]
        max_candidates=_env_int("SEGMENT_TUNING_MAX_CANDIDATES", default_candidates),
        include_all_candidate_reports=_env_bool("SEGMENT_TUNING_REPORT_CANDIDATES", True),
        join_velocity_harmonization=_env_bool("JOIN_VELOCITY_HARMONIZATION", True),
        adaptive_resegmentation_enabled=_env_bool("ADAPTIVE_RESEGMENTATION", True),
        adaptive_resegmentation_max_passes=_env_int("ADAPTIVE_RESEGMENTATION_MAX_PASSES", 2),
        adaptive_resegmentation_bad_rmse_m=_env_float("BAD_SEGMENT_RMSE_M", default_rmse),
        adaptive_resegmentation_bad_p95_m=_env_float("BAD_SEGMENT_P95_M", default_p95),
        adaptive_resegmentation_bad_max_m=_env_float("BAD_SEGMENT_MAX_M", default_max),
        adaptive_resegmentation_bad_vertical_rmse_m=_env_float("BAD_SEGMENT_VERTICAL_RMSE_M", default_vertical_rmse),
        adaptive_resegmentation_bad_vertical_p95_m=_env_float("BAD_SEGMENT_VERTICAL_P95_M", default_vertical_p95),
        adaptive_resegmentation_bad_vertical_max_m=_env_float("BAD_SEGMENT_VERTICAL_MAX_M", default_vertical_max),
        adaptive_resegmentation_bad_vertical_window_m=_env_float("BAD_SEGMENT_VERTICAL_WINDOW_M", default_vertical_window),
        adaptive_resegmentation_vertical_run_min_points=_env_int("ADAPTIVE_RESEGMENTATION_VERTICAL_RUN_MIN_POINTS", 3),
        adaptive_resegmentation_vertical_run_min_duration_s=_env_float("ADAPTIVE_RESEGMENTATION_VERTICAL_RUN_MIN_DURATION_S", 3.0),
        adaptive_resegmentation_min_points=_env_int("ADAPTIVE_RESEGMENTATION_MIN_POINTS", 8),
        adaptive_resegmentation_min_duration_s=_env_float("ADAPTIVE_RESEGMENTATION_MIN_DURATION_S", 8.0),
        adaptive_resegmentation_min_boundary_spacing_s=_env_float("ADAPTIVE_RESEGMENTATION_MIN_BOUNDARY_SPACING_S", 8.0),
        adaptive_resegmentation_max_segments_per_component=_env_int("ADAPTIVE_RESEGMENTATION_MAX_SEGMENTS_PER_COMPONENT", 18),
    )


def build_config() -> TrackOutputPipelineConfig:
    preset = _preset()
    output_presets = _preset_tuple()
    db_path = _resolve(Path(os.getenv("ADSB_SQLITE_PATH", "adsb_raw.sqlite")))
    output_dir = _resolve(Path(os.getenv("TRACK_OUTPUT_DIR", "track_output")))
    log_dir = _resolve(Path(os.getenv("TRACK_LOG_DIR", "logs")))
    rules_path = _resolve_optional(Path(os.getenv("TRACK_RULES_PATH"))) if os.getenv("TRACK_RULES_PATH") else None

    default_v_spline_backends = (
        "bspline_piecewise",
        "hermite_piecewise",
        "bspline_component_global",
        "bspline_overlap",
        "bspline_join_smooth",
        "hermite_stable",
        "quintic_bspline",
    )
    if _env_bool("V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR", False):
        default_v_spline_backends = (*default_v_spline_backends, "quintic_kalman_boundary")

    return TrackOutputPipelineConfig(
        paths=PipelinePaths(
            database_path=db_path,
            output_dir=output_dir,
            rules_path=rules_path,
        ),
        log_dir=log_dir,
        icao_list=_icao_list(),
        clean_output_dir=_env_bool("CLEAN_OUTPUT_DIR", False),
        v_spline_time_step_s=_env_float("V_SPLINE_TIME_STEP_S", 0.25),
        v_spline_output_frequency_hz=_env_float("V_SPLINE_OUTPUT_FREQUENCY_HZ", 4.0),
        write_debug_artifacts=_env_bool("WRITE_DEBUG_ARTIFACTS", True),
        dynamic_segmentation_config=_segmentation_config(),
        boundary_state_config=_boundary_config(),
        local_segment_policy_config=_local_policy_config(preset),
        local_segment_tuning_config=_local_tuning_config(preset),
        show_progress=_env_bool("PROGRESS", True),
        kalman_rts_output_enabled=_env_bool("KALMAN_RTS_OUTPUT", True),
        kalman_rts_output_presets=output_presets,
        v_spline_output_backends=_env_str_tuple("V_SPLINE_OUTPUT_BACKENDS", default_v_spline_backends),
        v_spline_output_presets=output_presets,
        use_kalman_boundary_prior=_env_bool("V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR", False),
        event_aware_evaluation_enabled=_env_bool("EVENT_AWARE_EVALUATION", True),
        holdout_evaluation_fraction=_env_float("HOLDOUT_EVALUATION_FRACTION", 0.15),
        bspline_config=_bspline_config(preset),
        hermite_config=_hermite_config(preset),
        kalman_rts_config=_kalman_rts_config(preset),
        bspline_config_by_preset={p: _bspline_config(p) for p in PRESETS},
        hermite_config_by_preset={p: _hermite_config(p) for p in PRESETS},
        kalman_rts_config_by_preset={p: _kalman_rts_config(p) for p in PRESETS},
        local_segment_policy_config_by_preset={p: _local_policy_config(p) for p in PRESETS},
        local_segment_tuning_config_by_preset={p: _local_tuning_config(p) for p in PRESETS},
        adapter_config=RawKeyframeVSplineAdapterConfig(
            max_gap_s=None,
            min_segment_observations=2,
            fail_on_short_segment=True,
            duplicate_time_tolerance_s=0.0,
            enable_position_motion_outlier_filter=True,
            position_outlier_speed_gate_mps=1200.0,
            position_outlier_speed_factor=5.0,
            position_outlier_max_iterations=4,
        ),
    )


def main() -> None:
    summary = TrackOutputPipeline(build_config()).run()
    print(json.dumps(_clean_json(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
