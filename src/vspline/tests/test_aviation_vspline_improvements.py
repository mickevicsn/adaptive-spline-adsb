from __future__ import annotations

import numpy as np

from boundary_state import SharedBoundaryState
from raw_keyframe_vspline_adapter import PreparedVSplineSample
from track_output_pipeline import (
    TrackOutputPipeline,
    TrackOutputPipelineConfig,
    _event_aware_component_continuity,
    _v_spline_output_specs,
)
from trajectory_segmentation import AcceptedBoundary, DynamicSegment, HardGapComponent, SegmentedComponent
from vspline.bspline_core import BSplineCoreConfig, BSplineCoreInput, BSplinePositionPrior, fit_b_spline_component
from vspline.local_tuning import LocalSegmentTuningConfig
from vspline.velocity_confidence import compute_velocity_confidence_scale


def _samples(n: int = 36) -> tuple[PreparedVSplineSample, ...]:
    t = np.arange(float(n))
    y = np.column_stack([90.0 * t, 30.0 * np.sin(t / 6.0), 100.0 + 0.2 * t])
    v = np.column_stack([np.full_like(t, 90.0), 5.0 * np.cos(t / 6.0), np.full_like(t, 0.2)])
    return tuple(
        PreparedVSplineSample(
            keyframe_id=f"kf_{i:03d}",
            t=float(ti),
            y=tuple(map(float, yi)),
            v=tuple(map(float, vi)),
            raw_index=i,
        )
        for i, (ti, yi, vi) in enumerate(zip(t, y, v))
    )


def test_velocity_confidence_scaling_downweights_stale_adsb_velocity() -> None:
    t = np.arange(0.0, 12.0)
    y = np.column_stack([80.0 * t, np.zeros_like(t), np.zeros_like(t)])
    v = np.column_stack([np.full_like(t, 80.0), np.zeros_like(t), np.zeros_like(t)])
    v[5] = np.array([-200.0, 150.0, 60.0])

    scale, report = compute_velocity_confidence_scale(t, y, v)

    assert report["enabled"] is True
    assert scale[5] < 0.25
    assert report["downweighted_count"] >= 1


def test_soft_boundary_position_prior_does_not_create_hard_raw_anchor() -> None:
    t = np.linspace(0.0, 20.0, 31)
    y = np.column_stack([70.0 * t, 0.1 * t * t, 50.0 + 0.2 * t])
    v = np.column_stack([np.full_like(t, 70.0), 0.2 * t, np.full_like(t, 0.2)])
    robust_position = y[15] + np.array([5.0, -2.0, 1.0])

    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y,
            v=v,
            position_priors=(
                BSplinePositionPrior(
                    prior_id="boundary_position",
                    t=float(t[15]),
                    position=robust_position,
                    weight=1.0,
                    confidence=0.8,
                    source="robust_boundary_state",
                ),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=4.0,
            velocity_weight=0.02,
            adaptive_eta=1_000.0,
            hard_boundary_positions=False,
            boundary_position_prior_weight=1.0,
            boundary_velocity_prior_weight=0.0,
            robust_position_loss="none",
        ),
    )

    assert fit.diagnostics["n_anchors"] == 0
    assert fit.diagnostics["boundary_position_priors"]["count"] == 1
    assert fit.diagnostics["solver"]["position_prior_row_count"] == 1


def test_hermite_stable_config_softens_velocity_and_disables_hard_endpoint_velocities() -> None:
    cfg = TrackOutputPipelineConfig()
    spec = _v_spline_output_specs(
        TrackOutputPipelineConfig(v_spline_output_backends=("hermite_stable",), v_spline_output_presets=("smooth",))
    )[0]
    method_cfg = TrackOutputPipeline(cfg)._method_config_for_spec(spec)

    assert method_cfg.hermite_config.hard_endpoint_velocities is False
    assert method_cfg.hermite_config.optimize is False
    assert method_cfg.hermite_config.velocity_weight <= 0.004
    assert method_cfg.hermite_config.adaptive_speed_floor_mps >= 25.0


def test_overlap_method_config_uses_robust_hard_c0_boundary_anchor() -> None:
    spec = _v_spline_output_specs(
        TrackOutputPipelineConfig(v_spline_output_backends=("bspline_overlap",), v_spline_output_presets=("balanced",))
    )[0]
    method_cfg = TrackOutputPipeline(TrackOutputPipelineConfig())._method_config_for_spec(spec)

    assert method_cfg.bspline_config.hard_boundary_positions is True
    assert method_cfg.bspline_config.boundary_position_prior_weight == 0.0
    assert method_cfg.boundary_state_config.position_raw_weight == 0.35
    assert method_cfg.boundary_state_config.position_robust_weight == 0.65
    assert method_cfg.boundary_state_config.blend_reported_velocity_weight == 0.0


def test_global_component_backend_disables_dynamic_join_fitting_policy() -> None:
    spec = _v_spline_output_specs(
        TrackOutputPipelineConfig(v_spline_output_backends=("bspline_component_global",), v_spline_output_presets=("balanced",))
    )[0]
    method_cfg = TrackOutputPipeline(TrackOutputPipelineConfig())._method_config_for_spec(spec)

    assert spec.method_id == "aviation_v_spline_bspline_global_balanced"
    assert method_cfg.bspline_config.degree == 3
    assert method_cfg.bspline_config.backend_name == "aviation_v_spline_bspline_global_balanced"
    assert method_cfg.local_segment_tuning_config.join_velocity_harmonization is False
    assert method_cfg.local_segment_tuning_config.adaptive_resegmentation_enabled is False


def test_quintic_backend_uses_degree_five_and_schema_method_id() -> None:
    spec = _v_spline_output_specs(
        TrackOutputPipelineConfig(v_spline_output_backends=("quintic_bspline",), v_spline_output_presets=("balanced",))
    )[0]
    cfg = TrackOutputPipelineConfig()
    method_cfg = TrackOutputPipeline(cfg)._method_config_for_spec(spec)

    assert spec.backend == "quintic_bspline"
    assert spec.method_id == "aviation_v_spline_quintic_balanced"
    assert method_cfg.bspline_config.degree == 5
    assert method_cfg.bspline_config.jerk_penalty_weight > 0.0
    assert method_cfg.bspline_config.snap_penalty_weight > 0.0


def test_event_aware_continuity_masks_true_discontinuity_from_normal_join_bucket() -> None:
    samples = _samples(20)
    component = HardGapComponent("comp_0001", 0, 19, samples)
    boundary = AcceptedBoundary(
        boundary_id="b1",
        component_id="comp_0001",
        sample_index=10,
        local_sample_index=10,
        t_boundary=float(samples[10].t),
        reasons=("surveillance_discontinuity",),
        score=100.0,
        is_hard_gap=False,
    )
    left = DynamicSegment("left", "comp_0001", 0, 10, samples[:11], samples[0].t, samples[10].t, {}, "airborne", end_boundary_id="b1")
    right = DynamicSegment("right", "comp_0001", 10, 19, samples[10:], samples[10].t, samples[-1].t, {}, "airborne", start_boundary_id="b1")
    segmented = SegmentedComponent(component, (boundary,), (left, right), {})
    continuity = {
        "boundaries": [
            {
                "left_segment_id": "left",
                "right_segment_id": "right",
                "position_jump_m": 100.0,
                "velocity_jump_mps": 25.0,
                "acceleration_jump_mps2": 10.0,
                "jerk_jump_mps3": 50.0,
            }
        ]
    }

    report = _event_aware_component_continuity(segmented, continuity)

    assert report["normal_segment_joins"]["count"] == 0
    assert report["by_event"]["surveillance_or_track_discontinuity"]["count"] == 1
    assert report["rows"][0]["excluded_from_normal_continuity_score"] is True


def test_overlap_save_does_not_borrow_across_true_discontinuity_boundary() -> None:
    samples = _samples(36)
    component = HardGapComponent("comp_0001", 0, 35, samples)
    boundary = AcceptedBoundary(
        boundary_id="b1",
        component_id="comp_0001",
        sample_index=18,
        local_sample_index=18,
        t_boundary=float(samples[18].t),
        reasons=("surveillance_discontinuity",),
        score=100.0,
        is_hard_gap=False,
    )
    left = DynamicSegment("left", "comp_0001", 0, 18, samples[:19], samples[0].t, samples[18].t, {"median_horizontal_speed_mps": 90.0}, "airborne", end_boundary_id="b1")
    right = DynamicSegment("right", "comp_0001", 18, 35, samples[18:], samples[18].t, samples[-1].t, {"median_horizontal_speed_mps": 90.0}, "airborne", start_boundary_id="b1")
    segmented = SegmentedComponent(component, (boundary,), (left, right), {})
    shared_state = SharedBoundaryState(
        boundary_id="b1",
        t_boundary=float(samples[18].t),
        position_m=samples[18].y,
        velocity_mps=samples[18].v,
        confidence=0.8,
        method="test_boundary_state",
        diagnostics={},
        acceleration_mps2=(0.0, 0.0, 0.0),
    )
    cfg = TrackOutputPipelineConfig(
        show_progress=False,
        holdout_evaluation_fraction=0.0,
        local_segment_tuning_config=LocalSegmentTuningConfig(enabled=False, join_velocity_harmonization=False, adaptive_resegmentation_enabled=False),
    )
    spec = _v_spline_output_specs(
        TrackOutputPipelineConfig(v_spline_output_backends=("bspline_overlap",), v_spline_output_presets=("balanced",))
    )[0]
    method_cfg = TrackOutputPipeline(cfg)._method_config_for_spec(spec)

    result = TrackOutputPipeline(method_cfg)._fit_component_with_local_b_spline(
        flight_id="synthetic",
        prepared_samples=list(samples),
        segmented_component=segmented,
        shared_states={"b1": shared_state},
        backend="bspline_overlap",
    )

    left_diag = result["segment_metadata"]["left"]["diagnostics"]["input_metadata"]["overlap_save"]
    right_diag = result["segment_metadata"]["right"]["diagnostics"]["input_metadata"]["overlap_save"]
    assert left_diag["enabled"] is True
    assert left_diag["borrowed_after_count"] == 0
    assert right_diag["borrowed_before_count"] == 0


def test_reference_free_trajectory_model_metrics_are_emitted_without_truth_data() -> None:
    from vspline.quality import evaluate_segment_quality

    samples = _samples(32)
    segment = DynamicSegment(
        "seg_metrics",
        "comp_metrics",
        0,
        len(samples) - 1,
        samples,
        float(samples[0].t),
        float(samples[-1].t),
        {"median_horizontal_speed_mps": 90.0},
        "airborne",
    )
    t = np.asarray([s.t for s in samples], dtype=float)
    y = np.asarray([s.y for s in samples], dtype=float)
    v = np.asarray([s.v for s in samples], dtype=float)
    fit = fit_b_spline_component(
        BSplineCoreInput(t=t, y=y, v=v, dim_names=("x", "y", "z")),
        BSplineCoreConfig(
            degree=5,
            knot_spacing_s=6.0,
            velocity_weight=0.03,
            adaptive_eta=10_000.0,
            hard_boundary_positions=False,
            boundary_velocity_prior_weight=0.0,
            jerk_penalty_weight=0.02,
            snap_penalty_weight=0.001,
            robust_position_loss="none",
        ),
    )

    quality = evaluate_segment_quality(segment, fit, render_step_s=0.5)
    metrics = quality.trajectory_model_metrics

    assert metrics["enabled"] is True
    assert metrics["truth_data_used"] is False
    assert metrics["metric_family"] == "reference_free_trajectory_model_metrics_v1"
    assert 0.0 <= metrics["weighted_score_0_100"] <= 100.0
    assert metrics["velocity_evidence"]["enabled"] is True
    assert metrics["smoothness"]["enabled"] is True
    assert "velocity_detail_retention_ratio" in metrics["dynamic_detail_preservation"]


def test_trajectory_model_aggregate_penalizes_rendering_through_hard_raw_gap() -> None:
    from vspline.quality import aggregate_trajectory_model_metrics

    segment_entry = {
        "segment_id": "seg",
        "t0": 0.0,
        "t1": 61.0,
        "n_observations": 5,
        "quality": {
            "trajectory_model_metrics": {
                "enabled": True,
                "weighted_score_0_100": 90.0,
                "regime_bucket": "airborne",
                "component_scores_0_100": {
                    "observation_position_score": 90.0,
                    "velocity_evidence_score": 90.0,
                    "finite_difference_kinematics_score": 90.0,
                    "trajectory_smoothness_score": 90.0,
                    "physical_plausibility_score": 90.0,
                    "dynamic_detail_preservation_score": 90.0,
                    "derivative_closure_score": 90.0,
                },
            }
        },
    }
    raw_times = [0.0, 1.0, 2.0, 60.0, 61.0]
    bridged = aggregate_trajectory_model_metrics(
        [segment_entry],
        raw_times=raw_times,
        render_times=list(np.arange(0.0, 62.0, 1.0)),
        fit_mode="whole_track_kalman_filter_plus_rts_smoother_no_segmentation",
        reconstruction_backend="kalman_rts",
    )
    honest = aggregate_trajectory_model_metrics(
        [segment_entry],
        raw_times=raw_times,
        render_times=[0.0, 1.0, 2.0, 60.0, 61.0],
        fit_mode="segmented_local_b_spline_overlap_save_render_trusted_interiors_only",
        reconstruction_backend="bspline_overlap",
    )

    assert bridged["truth_data_used"] is False
    assert bridged["hard_gap_honesty"]["hard_gap_count"] == 1
    assert bridged["hard_gap_honesty"]["bridged_gap_count"] == 1
    assert bridged["method_component_scores_0_100"]["hard_gap_honesty_score"] < 100.0
    assert honest["hard_gap_honesty"]["bridged_gap_count"] == 0
    assert honest["method_component_scores_0_100"]["hard_gap_honesty_score"] == 100.0
    assert honest["weighted_score_0_100"] > bridged["weighted_score_0_100"]
