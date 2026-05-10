from __future__ import annotations

import math

import numpy as np

from raw_keyframe_vspline_adapter import PreparedVSplineSample
from trajectory_segmentation import DynamicSegmentationConfig, _choose_boundary_sample_for_transition, segment_prepared_samples


def _samples_from_velocity(t: np.ndarray, v: np.ndarray, z0: float = 1000.0) -> list[PreparedVSplineSample]:
    y = np.zeros_like(v, dtype=float)
    y[0, 2] = z0
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        y[i] = y[i - 1] + 0.5 * (v[i - 1] + v[i]) * dt
    return [
        PreparedVSplineSample(
            keyframe_id=f"kf_{i:04d}",
            t=float(ti),
            y=tuple(map(float, yi)),
            v=tuple(map(float, vi)),
            raw_index=i,
        )
        for i, (ti, yi, vi) in enumerate(zip(t, y, v))
    ]


def _base_config(**kwargs) -> DynamicSegmentationConfig:
    base = dict(
        enabled=True,
        segmentation_feature_source="raw",
        min_segment_points=5,
        min_segment_duration_s=5.0,
        min_boundary_spacing_s=5.0,
        energy_state_min_points=5,
        energy_state_min_duration_s=5.0,
        energy_smoothing_window_points=3,
        enable_go_around_detection=False,
        enable_pelt_boundaries=False,
        enable_motion_spike_boundaries=False,
        segment_horizontal_turns=True,
        turn_rate_deadband_degps=1.0,
        turn_min_heading_change_deg=10.0,
        lateral_accel_deadband_mps2=0.4,
        enable_rough_air_segmentation=True,
        rough_air_score_threshold=3.0,
    )
    base.update(kwargs)
    return DynamicSegmentationConfig(**base)


def test_boundary_optimizer_caps_large_transition_band_shift() -> None:
    raw = np.zeros((100, 3), dtype=float)
    names = [
        "velocity_mismatch_mps",
        "local_residual_proxy_m",
        "acceleration_mps2",
        "vertical_acceleration_mps2",
        "speed_acceleration_mps2",
    ]
    z = np.zeros((100, len(names)), dtype=float)
    z[:, :] = 5.0
    # Make the far left edge look artificially attractive.  Without the shift cap
    # the optimizer can choose it; with the cap it must stay near the actual state
    # transition at index 80.
    z[10, :] = 0.0
    z[80, :] = 1.0

    selected, report = _choose_boundary_sample_for_transition(
        left={"end_idx": 10},
        right={"start_idx": 80},
        raw=raw,
        z=z,
        names=names,
        config=_base_config(boundary_search_padding_points=3, max_boundary_shift_points=12),
    )

    assert selected >= 68
    assert abs(selected - 80) <= 12
    assert report["search_was_clipped"] is True
    assert report["max_shift_points"] == 12


def test_sustained_level_turn_gets_regime_segment() -> None:
    t = np.arange(0.0, 91.0, 1.0)
    speed = 90.0
    heading = np.zeros_like(t)
    turn = (t >= 30.0) & (t <= 60.0)
    heading[turn] = np.deg2rad((t[turn] - 30.0) * 2.5)
    heading[t > 60.0] = np.deg2rad(75.0)
    v = np.column_stack([speed * np.cos(heading), speed * np.sin(heading), np.zeros_like(t)])
    samples = _samples_from_velocity(t, v)

    components, diag = segment_prepared_samples(samples, _base_config())

    labels = [seg.regime_label for comp in components for seg in comp.segments]
    reasons = [reason for comp in components for b in comp.boundaries for reason in b.reasons]
    assert diag["boundary_count"] >= 2
    assert any("turn_" in label for label in labels)
    assert "turn_state_transition" in reasons


def test_small_heading_jitter_does_not_split_straight_energy_constant_track() -> None:
    t = np.arange(0.0, 80.0, 1.0)
    speed = 100.0
    heading = np.deg2rad(0.15 * np.sin(t / 2.0))
    v = np.column_stack([speed * np.cos(heading), speed * np.sin(heading), np.zeros_like(t)])
    samples = _samples_from_velocity(t, v)

    components, diag = segment_prepared_samples(samples, _base_config())

    assert diag["segment_count"] == 1
    assert len(components[0].boundaries) == 0


def test_climb_transition_still_gets_energy_boundary() -> None:
    t = np.arange(0.0, 75.0, 1.0)
    speed = 85.0
    vz = np.zeros_like(t)
    vz[t >= 35.0] = 4.0
    v = np.column_stack([np.full_like(t, speed), np.zeros_like(t), vz])
    samples = _samples_from_velocity(t, v)

    components, diag = segment_prepared_samples(samples, _base_config(segment_horizontal_turns=False))

    labels = [seg.regime_label for comp in components for seg in comp.segments]
    reasons = [reason for comp in components for b in comp.boundaries for reason in b.reasons]
    assert diag["boundary_count"] >= 1
    assert any("climb" in label or "energy_gain" in label for label in labels)
    assert "energy_state_transition" in reasons


def test_rough_air_like_middle_window_is_labeled_without_motion_spike_detector() -> None:
    t = np.arange(0.0, 90.0, 1.0)
    speed = 95.0
    v = np.column_stack([np.full_like(t, speed), np.zeros_like(t), np.zeros_like(t)])
    samples = _samples_from_velocity(t, v)
    rough_samples = []
    for i, sample in enumerate(samples):
        y = np.asarray(sample.y, dtype=float)
        vv = np.asarray(sample.v, dtype=float)
        if 35 <= i <= 55:
            # Deterministic position/velocity inconsistency over a sustained
            # window.  This is a rough-air/surveillance-noise proxy, not a turn.
            y = y + np.array([35.0 * math.sin(i), 20.0 * math.cos(0.7 * i), 15.0 * math.sin(0.5 * i)])
            vv = vv + np.array([12.0 * math.sin(0.4 * i), 8.0 * math.cos(0.3 * i), 5.0 * math.sin(0.6 * i)])
        rough_samples.append(
            PreparedVSplineSample(sample.keyframe_id, sample.t, tuple(map(float, y)), tuple(map(float, vv)), sample.row_ids, sample.source_event_ids, sample.raw_index)
        )

    components, diag = segment_prepared_samples(rough_samples, _base_config(segment_horizontal_turns=False, rough_air_score_threshold=1.2))

    labels = [seg.regime_label for comp in components for seg in comp.segments]
    assert diag["boundary_count"] >= 1
    assert any("rough_air" in label or "surveillance_noisy" in label for label in labels)


def test_large_climb_to_level_or_descent_lobe_gets_vertical_reversal_boundary() -> None:
    t = np.arange(0.0, 86.0, 1.0)
    speed = 90.0
    vz = np.zeros_like(t)
    vz[(t >= 10.0) & (t <= 35.0)] = 5.0
    vz[(t > 35.0) & (t <= 65.0)] = -1.5
    v = np.column_stack([np.full_like(t, speed), np.zeros_like(t), vz])
    samples = _samples_from_velocity(t, v, z0=600.0)

    components, diag = segment_prepared_samples(
        samples,
        _base_config(
            segment_horizontal_turns=False,
            enable_rough_air_segmentation=False,
            enable_go_around_detection=False,
            enable_vertical_reversal_segmentation=True,
            min_segment_points=8,
            min_segment_duration_s=15.0,
            min_boundary_spacing_s=10.0,
            energy_state_min_points=20,
            energy_state_min_duration_s=30.0,
            vertical_reversal_min_altitude_excursion_m=25.0,
        ),
    )

    reasons = [reason for comp in components for b in comp.boundaries for reason in b.reasons]
    assert diag["boundary_count"] >= 1
    assert "vertical_reversal_transition" in reasons
    assert any("climb_to" in reason for reason in reasons)


def test_energy_constant_speed_height_exchange_gets_altitude_lobe_boundary() -> None:
    t = np.arange(0.0, 181.0, 1.0)
    base_speed = 95.0
    # The aircraft climbs while decelerating, then descends while accelerating.
    # Total specific energy is nearly constant, so an energy-only label can call
    # this whole interval energy_constant.  The altitude-lobe detector should
    # still split the climb/go-around-like state from the following descent.
    z = np.zeros_like(t, dtype=float) + 700.0
    z[(t >= 30.0) & (t <= 85.0)] = 700.0 + 2.0 * (t[(t >= 30.0) & (t <= 85.0)] - 30.0)
    z[t > 85.0] = 810.0 - 1.4 * (t[t > 85.0] - 85.0)
    z = np.maximum(z, 690.0)
    vz = np.gradient(z, t)
    # Reported speed trades against altitude so total-energy labeling is weak.
    speed = base_speed - 0.18 * (z - 700.0)
    speed = np.maximum(speed, 65.0)
    v = np.column_stack([speed, np.zeros_like(t), vz])
    y = np.zeros_like(v)
    y[:, 0] = np.cumsum(np.r_[0.0, 0.5 * (speed[:-1] + speed[1:]) * np.diff(t)])
    y[:, 2] = z
    samples = [
        PreparedVSplineSample(
            keyframe_id=f"kf_{i:04d}",
            t=float(ti),
            y=tuple(map(float, yi)),
            v=tuple(map(float, vi)),
            raw_index=i,
        )
        for i, (ti, yi, vi) in enumerate(zip(t, y, v))
    ]

    components, diag = segment_prepared_samples(
        samples,
        _base_config(
            segmentation_feature_source="raw",
            segment_horizontal_turns=False,
            enable_rough_air_segmentation=False,
            enable_go_around_detection=False,
            enable_vertical_reversal_segmentation=False,
            enable_altitude_lobe_segmentation=True,
            min_segment_points=10,
            min_segment_duration_s=20.0,
            min_boundary_spacing_s=15.0,
            energy_state_min_points=200,
            energy_state_min_duration_s=200.0,
            altitude_lobe_min_prominence_m=45.0,
            altitude_lobe_min_side_prominence_m=20.0,
            altitude_lobe_gradient_gate_mps=0.3,
        ),
    )

    reasons = [reason for comp in components for b in comp.boundaries for reason in b.reasons]
    boundaries = [b.local_sample_index for comp in components for b in comp.boundaries]
    assert diag["boundary_count"] >= 1
    assert "altitude_lobe_transition" in reasons
    assert any(abs(idx - 86) <= 6 for idx in boundaries)
