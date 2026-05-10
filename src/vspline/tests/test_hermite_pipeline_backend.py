from __future__ import annotations

import numpy as np

from raw_keyframe_vspline_adapter import PreparedVSplineSample
from track_output_pipeline import TrackOutputPipeline, TrackOutputPipelineConfig
from trajectory_segmentation import DynamicSegment, HardGapComponent, SegmentedComponent
from vspline.local_tuning import LocalSegmentTuningConfig


def test_piecewise_pipeline_can_fit_hermite_backend_with_same_component_path() -> None:
    t = np.linspace(0.0, 20.0, 21)
    y = np.column_stack([80.0 * t, 20.0 * np.sin(t / 5.0), 100.0 + 0.2 * t])
    v = np.column_stack([np.full_like(t, 80.0), 4.0 * np.cos(t / 5.0), np.full_like(t, 0.2)])
    samples = tuple(
        PreparedVSplineSample(
            keyframe_id=f"kf_{i:03d}",
            t=float(ti),
            y=tuple(map(float, yi)),
            v=tuple(map(float, vi)),
            raw_index=i,
        )
        for i, (ti, yi, vi) in enumerate(zip(t, y, v))
    )
    component = HardGapComponent(
        component_id="comp_0001",
        start_sample_index=0,
        end_sample_index=len(samples) - 1,
        samples=samples,
    )
    segment = DynamicSegment(
        segment_id="comp_0001_seg_0001",
        component_id=component.component_id,
        start_sample_index=0,
        end_sample_index=len(samples) - 1,
        samples=samples,
        t0=float(t[0]),
        t1=float(t[-1]),
        features={},
        regime_label="steady",
    )
    segmented = SegmentedComponent(component=component, boundaries=(), segments=(segment,))
    cfg = TrackOutputPipelineConfig(
        show_progress=False,
        local_segment_tuning_config=LocalSegmentTuningConfig(
            enabled=False,
            join_velocity_harmonization=False,
            adaptive_resegmentation_enabled=False,
        ),
    )

    result = TrackOutputPipeline(cfg)._fit_component_with_local_b_spline(
        flight_id="synthetic",
        prepared_samples=list(samples),
        segmented_component=segmented,
        shared_states={},
        backend="hermite_piecewise",
    )

    assert result["piecewise_component"]["solver"]["backend"] == "hermite_piecewise"
    assert len(result["fits"]) == 1
    fit = result["fits"][0][2]
    assert fit.diagnostics["backend"] == "hermite_piecewise"
    assert fit.diagnostics["n_basis"] == 2 * len(samples)
    assert np.all(np.isfinite(fit.evaluate(t, deriv=2)))
