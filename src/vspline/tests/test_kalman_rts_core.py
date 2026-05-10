from __future__ import annotations

import numpy as np

from kalman_rts_core import KalmanRTSInput, default_kalman_rts_config_for_preset, fit_kalman_rts_component
from raw_keyframe_vspline_adapter import PreparedVSplineSample
from track_output_pipeline import TrackOutputPipeline, TrackOutputPipelineConfig


def _reference_track(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = t - t[0]
    y = np.column_stack(
        [
            92.0 * tau,
            0.35 * tau * tau,
            750.0 + 0.08 * tau * tau,
        ]
    )
    v = np.column_stack(
        [
            np.full_like(tau, 92.0),
            0.70 * tau,
            0.16 * tau,
        ]
    )
    return y, v


def test_kalman_rts_core_evaluates_finite_whole_track_state() -> None:
    t = np.linspace(0.0, 45.0, 46)
    y, v = _reference_track(t)
    fit = fit_kalman_rts_component(
        KalmanRTSInput(t=t, y=y, v=v),
        default_kalman_rts_config_for_preset("accurate"),
        component_id="synthetic_whole_track",
    )

    grid = np.linspace(t[0], t[-1], 121)
    for deriv in (0, 1, 2, 3):
        evaluated = fit.evaluate(grid, deriv=deriv)
        assert evaluated.shape == (grid.size, 3)
        assert np.all(np.isfinite(evaluated))

    assert fit.diagnostics["backend"] == "kalman_rts"
    assert fit.diagnostics["segmentation_used"] is False
    assert fit.diagnostics["state_dimension"] == 9
    assert fit.lambda_intervals.shape == (t.size - 1,)
    assert fit.diagnostics["position_residual_rmse_3d_m"] < 2.0


def test_kalman_rts_pipeline_fit_uses_single_renderer_wrapper_without_segmentation() -> None:
    t = np.linspace(10.0, 35.0, 26)
    y, v = _reference_track(t)
    samples = [
        PreparedVSplineSample(
            keyframe_id=f"kf_{i:03d}",
            t=float(ti),
            y=tuple(map(float, yi)),
            v=tuple(map(float, vi)),
            raw_index=i,
        )
        for i, (ti, yi, vi) in enumerate(zip(t, y, v))
    ]
    cfg = TrackOutputPipelineConfig(show_progress=False)
    result = TrackOutputPipeline(cfg)._fit_kalman_rts_components(
        flight_id="synthetic",
        prepared_samples=samples,
        method_config=cfg,
    )

    assert len(result["fits"]) == 1
    assert result["piecewise_report"]["enabled"] is False
    assert result["piecewise_report"]["mode"] == "kalman_rts_whole_track"
    assert result["piecewise_report"]["segmentation_diagnostics"]["dynamic_segmentation_applied"] is False
    assert result["piecewise_report"]["segmentation_diagnostics"]["hard_gap_splitting_applied"] is False

    segment_id, segment, fit = result["fits"][0]
    meta = result["segment_metadata"][segment_id]
    assert segment.start_sample_index == 0
    assert segment.end_sample_index == len(samples) - 1
    assert meta["dynamic_segmentation_applied"] is False
    assert meta["fit_mode"] == "one_whole_track_state_space_smoother_no_segmentation"
    assert fit.diagnostics["segmentation_used"] is False
