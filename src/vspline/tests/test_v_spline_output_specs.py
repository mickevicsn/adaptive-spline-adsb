from __future__ import annotations

from track_output_pipeline import TrackOutputPipelineConfig, _kalman_rts_output_specs, _synthetic_gap_output_specs, _v_spline_output_specs


def test_default_v_spline_output_specs_emit_compact_representative_methods() -> None:
    specs = _v_spline_output_specs(TrackOutputPipelineConfig())
    method_ids = [s.method_id for s in specs]
    assert method_ids == [
        "aviation_v_spline_quintic_balanced",
        "aviation_v_spline_quintic_accurate",
        "aviation_v_spline_quintic_smooth",
        "v_spline_bspline_overlap_balanced",
        "v_spline_bspline_overlap_accurate",
        "v_spline_bspline_overlap_smooth",
    ]
    assert all(s.file_stem == s.method_id for s in specs)


def test_new_v_spline_output_specs_emit_required_experimental_variants() -> None:
    cfg = TrackOutputPipelineConfig(
        v_spline_output_backends=(
            "bspline_component_global",
            "bspline_overlap",
            "bspline_join_smooth",
            "hermite_stable",
            "quintic_bspline",
            "quintic_kalman_boundary",
        ),
        v_spline_output_presets=("accurate",),
    )
    specs = _v_spline_output_specs(cfg)
    assert [s.method_id for s in specs] == [
        "aviation_v_spline_bspline_global_accurate",
        "v_spline_bspline_overlap_accurate",
        "v_spline_bspline_join_smooth_accurate",
        "v_spline_hermite_stable_accurate",
        "aviation_v_spline_quintic_accurate",
        "aviation_v_spline_quintic_kalman_boundary_accurate",
    ]


def test_global_backend_aliases_emit_component_global_method_id() -> None:
    cfg = TrackOutputPipelineConfig(
        v_spline_output_backends=("global_bspline",),
        v_spline_output_presets=("balanced",),
    )
    specs = _v_spline_output_specs(cfg)
    assert [s.method_id for s in specs] == ["aviation_v_spline_bspline_global_balanced"]
    assert specs[0].backend == "bspline_component_global"


def test_v_spline_output_specs_can_select_hermite_smooth_only() -> None:
    cfg = TrackOutputPipelineConfig(
        v_spline_output_backends=("hermite_piecewise",),
        v_spline_output_presets=("smooth",),
    )
    specs = _v_spline_output_specs(cfg)
    assert len(specs) == 1
    assert specs[0].method_id == "v_spline_hermite_smooth"
    assert specs[0].backend == "hermite_piecewise"
    assert specs[0].preset == "smooth"


def test_default_kalman_rts_output_specs_emit_all_presets() -> None:
    specs = _kalman_rts_output_specs(TrackOutputPipelineConfig())
    assert [s.method_id for s in specs] == [
        "kalman_rts_balanced",
        "kalman_rts_accurate",
        "kalman_rts_smooth",
    ]
    assert all(s.file_stem == s.method_id for s in specs)


def test_default_synthetic_gap_specs_reference_existing_compact_methods() -> None:
    cfg = TrackOutputPipelineConfig()
    specs = _synthetic_gap_output_specs(cfg, _kalman_rts_output_specs(cfg), _v_spline_output_specs(cfg))
    assert [s.method_id for s in specs] == [
        "kalman_rts_balanced_synthetic_gap",
        "aviation_v_spline_quintic_balanced_synthetic_gap",
        "v_spline_bspline_overlap_smooth_synthetic_gap",
    ]
    assert [s.base_method_id for s in specs] == list(cfg.synthetic_gap_holdout_methods)


def test_kalman_rts_output_specs_can_be_disabled() -> None:
    cfg = TrackOutputPipelineConfig(kalman_rts_output_enabled=False)
    assert _kalman_rts_output_specs(cfg) == []
