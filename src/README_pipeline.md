# Source Module Layout

The source tree is organized around a step-by-step production pipeline.

```text
track_output_pipeline.py        orchestration, logging, debug artifacts
track_rules.py                  mandatory curated flight rules
sql_loader.py                   SQLite-only ADS-B extraction
geo_utils.py                    unit constants and WGS84 coordinate helpers
raw_adsb.py                     raw row normalization and keyframe creation
raw_keyframe_vspline_adapter.py paired position/velocity sample preparation
segmentation_kalman.py          optional Kalman/RTS feature smoothing for segmentation
trajectory_segmentation.py      energy-state and motion-regime segmentation
boundary_state.py               shared boundary position/velocity/acceleration estimates
vspline/bspline_core.py         B-spline V-Spline solver
vspline/hermite_core.py         paper-oriented Hermite V-Spline solver
vspline/segment_policy.py       energy/regime-aware local parameter policy
vspline/local_tuning.py         per-segment candidate tuning and adaptive resegmentation support
vspline/quality.py              residual and continuity metrics
```

Production reconstruction emits comparable V-Spline variants by default:
`bspline_piecewise` and `hermite_piecewise`, each for the `balanced`, `accurate`,
and `smooth` presets.  Legacy demo backends, Excel loading, inferred broad
rules, and global tuning switches have been removed from the pipeline path.
