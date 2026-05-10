"""Production V-Spline package.

Submodules are imported directly by the pipeline to avoid circular imports:

- ``vspline.bspline_core``: B-spline V-Spline solver
- ``vspline.hermite_core``: paper-oriented nodal Hermite V-Spline solver
- ``vspline.segment_policy``: regime-aware local parameter policy
- ``vspline.local_tuning``: per-segment candidate tuning
- ``vspline.quality``: residual and continuity metrics
- ``geo_utils``: coordinate conversion and unit helpers outside the package
"""

__all__: list[str] = []
