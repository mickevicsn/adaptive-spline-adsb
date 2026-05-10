from __future__ import annotations

import numpy as np

from vspline.hermite_core import (
    VSplineCoreConfig,
    VSplineCoreInput,
    VSplineEndpointConstraints,
    VSplineEndpointState,
    fit_v_spline_core,
)


def _reference(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = t - t[0]
    y = np.column_stack(
        [
            70.0 * tau,
            250.0 * np.sin(tau / 12.0),
            100.0 + 0.04 * tau * tau,
        ]
    )
    v = np.column_stack(
        [
            np.full_like(tau, 70.0),
            (250.0 / 12.0) * np.cos(tau / 12.0),
            0.08 * tau,
        ]
    )
    return y, v


def test_paper_hermite_core_fits_finite_derivatives_and_reports_basis_count() -> None:
    t = np.linspace(0.0, 30.0, 31)
    y, v = _reference(t)
    fit = fit_v_spline_core(
        VSplineCoreInput(t=t, y=y, v=v),
        VSplineCoreConfig(
            penalty_mode="adaptive",
            velocity_weight=0.03,
            adaptive_eta=1_000.0,
            adaptive_speed_floor_mps=1.0,
            compute_loocv_score=False,
        ),
    )

    grid = np.linspace(t[0], t[-1], 101)
    for deriv in (0, 1, 2, 3):
        evaluated = fit.evaluate(grid, deriv=deriv)
        assert evaluated.shape == (grid.size, 3)
        assert np.all(np.isfinite(evaluated))

    assert fit.diagnostics["basis"] == "nodal_cubic_hermite"
    assert fit.diagnostics["n_basis"] == 2 * t.size
    assert fit.lambda_intervals.shape == (t.size - 1,)


def test_external_endpoint_constraints_are_enforced_exactly() -> None:
    t = np.linspace(0.0, 12.0, 13)
    y, v = _reference(t)
    start_position = y[0] + np.array([5.0, -2.0, 1.0])
    end_velocity = v[-1] + np.array([0.0, 3.0, -0.5])
    constraints = VSplineEndpointConstraints(
        start=VSplineEndpointState(position=start_position, velocity=v[0]),
        end=VSplineEndpointState(position=y[-1], velocity=end_velocity),
        hard_start_position=True,
        hard_start_velocity=True,
        hard_end_position=True,
        hard_end_velocity=True,
    )

    fit = fit_v_spline_core(
        VSplineCoreInput(t=t, y=y, v=v),
        VSplineCoreConfig(
            penalty_mode="constant",
            smoothing_lambda=10.0,
            velocity_weight=0.02,
            compute_loocv_score=False,
        ),
        endpoint_constraints=constraints,
    )

    assert np.linalg.norm(fit.evaluate([t[0]], deriv=0)[0] - start_position) < 1e-9
    assert np.linalg.norm(fit.evaluate([t[-1]], deriv=1)[0] - end_velocity) < 1e-9
    assert fit.diagnostics["hard_endpoint_constraint_max_abs_error"] < 1e-9
