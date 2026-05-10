from __future__ import annotations

import os

import numpy as np
import pytest

from vspline.bspline_core import (
    BSplineAccelerationPrior,
    BSplineAnchor,
    BSplineCoreConfig,
    BSplineCoreInput,
    BSplineVelocityConstraint,
    BSplineVelocityPrior,
    fit_b_spline_component,
)


def _smooth_reference(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = t - t[0]
    y = np.column_stack(
        [
            90.0 * tau,
            600.0 * np.sin(tau / 18.0),
            120.0 + 45.0 * np.cos(tau / 16.0),
        ]
    )
    v = np.column_stack(
        [
            np.full_like(tau, 90.0),
            (600.0 / 18.0) * np.cos(tau / 18.0),
            -(45.0 / 16.0) * np.sin(tau / 16.0),
        ]
    )
    return y, v


def test_cubic_bspline_keeps_raw_boundary_anchor_and_c2_continuity() -> None:
    t = np.linspace(1_700_000_000.0, 1_700_000_060.0, 91)
    y, v = _smooth_reference(t)
    boundary_idx = 45
    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y,
            v=v,
            anchors=(
                BSplineAnchor(
                    anchor_id="synthetic_boundary",
                    t=float(t[boundary_idx]),
                    position=y[boundary_idx],
                    source="raw_segment_boundary_sample",
                    sample_index=boundary_idx,
                ),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=5.0,
            velocity_weight=0.05,
            adaptive_eta=1e3,
            boundary_velocity_prior_weight=0.0,
            robust_position_loss="none",
        ),
    )
    anchor_error = fit.diagnostics["max_anchor_error_m"]
    assert anchor_error < 1e-7

    tb = float(t[boundary_idx])
    eps = 1e-5
    a_left = fit.evaluate([tb - eps], deriv=2)[0]
    a_right = fit.evaluate([tb + eps], deriv=2)[0]
    assert np.linalg.norm(a_left - a_right) < 1e-3


def test_noisy_interior_data_does_not_move_hard_boundary_anchor() -> None:
    rng = np.random.default_rng(123)
    t = np.linspace(0.0, 40.0, 80)
    y_true, v_true = _smooth_reference(t)
    y_noisy = y_true + rng.normal(scale=[8.0, 8.0, 4.0], size=y_true.shape)
    boundary_idx = 40
    # The hard anchor is intentionally the actual selected raw sample, not the
    # clean reference.  Interior observations are noisy and should be smoothed.
    raw_boundary = y_noisy[boundary_idx].copy()
    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y_noisy,
            v=v_true,
            anchors=(
                BSplineAnchor(
                    anchor_id="raw_noisy_boundary",
                    t=float(t[boundary_idx]),
                    position=raw_boundary,
                    source="raw_segment_boundary_sample",
                    sample_index=boundary_idx,
                ),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=6.0,
            velocity_weight=0.02,
            adaptive_eta=5e3,
            boundary_velocity_prior_weight=0.0,
            robust_position_loss="huber",
            robust_iterations=2,
        ),
    )
    fitted_boundary = fit.evaluate([t[boundary_idx]], deriv=0)[0]
    assert np.linalg.norm(fitted_boundary - raw_boundary) < 1e-7
    assert fit.diagnostics["position_residual_rmse_3d_m"] < 25.0


def test_soft_boundary_velocity_prior_limits_bad_velocity_damage() -> None:
    t = np.linspace(0.0, 25.0, 40)
    y, v = _smooth_reference(t)
    boundary_idx = 20
    bad_v = v[boundary_idx] + np.array([50.0, -80.0, 20.0])
    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y,
            v=v,
            anchors=(
                BSplineAnchor(
                    anchor_id="b",
                    t=float(t[boundary_idx]),
                    position=y[boundary_idx],
                    source="raw_segment_boundary_sample",
                    sample_index=boundary_idx,
                ),
            ),
            velocity_priors=(
                BSplineVelocityPrior(
                    prior_id="b",
                    t=float(t[boundary_idx]),
                    velocity=bad_v,
                    weight=0.01,
                    confidence=0.1,
                ),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=4.0,
            velocity_weight=0.05,
            adaptive_eta=1e4,
            boundary_velocity_prior_weight=0.01,
            robust_position_loss="none",
        ),
    )
    # The bad prior is soft, so the fitted velocity should stay closer to the
    # actual trajectory velocity than to the deliberately biased prior.
    fitted_v = fit.evaluate([t[boundary_idx]], deriv=1)[0]
    assert np.linalg.norm(fitted_v - v[boundary_idx]) < np.linalg.norm(fitted_v - bad_v)



def test_soft_boundary_acceleration_prior_is_reported_and_finite() -> None:
    t = np.linspace(0.0, 30.0, 61)
    tau = t - t[0]
    y = np.column_stack([80.0 * tau, 0.5 * tau * tau, 50.0 + 0.04 * tau * tau])
    v = np.column_stack([np.full_like(tau, 80.0), tau, 0.08 * tau])
    boundary_idx = len(t) // 2
    expected_acc = np.array([0.0, 1.0, 0.08], dtype=float)

    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y,
            v=v,
            anchors=(
                BSplineAnchor(
                    anchor_id="mid_position",
                    t=float(t[boundary_idx]),
                    position=y[boundary_idx],
                    source="shared_boundary_state",
                    sample_index=boundary_idx,
                ),
            ),
            acceleration_priors=(
                BSplineAccelerationPrior(
                    prior_id="mid_acceleration",
                    t=float(t[boundary_idx]),
                    acceleration=expected_acc,
                    weight=1.0,
                    confidence=0.75,
                    source="shared_boundary_state:test",
                ),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=4.0,
            velocity_weight=0.05,
            adaptive_eta=1e3,
            boundary_velocity_prior_weight=0.0,
            boundary_acceleration_prior_weight=1.0,
            robust_position_loss="none",
        ),
    )

    diag = fit.diagnostics["boundary_acceleration_priors"]
    assert diag["count"] == 1
    assert fit.diagnostics["solver"]["acceleration_prior_row_count"] == 1
    assert diag["priors"][0]["prior_id"] == "mid_acceleration"
    assert np.isfinite(diag["priors"][0]["error_norm_mps2"])
    assert np.linalg.norm(fit.evaluate([t[boundary_idx]], deriv=0)[0] - y[boundary_idx]) < 1e-7

def test_short_component_stress_does_not_generate_nan_derivatives() -> None:
    t = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)
    y, v = _smooth_reference(t)
    fit = fit_b_spline_component(
        BSplineCoreInput(t=t, y=y, v=v),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=3.0,
            velocity_weight=0.01,
            adaptive_eta=1e4,
            boundary_velocity_prior_weight=0.0,
            robust_position_loss="none",
        ),
    )
    grid = np.linspace(t[0], t[-1], 21)
    assert np.all(np.isfinite(fit.evaluate(grid, deriv=0)))
    assert np.all(np.isfinite(fit.evaluate(grid, deriv=1)))
    assert np.all(np.isfinite(fit.evaluate(grid, deriv=2)))
    assert fit.diagnostics["max_anchor_error_m"] < 1e-7


def test_go_around_like_reversal_preserves_altitude_minimum_anchor() -> None:
    t = np.linspace(0.0, 50.0, 80)
    tau = t - 25.0
    y = np.column_stack([80.0 * t, 120.0 * np.sin(t / 15.0), 0.08 * tau * tau + 30.0])
    v = np.column_stack([np.full_like(t, 80.0), 8.0 * np.cos(t / 15.0), 0.16 * tau])
    idx_min = int(np.argmin(y[:, 2]))
    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y,
            v=v,
            anchors=(
                BSplineAnchor(
                    anchor_id="go_around_minimum",
                    t=float(t[idx_min]),
                    position=y[idx_min],
                    source="raw_segment_boundary_sample",
                    sample_index=idx_min,
                ),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=5.0,
            velocity_weight=0.03,
            adaptive_eta=5e3,
            boundary_velocity_prior_weight=0.0,
            robust_position_loss="none",
        ),
    )
    assert np.linalg.norm(fit.evaluate([t[idx_min]], deriv=0)[0] - y[idx_min]) < 1e-7
    assert fit.diagnostics["accel_max_mps2"] < 20.0


def test_hard_velocity_constraints_are_exact_at_segment_endpoints() -> None:
    t = np.linspace(0.0, 30.0, 50)
    y, v = _smooth_reference(t)
    fit = fit_b_spline_component(
        BSplineCoreInput(
            t=t,
            y=y,
            v=v,
            anchors=(
                BSplineAnchor("start", float(t[0]), y[0], source="raw_segment_boundary_sample", sample_index=0),
                BSplineAnchor("end", float(t[-1]), y[-1], source="raw_segment_boundary_sample", sample_index=len(t) - 1),
            ),
            velocity_constraints=(
                BSplineVelocityConstraint("start_v", float(t[0]), v[0], sample_index=0),
                BSplineVelocityConstraint("end_v", float(t[-1]), v[-1], sample_index=len(t) - 1),
            ),
        ),
        BSplineCoreConfig(
            degree=3,
            knot_spacing_s=5.0,
            velocity_weight=0.02,
            adaptive_eta=1e3,
            boundary_velocity_prior_weight=0.0,
            robust_position_loss="none",
        ),
    )

    assert np.linalg.norm(fit.evaluate([t[0]], deriv=0)[0] - y[0]) < 1e-7
    assert np.linalg.norm(fit.evaluate([t[-1]], deriv=0)[0] - y[-1]) < 1e-7
    assert np.linalg.norm(fit.evaluate([t[0]], deriv=1)[0] - v[0]) < 1e-7
    assert np.linalg.norm(fit.evaluate([t[-1]], deriv=1)[0] - v[-1]) < 1e-7
    assert fit.diagnostics["hard_velocity_constraints"]["max_error_mps"] < 1e-7


@pytest.mark.skipif(os.getenv("ADSB_REGRESSION_SQLITE") is None, reason="set ADSB_REGRESSION_SQLITE to run real-track regression")
def test_real_adsb_regression_fixture_placeholder() -> None:
    # Keep the real-track regression wired into CI without requiring the private
    # fixture everywhere.  A full pipeline-level regression should assert: hard
    # anchor error remains ~0, C0/C1 continuity holds, adaptive resegmentation
    # is reported when used, and position RMSE does not materially worsen.
    assert os.path.exists(os.environ["ADSB_REGRESSION_SQLITE"])
