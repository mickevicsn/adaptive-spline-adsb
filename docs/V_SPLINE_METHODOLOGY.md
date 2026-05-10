# V-Spline Methodology for ADS-B Reconstruction

The V-Spline idea fits a trajectory using position observations, velocity observations, and a smoothness penalty.  This project adapts that idea to ADS-B tracks by adding strict event pairing, local metric coordinates, conservative segmentation, robust boundary states, velocity confidence, per-segment tuning, and aviation-focused evaluation.

## Why V-Spline is useful for ADS-B

A position-only smoother ignores a major ADS-B signal: ground speed, track, and vertical rate.  A pure interpolation can over-trust noisy endpoint velocities or make unrealistic excursions in sparse intervals.  V-Spline sits between those extremes:

```text
fit position observations
+ fit velocity observations
+ penalize excessive acceleration
```

Approach and local flight tracks need this balance.  Velocity matters, derivative artifacts matter, and path shape must remain plausible through descent, turns, speed reduction, level-off, runway-adjacent motion, and go-around-like transitions.

## The objective in plain language

For each local segment, the fit estimates a continuous curve `x(t)` in local coordinates.  The simplified objective is:

```text
sum position residuals
+ gamma * sum velocity residuals
+ integral lambda(t) * ||x''(t)||^2 dt
```

The production B-spline core adds robust position loss, velocity confidence scaling, ridge regularization, optional jerk and snap penalties, endpoint guards, and boundary priors.  Those terms are safeguards around the same position/velocity/smoothness idea.

## Adaptive acceleration penalty

The adaptive penalty varies by interval.  In the B-spline core the interval weight follows the V-Spline form:

```text
lambda_i = eta * dt_i / max(speed_i, speed_floor)^2
```

This makes long sparse intervals and low-speed/noisy intervals less likely to receive high-frequency wiggles from unreliable velocity evidence.  Presets and local segment policy control `eta`, speed floors, knot spacing, and velocity weights.

## Why local segments are used

A single global smoothing value is rarely suitable for a mixed ADS-B track.  A track can contain steady descent, speed reduction, horizontal turn, level-off, runway-adjacent motion, sparse receiver intervals, and noisy messages.  A setting that is smooth enough for one region may flatten another region or overfit a quiet one.

Local segmentation allows different effective capacity and smoothness in different modelling regions while preserving stated continuity at ordinary joins.

## Basis families

The project emits several spline backends.

### Cubic B-spline V-Spline

`v_spline_bspline_<preset>` fits clamped cubic B-spline curves on local segments.  It uses position residuals, confidence-weighted velocity residuals, adaptive acceleration penalties, robust position loss, and boundary-related constraints or priors.

### Hermite V-Spline

`v_spline_hermite_<preset>` uses nodal cubic Hermite states at observation times.  Each node has position and velocity variables.  The objective follows the paper-oriented V-Spline structure: position residuals, velocity residuals, and an acceleration penalty scaled by sample count.

### Global-component cubic B-spline

`aviation_v_spline_bspline_global_<preset>` fits one cubic B-spline V-Spline per hard-gap component and does not use dynamic-regime internal boundaries.  It is a segmentation ablation backend.

### Overlap-save cubic B-spline

`v_spline_bspline_overlap_<preset>` fits local cubic B-spline segments and renders trusted interiors around joins.  It enforces robust hard C0 position anchors at ordinary joins and uses soft derivative priors.

### Join-smoothed cubic B-spline

`v_spline_bspline_join_smooth_<preset>` emphasizes soft higher-order join behaviour with stronger acceleration priors and jerk penalties while retaining robust hard C0 position anchors at ordinary joins.

### Stable Hermite

`v_spline_hermite_stable_<preset>` is a Hermite diagnostic backend that de-trusts ADS-B velocity more strongly and avoids hard endpoint velocity constraints.

### Aviation quintic V-Spline

`aviation_v_spline_quintic_<preset>` uses degree-5 B-splines.  It includes acceleration, jerk, and snap regularization, endpoint guards, velocity confidence, robust hard C0 position anchors, and soft velocity/acceleration priors at ordinary joins.

### Optional quintic with Kalman boundary prior

`aviation_v_spline_quintic_kalman_boundary_<preset>` is emitted when `V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR=1` or the backend is requested explicitly.  It uses the same quintic family with a Kalman-assisted boundary prior path.

## Continuity claims

Continuity is claimed only where the method creates and constrains an ordinary local modelling join.  Hard gaps and true discontinuity boundaries are not treated as ordinary joins.

For aviation B-spline variants, ordinary internal joins use a shared robust boundary position.  Velocity and acceleration continuity are encouraged through soft priors and method-specific join logic rather than treated as universal truth.  Overlap-save methods render trusted interiors to reduce endpoint artifacts near joins.

Hermite methods can harmonize join velocities when configured, but stable Hermite explicitly avoids over-trusting endpoint velocity constraints.

## Velocity confidence

ADS-B velocity is valuable but not always reliable.  `vspline/velocity_confidence.py` computes a confidence scale from:

- reported velocity versus finite-difference position velocity mismatch,
- vertical mismatch,
- track direction versus displacement direction,
- near-duplicate or quantized positions,
- low-speed ambiguity.

The spline objective scales velocity residuals by this confidence so questionable velocity evidence is downweighted instead of discarded blindly.

## Robust position loss

The B-spline core can use Huber position loss.  This limits the influence of retained noisy positions while still preserving ordinary observations.  Robust iterations are recorded in method quality diagnostics.

## Boundary terms

The B-spline core supports:

- hard boundary positions,
- hard component endpoint positions,
- optional endpoint velocity constraints,
- soft boundary velocity priors,
- soft boundary acceleration priors,
- endpoint guard multipliers for jerk and snap regularization.

The root configuration hard-anchors positions at relevant ordinary boundaries and component endpoints, while endpoint velocities are not hard constrained.

## Local policy and tuning

`vspline/segment_policy.py` supplies a first parameter guess by regime label.  `vspline/local_tuning.py` searches a compact candidate set around that guess.  The selected candidate depends on the preset objective:

```text
accurate -> position-focused
balanced -> fidelity/smoothness tradeoff
smooth   -> derivative-smoothness focused
```

Adaptive resegmentation can split persistently poor local fits when residual thresholds are exceeded.

## Presets

The root entrypoint emits `accurate`, `balanced`, and `smooth` by default.

`accurate` trusts observations more and uses more local flexibility.  `smooth` applies stronger regularization and de-trusts velocity more.  `balanced` sits between them.

Presets are not truth levels.  They are comparable modelling choices.
