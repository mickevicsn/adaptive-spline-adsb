# Segmentation and Boundary Methodology

Segmentation is a modelling step for local spline fitting.  It is not a certified flight-phase detector.  The goal is to isolate sustained motion regimes well enough that local V-Spline fits can use appropriate capacity, regularization, and boundary constraints.

## Inputs

Segmentation receives strict paired reconstruction samples from the raw-keyframe adapter.  Each sample has local 3D position and local 3D velocity.

The root configuration uses `segmentation_feature_source="kalman_rts"`, meaning the segmentation features are computed from a lightweight Kalman-smoothed feature stream.  The original paired samples still remain the fitting observations.

## Hard-gap components

The first split is by hard time gaps.  The root configuration uses:

```text
hard_gap_s:          30.0
relative_gap_factor: 5.0
```

A gap can be treated as hard when it exceeds the fixed threshold and the relative threshold based on the local median sample spacing.  Hard-gap components are separate modelling domains.  The spline pipeline does not claim ordinary join continuity across those gaps.

## Dynamic local segments

Inside each hard-gap component, the segmenter looks for sustained evidence rather than single-sample spikes.  The root configuration uses:

```text
min_segment_points:       8
min_segment_duration_s:   15.0
min_boundary_spacing_s:   12.0
max_segments_per_component: 48
prefer_under_segmentation: true
```

Candidate boundaries can come from:

- energy-state transitions,
- vertical-mode transitions,
- speed-acceleration evidence,
- horizontal turn behaviour,
- go-around-like descent-to-climb patterns,
- vertical reversals and level-off lobes,
- altitude lobe shape evidence,
- rough-air evidence when enabled.

Motion-spike and PELT boundary sources are disabled in the root configuration.

## Energy-state segmentation

Energy-state segmentation is enabled.  It combines altitude-rate and speed-transition evidence to identify sustained regions where a single smoothness setting may be inappropriate.

The root configuration protects energy boundaries and smooths energy features over a small window.  This keeps sustained transitions visible while reducing sensitivity to isolated ADS-B rows.

## Vertical reversal and altitude lobe detection

Vertical reversal segmentation identifies descent/climb reversals that persist long enough and move far enough vertically.  The root configuration uses:

```text
vertical_reversal_min_points:               4
vertical_reversal_min_duration_s:           6.0
vertical_reversal_min_altitude_excursion_m: 25.0
```

Altitude lobe segmentation looks for local vertical shape features.  The root configuration uses minimum duration, point-count, prominence, side-prominence, and gradient gates to avoid reacting to small vertical noise.

## Horizontal turn segmentation

Horizontal turn segmentation is enabled by default.  Turn-rate and lateral-acceleration deadbands reduce sensitivity to weak or noisy turns.  Segment labels can therefore represent steady, turning, transitional, noisy, ground-slow, or composite regimes.

## Boundary placement and cleanup

Candidate boundaries are filtered by minimum points, minimum duration, minimum spacing, and per-component segment limits.  The boundary can be shifted within a small local band to a better transition sample, capped by `MAX_BOUNDARY_SHIFT_POINTS`.

This cleanup step is important: it keeps the model local without turning every noisy row into a separate segment.

## Shared boundary states

For each accepted spline boundary, `boundary_state.py` estimates a shared boundary state.  The root configuration uses:

```text
position_source:              weighted_compromise
position_raw_weight:          0.35
position_robust_weight:       0.65
window_points:                11
min_side_points:              4
poly_order:                   2
robust_iters:                 3
huber_k:                      1.345
blend_reported_velocity_weight: 0.0
max_velocity_factor:          2.5
```

The estimator combines the raw boundary sample with a robust local polynomial fit.  Position is the weighted compromise.  Velocity and acceleration come from robust local derivatives and are clipped when necessary.

## How spline methods use boundaries

Aviation B-spline variants use shared robust boundary positions for ordinary internal joins.  They can also apply soft velocity and acceleration priors at those boundaries.

Overlap-save variants render trusted interiors around joins.  Join-smoothed variants use stronger higher-order penalties near joins.  Quintic variants add jerk and snap regularization plus endpoint guards.

Hard gaps and true event discontinuities are represented as separate continuity categories in quality metrics.

## Local tuning

The local tuning module evaluates a compact set of candidate configurations for each segment.  Candidates vary values such as velocity weight, adaptive penalty, knot spacing, minimum observations per basis, jerk penalty, and boundary acceleration prior.

The active objective depends on preset:

```text
accurate -> position
balanced -> balanced
smooth   -> smooth
```

Candidate reports are written when `SEGMENT_TUNING_REPORT_CANDIDATES` is true.  The selected candidate and residual diagnostics are recorded in method quality and `segment_metrics.csv`.

## Adaptive resegmentation

Adaptive resegmentation is enabled by default.  A segment can be split when residual thresholds show that a local fit is persistently poor.  The root configuration allows two passes and uses preset-specific horizontal and vertical residual thresholds.

Resegmentation still respects minimum points, duration, boundary spacing, and per-component limits.

## Inspection workflow

Use these debug files to inspect segmentation and boundaries:

```text
segmentation.json
boundary_states.json
segment_metrics.csv
join_metrics.csv
```

Useful questions:

- Did hard gaps split only true sparse intervals?
- Are accepted boundaries supported by sustained evidence?
- Are boundary states dominated by a single raw row or by a robust local fit?
- Do join metrics show a method claiming continuity where a hard gap exists?
- Do bad residual regions align with segment boundaries or local tuning choices?
