# Kalman/RTS and Baseline Methods

Kalman/RTS is the main non-spline comparator.  It uses the same strict paired ADS-B observations as the V-Spline methods, but it is a state-space smoother rather than a spline.

## Model

`kalman_rts_core.py` uses a 3D constant-acceleration model driven by white jerk.  The state is:

```text
[x, y, z, vx, vy, vz, ax, ay, az]
```

Position and velocity ADS-B observations are used as measurements when velocity observations are enabled.  The forward Kalman filter produces filtered states; the Rauch--Tung--Striebel pass produces fixed-interval smoothed states.

The default rendering interpolation is a local quintic state bridge through neighbouring smoothed position, velocity, and acceleration states.  This interpolation is a rendering adapter, not segmentation.

## Whole-track behavior

The pipeline fits Kalman/RTS as one whole-track smoother over all prepared paired samples for the selected flight.  It does not use dynamic spline segmentation, hard-gap components, shared spline boundary states, local tuning, join harmonization, or adaptive resegmentation.

The method payload marks this explicitly in its `piecewise` and segment metadata.  Its single segment wrapper exists so the renderer and evaluator can use the same JSON contract as spline methods.

## Presets

The root entrypoint emits:

```text
kalman_rts_accurate
kalman_rts_balanced
kalman_rts_smooth
```

Preset intent mirrors the spline presets:

- `accurate` uses lower measurement standard deviations and a more agile white-jerk process;
- `balanced` uses the base Kalman/RTS noise settings;
- `smooth` uses higher observation noise and lower process freedom.

Key balanced defaults are:

```text
position_std_xy_m:       25.0
position_std_z_m:        40.0
velocity_std_xy_mps:     8.0
velocity_std_z_mps:      4.0
jerk_std_xy_mps3:        1.2
jerk_std_z_mps3:         0.7
gate_sigma:              4.5
robust_measurement_scaling: true
```

## What Kalman/RTS is good for

Kalman/RTS is useful as a strong baseline because it is:

- sequential and probabilistic,
- derivative-aware through the state model,
- robust to noisy position and velocity measurements through measurement noise and scaling,
- independent of spline knot placement and local segment policies.

## What Kalman/RTS does not test

Kalman/RTS does not test whether dynamic segmentation is helpful.  It also does not test spline basis capacity, hard C0 boundary anchors, overlap-save rendering, or local candidate tuning.  Those are spline-family design choices.

Because it is a whole-track smoother, long sparse intervals should be read through gap-behaviour metrics and visual inspection.  It can be a strong comparator and still behave differently from local methods around hard surveillance gaps.

## Baseline families

The project also emits spline baselines:

```text
v_spline_bspline_<preset>
v_spline_hermite_<preset>
aviation_v_spline_bspline_global_<preset>
```

These are useful for comparing basis family, segmentation use, and local fitting behavior against the aviation overlap, join-smoothed, stable-Hermite, and quintic variants.

## Comparison guidance

Use Kalman/RTS to answer:

- Does a state-space smoother already solve this track acceptably?
- Are spline methods gaining fidelity or derivative quality beyond the baseline?
- Are local spline joins adding artifacts relative to one whole-track smoother?
- Does the whole-track smoother behave plausibly across long raw gaps?

Use the spline baselines to answer:

- Does dynamic segmentation help or hurt compared with one spline per hard-gap component?
- Do Hermite nodal states behave differently from B-spline bases?
- Do overlap-save and quintic variants reduce endpoint and join artifacts?
