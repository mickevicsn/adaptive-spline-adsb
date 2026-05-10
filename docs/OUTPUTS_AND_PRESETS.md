# Outputs, Method IDs, and Presets

The pipeline writes one top-level manifest and one folder per processed flight.

```text
track_output/
  flights.json
  flights/<track_id>/
    flight.json
    methods/
      raw_adsb.json
      <method_id>.json
      minimal/
        raw_adsb.json
        <method_id>.json
    debug/
      ...
```

`flights.json` is the viewer manifest.  Each per-flight manifest entry lists raw ADS-B, reconstruction methods, detailed method files, minimal method files, and the debug directory.

## Raw ADS-B method

The raw method id is:

```text
raw_adsb
```

The detailed raw payload contains raw keyframes, render keyframes, reference metadata, loader diagnostics, and normalization diagnostics.  With the root entrypoint, raw events are not stored inline; the payload stores `raw_event_count`.  The pipeline class can include raw events inline when `include_raw_events_inline=True`.

## Method JSON contract

Each detailed method payload contains:

```text
schema_version
track_id
icao
method
reference
raw_keyframes
render_keyframes
quality
```

Spline and Kalman payloads also include method-specific quality metadata such as backend, objective, preset, fit mode, segment count, adapter diagnostics, piecewise reports, segment metadata, and reference-free trajectory-model metrics.

Minimal method files point back to the detailed file and contain compact render data for viewers.

## Default method

The top-level pipeline constant uses:

```text
aviation_v_spline_quintic_balanced
```

A per-flight manifest uses that method as `defaultMethod` when it was emitted.  Otherwise, it falls back to the first emitted reconstruction method or raw ADS-B.

## Root entrypoint output set

The root `main.py` entrypoint emits all configured presets for these V-Spline backends by default:

```text
bspline_piecewise
hermite_piecewise
bspline_component_global
bspline_overlap
bspline_join_smooth
hermite_stable
quintic_bspline
```

If `V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR=1`, it also emits:

```text
quintic_kalman_boundary
```

Kalman/RTS output is enabled by default.

## Presets

The default preset list is:

```text
balanced,accurate,smooth
```

It is controlled by `RECONSTRUCTION_PRESETS`.  Presets are de-duplicated and must be chosen from:

```text
accurate
balanced
smooth
```

`accurate` emphasizes observation fidelity.  `smooth` emphasizes derivative regularity.  `balanced` sits between those objectives.

## V-Spline method ids

For each preset, V-Spline backend names map to method ids as follows.

| Backend | Method id pattern | Label |
|---|---|---|
| `bspline_piecewise` | `v_spline_bspline_<preset>` | B-Spline V-Spline |
| `hermite_piecewise` | `v_spline_hermite_<preset>` | Hermite V-Spline |
| `bspline_component_global` | `aviation_v_spline_bspline_global_<preset>` | Aviation Global B-Spline V-Spline |
| `bspline_overlap` | `v_spline_bspline_overlap_<preset>` | B-Spline V-Spline overlap-save |
| `bspline_join_smooth` | `v_spline_bspline_join_smooth_<preset>` | B-Spline V-Spline join-smoothed |
| `hermite_stable` | `v_spline_hermite_stable_<preset>` | Stable Hermite V-Spline |
| `quintic_bspline` | `aviation_v_spline_quintic_<preset>` | Aviation Quintic V-Spline |
| `quintic_kalman_boundary` | `aviation_v_spline_quintic_kalman_boundary_<preset>` | Aviation Quintic V-Spline + Kalman boundary prior |

Backend aliases such as `quintic`, `overlap`, and `global_bspline` are accepted by `_v_spline_output_specs`, but method ids and filenames use the explicit patterns above.

## Kalman/RTS method ids

Kalman/RTS emits:

```text
kalman_rts_accurate
kalman_rts_balanced
kalman_rts_smooth
```

The emitted preset list follows `RECONSTRUCTION_PRESETS`.

## Synthetic-gap diagnostic method ids

Synthetic-gap diagnostics append `_synthetic_gap` to the base method id.  With root defaults, the diagnostic set is:

```text
kalman_rts_balanced_synthetic_gap
aviation_v_spline_quintic_balanced_synthetic_gap
v_spline_bspline_overlap_smooth_synthetic_gap
```

These methods train on paired samples with deterministic interior windows withheld.  They are useful for interpolation diagnostics.  The evaluator reads them like any other manifest method, so production-only comparisons should filter methods explicitly.

## Main output controls

```text
ADSB_SQLITE_PATH        SQLite database path; default adsb_raw.sqlite
TRACK_OUTPUT_DIR        output folder; default track_output
TRACK_LOG_DIR           log folder; default logs
TRACK_RULES_PATH        alternate rule file
ICAO_LIST               track ids or ICAO values to process
CLEAN_OUTPUT_DIR        clear output directory before run; default true
WRITE_DEBUG_ARTIFACTS   write per-flight debug artifacts; default true
PROGRESS                progress bars; default true
```

## Reconstruction controls

```text
RECONSTRUCTION_PRESETS          default balanced,accurate,smooth
V_SPLINE_OUTPUT_BACKENDS        comma-separated backend names
V_SPLINE_TIME_STEP_S            render step; default 0.25
V_SPLINE_OUTPUT_FREQUENCY_HZ    output frequency; default 4.0
V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR  optional quintic Kalman-boundary method
KALMAN_RTS_OUTPUT               emit Kalman/RTS methods; default true
EVENT_AWARE_EVALUATION          event-aware quality aggregation; default true
HOLDOUT_EVALUATION_FRACTION     segment holdout fraction; default 0.15
```

## Segmentation and tuning controls

Common controls include:

```text
DYNAMIC_SEGMENTATION
DYNAMIC_HARD_GAP_S
MIN_SEGMENT_POINTS
MIN_SEGMENT_DURATION_S
MIN_BOUNDARY_SPACING_S
MAX_SEGMENTS_PER_COMPONENT
SEGMENT_HORIZONTAL_TURNS
ENABLE_ROUGH_AIR_SEGMENTATION
ENABLE_GO_AROUND_DETECTION
ENABLE_VERTICAL_REVERSAL_SEGMENTATION
ENABLE_ALTITUDE_LOBE_SEGMENTATION
MAX_BOUNDARY_SHIFT_POINTS
SEGMENT_TUNING
SEGMENT_TUNING_MAX_CANDIDATES
JOIN_VELOCITY_HARMONIZATION
ADAPTIVE_RESEGMENTATION
```

Boundary-state controls include:

```text
BOUNDARY_POSITION_RAW_WEIGHT
BOUNDARY_POSITION_ROBUST_WEIGHT
BOUNDARY_WINDOW_POINTS
BOUNDARY_MIN_SIDE_POINTS
BOUNDARY_REPORTED_VELOCITY_WEIGHT
BOUNDARY_MAX_VELOCITY_FACTOR
```
