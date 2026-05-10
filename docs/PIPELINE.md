# Pipeline Methodology

The pipeline is an end-to-end output builder.  It starts from curated SQLite ADS-B rows and writes a viewer/evaluation folder under `track_output/`.  Every stage preserves enough metadata to review what was loaded, filtered, fitted, and scored.

The root entrypoint is `main.py`.  It constructs an explicit `TrackOutputPipelineConfig` and calls `TrackOutputPipeline.run()`.

## Stage 1: select tracks from rules

The pipeline loads `src/config/flight_rules.json` unless `TRACK_RULES_PATH` points to another rule file.  Each requested item in `ICAO_LIST` is resolved as either an exact `track_id` or an unambiguous ICAO.  With no `ICAO_LIST`, every rule is processed.

The rule is part of the experiment definition.  It defines the aircraft/track id, time window, field elevation, optional on-ground window, spatial boundary, event-kind policy, CRC policy, raw column names, and keyframe quantization.

## Stage 2: load SQLite rows

`sql_loader.py` reads SQLite ADS-B records.  If the table name is not supplied, the loader chooses a table with the required timestamp and ICAO columns, using common ADS-B table names as preferences.

The root pipeline asks the SQL loader for broad extraction and applies rule-level time-window and CRC filtering during normalization.  This keeps the loader generic and keeps per-track decisions in `track_rules.py` and `flight_rules.json`.

## Stage 3: normalize ADS-B events and keyframes

`raw_adsb.py` turns raw rows into traceable ADS-B events and time-bucketed keyframes.

Events preserve source row identity and row type.  Keyframes aggregate events around a quantized timestamp.  A keyframe can hold position, velocity, both, or neither.

Normalization performs unit conversion, event-kind handling, vertical-rate parsing, field-relative vertical reference, on-ground altitude handling, and rule-based outlier filtering.  Horizontal velocity comes from ground speed and track.  Vertical velocity comes from explicit vertical-rate fields or `decoded_json.velocity_raw` when that fallback is available.

## Stage 4: choose origin and project to a local frame

The pipeline chooses a local horizontal origin from the rule override or the first clean point.  Latitude/longitude are projected to local metres.  The vertical channel is field-relative height when the rule provides field elevation.

The resulting local frame is documented in each method payload under `reference`.  The reconstruction objective and evaluation metrics work in metres, metres per second, metres per second squared, and related derivative units.

## Stage 5: write the raw ADS-B payload

The raw method is written as:

```text
track_output/flights/<track_id>/methods/raw_adsb.json
track_output/flights/<track_id>/methods/minimal/raw_adsb.json
```

The root entrypoint writes raw keyframes inline and stores a raw event count.  The pipeline can include raw events inline through `include_raw_events_inline=True`; the root entrypoint leaves that disabled for compact method files.

## Stage 6: prepare strict reconstruction samples

`raw_keyframe_vspline_adapter.py` extracts only keyframes that have:

```text
full local 3D position:  x_m, y_m, z_m
full local 3D velocity:  east_mps, north_mps, up_mps
unique timestamp:        t
```

The adapter rejects duplicate paired times and validates the local ENU velocity frame.  It also runs a conservative paired-position motion outlier filter.  Acceleration derived from raw velocity differences is diagnostic only and is not passed as an observation to the fitting cores.

In the root entrypoint, the adapter does not split samples by time gap.  Hard-gap handling is performed by the dynamic segmentation stage.

## Stage 7: segment paired samples for local spline fitting

`trajectory_segmentation.py` splits paired samples into hard-gap components and local dynamic modelling regions.  The root configuration uses a hard-gap threshold of 30 seconds with a relative-gap guard, then searches for sustained evidence of energy-state transitions, vertical reversals, altitude lobes, go-around-like patterns, and horizontal turns.

Segmentation features are derived from a lightweight Kalman-smoothed feature source by default.  The raw paired observations remain the fit data; smoothing the segmentation features does not replace the reconstruction samples.

## Stage 8: estimate shared boundary states

`boundary_state.py` estimates a shared state for accepted spline boundaries.  The root configuration uses a weighted compromise between the raw boundary sample and a robust local polynomial fit:

```text
raw position weight:     0.35
robust fit weight:       0.65
window points:           11
polynomial order:        2
reported velocity blend: 0.0
```

These boundary states are used by aviation spline variants for shared position anchors and soft derivative priors.  Hard gaps and true discontinuity boundaries are treated separately from ordinary local modelling joins.

## Stage 9: fit reconstruction methods

The root entrypoint emits Kalman/RTS and V-Spline methods for the requested presets.

Kalman/RTS fits one whole-track state-space smoother over the prepared paired samples for each preset.  It skips spline segmentation, boundary states, local tuning, join harmonization, and adaptive resegmentation.

V-Spline methods fit local curves using cubic or quintic B-spline bases, or nodal cubic Hermite states, depending on the backend.  Local B-spline and Hermite methods use the dynamic segmentation output, boundary states, local policy, and optional per-segment candidate tuning.

## Stage 10: fit synthetic-gap diagnostics

The pipeline can fit diagnostic methods with deterministic interior raw windows withheld from the training samples.  The default diagnostic base methods are:

```text
kalman_rts_balanced
aviation_v_spline_quintic_balanced
v_spline_bspline_overlap_smooth
```

When those base methods are emitted, the diagnostic methods use the suffix `_synthetic_gap`.  They are useful for interpolation stress tests and should be interpreted separately from normal method files.

## Stage 11: write debug artifacts and manifests

Per-flight debug artifacts are written under:

```text
track_output/flights/<track_id>/debug/
```

The flight manifest and method files are written under:

```text
track_output/flights/<track_id>/flight.json
track_output/flights/<track_id>/methods/*.json
track_output/flights/<track_id>/methods/minimal/*.json
```

The top-level manifest is:

```text
track_output/flights.json
```

## Main runtime controls

Common environment variables:

```text
ADSB_SQLITE_PATH                 SQLite database path; default adsb_raw.sqlite
TRACK_OUTPUT_DIR                 output folder; default track_output
TRACK_LOG_DIR                    log folder; default logs
TRACK_RULES_PATH                 optional alternate flight rules file
ICAO_LIST                        comma-separated track ids or ICAO values
CLEAN_OUTPUT_DIR                 clear output directory before run; default true
RECONSTRUCTION_PRESETS           presets to emit; default balanced,accurate,smooth
V_SPLINE_OUTPUT_BACKENDS         backend list; see OUTPUTS_AND_PRESETS.md
V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR  enable optional quintic Kalman-boundary method
KALMAN_RTS_OUTPUT                emit Kalman/RTS methods; default true
WRITE_DEBUG_ARTIFACTS            write debug files; default true
PROGRESS                         progress bars; default true
```
