# Source Module Layout

This document maps source files to project responsibilities.  It is useful when reading the methodology docs alongside the code.

## Entrypoints

### `main.py`

Production entrypoint.  It reads environment variables, builds `TrackOutputPipelineConfig`, and runs the pipeline.  It sets the root default output set: Kalman/RTS plus the configured V-Spline backends for the `balanced`, `accurate`, and `smooth` presets.

### `evaluate_reconstructions.py`

Evaluation CLI.  It reads generated `track_output` manifests, loads raw ADS-B and non-raw method payloads listed in the selected flight manifest, computes grouped metrics, and writes single-flight or dataset-level evaluation reports.

## Data loading and rules

### `src/sql_loader.py`

SQLite ADS-B loader.  It selects a compatible table, loads raw rows, parses timestamps and optional vertical-rate data, and returns a broad dataframe plus a loader report.

### `src/track_rules.py`

Rule schema and registry.  It validates rule files, resolves track ids and ICAO values, applies ICAO/CRC/event/time filters, and exposes field elevation, on-ground intervals, spatial boundaries, and outlier rules.

### `src/config/flight_rules.json`

Default curated rule file.  Each rule defines one processed track.

## Raw ADS-B preprocessing

### `src/raw_adsb.py`

Raw ADS-B normalizer.  It converts rows to events, keyframes, and a flat normalized dataframe.  It handles unit conversion, event-kind inference, vertical-rate parsing, keyframe quantization, field-relative altitude, on-ground interval behavior, and rule-driven outlier filtering.

### `src/raw_keyframe_vspline_adapter.py`

Strict keyframe-to-sample adapter.  It extracts only full 3D position plus full 3D velocity samples with unique times and validated local ENU velocity frame.  It reports rejected/invalid keyframes and paired-position outlier filtering.

### `src/geo_utils.py`

Local coordinate utilities for converting latitude/longitude to metric local coordinates and back.

## Segmentation and boundaries

### `src/segmentation_kalman.py`

Lightweight constant-velocity Kalman smoother used to create stable segmentation features.  It is used for segmentation evidence, not as replacement fit data.

### `src/trajectory_segmentation.py`

Dynamic segmentation.  It splits paired samples by hard gaps and sustained motion evidence, then produces components, local segments, boundary records, regime labels, and diagnostics.

### `src/boundary_state.py`

Shared boundary-state estimator.  It combines raw boundary samples with robust local polynomial fits and outputs boundary position, velocity, acceleration, confidence, and diagnostics.

## Reconstruction cores

### `src/kalman_rts_core.py`

Whole-track constant-acceleration Kalman filter plus RTS smoother.  It consumes strict paired samples and returns smoothed position, velocity, and acceleration states.

### `src/vspline/bspline_core.py`

B-spline V-Spline core.  It supports cubic and quintic bases, position and velocity residuals, adaptive acceleration penalties, jerk and snap penalties, Huber position loss, hard anchors, soft boundary priors, endpoint guards, and solver diagnostics.

### `src/vspline/hermite_core.py`

Nodal cubic Hermite V-Spline core.  It uses position and velocity variables at observation times and a paper-oriented V-Spline objective.

### `src/vspline/velocity_confidence.py`

Velocity confidence scaling.  It compares reported velocity to finite-difference evidence and downweights low-confidence velocity observations.

### `src/vspline/segment_policy.py`

Local first-guess policy.  It maps regime labels to parameter multipliers such as adaptive penalty, velocity weight, knot spacing, and boundary priors.

### `src/vspline/local_tuning.py`

Per-segment candidate search and adaptive resegmentation support.  It scores candidate configurations according to the active preset objective.

### `src/vspline/quality.py`

Local fit quality, join quality, and reference-free trajectory-model metrics.

## Output builder

### `src/track_output_pipeline.py`

End-to-end orchestration.  It runs loading, normalization, local projection, paired-sample preparation, segmentation, boundary estimation, method fitting, synthetic-gap diagnostics, debug artifact writing, method JSON writing, and manifest writing.

The method id mapping, output layout, schema version, default viewer method, and synthetic-gap output specs live in this module.
