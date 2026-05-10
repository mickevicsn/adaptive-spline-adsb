# Debug and Academic Artifacts

Debug artifacts are part of the methodology.  They make each reconstruction auditable: what data was loaded, what filters were applied, which samples were paired, why segments were chosen, how boundaries were estimated, what parameters were selected, and how methods were scored.

Per flight, debug files live under:

```text
track_output/flights/<track_id>/debug/
```

## Run traceability

### `debug_manifest.json`

Records pipeline stages, status, duration, and emitted artifacts.  Use it to confirm which stages completed and which artifacts were written.

### `flight.log`

Per-flight log messages.  Use this file for warnings, fallback notes, and stage-level context.

### `config_snapshot.json`

The effective configuration used for the flight.  This includes output backends, presets, segmentation thresholds, boundary settings, local tuning settings, Kalman/RTS settings, adapter settings, and synthetic-gap configuration.

### `flight_rule.json`

The exact curated flight rule used for the run.  This is part of the experiment definition.

## Data preparation artifacts

### `raw_loader_report.json`

SQL extraction report.  Use it to verify table choice, row counts, selected columns, vertical-rate extraction, and broad load behavior.

### `normalization_report.json`

Normalization report.  It contains normalizer configuration, rule-filter counts, event-kind counts, vertical-rate source counts, keyframe counts, outlier filtering, and start trimming.

### `prepared_samples_report.json`

Strict paired-sample report.  It shows paired sample count, adapter segment count, duplicate-time handling, invalid paired keyframes, and paired-position outlier filtering.

## Segmentation artifacts

### `segmentation.json`

Contains hard-gap components, dynamic segments, accepted boundaries, candidate reasons, regime labels, and diagnostics.  Read this as modelling evidence rather than certified flight-phase truth.

### `boundary_states.json`

Contains shared boundary states used by spline methods.  Inspect this when a join kink, velocity discontinuity, or suspicious boundary anchor appears in the output.

### `synthetic_gap_holdout.json`

Contains the synthetic-gap diagnostic plan when enabled: selected windows, withheld sample indices, training sample count, and configuration values.

## Reconstruction artifacts

### `reconstruction_quality.json`

Global quality summary for generated reconstructions.  It is a quick overview of method quality blocks and trajectory-model metrics.

### `segment_metrics.csv`

Local segment table.  Useful fields include:

- method id and backend,
- segment id and regime label,
- observation count and time span,
- selected preset/config values,
- candidate score and local tuning metadata,
- residual RMSE, p95, and max values,
- acceleration and jerk metrics,
- adaptive resegmentation metadata.

This file is the main entry point for diagnosing local regions.

### `join_metrics.csv`

Join continuity table.  It reports position jump, velocity jump, acceleration jump or proxy, boundary category, and method metadata.  Use it to support or reject method-specific continuity claims.

### `trajectory_model_metrics.csv`

Reference-free trajectory-model score components.  It reports observation consistency, velocity evidence, finite-difference kinematics, trajectory smoothness, physical plausibility, dynamic-detail preservation, derivative closure, event-aware joins, hard-gap honesty, and locality scope where available.  The evaluator also writes a file with this name in the same debug directory; run order determines which producer supplied the file on disk.

### `flight_summary.json`

Compact summary of manifest entry, output method paths, raw keyframe count, and paired sample count.

## Evaluation artifacts

Running `evaluate_reconstructions.py` writes these files into the same debug directory:

```text
evaluation_summary.json
evaluation_metrics.csv
evaluation_group_scores.csv
trajectory_model_metrics.csv
evaluation_report.md
```

Dataset mode writes dataset-level reports under `track_output/dataset_evaluation/` unless another directory is supplied.

## Debugging checklist

When output looks wrong, inspect artifacts in this order:

1. `flight_rule.json` — confirm the intended track, time window, field elevation, and event policy.
2. `raw_loader_report.json` — confirm SQL rows and columns.
3. `normalization_report.json` — confirm event counts, vertical-rate availability, keyframe counts, and filters.
4. `prepared_samples_report.json` — confirm how many keyframes became strict paired samples.
5. `segmentation.json` — confirm hard gaps and dynamic boundaries.
6. `boundary_states.json` — confirm join anchors and robust boundary estimates.
7. `segment_metrics.csv` — find local fit failures or excessive residuals.
8. `join_metrics.csv` — inspect continuity claims.
9. `evaluation_metrics.csv` and `evaluation_group_scores.csv` — compare method behavior.

## Reading failures

A failure at preparation usually indicates insufficient strict paired samples, duplicate paired timestamps, missing local frame fields, or missing vertical velocity.

A failure during fitting usually indicates a segment with too few observations, ill-conditioned basis settings, or invalid sample data.

A suspiciously good raw-fidelity score can still hide derivative artifacts.  Always inspect smoothness, dynamics, gap behaviour, and trajectory-model metrics alongside raw position error.
