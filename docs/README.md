# ADS-B Reconstruction Documentation

This documentation describes the ADS-B reconstruction project as it is represented by the supplied source files.  The project converts curated raw ADS-B records from SQLite into traceable raw keyframes, strict paired position/velocity samples, and comparable reconstructed trajectories.

The central modelling idea is an aviation-adapted V-Spline workflow: use ADS-B position, reported velocity, and smoothness penalties in a local metric frame, then compare that spline family against a whole-track Kalman/RTS state-space baseline.

## Recommended reading order

1. `ADSB_PREPROCESSING.md` explains raw ADS-B normalization, event/keyframe/sample semantics, and why paired samples are strict.
2. `FLIGHT_RULES.md` explains the curated per-track rule file and the role of manual experiment definition.
3. `PIPELINE.md` follows the root `main.py` entrypoint through SQL loading, normalization, segmentation, fitting, output writing, and evaluation.
4. `V_SPLINE_METHODOLOGY.md` explains the spline objective, local fitting strategy, method families, and continuity claims.
5. `SEGMENTATION_AND_BOUNDARIES.md` explains hard-gap components, dynamic modelling regions, boundary states, local tuning, and join handling.
6. `KALMAN_RTS_AND_BASELINES.md` explains the state-space comparator and the non-spline baselines.
7. `OUTPUTS_AND_PRESETS.md` lists generated files, method ids, presets, and key environment variables.
8. `EVALUATION.md` explains the evaluator metrics, score groups, dataset mode, and how to treat diagnostic methods.
9. `DEBUG_ARTIFACTS.md` explains the per-flight review files under `track_output/flights/<track_id>/debug/`.
10. `SOURCE_MODULE_LAYOUT.md` maps source modules to pipeline responsibilities.
11. `REFERENCES.md` lists the external methodology references behind the project.

## Core mental model

The repository keeps three ideas separate.

**Raw ADS-B evidence** is preserved as normalized events and keyframes.  Keyframes can contain position only, velocity only, both, or neither.

**Reconstruction samples** are stricter.  A sample must have a unique timestamp, full local 3D position, and full local 3D velocity.  The V-Spline cores and Kalman/RTS core consume only these paired samples.

**Rendered reconstructions** are method outputs on a common time grid.  They are not treated as ground truth.  The evaluator compares fidelity, smoothness, dynamics, continuity, gap behaviour, and physical-envelope metrics against raw ADS-B evidence.

## Project scope

The project reconstructs track geometry and derivatives from ADS-B observations.  It does not claim an independent truth trajectory, direct aircraft intent, wind estimates, or certified flight-phase labels.  Segmentation labels are modelling regions used to fit local methods; they are not authoritative operational classifications.

## Main entrypoint

The production entrypoint is:

```bash
python main.py
```

By default, it processes every rule in `src/config/flight_rules.json`, unless `ICAO_LIST` is provided.  Requested values must match an exact `track_id` or an unambiguous ICAO rule.  The entrypoint reads SQLite data from `ADSB_SQLITE_PATH` or `adsb_raw.sqlite`, writes output to `TRACK_OUTPUT_DIR` or `track_output`, and writes logs to `TRACK_LOG_DIR` or `logs`.

The root entrypoint emits Kalman/RTS methods and the configured V-Spline backends for the `balanced`, `accurate`, and `smooth` presets.  The default viewer method is `aviation_v_spline_quintic_balanced` when that method is available in the run.
