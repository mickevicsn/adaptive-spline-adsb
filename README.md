# ADS-B V-Spline Reconstruction Pipeline

This repository builds a viewer-ready `track_output/` folder from a production
SQLite ADS-B database.  The reconstruction path is intentionally single-purpose:

**SQLite raw ADS-B rows → curated `flight_rules.json` → dynamic flight-regime
segmentation → local V-Spline reconstruction with a B-spline basis → debug and
quality artifacts.**

## Run

```bash
python main.py
```

Useful environment variables:

```bash
ADSB_SQLITE_PATH=adsb_raw.sqlite       # production database
TRACK_RULES_PATH=src/config/flight_rules.json
TRACK_OUTPUT_DIR=track_output
ICAO_LIST=4BAAD9,502D5A               # optional subset; unset means all rules
RECONSTRUCTION_PRESET=balanced         # balanced | accurate | smooth
SEGMENT_TUNING=1
ADAPTIVE_RESEGMENTATION=1
WRITE_DEBUG_ARTIFACTS=1
```

## Output structure

```text
track_output/
  flights.json
  flights/<track_id>/
    flight.json
    methods/raw_adsb.json
    methods/v_spline_bspline.json
    methods/minimal/raw_adsb.json
    methods/minimal/v_spline_bspline.json
    debug/
      debug_manifest.json
      flight_rule.json
      config_snapshot.json
      raw_loader_report.json
      normalization_report.json
      prepared_samples_report.json
      segmentation.json
      boundary_states.json
      reconstruction_quality.json
      segment_metrics.csv
      join_metrics.csv
      flight_summary.json
      flight.log
```

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) explains the execution stages.
- [`docs/V_SPLINE_METHODOLOGY.md`](docs/V_SPLINE_METHODOLOGY.md) explains the mathematical reconstruction method.
- [`docs/DEBUG_ARTIFACTS.md`](docs/DEBUG_ARTIFACTS.md) explains the per-flight academic/debug files.
- [`docs/FLIGHT_RULES.md`](docs/FLIGHT_RULES.md) explains why `flight_rules.json` is mandatory.
