# Flight Rules

Flight rules are curated per-track experiment definitions.  They belong outside the mathematical core because track selection, field reference, data quality policy, and runway-adjacent assumptions are data decisions rather than spline decisions.

The default rule file is:

```text
src/config/flight_rules.json
```

An alternate rule file can be supplied with `TRACK_RULES_PATH`.

## Rule identity

Each rule has:

```text
track_id
icao
time_window
field_elevation
```

`track_id` must be unique.  `ICAO_LIST` can select a track by exact `track_id` or by ICAO when that ICAO is unambiguous in the rule file.

## Time window

The time window defines the processed interval:

```json
"time_window": {
  "first_point_unix": 1760000000,
  "last_point_unix": 1760001200
}
```

The window is applied during rule filtering.  It protects the reconstruction from unrelated earlier or later surveillance records for the same aircraft.

## Field elevation

The field elevation rule defines the vertical reference used by the pipeline:

```json
"field_elevation": {
  "method": "fixed_ft_msl",
  "value": 273.0
}
```

When `baro_z_reference` is `field`, the normalized vertical channel is:

```text
z_m = altitude_msl_m - field_elevation_m
```

This makes vertical diagnostics more interpretable around approach, runway, and on-ground intervals.

## On-ground interval

A rule can define an on-ground interval:

```json
"on_ground_window": {
  "start_unix": 1760001100,
  "end_unix": 1760001200
}
```

During this interval, the normalized keyframe altitude can be anchored to the field reference for display and fitting consistency.  Raw altitude fields remain available in the detailed raw payload.

## CRC policy

`require_crc_ok` controls whether the rule filter keeps only rows marked CRC-valid when the CRC column exists.  The root pipeline loads broad SQL rows, then applies this policy in normalization.

## Event-kind policy

`allowed_events` controls which normalized event kinds are used:

```json
"allowed_events": ["velocity", "position"]
```

When both position and velocity are allowed, combined position/velocity aliases are also retained.  This supports datasets where the raw event-kind column distinguishes position-only, velocity-only, and combined rows.

If the raw event-kind column is missing, the code infers event kind from available latitude/longitude, ground speed, and track fields.

## Spatial boundary

`boundary_rix` is a project-specific spatial boundary rule.  It limits accepted position evidence by distance from a known reference point, expressed in the configured unit.  This removes rows that are inconsistent with the intended local track.

## Outlier multiplier

`outlier_multiplier` controls IQR-style outlier filtering used by the raw normalizer.  It applies to position-speed spikes and velocity-acceleration spikes computed from raw evidence.

This step removes observations that are physically incompatible with the surrounding track evidence.  It does not smooth the retained observations.

## Origin override

A rule may provide `origin_lat_deg` and `origin_lon_deg`.  If omitted, the pipeline uses the first clean point for the local horizontal origin.  The altitude reference still comes from field elevation.

## Raw column names

The rule can name the raw timestamp, ICAO, CRC, and event-kind columns:

```json
"raw_time_column": "ts_utc",
"raw_icao_column": "icao",
"raw_crc_ok_column": "crc_ok",
"raw_event_kind_column": "event_kind"
```

This keeps the loader usable across SQLite extracts with compatible content but different column naming.

## Keyframe quantization

`keyframe_time_quantization_s` controls time bucketing before keyframe aggregation.  The default is 1 second.  The normalizer rounds timestamps to the nearest quantized bucket, aggregates position and velocity evidence within that bucket, and keeps source row identifiers for traceability.

## Registry behavior

The rule registry validates duplicate `track_id` values.  It resolves exact track ids directly.  ICAO lookup is allowed only when the ICAO maps to one rule.  Multiple rules for the same ICAO require the caller to request a specific `track_id`.

## Practical rule checklist

A useful rule should answer these questions:

- Which surveillance interval belongs to this track?
- What field elevation defines the vertical reference?
- Is an on-ground interval known?
- Should CRC-invalid rows be rejected?
- Which ADS-B event kinds are useful for this reconstruction?
- Is a spatial boundary needed for receiver or track contamination?
- Does the raw table use standard column names?
