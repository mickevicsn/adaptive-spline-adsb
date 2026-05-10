# ADS-B Preprocessing Methodology

Preprocessing is the first critical modelling step.  ADS-B is not a regular synchronized time series, and most reconstruction failures begin with the wrong time window, mixed event types, missing vertical rate, impossible jumps, duplicate timestamps, or hidden interpolation.

## Why ADS-B is not a regular time series

ADS-B is a broadcast surveillance stream.  The aircraft broadcasts different kinds of state information, but position, velocity, identification, status, and operational messages are not one synchronized record.  Airborne position and airborne velocity are distinct message types.  A receiver may see them at different times, miss one of them, or record repeated and irregular timestamps.

For reconstruction this creates three practical constraints.

First, position and velocity are asynchronous.  A row with latitude/longitude may not contain the velocity needed for a V-Spline paired observation.  A velocity row may not contain position.

Second, the stream is irregular.  The dataset can contain gaps, bursts, duplicates, quantization, and receiver timing effects.

Third, field meaning is not uniform.  Barometric altitude, geometric altitude, vertical rate, ground speed, and track can come from different decoded fields and may carry different noise or latency.

## Event, keyframe, sample

The pipeline uses three raw-data representations.

### Event

An event is one normalized raw ADS-B row.  It keeps source row identifiers and event metadata.  It may be a position event, a velocity event, a combined position/velocity event, or another row.  Events are used for traceability and debugging.

### Keyframe

A keyframe is a time-bucketed aggregation of events around a quantized timestamp.  The default quantization is 1 second unless the rule overrides it.  Keyframes can contain position, velocity, both, or neither.  Raw viewer output and raw diagnostics are keyframe-based.

### Reconstruction sample

A reconstruction sample is stricter than a keyframe.  It must contain:

```text
full local 3D position:  x_m, y_m, z_m
full local 3D velocity:  east_mps, north_mps, up_mps
unique time:             t
```

Only these paired samples are passed to V-Spline and Kalman/RTS reconstruction cores.  The preprocessing layer does not silently synthesize missing position or velocity for the core methods.

## Rule filtering

Curated flight rules can filter by:

- track id or ICAO,
- time window,
- CRC validity,
- allowed event kinds,
- spatial boundary,
- field elevation,
- on-ground interval,
- raw column names,
- keyframe time quantization.

The SQL loader keeps extraction broad.  The normalizer applies the rule filters so that the rule remains the place where per-track assumptions are visible.

## Unit and coordinate normalization

The raw stream is normalized into physical units:

- ground speed to metres per second,
- altitude to metres,
- vertical rate to metres per second,
- horizontal velocity from ground speed and track,
- latitude/longitude to local metric coordinates.

The local horizontal frame is ENU-like, and the velocity frame is local ENU.  Reconstruction objectives and diagnostics are therefore expressed in metric units.

## Vertical reference

Approach reconstruction needs a meaningful vertical channel.  When the flight rule provides field elevation, `z_m` is height above that reference.  This makes ground, approach, flare, and runway-adjacent diagnostics easier to interpret than raw MSL altitude alone.

The original altitude fields are preserved in raw outputs so the vertical reference is auditable.

## Vertical-rate handling

Vertical velocity is part of the strict paired sample.  The normalizer accepts explicit vertical-rate columns and can parse `decoded_json.velocity_raw` when that fallback exists.  Values are normalized to metres per second and paired with horizontal velocity derived from ground speed and track.

A keyframe without a complete vertical velocity component is not a V-Spline sample, even when it has position and horizontal speed.

## Outlier handling

Outlier handling removes observations that would make reconstruction nonsensical.  It is not a smoothing substitute.

The normalizer can apply:

- spatial boundary checks, such as maximum distance from a known reference point;
- apparent position-speed spike checks from consecutive position messages;
- velocity-acceleration spike checks from consecutive velocity messages;
- start trimming before the first usable clean position.

The adapter also runs a conservative paired-position motion filter with a high absolute speed gate and a reported-speed factor.  This filter is intended for catastrophic paired samples, not ordinary ADS-B noise.

## On-ground interval handling

A rule can define an on-ground interval.  Inside that interval, the keyframe altitude can be anchored to the configured field elevation so the local vertical channel is zero at the field reference.  Raw altitude evidence is still preserved for inspection.

## Missing data policy

Unpaired timestamps are kept as raw keyframes for traceability, but they are not passed to reconstruction cores.  This avoids a hidden interpolation step before the modelling stage.

The fitting cores receive the exact strict paired sample set described by `prepared_samples_report.json`.

## Acceleration semantics

Raw acceleration in keyframes is diagnostic.  It is derived from adjacent velocity keyframes when enabled.  It is useful for inspecting spikes and dynamics, but it is not an observation supplied to the V-Spline or Kalman/RTS fit.

Spline acceleration and jerk are derivatives of the reconstructed curve.  Kalman/RTS acceleration is part of the smoothed state.

## Event-kind semantics

The rule can accept event kinds such as `position`, `velocity`, and combined aliases.  When both position and velocity are allowed, combined position/velocity rows are kept automatically.

If an event-kind column is missing, the code infers event kind from available fields: latitude/longitude imply position evidence, ground speed and track imply velocity evidence, and rows with both are combined evidence.

## Inspection checklist

When a reconstruction looks suspicious, inspect these debug files first:

```text
raw_loader_report.json
normalization_report.json
prepared_samples_report.json
segmentation.json
boundary_states.json
```

Typical questions:

- Did the rule select the intended time window?
- Did CRC and event-kind filters leave enough rows?
- Are vertical-rate sources available?
- How many keyframes became strict paired samples?
- Did the motion outlier filter reject any paired samples?
- Did the hard-gap threshold split the track at expected places?
