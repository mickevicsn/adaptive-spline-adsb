# Adaptive Spline ADS-B

A research and engineering pipeline for reconstructing aircraft trajectories from ADS-B surveillance data using velocity-aware V-Spline methods and Kalman/RTS baselines.

The project converts curated ADS-B records from a SQLite database into viewer-ready trajectory files, debug artifacts, evaluation reports, and an interactive 3D flight viewer. It is designed for comparing reconstruction methods under a shared evidence contract rather than treating raw ADS-B points as perfect ground truth.

## What this project does

```text
SQLite ADS-B database
        ↓
curated flight rules
        ↓
raw ADS-B events and keyframes
        ↓
strict paired position + velocity samples
        ↓
dynamic segmentation and boundary-state modelling
        ↓
Kalman/RTS and V-Spline reconstruction methods
        ↓
debug artifacts, evaluation reports, and 3D viewer output
```

The main goal is to evaluate whether aviation-adapted V-Spline reconstructions can approach strong Kalman/RTS smoothing performance while retaining spline-specific advantages such as local controllability, differentiability, segment diagnostics, and explicit boundary inspection.

## Key features

- **ADS-B evidence handling**: loads selected SQLite ADS-B rows and applies per-flight rules for ICAO or track selection, time windows, field elevation, CRC policy, event-kind filtering, and outlier control.
- **Strict reconstruction samples**: uses only samples with complete local 3D position and complete local 3D velocity at a unique timestamp.
- **Local metric frame**: converts latitude, longitude, altitude, ground speed, track, and vertical rate into metre-based local coordinates and velocities.
- **Multiple reconstruction families**:
  - raw ADS-B baseline,
  - Kalman/RTS whole-track smoothing,
  - cubic B-spline V-Spline,
  - Hermite V-Spline,
  - global-component B-spline V-Spline,
  - overlap-save B-spline V-Spline,
  - join-smoothed B-spline V-Spline,
  - stable Hermite diagnostic variant,
  - aviation quintic V-Spline.
- **Presets**: `accurate`, `balanced`, and `smooth` reconstruction settings.
- **Dynamic segmentation**: detects hard gaps and local modelling regions for turns, vertical changes, energy-state transitions, altitude lobes, and go-around-like behaviour.
- **Boundary-state modelling**: estimates shared spline boundary states using a weighted compromise between raw boundary evidence and robust local fitting.
- **Synthetic-gap diagnostics**: optionally withholds deterministic interior evidence windows to stress-test interpolation behaviour.
- **Evaluation framework**: writes per-flight and dataset-level scores for fidelity, smoothness, aircraft-dynamics proxies, gap behaviour, endpoint artefacts, shape similarity, envelope violations, and reference-free trajectory-model quality.
- **Interactive 3D viewer**: launches a browser-based ADS-B trajectory viewer for comparing raw evidence with reconstructed tracks.

## Repository layout

```text
.
├── main.py                         # launches the ADS-B 3D viewer
├── main_pipeline.py                # builds track_output from SQLite + flight rules
├── evaluate_reconstructions.py     # evaluates generated reconstruction JSON files
├── pyproject.toml
├── docs/
│   ├── PIPELINE.md
│   ├── V_SPLINE_METHODOLOGY.md
│   ├── DEBUG_ARTIFACTS.md
│   └── FLIGHT_RULES.md
└── src/
    ├── adsb_viewer/                # Dash/Flask + Cesium-style viewer assets
    ├── config/                     # default flight_rules.json location
    ├── vspline/                    # V-Spline fitting cores and local policies
    └── ...                         # pipeline, segmentation, Kalman/RTS, adapters
```

## Requirements

- Python 3.9 or newer
- A SQLite database containing ADS-B rows
- A curated `flight_rules.json` file
- Python packages used by the pipeline, evaluation, and viewer

The project metadata declares the package name `adsb-collector` and core dependencies such as `redis` and `pyModeS`. The reconstruction, evaluation, and viewer code also uses scientific and web packages such as NumPy, pandas, Dash, and Flask. If your environment does not already include them, install them before running the full workflow.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy pandas dash flask
```

Depending on the enabled reconstruction backends, your local environment may also require additional scientific packages used by the spline and filtering code.

## Input data

This repository expects a local SQLite ADS-B database. The database is not included in the repository.

At minimum, the data source should provide enough information to derive:

- timestamp,
- ICAO or track identity,
- latitude and longitude,
- altitude or vertical position,
- ground speed,
- track angle,
- vertical rate when available,
- optional CRC or message-quality fields.

The exact raw column names can be defined in `flight_rules.json`.

## Flight rules

Flight rules are curated experiment definitions. They decide which records belong to a reconstruction case and how vertical reference, event filtering, and quality policy should be applied.

Default path:

```text
src/config/flight_rules.json
```

A rule normally defines:

```json
{
  "track_id": "4BAAD9",
  "icao": "4BAAD9",
  "time_window": {
    "first_point_unix": 1760000000,
    "last_point_unix": 1760001200
  },
  "field_elevation": {
    "method": "fixed_ft_msl",
    "value": 273.0
  },
  "allowed_events": ["position", "velocity"],
  "require_crc_ok": true
}
```

Useful rule fields include:

- `track_id`: unique case identifier.
- `icao`: aircraft ICAO address.
- `time_window`: accepted surveillance interval.
- `field_elevation`: local vertical reference.
- `on_ground_window`: optional runway/on-ground interval.
- `allowed_events`: accepted event kinds.
- `require_crc_ok`: whether to reject CRC-invalid rows when that field exists.
- `origin_lat_deg` / `origin_lon_deg`: optional local horizontal origin override.
- `raw_time_column`, `raw_icao_column`, `raw_crc_ok_column`, `raw_event_kind_column`: column-name mapping for different SQLite extracts.
- `keyframe_time_quantization_s`: time bucket size before keyframe aggregation.

## Quick start

### 1. Configure your input paths

```bash
export ADSB_SQLITE_PATH=/path/to/adsb_raw.sqlite
export TRACK_RULES_PATH=src/config/flight_rules.json
export TRACK_OUTPUT_DIR=track_output
```

On Windows PowerShell:

```powershell
$env:ADSB_SQLITE_PATH="C:\path\to\adsb_raw.sqlite"
$env:TRACK_RULES_PATH="src\config\flight_rules.json"
$env:TRACK_OUTPUT_DIR="track_output"
```

### 2. Build reconstruction outputs

```bash
python main_pipeline.py
```

To process only selected aircraft or track IDs:

```bash
ICAO_LIST=4BAAD9,502D5A python main_pipeline.py
```

To emit only selected presets:

```bash
RECONSTRUCTION_PRESETS=balanced,smooth python main_pipeline.py
```

### 3. Evaluate reconstructions

Evaluate the default or selected flight:

```bash
python evaluate_reconstructions.py
python evaluate_reconstructions.py --flight-id 4BAAD9
python evaluate_reconstructions.py --icao 4BAAD9
```

Evaluate all configured dataset flights:

```bash
python evaluate_reconstructions.py --all-flights
```

Dataset-level reports are written under:

```text
track_output/dataset_evaluation/
```

unless another output directory is supplied.

### 4. Launch the 3D viewer

```bash
python main.py
```

By default the viewer opens:

```text
http://127.0.0.1:8050
```

The viewer reads from:

```text
track_output/
```

and allows available flights and reconstruction methods to be selected through the browser interface.

## Common runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `ADSB_SQLITE_PATH` | `adsb_raw.sqlite` | SQLite ADS-B database path. |
| `TRACK_OUTPUT_DIR` | `track_output` | Output folder for manifests, methods, and debug artifacts. |
| `TRACK_LOG_DIR` | `logs` | Pipeline log folder. |
| `TRACK_RULES_PATH` | `src/config/flight_rules.json` | Optional alternate rule file. |
| `ICAO_LIST` | unset | Comma-separated track IDs or ICAO values to process. |
| `CLEAN_OUTPUT_DIR` | `false` | Whether to clear the output directory before running. |
| `RECONSTRUCTION_PRESETS` | `balanced,accurate,smooth` | Presets to emit. |
| `V_SPLINE_OUTPUT_BACKENDS` | project default set | Comma-separated V-Spline backends to emit. |
| `KALMAN_RTS_OUTPUT` | `true` | Emit Kalman/RTS methods. |
| `WRITE_DEBUG_ARTIFACTS` | `true` | Write per-flight debug artifacts. |
| `PROGRESS` | `true` | Show progress bars. |
| `SEGMENT_TUNING` | `true` | Enable per-segment local candidate tuning. |
| `ADAPTIVE_RESEGMENTATION` | `true` | Split persistently poor local fits when enabled. |
| `DYNAMIC_HARD_GAP_S` | `30.0` | Hard-gap threshold used by segmentation. |
| `V_SPLINE_USE_KALMAN_BOUNDARY_PRIOR` | `false` | Enable optional quintic Kalman-boundary method. |

## Output structure

A successful pipeline run creates a `track_output/` folder similar to:

```text
track_output/
├── flights.json
├── dataset_evaluation/
│   ├── dataset_summary.json
│   ├── dataset_metrics.csv
│   ├── dataset_group_scores.csv
│   └── dataset_report.md
└── flights/
    └── <track_id>/
        ├── flight.json
        ├── methods/
        │   ├── raw_adsb.json
        │   ├── kalman_rts_balanced.json
        │   ├── kalman_rts_accurate.json
        │   ├── kalman_rts_smooth.json
        │   ├── aviation_v_spline_quintic_balanced.json
        │   └── minimal/
        │       └── ...
        └── debug/
            ├── debug_manifest.json
            ├── flight_rule.json
            ├── config_snapshot.json
            ├── raw_loader_report.json
            ├── normalization_report.json
            ├── prepared_samples_report.json
            ├── segmentation.json
            ├── boundary_states.json
            ├── reconstruction_quality.json
            ├── segment_metrics.csv
            ├── join_metrics.csv
            ├── trajectory_model_metrics.csv
            ├── evaluation_metrics.csv
            ├── evaluation_group_scores.csv
            ├── evaluation_report.md
            ├── flight_summary.json
            └── flight.log
```

Exact method filenames depend on the enabled presets and backends.

## Reconstruction method families

### Raw ADS-B baseline

`raw_adsb` stores accepted keyframes and raw evidence for comparison. It is a measurement baseline, not a ground-truth trajectory.

### Kalman/RTS

`kalman_rts_<preset>` fits a whole-track state-space smoother. It provides the main classical reference family and is emitted for `accurate`, `balanced`, and `smooth` presets when enabled.

### V-Spline families

The V-Spline methods fit continuous local curves using position residuals, velocity residuals, and smoothness penalties. Depending on the backend, the pipeline can emit:

```text
v_spline_bspline_<preset>
v_spline_hermite_<preset>
aviation_v_spline_bspline_global_<preset>
v_spline_bspline_overlap_<preset>
v_spline_bspline_join_smooth_<preset>
v_spline_hermite_stable_<preset>
aviation_v_spline_quintic_<preset>
aviation_v_spline_quintic_kalman_boundary_<preset>
```

The aviation quintic V-Spline uses degree-5 B-splines with acceleration, jerk, and snap regularisation, endpoint guards, velocity confidence, robust boundary anchors, and soft derivative priors at ordinary joins.

### Synthetic-gap diagnostics

Synthetic-gap methods use the suffix:

```text
_synthetic_gap
```

They are diagnostic stress tests where interior evidence is withheld. They should be interpreted separately from ordinary reconstruction methods.

## Evaluation

The evaluator compares generated method JSON files against the available ADS-B evidence and writes both metric-level and group-level outputs.

Main metric groups:

- smoothness,
- aircraft dynamics,
- raw position fidelity,
- raw velocity fidelity,
- shape similarity,
- endpoint artifacts,
- gap behaviour,
- reference-free trajectory-model quality,
- envelope violations.

The overall score is a weighted engineering score. It is useful for comparing methods under the project’s rules, but it is not a universal physical-truth score.

## Viewer

The viewer is a diagnostic and communication tool. It displays raw observations and reconstructed method outputs in a time-synchronised 3D scene.

Typical uses:

- inspect raw ADS-B evidence gaps,
- compare raw dots with reconstructed paths,
- check segment boundaries,
- find suspicious joins or endpoint behaviour,
- inspect speed, heading, altitude, velocity, and acceleration traces,
- prepare visual case studies.

Launch:

```bash
python main.py
```

Then open:

```text
http://127.0.0.1:8050
```

## Methodology notes

This project intentionally separates measurement evidence from model output.

Raw ADS-B data can be irregular, asynchronous, incomplete, or receiver-dependent. Position and velocity evidence may not arrive as a single clean state vector. For that reason, the pipeline preserves raw events and keyframes for audit, but reconstruction methods use a stricter sample contract:

```text
time t
local position: x_m, y_m, z_m
local velocity: east_mps, north_mps, up_mps
```

This makes method comparison fairer because Kalman/RTS and V-Spline methods receive the same paired evidence.

## Troubleshooting

### `Missing manifest: track_output/flights.json`

Run the reconstruction pipeline first:

```bash
python main_pipeline.py
```

or pass the correct output folder:

```bash
python evaluate_reconstructions.py --output-dir /path/to/track_output
```

### No flights appear in the viewer

Check that `track_output/flights.json` exists and that each flight folder contains method JSON files under:

```text
track_output/flights/<track_id>/methods/
```

### A selected ICAO is not found

Check `src/config/flight_rules.json`. `ICAO_LIST` must match either a unique `track_id` or an unambiguous ICAO in the rule file.

### Reconstruction fails during sample preparation

Inspect:

```text
prepared_samples_report.json
normalization_report.json
flight_rule.json
```

Common causes are missing local position fields, missing local velocity fields, duplicate paired times, or too few strict paired samples.

### Spline output has suspicious joins or spikes

Inspect:

```text
segmentation.json
boundary_states.json
segment_metrics.csv
join_metrics.csv
```

A visually smooth line can still have poor derivative behaviour, so check smoothness, dynamics, gap metrics, and trajectory-model metrics alongside raw position errors.

## Documentation

The `docs/` folder gives deeper methodology details:

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — execution stages and runtime controls.
- [`docs/V_SPLINE_METHODOLOGY.md`](docs/V_SPLINE_METHODOLOGY.md) — V-Spline objective, basis families, presets, continuity claims, velocity confidence, and boundary terms.
- [`docs/DEBUG_ARTIFACTS.md`](docs/DEBUG_ARTIFACTS.md) — debug files, evaluation files, and diagnostic workflow.
- [`docs/FLIGHT_RULES.md`](docs/FLIGHT_RULES.md) — curated flight-rule format and practical rule checklist.

## Limitations

- Raw ADS-B observations are treated as surveillance evidence, not independent ground truth.
- Reconstruction scores are project-specific engineering metrics, not universal accuracy claims.
- Synthetic-gap outputs are robustness diagnostics and should not be mixed with ordinary production methods.
- Segment labels are modelling regions, not certified flight phases.
- Results depend on receiver coverage, SQLite field availability, configured flight rules, and enabled reconstruction backends.

## Citation / thesis context

This repository supports research on adaptive spline smoothing for ADS-B aircraft trajectory reconstruction. When using the code in academic work, describe both the data contract and the evaluation limits: the methods reconstruct plausible trajectory models from ADS-B evidence, but they do not prove the exact physical aircraft path without an independent reference trajectory.

## License

No license file is currently shown in the repository. Add a `LICENSE` file before distributing or reusing the project outside a private research context.
