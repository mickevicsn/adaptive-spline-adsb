#!/usr/bin/env python3
"""Evaluate generated reconstruction JSONs for one configured flight.

The script reads the already processed ``track_output`` folder, chooses a flight
from CLI/env/main.py config, loads every non-raw method in the flight manifest,
and compares them against the raw ADS-B keyframes over the whole processed
flight.  It writes JSON/CSV/Markdown reports under the flight debug directory.

Typical use from the project root after running ``python main.py``:

    python evaluate_reconstructions.py
    python evaluate_reconstructions.py --flight-id MY_TRACK_ID
    python evaluate_reconstructions.py --icao 4BAAD9
    python evaluate_reconstructions.py --all-flights
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

HARDCODED_DATASET_ICAOS: tuple[str, ...] = (
    "4ACA4A",
    "4BAAD9",
    "48ADA7",
    "48C229",
    "502CC7",
    "502CCF",
    "502CE3",
    "502CE4",
    "502D5A",
    "502D5C",
    "502D5D",
    "502D9F",
    "502D59",
    "502D96",
    "4D22BF",
    "48C1A7",
    "502D5E",
    "4601F5",
)

RAW_METHOD_ID = "raw_adsb"
GROUND_SPEED_THRESHOLD_MPS = 5.0
GROUND_ALT_THRESHOLD_M = 30.0
AIRBORNE_SPEED_MIN_MPS = 30.0
HF_CUTOFF_HZ = 0.4
G = 9.80665
LONG_GAP_THRESHOLD_S = 10.0
ENDPOINT_WINDOW_S = 10.0
TRAJECTORY_MODEL_METRIC_FAMILY = "reference_free_trajectory_model_metrics_v1"
TRAJECTORY_MODEL_MAX_SCORE = 100.0
MAX_SHAPE_POINTS = 2000
MAX_DTW_POINTS = 700
MAX_FRECHET_POINTS = 700

MAX_REASONABLE_GROUNDSPEED_MPS = 350.0
MAX_REASONABLE_VERTICAL_RATE_MPS = 35.0
MAX_REASONABLE_ACCEL_MPS2 = 20.0
MAX_REASONABLE_JERK_MPS3 = 20.0
MAX_REASONABLE_BANK_DEG = 75.0
MAX_REASONABLE_TURN_RATE_DEGPS = 20.0

LOWER_IS_BETTER_METRIC_GROUPS: dict[str, list[str]] = {
    "smoothness": [
        "accel_rms_mps2",
        "accel_p95_mps2",
        "accel_max_mps2",
        "jerk_rms_mps3",
        "jerk_p95_mps3",
        "jerk_max_mps3",
        "hf_accel_energy_ratio",
        "hf_jerk_energy_ratio",
    ],
    "aircraft_dynamics": [
        "curvature_airborne_rms",
        "curvature_airborne_p95",
        "turn_rate_airborne_p95_deg_s",
        "bank_angle_airborne_p95_deg",
    ],
    "raw_position_fidelity": [
        "raw_horizontal_position_error_m_rmse",
        "raw_horizontal_position_error_m_p95_abs",
        "raw_vertical_position_error_m_rmse",
        "raw_vertical_position_error_m_p95_abs",
        "raw_position_3d_error_m_rmse",
        "raw_position_3d_error_m_p95_abs",
        "raw_along_track_error_m_rmse",
        "raw_cross_track_error_m_rmse",
    ],
    "raw_velocity_fidelity": [
        "raw_horizontal_velocity_error_mps_rmse",
        "raw_groundspeed_error_mps_rmse",
        "raw_track_angle_error_deg_rmse",
    ],
    "shape_similarity": [
        "raw_to_reconstruction_hausdorff_distance_m",
        "symmetric_hausdorff_distance_m",
        "dtw_mean_step_distance_m",
        "discrete_frechet_distance_m",
    ],
    "endpoint_artifacts": [
        "endpoint_jerk_rms_ratio",
        "endpoint_accel_rms_ratio",
    ],
    "gap_behavior": [
        "gap_accel_p95_mps2",
        "gap_jerk_p95_mps3",
    ],
    "trajectory_model_reference_free": [
        "trajectory_model_score_loss_lower_is_better",
    ],
    "envelope_violations": [
        "envelope_violation_groundspeed_count",
        "envelope_violation_vertical_rate_count",
        "envelope_violation_accel_count",
        "envelope_violation_jerk_count",
        "envelope_violation_bank_count",
        "envelope_violation_turn_rate_count",
    ],
}

COMPARISON_GROUP_WEIGHTS: dict[str, float] = {
    "smoothness": 2.0,
    "aircraft_dynamics": 1.0,
    "raw_position_fidelity": 2.0,
    "raw_velocity_fidelity": 1.5,
    "shape_similarity": 1.0,
    "endpoint_artifacts": 0.5,
    "gap_behavior": 0.5,
    "trajectory_model_reference_free": 2.0,
    "envelope_violations": 1.0,
}


@dataclass(frozen=True)
class MethodPayload:
    method_id: str
    label: str
    detailed_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    flight_id: str
    icao: str
    output_dir: Path
    evaluation_dir: Path
    metrics_by_method: dict[str, dict[str, Any]]
    group_scores: dict[str, dict[str, Any]]
    overall_scores: dict[str, float]
    metric_winners: dict[str, dict[str, Any]]
    ranked_methods: list[dict[str, Any]]


@dataclass(frozen=True)
class DatasetEvaluationResult:
    output_dir: Path
    evaluation_dir: Path
    flight_results: list[EvaluationResult]
    flight_errors: list[dict[str, Any]]
    selected_flight_count: int
    weight_metric: str
    require_all_methods: bool
    metrics_by_method: dict[str, dict[str, Any]]
    group_scores: dict[str, dict[str, Any]]
    overall_scores: dict[str, float]
    metric_winners: dict[str, dict[str, Any]]
    ranked_methods: list[dict[str, Any]]
    method_coverage: dict[str, dict[str, Any]]
    per_flight_rows: list[dict[str, Any]]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = load_project_config(project_root)
    output_dir = Path(args.output_dir) if args.output_dir else get_config_output_dir(cfg, project_root)
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    manifest_path = output_dir / "flights.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run main.py first or pass --output-dir.")

    manifest = read_json(manifest_path)
    if args.all_flights:
        flight_entries = select_flights(manifest)
        dataset_result = evaluate_dataset(
            output_dir=output_dir,
            flight_entries=flight_entries,
            method_filter=args.methods,
            weight_metric=args.dataset_weight,
            require_all_methods=args.dataset_require_all_methods,
            skip_flight_errors=args.skip_flight_errors,
            dataset_output_dir=Path(args.dataset_output_dir) if args.dataset_output_dir else None,
        )
        write_dataset_reports(dataset_result)
        print_dataset_summary(dataset_result)
        return

    flight_entry = select_flight(manifest, cfg, args.flight_id, args.icao)
    result = evaluate_flight(output_dir=output_dir, flight_entry=flight_entry, method_filter=args.methods)
    write_reports(result)
    print_summary(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated reconstruction methods for one flight or the whole processed dataset.")
    parser.add_argument("--output-dir", type=str, default=None, help="track_output directory; defaults to main.py config.")
    parser.add_argument("--flight-id", type=str, default=os.getenv("EVALUATION_FLIGHT_ID"), help="Manifest flightId to evaluate. In --all-flights mode this may be a comma-separated filter.")
    parser.add_argument("--icao", type=str, default=os.getenv("EVALUATION_ICAO"), help="ICAO to evaluate if flight-id is not supplied. In --all-flights mode this may be a comma-separated filter.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method ids to evaluate. Defaults to every non-raw method.")
    parser.add_argument("--all-flights", action="store_true", default=env_bool("EVALUATE_ALL_FLIGHTS", False), help="Evaluate every selected flight in flights.json and write dataset-level weighted reports.")
    parser.add_argument("--dataset-output-dir", type=str, default=os.getenv("DATASET_EVALUATION_DIR"), help="Directory for dataset-level reports. Defaults to <output-dir>/dataset_evaluation.")
    parser.add_argument("--dataset-weight", choices=("raw_sample_count", "raw_position_overlap_count", "duration_s", "uniform"), default=os.getenv("DATASET_EVALUATION_WEIGHT", "raw_sample_count"), help="Per-flight weight used when aggregating numeric metrics across the dataset.")
    parser.add_argument("--dataset-require-all-methods", action="store_true", default=env_bool("DATASET_REQUIRE_ALL_METHODS", False), help="Only rank methods present for every successfully evaluated flight.")
    parser.add_argument("--skip-flight-errors", action="store_true", default=env_bool("SKIP_FLIGHT_ERRORS", False), help="Continue dataset evaluation when a flight cannot be evaluated; errors are written to dataset_flight_errors.csv/json.")
    return parser.parse_args()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def load_project_config(project_root: Path) -> Any | None:
    src_dir = project_root / "src"
    for path in (project_root, src_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import main as project_main  # type: ignore

        if hasattr(project_main, "build_config"):
            return project_main.build_config()
    except Exception:
        return None
    return None


def get_config_output_dir(cfg: Any | None, project_root: Path) -> Path:
    if cfg is not None:
        try:
            return Path(cfg.paths.output_dir)
        except Exception:
            pass
    raw = os.getenv("TRACK_OUTPUT_DIR", "track_output")
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def select_flight(manifest: dict[str, Any], cfg: Any | None, flight_id: str | None, icao: str | None) -> dict[str, Any]:
    flights = manifest.get("flights") or []
    if not isinstance(flights, list) or not flights:
        raise ValueError("flights.json does not contain any flights")

    def norm(value: Any) -> str:
        return str(value or "").strip().upper()

    candidates: list[str] = []
    if flight_id:
        candidates.append(norm(flight_id))
    if icao:
        candidates.append(norm(icao))
    if not candidates and cfg is not None:
        icao_list = getattr(cfg, "icao_list", None)
        if icao_list:
            candidates.extend(norm(item) for item in icao_list)

    for candidate in candidates:
        for entry in flights:
            if norm(entry.get("flightId")) == candidate or norm(entry.get("icao")) == candidate:
                return entry

    if len(flights) == 1:
        return flights[0]
    return flights[0]


def select_flights(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the hard-coded dataset flights from flights.json.

    This deliberately ignores:
    - --icao
    - --flight-id
    - ICAO_LIST
    - EVALUATION_ICAO
    - EVALUATION_FLIGHT_ID
    - main.py cfg.icao_list
    """
    flights = manifest.get("flights") or []
    if not isinstance(flights, list) or not flights:
        raise ValueError("flights.json does not contain any flights")

    def norm(value: Any) -> str:
        return str(value or "").strip().upper()

    wanted = {norm(item) for item in HARDCODED_DATASET_ICAOS}

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entry in flights:
        if not isinstance(entry, dict):
            continue

        flight_id = norm(entry.get("flightId"))
        icao = norm(entry.get("icao"))

        if flight_id in wanted or icao in wanted:
            unique = flight_id or icao
            if unique and unique not in seen:
                selected.append(entry)
                seen.add(unique)

    found = {
        norm(entry.get("flightId")) for entry in selected
    } | {
        norm(entry.get("icao")) for entry in selected
    }

    missing = [item for item in HARDCODED_DATASET_ICAOS if norm(item) not in found]
    if missing:
        raise ValueError(
            "These hard-coded flights are missing from flights.json: "
            + ", ".join(missing)
        )

    print(
        "Hard-coded dataset flights selected: "
        + ", ".join(str(entry.get("flightId") or entry.get("icao")) for entry in selected)
    )

    return selected

def evaluate_flight(*, output_dir: Path, flight_entry: dict[str, Any], method_filter: Iterable[str] | None = None) -> EvaluationResult:
    flight_id = str(flight_entry.get("flightId") or "")
    icao = str(flight_entry.get("icao") or "")
    if not flight_id:
        raise ValueError("Selected manifest entry has no flightId")
    flight_dir = output_dir / "flights" / flight_id
    evaluation_dir = flight_dir / "debug"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    methods = list(flight_entry.get("methods") or [])
    raw_entry = next((m for m in methods if str(m.get("methodId")) == RAW_METHOD_ID), None)
    if raw_entry is None:
        raise ValueError(f"Flight {flight_id} has no raw_adsb method in manifest")
    raw_payload = load_method_payload(output_dir, raw_entry).payload
    raw_df = parse_keyframes(raw_payload.get("raw_keyframes") or raw_payload.get("render_keyframes") or [])
    raw_df = raw_df.dropna(subset=["t", "x", "y", "z"], how="any")
    if raw_df.empty:
        raise ValueError(f"Raw payload for {flight_id} has no local x/y/z position samples")

    allowed = {str(m) for m in method_filter} if method_filter else None
    method_payloads: list[MethodPayload] = []
    for entry in methods:
        method_id = str(entry.get("methodId") or "")
        if not method_id or method_id == RAW_METHOD_ID:
            continue
        if allowed is not None and method_id not in allowed:
            continue
        try:
            method_payloads.append(load_method_payload(output_dir, entry))
        except FileNotFoundError as exc:
            print(f"Skipping {method_id}: {exc}")

    if len(method_payloads) < 1:
        raise ValueError("No reconstruction methods found to evaluate")

    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method in method_payloads:
        render_df = parse_keyframes(method.payload.get("render_keyframes") or [])
        render_df = render_df.dropna(subset=["t", "x", "y", "z"], how="any")
        metrics_by_method[method.method_id] = evaluate_method(raw_df, render_df, method)

    group_scores, overall_scores, metric_winners = score_methods(metrics_by_method)
    ranked_methods = [
        {"method_id": method_id, "overall_score_lower_is_better": score, "rank": rank + 1}
        for rank, (method_id, score) in enumerate(sorted(overall_scores.items(), key=lambda item: item[1]))
    ]
    return EvaluationResult(
        flight_id=flight_id,
        icao=icao,
        output_dir=output_dir,
        evaluation_dir=evaluation_dir,
        metrics_by_method=metrics_by_method,
        group_scores=group_scores,
        overall_scores=overall_scores,
        metric_winners=metric_winners,
        ranked_methods=ranked_methods,
    )


def load_method_payload(output_dir: Path, method_entry: dict[str, Any]) -> MethodPayload:
    method_id = str(method_entry.get("methodId") or "")
    label = str(method_entry.get("label") or method_id)
    rel = method_entry.get("detailedFile") or method_entry.get("file")
    if not rel:
        raise FileNotFoundError(f"method {method_id} has no file path")
    path = output_dir / str(rel)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return MethodPayload(method_id=method_id, label=label, detailed_path=path, payload=read_json(path))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_keyframes(keyframes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in keyframes:
        if not isinstance(item, dict):
            continue
        pos = item.get("position") if isinstance(item.get("position"), dict) else {}
        vel = item.get("velocity") if isinstance(item.get("velocity"), dict) else {}
        acc = item.get("acceleration") if isinstance(item.get("acceleration"), dict) else {}
        rows.append(
            {
                "id": item.get("id"),
                "t": finite_or_nan(item.get("t")),
                "x": first_finite(pos, "x_m", "east_m", "east"),
                "y": first_finite(pos, "y_m", "north_m", "north"),
                "z": first_finite(pos, "z_m", "up_m", "alt_m", "altitude_m"),
                "vx": first_finite(vel, "east_mps", "vx_mps", "x_mps", "east"),
                "vy": first_finite(vel, "north_mps", "vy_mps", "y_mps", "north"),
                "vz": first_finite(vel, "up_mps", "vertical_mps", "vz_mps", "z_mps", "up"),
                "ax": first_finite(acc, "east_mps2", "ax_mps2", "x_mps2", "east"),
                "ay": first_finite(acc, "north_mps2", "ay_mps2", "y_mps2", "north"),
                "az": first_finite(acc, "up_mps2", "vertical_mps2", "az_mps2", "z_mps2", "up"),
                "segment_id": item.get("segment_id") or (item.get("quality") or {}).get("segment_id") if isinstance(item.get("quality"), dict) else item.get("segment_id"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"])
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["t"]).sort_values("t")
    numeric_cols = [c for c in ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"] if c in df.columns]
    if not df.empty:
        grouped = df.groupby("t", as_index=False)
        means = grouped[numeric_cols].mean()
        firsts = grouped[[c for c in df.columns if c not in numeric_cols and c != "t"]].first()
        df = pd.merge(means, firsts, on="t", how="left") if not firsts.empty else means
        df = df.sort_values("t").reset_index(drop=True)
    return df


def first_finite(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in mapping:
            value = finite_or_nan(mapping.get(key))
            if math.isfinite(value):
                return value
    return float("nan")


def finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def evaluate_method(raw_df: pd.DataFrame, render_df: pd.DataFrame, method: MethodPayload) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "method_id": method.method_id,
        "label": method.label,
        "detailed_file": str(method.detailed_path),
        "raw_sample_count": int(len(raw_df)),
        "render_sample_count": int(len(render_df)),
    }
    if render_df.empty or len(render_df) < 2:
        metrics["error"] = "render_keyframes missing or too short"
        return metrics

    metrics.update(time_grid_metrics(render_df))
    metrics.update(raw_position_fidelity(raw_df, render_df))
    metrics.update(raw_velocity_fidelity(raw_df, render_df))
    metrics.update(shape_similarity(raw_df, render_df))
    motion = motion_arrays(render_df)
    metrics.update(smoothness_metrics(render_df, motion))
    metrics.update(aircraft_dynamics_metrics(render_df, motion))
    metrics.update(endpoint_artifact_metrics(render_df, motion))
    metrics.update(gap_behavior_metrics(raw_df, render_df, motion))
    metrics.update(envelope_violation_metrics(render_df, motion))
    metrics.update(trajectory_model_metrics(raw_df, render_df, method, motion))

    quality = method.payload.get("quality") if isinstance(method.payload.get("quality"), dict) else {}
    if quality:
        metrics["quality_backend"] = quality.get("reconstruction_backend")
        metrics["quality_fit_mode"] = quality.get("fit_mode")
        metrics["quality_segment_count"] = quality.get("segment_count")
        metrics["quality_preset"] = quality.get("preset")
    return clean_json(metrics)


def time_grid_metrics(render_df: pd.DataFrame) -> dict[str, Any]:
    t = render_df["t"].to_numpy(float)
    dt = np.diff(t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    return {
        "duration_s": float(t[-1] - t[0]) if t.size else None,
        "render_dt_median_s": float(np.median(finite_dt)) if finite_dt.size else None,
        "render_dt_p95_s": float(np.quantile(finite_dt, 0.95)) if finite_dt.size else None,
        "render_long_gap_count": int(np.sum(finite_dt > LONG_GAP_THRESHOLD_S)) if finite_dt.size else 0,
    }


def raw_position_fidelity(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> dict[str, Any]:
    raw = raw_df.dropna(subset=["x", "y", "z"])
    if raw.empty:
        return {}
    pred = interpolate_columns(render_df, raw["t"].to_numpy(float), ["x", "y", "z"])
    mask = np.all(np.isfinite(pred), axis=1)
    true = raw[["x", "y", "z"]].to_numpy(float)[mask]
    pred = pred[mask]
    if true.size == 0:
        return {}
    delta = pred - true
    h = np.linalg.norm(delta[:, :2], axis=1)
    v = np.abs(delta[:, 2])
    e3 = np.linalg.norm(delta, axis=1)
    out = {
        "raw_position_overlap_count": int(true.shape[0]),
        "raw_horizontal_position_error_m_rmse": rms(h),
        "raw_horizontal_position_error_m_median_abs": q(h, 0.50),
        "raw_horizontal_position_error_m_p95_abs": q(h, 0.95),
        "raw_vertical_position_error_m_rmse": rms(v),
        "raw_vertical_position_error_m_p95_abs": q(v, 0.95),
        "raw_position_3d_error_m_rmse": rms(e3),
        "raw_position_3d_error_m_p95_abs": q(e3, 0.95),
        "raw_position_3d_error_m_max_abs": max_or_none(e3),
    }
    if {"vx", "vy"}.issubset(raw.columns):
        vv = raw[["vx", "vy"]].to_numpy(float)[mask]
        speed = np.linalg.norm(vv, axis=1)
        good = np.isfinite(speed) & (speed > GROUND_SPEED_THRESHOLD_MPS)
        if np.any(good):
            direction = vv[good] / speed[good, None]
            horiz_delta = delta[good, :2]
            along = np.sum(horiz_delta * direction, axis=1)
            cross = horiz_delta[:, 0] * (-direction[:, 1]) + horiz_delta[:, 1] * direction[:, 0]
            out["raw_along_track_error_m_rmse"] = rms(along)
            out["raw_cross_track_error_m_rmse"] = rms(cross)
    return out


def raw_velocity_fidelity(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> dict[str, Any]:
    cols = ["vx", "vy", "vz"]
    raw = raw_df.dropna(subset=cols, how="any")
    if raw.empty:
        return {}
    pred = interpolate_columns(render_df, raw["t"].to_numpy(float), cols)
    mask = np.all(np.isfinite(pred), axis=1)
    true = raw[cols].to_numpy(float)[mask]
    pred = pred[mask]
    if true.size == 0:
        return {}
    delta = pred - true
    h = np.linalg.norm(delta[:, :2], axis=1)
    speed_true = np.linalg.norm(true[:, :2], axis=1)
    speed_pred = np.linalg.norm(pred[:, :2], axis=1)
    speed_err = speed_pred - speed_true
    track_true = np.degrees(np.arctan2(true[:, 0], true[:, 1]))
    track_pred = np.degrees(np.arctan2(pred[:, 0], pred[:, 1]))
    track_err = angle_diff_deg(track_pred, track_true)
    moving = speed_true > GROUND_SPEED_THRESHOLD_MPS
    return {
        "raw_velocity_overlap_count": int(true.shape[0]),
        "raw_horizontal_velocity_error_mps_rmse": rms(h),
        "raw_groundspeed_error_mps_rmse": rms(speed_err),
        "raw_track_angle_error_deg_rmse": rms(track_err[moving]) if np.any(moving) else None,
        "raw_vertical_velocity_error_mps_rmse": rms(delta[:, 2]),
    }


def interpolate_columns(df: pd.DataFrame, t_query: np.ndarray, cols: list[str]) -> np.ndarray:
    t = df["t"].to_numpy(float)
    out = np.full((len(t_query), len(cols)), np.nan, dtype=float)
    valid_time = np.isfinite(t)
    if np.sum(valid_time) < 2:
        return out
    t = t[valid_time]
    order = np.argsort(t)
    t = t[order]
    for j, col in enumerate(cols):
        if col not in df.columns:
            continue
        values_full = df[col].to_numpy(float)[valid_time][order]
        good = np.isfinite(values_full)
        if np.sum(good) < 2:
            continue
        tq = np.asarray(t_query, dtype=float)
        inside = np.isfinite(tq) & (tq >= t[good][0]) & (tq <= t[good][-1])
        out[inside, j] = np.interp(tq[inside], t[good], values_full[good])
    return out


def shape_similarity(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> dict[str, Any]:
    raw = raw_df.dropna(subset=["x", "y", "z"])[["x", "y", "z"]].to_numpy(float)
    recon = render_df.dropna(subset=["x", "y", "z"])[["x", "y", "z"]].to_numpy(float)
    if raw.shape[0] < 2 or recon.shape[0] < 2:
        return {}
    raw_h = downsample_points(raw, MAX_SHAPE_POINTS)
    recon_h = downsample_points(recon, MAX_SHAPE_POINTS)
    d_raw_to_rec = directed_hausdorff(raw_h, recon_h)
    d_rec_to_raw = directed_hausdorff(recon_h, raw_h)
    raw_dtw = downsample_points(raw, MAX_DTW_POINTS)
    recon_dtw = downsample_points(recon, MAX_DTW_POINTS)
    raw_fr = downsample_points(raw, MAX_FRECHET_POINTS)
    recon_fr = downsample_points(recon, MAX_FRECHET_POINTS)
    return {
        "raw_to_reconstruction_hausdorff_distance_m": d_raw_to_rec,
        "reconstruction_to_raw_hausdorff_distance_m": d_rec_to_raw,
        "symmetric_hausdorff_distance_m": max(d_raw_to_rec, d_rec_to_raw),
        "dtw_mean_step_distance_m": dtw_mean_distance(raw_dtw, recon_dtw),
        "discrete_frechet_distance_m": discrete_frechet(raw_fr, recon_fr),
    }


def motion_arrays(render_df: pd.DataFrame) -> dict[str, np.ndarray]:
    t = render_df["t"].to_numpy(float)
    pos = render_df[["x", "y", "z"]].to_numpy(float)
    vel = render_df[["vx", "vy", "vz"]].to_numpy(float) if {"vx", "vy", "vz"}.issubset(render_df.columns) else np.full_like(pos, np.nan)
    acc = render_df[["ax", "ay", "az"]].to_numpy(float) if {"ax", "ay", "az"}.issubset(render_df.columns) else np.full_like(pos, np.nan)
    if not np.all(np.isfinite(vel)) and len(t) >= 2:
        vel = gradient_vector(pos, t)
    if not np.all(np.isfinite(acc)) and len(t) >= 2:
        acc = gradient_vector(vel, t)
    jerk = gradient_vector(acc, t) if len(t) >= 2 else np.zeros_like(acc)
    return {"t": t, "pos": pos, "vel": vel, "acc": acc, "jerk": jerk}


def gradient_vector(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    if values.shape[0] < 2 or not np.all(np.isfinite(t)):
        return out
    for j in range(values.shape[1]):
        y = values[:, j]
        if np.sum(np.isfinite(y)) < 2:
            continue
        filled = pd.Series(y).interpolate(limit_direction="both").to_numpy(float)
        out[:, j] = np.gradient(filled, t, edge_order=1)
    return out


def smoothness_metrics(render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    acc_norm = vec_norm(motion["acc"])
    jerk_norm = vec_norm(motion["jerk"])
    t = motion["t"]
    return {
        "accel_rms_mps2": rms(acc_norm),
        "accel_p95_mps2": q(acc_norm, 0.95),
        "accel_max_mps2": max_or_none(acc_norm),
        "jerk_rms_mps3": rms(jerk_norm),
        "jerk_p95_mps3": q(jerk_norm, 0.95),
        "jerk_max_mps3": max_or_none(jerk_norm),
        "hf_accel_energy_ratio": high_frequency_energy_ratio(acc_norm, t, HF_CUTOFF_HZ),
        "hf_jerk_energy_ratio": high_frequency_energy_ratio(jerk_norm, t, HF_CUTOFF_HZ),
    }


def aircraft_dynamics_metrics(render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    t = motion["t"]
    vel = motion["vel"]
    speed_h = np.linalg.norm(vel[:, :2], axis=1)
    vz = vel[:, 2]
    heading = np.unwrap(np.arctan2(vel[:, 0], vel[:, 1]))
    turn_rate_rad = gradient_scalar(heading, t)
    turn_rate_deg = np.degrees(turn_rate_rad)
    curvature = np.abs(turn_rate_rad) / np.maximum(speed_h, 1e-9)
    bank = np.degrees(np.arctan2(speed_h * turn_rate_rad, G))
    z = render_df["z"].to_numpy(float) if "z" in render_df.columns else np.full_like(speed_h, np.nan)
    airborne = np.isfinite(speed_h) & (speed_h >= AIRBORNE_SPEED_MIN_MPS) & (~np.isfinite(z) | (z >= GROUND_ALT_THRESHOLD_M))
    return {
        "groundspeed_mps_min": min_or_none(speed_h),
        "groundspeed_mps_median": q(speed_h, 0.50),
        "groundspeed_mps_p95": q(speed_h, 0.95),
        "vertical_rate_mps_p95_abs": q(np.abs(vz), 0.95),
        "curvature_airborne_rms": rms(curvature[airborne]) if np.any(airborne) else None,
        "curvature_airborne_p95": q(curvature[airborne], 0.95) if np.any(airborne) else None,
        "turn_rate_airborne_p95_deg_s": q(np.abs(turn_rate_deg[airborne]), 0.95) if np.any(airborne) else None,
        "bank_angle_airborne_p95_deg": q(np.abs(bank[airborne]), 0.95) if np.any(airborne) else None,
    }


def endpoint_artifact_metrics(render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    t = motion["t"]
    if t.size < 3:
        return {}
    endpoint = (t <= t[0] + ENDPOINT_WINDOW_S) | (t >= t[-1] - ENDPOINT_WINDOW_S)
    middle = ~endpoint
    acc = vec_norm(motion["acc"])
    jerk = vec_norm(motion["jerk"])
    return {
        "endpoint_accel_rms_ratio": ratio(rms(acc[endpoint]), rms(acc[middle])),
        "endpoint_jerk_rms_ratio": ratio(rms(jerk[endpoint]), rms(jerk[middle])),
    }


def gap_behavior_metrics(raw_df: pd.DataFrame, render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    raw_t = raw_df["t"].to_numpy(float)
    raw_t = raw_t[np.isfinite(raw_t)]
    raw_t.sort()
    if raw_t.size < 2:
        return {"raw_long_gap_count": 0, "gap_accel_p95_mps2": 0.0, "gap_jerk_p95_mps3": 0.0}
    gaps = [(raw_t[i], raw_t[i + 1]) for i in range(raw_t.size - 1) if raw_t[i + 1] - raw_t[i] > LONG_GAP_THRESHOLD_S]
    t = motion["t"]
    mask = np.zeros(t.size, dtype=bool)
    for a, b in gaps:
        mask |= (t > a) & (t < b)
    acc = vec_norm(motion["acc"])
    jerk = vec_norm(motion["jerk"])
    return {
        "raw_long_gap_count": int(len(gaps)),
        "gap_render_sample_count": int(np.sum(mask)),
        "gap_accel_p95_mps2": q(acc[mask], 0.95) if np.any(mask) else 0.0,
        "gap_jerk_p95_mps3": q(jerk[mask], 0.95) if np.any(mask) else 0.0,
    }


def envelope_violation_metrics(render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    vel = motion["vel"]
    speed_h = np.linalg.norm(vel[:, :2], axis=1)
    vz = np.abs(vel[:, 2])
    acc = vec_norm(motion["acc"])
    jerk = vec_norm(motion["jerk"])
    t = motion["t"]
    heading = np.unwrap(np.arctan2(vel[:, 0], vel[:, 1]))
    turn_rate_deg = np.abs(np.degrees(gradient_scalar(heading, t)))
    bank = np.abs(np.degrees(np.arctan2(speed_h * np.radians(turn_rate_deg), G)))
    return {
        "envelope_violation_groundspeed_count": count_gt(speed_h, MAX_REASONABLE_GROUNDSPEED_MPS),
        "envelope_violation_vertical_rate_count": count_gt(vz, MAX_REASONABLE_VERTICAL_RATE_MPS),
        "envelope_violation_accel_count": count_gt(acc, MAX_REASONABLE_ACCEL_MPS2),
        "envelope_violation_jerk_count": count_gt(jerk, MAX_REASONABLE_JERK_MPS3),
        "envelope_violation_bank_count": count_gt(bank, MAX_REASONABLE_BANK_DEG),
        "envelope_violation_turn_rate_count": count_gt(turn_rate_deg, MAX_REASONABLE_TURN_RATE_DEGPS),
    }


def gradient_scalar(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(t)):
        return np.full(values.shape, np.nan)
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(float)
    return np.gradient(filled, t, edge_order=1)


def high_frequency_energy_ratio(values: np.ndarray, t: np.ndarray, cutoff_hz: float) -> float | None:
    x = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    good = np.isfinite(x) & np.isfinite(t)
    x = x[good]
    t = t[good]
    if x.size < 8:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        return None
    x = x - float(np.mean(x))
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=median_dt)
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    return float(np.sum(power[freqs >= cutoff_hz]) / total)


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return points[idx]


def directed_hausdorff(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    max_min = 0.0
    for start in range(0, a.shape[0], chunk):
        aa = a[start : start + chunk]
        d2 = np.sum((aa[:, None, :] - b[None, :, :]) ** 2, axis=2)
        nearest = np.sqrt(np.min(d2, axis=1))
        max_min = max(max_min, float(np.max(nearest)))
    return max_min


def dtw_mean_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return None
    prev = np.full(b.shape[0] + 1, np.inf, dtype=float)
    curr = np.full(b.shape[0] + 1, np.inf, dtype=float)
    prev[0] = 0.0
    for i in range(1, a.shape[0] + 1):
        curr[0] = np.inf
        for j in range(1, b.shape[0] + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    path_len = a.shape[0] + b.shape[0]
    return float(prev[b.shape[0]] / max(1, path_len))


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return None
    ca = np.full((a.shape[0], b.shape[0]), np.inf, dtype=float)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            d = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[-1, -1])



def trajectory_model_metrics(raw_df: pd.DataFrame, render_df: pd.DataFrame, method: MethodPayload, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reference-free scores for the trajectory as a model, not as a truth match.

    These metrics intentionally do not use external truth data.  Raw ADS-B is
    treated as noisy evidence, while internal derivative closure, event-aware
    continuity, physical plausibility, useful dynamic detail, and hard-gap
    honesty are scored as trajectory-model properties.
    """
    t = motion["t"]
    pos = motion["pos"]
    vel = motion["vel"]
    acc = motion["acc"]
    jerk = motion["jerk"]
    snap = gradient_vector(jerk, t) if len(t) >= 2 else np.full_like(jerk, np.nan)
    speed_h = np.linalg.norm(vel[:, :2], axis=1)
    speed_3d = np.linalg.norm(vel, axis=1)
    z = render_df["z"].to_numpy(float) if "z" in render_df.columns else np.full_like(speed_h, np.nan)
    airborne = np.isfinite(speed_h) & ((speed_h >= AIRBORNE_SPEED_MIN_MPS) | (np.isfinite(z) & (z >= GROUND_ALT_THRESHOLD_M)))
    speed_floor = np.where(airborne, 35.0, 6.0)

    raw_position_rmse = _position_evidence_rmse(raw_df, render_df)
    raw_velocity_rmse = _velocity_evidence_rmse(raw_df, render_df)
    fd_velocity_rmse = _finite_difference_velocity_evidence_rmse(raw_df, render_df)
    closure = _derivative_closure_metrics(render_df, t, pos)
    dynamics = _trajectory_dynamic_metrics(t, pos, vel, acc, jerk, snap, speed_h, speed_3d, speed_floor)
    joins = _event_aware_join_metrics(raw_df, render_df, motion)
    gaps = _hard_gap_honesty_metrics(raw_df, render_df)
    detail = _dynamic_detail_metrics(raw_df, render_df, motion)
    locality_score = _locality_scope_score(method)

    position_score = score_lower_is_better(raw_position_rmse, good=12.0, poor=160.0, default=65.0)
    velocity_score = score_lower_is_better(raw_velocity_rmse, good=2.5, poor=32.0, default=55.0)
    fd_kinematics_score = score_lower_is_better(fd_velocity_rmse, good=4.0, poor=45.0, default=55.0)
    derivative_closure_score = weighted_score(
        {
            "velocity_closure": score_lower_is_better(closure.get("velocity_closure_rmse_mps"), 0.35, 8.0, 70.0),
            "acceleration_closure": score_lower_is_better(closure.get("acceleration_closure_rmse_mps2"), 0.50, 6.0, 70.0),
        },
        {"velocity_closure": 0.55, "acceleration_closure": 0.45},
    )
    smoothness_score = weighted_score(
        {
            "speed_normalized_jerk": score_lower_is_better(dynamics.get("speed_normalized_jerk_p95_1_s2"), 0.012, 0.22, 60.0),
            "snap_proxy": score_lower_is_better(dynamics.get("snap_proxy_p95_mps4"), 0.8, 12.0, 60.0),
            "jerk_hf_energy": score_lower_is_better(dynamics.get("jerk_high_frequency_energy_ratio"), 0.015, 0.40, 65.0),
            "accel_variation": score_lower_is_better(dynamics.get("accel_variation_p95_mps2"), 0.9, 7.0, 65.0),
        },
        {"speed_normalized_jerk": 0.35, "snap_proxy": 0.25, "jerk_hf_energy": 0.25, "accel_variation": 0.15},
    )
    physical_score = weighted_score(
        {
            "acceleration": score_lower_is_better(dynamics.get("accel_p95_mps2"), 2.2, 9.0, 60.0),
            "vertical_rate": score_lower_is_better(dynamics.get("vertical_rate_p95_abs_mps"), 12.0, 35.0, 70.0),
            "turn_rate": score_lower_is_better(dynamics.get("turn_rate_p95_abs_deg_s"), 3.5, 15.0, 65.0),
            "bank": score_lower_is_better(dynamics.get("bank_angle_p95_abs_deg"), 35.0, 75.0, 70.0),
        },
        {"acceleration": 0.35, "vertical_rate": 0.20, "turn_rate": 0.25, "bank": 0.20},
    )
    detail_score = weighted_score(
        {
            "speed_detail_ratio": score_ratio_band(detail.get("speed_detail_ratio"), low_good=0.45, high_good=1.25, low_bad=0.08, high_bad=2.8, default=55.0),
            "vertical_detail_ratio": score_ratio_band(detail.get("vertical_detail_ratio"), low_good=0.40, high_good=1.35, low_bad=0.05, high_bad=3.0, default=55.0),
            "noise_chasing_guard": score_lower_is_better(dynamics.get("jerk_high_frequency_energy_ratio"), 0.02, 0.45, 65.0),
        },
        {"speed_detail_ratio": 0.35, "vertical_detail_ratio": 0.35, "noise_chasing_guard": 0.30},
    )
    join_score = weighted_score(
        {
            "position_join_closure": score_lower_is_better(joins.get("normal_join_position_closure_p95_m"), 4.0, 70.0, 85.0),
            "velocity_jump": score_lower_is_better(joins.get("normal_join_velocity_jump_p95_mps"), 1.5, 18.0, 85.0),
            "acceleration_jump": score_lower_is_better(joins.get("normal_join_accel_jump_p95_mps2"), 0.8, 10.0, 85.0),
        },
        {"position_join_closure": 0.35, "velocity_jump": 0.35, "acceleration_jump": 0.30},
    )
    hard_gap_score = score_lower_is_better(gaps.get("hard_gap_continuity_coverage_ratio"), 0.0, 0.80, 100.0)

    component_scores = {
        "position_observation_evidence_score": position_score,
        "velocity_evidence_score": velocity_score,
        "finite_difference_kinematics_score": fd_kinematics_score,
        "derivative_closure_score": derivative_closure_score,
        "trajectory_smoothness_score": smoothness_score,
        "physical_plausibility_score": physical_score,
        "dynamic_detail_preservation_score": detail_score,
        "event_aware_join_score": join_score,
        "hard_gap_honesty_score": hard_gap_score,
        "locality_scope_score": locality_score,
    }
    component_weights = {
        "position_observation_evidence_score": 0.10,
        "velocity_evidence_score": 0.15,
        "finite_difference_kinematics_score": 0.13,
        "derivative_closure_score": 0.12,
        "trajectory_smoothness_score": 0.18,
        "physical_plausibility_score": 0.12,
        "dynamic_detail_preservation_score": 0.10,
        "event_aware_join_score": 0.06,
        "hard_gap_honesty_score": 0.03,
        "locality_scope_score": 0.01,
    }
    weighted = weighted_score(component_scores, component_weights)

    out: dict[str, Any] = {
        "trajectory_model_metric_family": TRAJECTORY_MODEL_METRIC_FAMILY,
        "trajectory_model_truth_data_used": False,
        "trajectory_model_weighted_score_higher_is_better": weighted,
        "trajectory_model_score_loss_lower_is_better": (TRAJECTORY_MODEL_MAX_SCORE - weighted) if weighted is not None else None,
        "trajectory_model_component_scores": component_scores,
        "trajectory_model_component_weights": component_weights,
        "trajectory_model_raw_position_evidence_rmse_m": raw_position_rmse,
        "trajectory_model_raw_velocity_evidence_rmse_mps": raw_velocity_rmse,
        "trajectory_model_fd_velocity_evidence_rmse_mps": fd_velocity_rmse,
        "trajectory_model_locality_scope_note": _locality_scope_note(method),
    }
    out.update({f"trajectory_model_component_{k}": v for k, v in component_scores.items()})
    out.update({f"trajectory_model_{k}": v for k, v in closure.items()})
    out.update({f"trajectory_model_{k}": v for k, v in dynamics.items()})
    out.update({f"trajectory_model_{k}": v for k, v in joins.items()})
    out.update({f"trajectory_model_{k}": v for k, v in gaps.items()})
    out.update({f"trajectory_model_{k}": v for k, v in detail.items()})
    return out


def _position_evidence_rmse(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> float | None:
    raw = raw_df.dropna(subset=["x", "y", "z"])
    if raw.empty:
        return None
    pred = interpolate_columns(render_df, raw["t"].to_numpy(float), ["x", "y", "z"])
    true = raw[["x", "y", "z"]].to_numpy(float)
    mask = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(true), axis=1)
    if not np.any(mask):
        return None
    return rms(np.linalg.norm(pred[mask] - true[mask], axis=1))


def _velocity_evidence_rmse(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> float | None:
    cols = ["vx", "vy", "vz"]
    raw = raw_df.dropna(subset=cols, how="any")
    if raw.empty:
        return None
    pred = interpolate_columns(render_df, raw["t"].to_numpy(float), cols)
    true = raw[cols].to_numpy(float)
    mask = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(true), axis=1)
    if not np.any(mask):
        return None
    return rms(np.linalg.norm(pred[mask] - true[mask], axis=1))


def _finite_difference_velocity_evidence_rmse(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> float | None:
    raw = raw_df.dropna(subset=["t", "x", "y", "z"])
    if len(raw) < 4:
        return None
    raw_t = raw["t"].to_numpy(float)
    raw_pos = raw[["x", "y", "z"]].to_numpy(float)
    raw_fd_vel = gradient_vector(raw_pos, raw_t)
    pred = interpolate_columns(render_df, raw_t, ["vx", "vy", "vz"])
    mask = np.all(np.isfinite(raw_fd_vel), axis=1) & np.all(np.isfinite(pred), axis=1)
    if np.sum(mask) < 3:
        return None
    dt = np.diff(raw_t)
    typical_dt = float(np.median(dt[np.isfinite(dt) & (dt > 0)])) if np.any(np.isfinite(dt) & (dt > 0)) else 1.0
    # Very long raw gaps make finite differences physically meaningless; exclude
    # points adjacent to such gaps from the evidence score.
    left_gap = np.r_[False, np.diff(raw_t) > max(LONG_GAP_THRESHOLD_S, 4.0 * typical_dt)]
    right_gap = np.r_[np.diff(raw_t) > max(LONG_GAP_THRESHOLD_S, 4.0 * typical_dt), False]
    mask &= ~(left_gap | right_gap)
    if np.sum(mask) < 3:
        return None
    return rms(np.linalg.norm(pred[mask] - raw_fd_vel[mask], axis=1))


def _derivative_closure_metrics(render_df: pd.DataFrame, t: np.ndarray, pos: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(t) < 3:
        return out
    vel_cols = ["vx", "vy", "vz"]
    acc_cols = ["ax", "ay", "az"]
    vel_fd = gradient_vector(pos, t)
    if all(c in render_df.columns for c in vel_cols):
        vel_given = render_df[vel_cols].to_numpy(float)
        mask = np.all(np.isfinite(vel_given), axis=1) & np.all(np.isfinite(vel_fd), axis=1)
        if np.sum(mask) >= 3:
            out["velocity_closure_rmse_mps"] = rms(np.linalg.norm(vel_given[mask] - vel_fd[mask], axis=1))
            out["velocity_closure_p95_mps"] = q(np.linalg.norm(vel_given[mask] - vel_fd[mask], axis=1), 0.95)
    if all(c in render_df.columns for c in vel_cols + acc_cols):
        vel_given = render_df[vel_cols].to_numpy(float)
        acc_given = render_df[acc_cols].to_numpy(float)
        acc_fd = gradient_vector(vel_given, t)
        mask = np.all(np.isfinite(acc_given), axis=1) & np.all(np.isfinite(acc_fd), axis=1)
        if np.sum(mask) >= 3:
            out["acceleration_closure_rmse_mps2"] = rms(np.linalg.norm(acc_given[mask] - acc_fd[mask], axis=1))
            out["acceleration_closure_p95_mps2"] = q(np.linalg.norm(acc_given[mask] - acc_fd[mask], axis=1), 0.95)
    return out


def _trajectory_dynamic_metrics(
    t: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    jerk: np.ndarray,
    snap: np.ndarray,
    speed_h: np.ndarray,
    speed_3d: np.ndarray,
    speed_floor: np.ndarray,
) -> dict[str, Any]:
    acc_norm = vec_norm(acc)
    jerk_norm = vec_norm(jerk)
    snap_norm = vec_norm(snap)
    normalized_jerk = jerk_norm / np.maximum(speed_3d, speed_floor)
    heading = np.unwrap(np.arctan2(vel[:, 0], vel[:, 1]))
    turn_rate_deg = np.abs(np.degrees(gradient_scalar(heading, t)))
    bank = np.abs(np.degrees(np.arctan2(speed_h * np.radians(turn_rate_deg), G)))
    accel_variation = vec_norm(gradient_vector(acc, t)) if len(t) >= 2 else np.full_like(acc_norm, np.nan)
    return {
        "accel_p95_mps2": q(acc_norm, 0.95),
        "accel_variation_p95_mps2": q(accel_variation, 0.95),
        "vertical_rate_p95_abs_mps": q(np.abs(vel[:, 2]), 0.95),
        "turn_rate_p95_abs_deg_s": q(turn_rate_deg, 0.95),
        "bank_angle_p95_abs_deg": q(bank, 0.95),
        "speed_normalized_jerk_rms_1_s2": rms(normalized_jerk),
        "speed_normalized_jerk_p95_1_s2": q(normalized_jerk, 0.95),
        "snap_proxy_rms_mps4": rms(snap_norm),
        "snap_proxy_p95_mps4": q(snap_norm, 0.95),
        "jerk_high_frequency_energy_ratio": high_frequency_energy_ratio(jerk_norm, t, HF_CUTOFF_HZ),
    }


def _event_aware_join_metrics(raw_df: pd.DataFrame, render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    if "segment_id" not in render_df.columns or len(render_df) < 3:
        return {"normal_join_count": 0}
    seg = render_df["segment_id"].astype(str).to_numpy()
    boundary_idx = np.where(seg[1:] != seg[:-1])[0] + 1
    if boundary_idx.size == 0:
        return {"normal_join_count": 0, "normal_join_score_defaulted": True}
    raw_gaps = _raw_hard_gaps(raw_df)
    t = motion["t"]
    pos = motion["pos"]
    vel = motion["vel"]
    acc = motion["acc"]
    pos_closure: list[float] = []
    vel_jump: list[float] = []
    acc_jump: list[float] = []
    skipped_hard = 0
    for idx in boundary_idx:
        if idx <= 0 or idx >= len(t):
            continue
        a = float(t[idx - 1])
        b = float(t[idx])
        if not math.isfinite(a) or not math.isfinite(b):
            continue
        if b - a > LONG_GAP_THRESHOLD_S or _interval_overlaps_gap(a, b, raw_gaps):
            skipped_hard += 1
            continue
        dt = max(b - a, 1e-6)
        pred_step = 0.5 * (vel[idx - 1] + vel[idx]) * dt
        pos_closure.append(float(np.linalg.norm((pos[idx] - pos[idx - 1]) - pred_step)))
        vel_jump.append(float(np.linalg.norm(vel[idx] - vel[idx - 1])))
        acc_jump.append(float(np.linalg.norm(acc[idx] - acc[idx - 1])))
    return {
        "normal_join_count": int(len(pos_closure)),
        "hard_event_join_count_skipped": int(skipped_hard),
        "normal_join_position_closure_p95_m": q(np.asarray(pos_closure), 0.95),
        "normal_join_velocity_jump_p95_mps": q(np.asarray(vel_jump), 0.95),
        "normal_join_accel_jump_p95_mps2": q(np.asarray(acc_jump), 0.95),
    }


def _hard_gap_honesty_metrics(raw_df: pd.DataFrame, render_df: pd.DataFrame) -> dict[str, Any]:
    gaps = _raw_hard_gaps(raw_df)
    t = render_df["t"].to_numpy(float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(dt)) if dt.size else 1.0
    if not gaps:
        return {
            "hard_gap_count": 0,
            "hard_gap_render_sample_count": 0,
            "hard_gap_continuity_coverage_ratio": 0.0,
        }
    coverage_values: list[float] = []
    sample_count = 0
    for a, b in gaps:
        inner = (t > a) & (t < b)
        count = int(np.sum(inner))
        sample_count += count
        expected = max(1.0, (b - a) / max(median_dt, 1e-6) - 1.0)
        coverage_values.append(min(1.0, count / expected))
    return {
        "hard_gap_count": int(len(gaps)),
        "hard_gap_render_sample_count": int(sample_count),
        "hard_gap_continuity_coverage_ratio": float(np.mean(coverage_values)) if coverage_values else 0.0,
    }


def _dynamic_detail_metrics(raw_df: pd.DataFrame, render_df: pd.DataFrame, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = raw_df.dropna(subset=["t", "x", "y", "z"])
    if len(raw) < 8:
        return {}
    raw_t = raw["t"].to_numpy(float)
    raw_pos = raw[["x", "y", "z"]].to_numpy(float)
    raw_vel = gradient_vector(raw_pos, raw_t)
    render_vel = interpolate_columns(render_df, raw_t, ["vx", "vy", "vz"])
    mask = np.all(np.isfinite(raw_vel), axis=1) & np.all(np.isfinite(render_vel), axis=1)
    if np.sum(mask) < 8:
        return {}
    raw_speed = np.linalg.norm(raw_vel[mask, :2], axis=1)
    render_speed = np.linalg.norm(render_vel[mask, :2], axis=1)
    raw_vz = raw_vel[mask, 2]
    render_vz = render_vel[mask, 2]
    return {
        "speed_detail_ratio": robust_variation_ratio(render_speed, raw_speed),
        "vertical_detail_ratio": robust_variation_ratio(render_vz, raw_vz),
    }


def _raw_hard_gaps(raw_df: pd.DataFrame) -> list[tuple[float, float]]:
    raw_t = raw_df["t"].to_numpy(float)
    raw_t = raw_t[np.isfinite(raw_t)]
    raw_t.sort()
    if raw_t.size < 2:
        return []
    dt = np.diff(raw_t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive)) if positive.size else 1.0
    threshold = max(LONG_GAP_THRESHOLD_S, 5.0 * median_dt)
    return [(float(raw_t[i]), float(raw_t[i + 1])) for i in range(raw_t.size - 1) if raw_t[i + 1] - raw_t[i] > threshold]


def _interval_overlaps_gap(a: float, b: float, gaps: list[tuple[float, float]]) -> bool:
    lo = min(a, b)
    hi = max(a, b)
    for ga, gb in gaps:
        if hi > ga and lo < gb:
            return True
    return False


def _locality_scope_score(method: MethodPayload) -> float:
    method_id = method.method_id.lower()
    quality = method.payload.get("quality") if isinstance(method.payload.get("quality"), dict) else {}
    backend = str(quality.get("reconstruction_backend") or quality.get("fit_mode") or "").lower()
    segment_count = finite_float_or_none(quality.get("segment_count")) if quality else None
    if "kalman" in method_id and "boundary" not in method_id:
        return 25.0
    if "global" in method_id or "whole" in backend:
        return 30.0
    if "piecewise" in backend or "segment" in backend or (segment_count is not None and segment_count > 1):
        return 100.0
    if "v_spline" in method_id or "spline" in method_id:
        return 85.0
    return 55.0


def _locality_scope_note(method: MethodPayload) -> str:
    method_id = method.method_id.lower()
    if "kalman" in method_id and "boundary" not in method_id:
        return "whole-track Kalman/RTS baseline; locality score intentionally low for spline-centered trajectory-model ranking"
    if "v_spline" in method_id or "spline" in method_id:
        return "spline/local trajectory model; locality is part of the reference-free scientific score"
    return "generic reconstruction method"


def robust_variation_ratio(model_values: np.ndarray, evidence_values: np.ndarray) -> float | None:
    m = finite_values(np.diff(np.asarray(model_values, dtype=float)))
    e = finite_values(np.diff(np.asarray(evidence_values, dtype=float)))
    if m.size < 4 or e.size < 4:
        return None
    m_scale = robust_scale(m)
    e_scale = robust_scale(e)
    if e_scale is None or e_scale < 1e-9:
        return None
    return float((m_scale or 0.0) / e_scale)


def robust_scale(values: np.ndarray) -> float | None:
    arr = finite_values(values)
    if arr.size == 0:
        return None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad > 1e-12:
        return 1.4826 * mad
    return float(np.std(arr))


def score_lower_is_better(value: Any, good: float, poor: float, default: float | None = None) -> float | None:
    x = finite_float_or_none(value)
    if x is None:
        return default
    if poor <= good:
        return default
    z = (x - good) / (poor - good)
    return float(np.clip(TRAJECTORY_MODEL_MAX_SCORE * (1.0 - z), 0.0, TRAJECTORY_MODEL_MAX_SCORE))


def score_ratio_band(
    value: Any,
    *,
    low_good: float,
    high_good: float,
    low_bad: float,
    high_bad: float,
    default: float | None = None,
) -> float | None:
    x = finite_float_or_none(value)
    if x is None:
        return default
    if low_good <= x <= high_good:
        return TRAJECTORY_MODEL_MAX_SCORE
    if x < low_good:
        if low_good <= low_bad:
            return default
        return float(np.clip(TRAJECTORY_MODEL_MAX_SCORE * (x - low_bad) / (low_good - low_bad), 0.0, TRAJECTORY_MODEL_MAX_SCORE))
    if high_bad <= high_good:
        return default
    return float(np.clip(TRAJECTORY_MODEL_MAX_SCORE * (high_bad - x) / (high_bad - high_good), 0.0, TRAJECTORY_MODEL_MAX_SCORE))


def weighted_score(scores: dict[str, Any], weights: dict[str, float]) -> float | None:
    acc = 0.0
    wsum = 0.0
    for key, weight in weights.items():
        value = finite_float_or_none(scores.get(key))
        if value is None:
            continue
        acc += float(value) * float(weight)
        wsum += float(weight)
    if wsum <= 0:
        return None
    return float(acc / wsum)

def score_methods(metrics_by_method: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, dict[str, Any]]]:
    group_scores: dict[str, dict[str, Any]] = {m: {} for m in metrics_by_method}
    overall_accum: dict[str, float] = {m: 0.0 for m in metrics_by_method}
    overall_weight: dict[str, float] = {m: 0.0 for m in metrics_by_method}
    metric_winners: dict[str, dict[str, Any]] = {}

    for group, metric_names in LOWER_IS_BETTER_METRIC_GROUPS.items():
        weight = float(COMPARISON_GROUP_WEIGHTS.get(group, 1.0))
        group_accum = {m: 0.0 for m in metrics_by_method}
        group_count = {m: 0 for m in metrics_by_method}
        for metric in metric_names:
            values = {
                method: finite_float_or_none(metrics.get(metric))
                for method, metrics in metrics_by_method.items()
            }
            values = {method: value for method, value in values.items() if value is not None}
            if len(values) < 2 or not metric_has_signal(values):
                continue
            best_method, best_value = min(values.items(), key=lambda item: item[1])
            sorted_values = sorted(values.items(), key=lambda item: item[1])
            second = sorted_values[1][1] if len(sorted_values) > 1 else None
            metric_winners[metric] = {
                "group": group,
                "lower_is_better": True,
                "winner": best_method,
                "best_value": best_value,
                "second_value": second,
                "margin_abs": (second - best_value) if second is not None else None,
                "values": values,
            }
            scale_values = np.asarray([abs(v) for v in values.values() if math.isfinite(v)], dtype=float)
            scale_values = scale_values[scale_values > 1e-12]
            scale = float(np.median(scale_values)) if scale_values.size else 1.0
            for method, value in values.items():
                group_accum[method] += float(value) / scale
                group_count[method] += 1
        for method in metrics_by_method:
            if group_count[method] == 0:
                continue
            score = group_accum[method] / float(group_count[method])
            group_scores[method][group] = {
                "score_lower_is_better": score,
                "metric_count": group_count[method],
                "weight": weight,
            }
            overall_accum[method] += score * weight
            overall_weight[method] += weight

    overall_scores = {
        method: (overall_accum[method] / overall_weight[method] if overall_weight[method] > 0 else float("inf"))
        for method in metrics_by_method
    }
    return group_scores, overall_scores, metric_winners


def metric_has_signal(values: dict[str, float]) -> bool:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if len(finite) < 2:
        return False
    if all(abs(v) < 1e-12 for v in finite):
        return False
    return max(finite) - min(finite) > 1e-9




def trajectory_model_ranking(metrics_by_method: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[tuple[str, float]] = []
    for method_id, metrics in metrics_by_method.items():
        value = finite_float_or_none(metrics.get("trajectory_model_weighted_score_higher_is_better"))
        if value is not None:
            values.append((method_id, value))
    values.sort(key=lambda item: item[1], reverse=True)
    return [
        {"rank": rank + 1, "method_id": method_id, "trajectory_model_score_higher_is_better": score}
        for rank, (method_id, score) in enumerate(values)
    ]

def trajectory_model_report_rows(metrics_by_method: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id, metrics in metrics_by_method.items():
        row = {
            "method_id": method_id,
            "metric_family": metrics.get("trajectory_model_metric_family"),
            "truth_data_used": metrics.get("trajectory_model_truth_data_used"),
            "weighted_score_higher_is_better": metrics.get("trajectory_model_weighted_score_higher_is_better"),
            "score_loss_lower_is_better": metrics.get("trajectory_model_score_loss_lower_is_better"),
            "raw_position_evidence_rmse_m": metrics.get("trajectory_model_raw_position_evidence_rmse_m"),
            "raw_velocity_evidence_rmse_mps": metrics.get("trajectory_model_raw_velocity_evidence_rmse_mps"),
            "fd_velocity_evidence_rmse_mps": metrics.get("trajectory_model_fd_velocity_evidence_rmse_mps"),
            "hard_gap_coverage_ratio": metrics.get("trajectory_model_hard_gap_continuity_coverage_ratio"),
            "normal_join_count": metrics.get("trajectory_model_normal_join_count"),
            "locality_note": metrics.get("trajectory_model_locality_scope_note"),
        }
        components = metrics.get("trajectory_model_component_scores")
        if isinstance(components, dict):
            for key, value in components.items():
                row[key] = value
        rows.append(clean_json(row))
    return rows

def evaluate_dataset(
    *,
    output_dir: Path,
    flight_entries: list[dict[str, Any]],
    method_filter: Iterable[str] | None = None,
    weight_metric: str = "raw_sample_count",
    require_all_methods: bool = False,
    skip_flight_errors: bool = False,
    dataset_output_dir: Path | None = None,
) -> DatasetEvaluationResult:
    """Evaluate multiple flights and aggregate metrics by method.

    Each flight is evaluated with the same per-flight machinery as the original
    single-flight mode, so existing debug artifacts and CSVs remain comparable.
    Dataset metrics are then computed from the per-flight method metrics using
    a configurable flight weight.  RMSE-like values are aggregated as weighted
    RMS of the per-flight RMSEs, count-like values are summed, and other numeric
    values are weighted averages.
    """
    evaluation_dir = dataset_output_dir or (output_dir / "dataset_evaluation")
    evaluation_dir = evaluation_dir if evaluation_dir.is_absolute() else output_dir / evaluation_dir
    flight_results: list[EvaluationResult] = []
    flight_errors: list[dict[str, Any]] = []

    for index, flight_entry in enumerate(flight_entries, start=1):
        flight_id = str(flight_entry.get("flightId") or "")
        icao = str(flight_entry.get("icao") or "")
        try:
            result = evaluate_flight(output_dir=output_dir, flight_entry=flight_entry, method_filter=method_filter)
            write_reports(result)
            flight_results.append(result)
            print(f"[{index}/{len(flight_entries)}] evaluated {flight_id or '<missing flightId>'} ({icao})")
        except Exception as exc:
            error = {"flight_id": flight_id, "icao": icao, "error": repr(exc)}
            flight_errors.append(error)
            if not skip_flight_errors:
                raise
            print(f"[{index}/{len(flight_entries)}] skipped {flight_id or '<missing flightId>'} ({icao}): {exc}")

    if not flight_results:
        raise ValueError("No flights were successfully evaluated")

    metrics_by_method, method_coverage, per_flight_rows = aggregate_dataset_metrics(
        flight_results=flight_results,
        weight_metric=weight_metric,
        require_all_methods=require_all_methods,
    )
    group_scores, overall_scores, metric_winners = score_methods(metrics_by_method) if metrics_by_method else ({}, {}, {})
    ranked_methods = [
        {"method_id": method_id, "overall_score_lower_is_better": score, "rank": rank + 1}
        for rank, (method_id, score) in enumerate(sorted(overall_scores.items(), key=lambda item: item[1]))
    ]
    return DatasetEvaluationResult(
        output_dir=output_dir,
        evaluation_dir=evaluation_dir,
        flight_results=flight_results,
        flight_errors=flight_errors,
        selected_flight_count=len(flight_entries),
        weight_metric=weight_metric,
        require_all_methods=require_all_methods,
        metrics_by_method=metrics_by_method,
        group_scores=group_scores,
        overall_scores=overall_scores,
        metric_winners=metric_winners,
        ranked_methods=ranked_methods,
        method_coverage=method_coverage,
        per_flight_rows=per_flight_rows,
    )


def aggregate_dataset_metrics(
    *,
    flight_results: list[EvaluationResult],
    weight_metric: str,
    require_all_methods: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    total_flights = len(flight_results)
    method_rows: dict[str, list[tuple[EvaluationResult, float, dict[str, Any]]]] = {}
    per_flight_rows: list[dict[str, Any]] = []

    for result in flight_results:
        rank_by_method = {item["method_id"]: item for item in result.ranked_methods}
        for method_id, metrics in result.metrics_by_method.items():
            weight = dataset_metric_weight(metrics, weight_metric)
            method_rows.setdefault(method_id, []).append((result, weight, metrics))
            row = {
                "flight_id": result.flight_id,
                "icao": result.icao,
                "method_id": method_id,
                "dataset_weight": weight,
                "dataset_weight_metric": weight_metric,
                "flight_rank": rank_by_method.get(method_id, {}).get("rank"),
                "flight_overall_score_lower_is_better": rank_by_method.get(method_id, {}).get("overall_score_lower_is_better"),
            }
            row.update(metrics)
            per_flight_rows.append(clean_json(row))

    if require_all_methods:
        method_rows = {method_id: rows for method_id, rows in method_rows.items() if len(rows) == total_flights}

    aggregated_by_method: dict[str, dict[str, Any]] = {}
    coverage_by_method: dict[str, dict[str, Any]] = {}
    for method_id, rows in sorted(method_rows.items()):
        flight_ids = [r.flight_id for r, _, _ in rows]
        icaos = [r.icao for r, _, _ in rows]
        weights = [max(float(w), 0.0) for _, w, _ in rows]
        total_weight = float(sum(weights))
        coverage = {
            "method_id": method_id,
            "flight_count": len(rows),
            "selected_flight_count": total_flights,
            "coverage_fraction": len(rows) / float(total_flights) if total_flights else None,
            "total_weight": total_weight,
            "weight_metric": weight_metric,
            "flight_ids": flight_ids,
            "icaos": icaos,
        }
        coverage_by_method[method_id] = clean_json(coverage)
        metric_keys = sorted({key for _, _, metrics in rows for key in metrics.keys()})
        aggregate: dict[str, Any] = {
            "method_id": method_id,
            "dataset_metric_scope": "weighted_across_flights",
            "dataset_weight_metric": weight_metric,
            "dataset_flight_count": len(rows),
            "dataset_selected_flight_count": total_flights,
            "dataset_coverage_fraction": coverage["coverage_fraction"],
            "dataset_total_weight": total_weight,
        }
        for key in metric_keys:
            if key in {"method_id", "label", "detailed_file"}:
                continue
            values: list[float] = []
            value_weights: list[float] = []
            for _, weight, metrics in rows:
                value = finite_float_or_none(metrics.get(key))
                if value is None:
                    continue
                values.append(value)
                value_weights.append(max(float(weight), 0.0))
            if not values:
                common = common_non_numeric_value([metrics.get(key) for _, _, metrics in rows])
                if common is not None:
                    aggregate[key] = common
                continue
            aggregate[key] = aggregate_metric_values(key, values, value_weights)
        aggregated_by_method[method_id] = clean_json(aggregate)
    return aggregated_by_method, coverage_by_method, per_flight_rows


def dataset_metric_weight(metrics: dict[str, Any], weight_metric: str) -> float:
    if weight_metric == "uniform":
        return 1.0
    value = finite_float_or_none(metrics.get(weight_metric))
    if value is None or value <= 0:
        value = finite_float_or_none(metrics.get("raw_sample_count"))
    if value is None or value <= 0:
        value = finite_float_or_none(metrics.get("duration_s"))
    if value is None or value <= 0:
        return 1.0
    return float(value)


def aggregate_metric_values(key: str, values: list[float], weights: list[float]) -> float | int | None:
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w) & (w >= 0)
    arr = arr[mask]
    w = w[mask]
    if arr.size == 0:
        return None
    if np.sum(w) <= 0:
        w = np.ones_like(arr)
    if is_count_metric_name(key):
        total = float(np.sum(arr))
        return int(total) if abs(total - round(total)) < 1e-9 else total
    if is_rmse_metric_name(key):
        return float(np.sqrt(np.average(arr * arr, weights=w)))
    return float(np.average(arr, weights=w))


def is_count_metric_name(key: str) -> bool:
    key_l = key.lower()
    return (
        key_l.endswith("_count")
        or key_l.endswith("_sample_count")
        or key_l.endswith("_violation_count")
        or ("envelope_violation" in key_l and key_l.endswith("count"))
    )


def is_rmse_metric_name(key: str) -> bool:
    key_l = key.lower()
    return key_l.endswith("_rmse") or "_rmse_" in key_l


def common_non_numeric_value(values: list[Any]) -> Any | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    first = present[0]
    if all(v == first for v in present):
        return first
    return None


def write_dataset_reports(result: DatasetEvaluationResult) -> None:
    result.evaluation_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(result.output_dir),
        "evaluation_dir": str(result.evaluation_dir),
        "selected_flight_count": result.selected_flight_count,
        "evaluated_flight_count": len(result.flight_results),
        "failed_flight_count": len(result.flight_errors),
        "weight_metric": result.weight_metric,
        "require_all_methods": result.require_all_methods,
        "ranked_methods": result.ranked_methods,
        "trajectory_model_ranking_higher_is_better": trajectory_model_ranking(result.metrics_by_method),
        "overall_scores_lower_is_better": result.overall_scores,
        "group_scores": result.group_scores,
        "metric_winners": result.metric_winners,
        "method_coverage": result.method_coverage,
        "flight_errors": result.flight_errors,
        "aggregation_note": "Dataset metrics aggregate per-flight method metrics. RMSE-like metrics are weighted RMS, count-like metrics are summed, and other numeric metrics are weighted averages using the configured weight_metric.",
    }
    write_json(result.evaluation_dir / "dataset_evaluation_summary.json", summary)
    metrics_df = pd.DataFrame.from_dict(result.metrics_by_method, orient="index")
    if not metrics_df.empty:
        metrics_df.insert(0, "method_key", metrics_df.index.astype(str))
    metrics_df.to_csv(result.evaluation_dir / "dataset_weighted_metrics.csv", index=False)
    pd.DataFrame(result.method_coverage.values()).to_csv(result.evaluation_dir / "dataset_method_coverage.csv", index=False)
    pd.DataFrame(result.per_flight_rows).to_csv(result.evaluation_dir / "dataset_method_flight_metrics.csv", index=False)
    pd.DataFrame(result.flight_errors).to_csv(result.evaluation_dir / "dataset_flight_errors.csv", index=False)
    flight_ranking_rows: list[dict[str, Any]] = []
    for flight_result in result.flight_results:
        for item in flight_result.ranked_methods:
            flight_ranking_rows.append({"flight_id": flight_result.flight_id, "icao": flight_result.icao, **item})
    pd.DataFrame(flight_ranking_rows).to_csv(result.evaluation_dir / "dataset_flight_rankings.csv", index=False)
    group_rows: list[dict[str, Any]] = []
    for method, groups in result.group_scores.items():
        for group, data in groups.items():
            group_rows.append({"method_id": method, "group": group, **data})
    pd.DataFrame(group_rows).to_csv(result.evaluation_dir / "dataset_group_scores.csv", index=False)
    trajectory_rows = trajectory_model_report_rows(result.metrics_by_method)
    pd.DataFrame(trajectory_rows).to_csv(result.evaluation_dir / "dataset_trajectory_model_metrics.csv", index=False)
    (result.evaluation_dir / "dataset_evaluation_report.md").write_text(render_dataset_markdown_report(result), encoding="utf-8")


def render_dataset_markdown_report(result: DatasetEvaluationResult) -> str:
    lines = [
        "# Dataset reconstruction evaluation",
        "",
        f"Evaluated flights: `{len(result.flight_results)}` / selected `{result.selected_flight_count}`",
        f"Weight metric: `{result.weight_metric}`",
        f"Require all methods: `{result.require_all_methods}`",
        "",
        "## Dataset overall ranking",
        "",
        "Lower is better. Scores are computed from weighted dataset-level metrics.",
        "",
        "| Rank | Method | Overall score ↓ | Flights | Coverage | Total weight |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in result.ranked_methods:
        method_id = item["method_id"]
        coverage = result.method_coverage.get(method_id, {})
        lines.append(
            f"| {item['rank']} | `{method_id}` | {item['overall_score_lower_is_better']:.4g} | "
            f"{coverage.get('flight_count', '')} | {format_float(coverage.get('coverage_fraction'))} | {format_float(coverage.get('total_weight'))} |"
        )
    lines.extend([
        "",
        "## Reference-free trajectory-model ranking",
        "",
        "Higher is better. These scores are weighted across evaluated flights.",
        "",
        "| Rank | Method | Trajectory-model score ↑ |",
        "|---:|---|---:|",
    ])
    for item in trajectory_model_ranking(result.metrics_by_method):
        lines.append(f"| {item['rank']} | `{item['method_id']}` | {item['trajectory_model_score_higher_is_better']:.4g} |")
    lines.extend(["", "## Group scores", "", "Lower is better. Blank means the group had no comparable signal.", ""])
    groups = list(LOWER_IS_BETTER_METRIC_GROUPS.keys())
    lines.append("| Method | " + " | ".join(groups) + " |")
    lines.append("|---|" + "---:|" * len(groups))
    for method in result.metrics_by_method:
        cells = []
        for group in groups:
            data = result.group_scores.get(method, {}).get(group)
            cells.append(f"{data['score_lower_is_better']:.4g}" if data else "")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    if result.flight_errors:
        lines.extend(["", "## Flight errors", "", "| Flight | ICAO | Error |", "|---|---|---|"])
        for error in result.flight_errors:
            lines.append(f"| `{error.get('flight_id')}` | `{error.get('icao')}` | `{error.get('error')}` |")
    lines.extend([
        "",
        "Generated files:",
        "",
        "- `dataset_evaluation_summary.json`",
        "- `dataset_weighted_metrics.csv`",
        "- `dataset_group_scores.csv`",
        "- `dataset_method_coverage.csv`",
        "- `dataset_method_flight_metrics.csv`",
        "- `dataset_flight_rankings.csv`",
        "- `dataset_trajectory_model_metrics.csv`",
        "- `dataset_flight_errors.csv`",
    ])
    return "\n".join(lines) + "\n"


def print_dataset_summary(result: DatasetEvaluationResult) -> None:
    print(f"Evaluated {len(result.flight_results)} / {result.selected_flight_count} selected flight(s)")
    print(f"Dataset reports written to: {result.evaluation_dir}")
    if result.flight_errors:
        print(f"Skipped/failed flights: {len(result.flight_errors)}")
    print(f"\nDataset overall ranking, lower is better (weight={result.weight_metric}):")
    for item in result.ranked_methods:
        method_id = item["method_id"]
        coverage = result.method_coverage.get(method_id, {})
        print(
            f"  {item['rank']:>2}. {method_id:<45} {item['overall_score_lower_is_better']:.6g} "
            f"flights={coverage.get('flight_count', 0)}/{coverage.get('selected_flight_count', 0)}"
        )


def write_reports(result: EvaluationResult) -> None:
    result.evaluation_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "flight_id": result.flight_id,
        "icao": result.icao,
        "output_dir": str(result.output_dir),
        "evaluation_dir": str(result.evaluation_dir),
        "ranked_methods": result.ranked_methods,
        "trajectory_model_ranking_higher_is_better": trajectory_model_ranking(result.metrics_by_method),
        "overall_scores_lower_is_better": result.overall_scores,
        "group_scores": result.group_scores,
        "metric_winners": result.metric_winners,
        "metrics_by_method": result.metrics_by_method,
        "scoring_note": "Most comparison groups are lower-is-better. The trajectory-model group uses 100 - reference-free trajectory_model_weighted_score_higher_is_better as its lower-is-better loss. Use group metrics and trajectory_model_metrics.csv to interpret why a method wins.",
    }
    write_json(result.evaluation_dir / "evaluation_summary.json", summary)
    metrics_df = pd.DataFrame.from_dict(result.metrics_by_method, orient="index")
    metrics_df.insert(0, "method_key", metrics_df.index.astype(str))
    metrics_df.to_csv(result.evaluation_dir / "evaluation_metrics.csv", index=False)
    trajectory_rows = trajectory_model_report_rows(result.metrics_by_method)
    pd.DataFrame(trajectory_rows).to_csv(result.evaluation_dir / "trajectory_model_metrics.csv", index=False)
    group_rows: list[dict[str, Any]] = []
    for method, groups in result.group_scores.items():
        for group, data in groups.items():
            group_rows.append({"method_id": method, "group": group, **data})
    pd.DataFrame(group_rows).to_csv(result.evaluation_dir / "evaluation_group_scores.csv", index=False)
    (result.evaluation_dir / "evaluation_report.md").write_text(render_markdown_report(result), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def render_markdown_report(result: EvaluationResult) -> str:
    lines = [
        f"# Reconstruction evaluation: {result.flight_id}",
        "",
        f"ICAO: `{result.icao}`",
        "",
        "## Overall ranking",
        "",
        "| Rank | Method | Overall score ↓ |",
        "|---:|---|---:|",
    ]
    for item in result.ranked_methods:
        lines.append(f"| {item['rank']} | `{item['method_id']}` | {item['overall_score_lower_is_better']:.4g} |")
    lines.extend(["", "## Reference-free trajectory-model ranking", "", "Higher is better. These scores use raw ADS-B only as noisy evidence, not as truth.", "", "| Rank | Method | Trajectory-model score ↑ |", "|---:|---|---:|"])
    for item in trajectory_model_ranking(result.metrics_by_method):
        lines.append(f"| {item['rank']} | `{item['method_id']}` | {item['trajectory_model_score_higher_is_better']:.4g} |")
    lines.extend(["", "## Group scores", "", "Lower is better. Blank means the group had no comparable signal for that method.", ""])
    groups = list(LOWER_IS_BETTER_METRIC_GROUPS.keys())
    lines.append("| Method | " + " | ".join(groups) + " |")
    lines.append("|---|" + "---:|" * len(groups))
    for method in result.metrics_by_method:
        cells = []
        for group in groups:
            data = result.group_scores.get(method, {}).get(group)
            cells.append(f"{data['score_lower_is_better']:.4g}" if data else "")
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    lines.extend(["", "## Metric winners", "", "| Metric | Group | Winner | Best value |", "|---|---|---|---:|"])
    for metric, info in sorted(result.metric_winners.items(), key=lambda item: (item[1].get("group", ""), item[0])):
        lines.append(f"| `{metric}` | {info.get('group')} | `{info.get('winner')}` | {format_float(info.get('best_value'))} |")
    lines.extend(["", "Generated files:", "", "- `evaluation_summary.json`", "- `evaluation_metrics.csv`", "- `evaluation_group_scores.csv`", "- `trajectory_model_metrics.csv`"])
    return "\n".join(lines) + "\n"


def print_summary(result: EvaluationResult) -> None:
    print(f"Evaluated flight {result.flight_id} ({result.icao})")
    print(f"Reports written to: {result.evaluation_dir}")
    print("\nOverall ranking, lower is better:")
    for item in result.ranked_methods:
        print(f"  {item['rank']:>2}. {item['method_id']:<35} {item['overall_score_lower_is_better']:.6g}")


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def vec_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.linalg.norm(arr, axis=1)


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def rms(values: np.ndarray) -> float | None:
    arr = finite_values(values)
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr * arr)))


def q(values: np.ndarray, quantile: float) -> float | None:
    arr = finite_values(values)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, quantile))


def max_or_none(values: np.ndarray) -> float | None:
    arr = finite_values(values)
    if arr.size == 0:
        return None
    return float(np.max(arr))


def min_or_none(values: np.ndarray) -> float | None:
    arr = finite_values(values)
    if arr.size == 0:
        return None
    return float(np.min(arr))


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or not math.isfinite(num) or not math.isfinite(den):
        return None
    if abs(den) < 1e-12:
        return None if abs(num) > 1e-12 else 1.0
    return float(num / den)


def count_gt(values: np.ndarray, threshold: float) -> int:
    arr = finite_values(values)
    return int(np.sum(arr > float(threshold))) if arr.size else 0


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def format_float(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.4g}"


if __name__ == "__main__":
    main()
