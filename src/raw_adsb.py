"""
Raw ADS-B event/keyframe normalization.

This layer is deliberately outside the V-Spline core. It converts raw decoder
rows into a traceable intermediate representation:

    raw SQL rows
      -> prepared raw events with event ids
      -> time-bucketed raw keyframes with source row ids
      -> flat aviation dataframe for later V-Spline preprocessing

The flat dataframe is still observed ADS-B data. It is not reconstructed,
smoothed, or interpolated. Horizontal velocity is computed from reported ground
speed and track. Vertical velocity uses the ADS-B vertical-rate fields provided
by the SQL loader when available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import copy
import json
import math

import numpy as np
import pandas as pd
from loguru import logger

from geo_utils import FT_TO_M, KNOT_TO_MPS
from track_rules import TrackRuleConfig

TWO_PI = 2.0 * math.pi
TrackUnit = Literal["deg", "rad"]

# Riga International Airport / RIX / EVRA reference point used by
# boundary_rix outlier filtering and distance diagnostics.
RIX_LAT_DEG = 56.923611
RIX_LON_DEG = 23.971111


@dataclass(frozen=True)
class RawAdsbColumnConfig:
    time_column: str = "ts_utc"
    icao_column: str = "icao"
    lat_column: str = "lat"
    lon_column: str = "lon"
    altitude_ft_column: str = "alt"
    ground_speed_kt_column: str = "gs"
    track_column: str = "track"
    crc_ok_column: str = "crc_ok"
    raw_id_column: str = "raw_id"
    row_id_column: str | None = None
    decoded_json_column: str = "decoded_json"
    event_kind_column: str = "event_kind"
    vertical_rate_fpm_column: str | None = "vertical_rate_fpm"
    vertical_rate_mps_column: str | None = "vertical_rate_mps"
    seen_utc_column: str | None = "seen_utc"


@dataclass(frozen=True)
class RawAdsbNormalizeConfig:
    columns: RawAdsbColumnConfig = field(default_factory=RawAdsbColumnConfig)
    apply_rule_filters: bool = True
    keep_unpaired_timestamps: bool = True
    parse_vertical_rate_from_decoded_json: bool = True
    keyframe_time_quantization_s: float = 1.0
    track_unit: TrackUnit = "deg"
    baro_z_reference: Literal["field", "msl"] = "field"
    clip_height_to_zero_for_display: bool = False
    derive_velocity_delta_acceleration: bool = True
    acceleration_max_dt_s: float | None = None

    def __post_init__(self) -> None:
        if self.keyframe_time_quantization_s <= 0:
            raise ValueError("keyframe_time_quantization_s must be > 0")
        if self.track_unit not in {"deg", "rad"}:
            raise ValueError("track_unit must be 'deg' or 'rad'")
        if self.baro_z_reference not in {"field", "msl"}:
            raise ValueError("baro_z_reference must be 'field' or 'msl'")
        if self.acceleration_max_dt_s is not None and self.acceleration_max_dt_s <= 0:
            raise ValueError("acceleration_max_dt_s must be None or > 0")


@dataclass
class RawAdsbNormalizeResult:
    """Output of raw ADS-B normalization.

    ``dataframe`` is the flat V-Spline-ready aviation table. ``events`` and
    ``keyframes`` keep the richer traceable event/keyframe representation for
    debugging and future viewer export.
    """

    dataframe: pd.DataFrame
    report: dict[str, Any]
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    keyframes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RawAdsbJsonSplitConfig:
    """Configuration for writing raw ADS-B viewer JSON.

    The historical viewer packet stored every raw event inline under
    ``raw_events``.  For large tracks that makes the primary viewer payload
    unnecessarily heavy because the renderer normally needs ``raw_keyframes``
    and ``render_keyframes`` first.  This config keeps the non-event packet
    byte-for-byte equivalent at the data level while moving raw events to a
    sibling JSON document.
    """

    events_key: str = "raw_events"
    events_ref_key: str = "raw_events_file"
    events_count_key: str = "raw_event_count"
    include_events_ref_in_main: bool = True
    include_events_metadata: bool = True
    indent: int = 2
    ensure_ascii: bool = False


def default_raw_adsb_events_path(path: str | Path) -> Path:
    """Return the default sibling path for split raw ADS-B events.

    Example: ``raw_adsb.json`` -> ``raw_adsb_events.json``.
    """

    main_path = Path(path)
    suffix = main_path.suffix or ".json"
    return main_path.with_name(f"{main_path.stem}_events{suffix}")


def split_raw_adsb_payload(
    payload: dict[str, Any],
    *,
    events_filename: str | None = None,
    config: RawAdsbJsonSplitConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a raw ADS-B viewer payload into main JSON and event JSON.

    The returned main payload is a deep copy of ``payload`` with ``raw_events``
    removed.  The returned event payload stores those same events under
    ``raw_events`` and, by default, includes only lightweight metadata copied
    from the main packet.
    """

    cfg = config or RawAdsbJsonSplitConfig()
    main_payload = _clean_json(copy.deepcopy(payload))
    raw_events = main_payload.pop(cfg.events_key, [])
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list):
        raise ValueError(f"{cfg.events_key!r} must be a list when splitting raw ADS-B JSON")

    if cfg.include_events_ref_in_main:
        if events_filename is not None:
            main_payload[cfg.events_ref_key] = str(events_filename)
        main_payload[cfg.events_count_key] = int(len(raw_events))

    if cfg.include_events_metadata:
        events_payload: dict[str, Any] = {
            "schema_version": main_payload.get("schema_version", payload.get("schema_version")),
            "track_id": main_payload.get("track_id", payload.get("track_id")),
            "icao": main_payload.get("icao", payload.get("icao")),
            cfg.events_count_key: int(len(raw_events)),
            cfg.events_key: raw_events,
        }
    else:
        events_payload = {cfg.events_key: raw_events}

    return _clean_json(main_payload), _clean_json(events_payload)


def write_json_packet(packet: dict[str, Any], path: str | Path, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Write a JSON packet using the formatting convention used by this package."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_json(packet), f, indent=indent, ensure_ascii=ensure_ascii)
        f.write("\n")


def write_raw_adsb_split_json(
    payload: dict[str, Any],
    path: str | Path,
    *,
    events_path: str | Path | None = None,
    config: RawAdsbJsonSplitConfig | None = None,
) -> tuple[Path, Path]:
    """Write raw ADS-B viewer JSON with raw events in a sibling file.

    Parameters
    ----------
    payload:
        Full raw ADS-B payload, including a ``raw_events`` list.
    path:
        Destination for the main viewer payload.  This file will not contain
        inline ``raw_events``.
    events_path:
        Destination for the event payload.  Defaults to
        ``<path stem>_events.json``.

    Returns
    -------
    (main_path, events_path)
    """

    cfg = config or RawAdsbJsonSplitConfig()
    main_path = Path(path)
    sidecar_path = Path(events_path) if events_path is not None else default_raw_adsb_events_path(main_path)
    events_filename = sidecar_path.name if sidecar_path.parent == main_path.parent else str(sidecar_path)
    main_payload, events_payload = split_raw_adsb_payload(payload, events_filename=events_filename, config=cfg)
    write_json_packet(main_payload, main_path, indent=cfg.indent, ensure_ascii=cfg.ensure_ascii)
    write_json_packet(events_payload, sidecar_path, indent=cfg.indent, ensure_ascii=cfg.ensure_ascii)
    return main_path, sidecar_path


def rewrite_raw_adsb_json_with_split_events(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    events_path: str | Path | None = None,
    config: RawAdsbJsonSplitConfig | None = None,
) -> tuple[Path, Path]:
    """Rewrite an existing monolithic raw ADS-B JSON file into split files.

    This is useful for migrating an already-produced ``raw_adsb.json`` fixture:
    the main output preserves all non-event sections and the sidecar receives
    the original ``raw_events`` list.
    """

    src = Path(source_path)
    dst = Path(output_path) if output_path is not None else src
    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return write_raw_adsb_split_json(payload, dst, events_path=events_path, config=config)


def _numeric_column(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column and column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    try:
        return bool(result)
    except Exception:
        return False


def _json_number(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if _is_missing(obj):
        return None
    return obj


def _event_id(n: int) -> str:
    return f"evt_{n:06d}"


def _keyframe_id(n: int) -> str:
    return f"kf_{n:06d}"


def _extract_vertical_rate_fpm_from_json(value: Any) -> float | None:
    """Extract vertical rate from decoded_json.velocity_raw when present.

    The observed ADS-B decoder payload often stores TC=19 velocity as
    ``[ground_speed_kt, track_deg, vertical_rate_fpm, source]``.
    """
    if _is_missing(value):
        return None
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("velocity_raw")
    if isinstance(raw, list) and len(raw) >= 3:
        try:
            return float(raw[2])
        except Exception:
            return None
    return None


def _normalize_angle_rad(angle_rad: float) -> float:
    return ((angle_rad % TWO_PI) + TWO_PI) % TWO_PI


def _track_to_rad(track: float | None, track_unit: TrackUnit) -> float | None:
    if track is None:
        return None
    track = float(track)
    if track_unit == "deg":
        return math.radians(track % 360.0)
    if track_unit == "rad":
        if abs(track) > TWO_PI + 1e-6:
            raise ValueError(
                f"track_unit='rad' but track={track} looks like degrees. "
                "ADS-B / pyModeS track is usually degrees; use track_unit='deg'."
            )
        return _normalize_angle_rad(track)
    raise ValueError("track_unit must be 'deg' or 'rad'")


def _track_from_velocity_components(east_mps: float | None, north_mps: float | None, track_unit: TrackUnit) -> float | None:
    if east_mps is None or north_mps is None:
        return None
    if abs(east_mps) < 1e-15 and abs(north_mps) < 1e-15:
        return None
    track_rad = _normalize_angle_rad(math.atan2(east_mps, north_mps))
    return math.degrees(track_rad) if track_unit == "deg" else track_rad


def _velocity_components_from_gs_track(gs_kt: float | None, track: float | None, track_unit: TrackUnit) -> tuple[float | None, float | None]:
    """Compute horizontal ADS-B velocity components from GS+track."""
    if gs_kt is None or track is None:
        return None, None
    chi = _track_to_rad(track, track_unit)
    if chi is None:
        return None, None
    speed_mps = float(gs_kt) * KNOT_TO_MPS
    return speed_mps * math.sin(chi), speed_mps * math.cos(chi)


def _mean_numeric(group: pd.DataFrame, column: str) -> float | None:
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _unique_strings(group: pd.DataFrame, column: str) -> list[str]:
    if column not in group.columns:
        return []
    out: list[str] = []
    for value in group[column].tolist():
        if _is_missing(value):
            continue
        text = str(value)
        if text not in out:
            out.append(text)
    return out


def _mean_vector(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _haversine_km(
    lat1: float | pd.Series,
    lon1: float | pd.Series,
    lat2: float | pd.Series,
    lon2: float | pd.Series,
) -> float | pd.Series:
    """Great-circle distance in kilometers for scalar or pandas-Series inputs."""
    earth_radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c


def _iqr_upper_threshold(series: pd.Series, multiplier: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if len(clean) < 4:
        return None

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr <= 0:
        return None

    return float(q3 + float(multiplier) * iqr)


def _count_outliers(mask: pd.Series, denominator: int) -> dict[str, Any]:
    count = int(mask.fillna(False).astype(bool).sum())
    return {
        "count": count,
        "denominator": int(denominator),
        "fraction": float(count / denominator) if denominator else None,
    }


class RawAdsbEventPreparer:
    """Validate, sort, enrich, and optionally rule-filter raw ADS-B rows."""

    def __init__(self, config: RawAdsbNormalizeConfig) -> None:
        self.config = config

    def _add_diagnostics(self, work: pd.DataFrame) -> pd.DataFrame:
        """Add raw-row diagnostics used by outlier detection and reports."""
        out = work.copy()

        for column in (
            "distance_from_rix_km",
            "pos_prev_ts_utc",
            "pos_dt_s",
            "pos_step_km",
            "pos_speed_kmh",
            "vel_prev_ts_utc",
            "vel_dt_s",
            "vel_delta_mps",
            "vel_horizontal_delta_mps",
            "vel_vertical_delta_mps",
            "vel_accel_mps2",
            "vel_horizontal_accel_mps2",
            "vel_vertical_accel_mps2",
        ):
            out[column] = np.nan

        position_mask = out["_has_position_xy"].fillna(False).astype(bool)
        if position_mask.any():
            pos = out.loc[position_mask, ["_time_s", "_lat_deg", "_lon_deg"]].copy()
            out.loc[position_mask, "distance_from_rix_km"] = _haversine_km(
                RIX_LAT_DEG,
                RIX_LON_DEG,
                pos["_lat_deg"],
                pos["_lon_deg"],
            )

            prev_time = pos["_time_s"].shift(1)
            prev_lat = pos["_lat_deg"].shift(1)
            prev_lon = pos["_lon_deg"].shift(1)
            dt_s = pos["_time_s"] - prev_time
            step_km = _haversine_km(prev_lat, prev_lon, pos["_lat_deg"], pos["_lon_deg"])
            speed_kmh = np.where(dt_s > 0, step_km / (dt_s / 3600.0), np.nan)

            out.loc[position_mask, "pos_prev_ts_utc"] = prev_time.to_numpy()
            out.loc[position_mask, "pos_dt_s"] = dt_s.to_numpy()
            out.loc[position_mask, "pos_step_km"] = np.asarray(step_km)
            out.loc[position_mask, "pos_speed_kmh"] = speed_kmh

        velocity_mask = (
            out["_has_horizontal_velocity"].fillna(False).astype(bool)
            | out["_has_vertical_velocity"].fillna(False).astype(bool)
        )
        if velocity_mask.any():
            vel = out.loc[velocity_mask, ["_time_s", "_east_mps", "_north_mps", "_up_mps"]].copy()
            prev_time = vel["_time_s"].shift(1)
            prev_east = vel["_east_mps"].shift(1)
            prev_north = vel["_north_mps"].shift(1)
            prev_up = vel["_up_mps"].shift(1)
            dt_s = vel["_time_s"] - prev_time

            horizontal_delta_mps = np.sqrt(
                (vel["_east_mps"] - prev_east) ** 2
                + (vel["_north_mps"] - prev_north) ** 2
            )
            vertical_delta_mps = (vel["_up_mps"] - prev_up).abs()
            delta_mps = np.sqrt(
                horizontal_delta_mps.fillna(0.0) ** 2
                + vertical_delta_mps.fillna(0.0) ** 2
            )
            delta_mps = delta_mps.where(horizontal_delta_mps.notna() | vertical_delta_mps.notna())

            horizontal_accel_mps2 = np.where(dt_s > 0, horizontal_delta_mps / dt_s, np.nan)
            vertical_accel_mps2 = np.where(dt_s > 0, vertical_delta_mps / dt_s, np.nan)
            accel_mps2 = np.where(dt_s > 0, delta_mps / dt_s, np.nan)

            out.loc[velocity_mask, "vel_prev_ts_utc"] = prev_time.to_numpy()
            out.loc[velocity_mask, "vel_dt_s"] = dt_s.to_numpy()
            out.loc[velocity_mask, "vel_delta_mps"] = delta_mps.to_numpy()
            out.loc[velocity_mask, "vel_horizontal_delta_mps"] = horizontal_delta_mps.to_numpy()
            out.loc[velocity_mask, "vel_vertical_delta_mps"] = vertical_delta_mps.to_numpy()
            out.loc[velocity_mask, "vel_accel_mps2"] = accel_mps2
            out.loc[velocity_mask, "vel_horizontal_accel_mps2"] = horizontal_accel_mps2
            out.loc[velocity_mask, "vel_vertical_accel_mps2"] = vertical_accel_mps2

        return out

    def _remove_rule_outliers(
        self,
        work: pd.DataFrame,
        *,
        rule: TrackRuleConfig | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply boundary_rix and IQR outlier removal before event IDs/keyframes."""
        if rule is None:
            return work, {"enabled": False, "reason": "no_track_rule"}

        max_distance_from_rix_km = (
            rule.boundary_rix.distance_km
            if getattr(rule, "boundary_rix", None) is not None
            else None
        )
        iqr_multiplier = (
            rule.outlier_multiplier.multiplier
            if getattr(rule, "outlier_multiplier", None) is not None
            else None
        )

        if max_distance_from_rix_km is None and iqr_multiplier is None:
            return work, {"enabled": False, "reason": "rule_has_no_outlier_fields"}

        position_count = int(work["_has_position"].fillna(False).astype(bool).sum())
        velocity_count = int(
            (
                work["_has_horizontal_velocity"].fillna(False).astype(bool)
                | work["_has_vertical_velocity"].fillna(False).astype(bool)
            ).sum()
        )

        if max_distance_from_rix_km is not None:
            distance_outlier_mask = (
                work["_has_position"]
                & work["distance_from_rix_km"].notna()
                & (work["distance_from_rix_km"] > float(max_distance_from_rix_km))
            )
        else:
            distance_outlier_mask = pd.Series(False, index=work.index)

        if iqr_multiplier is not None:
            pos_speed_threshold = _iqr_upper_threshold(
                work.loc[work["_has_position"], "pos_speed_kmh"],
                float(iqr_multiplier),
            )
            vel_accel_threshold = _iqr_upper_threshold(
                work.loc[
                    work["_has_horizontal_velocity"].fillna(False).astype(bool)
                    | work["_has_vertical_velocity"].fillna(False).astype(bool),
                    "vel_accel_mps2",
                ],
                float(iqr_multiplier),
            )
        else:
            pos_speed_threshold = None
            vel_accel_threshold = None

        if pos_speed_threshold is not None:
            pos_speed_outlier_mask = (
                work["_has_position"]
                & work["pos_speed_kmh"].notna()
                & (work["pos_speed_kmh"] > pos_speed_threshold)
            )
        else:
            pos_speed_outlier_mask = pd.Series(False, index=work.index)

        if vel_accel_threshold is not None:
            vel_accel_outlier_mask = (
                (
                    work["_has_horizontal_velocity"].fillna(False).astype(bool)
                    | work["_has_vertical_velocity"].fillna(False).astype(bool)
                )
                & work["vel_accel_mps2"].notna()
                & (work["vel_accel_mps2"] > vel_accel_threshold)
            )
        else:
            vel_accel_outlier_mask = pd.Series(False, index=work.index)

        any_outlier_mask = (
            distance_outlier_mask
            | pos_speed_outlier_mask
            | vel_accel_outlier_mask
        ).fillna(False).astype(bool)

        report = {
            "enabled": True,
            "input_rows": int(len(work)),
            "output_rows": int((~any_outlier_mask).sum()),
            "removed_rows": int(any_outlier_mask.sum()),
            "boundary_rix": rule.boundary_rix.to_dict() if getattr(rule, "boundary_rix", None) is not None else None,
            "outlier_multiplier": rule.outlier_multiplier.to_dict() if getattr(rule, "outlier_multiplier", None) is not None else None,
            "thresholds": {
                "distance_from_rix_km": max_distance_from_rix_km,
                "pos_speed_kmh": pos_speed_threshold,
                "vel_accel_mps2": vel_accel_threshold,
            },
            "distance_from_rix": _count_outliers(distance_outlier_mask, position_count),
            "pos_speed": _count_outliers(pos_speed_outlier_mask, position_count),
            "vel_accel": _count_outliers(vel_accel_outlier_mask, velocity_count),
            "any": _count_outliers(any_outlier_mask, int(len(work))),
        }

        cleaned = work.loc[~any_outlier_mask].copy().reset_index(drop=True)
        return cleaned, report

    def _trim_before_first_position(
        self,
        work: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Drop leading rows before the first usable position after all filters.

        Raw tracks may begin with velocity-only ADS-B messages.  Those are useful
        diagnostics, but the raw_keyframes dataset consumed by viewers and later
        reconstruction stages should start at the first post-filter row that has
        at least a horizontal position observation, meaning lat/lon are present.
        """

        position_mask = work["_has_position"].fillna(False).astype(bool)
        if not position_mask.any():
            raise ValueError(
                "no raw ADS-B position rows remain after rule filters, "
                "outlier removal, and start trimming"
            )

        first_position_index = int(position_mask[position_mask].index[0])
        first_position_time_s = _json_number(work.loc[first_position_index, "_time_s"])
        first_position_keyframe_t = _json_number(work.loc[first_position_index, "_keyframe_t"])
        removed_rows = first_position_index

        trimmed = work.loc[first_position_index:].copy().reset_index(drop=True)
        report = {
            "enabled": True,
            "rule": "drop_rows_before_first_post_filter_position",
            "input_rows": int(len(work)),
            "output_rows": int(len(trimmed)),
            "removed_rows": int(removed_rows),
            "first_position_time_s": first_position_time_s,
            "first_position_keyframe_t": first_position_keyframe_t,
            "first_position_original_index": first_position_index,
            "position_definition": "lat/lon present after rule filters and outlier removal",
        }
        return trimmed, report

    def prepare(self, df: pd.DataFrame, *, rule: TrackRuleConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        cfg = self.config
        c = cfg.columns
        raw_count = int(len(df))
        work = df.copy()

        if rule is not None and cfg.apply_rule_filters:
            work = rule.filter_raw_dataframe(work)

        filtered_count = int(len(work))
        if filtered_count == 0:
            raise ValueError("no raw ADS-B rows remain after rule filters")

        if c.time_column not in work.columns:
            raise ValueError(f"raw ADS-B dataframe has no time column {c.time_column!r}")

        work["_time_s"] = _numeric_column(work, c.time_column)
        work = work[work["_time_s"].notna()].copy()
        if work.empty:
            raise ValueError("raw ADS-B dataframe contains no valid timestamps after parsing")

        if c.icao_column in work.columns:
            work["_icao"] = work[c.icao_column].astype(str).str.upper()
        else:
            work["_icao"] = rule.icao.upper() if rule is not None else None

        work["_lat_deg"] = _numeric_column(work, c.lat_column)
        work["_lon_deg"] = _numeric_column(work, c.lon_column)
        work["_altitude_ft_msl"] = _numeric_column(work, c.altitude_ft_column)
        work["_ground_speed_kt"] = _numeric_column(work, c.ground_speed_kt_column)
        work["_track"] = _numeric_column(work, c.track_column)

        if c.vertical_rate_fpm_column and c.vertical_rate_fpm_column in work.columns:
            work["_vertical_rate_fpm"] = _numeric_column(work, c.vertical_rate_fpm_column)
            vertical_rate_source = c.vertical_rate_fpm_column
        else:
            work["_vertical_rate_fpm"] = np.nan
            vertical_rate_source = None

        if c.vertical_rate_mps_column and c.vertical_rate_mps_column in work.columns:
            work["_vertical_rate_mps"] = _numeric_column(work, c.vertical_rate_mps_column)
            vertical_rate_mps_source = c.vertical_rate_mps_column
        else:
            work["_vertical_rate_mps"] = np.nan
            vertical_rate_mps_source = None

        # Prefer explicit SQL-loader columns. Fill missing fpm/mps from the other
        # unit, then fall back to decoded_json.velocity_raw[2] for older extracts.
        missing_fpm = work["_vertical_rate_fpm"].isna() & work["_vertical_rate_mps"].notna()
        work.loc[missing_fpm, "_vertical_rate_fpm"] = work.loc[missing_fpm, "_vertical_rate_mps"] / (FT_TO_M / 60.0)

        missing_mps = work["_vertical_rate_mps"].isna() & work["_vertical_rate_fpm"].notna()
        work.loc[missing_mps, "_vertical_rate_mps"] = work.loc[missing_mps, "_vertical_rate_fpm"] * FT_TO_M / 60.0

        if cfg.parse_vertical_rate_from_decoded_json and c.decoded_json_column in work.columns:
            parsed = work[c.decoded_json_column].map(_extract_vertical_rate_fpm_from_json)
            parsed = pd.to_numeric(parsed, errors="coerce")
            missing = work["_vertical_rate_fpm"].isna()
            work.loc[missing, "_vertical_rate_fpm"] = parsed.loc[missing]
            missing_mps = work["_vertical_rate_mps"].isna() & work["_vertical_rate_fpm"].notna()
            work.loc[missing_mps, "_vertical_rate_mps"] = work.loc[missing_mps, "_vertical_rate_fpm"] * FT_TO_M / 60.0
            if parsed.notna().any() and vertical_rate_source is None:
                vertical_rate_source = "decoded_json.velocity_raw[2]"

        if vertical_rate_mps_source is not None:
            vertical_rate_source = vertical_rate_mps_source

        # Event-kind semantics are channel based.  A position event only
        # needs lat/lon; altitude is optional at this raw-loader stage.  A later
        # V-Spline 3D fit still requires full position (lat/lon/altitude) and
        # full velocity (horizontal + vertical rate).
        work["_has_position_xy"] = work["_lat_deg"].notna() & work["_lon_deg"].notna()
        work["_has_full_position"] = work["_has_position_xy"] & work["_altitude_ft_msl"].notna()
        work["_has_position"] = work["_has_position_xy"]
        work["_has_horizontal_velocity"] = work["_ground_speed_kt"].notna() & work["_track"].notna()
        work["_has_vertical_velocity"] = work["_vertical_rate_mps"].notna()
        work["_has_full_velocity"] = work["_has_horizontal_velocity"] & work["_has_vertical_velocity"]

        east_values: list[float | None] = []
        north_values: list[float | None] = []
        for gs, track in zip(work["_ground_speed_kt"].tolist(), work["_track"].tolist()):
            east, north = _velocity_components_from_gs_track(
                _json_number(gs),
                _json_number(track),
                cfg.track_unit,
            )
            east_values.append(east)
            north_values.append(north)
        work["_east_mps"] = east_values
        work["_north_mps"] = north_values
        work["_up_mps"] = work["_vertical_rate_mps"]

        q = getattr(rule, "keyframe_time_quantization_s", None) if rule is not None else None
        q = float(q) if q is not None else float(cfg.keyframe_time_quantization_s)
        if q <= 0:
            raise ValueError("keyframe time quantization must be > 0")
        work["_keyframe_t"] = (work["_time_s"] / q).round() * q

        sort_cols = ["_time_s"]
        if c.raw_id_column in work.columns:
            work["_row_sort"] = pd.to_numeric(work[c.raw_id_column], errors="coerce")
            sort_cols.append("_row_sort")
        elif c.row_id_column and c.row_id_column in work.columns:
            work["_row_sort"] = pd.to_numeric(work[c.row_id_column], errors="coerce")
            sort_cols.append("_row_sort")
        work = work.sort_values(sort_cols).reset_index(drop=True)

        # Diagnostics are computed before outlier removal so thresholds can use
        # the original sequence of accepted ICAO/CRC/time-window rows.
        work = self._add_diagnostics(work)
        before_outlier_count = int(len(work))
        work, outlier_report = self._remove_rule_outliers(work, rule=rule)
        after_outlier_count = int(len(work))
        if work.empty:
            raise ValueError("no raw ADS-B rows remain after outlier removal")

        before_start_trim_count = int(len(work))
        work, start_trim_report = self._trim_before_first_position(work)
        after_start_trim_count = int(len(work))
        if work.empty:
            raise ValueError("no raw ADS-B rows remain after trimming to first position")

        # Recompute final diagnostics after outlier removal and start trimming so
        # exported events/keyframes are self-consistent and the first retained
        # position has no diagnostic dependency on a dropped row.
        work = self._add_diagnostics(work)

        work["_event_id"] = [_event_id(i + 1) for i in range(len(work))]

        # Canonical event kind used by keyframe aggregation and reports.
        # Prefer the rule's event-kind inference so the allowed_events filter and
        # the normalizer use identical semantics.  Preserve any source column in
        # _source_event_kind before overwriting event_kind with the canonical value.
        if c.event_kind_column in work.columns:
            work["_source_event_kind"] = work[c.event_kind_column].astype(str)
        else:
            work["_source_event_kind"] = None

        if rule is not None:
            work["event_kind"] = rule.event_kind_series(work).astype(str).str.strip().str.lower()
        else:
            work["event_kind"] = np.where(
                work["_has_position_xy"] & work["_has_horizontal_velocity"],
                "position_velocity",
                np.where(
                    work["_has_position_xy"],
                    "position",
                    np.where(
                        work["_has_horizontal_velocity"] | work["_has_vertical_velocity"],
                        "velocity",
                        "other",
                    ),
                ),
            )

        report = {
            "raw_row_count": raw_count,
            "filtered_row_count": filtered_count,
            "valid_time_row_count": int(len(work)),
            "vertical_rate_source": vertical_rate_source,
            "keyframe_time_quantization_s": q,
            "allowed_events": list(rule.allowed_events) if rule is not None and rule.allowed_events is not None else None,
            "pre_outlier_row_count": before_outlier_count,
            "post_outlier_row_count": after_outlier_count,
            "pre_start_trim_row_count": before_start_trim_count,
            "post_start_trim_row_count": after_start_trim_count,
            "outlier_removal": outlier_report,
            "start_trim": start_trim_report,
            "event_kind_counts": {str(k): int(v) for k, v in work["event_kind"].value_counts(dropna=False).items()},
            "position_xy_message_count": int(work["_has_position_xy"].sum()),
            "full_position_message_count": int(work["_has_full_position"].sum()),
            "horizontal_velocity_message_count": int(work["_has_horizontal_velocity"].sum()),
            "vertical_velocity_message_count": int(work["_has_vertical_velocity"].sum()),
            "full_velocity_message_count": int(work["_has_full_velocity"].sum()),
        }
        return work, report


class RawAdsbKeyframeBuilder:
    """Build traceable time-bucket keyframes from prepared raw ADS-B events."""

    def __init__(self, config: RawAdsbNormalizeConfig) -> None:
        self.config = config

    def build(self, events: pd.DataFrame, *, rule: TrackRuleConfig | None = None) -> list[dict[str, Any]]:
        keyframes: list[dict[str, Any]] = []
        for n, (t_key, group) in enumerate(events.groupby("_keyframe_t", sort=True), start=1):
            keyframe = self._build_keyframe(n, float(t_key), group, rule=rule)
            if self.config.keep_unpaired_timestamps or bool(keyframe.get("paired_for_vspline")):
                keyframes.append(keyframe)

        if self.config.derive_velocity_delta_acceleration:
            keyframes = self._add_derived_velocity_delta_acceleration(keyframes)

        return _clean_json(keyframes)

    def _build_keyframe(self, n: int, t: float, group: pd.DataFrame, *, rule: TrackRuleConfig | None) -> dict[str, Any]:
        source_event_ids = [x for x in group["_event_id"].tolist() if not _is_missing(x)]
        row_ids = self._row_ids(group)
        position = self._build_position(t=t, group=group, rule=rule)
        velocity = self._build_velocity(group)
        acceleration = self._build_acceleration(group)

        has_position = position is not None
        has_full_position = bool(position is not None and position.get("z_m") is not None)
        has_full_velocity = bool(
            velocity is not None
            and velocity.get("east_mps") is not None
            and velocity.get("north_mps") is not None
            and velocity.get("up_mps") is not None
        )
        event_kind = self._aggregate_event_kind(group, has_position=has_position, has_velocity=velocity is not None)

        return _clean_json(
            {
                "id": _keyframe_id(n),
                "t": t,
                "source_event_ids": source_event_ids,
                "row_ids": row_ids,
                "event_kind": event_kind,
                "event_kinds": _unique_strings(group, "event_kind"),
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
                "full_position_for_vspline": has_full_position,
                "full_velocity_for_vspline": has_full_velocity,
                "paired_for_vspline": bool(has_full_position and has_full_velocity),
            }
        )

    def _row_ids(self, group: pd.DataFrame) -> list[int]:
        c = self.config.columns
        candidates = []
        if c.raw_id_column in group.columns:
            candidates = pd.to_numeric(group[c.raw_id_column], errors="coerce").dropna().tolist()
        elif c.row_id_column and c.row_id_column in group.columns:
            candidates = pd.to_numeric(group[c.row_id_column], errors="coerce").dropna().tolist()
        return [int(x) for x in candidates]

    def _build_position(self, *, t: float, group: pd.DataFrame, rule: TrackRuleConfig | None) -> dict[str, Any] | None:
        # Position keyframes are allowed with lat/lon even if altitude is absent.
        # Such keyframes are useful for traceability and horizontal display, but
        # they are not full 3D V-Spline observations until altitude is present.
        pos = group[group["_has_position_xy"].fillna(False).astype(bool)].copy()
        if pos.empty:
            return None

        lat = _mean_numeric(pos, "_lat_deg")
        lon = _mean_numeric(pos, "_lon_deg")
        alt_ft = _mean_numeric(pos, "_altitude_ft_msl")
        if lat is None or lon is None:
            return None

        field_ft = rule.field_elevation.elevation_ft_msl if rule is not None else 0.0
        field_m = rule.field_elevation.elevation_m_msl if rule is not None else 0.0
        on_ground_applied = bool(
            rule is not None
            and rule.on_ground_window is not None
            and rule.on_ground_window.contains(t)
        )

        raw_alt_ft_before_override = alt_ft
        if on_ground_applied:
            # Manual flight-specific ground rule: inside this window, trust the
            # configured field elevation for vertical position. Raw altitude is
            # preserved diagnostically, but normalized altitude becomes field elevation.
            alt_ft = field_ft
            altitude_m_msl = field_m
            height_above_reference_m = 0.0
            display_height_m = 0.0
            z_m = 0.0 if self.config.baro_z_reference == "field" else altitude_m_msl
            altitude_source = "on_ground_window_field_elevation_override"
        else:
            altitude_m_msl = alt_ft * FT_TO_M if alt_ft is not None else None
            height_above_reference_m = (alt_ft - field_ft) * FT_TO_M if alt_ft is not None else None
            display_height_m = (
                max(0.0, height_above_reference_m)
                if height_above_reference_m is not None and self.config.clip_height_to_zero_for_display
                else height_above_reference_m
            )
            z_m = display_height_m if self.config.baro_z_reference == "field" else altitude_m_msl
            altitude_source = "observed_altitude_ft_msl" if alt_ft is not None else "missing_altitude"

        return {
            "source": "observed_adsb_position",
            "frame": "horizontal_enu_plus_barometric_z",
            "aggregation": "mean_within_t_bucket",
            "observation_count": int(len(pos)),
            "lat": lat,
            "lon": lon,
            "lat_deg": lat,
            "lon_deg": lon,
            "altitude_ft_msl": alt_ft,
            "raw_altitude_ft_msl_before_override": raw_alt_ft_before_override if on_ground_applied else None,
            "altitude_source": altitude_source,
            "altitude_override_applied": on_ground_applied,
            "on_ground_window_applied": on_ground_applied,
            "altitude_m_msl": altitude_m_msl,
            "height_above_reference_m": height_above_reference_m,
            "height_above_reference_m_display": display_height_m,
            "height_above_field_m": height_above_reference_m,
            "distance_from_rix_km": _mean_numeric(pos, "distance_from_rix_km"),
            "height_clipped_for_display": bool(
                display_height_m is not None
                and height_above_reference_m is not None
                and display_height_m != height_above_reference_m
            ),
            "full_position_for_vspline": z_m is not None,
            "z_m": z_m,
            "z_reference": self.config.baro_z_reference,
            "vertical_reference_ft_msl": field_ft,
            "vertical_reference_source": "track_rule_field_elevation" if rule is not None else "default_zero_reference",
            "field_elevation_ft_msl": field_ft,
        }

    def _build_velocity(self, group: pd.DataFrame) -> dict[str, Any] | None:
        vel = group[group["_has_horizontal_velocity"].fillna(False).astype(bool)].copy()
        if vel.empty:
            return None

        east_values = [float(x) for x in pd.to_numeric(vel["_east_mps"], errors="coerce").dropna().tolist()]
        north_values = [float(x) for x in pd.to_numeric(vel["_north_mps"], errors="coerce").dropna().tolist()]
        if not east_values or not north_values:
            return None

        east_mps = _mean_vector(east_values)
        north_mps = _mean_vector(north_values)
        if east_mps is None or north_mps is None:
            return None

        vector_mean_ground_speed_kt = math.hypot(east_mps, north_mps) / KNOT_TO_MPS
        track = _track_from_velocity_components(east_mps, north_mps, self.config.track_unit)
        track_rad = _track_to_rad(track, self.config.track_unit) if track is not None else None
        track_deg = math.degrees(track_rad) if track_rad is not None else None

        vertical_vel = vel[vel["_has_vertical_velocity"].fillna(False).astype(bool)].copy()
        vertical_rate_fpm = _mean_numeric(vertical_vel, "_vertical_rate_fpm") if not vertical_vel.empty else None
        up_mps = _mean_numeric(vertical_vel, "_vertical_rate_mps") if not vertical_vel.empty else None

        return {
            "source": "reported_adsb_ground_speed_track_plus_adsb_vertical_rate",
            "frame": "local_enu",
            "dimension": "3d" if up_mps is not None else "horizontal",
            "aggregation": "mean_vector_within_t_bucket_from_reported_gs_track_and_vertical_rate",
            "observation_count": int(len(vel)),
            "full_3d_observation_count": int(len(vel[vel["_has_full_velocity"].fillna(False).astype(bool)])),
            "vertical_observation_count": int(len(vertical_vel)),
            "ground_speed_kt": vector_mean_ground_speed_kt,
            "mean_reported_ground_speed_kt": _mean_numeric(vel, "_ground_speed_kt"),
            "track": track,
            "track_unit": self.config.track_unit,
            "track_rad": track_rad,
            "track_deg": track_deg,
            "east_mps": east_mps,
            "north_mps": north_mps,
            "up_mps": up_mps,
            "vertical_rate_fpm": vertical_rate_fpm,
            "vertical_component_available": up_mps is not None,
            "ignored_precomputed_components": "vel_east_mps/vel_north_mps are ignored if present; horizontal vector source is reported GS+track; vertical vector source is ADS-B vertical_rate_mps/fpm",
        }

    def _build_acceleration(self, group: pd.DataFrame) -> dict[str, Any] | None:
        # Placeholder channel. Directional acceleration is added later from
        # consecutive velocity keyframes. Loader diagnostic magnitudes can be
        # added here in the future if available.
        has_velocity = (
            group["_has_horizontal_velocity"].fillna(False).astype(bool)
            | group["_has_vertical_velocity"].fillna(False).astype(bool)
        )
        if not has_velocity.any():
            return None
        return {
            "source": "unavailable_until_consecutive_velocity_keyframes",
            "frame": "local_enu",
            "dimension": "3d_when_vertical_rate_available",
            "aggregation": "velocity_keyframe_placeholder",
            "observation_count": int(has_velocity.sum()),
            "east_mps2": None,
            "north_mps2": None,
            "up_mps2": None,
            "horizontal_mps2": None,
            "vertical_mps2": None,
            "magnitude_mps2": None,
            "vertical_component_available": bool(group["_has_vertical_velocity"].fillna(False).astype(bool).any()),
            "used_as_vspline_observation": False,
        }

    def _add_derived_velocity_delta_acceleration(self, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = [dict(kf) for kf in keyframes]
        previous: dict[str, Any] | None = None

        for keyframe in result:
            velocity = keyframe.get("velocity")
            if not isinstance(velocity, dict):
                continue

            east = _json_number(velocity.get("east_mps"))
            north = _json_number(velocity.get("north_mps"))
            up = _json_number(velocity.get("up_mps"))
            t = _json_number(keyframe.get("t"))

            if t is None or (east is None and north is None and up is None):
                continue

            acceleration = dict(keyframe.get("acceleration") or {})
            acceleration.update(
                {
                    "frame": "local_enu",
                    "dimension": "3d" if up is not None else "horizontal",
                    "vertical_component_available": up is not None,
                    "used_as_vspline_observation": False,
                }
            )

            if previous is None:
                acceleration.update(
                    {
                        "source": "unavailable_no_previous_velocity_keyframe",
                        "dt_s": None,
                        "previous_keyframe_id": None,
                        "current_keyframe_id": keyframe.get("id"),
                        "east_mps2": None,
                        "north_mps2": None,
                        "up_mps2": None,
                        "horizontal_mps2": None,
                        "vertical_mps2": None,
                        "magnitude_mps2": None,
                    }
                )
            else:
                prev_t = _json_number(previous.get("t"))
                prev_e = _json_number(previous.get("east_mps"))
                prev_n = _json_number(previous.get("north_mps"))
                prev_u = _json_number(previous.get("up_mps"))
                dt = t - prev_t if prev_t is not None else None

                if dt is None or dt <= 0:
                    acceleration.update({"source": "unavailable_non_positive_dt", "dt_s": dt})
                elif self.config.acceleration_max_dt_s is not None and dt > self.config.acceleration_max_dt_s:
                    acceleration.update(
                        {
                            "source": "unavailable_dt_exceeds_acceleration_max_dt_s",
                            "dt_s": dt,
                            "acceleration_max_dt_s": self.config.acceleration_max_dt_s,
                        }
                    )
                else:
                    east_mps2 = (east - prev_e) / dt if east is not None and prev_e is not None else None
                    north_mps2 = (north - prev_n) / dt if north is not None and prev_n is not None else None
                    up_mps2 = (up - prev_u) / dt if up is not None and prev_u is not None else None

                    horizontal_mps2 = (
                        math.hypot(east_mps2, north_mps2)
                        if east_mps2 is not None and north_mps2 is not None
                        else None
                    )
                    vertical_mps2 = abs(up_mps2) if up_mps2 is not None else None

                    components = [x for x in (east_mps2, north_mps2, up_mps2) if x is not None]
                    magnitude_mps2 = math.sqrt(sum(x * x for x in components)) if components else None

                    if magnitude_mps2 is None:
                        acceleration.update({"source": "unavailable_invalid_previous_velocity", "dt_s": dt})
                    else:
                        acceleration.update(
                            {
                                "source": "derived_from_consecutive_adsb_velocity_delta",
                                "dt_s": dt,
                                "previous_keyframe_id": previous.get("id"),
                                "current_keyframe_id": keyframe.get("id"),
                                "east_mps2": east_mps2,
                                "north_mps2": north_mps2,
                                "up_mps2": up_mps2,
                                "horizontal_mps2": horizontal_mps2,
                                "vertical_mps2": vertical_mps2,
                                "magnitude_mps2": magnitude_mps2,
                            }
                        )

            keyframe["acceleration"] = acceleration
            previous = {
                "id": keyframe.get("id"),
                "t": t,
                "east_mps": east,
                "north_mps": north,
                "up_mps": up,
            }

        return result

    def _aggregate_event_kind(self, group: pd.DataFrame, *, has_position: bool, has_velocity: bool) -> str:
        if has_position and has_velocity:
            return "position_velocity"
        if has_position:
            return "position"
        if has_velocity:
            return "velocity"
        kinds = _unique_strings(group, "event_kind")
        if len(kinds) == 1:
            return kinds[0]
        if len(kinds) > 1:
            return "mixed"
        return "unknown"


class RawAdsbNormalizer:
    """Normalize raw ADS-B rows into events, keyframes, and flat aviation rows."""

    def __init__(self, config: RawAdsbNormalizeConfig | None = None) -> None:
        self.config = config or RawAdsbNormalizeConfig()
        self.preparer = RawAdsbEventPreparer(self.config)
        self.keyframe_builder = RawAdsbKeyframeBuilder(self.config)

    def normalize(self, df: pd.DataFrame, *, rule: TrackRuleConfig | None = None) -> RawAdsbNormalizeResult:
        raw_count = int(len(df))
        logger.info(
            "Raw ADS-B normalization started: {}",
            {"input_rows": raw_count, "rule": rule.track_id if rule is not None else None},
        )

        events, prepare_report = self.preparer.prepare(df, rule=rule)
        keyframes = self.keyframe_builder.build(events, rule=rule)
        flat = self._keyframes_to_flat_dataframe(keyframes)

        paired_count = int(flat.get("paired_for_vspline", pd.Series(False, index=flat.index)).fillna(False).sum())
        on_ground_override_count = int(
            flat.get("on_ground_window_applied", pd.Series(False, index=flat.index))
            .fillna(False)
            .astype(bool)
            .sum()
        )
        on_ground_report = None
        if rule is not None and rule.on_ground_window is not None:
            on_ground_report = {
                "enabled": True,
                **rule.on_ground_window.to_dict(),
                "altitude_override": "field_elevation",
                "field_elevation_ft_msl": rule.field_elevation.elevation_ft_msl,
                "field_elevation_m_msl": rule.field_elevation.elevation_m_msl,
                "affected_keyframe_count": on_ground_override_count,
            }
        report = {
            "raw_row_count": raw_count,
            "filtered_row_count": prepare_report["filtered_row_count"],
            "valid_time_row_count": prepare_report["valid_time_row_count"],
            "event_count": int(len(events)),
            "keyframe_count": int(len(keyframes)),
            "output_timestamp_count": int(len(flat)),
            "paired_timestamp_count": paired_count,
            "on_ground_window": on_ground_report,
            "rule_applied": rule.to_dict() if rule is not None and self.config.apply_rule_filters else None,
            "normalizer_config": asdict(self.config),
            **prepare_report,
        }

        logger.info(
            "Raw ADS-B normalization completed: {}",
            {
                "raw_row_count": report["raw_row_count"],
                "filtered_row_count": report["filtered_row_count"],
                "keyframe_count": report["keyframe_count"],
                "paired_timestamp_count": report["paired_timestamp_count"],
                "on_ground_override_count": on_ground_override_count,
            },
        )
        return RawAdsbNormalizeResult(dataframe=flat, report=report, events=events, keyframes=keyframes)

    def _keyframes_to_flat_dataframe(self, keyframes: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for keyframe in keyframes:
            position = keyframe.get("position") if isinstance(keyframe.get("position"), dict) else {}
            velocity = keyframe.get("velocity") if isinstance(keyframe.get("velocity"), dict) else {}
            acceleration = keyframe.get("acceleration") if isinstance(keyframe.get("acceleration"), dict) else {}

            row = {
                "time_s": keyframe.get("t"),
                "keyframe_id": keyframe.get("id"),
                "event_kind": keyframe.get("event_kind"),
                "event_kinds": json.dumps(keyframe.get("event_kinds", []), ensure_ascii=False),
                "source_event_ids": json.dumps(keyframe.get("source_event_ids", []), ensure_ascii=False),
                "row_ids": json.dumps(keyframe.get("row_ids", []), ensure_ascii=False),
                "paired_for_vspline": bool(keyframe.get("paired_for_vspline", False)),
                "full_position_for_vspline": bool(keyframe.get("full_position_for_vspline", False)),
                "full_velocity_for_vspline": bool(keyframe.get("full_velocity_for_vspline", False)),
                "lat": position.get("lat"),
                "lon": position.get("lon"),
                "lat_deg": position.get("lat_deg"),
                "lon_deg": position.get("lon_deg"),
                "altitude_ft_msl": position.get("altitude_ft_msl"),
                "raw_altitude_ft_msl_before_override": position.get("raw_altitude_ft_msl_before_override"),
                "altitude_source": position.get("altitude_source"),
                "altitude_override_applied": position.get("altitude_override_applied"),
                "on_ground_window_applied": position.get("on_ground_window_applied"),
                "altitude_m_msl": position.get("altitude_m_msl"),
                "height_above_reference_m": position.get("height_above_reference_m"),
                "height_above_reference_m_display": position.get("height_above_reference_m_display"),
                "height_above_field_m": position.get("height_above_field_m"),
                "distance_from_rix_km": position.get("distance_from_rix_km"),
                "z_m": position.get("z_m"),
                "z_reference": position.get("z_reference"),
                "vertical_reference_ft_msl": position.get("vertical_reference_ft_msl"),
                "vertical_reference_source": position.get("vertical_reference_source"),
                "field_elevation_ft_msl": position.get("field_elevation_ft_msl"),
                "ground_speed_kt": velocity.get("ground_speed_kt"),
                "mean_reported_ground_speed_kt": velocity.get("mean_reported_ground_speed_kt"),
                "track_deg": velocity.get("track_deg"),
                "track_rad": velocity.get("track_rad"),
                "east_mps": velocity.get("east_mps"),
                "north_mps": velocity.get("north_mps"),
                "up_mps": velocity.get("up_mps"),
                "vertical_rate_fpm": velocity.get("vertical_rate_fpm"),
                "vertical_component_available": velocity.get("vertical_component_available"),
                "position_observation_count": position.get("observation_count"),
                "velocity_observation_count": velocity.get("observation_count"),
                "velocity_full_3d_observation_count": velocity.get("full_3d_observation_count"),
                "position_aggregation": position.get("aggregation"),
                "velocity_aggregation": velocity.get("aggregation"),
                "acc_east_mps2": acceleration.get("east_mps2"),
                "acc_north_mps2": acceleration.get("north_mps2"),
                "acc_up_mps2": acceleration.get("up_mps2"),
                "acc_horizontal_mps2": acceleration.get("horizontal_mps2"),
                "acc_vertical_mps2": acceleration.get("vertical_mps2"),
                "acc_magnitude_mps2": acceleration.get("magnitude_mps2"),
                "acc_source": acceleration.get("source"),
                "acc_used_as_vspline_observation": acceleration.get("used_as_vspline_observation"),
            }
            rows.append(_clean_json(row))

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("time_s").reset_index(drop=True)
        return out
