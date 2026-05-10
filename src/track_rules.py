"""
Rule-based per-track configuration.

This module is intentionally separated from the V-Spline math.  It defines
schemas and loaders for human-curated flight rules.  The actual per-flight
values live in JSON, usually:

    config/flight_rules.json

Each rule can define:
- first/last Unix timestamp to reconstruct,
- ICAO/track identity,
- field-elevation reference used for vertical coordinates,
- optional fixed horizontal ENU origin,
- future manual rules that are too track-specific to belong in the core.

The field-elevation behavior mirrors the old reconstructor's field-relative
vertical convention: z_m = (altitude_ft_msl - field_elevation_ft_msl) * 0.3048.
In the aviation preprocessor, this is achieved by using MSL altitude input but
setting the local ENU origin altitude to the configured field elevation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import json

import numpy as np
import pandas as pd
from loguru import logger

from geo_utils import FT_TO_M


FieldElevationMethod = Literal["fixed_ft_msl", "fixed_m_msl"]


@dataclass(frozen=True)
class FieldElevationRule:
    """Human-curated field-elevation rule for one track.

    Use ``fixed_ft_msl`` when copying the behavior of the old script, where
    ``VSplineConfig.field_elevation_ft_msl`` was a fixed value and vertical
    fitting used field-relative height.
    """

    method: FieldElevationMethod = "fixed_ft_msl"
    value: float = 0.0
    source: str = "manual"
    notes: str | None = None

    @classmethod
    def fixed_ft_msl(
        cls,
        value_ft_msl: float,
        *,
        source: str = "manual",
        notes: str | None = None,
    ) -> "FieldElevationRule":
        return cls(method="fixed_ft_msl", value=float(value_ft_msl), source=source, notes=notes)

    @classmethod
    def fixed_m_msl(
        cls,
        value_m_msl: float,
        *,
        source: str = "manual",
        notes: str | None = None,
    ) -> "FieldElevationRule":
        return cls(method="fixed_m_msl", value=float(value_m_msl), source=source, notes=notes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldElevationRule":
        method = data.get("method", "fixed_ft_msl")
        if method not in ("fixed_ft_msl", "fixed_m_msl"):
            raise ValueError(f"unsupported field elevation method: {method!r}")
        return cls(
            method=method,
            value=float(data.get("value", 0.0)),
            source=str(data.get("source", "manual")),
            notes=data.get("notes"),
        )

    @property
    def elevation_ft_msl(self) -> float:
        if self.method == "fixed_ft_msl":
            return float(self.value)
        if self.method == "fixed_m_msl":
            return float(self.value) / FT_TO_M
        raise ValueError(f"unsupported field elevation method: {self.method!r}")

    @property
    def elevation_m_msl(self) -> float:
        if self.method == "fixed_ft_msl":
            return float(self.value) * FT_TO_M
        if self.method == "fixed_m_msl":
            return float(self.value)
        raise ValueError(f"unsupported field elevation method: {self.method!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "elevation_ft_msl": self.elevation_ft_msl,
            "elevation_m_msl": self.elevation_m_msl,
        }


@dataclass(frozen=True)
class TrackTimeWindow:
    """Inclusive Unix-second time window for the hand-curated track."""

    first_point_unix: float
    last_point_unix: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.first_point_unix) or not np.isfinite(self.last_point_unix):
            raise ValueError("time-window endpoints must be finite")
        if self.last_point_unix < self.first_point_unix:
            raise ValueError("last_point_unix must be >= first_point_unix")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackTimeWindow":
        return cls(
            first_point_unix=float(data["first_point_unix"]),
            last_point_unix=float(data["last_point_unix"]),
        )

    def mask(self, t: pd.Series) -> pd.Series:
        tt = pd.to_numeric(t, errors="coerce")
        return tt.notna() & (tt >= float(self.first_point_unix)) & (tt <= float(self.last_point_unix))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class OnGroundWindow:
    """Inclusive Unix-second window where aircraft is manually treated as on the field.

    During raw ADS-B normalization, position keyframes in this window may have
    altitude overwritten to the configured field elevation. This produces
    field-relative z_m = 0 while preserving the raw altitude diagnostically.
    """

    start_unix: float
    end_unix: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_unix) or not np.isfinite(self.end_unix):
            raise ValueError("on-ground window endpoints must be finite")
        if self.end_unix < self.start_unix:
            raise ValueError("on_ground_window.end_unix must be >= start_unix")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OnGroundWindow | None":
        if data is None:
            return None
        return cls(
            start_unix=float(data["start_unix"]),
            end_unix=float(data["end_unix"]),
        )

    def contains(self, t: float | int | None) -> bool:
        if t is None:
            return False
        try:
            tt = float(t)
        except Exception:
            return False
        return float(self.start_unix) <= tt <= float(self.end_unix)

    def mask(self, t: pd.Series) -> pd.Series:
        tt = pd.to_numeric(t, errors="coerce")
        return tt.notna() & (tt >= float(self.start_unix)) & (tt <= float(self.end_unix))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryRixRule:
    """Maximum accepted horizontal distance from Riga/RIX airport.

    The JSON shape is intentionally small::

        {"value": 90.0, "unit": "Km"}

    The normalizer converts the value to kilometers and removes position rows
    farther than this radius before keyframe aggregation.
    """

    value: float
    unit: str = "km"

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("boundary_rix.value must be finite and > 0")
        normalized_unit = self.unit.strip().lower()
        if normalized_unit not in {"km", "kilometer", "kilometers"}:
            raise ValueError(f"unsupported boundary_rix unit: {self.unit!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BoundaryRixRule | None":
        if data is None:
            return None
        return cls(
            value=float(data["value"]),
            unit=str(data.get("unit", "km")),
        )

    @property
    def distance_km(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"value": float(self.value), "unit": self.unit, "distance_km": self.distance_km}


@dataclass(frozen=True)
class OutlierMultiplierRule:
    """IQR multiplier used by raw ADS-B outlier filters."""

    value: float
    unit: str = "IQR"

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("outlier_multiplier.value must be finite and > 0")
        normalized_unit = self.unit.strip().lower()
        if normalized_unit != "iqr":
            raise ValueError(f"unsupported outlier_multiplier unit: {self.unit!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutlierMultiplierRule | None":
        if data is None:
            return None
        return cls(
            value=float(data["value"]),
            unit=str(data.get("unit", "IQR")),
        )

    @property
    def multiplier(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"value": float(self.value), "unit": self.unit, "multiplier": self.multiplier}


@dataclass(frozen=True)
class TrackRuleConfig:
    """Per-track rule packet.

    This should be the only place where a specific aircraft/track gets manual
    decisions such as time trimming or field elevation.  The mathematical core
    should never import this class.
    """

    track_id: str
    icao: str
    time_window: TrackTimeWindow
    field_elevation: FieldElevationRule
    on_ground_window: OnGroundWindow | None = None
    boundary_rix: BoundaryRixRule | None = None
    outlier_multiplier: OutlierMultiplierRule | None = None

    # Optional horizontal origin override. If omitted, the aviation preprocessor
    # uses the first clean point's lat/lon.  The altitude origin is always set
    # from field_elevation to reproduce the old field-relative vertical convention.
    origin_lat_deg: float | None = None
    origin_lon_deg: float | None = None

    require_crc_ok: bool = True
    # Optional event-kind filter, usually from JSON as ["velocity", "position"].
    # If both "position" and "velocity" are allowed, combined
    # "position_velocity" rows are kept automatically.
    allowed_events: tuple[str, ...] | None = None

    raw_time_column: str = "ts_utc"
    raw_icao_column: str = "icao"
    raw_crc_ok_column: str = "crc_ok"
    raw_event_kind_column: str = "event_kind"
    keyframe_time_quantization_s: float = 1.0

    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackRuleConfig":
        return cls(
            track_id=str(data["track_id"]),
            icao=str(data["icao"]).upper(),
            time_window=TrackTimeWindow.from_dict(data["time_window"]),
            field_elevation=FieldElevationRule.from_dict(data["field_elevation"]),
            on_ground_window=OnGroundWindow.from_dict(data.get("on_ground_window")),
            boundary_rix=BoundaryRixRule.from_dict(data.get("boundary_rix")),
            outlier_multiplier=OutlierMultiplierRule.from_dict(data.get("outlier_multiplier")),
            origin_lat_deg=_optional_float(data.get("origin_lat_deg")),
            origin_lon_deg=_optional_float(data.get("origin_lon_deg")),
            require_crc_ok=bool(data.get("require_crc_ok", True)),
            allowed_events=_parse_allowed_events(data.get("allowed_events")),
            raw_time_column=str(data.get("raw_time_column", "ts_utc")),
            raw_icao_column=str(data.get("raw_icao_column", "icao")),
            raw_crc_ok_column=str(data.get("raw_crc_ok_column", "crc_ok")),
            raw_event_kind_column=str(data.get("raw_event_kind_column", "event_kind")),
            keyframe_time_quantization_s=float(data.get("keyframe_time_quantization_s", 1.0)),
            notes=dict(data.get("notes", {})),
        )

    def filter_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ICAO, CRC, allowed-event, and hand-curated time-window filters."""
        work = df.copy()
        start_count = int(len(work))

        if self.raw_icao_column in work.columns:
            work = work[work[self.raw_icao_column].astype(str).str.upper() == self.icao.upper()].copy()

        after_icao = int(len(work))

        if self.require_crc_ok and self.raw_crc_ok_column in work.columns:
            work = work[pd.to_numeric(work[self.raw_crc_ok_column], errors="coerce").eq(1)].copy()

        after_crc = int(len(work))

        if self.allowed_events is not None:
            event_kind = self.event_kind_series(work)
            allowed_mask = self.allowed_event_mask(event_kind)
            work = work[allowed_mask].copy()

        after_allowed_events = int(len(work))

        if self.raw_time_column not in work.columns:
            raise ValueError(f"raw dataframe has no time column {self.raw_time_column!r}")

        work = work[self.time_window.mask(work[self.raw_time_column])].copy()
        out = work.sort_values(self.raw_time_column).reset_index(drop=True)

        logger.info(
            "Track rule filter applied: {}",
            {
                "track_id": self.track_id,
                "icao": self.icao,
                "input_rows": start_count,
                "after_icao_rows": after_icao,
                "after_crc_rows": after_crc,
                "allowed_events": list(self.allowed_events) if self.allowed_events is not None else None,
                "after_allowed_events_rows": after_allowed_events,
                "after_time_window_rows": int(len(out)),
                "first_point_unix": self.time_window.first_point_unix,
                "last_point_unix": self.time_window.last_point_unix,
                "on_ground_window": self.on_ground_window.to_dict() if self.on_ground_window is not None else None,
                "boundary_rix": self.boundary_rix.to_dict() if self.boundary_rix is not None else None,
                "outlier_multiplier": self.outlier_multiplier.to_dict() if self.outlier_multiplier is not None else None,
            },
        )
        return out

    def expanded_allowed_events(self) -> set[str] | None:
        """Return normalized allowed event kinds with useful combined aliases."""
        if self.allowed_events is None:
            return None

        allowed = {str(x).strip().lower() for x in self.allowed_events if str(x).strip()}

        # A raw row/keyframe containing both channels must be retained when both
        # component event types are allowed.
        if "position" in allowed and "velocity" in allowed:
            allowed.add("position_velocity")
            allowed.add("velocity_position")

        return allowed

    def event_kind_series(self, df: pd.DataFrame) -> pd.Series:
        """Return raw event-kind series, deriving it if the source column is absent.

        The SQL/raw source may not always contain an event_kind column.  In that
        case we infer a conservative kind from common ADS-B columns:

        - position: lat/lon present, altitude optional
        - velocity: gs/track present
        - position_velocity: both present
        - other: neither present
        """
        if self.raw_event_kind_column in df.columns:
            return df[self.raw_event_kind_column].astype(str).str.strip().str.lower()

        lat = pd.to_numeric(df["lat"], errors="coerce") if "lat" in df.columns else pd.Series(np.nan, index=df.index)
        lon = pd.to_numeric(df["lon"], errors="coerce") if "lon" in df.columns else pd.Series(np.nan, index=df.index)
        gs = pd.to_numeric(df["gs"], errors="coerce") if "gs" in df.columns else pd.Series(np.nan, index=df.index)
        track = pd.to_numeric(df["track"], errors="coerce") if "track" in df.columns else pd.Series(np.nan, index=df.index)

        has_position = lat.notna() & lon.notna()
        has_velocity = gs.notna() & track.notna()

        return pd.Series(
            np.where(
                has_position & has_velocity,
                "position_velocity",
                np.where(has_position, "position", np.where(has_velocity, "velocity", "other")),
            ),
            index=df.index,
            dtype="string",
        )

    def allowed_event_mask(self, event_kind: pd.Series) -> pd.Series:
        """Return True where event_kind is allowed by this rule."""
        allowed = self.expanded_allowed_events()
        if allowed is None:
            return pd.Series(True, index=event_kind.index)
        normalized = event_kind.astype(str).str.strip().str.lower()
        return normalized.isin(allowed).fillna(False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "icao": self.icao,
            "time_window": self.time_window.to_dict(),
            "field_elevation": self.field_elevation.to_dict(),
            "on_ground_window": self.on_ground_window.to_dict() if self.on_ground_window is not None else None,
            "boundary_rix": self.boundary_rix.to_dict() if self.boundary_rix is not None else None,
            "outlier_multiplier": self.outlier_multiplier.to_dict() if self.outlier_multiplier is not None else None,
            "origin_lat_deg": self.origin_lat_deg,
            "origin_lon_deg": self.origin_lon_deg,
            "require_crc_ok": self.require_crc_ok,
            "allowed_events": list(self.allowed_events) if self.allowed_events is not None else None,
            "raw_time_column": self.raw_time_column,
            "raw_icao_column": self.raw_icao_column,
            "raw_crc_ok_column": self.raw_crc_ok_column,
            "raw_event_kind_column": self.raw_event_kind_column,
            "keyframe_time_quantization_s": self.keyframe_time_quantization_s,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TrackRuleRegistry:
    """A collection of per-flight rules loaded from JSON."""

    schema_version: str
    description: str | None
    rules: tuple[TrackRuleConfig, ...]
    source_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str | None = None) -> "TrackRuleRegistry":
        rules = tuple(TrackRuleConfig.from_dict(item) for item in data.get("rules", []))
        seen: set[str] = set()
        duplicates: list[str] = []
        for rule in rules:
            key = rule.track_id.upper()
            if key in seen:
                duplicates.append(rule.track_id)
            seen.add(key)
        if duplicates:
            raise ValueError(f"duplicate track_id values in flight rule registry: {duplicates}")
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            description=data.get("description"),
            rules=rules,
            source_path=source_path,
        )

    def get(self, track_id_or_icao: str) -> TrackRuleConfig:
        """Return a rule by unique track_id, or by ICAO when unambiguous."""
        key = str(track_id_or_icao).upper()

        by_track = [r for r in self.rules if r.track_id.upper() == key]
        if len(by_track) == 1:
            return by_track[0]

        by_icao = [r for r in self.rules if r.icao.upper() == key]
        if len(by_icao) == 1:
            return by_icao[0]
        if len(by_icao) > 1:
            track_ids = [r.track_id for r in by_icao]
            raise KeyError(
                f"multiple flight rules exist for ICAO {track_id_or_icao!r}; use exact track_id. "
                f"Candidates: {track_ids}"
            )
        raise KeyError(f"no TrackRuleConfig for {track_id_or_icao!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "source_path": self.source_path,
            "rules": [r.to_dict() for r in self.rules],
        }


def _parse_allowed_events(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None

    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError("allowed_events must be null, a string, or a list of strings") from exc

    normalized: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if not text:
            continue
        if text not in {"position", "velocity", "position_velocity", "velocity_position", "other"}:
            raise ValueError(
                "unsupported allowed event kind "
                f"{item!r}; expected position, velocity, position_velocity, or other"
            )
        if text not in normalized:
            normalized.append(text)

    return tuple(normalized) if normalized else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if np.isfinite(x) else None


def default_track_rules_path() -> Path:
    """Path to the package default flight-rule JSON registry."""
    return Path(__file__).resolve().parent / "config" / "flight_rules.json"


def load_track_rule_registry(path: str | Path | None = None) -> TrackRuleRegistry:
    """Load a flight-rule registry from JSON."""
    registry_path = Path(path) if path is not None else default_track_rules_path()
    with registry_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    registry = TrackRuleRegistry.from_dict(data, source_path=str(registry_path))
    logger.info(
        "Loaded track rule registry: {}",
        {
            "path": str(registry_path),
            "rule_count": len(registry.rules),
            "schema_version": registry.schema_version,
        },
    )
    return registry

