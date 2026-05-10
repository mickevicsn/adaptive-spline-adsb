"""
SQLite/raw-SQL ADS-B loader.

This module is intentionally outside the V-Spline math. Its job is only to
extract raw ADS-B rows for one ICAO from the production SQLite dataset:

    raw_id, ts_utc, icao, df, tc, bds, lat, lon, alt, gs, track,
    callsign, squawk, crc, crc_ok, decoded_json, seen_utc

The resulting dataframe can be sent to ``RawAdsbNormalizer`` and then to the
aviation V-Spline preprocessing layer.

This version only derives explicit vertical-rate fields from
``decoded_json.velocity_raw``:

    vertical_rate_fpm
    vertical_rate_mps
    vertical_rate_source

It deliberately does not derive or discuss horizontal velocity components from
the velocity message. Horizontal motion should be handled by the later
normalization/preprocessing layer from whatever source is currently considered
valid for the project.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import json
import sqlite3

import pandas as pd
from loguru import logger

from track_rules import TrackRuleConfig


FT_TO_M = 0.3048
FPM_TO_MPS = FT_TO_M / 60.0


DEFAULT_RAW_ADSB_COLUMNS: tuple[str, ...] = (
    "raw_id",
    "ts_utc",
    "icao",
    "df",
    "tc",
    "bds",
    "lat",
    "lon",
    "alt",
    "gs",
    "track",
    "callsign",
    "squawk",
    "crc",
    "crc_ok",
    "decoded_json",
    "seen_utc",
)


DERIVED_ADSB_VERTICAL_RATE_COLUMNS: tuple[str, ...] = (
    "vertical_rate_fpm",
    "vertical_rate_mps",
    "vertical_rate_source",
    "vertical_rate_available",
)


@dataclass(frozen=True)
class SqlAdsbColumnConfig:
    """Column names used by the raw ADS-B SQL table."""

    raw_id_column: str = "raw_id"
    time_column: str = "ts_utc"
    icao_column: str = "icao"
    crc_ok_column: str = "crc_ok"
    default_output_columns: tuple[str, ...] = DEFAULT_RAW_ADSB_COLUMNS


@dataclass(frozen=True)
class SqlAdsbLoadConfig:
    """Configuration for loading one ICAO from a SQLite ADS-B dataset.

    Parameters
    ----------
    table_name:
        SQL table to read. If omitted, the loader inspects the database and
        chooses the best matching table by column overlap.
    include_crc_false:
        Defaults to True so rule-based CRC filtering can be applied later by
        ``TrackRuleConfig`` / ``RawAdsbNormalizer`` instead of being baked into
        the SQL extraction step.
    first_point_unix, last_point_unix:
        Optional inclusive SQL-side time filter. Leave as None to load all rows
        for the ICAO, matching the old extraction behavior.
    strict_output_columns:
        If True, every ``default_output_columns`` entry must exist in the SQL
        table. If False, missing optional columns are skipped.
    derive_vertical_rate_columns:
        If True, parse ``decoded_json.velocity_raw`` and add explicit vertical
        rate fields.
    """

    table_name: str | None = None
    columns: SqlAdsbColumnConfig = field(default_factory=SqlAdsbColumnConfig)
    include_crc_false: bool = True
    first_point_unix: float | None = None
    last_point_unix: float | None = None
    strict_output_columns: bool = False
    derive_vertical_rate_columns: bool = True


@dataclass
class SqlAdsbLoadResult:
    """Result packet for one SQL ICAO extraction."""

    dataframe: pd.DataFrame
    report: dict[str, Any]

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(path, index=False)


def _quote_identifier(identifier: str) -> str:
    if not identifier:
        raise ValueError("SQL identifier cannot be empty")
    return '"' + identifier.replace('"', '""') + '"'


def _sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [str(r[0]) for r in rows]


def _sqlite_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    return [str(r[1]) for r in rows]


def _choose_table(
    conn: sqlite3.Connection,
    *,
    requested: str | None,
    required_columns: Iterable[str],
    preferred_tables: tuple[str, ...] = (
        "adsb_raw",
        "raw_adsb",
        "adsb",
        "messages",
        "matching_rows",
        "parsed_messages",
    ),
) -> tuple[str, list[str]]:
    if requested:
        cols = _sqlite_table_columns(conn, requested)
        if not cols:
            raise ValueError(f"table {requested!r} not found or has no columns")
        return requested, cols

    tables = _sqlite_tables(conn)
    if not tables:
        raise ValueError("SQLite database contains no user tables")

    required = {c.lower() for c in required_columns}
    best: tuple[int, int, str, list[str]] | None = None

    for table in tables:
        cols = _sqlite_table_columns(conn, table)
        lower = {c.lower() for c in cols}
        overlap = len(required & lower)
        preferred_rank = preferred_tables.index(table) if table in preferred_tables else len(preferred_tables)
        score = (overlap, -preferred_rank)
        candidate = (score[0], score[1], table, cols)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    if best[0] == 0:
        raise ValueError(
            "could not infer ADS-B table; pass SqlAdsbLoadConfig(table_name=...) explicitly"
        )
    return best[2], best[3]


def _present_columns(
    table_columns: list[str],
    requested_columns: Iterable[str],
    *,
    strict: bool,
) -> list[str]:
    by_lower = {c.lower(): c for c in table_columns}
    selected: list[str] = []
    missing: list[str] = []

    for col in requested_columns:
        real = by_lower.get(col.lower())
        if real is None:
            missing.append(col)
        else:
            selected.append(real)

    if missing and strict:
        raise ValueError(f"SQL table is missing expected columns: {missing}")

    return selected


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _parse_json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if value is None:
        return {}

    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return {}

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _extract_vertical_rate_fields(decoded_json: Any) -> dict[str, Any]:
    """Extract vertical-rate fields from decoded_json.

    Supported legacy shape:
        velocity_raw = [speed, angle, vertical_rate_fpm, type]

    Supported extended shape:
        velocity_raw = [
            speed,
            angle,
            vertical_rate_fpm,
            type,
            direction_source,
            vertical_rate_source,
        ]

    Dict-like velocity payloads are also supported defensively.
    """
    decoded = _parse_json_obj(decoded_json)
    vel = decoded.get("velocity_raw")

    # Some pipelines accidentally double-encode nested JSON values.
    if isinstance(vel, str):
        try:
            parsed = json.loads(vel)
            vel = parsed
        except Exception:
            pass

    vertical_rate_fpm = None
    vertical_rate_source = None

    if isinstance(vel, (list, tuple)):
        if len(vel) >= 3:
            vertical_rate_fpm = _to_float_or_none(vel[2])
        if len(vel) >= 6:
            vertical_rate_source = vel[5]

    elif isinstance(vel, dict):
        vertical_rate_fpm = _to_float_or_none(
            vel.get("vertical_rate")
            or vel.get("vertical_rate_fpm")
            or vel.get("rocd")
        )
        vertical_rate_source = (
            vel.get("vertical_rate_source")
            or vel.get("vr_source")
            or vel.get("source")
        )

    # Fallbacks if future decoders store explicit top-level keys.
    if vertical_rate_fpm is None:
        vertical_rate_fpm = _to_float_or_none(
            decoded.get("vertical_rate_fpm")
            or decoded.get("baro_rate")
            or decoded.get("rocd")
        )

    if vertical_rate_source is None:
        vertical_rate_source = (
            decoded.get("vertical_rate_source")
            or decoded.get("adsb_vertical_rate_source")
        )

    return {
        "vertical_rate_fpm": vertical_rate_fpm,
        "vertical_rate_source": None if vertical_rate_source is None else str(vertical_rate_source),
    }


def add_adsb_vertical_rate_derived_columns(
    df: pd.DataFrame,
    *,
    decoded_json_column: str = "decoded_json",
) -> pd.DataFrame:
    """Add explicit vertical-rate columns from decoded_json.velocity_raw.

    Intended downstream V-Spline semantics:
      - ``vertical_rate_mps`` can be used as z velocity when finite.
      - no horizontal derivative information is produced here.
    """
    out = df.copy()

    if decoded_json_column not in out.columns:
        for col in DERIVED_ADSB_VERTICAL_RATE_COLUMNS:
            out[col] = None
        return out

    records: list[dict[str, Any]] = []

    for _, row in out.iterrows():
        fields = _extract_vertical_rate_fields(row.get(decoded_json_column))
        vertical_rate_fpm = fields["vertical_rate_fpm"]
        vertical_rate_mps = (
            vertical_rate_fpm * FPM_TO_MPS
            if vertical_rate_fpm is not None
            else None
        )

        records.append(
            {
                "vertical_rate_fpm": vertical_rate_fpm,
                "vertical_rate_mps": vertical_rate_mps,
                "vertical_rate_source": fields["vertical_rate_source"],
                "vertical_rate_available": bool(vertical_rate_mps is not None),
            }
        )

    derived = pd.DataFrame.from_records(records, index=out.index)

    for col in derived.columns:
        out[col] = derived[col]

    return out


def _derived_vertical_rate_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "vertical_rate_row_count": 0,
            "vertical_rate_source_counts": {},
        }

    report: dict[str, Any] = {}

    report["vertical_rate_row_count"] = (
        int(df["vertical_rate_fpm"].notna().sum())
        if "vertical_rate_fpm" in df.columns
        else 0
    )

    report["vertical_rate_available_row_count"] = (
        int(df["vertical_rate_available"].fillna(False).astype(bool).sum())
        if "vertical_rate_available" in df.columns
        else 0
    )

    if "vertical_rate_source" in df.columns:
        counts = (
            df["vertical_rate_source"]
            .fillna("MISSING")
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
        )
        report["vertical_rate_source_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        report["vertical_rate_source_counts"] = {}

    return report


class SqlAdsbLoader:
    """Load raw ADS-B rows for one ICAO from a SQLite dataset."""

    def __init__(self, database_path: str | Path, config: SqlAdsbLoadConfig | None = None) -> None:
        self.database_path = Path(database_path)
        self.config = config or SqlAdsbLoadConfig()

    def load_icao(self, icao: str) -> SqlAdsbLoadResult:
        """Load all raw SQL rows for one ICAO, optionally time/CRC filtered."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"SQL database not found: {self.database_path}")

        cfg = self.config
        logger.info(
            "SQL ICAO load started: {}",
            {
                "database_path": str(self.database_path),
                "icao": str(icao).upper(),
                "table_name": cfg.table_name,
                "include_crc_false": cfg.include_crc_false,
                "first_point_unix": cfg.first_point_unix,
                "last_point_unix": cfg.last_point_unix,
                "derive_vertical_rate_columns": cfg.derive_vertical_rate_columns,
            },
        )

        cc = cfg.columns

        with sqlite3.connect(str(self.database_path)) as conn:
            table, table_columns = _choose_table(
                conn,
                requested=cfg.table_name,
                required_columns=(cc.time_column, cc.icao_column),
            )

            lower_cols = {c.lower(): c for c in table_columns}
            time_col = lower_cols.get(cc.time_column.lower())
            icao_col = lower_cols.get(cc.icao_column.lower())
            crc_col = lower_cols.get(cc.crc_ok_column.lower())
            raw_id_col = lower_cols.get(cc.raw_id_column.lower())

            if time_col is None:
                raise ValueError(f"table {table!r} has no time column {cc.time_column!r}")

            if icao_col is None:
                raise ValueError(f"table {table!r} has no ICAO column {cc.icao_column!r}")

            if not cfg.include_crc_false and crc_col is None:
                raise ValueError(
                    f"include_crc_false=False but table {table!r} has no CRC column "
                    f"{cc.crc_ok_column!r}"
                )

            output_cols = _present_columns(
                table_columns,
                cc.default_output_columns,
                strict=cfg.strict_output_columns,
            )
            if not output_cols:
                output_cols = table_columns

            where = [f"UPPER(CAST({_quote_identifier(icao_col)} AS TEXT)) = UPPER(?)"]
            params: list[Any] = [str(icao)]

            if cfg.first_point_unix is not None:
                where.append(f"{_quote_identifier(time_col)} >= ?")
                params.append(float(cfg.first_point_unix))

            if cfg.last_point_unix is not None:
                where.append(f"{_quote_identifier(time_col)} <= ?")
                params.append(float(cfg.last_point_unix))

            if not cfg.include_crc_false:
                where.append(f"CAST({_quote_identifier(crc_col)} AS INTEGER) = 1")

            order_cols = [time_col]
            if raw_id_col is not None:
                order_cols.append(raw_id_col)

            sql = (
                "SELECT "
                + ", ".join(_quote_identifier(c) for c in output_cols)
                + f" FROM {_quote_identifier(table)}"
                + " WHERE "
                + " AND ".join(where)
                + " ORDER BY "
                + ", ".join(_quote_identifier(c) for c in order_cols)
            )

            df = pd.read_sql_query(sql, conn, params=params)

        # Keep the old extraction shape predictable: same column order, numeric
        # ts_utc when available, sorted by ts_utc/raw_id.
        if time_col in df.columns:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

        if raw_id_col and raw_id_col in df.columns:
            df[raw_id_col] = pd.to_numeric(df[raw_id_col], errors="coerce").astype("Int64")

        if cfg.derive_vertical_rate_columns:
            df = add_adsb_vertical_rate_derived_columns(
                df,
                decoded_json_column="decoded_json",
            )

        first_seen = None
        last_seen = None
        if time_col in df.columns and not df.empty:
            t = pd.to_numeric(df[time_col], errors="coerce").dropna()
            if not t.empty:
                first_seen = float(t.min())
                last_seen = float(t.max())

        derived_vertical_rate_report = (
            _derived_vertical_rate_report(df)
            if cfg.derive_vertical_rate_columns
            else {}
        )

        report = {
            "database_path": str(self.database_path),
            "table_name": table,
            "target_icao": str(icao).upper(),
            "row_count": int(len(df)),
            "first_seen_unix": first_seen,
            "last_seen_unix": last_seen,
            "include_crc_false": bool(cfg.include_crc_false),
            "first_point_unix_filter": cfg.first_point_unix,
            "last_point_unix_filter": cfg.last_point_unix,
            "selected_columns": list(df.columns),
            "sql": sql,
            "loader_config": asdict(cfg),
            "derived_vertical_rate_report": derived_vertical_rate_report,
        }

        result = SqlAdsbLoadResult(dataframe=df, report=report)

        logger.info(
            "SQL ICAO load completed: {}",
            {
                "table_name": table,
                "target_icao": report["target_icao"],
                "row_count": report["row_count"],
                "first_seen_unix": report["first_seen_unix"],
                "last_seen_unix": report["last_seen_unix"],
                "derived_vertical_rate_report": report["derived_vertical_rate_report"],
            },
        )

        return result

    def load_rule(
        self,
        rule: TrackRuleConfig,
        *,
        apply_rule_time_window: bool = False,
        apply_rule_crc: bool = False,
    ) -> SqlAdsbLoadResult:
        """Load rows for a TrackRuleConfig.

        Defaults keep the SQL extract broad: all rows for the ICAO are loaded.
        Set ``apply_rule_time_window`` and/or ``apply_rule_crc`` to push those
        rule filters down into SQL.
        """
        base = self.config
        cfg = SqlAdsbLoadConfig(
            table_name=base.table_name,
            columns=base.columns,
            include_crc_false=not apply_rule_crc,
            first_point_unix=rule.time_window.first_point_unix
            if apply_rule_time_window
            else base.first_point_unix,
            last_point_unix=rule.time_window.last_point_unix
            if apply_rule_time_window
            else base.last_point_unix,
            strict_output_columns=base.strict_output_columns,
            derive_vertical_rate_columns=base.derive_vertical_rate_columns,
        )
        return SqlAdsbLoader(self.database_path, cfg).load_icao(rule.icao)


def load_icao_from_sqlite(
    database_path: str | Path,
    icao: str,
    *,
    config: SqlAdsbLoadConfig | None = None,
) -> SqlAdsbLoadResult:
    """Convenience wrapper for ``SqlAdsbLoader(...).load_icao(...)``."""
    return SqlAdsbLoader(database_path, config).load_icao(icao)
