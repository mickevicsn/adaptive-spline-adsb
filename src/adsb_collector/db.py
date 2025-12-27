# src/adsb_collector/db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Any


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """
    Create a SQLite connection with performance settings suitable for high-write workloads.
    """
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)  # autocommit mode
    conn.row_factory = sqlite3.Row

    # Performance + durability tradeoffs:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")

    # Optional tuning (safe defaults):
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB cache if RAM allows (negative = KB units)
    conn.execute("PRAGMA busy_timeout=30000;")  # ms

    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create tables/indexes if they don't exist.
    Raw table stores every Mode-S frame (Beast types 2/3).
    Parsed table stores decoded fields + JSON blob, keyed by raw_id.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Every raw Mode-S / ADS-B frame we captured
        CREATE TABLE IF NOT EXISTS raw_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc INTEGER NOT NULL,          -- unix seconds (receiver time)
            mlat48 INTEGER NOT NULL,          -- 48-bit beast timestamp
            signal INTEGER NOT NULL,          -- beast RSSI-like 0..255
            beast_type INTEGER NOT NULL,      -- 0x32 (short) or 0x33 (long) typically
            df INTEGER,                       -- downlink format (optional)
            icao TEXT,                        -- ICAO address (optional)
            payload BLOB NOT NULL,            -- 7 or 14 bytes (Mode-S)
            payload_hex TEXT NOT NULL         -- hex string (for easy tooling)
        );

        CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_messages(ts_utc);
        CREATE INDEX IF NOT EXISTS idx_raw_icao_ts ON raw_messages(icao, ts_utc);

        -- Parsed/decoded interpretation (best effort)
        CREATE TABLE IF NOT EXISTS parsed_messages (
            raw_id INTEGER PRIMARY KEY,       -- 1:1 with raw_messages.id
            ts_utc INTEGER NOT NULL,
            icao TEXT,
            df INTEGER,
            tc INTEGER,                       -- ADS-B typecode (DF17/18)
            bds TEXT,                         -- Comm-B BDS inference (DF20/21), if known

            lat REAL,
            lon REAL,
            alt INTEGER,
            gs REAL,
            track REAL,
            callsign TEXT,
            squawk TEXT,

            crc INTEGER,
            crc_ok INTEGER,

            decoded_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_parsed_ts ON parsed_messages(ts_utc);
        CREATE INDEX IF NOT EXISTS idx_parsed_icao_ts ON parsed_messages(icao, ts_utc);
        """
    )

    # Track schema version
    conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1');")


def insert_raw_batch(
    conn: sqlite3.Connection,
    rows: Sequence[Tuple[Any, ...]],
) -> None:
    """
    Insert many raw messages in one transaction.
    rows item format:
      (ts_utc, mlat48, signal, beast_type, df, icao, payload_bytes, payload_hex)
    """
    if not rows:
        return
    conn.execute("BEGIN;")
    conn.executemany(
        """
        INSERT INTO raw_messages
        (ts_utc, mlat48, signal, beast_type, df, icao, payload, payload_hex)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    conn.execute("COMMIT;")


def insert_parsed_batch(
    conn: sqlite3.Connection,
    rows: Sequence[Tuple[Any, ...]],
) -> None:
    """
    Insert many parsed rows (idempotent via PRIMARY KEY raw_id).
    rows item format:
      (raw_id, ts_utc, icao, df, tc, bds, lat, lon, alt, gs, track, callsign, squawk, crc, crc_ok, decoded_json)
    """
    if not rows:
        return
    conn.execute("BEGIN;")
    conn.executemany(
        """
        INSERT OR IGNORE INTO parsed_messages
        (raw_id, ts_utc, icao, df, tc, bds, lat, lon, alt, gs, track, callsign, squawk, crc, crc_ok, decoded_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    conn.execute("COMMIT;")


def get_last_parsed_raw_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(raw_id), 0) AS m FROM parsed_messages;").fetchone()
    return int(row["m"]) if row else 0
