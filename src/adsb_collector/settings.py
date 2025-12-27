# src/adsb_collector/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None or v == "" else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None or v == "" else float(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    # Redis / stream
    redis_url: str
    stream_key: str

    # Beast input
    beast_host: str
    beast_port: int

    # Storage
    db_path: Path
    spool_dir: Optional[Path]

    # Decode reference (receiver location)
    lat_ref: float
    lon_ref: float

    # Runtime tuning
    flush_seconds: int
    producer_pipe: int
    include_modeac: bool

    # Decoder tuning
    decode_batch: int
    decode_sleep: float

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.spool_dir:
            self.spool_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not (-90.0 <= self.lat_ref <= 90.0):
            raise ValueError(f"LAT_REF out of range: {self.lat_ref}")
        if not (-180.0 <= self.lon_ref <= 180.0):
            raise ValueError(f"LON_REF out of range: {self.lon_ref}")
        if self.beast_port <= 0 or self.beast_port > 65535:
            raise ValueError(f"Invalid BEAST_PORT: {self.beast_port}")
        if self.flush_seconds < 1:
            raise ValueError(f"FLUSH_SECONDS must be >= 1, got {self.flush_seconds}")


def load_settings() -> Settings:
    spool_dir_raw = os.getenv("SPOOL_DIR", "").strip()
    spool_dir = Path(spool_dir_raw) if spool_dir_raw else None

    s = Settings(
        redis_url=_env("REDIS_URL", "redis://127.0.0.1:6379/0"),
        stream_key=_env("STREAM_KEY", "adsb:raw"),

        beast_host=_env("BEAST_HOST", "127.0.0.1"),
        beast_port=_env_int("BEAST_PORT", 30005),

        db_path=Path(_env("DB_PATH", str(Path.home() / "adsb-collector" / "data" / "adsb_raw.sqlite"))),
        spool_dir=spool_dir,

        lat_ref=_env_float("LAT_REF", 56.9236),
        lon_ref=_env_float("LON_REF", 23.9711),

        flush_seconds=_env_int("FLUSH_SECONDS", 60),
        producer_pipe=_env_int("PRODUCER_PIPE", 500),
        include_modeac=_env_bool("INCLUDE_MODEAC", False),

        decode_batch=_env_int("DECODE_BATCH", 5000),
        decode_sleep=_env_float("DECODE_SLEEP", 1.0),
    )

    s.validate()
    s.ensure_dirs()
    return s


SETTINGS = load_settings()