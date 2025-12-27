from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import redis

from adsb_collector.settings import load_settings
from adsb_collector.ingest.beast_parser import (
    BeastStreamParser,
    T_MODEAC,
    T_SHORT,
    T_LONG,
    BeastFrame,
)

SETTINGS = load_settings()

def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _open_spool(spool_dir: Path, day: str):
    spool_dir.mkdir(parents=True, exist_ok=True)
    p = spool_dir / f"beast_{day}.bin"
    # large buffer reduces SD card writes
    return open(p, "ab", buffering=1024 * 1024)


def _spool_write(fh, ts_utc: int, frame: BeastFrame) -> None:
    """
    Minimal binary record format (fast append):
      ts_utc (4 bytes BE)
      mlat48 (6 bytes BE)
      signal (1 byte)
      msgtype (1 byte)
      payload_len (1 byte)
      payload (N bytes)
    """
    payload = frame.payload
    fh.write(ts_utc.to_bytes(4, "big"))
    fh.write(frame.mlat48.to_bytes(6, "big"))
    fh.write(bytes([frame.signal & 0xFF, frame.msgtype & 0xFF, len(payload) & 0xFF]))
    fh.write(payload)


def _connect_beast(host: str, port: int) -> socket.socket:
    s = socket.create_connection((host, port), timeout=5)
    s.settimeout(2.0)
    return s


def main() -> int:
    stop_evt = threading.Event()

    def _stop(*_):
        stop_evt.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Redis: keep bytes in/out
    r = redis.Redis.from_url(SETTINGS.redis_url, decode_responses=False)

    parser = BeastStreamParser()

    spool_fh = None
    spool_day = None
    if SETTINGS.spool_dir:
        SETTINGS.spool_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[producer] beast={SETTINGS.beast_host}:{SETTINGS.beast_port} "
        f"redis={SETTINGS.redis_url} stream={SETTINGS.stream_key}"
    )
    print(f"[producer] Mode-S only = {not SETTINGS.include_modeac}")
    print(f"[producer] spool_dir = {SETTINGS.spool_dir if SETTINGS.spool_dir else '(disabled)'}")

    while not stop_evt.is_set():
        try:
            sock = _connect_beast(SETTINGS.beast_host, SETTINGS.beast_port)
            print("[producer] connected to Beast")
        except Exception as e:
            print(f"[producer] connect failed: {e} (retrying)", file=sys.stderr)
            time.sleep(2)
            continue

        with sock:
            pipe = r.pipeline(transaction=False)
            pending = 0
            last_flush = time.time()

            while not stop_evt.is_set():
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("socket closed")
                except Exception as e:
                    print(f"[producer] recv error: {e} (reconnecting)", file=sys.stderr)
                    break

                frames = parser.feed(chunk)
                if not frames:
                    continue

                ts_utc = int(time.time())

                # rotate spool daily (UTC)
                if SETTINGS.spool_dir:
                    d = _utc_day()
                    if spool_fh is None or spool_day != d:
                        try:
                            if spool_fh:
                                spool_fh.flush()
                                spool_fh.close()
                        except Exception:
                            pass
                        spool_day = d
                        spool_fh = _open_spool(SETTINGS.spool_dir, d)

                for fr in frames:
                    # Keep only Mode-S (Beast types 2/3) unless include_modeac is enabled
                    if fr.msgtype == T_MODEAC and not SETTINGS.include_modeac:
                        continue
                    if fr.msgtype not in (T_SHORT, T_LONG, T_MODEAC):
                        continue

                    # Disk spool first (best-effort “no loss” even if Redis is down)
                    if spool_fh:
                        try:
                            _spool_write(spool_fh, ts_utc, fr)
                        except Exception as e:
                            print(f"[producer] WARN spool write failed: {e}", file=sys.stderr)

                    # Prepare stream fields
                    # Store raw payload bytes + a few convenience fields.
                    fields: Dict[bytes, bytes] = {
                        b"ts": str(ts_utc).encode(),
                        b"mlat48": str(fr.mlat48).encode(),
                        b"sig": str(fr.signal).encode(),
                        b"t": bytes([fr.msgtype]),     # one byte
                        b"payload": fr.payload,        # raw bytes (7/14 for Mode-S)
                    }

                    df = fr.df()
                    icao = fr.icao()
                    if df is not None:
                        fields[b"df"] = str(df).encode()
                    if icao is not None:
                        fields[b"icao"] = icao.encode()

                    try:
                        pipe.xadd(SETTINGS.stream_key.encode(), fields=fields)
                        pending += 1
                    except Exception as e:
                        # If Redis is down, spool still holds data (if enabled)
                        print(f"[producer] WARN redis xadd failed: {e}", file=sys.stderr)

                    # flush pipeline periodically
                    if pending >= SETTINGS.producer_pipe:
                        try:
                            pipe.execute()
                        except Exception as e:
                            print(f"[producer] WARN redis pipeline execute failed: {e}", file=sys.stderr)
                        pending = 0
                        last_flush = time.time()

                # time-based flush too
                if pending and (time.time() - last_flush) > 1.0:
                    try:
                        pipe.execute()
                    except Exception as e:
                        print(f"[producer] WARN redis pipeline execute failed: {e}", file=sys.stderr)
                    pending = 0
                    last_flush = time.time()

    # cleanup
    try:
        if spool_fh:
            spool_fh.flush()
            spool_fh.close()
    except Exception:
        pass

    print("[producer] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
