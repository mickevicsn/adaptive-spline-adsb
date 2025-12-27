from __future__ import annotations

import os
import time
import sys
import sqlite3
from typing import Any, Dict, List, Tuple, Optional

import redis

from adsb_collector.settings import load_settings
from adsb_collector.db import connect_sqlite, init_schema, insert_raw_batch

SETTINGS = load_settings()

def _b2i(x: bytes) -> int:
    return int(x.decode("ascii", errors="strict"))


def _b2s(x: bytes) -> str:
    return x.decode("ascii", errors="strict")


def _maybe_int_field(fields: Dict[bytes, bytes], key: bytes) -> Optional[int]:
    if key not in fields:
        return None
    try:
        return _b2i(fields[key])
    except Exception:
        return None


def _maybe_str_field(fields: Dict[bytes, bytes], key: bytes) -> Optional[str]:
    if key not in fields:
        return None
    try:
        return _b2s(fields[key])
    except Exception:
        return None


def _rows_from_messages(messages: List[Tuple[bytes, Dict[bytes, bytes]]]) -> Tuple[List[Tuple[Any, ...]], List[bytes]]:
    """
    Convert Redis stream messages to DB rows.
    Returns (rows, ids)
    Row format for insert_raw_batch:
      (ts_utc, mlat48, signal, beast_type, df, icao, payload_bytes, payload_hex)
    """
    rows: List[Tuple[Any, ...]] = []
    ids: List[bytes] = []

    for msgid, fields in messages:
        ids.append(msgid)

        try:
            ts_utc = _b2i(fields[b"ts"])
            mlat48 = _b2i(fields[b"mlat48"])
            signal = _b2i(fields[b"sig"])
            beast_type = fields[b"t"][0]  # one byte
            payload = fields[b"payload"]  # raw bytes

            df = _maybe_int_field(fields, b"df")
            icao = _maybe_str_field(fields, b"icao")

            rows.append((
                ts_utc,
                mlat48,
                signal,
                beast_type,
                df,
                icao,
                sqlite3.Binary(payload),
                payload.hex().upper(),
            ))
        except Exception:
            # Malformed message: we still ACK it (so it doesn't block the consumer group)
            # but we don't insert it.
            continue

    return rows, ids


def main() -> int:
    redis_url = SETTINGS.redis_url
    stream = SETTINGS.stream_key
    group = os.getenv("REDIS_GROUP", "adsb-sql")
    consumer = os.getenv("REDIS_CONSUMER", "c1")

    # Optional: keep Redis from growing forever (safe because SQLite is the durable store)
    # Set e.g. TRIM_MAXLEN=500000 to keep ~last 500k entries.
    trim_maxlen = int(os.getenv("TRIM_MAXLEN", "0"))

    flush_seconds = SETTINGS.flush_seconds
    read_count = int(os.getenv("READ_COUNT", "5000"))
    block_ms = int(os.getenv("BLOCK_MS", "2000"))
    max_batch = int(os.getenv("MAX_BATCH", "200000"))

    # Pending reclaim (if flusher crashed after reading but before ACK)
    # Claim messages idle > this many ms
    claim_idle_ms = int(os.getenv("CLAIM_IDLE_MS", "60000"))

    r = redis.Redis.from_url(redis_url, decode_responses=False)

    # Create consumer group (OK if exists)
    try:
        r.xgroup_create(stream.encode(), group.encode(), id=b"0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    conn = connect_sqlite(SETTINGS.db_path)
    init_schema(conn)

    print(f"[flusher] redis={redis_url} stream={stream} group={group}/{consumer}")
    print(f"[flusher] db={SETTINGS.db_path} flush={flush_seconds}s read_count={read_count} block={block_ms}ms")
    if trim_maxlen:
        print(f"[flusher] redis trim enabled ~maxlen={trim_maxlen}")
    else:
        print("[flusher] redis trim disabled")

    buffer_rows: List[Tuple[Any, ...]] = []
    buffer_ids: List[bytes] = []
    last_flush = time.time()

    # For XAUTOCLAIM scanning
    claim_start_id = b"0-0"

    while True:
        # 1) Try to reclaim pending messages occasionally (best-effort)
        try:
            # redis-py supports xautoclaim on newer Redis. If not available, it will raise.
            next_id, claimed = r.xautoclaim(
                stream.encode(),
                group.encode(),
                consumer.encode(),
                min_idle_time=claim_idle_ms,
                start_id=claim_start_id,
                count=read_count,
            )
            claim_start_id = next_id if isinstance(next_id, (bytes, bytearray)) else claim_start_id

            if claimed:
                rows, ids = _rows_from_messages(claimed)
                buffer_rows.extend(rows)
                buffer_ids.extend(ids)
        except Exception:
            # Ignore if Redis/command not available; normal XREADGROUP still works.
            pass

        # 2) Read new messages
        try:
            resp = r.xreadgroup(
                groupname=group.encode(),
                consumername=consumer.encode(),
                streams={stream.encode(): b">"},
                count=read_count,
                block=block_ms,
            )
        except Exception as e:
            print(f"[flusher] WARN xreadgroup failed: {e}", file=sys.stderr)
            time.sleep(1.0)
            continue

        if resp:
            # resp: [(stream_name, [(id, fields), ...])]
            for _stream_name, msgs in resp:
                rows, ids = _rows_from_messages(msgs)
                buffer_rows.extend(rows)
                buffer_ids.extend(ids)

        now = time.time()
        should_flush = (
            buffer_ids
            and (now - last_flush >= flush_seconds or len(buffer_rows) >= max_batch or len(buffer_ids) >= max_batch)
        )

        if should_flush:
            try:
                # Write rows (if any)
                if buffer_rows:
                    insert_raw_batch(conn, buffer_rows)

                # ACK everything we read (even malformed ones) AFTER DB commit succeeded
                r.xack(stream.encode(), group.encode(), *buffer_ids)

                # Optionally trim the stream to keep Redis bounded
                if trim_maxlen:
                    try:
                        r.xtrim(stream.encode(), maxlen=trim_maxlen, approximate=True)
                    except Exception:
                        pass

                print(f"[flusher] flushed rows={len(buffer_rows)} acked={len(buffer_ids)}")
                buffer_rows.clear()
                buffer_ids.clear()
                last_flush = now
            except Exception as e:
                # Do NOT ACK on failure; messages stay pending and will be retried/claimed
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                print(f"[flusher] ERROR flush failed (not acking, will retry): {e}", file=sys.stderr)
                time.sleep(1.0)

    # unreachable
    # return 0


if __name__ == "__main__":
    raise SystemExit(main())
