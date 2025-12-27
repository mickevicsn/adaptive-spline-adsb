from __future__ import annotations

import time
import sys
from typing import Any, List, Tuple

from adsb_collector.settings import load_settings
from adsb_collector.db import (
    connect_sqlite,
    init_schema,
    insert_parsed_batch,
    get_last_parsed_raw_id,
)
from adsb_collector.decode.decode_pymodes import PyModeSDecoder

SETTINGS = load_settings()

def main() -> int:
    conn = connect_sqlite(SETTINGS.db_path)
    init_schema(conn)

    dec = PyModeSDecoder(lat_ref=SETTINGS.lat_ref, lon_ref=SETTINGS.lon_ref)

    batch = SETTINGS.decode_batch
    sleep_s = SETTINGS.decode_sleep

    print(f"[decoder] db={SETTINGS.db_path}")
    print(f"[decoder] lat_ref={SETTINGS.lat_ref} lon_ref={SETTINGS.lon_ref}")
    print(f"[decoder] batch={batch} sleep={sleep_s}s")

    while True:
        last_id = get_last_parsed_raw_id(conn)

        rows = conn.execute(
            """
            SELECT id, ts_utc, payload_hex
            FROM raw_messages
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (last_id, batch),
        ).fetchall()

        if not rows:
            time.sleep(sleep_s)
            continue

        out: List[Tuple[Any, ...]] = []

        for r in rows:
            raw_id = int(r["id"])
            ts_utc = int(r["ts_utc"])
            msg_hex = r["payload_hex"]

            decoded = dec.decode(msg_hex, ts_utc)
            decoded_json = dec.to_json(decoded)

            out.append((
                raw_id,
                ts_utc,
                decoded.get("icao"),
                decoded.get("df"),
                decoded.get("tc"),
                decoded.get("bds"),
                decoded.get("lat"),
                decoded.get("lon"),
                decoded.get("alt"),
                decoded.get("gs"),
                decoded.get("track"),
                decoded.get("callsign"),
                decoded.get("squawk"),
                decoded.get("crc"),
                decoded.get("crc_ok"),
                decoded_json,
            ))

        try:
            insert_parsed_batch(conn, out)
            print(f"[decoder] parsed={len(out)} raw_id {out[0][0]}..{out[-1][0]}")
        except Exception as e:
            print(f"[decoder] ERROR insert failed: {e}", file=sys.stderr)
            time.sleep(1.0)

    # unreachable
    # return 0


if __name__ == "__main__":
    raise SystemExit(main())
