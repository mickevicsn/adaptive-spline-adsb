# src/adsb_collector/ingest/beast_parser.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

ESC = 0x1A

# Beast message types (ASCII '1','2','3')
T_MODEAC = 0x31  # "1" -> Mode A/C (2 bytes)
T_SHORT  = 0x32  # "2" -> Mode-S short frame (7 bytes)
T_LONG   = 0x33  # "3" -> Mode-S long frame (14 bytes)

TYPE_TO_PAYLOAD_LEN = {
    T_MODEAC: 2,
    T_SHORT: 7,
    T_LONG: 14,
}


@dataclass(frozen=True)
class BeastFrame:
    """
    One Beast frame parsed from the stream.

    - msgtype: 0x31/0x32/0x33 (type '1','2','3')
    - mlat48: 48-bit timestamp from Beast (6 bytes)
    - signal: 0..255 (1 byte, "RSSI-like")
    - payload: Mode A/C (2B) or Mode-S (7B/14B)
    """
    msgtype: int
    mlat48: int
    signal: int
    payload: bytes

    def payload_hex(self) -> str:
        return self.payload.hex().upper()

    def df(self) -> Optional[int]:
        # For Mode-S (7/14 bytes), DF is top 5 bits of first byte
        if self.msgtype not in (T_SHORT, T_LONG) or not self.payload:
            return None
        return self.payload[0] >> 3

    def icao(self) -> Optional[str]:
        # Many Mode-S DFs contain ICAO addr at bytes 1..3 (best-effort)
        if self.msgtype not in (T_SHORT, T_LONG) or len(self.payload) < 4:
            return None
        return self.payload[1:4].hex().upper()


class BeastStreamParser:
    """
    Parse Mode-S Beast escaped binary messages from a TCP stream.

    Message on the wire (escaped):
      0x1a <type> <escaped remainder>

    Remainder (unescaped) length:
      6-byte timestamp + 1-byte signal + payload_len(type)

    Escaping:
      A literal 0x1a in the remainder is encoded as 0x1a 0x1a.

    This parser is designed to:
      - handle arbitrary chunk boundaries
      - resync on bad data
      - return as many complete frames as possible per feed()
    """
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> List[BeastFrame]:
        frames: List[BeastFrame] = []
        if not chunk:
            return frames

        self._buf.extend(chunk)

        while True:
            # find start ESC
            try:
                i = self._buf.index(ESC)
            except ValueError:
                # no ESC found; keep buffer from growing forever
                if len(self._buf) > 4096:
                    del self._buf[:-1]
                break

            # need type byte
            if i + 1 >= len(self._buf):
                if i > 0:
                    del self._buf[:i]
                break

            t = self._buf[i + 1]

            # ESC ESC outside a message start: drop one ESC and continue
            if t == ESC:
                del self._buf[:i + 1]
                continue

            if t not in TYPE_TO_PAYLOAD_LEN:
                # not a known message type; drop up to after ESC and keep scanning
                del self._buf[:i + 1]
                continue

            need_unescaped = 6 + 1 + TYPE_TO_PAYLOAD_LEN[t]
            rem = bytearray()
            j = i + 2  # read position in escaped buffer, after ESC+type

            # build unescaped remainder
            while len(rem) < need_unescaped:
                if j >= len(self._buf):
                    # not enough data yet; keep from ESC start
                    if i > 0:
                        del self._buf[:i]
                    return frames

                b = self._buf[j]
                if b != ESC:
                    rem.append(b)
                    j += 1
                    continue

                # b is ESC; expect escaped ESC (ESC ESC)
                if j + 1 >= len(self._buf):
                    if i > 0:
                        del self._buf[:i]
                    return frames

                b2 = self._buf[j + 1]
                if b2 == ESC:
                    rem.append(ESC)
                    j += 2
                    continue

                # unexpected ESC + non-ESC in middle: desync, rescan from this ESC
                del self._buf[:j]
                break
            else:
                # got complete remainder
                ts6 = rem[0:6]
                sig = rem[6]
                payload = bytes(rem[7:])

                mlat48 = int.from_bytes(ts6, "big", signed=False)
                frames.append(BeastFrame(
                    msgtype=t,
                    mlat48=mlat48,
                    signal=int(sig),
                    payload=payload,
                ))

                # remove consumed bytes (ESC..end-of-message)
                del self._buf[:j]
                continue

            # if we broke due to desync, keep looping to find next ESC
            continue

        return frames
