from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


try:
    import pyModeS as pms
except ImportError as e:
    raise RuntimeError("pyModeS is not installed. Activate venv and run: pip install pyModeS") from e


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _safe_getattr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _to_float(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _to_int(x) -> Optional[int]:
    try:
        return None if x is None else int(x)
    except Exception:
        return None


@dataclass
class CPRPair:
    msg_hex: str
    ts_utc: int


class CPRCache:
    """
    Stores last even/odd CPR frames per ICAO, separately for airborne and surface.
    """
    def __init__(self) -> None:
        self.air: Dict[str, Dict[str, CPRPair]] = {}
        self.sfc: Dict[str, Dict[str, CPRPair]] = {}

    def update(self, store: Dict[str, Dict[str, CPRPair]], icao: str, oe: int, msg_hex: str, ts_utc: int) -> None:
        d = store.setdefault(icao, {})
        d["odd" if oe else "even"] = CPRPair(msg_hex=msg_hex, ts_utc=ts_utc)

    def get_pair(self, store: Dict[str, Dict[str, CPRPair]], icao: str) -> Optional[Tuple[CPRPair, CPRPair]]:
        d = store.get(icao)
        if not d:
            return None
        if "even" in d and "odd" in d:
            return d["even"], d["odd"]
        return None


def _infer_bds(msg_hex: str) -> Optional[str]:
    """
    Comm-B (DF20/21) BDS inference varies across pyModeS versions.
    We'll try a few common locations/names and return string if we get something.
    """
    # candidate modules in different versions
    for mod_name in ("commb", "ehs"):
        mod = _safe_getattr(pms, mod_name)
        if not mod:
            continue
        for fn_name in ("BDS", "bds", "inferBDS", "infer_bds"):
            fn = _safe_getattr(mod, fn_name)
            if callable(fn):
                val = _safe_call(fn, msg_hex)
                if val is not None:
                    return str(val)
    return None


def decode_message(msg_hex: str, ts_utc: int, lat_ref: float, lon_ref: float, cache: CPRCache) -> Dict[str, Any]:
    """
    Best-effort decode for Mode-S/ADS-B frames.
    Returns a dict with standard columns + extra detail.
    """
    d: Dict[str, Any] = {"msg_hex": msg_hex}

    # CRC remainder (0 usually means OK in pyModeS)
    crc = _safe_call(pms.crc, msg_hex, encode=False)
    d["crc"] = _to_int(crc)
    d["crc_ok"] = 1 if _to_int(crc) == 0 else 0 if crc is not None else None

    df = _safe_call(pms.df, msg_hex)
    d["df"] = _to_int(df)

    icao: Optional[str] = None
    tc: Optional[int] = None

    # Try general ICAO if available
    icao_general = _safe_call(_safe_getattr(pms, "icao"), msg_hex) if callable(_safe_getattr(pms, "icao")) else None
    if isinstance(icao_general, str):
        icao = icao_general.upper()

    # ADS-B DF17/18
    if d["df"] in (17, 18):
        adsb = _safe_getattr(pms, "adsb")
        if adsb:
            icao_adsb = _safe_call(adsb.icao, msg_hex)
            if isinstance(icao_adsb, str):
                icao = icao_adsb.upper()

            tc = _safe_call(adsb.typecode, msg_hex)
            tc = _to_int(tc)

            d["icao"] = icao
            d["tc"] = tc

            # Callsign (TC 1-4)
            callsign = None
            if tc is not None and 1 <= tc <= 4:
                callsign = _safe_call(adsb.callsign, msg_hex)
                if isinstance(callsign, str):
                    callsign = callsign.strip() or None

            # Altitude for airborne position TCs
            alt = None
            if tc is not None and (9 <= tc <= 18 or 20 <= tc <= 22):
                alt = _safe_call(adsb.altitude, msg_hex)

            # Position: try with ref (single-frame) first
            lat = lon = None
            if tc is not None and (5 <= tc <= 8 or 9 <= tc <= 18 or 20 <= tc <= 22):
                latlon = _safe_call(adsb.position_with_ref, msg_hex, lat_ref, lon_ref)
                if latlon and isinstance(latlon, (tuple, list)) and len(latlon) == 2:
                    lat, lon = latlon[0], latlon[1]

            # Position: try even/odd pairing when possible
            if icao and tc is not None and (5 <= tc <= 8 or 9 <= tc <= 18 or 20 <= tc <= 22):
                oe = _safe_call(adsb.oe_flag, msg_hex)
                oe_i = _to_int(oe)
                if oe_i in (0, 1):
                    store = cache.sfc if 5 <= tc <= 8 else cache.air
                    cache.update(store, icao, oe_i, msg_hex, ts_utc)
                    pair = cache.get_pair(store, icao)
                    if pair:
                        even, odd = pair
                        latlon2 = _safe_call(
                            adsb.position,
                            even.msg_hex,
                            odd.msg_hex,
                            even.ts_utc,
                            odd.ts_utc,
                            lat_ref=lat_ref,
                            lon_ref=lon_ref,
                        )
                        if latlon2 and isinstance(latlon2, (tuple, list)) and len(latlon2) == 2:
                            lat, lon = latlon2[0], latlon2[1]

            # Velocity (TC 19)
            vel = None
            gs = track = None
            if tc == 19:
                vel = _safe_call(adsb.velocity, msg_hex)
                # vel is sometimes dict-like, sometimes tuple-like depending on pyModeS version/subtype
                if isinstance(vel, dict):
                    gs = vel.get("gs") or vel.get("ground_speed")
                    track = vel.get("track") or vel.get("heading")
                elif isinstance(vel, (tuple, list)):
                    # common: (speed, heading, vertical_rate, "GS"/"TAS"...)
                    if len(vel) >= 2:
                        gs = vel[0]
                        track = vel[1]

            # Squawk/emergency (optional; varies by version)
            squawk = None
            if hasattr(adsb, "squawk"):
                squawk = _safe_call(adsb.squawk, msg_hex)
            if squawk is None and hasattr(adsb, "emergency_squawk"):
                squawk = _safe_call(adsb.emergency_squawk, msg_hex)

            d.update({
                "bds": None,
                "lat": _to_float(lat),
                "lon": _to_float(lon),
                "alt": _to_int(alt),
                "gs": _to_float(gs),
                "track": _to_float(track),
                "callsign": callsign,
                "squawk": squawk if isinstance(squawk, str) else None,
                "velocity_raw": vel,  # keep full object in JSON
            })
            return d

    # Mode-S ELS / Comm-B (DF4/5/20/21) best-effort
    icao_top = icao or None
    if icao_top is None and callable(_safe_getattr(pms, "icao")):
        v = _safe_call(pms.icao, msg_hex)
        if isinstance(v, str):
            icao_top = v.upper()

    d["icao"] = icao_top
    d["tc"] = None

    if d["df"] in (20, 21):
        d["bds"] = _infer_bds(msg_hex)
    else:
        d["bds"] = None

    # Try some common helpers if present in your pyModeS version
    # (we keep it safe, and still store everything in decoded_json)
    d["callsign"] = None
    d["squawk"] = None
    d["alt"] = None
    d["lat"] = None
    d["lon"] = None
    d["gs"] = None
    d["track"] = None

    # Keep raw decoded details for later analysis
    return d


class PyModeSDecoder:
    """
    Stateful decoder (keeps CPR cache across messages).
    """
    def __init__(self, lat_ref: float, lon_ref: float) -> None:
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref
        self.cache = CPRCache()

    def decode(self, msg_hex: str, ts_utc: int) -> Dict[str, Any]:
        return decode_message(msg_hex, ts_utc, self.lat_ref, self.lon_ref, self.cache)

    @staticmethod
    def to_json(decoded: Dict[str, Any]) -> str:
        # ensure everything is JSON-able
        return json.dumps(decoded, separators=(",", ":"), ensure_ascii=False, default=str)
