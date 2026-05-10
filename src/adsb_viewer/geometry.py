from __future__ import annotations

import math
from typing import Any, Optional


EARTH_RADIUS_M = 6_371_000.0


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(result) or math.isinf(result):
        return None

    return result


def heading_from_velocity(
    east_mps: Optional[float],
    north_mps: Optional[float],
) -> Optional[float]:
    if east_mps is None or north_mps is None:
        return None

    if math.hypot(east_mps, north_mps) < 1e-6:
        return None

    return math.degrees(math.atan2(east_mps, north_mps)) % 360.0


def offset_lat_lon(
    lat_deg: float,
    lon_deg: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * max(1e-9, math.cos(lat_rad)))

    lat2 = lat_deg + math.degrees(dlat)
    lon2 = lon_deg + math.degrees(dlon)

    return lat2, lon2


def lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def lerp_optional(
    a: Optional[float],
    b: Optional[float],
    alpha: float,
) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + (b - a) * alpha
