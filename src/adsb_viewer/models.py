from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FlightSample:
    t: float
    t_rel_s: float
    lon: float
    lat: float
    z_m: float
    x_m: float
    y_m: float
    heading_deg: float
    ground_speed_kt: Optional[float]
    vel_east_mps: Optional[float]
    vel_north_mps: Optional[float]
    vel_up_mps: Optional[float]
    accel_east_mps2: Optional[float]
    accel_north_mps2: Optional[float]
    accel_up_mps2: Optional[float]
    accel_horizontal_mps2: Optional[float]
    accel_total_mps2: Optional[float]
    altitude_ft_msl: Optional[float]
