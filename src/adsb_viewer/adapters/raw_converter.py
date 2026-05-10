from __future__ import annotations

import math
from typing import Any, Optional

from .base import AdapterContext, NormalizedTrack
from ..geometry import as_float, heading_from_velocity
from ..models import FlightSample

KNOT_TO_MPS = 0.5144444444444445
MPS_TO_KNOT = 1.0 / KNOT_TO_MPS


class RawConverterJsonAdapter:
    """Adapter for JsonTrackConverter-style method files.

    This adapter contains the current raw ADS-B/converter JSON assumptions. It
    is intentionally isolated from the renderer and from future reconstruction
    methods. Future methods may either emit this same converter-style schema or
    add a new adapter without changing Cesium/JS rendering code.
    """

    adapter_id = "json_track_converter"
    label = "JsonTrackConverter output"

    def can_load(self, data: dict[str, Any], _context: AdapterContext) -> bool:
        return isinstance(data.get("raw_keyframes"), list)

    def normalize(self, data: dict[str, Any], context: AdapterContext) -> NormalizedTrack:
        raw_position_points = self._extract_raw_position_points(data)
        if not raw_position_points:
            raise RuntimeError("No usable raw position events found in raw_events or raw_keyframes.")

        sparse_samples = self._extract_render_timeline(data, context)
        if not sparse_samples:
            sparse_samples = self._extract_animation_timeline(data, raw_position_points, context)

        if not sparse_samples:
            raise RuntimeError("No usable render samples found in the selected method JSON.")

        track_id = str(
            data.get("track_id")
            or data.get("flightId")
            or data.get("icao")
            or context.source_path.stem
            or "track"
        )

        return NormalizedTrack(
            track_json=data,
            raw_position_points=raw_position_points,
            samples=sparse_samples,
            track_id=track_id,
            source_schema=self.adapter_id,
            interpolate_samples=True,
            metadata={
                "adapterId": self.adapter_id,
                "methodId": context.method_id,
                "methodLabel": context.method_label,
            },
        )

    def _extract_raw_position_points(self, track_json: dict[str, Any]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []

        raw_keyframes = track_json.get("raw_keyframes") or []
        for kf in raw_keyframes:
            position = kf.get("position")
            t = as_float(kf.get("t"))
            if not position or t is None:
                continue

            row_ids = kf.get("row_ids") or [None]
            point = self._position_payload_to_point(
                position=position,
                t=t,
                row_id=row_ids[0],
                source_id=kf.get("id"),
                kind=kf.get("event_kind", "keyframe"),
            )
            if point:
                points.append(point)

        return self._sort_and_dedupe_points(points)

    def _position_payload_to_point(
        self,
        position: dict[str, Any],
        t: Optional[float],
        row_id: Any,
        source_id: Any,
        kind: Any,
    ) -> Optional[dict[str, Any]]:
        lat = as_float(position.get("lat"))
        lon = as_float(position.get("lon"))

        if lat is None or lon is None:
            return None

        z_m = as_float(position.get("z_m")) or 0.0

        return {
            "t": t,
            "row_id": row_id,
            "source_id": source_id,
            "kind": kind,
            "lon": lon,
            "lat": lat,
            "z_m": z_m,
            "x_m": as_float(position.get("x_m")) or 0.0,
            "y_m": as_float(position.get("y_m")) or 0.0,
            "altitude_ft_msl": as_float(position.get("altitude_ft_msl")),
            "height_above_field_m": as_float(position.get("height_above_field_m")),
        }

    @staticmethod
    def _sort_and_dedupe_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points = sorted(points, key=lambda p: (p["t"], p.get("row_id") or 0))

        result: list[dict[str, Any]] = []
        seen: set[tuple[float, float, float, float]] = set()

        for p in points:
            key = (
                round(p["t"], 3),
                round(p["lon"], 8),
                round(p["lat"], 8),
                round(p["z_m"], 2),
            )
            if key in seen:
                continue

            seen.add(key)
            result.append(p)

        return result

    def _extract_render_timeline(self, track_json: dict[str, Any], context: AdapterContext) -> list[FlightSample]:
        """Use viewer-ready render_keyframes when the converter provides them."""
        render_keyframes = track_json.get("render_keyframes") or []
        if not render_keyframes:
            return []

        velocity_events = [
            payload
            for payload in (self._velocity_payload_from_keyframe(kf) for kf in render_keyframes)
            if payload is not None
        ]
        acceleration_events = [
            payload
            for payload in (self._acceleration_payload_from_keyframe(kf) for kf in render_keyframes)
            if payload is not None
        ]
        velocity_events.sort(key=lambda e: e["t"])
        acceleration_events.sort(key=lambda e: e["t"])

        samples: list[FlightSample] = []
        first_t: Optional[float] = None

        for index, kf in enumerate(render_keyframes):
            position = kf.get("position")
            t = as_float(kf.get("t"))
            if not position or t is None:
                continue

            row_ids = kf.get("row_ids") or [None]
            point = self._position_payload_to_point(
                position=position,
                t=t,
                row_id=row_ids[0],
                source_id=kf.get("id"),
                kind=kf.get("event_kind", "render_keyframe"),
            )
            if not point:
                continue

            if first_t is None:
                first_t = t

            velocity = self._velocity_payload_from_keyframe(kf)
            if velocity is None:
                velocity = self._nearest_recent_payload(
                    payloads=velocity_events,
                    current_t=t,
                    max_age_s=context.config.max_carried_velocity_age_s,
                )

            acceleration = self._nearest_recent_payload(
                payloads=acceleration_events,
                current_t=t,
                max_age_s=context.config.max_carried_acceleration_age_s,
            )

            vel_east = as_float((velocity or {}).get("east_mps"))
            vel_north = as_float((velocity or {}).get("north_mps"))
            vel_up = as_float((velocity or {}).get("up_mps"))
            ground_speed_kt = as_float((velocity or {}).get("ground_speed_kt"))
            accel_east = as_float((acceleration or {}).get("east_mps2"))
            accel_north = as_float((acceleration or {}).get("north_mps2"))
            accel_up = as_float((acceleration or {}).get("up_mps2"))
            accel_horizontal = as_float((acceleration or {}).get("horizontal_mps2"))
            accel_total = as_float((acceleration or {}).get("total_mps2"))

            heading = as_float((velocity or {}).get("track_deg"))
            if heading is None:
                heading = heading_from_velocity(vel_east, vel_north)
            if heading is None:
                heading = self._heading_from_render_neighbors(render_keyframes, index)

            if ground_speed_kt is None and vel_east is not None and vel_north is not None:
                ground_speed_kt = math.hypot(vel_east, vel_north) * MPS_TO_KNOT

            if accel_horizontal is None and accel_east is not None and accel_north is not None:
                accel_horizontal = math.hypot(accel_east, accel_north)
            if accel_total is None and (accel_east is not None or accel_north is not None or accel_up is not None):
                accel_total = math.sqrt((accel_east or 0.0) ** 2 + (accel_north or 0.0) ** 2 + (accel_up or 0.0) ** 2)

            samples.append(
                FlightSample(
                    t=t,
                    t_rel_s=t - first_t,
                    lon=point["lon"],
                    lat=point["lat"],
                    z_m=point["z_m"],
                    x_m=point["x_m"],
                    y_m=point["y_m"],
                    heading_deg=heading,
                    ground_speed_kt=ground_speed_kt,
                    vel_east_mps=vel_east,
                    vel_north_mps=vel_north,
                    vel_up_mps=vel_up,
                    accel_east_mps2=accel_east,
                    accel_north_mps2=accel_north,
                    accel_up_mps2=accel_up,
                    accel_horizontal_mps2=accel_horizontal,
                    accel_total_mps2=accel_total,
                    altitude_ft_msl=point.get("altitude_ft_msl"),
                )
            )

        return samples

    def _velocity_payload_from_keyframe(self, kf: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Build velocity for a converter keyframe.

        Prefer explicit ENU components when the converter already emitted them.
        Fall back to ADS-B ground speed + track only when components are absent.
        """
        t = as_float(kf.get("t"))
        if t is None:
            return None

        velocity = kf.get("velocity") or {}
        if not isinstance(velocity, dict):
            return None

        return self._velocity_payload_from_payloads(t=t, primary=velocity, fallback=kf)

    def _velocity_payload_from_payloads(
        self,
        t: float,
        primary: dict[str, Any],
        fallback: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Normalize velocity from one or two payload dictionaries.

        ``primary`` is normally the nested ``velocity`` object. ``fallback`` is
        normally the keyframe itself, kept for older converter files that placed
        speed/track fields at the keyframe level.
        """
        fallback = fallback or {}

        east = self._first_numeric(primary, ["east_mps", "vel_east_mps", "vx_mps"])
        north = self._first_numeric(primary, ["north_mps", "vel_north_mps", "vy_mps"])
        if east is None:
            east = self._first_numeric(fallback, ["east_mps", "vel_east_mps", "vx_mps"])
        if north is None:
            north = self._first_numeric(fallback, ["north_mps", "vel_north_mps", "vy_mps"])

        # Use already-normalized vertical velocity components directly.
        # Do not infer or convert this from other vertical-rate fields.
        up = self._extract_vertical_velocity_mps(primary)
        if up is None:
            up = self._extract_vertical_velocity_mps(fallback)

        ground_speed_kt = self._first_numeric(
            primary,
            ["ground_speed_kt", "gs_kt", "speed_kt", "speed_knots", "gs"],
        )
        if ground_speed_kt is None:
            ground_speed_kt = self._first_numeric(
                fallback,
                ["ground_speed_kt", "gs_kt", "speed_kt", "speed_knots", "gs"],
            )

        track_deg = self._track_deg_from_payload(primary)
        if track_deg is None:
            track_deg = self._track_deg_from_payload(fallback)

        # Preferred path: consume provided ENU velocity components directly.
        # This avoids recomputing and possibly drifting from converter output.
        if east is not None and north is not None:
            if ground_speed_kt is None:
                ground_speed_kt = math.hypot(east, north) * MPS_TO_KNOT
            if track_deg is None:
                track_deg = math.degrees(math.atan2(east, north)) % 360.0

            return {
                "t": t,
                "east_mps": east,
                "north_mps": north,
                "up_mps": up,
                "ground_speed_kt": ground_speed_kt,
                "track_deg": track_deg,
                "source": "provided_enu_velocity_components",
            }

        # Fallback path: derive ENU components from ADS-B ground speed + track.
        if ground_speed_kt is not None and track_deg is not None:
            speed_mps = ground_speed_kt * KNOT_TO_MPS
            angle = math.radians(track_deg)

            return {
                "t": t,
                "east_mps": speed_mps * math.sin(angle),
                "north_mps": speed_mps * math.cos(angle),
                "up_mps": up,
                "ground_speed_kt": ground_speed_kt,
                "track_deg": track_deg,
                "source": "reported_adsb_ground_speed_track_only",
            }

        return None

    def _acceleration_payload_from_keyframe(self, kf: dict[str, Any]) -> Optional[dict[str, Any]]:
        t = as_float(kf.get("t"))
        if t is None:
            return None

        acceleration = kf.get("acceleration") or {}
        if not isinstance(acceleration, dict):
            return None

        east = self._first_numeric(acceleration, ["east_mps2", "ax_mps2", "ax", "east"])
        north = self._first_numeric(acceleration, ["north_mps2", "ay_mps2", "ay", "north"])
        # Use already-normalized vertical acceleration directly.
        # vertical_mps2 is the converter-provided vertical acceleration component.
        up = self._first_numeric(acceleration, ["up_mps2", "vertical_mps2", "accel_up_mps2", "az_mps2", "az", "up"])
        horizontal = self._first_numeric(acceleration, ["horizontal_mps2", "diagnostic_horizontal_mps2", "magnitude_mps2"])
        total = self._first_numeric(acceleration, ["total_mps2", "magnitude_3d_mps2"])

        if horizontal is None and east is not None and north is not None:
            horizontal = math.hypot(east, north)
        if total is None and (east is not None or north is not None or up is not None):
            total = math.sqrt((east or 0.0) ** 2 + (north or 0.0) ** 2 + (up or 0.0) ** 2)

        if east is None and north is None and up is None and horizontal is None and total is None:
            return None

        return {"t": t, "east_mps2": east, "north_mps2": north, "up_mps2": up, "horizontal_mps2": horizontal, "total_mps2": total}

    @staticmethod
    def _track_deg_from_payload(payload: dict[str, Any]) -> Optional[float]:
        for key in ["track_deg_normalized", "track_deg", "heading_deg", "heading_deg_normalized", "true_track_deg"]:
            value = as_float(payload.get(key))
            if value is not None:
                return value % 360.0

        for key in ["track_rad_normalized", "track_rad"]:
            value = as_float(payload.get(key))
            if value is not None:
                return math.degrees(value) % 360.0

        value = as_float(payload.get("track"))
        if value is None:
            return None

        unit = str(payload.get("track_unit") or "").lower()
        if unit in {"rad", "radian", "radians"}:
            return math.degrees(value) % 360.0
        if unit in {"deg", "degree", "degrees"}:
            return value % 360.0

        if abs(value) <= (2.0 * math.pi + 1e-6):
            return math.degrees(value) % 360.0
        return value % 360.0

    @staticmethod
    def _heading_from_render_neighbors(render_keyframes: list[dict[str, Any]], index: int) -> float:
        current_pos = (render_keyframes[index] or {}).get("position") or {}
        current_x = as_float(current_pos.get("x_m")) or 0.0
        current_y = as_float(current_pos.get("y_m")) or 0.0

        for other_kf in render_keyframes[index + 1:]:
            other_pos = other_kf.get("position") or {}
            if not other_pos:
                continue
            dx = (as_float(other_pos.get("x_m")) or 0.0) - current_x
            dy = (as_float(other_pos.get("y_m")) or 0.0) - current_y
            if math.hypot(dx, dy) > 0.5:
                return math.degrees(math.atan2(dx, dy)) % 360.0

        for other_kf in reversed(render_keyframes[:index]):
            other_pos = other_kf.get("position") or {}
            if not other_pos:
                continue
            dx = current_x - (as_float(other_pos.get("x_m")) or 0.0)
            dy = current_y - (as_float(other_pos.get("y_m")) or 0.0)
            if math.hypot(dx, dy) > 0.5:
                return math.degrees(math.atan2(dx, dy)) % 360.0

        return 0.0

    def _extract_animation_timeline(
        self,
        track_json: dict[str, Any],
        raw_position_points: list[dict[str, Any]],
        context: AdapterContext,
    ) -> list[FlightSample]:
        velocity_events = self._extract_velocity_events(track_json)
        acceleration_events = self._extract_acceleration_events(velocity_events)

        first_t = raw_position_points[0]["t"]
        samples: list[FlightSample] = []

        for index, point in enumerate(raw_position_points):
            t = point["t"]

            velocity = self._nearest_recent_payload(
                payloads=velocity_events,
                current_t=t,
                max_age_s=context.config.max_carried_velocity_age_s,
            )
            acceleration = self._nearest_recent_payload(
                payloads=acceleration_events,
                current_t=t,
                max_age_s=context.config.max_carried_acceleration_age_s,
            )

            vel_east = as_float((velocity or {}).get("east_mps"))
            vel_north = as_float((velocity or {}).get("north_mps"))
            vel_up = as_float((velocity or {}).get("up_mps"))
            ground_speed_kt = as_float((velocity or {}).get("ground_speed_kt"))
            accel_east = as_float((acceleration or {}).get("east_mps2"))
            accel_north = as_float((acceleration or {}).get("north_mps2"))
            accel_up = as_float((acceleration or {}).get("up_mps2"))
            accel_horizontal = as_float((acceleration or {}).get("horizontal_mps2"))
            accel_total = as_float((acceleration or {}).get("total_mps2"))

            heading = as_float((velocity or {}).get("track_deg"))
            if heading is None:
                heading = heading_from_velocity(vel_east, vel_north)
            if heading is None:
                heading = self._heading_from_neighbor_points(raw_position_points, index)

            if ground_speed_kt is None and vel_east is not None and vel_north is not None:
                ground_speed_kt = math.hypot(vel_east, vel_north) * MPS_TO_KNOT

            if accel_horizontal is None and accel_east is not None and accel_north is not None:
                accel_horizontal = math.hypot(accel_east, accel_north)
            if accel_total is None and (accel_east is not None or accel_north is not None or accel_up is not None):
                accel_total = math.sqrt((accel_east or 0.0) ** 2 + (accel_north or 0.0) ** 2 + (accel_up or 0.0) ** 2)

            samples.append(
                FlightSample(
                    t=t,
                    t_rel_s=t - first_t,
                    lon=point["lon"],
                    lat=point["lat"],
                    z_m=point["z_m"],
                    x_m=point["x_m"],
                    y_m=point["y_m"],
                    heading_deg=heading,
                    ground_speed_kt=ground_speed_kt,
                    vel_east_mps=vel_east,
                    vel_north_mps=vel_north,
                    vel_up_mps=vel_up,
                    accel_east_mps2=accel_east,
                    accel_north_mps2=accel_north,
                    accel_up_mps2=accel_up,
                    accel_horizontal_mps2=accel_horizontal,
                    accel_total_mps2=accel_total,
                    altitude_ft_msl=point.get("altitude_ft_msl"),
                )
            )

        return samples

    def _extract_velocity_events(
        self,
        track_json: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        for kf in track_json.get("raw_keyframes") or []:
            t = as_float(kf.get("t"))
            if t is None:
                continue

            velocity = kf.get("velocity") or {}
            if not isinstance(velocity, dict):
                velocity = {}

            payload = self._velocity_payload_from_payloads(t=t, primary=velocity, fallback=kf)
            if payload is None:
                continue

            events.append(payload)

        events.sort(key=lambda e: e["t"])
        return events

    def _extract_acceleration_events(self, velocity_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        for left, right in zip(velocity_events[:-1], velocity_events[1:]):
            t0 = as_float(left.get("t"))
            t1 = as_float(right.get("t"))
            ve0 = as_float(left.get("east_mps"))
            vn0 = as_float(left.get("north_mps"))
            ve1 = as_float(right.get("east_mps"))
            vn1 = as_float(right.get("north_mps"))

            if None in (t0, t1, ve0, vn0, ve1, vn1):
                continue

            dt = t1 - t0
            if dt <= 1e-6:
                continue

            ax = (ve1 - ve0) / dt
            ay = (vn1 - vn0) / dt
            # Fallback raw-keyframe mode derives only horizontal acceleration.
            # Vertical acceleration is consumed directly from render_keyframes[].acceleration
            # and is not calculated from vertical velocity/rate fields.
            az = None
            horizontal = math.hypot(ax, ay)
            total = horizontal

            events.append(
                {
                    "t": t1,
                    "east_mps2": ax,
                    "north_mps2": ay,
                    "up_mps2": az,
                    "horizontal_mps2": horizontal,
                    "total_mps2": total,
                }
            )

        return events

    @staticmethod
    def _nearest_recent_payload(
        payloads: list[dict[str, Any]],
        current_t: float,
        max_age_s: float,
    ) -> Optional[dict[str, Any]]:
        best: Optional[dict[str, Any]] = None
        best_age = float("inf")

        for payload in payloads:
            age = current_t - payload["t"]

            if age < 0:
                break

            if age <= max_age_s and age < best_age:
                best = payload
                best_age = age

        return best

    @staticmethod
    def _heading_from_neighbor_points(points: list[dict[str, Any]], index: int) -> float:
        current = points[index]

        for other in points[index + 1:]:
            dx = other["x_m"] - current["x_m"]
            dy = other["y_m"] - current["y_m"]
            if math.hypot(dx, dy) > 0.5:
                return math.degrees(math.atan2(dx, dy)) % 360.0

        for other in reversed(points[:index]):
            dx = current["x_m"] - other["x_m"]
            dy = current["y_m"] - other["y_m"]
            if math.hypot(dx, dy) > 0.5:
                return math.degrees(math.atan2(dx, dy)) % 360.0

        return 0.0

    @staticmethod
    def _first_numeric(payload: dict[str, Any], keys: list[str]) -> Optional[float]:
        for key in keys:
            if key in payload:
                value = as_float(payload.get(key))
                if value is not None:
                    return value
        return None

    def _extract_vertical_velocity_mps(self, payload: dict[str, Any]) -> Optional[float]:
        """Return a provided vertical velocity component in m/s.

        The converter now emits a real local-ENU vertical velocity component.
        The viewer must use that component directly and must not calculate it
        from other vertical-rate fields or unit conversions.
        """
        return self._first_numeric(payload, ["up_mps", "vertical_mps", "vel_up_mps", "vz_mps"])
