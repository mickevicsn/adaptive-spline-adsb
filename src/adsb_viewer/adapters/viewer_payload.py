from __future__ import annotations

from typing import Any, Optional

from .base import AdapterContext, NormalizedTrack
from ..geometry import as_float
from ..models import FlightSample


class ViewerPayloadAdapter:
    """Adapter for files that already use the browser-viewer payload contract."""

    adapter_id = "viewer_payload_v1"
    label = "Viewer payload"

    def can_load(self, data: dict[str, Any], _context: AdapterContext) -> bool:
        return isinstance(data.get("samples"), list) and (
            isinstance(data.get("rawPositionPoints"), list)
            or data.get("viewerPayloadVersion") is not None
        )

    def normalize(self, data: dict[str, Any], context: AdapterContext) -> NormalizedTrack:
        raw_points = data.get("rawPositionPoints") or []
        samples = [self._sample_from_payload(row) for row in data.get("samples") or [] if isinstance(row, dict)]
        samples = [sample for sample in samples if sample is not None]

        if not raw_points and samples:
            raw_points = [self._point_from_sample(sample, index) for index, sample in enumerate(samples)]

        if not raw_points:
            raise RuntimeError("Viewer payload has no rawPositionPoints and no samples to derive them from.")
        if not samples:
            raise RuntimeError("Viewer payload has no usable samples.")

        track_id = str(
            data.get("track_id")
            or data.get("flightId")
            or data.get("icao")
            or context.source_path.stem
            or "track"
        )

        return NormalizedTrack(
            track_json=data,
            raw_position_points=[dict(point) for point in raw_points],
            samples=samples,
            track_id=track_id,
            source_schema=self.adapter_id,
            interpolate_samples=False,
            metadata={
                "adapterId": self.adapter_id,
                "methodId": context.method_id,
                "methodLabel": context.method_label,
            },
        )

    @staticmethod
    def _sample_from_payload(row: dict[str, Any]) -> Optional[FlightSample]:
        t = as_float(row.get("t"))
        lon = as_float(row.get("lon"))
        lat = as_float(row.get("lat"))
        if t is None or lon is None or lat is None:
            return None

        z_m = as_float(row.get("z_m")) or 0.0
        return FlightSample(
            t=t,
            t_rel_s=as_float(row.get("t_rel_s")) or 0.0,
            lon=lon,
            lat=lat,
            z_m=z_m,
            x_m=as_float(row.get("x_m")) or 0.0,
            y_m=as_float(row.get("y_m")) or 0.0,
            heading_deg=as_float(row.get("heading_deg")) or 0.0,
            ground_speed_kt=as_float(row.get("ground_speed_kt")),
            vel_east_mps=as_float(row.get("vel_east_mps")),
            vel_north_mps=as_float(row.get("vel_north_mps")),
            vel_up_mps=as_float(row.get("vel_up_mps")),
            accel_east_mps2=as_float(row.get("accel_east_mps2")),
            accel_north_mps2=as_float(row.get("accel_north_mps2")),
            accel_up_mps2=as_float(row.get("accel_up_mps2")),
            accel_horizontal_mps2=as_float(row.get("accel_horizontal_mps2")),
            accel_total_mps2=as_float(row.get("accel_total_mps2")),
            altitude_ft_msl=as_float(row.get("altitude_ft_msl")),
        )

    @staticmethod
    def _point_from_sample(sample: FlightSample, index: int) -> dict[str, Any]:
        return {
            "t": sample.t,
            "row_id": index,
            "source_id": f"sample_{index}",
            "kind": "viewer_sample",
            "lon": sample.lon,
            "lat": sample.lat,
            "z_m": sample.z_m,
            "x_m": sample.x_m,
            "y_m": sample.y_m,
            "altitude_ft_msl": sample.altitude_ft_msl,
            "height_above_field_m": None,
        }
