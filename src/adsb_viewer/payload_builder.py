from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .adapters import AdapterContext, NormalizedTrack
from .config import ViewerConfig
from .geometry import heading_from_velocity, lerp, lerp_optional, offset_lat_lon
from .method_registry import MethodRegistry, default_method_registry
from .models import FlightSample


class TrackPayloadBuilder:
    """Build the browser/Cesium payload from a method-pluggable input file.

    This class is intentionally method-neutral. It loads JSON, asks the method
    registry for an adapter, receives a normalized track, and then performs only
    viewer-common work: optional position interpolation, vector endpoint creation,
    and config metadata assembly.
    """

    def __init__(
        self,
        config: Optional[ViewerConfig] = None,
        registry: Optional[MethodRegistry] = None,
    ):
        self.config = config or ViewerConfig()
        self.registry = registry or default_method_registry()

    def build_payload(
        self,
        json_path: str | Path,
        *,
        method_id: str = "raw_adsb",
        method_label: str = "Raw ADS-B",
    ) -> dict[str, Any]:
        json_path = Path(json_path)
        method_json = self._load_json(json_path)
        context = AdapterContext(
            method_id=method_id,
            method_label=method_label,
            source_path=json_path,
            config=self.config,
        )
        adapter = self.registry.resolve(method_json, context)
        normalized = adapter.normalize(method_json, context)

        samples = normalized.samples
        if normalized.interpolate_samples:
            samples = self._interpolate_samples(samples)

        return self._build_browser_payload(
            normalized=normalized,
            samples=samples,
            adapter_id=getattr(adapter, "adapter_id", adapter.__class__.__name__),
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")

        return data

    def _interpolate_samples(self, sparse: list[FlightSample]) -> list[FlightSample]:
        if len(sparse) <= 1:
            return sparse

        dense: list[FlightSample] = []

        for left, right in zip(sparse[:-1], sparse[1:]):
            dt = right.t_rel_s - left.t_rel_s
            if dt <= 0:
                continue

            steps = max(1, int(math.ceil(dt / self.config.sample_period_s)))

            for step in range(steps):
                alpha = min(1.0, (step * self.config.sample_period_s) / dt)
                dense.append(self._lerp_sample(left, right, alpha))

        dense.append(sparse[-1])
        return dense

    @staticmethod
    def _normalize_segments(track_json: dict[str, Any]) -> list[dict[str, Any]]:
        """Return lightweight segment refs from minimal or detailed method JSON.

        The current V-Spline minimal JSON may not contain top-level ``segments``
        yet, but it does contain ``segment_id`` on each ``render_keyframe``.
        In that case derive segment time/index ranges from consecutive runs of
        render samples so raw ADS-B dots can still be coloured by segment.
        """
        segments = track_json.get("segments")
        if isinstance(segments, list) and segments:
            return [dict(seg) for seg in segments if isinstance(seg, dict)]

        # Detailed piecewise payloads may store segments under connected components.
        result: list[dict[str, Any]] = []
        for component in track_json.get("connected_components") or []:
            if not isinstance(component, dict):
                continue
            component_id = component.get("component_id")
            for seg in component.get("segments") or []:
                if not isinstance(seg, dict):
                    continue
                row = dict(seg)
                row.setdefault("component_id", component_id)
                result.append(row)
        if result:
            return result

        # Backward-compatible fallback for schema_version 1.3 files where
        # segment refs live only on render_keyframes.
        return TrackPayloadBuilder._derive_segments_from_render_keyframes(track_json)

    @staticmethod
    def _derive_segments_from_render_keyframes(track_json: dict[str, Any]) -> list[dict[str, Any]]:
        render_keyframes = track_json.get("render_keyframes") or []
        if not isinstance(render_keyframes, list) or not render_keyframes:
            return []

        segments: list[dict[str, Any]] = []
        current_id: Optional[str] = None
        start_index: Optional[int] = None
        start_t: Optional[float] = None
        last_t: Optional[float] = None
        last_index: Optional[int] = None

        def flush() -> None:
            nonlocal current_id, start_index, start_t, last_t, last_index
            if current_id is None or start_index is None or last_index is None:
                return
            component_id = None
            if "_seg_" in current_id:
                component_id = current_id.split("_seg_", 1)[0]
            segments.append({
                "segment_id": current_id,
                "component_id": component_id,
                "segment_index": len(segments),
                "segment_color_index": len(segments),
                "regime_label": "segment",
                "t0": start_t,
                "t1": last_t,
                "render_time_start": start_t,
                "render_time_end": last_t,
                "render_keyframe_start_index": start_index,
                "render_keyframe_end_index": last_index,
            })

        for index, row in enumerate(render_keyframes):
            if not isinstance(row, dict):
                continue
            segment_id = row.get("segment_id") or row.get("segmentId")
            t = TrackPayloadBuilder._as_number(row.get("t"))
            if segment_id is None or t is None:
                continue
            segment_id = str(segment_id)
            if current_id is None:
                current_id = segment_id
                start_index = index
                start_t = t
            elif segment_id != current_id:
                flush()
                current_id = segment_id
                start_index = index
                start_t = t
            last_index = index
            last_t = t

        flush()
        return segments

    @staticmethod
    def _normalize_segment_boundaries(track_json: dict[str, Any]) -> list[dict[str, Any]]:
        boundaries = track_json.get("segment_boundaries") or track_json.get("boundaries")
        if isinstance(boundaries, list):
            return [dict(boundary) for boundary in boundaries if isinstance(boundary, dict)]

        result: list[dict[str, Any]] = []
        for component in track_json.get("connected_components") or []:
            if not isinstance(component, dict):
                continue
            component_id = component.get("component_id")
            for boundary in component.get("boundaries") or []:
                if not isinstance(boundary, dict):
                    continue
                row = dict(boundary)
                row.setdefault("component_id", component_id)
                result.append(row)
        return result

    @staticmethod
    def _as_number(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            number = float(value)
            if math.isfinite(number):
                return number
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _segment_index_bounds(segment: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
        start = (
            segment.get("raw_point_start_index")
            if segment.get("raw_point_start_index") is not None
            else segment.get("start_sample_index")
        )
        if start is None:
            start = segment.get("sample_index_start")
        if start is None:
            start = segment.get("render_keyframe_start_index")

        end = (
            segment.get("raw_point_end_index")
            if segment.get("raw_point_end_index") is not None
            else segment.get("end_sample_index")
        )
        if end is None:
            end = segment.get("sample_index_end")
        if end is None:
            end = segment.get("render_keyframe_end_index")

        try:
            start_i = int(start) if start is not None else None
        except (TypeError, ValueError):
            start_i = None
        try:
            end_i = int(end) if end is not None else None
        except (TypeError, ValueError):
            end_i = None
        return start_i, end_i

    @classmethod
    def _segment_time_bounds(cls, segment: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        t0 = cls._as_number(segment.get("t0"))
        if t0 is None:
            t0 = cls._as_number(segment.get("render_time_start"))
        if t0 is None:
            t0 = cls._as_number(segment.get("start_time"))

        t1 = cls._as_number(segment.get("t1"))
        if t1 is None:
            t1 = cls._as_number(segment.get("render_time_end"))
        if t1 is None:
            t1 = cls._as_number(segment.get("end_time"))

        return t0, t1

    @classmethod
    def _annotate_raw_points_with_segments(
        cls,
        points: list[dict[str, Any]],
        segments: list[dict[str, Any]],
    ) -> int:
        """Attach segment metadata to raw points for browser-side colorizing."""
        if not points or not segments:
            return 0

        annotated = 0
        for seg_index, segment in enumerate(segments):
            segment_id = str(segment.get("segment_id") or f"seg_{seg_index:04d}")
            component_id = segment.get("component_id")
            regime = segment.get("regime_label")
            start_i, end_i = cls._segment_index_bounds(segment)
            t0, t1 = cls._segment_time_bounds(segment)

            matched_any = False
            if start_i is not None and end_i is not None:
                lo = max(0, min(start_i, end_i))
                hi = min(len(points) - 1, max(start_i, end_i))
                for point_index in range(lo, hi + 1):
                    point = points[point_index]
                    if point.get("segment_id") is None:
                        annotated += 1
                    point["segment_id"] = segment_id
                    point["segment_index"] = seg_index
                    point["segment_color_index"] = seg_index
                    point["segment_regime_label"] = regime
                    point["component_id"] = component_id
                    matched_any = True

            if matched_any:
                continue

            if t0 is None or t1 is None:
                continue
            lo_t = min(t0, t1)
            hi_t = max(t0, t1)
            eps = 1e-6
            for point in points:
                t = cls._as_number(point.get("t"))
                if t is None or t < lo_t - eps or t > hi_t + eps:
                    continue
                if point.get("segment_id") is None:
                    annotated += 1
                point["segment_id"] = segment_id
                point["segment_index"] = seg_index
                point["segment_color_index"] = seg_index
                point["segment_regime_label"] = regime
                point["component_id"] = component_id

        return annotated

    def _build_browser_payload(
        self,
        normalized: NormalizedTrack,
        samples: list[FlightSample],
        adapter_id: str,
    ) -> dict[str, Any]:
        browser_points = []
        for index, p in enumerate(normalized.raw_position_points):
            row = dict(p)
            row["point_index"] = index
            row["z_m_visual"] = (p.get("z_m") or 0.0) * self.config.altitude_visual_scale
            browser_points.append(row)

        browser_samples = []
        for sample in samples:
            row = asdict(sample)
            row["z_m_visual"] = sample.z_m * self.config.altitude_visual_scale
            row["velocity_vector"] = self._make_velocity_vector(sample)
            row["acceleration_vector"] = self._make_acceleration_vector(sample)
            browser_samples.append(row)

        segments = self._normalize_segments(normalized.track_json)
        segment_boundaries = self._normalize_segment_boundaries(normalized.track_json)
        colorized_raw_points = self._annotate_raw_points_with_segments(browser_points, segments)

        max_z_m = max(((p.get("z_m") or 0.0) for p in normalized.raw_position_points), default=0.0)

        method_id = str(normalized.metadata.get("methodId") or "").lower()
        display_meta = normalized.track_json.get("display") or {}
        is_reconstructed_curve_method = (
            method_id in {"kalman_rts", "v_spline"}
            or bool(display_meta.get("kalman_rts_animation"))
            or bool(display_meta.get("v_spline_animation"))
        )

        payload = {
            "title": self.config.title,
            "track_id": normalized.track_id,
            "method": {
                "methodId": normalized.metadata.get("methodId"),
                "methodLabel": normalized.metadata.get("methodLabel"),
                "adapterId": adapter_id,
                "sourceSchema": normalized.source_schema,
            },
            "config": {
                "initialSpeed": self.config.initial_speed,
                "minSimulationSpeed": self.config.min_simulation_speed,
                "maxSimulationSpeed": self.config.max_simulation_speed,
                "simulationSpeedStep": self.config.simulation_speed_step,
                "trailSeconds": self.config.trail_seconds,
                "cameraInitialHeadingDeg": self.config.camera_initial_heading_deg,
                "cameraInitialPitchDeg": self.config.camera_initial_pitch_deg,
                "cameraMinPitchDeg": self.config.camera_min_pitch_deg,
                "cameraMaxPitchDeg": self.config.camera_max_pitch_deg,
                "cameraHeadingStepDeg": self.config.camera_heading_step_deg,
                "cameraPitchStepDeg": self.config.camera_pitch_step_deg,
                "cameraRangeM": self.config.camera_range_m,
                "cameraAltitudeRangeFactor": self.config.camera_altitude_range_factor,
                "cameraMinRangeM": self.config.camera_min_range_m,
                "cameraMaxRangeM": self.config.camera_max_range_m,
                "cameraWheelZoomFactor": self.config.camera_wheel_zoom_factor,
                "cameraTargetScreenOffsetRightRatio": self.config.camera_target_screen_offset_right_ratio,
                "cameraTargetScreenOffsetDownRatio": self.config.camera_target_screen_offset_down_ratio,
                "altitudeVisualScale": self.config.altitude_visual_scale,
                "rawPointPixelSize": self.config.raw_point_pixel_size,
                "connectionLineWidthPx": self.config.connection_line_width_px,
                "connectionLineSideOffsetM": self.config.connection_line_side_offset_m,
                "connectionLineEndpointGapM": self.config.connection_line_endpoint_gap_m,
                # Raw ADS-B uses blue connection segments between raw dots.
                # Reconstruction methods such as Kalman RTS and V-Spline show
                # raw observations as disconnected points and the reconstructed
                # trajectory as a separate continuous line.
                "showRawConnectionLines": not is_reconstructed_curve_method,
                "showReconstructedPathLine": is_reconstructed_curve_method,
                "reconstructedPathLineWidthPx": 4.0 if is_reconstructed_curve_method else 0.0,
                "futureTrackFadeStartS": self.config.future_track_fade_start_s,
                "futureTrackFadeDurationS": self.config.future_track_fade_duration_s,
                "futureTrackMinAlpha": self.config.future_track_min_alpha,
                "trailWidthPx": self.config.trail_width_px,
                "aircraftPointPixelSize": self.config.aircraft_point_pixel_size,
                "maxZM": max_z_m,
                "vectorDisplayInterpolation": "none",
                "vectorDisplayMode": "zero_order_hold_raw_reported_values",
                "vectorDisplayDescription": "Velocity/acceleration vectors are not linearly interpolated; latest method-provided vector sample is held until the next vector keyframe.",
            },
            "rawPositionPoints": browser_points,
            "samples": browser_samples,
            "piecewise": normalized.track_json.get("piecewise") or {},
            "segments": segments,
            "segmentBoundaries": segment_boundaries,
            "segmentColoring": {
                "enabled": bool(segments),
                "colorizedRawPoints": colorized_raw_points,
                "rawPointCount": len(browser_points),
            },
        }

        return payload

    def _make_velocity_vector(self, sample: FlightSample) -> Optional[dict[str, Any]]:
        if sample.vel_east_mps is None or sample.vel_north_mps is None:
            return None

        east_m = sample.vel_east_mps * self.config.velocity_vector_seconds
        north_m = sample.vel_north_mps * self.config.velocity_vector_seconds
        horizontal_m = math.hypot(east_m, north_m)
        min_length_m = max(0.0, self.config.velocity_vector_min_length_m)
        if 0.0 < horizontal_m < min_length_m:
            scale = min_length_m / horizontal_m
            east_m *= scale
            north_m *= scale

        up_m = (sample.vel_up_mps or 0.0) * self.config.velocity_vector_seconds
        lat2, lon2 = offset_lat_lon(sample.lat, sample.lon, east_m, north_m)

        z = sample.z_m * self.config.altitude_visual_scale + self.config.vector_lift_m

        return {
            "source": [sample.lon, sample.lat, z],
            "target": [lon2, lat2, z + up_m],
        }

    def _make_acceleration_vector(self, sample: FlightSample) -> Optional[dict[str, Any]]:
        if sample.accel_east_mps2 is None and sample.accel_north_mps2 is None and sample.accel_up_mps2 is None:
            return None

        east_m = (sample.accel_east_mps2 or 0.0) * self.config.acceleration_vector_scale
        north_m = (sample.accel_north_mps2 or 0.0) * self.config.acceleration_vector_scale
        up_m = (sample.accel_up_mps2 or 0.0) * self.config.acceleration_vector_scale
        total_m = math.sqrt(east_m * east_m + north_m * north_m + up_m * up_m)
        min_length_m = max(0.0, self.config.acceleration_vector_min_length_m)
        if 0.0 < total_m < min_length_m:
            scale = min_length_m / total_m
            east_m *= scale
            north_m *= scale
            up_m *= scale

        lat2, lon2 = offset_lat_lon(sample.lat, sample.lon, east_m, north_m)
        z = sample.z_m * self.config.altitude_visual_scale + self.config.vector_lift_m

        return {
            "source": [sample.lon, sample.lat, z],
            "target": [lon2, lat2, z + up_m],
        }

    @staticmethod
    def _lerp_sample(left: FlightSample, right: FlightSample, alpha: float) -> FlightSample:
        # Position is linearly interpolated for smooth movement. Vector fields are
        # zero-order held from the left sample so method-provided vector values are
        # never visually smoothed by the viewer.
        vel_east = left.vel_east_mps
        vel_north = left.vel_north_mps

        heading = left.heading_deg
        if heading is None:
            heading = heading_from_velocity(vel_east, vel_north)
        if heading is None:
            heading = math.degrees(math.atan2(right.x_m - left.x_m, right.y_m - left.y_m)) % 360.0

        return FlightSample(
            t=lerp(left.t, right.t, alpha),
            t_rel_s=lerp(left.t_rel_s, right.t_rel_s, alpha),
            lon=lerp(left.lon, right.lon, alpha),
            lat=lerp(left.lat, right.lat, alpha),
            z_m=lerp(left.z_m, right.z_m, alpha),
            x_m=lerp(left.x_m, right.x_m, alpha),
            y_m=lerp(left.y_m, right.y_m, alpha),
            heading_deg=heading,
            ground_speed_kt=left.ground_speed_kt,
            vel_east_mps=vel_east,
            vel_north_mps=vel_north,
            vel_up_mps=left.vel_up_mps,
            accel_east_mps2=left.accel_east_mps2,
            accel_north_mps2=left.accel_north_mps2,
            accel_up_mps2=left.accel_up_mps2,
            accel_horizontal_mps2=left.accel_horizontal_mps2,
            accel_total_mps2=left.accel_total_mps2,
            altitude_ft_msl=lerp_optional(left.altitude_ft_msl, right.altitude_ft_msl, alpha),
        )
