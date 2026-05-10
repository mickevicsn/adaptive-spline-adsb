from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ViewerConfig:
    title: str = "ADS-B 3D Flight Viewer"

    # Kept only so older main scripts do not break.
    output_dir: Path = Path("viewer_output")

    # Animation
    sample_period_s: float = 0.25
    initial_speed: float = 1.0
    min_simulation_speed: float = 0.25
    max_simulation_speed: float = 8.0
    simulation_speed_step: float = 0.25
    trail_seconds: float = 90.0

    # Cesium chase/orbit camera
    camera_initial_heading_deg: float = 0.0
    camera_initial_pitch_deg: float = -28.0
    camera_min_pitch_deg: float = -85.0
    camera_max_pitch_deg: float = -5.0
    camera_heading_step_deg: float = 10.0
    camera_pitch_step_deg: float = 4.0
    camera_range_m: float = 650.0
    camera_altitude_range_factor: float = 0.35

    # Mouse-wheel zoom for the chase/orbit camera.
    camera_min_range_m: float = 80.0
    camera_max_range_m: float = 50_000.0
    camera_wheel_zoom_factor: float = 1.12

    # Screen-space chase offset. Positive values shift the aircraft toward
    # the bottom-right of the viewport so more forward/upper-left trajectory
    # remains visible.
    camera_target_screen_offset_right_ratio: float = 0.18
    camera_target_screen_offset_down_ratio: float = 0.12

    # Visual scaling
    altitude_visual_scale: float = 1.0

    # Raw dots
    raw_point_pixel_size: float = 10.0

    # Connection line between raw dots.
    connection_line_width_px: float = 2.0
    connection_line_side_offset_m: float = 7.0
    connection_line_endpoint_gap_m: float = 5.0

    # Future track fade. Points/segments up to this far ahead stay solid;
    # farther future track becomes progressively more transparent.
    future_track_fade_start_s: float = 120.0
    future_track_fade_duration_s: float = 480.0
    future_track_min_alpha: float = 0.08

    trail_width_px: float = 7.0

    aircraft_point_pixel_size: float = 14.0

    # Vectors
    velocity_vector_seconds: float = 8.0
    acceleration_vector_scale: float = 140.0
    velocity_vector_min_length_m: float = 160.0
    acceleration_vector_min_length_m: float = 56.0
    vector_lift_m: float = 0.0
    max_carried_velocity_age_s: float = 8.0
    max_carried_acceleration_age_s: float = 8.0
