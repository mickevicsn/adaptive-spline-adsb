"""
Adapter from raw_keyframes_*.json to the paper-oriented V-Spline core input.

This module does NOT change the V-Spline mathematical core.  It only prepares
strict paired 3D observations:

    t_i
    y_i = [x_m, y_m, z_m]
    v_i = [east_mps, north_mps, up_mps]

from the viewer/raw-keyframe JSON payload.

Important policy:
    * Keep only keyframes explicitly marked paired_for_vspline/full position/full velocity.
    * Do not deduplicate timestamps.
    * Sort by t, then fail loudly if duplicate t values exist.
    * Split only after paired extraction and duplicate validation.
    * Acceleration fields are never used as V-Spline observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math

import numpy as np


DIM_NAMES_3D: tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class RawKeyframeVSplineAdapterConfig:
    """Configuration for raw-keyframe -> V-Spline core preparation.

    Parameters
    ----------
    require_paired_flag:
        Require kf["paired_for_vspline"] is True.
    require_full_position_flag:
        Require kf["full_position_for_vspline"] is True.
    require_full_velocity_flag:
        Require kf["full_velocity_for_vspline"] is True.
    require_velocity_dimension_3d:
        Require velocity["dimension"] == "3d" when present.
    require_vertical_component_available:
        Require velocity["vertical_component_available"] is True.
    require_local_enu_velocity_frame:
        Require velocity["frame"] starts with "local_enu" when present.
    allow_position_anchor_only:
        If False, rejects render/interpolated anchors that are not raw position
        fields.  The adapter expects actual kf["position"] values.
    max_gap_s:
        If not None, split paired samples into new segments whenever
        consecutive paired times differ by more than this threshold.
    min_segment_observations:
        Drop/fail segments shorter than this many paired samples depending on
        fail_on_short_segment.
    fail_on_short_segment:
        Raise if a split segment has fewer than min_segment_observations.
        If False, short segments are skipped.
    duplicate_time_tolerance_s:
        Default 0.0 means exact duplicate float values after JSON parsing.
        Set only if an upstream writer may produce timestamps that should be
        treated as identical within a tolerance.  No deduplication is performed;
        this only changes duplicate detection.
    fail_on_unpaired_keyframes:
        If True, raise when raw_keyframes contain unpaired records.  Usually
        False because velocity-only rows are expected before/after paired data.
    fail_on_invalid_paired_keyframe:
        If True, raise when a keyframe passes flags but has missing/non-finite
        numeric fields.  If False, invalid paired keyframes are skipped and
        reported in diagnostics.
    """

    require_paired_flag: bool = True
    require_full_position_flag: bool = True
    require_full_velocity_flag: bool = True
    require_velocity_dimension_3d: bool = True
    require_vertical_component_available: bool = True
    require_local_enu_velocity_frame: bool = True
    allow_position_anchor_only: bool = False

    max_gap_s: float | None = 30.0
    min_segment_observations: int = 2
    fail_on_short_segment: bool = True

    duplicate_time_tolerance_s: float = 0.0
    fail_on_unpaired_keyframes: bool = False
    fail_on_invalid_paired_keyframe: bool = True

    # Data-driven paired-position sanity filter.  This catches impossible ADS-B
    # position jumps that survive raw CRC/event filtering and broad inferred
    # track rules.
    enable_position_motion_outlier_filter: bool = True
    position_outlier_speed_gate_mps: float = 1200.0
    position_outlier_speed_factor: float = 5.0
    position_outlier_max_iterations: int = 4
    position_outlier_min_samples_remaining: int = 8


@dataclass(frozen=True)
class PreparedVSplineSample:
    """One strict paired 3D V-Spline observation extracted from a raw keyframe."""

    keyframe_id: str
    t: float
    y: tuple[float, float, float]
    v: tuple[float, float, float]
    row_ids: tuple[int, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    raw_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedVSplineSegment:
    """A contiguous block of paired samples ready for local B-spline fitting."""

    segment_id: str
    samples: tuple[PreparedVSplineSample, ...]
    split_reason: str
    t0: float
    t1: float
    dt_min_s: float | None
    dt_max_s: float | None

    @property
    def n_observations(self) -> int:
        return len(self.samples)

    @property
    def keyframe_ids(self) -> list[str]:
        return [s.keyframe_id for s in self.samples]

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "split_reason": self.split_reason,
            "n_observations": self.n_observations,
            "t0": self.t0,
            "t1": self.t1,
            "dt_min_s": self.dt_min_s,
            "dt_max_s": self.dt_max_s,
            "keyframe_ids": self.keyframe_ids,
        }


@dataclass
class RawKeyframeVSplinePreparation:
    """Prepared output and diagnostics from RawKeyframeVSplineAdapter."""

    source_path: str
    track_id: str | None
    icao: str | None
    schema_version: str | None
    samples: list[PreparedVSplineSample]
    segments: list[PreparedVSplineSegment]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, include_samples: bool = False) -> dict[str, Any]:
        out = {
            "source_path": self.source_path,
            "track_id": self.track_id,
            "icao": self.icao,
            "schema_version": self.schema_version,
            "n_samples": len(self.samples),
            "n_segments": len(self.segments),
            "segments": [s.as_dict() for s in self.segments],
            "diagnostics": self.diagnostics,
        }
        if include_samples:
            out["samples"] = [s.as_dict() for s in self.samples]
        return out


class RawKeyframeVSplineAdapter:
    """Convert raw-keyframe JSON into strict paired 3D V-Spline samples."""

    def __init__(
        self,
        json_path: str | Path,
        config: RawKeyframeVSplineAdapterConfig | None = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.config = config or RawKeyframeVSplineAdapterConfig()
        self.payload: dict[str, Any] | None = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load the JSON payload from disk."""
        with self.json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Top-level raw keyframe JSON must be an object/dict")
        self.payload = payload
        return payload

    def prepare(self) -> RawKeyframeVSplinePreparation:
        """Load, validate, extract paired samples, verify time uniqueness, split."""
        payload = self.payload if self.payload is not None else self.load()

        self.validate_reference(payload)

        raw_keyframes = payload.get("raw_keyframes")
        if not isinstance(raw_keyframes, list):
            raise ValueError('JSON payload must contain list field "raw_keyframes"')

        samples, extraction_diag = self.extract_paired_samples(raw_keyframes)
        ordered = self.sort_and_verify_unique_times(samples)
        ordered, position_outlier_diag = self.filter_position_motion_outliers(ordered)
        self.verify_strictly_increasing(ordered)

        segments, split_diag = self.split_samples(ordered)

        diagnostics = {
            "adapter": "RawKeyframeVSplineAdapter",
            "dimension": "3d",
            "dim_names": list(DIM_NAMES_3D),
            "uses_acceleration_observations": False,
            "sort_policy": "sort_by_t_then_verify_unique",
            "deduplication_policy": "none_raise_on_duplicate_t",
            "extraction": extraction_diag,
            "position_motion_outlier_filter": position_outlier_diag,
            "split": split_diag,
            "config": asdict(self.config),
        }

        return RawKeyframeVSplinePreparation(
            source_path=str(self.json_path),
            track_id=_optional_str(payload.get("track_id")),
            icao=_optional_str(payload.get("icao")),
            schema_version=_optional_str(payload.get("schema_version")),
            samples=ordered,
            segments=segments,
            diagnostics=diagnostics,
        )


    # ---------------------------------------------------------------------
    # Validation / extraction
    # ---------------------------------------------------------------------

    def validate_reference(self, payload: dict[str, Any]) -> None:
        """Validate reference metadata expected by this adapter.

        This method validates only metadata that affects interpretation of
        x/y/z and velocity components.  It does not require origin lat/lon,
        because this JSON is assumed to already contain local-frame positions.
        """
        reference = payload.get("reference")
        if not isinstance(reference, dict):
            raise ValueError('JSON payload must contain object field "reference"')

        local_frame = reference.get("local_frame")
        if local_frame != "horizontal_enu_plus_barometric_z":
            raise ValueError(
                "Unsupported reference.local_frame. Expected "
                "'horizontal_enu_plus_barometric_z', got "
                f"{local_frame!r}"
            )

        time_unit = reference.get("time_unit")
        if time_unit != "unix_seconds":
            raise ValueError(
                f"Unsupported reference.time_unit. Expected 'unix_seconds', got {time_unit!r}"
            )

        velocity_obs = reference.get("velocity_observation")
        if isinstance(velocity_obs, dict):
            frame = velocity_obs.get("frame")
            if self.config.require_local_enu_velocity_frame:
                if not (isinstance(frame, str) and frame.startswith("local_enu")):
                    raise ValueError(
                        "Unsupported reference.velocity_observation.frame. "
                        f"Expected local_enu*, got {frame!r}"
                    )

            # Some older payloads said "horizontal" even if keyframes now carry
            # vertical rate.  Do not fail solely on reference.dimension here.
            # The authoritative check is per-keyframe velocity["dimension"] and
            # vertical_component_available.

    def extract_paired_samples(
        self,
        raw_keyframes: Sequence[dict[str, Any]],
    ) -> tuple[list[PreparedVSplineSample], dict[str, Any]]:
        """Extract strict paired 3D samples from raw keyframes."""
        samples: list[PreparedVSplineSample] = []
        skipped: dict[str, int] = {
            "not_dict": 0,
            "unpaired_flag": 0,
            "missing_full_position_flag": 0,
            "missing_full_velocity_flag": 0,
            "missing_position_object": 0,
            "missing_velocity_object": 0,
            "invalid_velocity_dimension": 0,
            "invalid_velocity_frame": 0,
            "vertical_component_unavailable": 0,
            "nonfinite_numeric_field": 0,
        }
        invalid_preview: list[dict[str, Any]] = []
        unpaired_count = 0

        for idx, kf in enumerate(raw_keyframes):
            if not isinstance(kf, dict):
                skipped["not_dict"] += 1
                continue

            keyframe_id = _optional_str(kf.get("id")) or f"raw_index_{idx}"

            if self.config.require_paired_flag and kf.get("paired_for_vspline") is not True:
                skipped["unpaired_flag"] += 1
                unpaired_count += 1
                continue

            if (
                self.config.require_full_position_flag
                and kf.get("full_position_for_vspline") is not True
            ):
                skipped["missing_full_position_flag"] += 1
                continue

            if (
                self.config.require_full_velocity_flag
                and kf.get("full_velocity_for_vspline") is not True
            ):
                skipped["missing_full_velocity_flag"] += 1
                continue

            position = kf.get("position")
            velocity = kf.get("velocity")
            if not isinstance(position, dict):
                skipped["missing_position_object"] += 1
                self._handle_invalid_paired(
                    invalid_preview,
                    keyframe_id,
                    idx,
                    "missing position object",
                )
                continue

            if not isinstance(velocity, dict):
                skipped["missing_velocity_object"] += 1
                self._handle_invalid_paired(
                    invalid_preview,
                    keyframe_id,
                    idx,
                    "missing velocity object",
                )
                continue

            if self.config.require_velocity_dimension_3d:
                if velocity.get("dimension") != "3d":
                    skipped["invalid_velocity_dimension"] += 1
                    self._handle_invalid_paired(
                        invalid_preview,
                        keyframe_id,
                        idx,
                        f"velocity.dimension={velocity.get('dimension')!r}",
                    )
                    continue

            if self.config.require_local_enu_velocity_frame:
                frame = velocity.get("frame")
                if not (isinstance(frame, str) and frame.startswith("local_enu")):
                    skipped["invalid_velocity_frame"] += 1
                    self._handle_invalid_paired(
                        invalid_preview,
                        keyframe_id,
                        idx,
                        f"velocity.frame={frame!r}",
                    )
                    continue

            if self.config.require_vertical_component_available:
                if velocity.get("vertical_component_available") is not True:
                    skipped["vertical_component_unavailable"] += 1
                    self._handle_invalid_paired(
                        invalid_preview,
                        keyframe_id,
                        idx,
                        "vertical_component_available is not true",
                    )
                    continue

            try:
                t = _finite_float(kf.get("t"), "t")
                y = (
                    _finite_float(position.get("x_m"), "position.x_m"),
                    _finite_float(position.get("y_m"), "position.y_m"),
                    _finite_float(position.get("z_m"), "position.z_m"),
                )
                v = (
                    _finite_float(velocity.get("east_mps"), "velocity.east_mps"),
                    _finite_float(velocity.get("north_mps"), "velocity.north_mps"),
                    _finite_float(velocity.get("up_mps"), "velocity.up_mps"),
                )
            except ValueError as exc:
                skipped["nonfinite_numeric_field"] += 1
                self._handle_invalid_paired(
                    invalid_preview,
                    keyframe_id,
                    idx,
                    str(exc),
                )
                continue

            samples.append(
                PreparedVSplineSample(
                    keyframe_id=keyframe_id,
                    t=t,
                    y=y,
                    v=v,
                    row_ids=tuple(_ints_from_any(kf.get("row_ids"))),
                    source_event_ids=tuple(_strings_from_any(kf.get("source_event_ids"))),
                    raw_index=idx,
                )
            )

        if self.config.fail_on_unpaired_keyframes and unpaired_count:
            raise ValueError(
                "Found unpaired keyframes although fail_on_unpaired_keyframes=True. "
                f"unpaired_count={unpaired_count}"
            )

        if not samples:
            raise ValueError(
                "No paired 3D V-Spline samples extracted. Need keyframes with "
                "paired_for_vspline=true, full_position_for_vspline=true, "
                "full_velocity_for_vspline=true, position x/y/z, and velocity "
                "east/north/up."
            )

        diag = {
            "raw_keyframe_count": len(raw_keyframes),
            "paired_sample_count": len(samples),
            "skipped_counts": skipped,
            "invalid_preview": invalid_preview[:10],
            "first_sample_t": float(samples[0].t) if samples else None,
            "last_sample_t": float(samples[-1].t) if samples else None,
        }
        return samples, diag

    def _handle_invalid_paired(
        self,
        invalid_preview: list[dict[str, Any]],
        keyframe_id: str,
        raw_index: int,
        reason: str,
    ) -> None:
        invalid_preview.append(
            {"keyframe_id": keyframe_id, "raw_index": raw_index, "reason": reason}
        )
        if self.config.fail_on_invalid_paired_keyframe:
            raise ValueError(
                f"Invalid paired V-Spline keyframe {keyframe_id} at raw index "
                f"{raw_index}: {reason}"
            )

    # ---------------------------------------------------------------------
    # Ordering / uniqueness
    # ---------------------------------------------------------------------

    def sort_and_verify_unique_times(
        self,
        samples: Sequence[PreparedVSplineSample],
    ) -> list[PreparedVSplineSample]:
        """Sort by t and fail if duplicate times exist.

        No deduplication or averaging is performed.
        """
        ordered = sorted(samples, key=lambda s: s.t)

        tol = float(self.config.duplicate_time_tolerance_s)
        if tol < 0.0:
            raise ValueError("duplicate_time_tolerance_s must be non-negative")

        duplicates: list[dict[str, Any]] = []
        i = 0
        while i < len(ordered):
            j = i + 1
            ids = [ordered[i].keyframe_id]
            t0 = ordered[i].t

            if tol == 0.0:
                while j < len(ordered) and ordered[j].t == t0:
                    ids.append(ordered[j].keyframe_id)
                    j += 1
            else:
                while j < len(ordered) and abs(ordered[j].t - t0) <= tol:
                    ids.append(ordered[j].keyframe_id)
                    j += 1

            if len(ids) > 1:
                duplicates.append(
                    {
                        "t": float(t0),
                        "count": len(ids),
                        "keyframe_ids": ids[:20],
                        "raw_indices": [
                            ordered[k].raw_index for k in range(i, j)
                        ][:20],
                    }
                )

            i = j

        if duplicates:
            raise ValueError(
                "Duplicate paired V-Spline keyframe timestamps found. "
                "This adapter intentionally does not deduplicate. Upstream "
                "same_t_bucket aggregation must produce exactly one paired "
                f"keyframe per t. Duplicate preview: {duplicates[:10]}"
            )

        return ordered

    @staticmethod
    def verify_strictly_increasing(samples: Sequence[PreparedVSplineSample]) -> None:
        """Verify t is strictly increasing after sorting and duplicate check."""
        if len(samples) < 2:
            raise ValueError("At least two paired samples are required for V-Spline")

        t = np.asarray([s.t for s in samples], dtype=float)
        dt = np.diff(t)
        if not np.all(dt > 0.0):
            raise ValueError(
                "V-Spline times must be strictly increasing after sorting. "
                f"min_dt={float(np.min(dt)) if dt.size else None}"
            )

    def filter_position_motion_outliers(
        self,
        ordered_samples: Sequence[PreparedVSplineSample],
    ) -> tuple[list[PreparedVSplineSample], dict[str, Any]]:
        """Drop paired samples that imply impossible motion between neighbors.

        The filter is intentionally conservative: it compares position-derived
        step speed to both an absolute gate and the reported ADS-B velocity near
        that edge.  It removes endpoint samples attached to a single impossible
        edge and interior samples attached to two impossible edges.  It does not
        try to smooth the path; it only removes catastrophic position rows that
        would poison segmentation and the final spline fit.
        """
        samples = list(ordered_samples)
        cfg = self.config
        if not cfg.enable_position_motion_outlier_filter:
            return samples, {"enabled": False, "reason": "disabled"}
        if len(samples) < max(3, int(cfg.position_outlier_min_samples_remaining)):
            return samples, {"enabled": False, "reason": "too_few_samples", "sample_count": len(samples)}
        gate_abs = float(cfg.position_outlier_speed_gate_mps)
        factor = float(cfg.position_outlier_speed_factor)
        max_iterations = max(1, int(cfg.position_outlier_max_iterations))
        min_remaining = max(2, int(cfg.position_outlier_min_samples_remaining))
        if gate_abs <= 0 or factor <= 0:
            raise ValueError("position outlier speed gate and speed factor must be positive")

        removed: list[dict[str, Any]] = []
        iteration_reports: list[dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            if len(samples) <= min_remaining:
                break
            t = np.asarray([s.t for s in samples], dtype=float)
            y = np.asarray([s.y for s in samples], dtype=float)
            v = np.asarray([s.v for s in samples], dtype=float)
            dt = np.diff(t)
            if dt.size == 0 or not np.all(dt > 0):
                break
            step_speed = np.linalg.norm(np.diff(y, axis=0), axis=1) / np.maximum(dt, 1e-9)
            reported_speed = np.linalg.norm(v, axis=1)
            edge_reported = 0.5 * (reported_speed[:-1] + reported_speed[1:])
            edge_gate = np.maximum(gate_abs, factor * np.maximum(edge_reported, 1.0))
            bad_edges = step_speed > edge_gate
            bad_edge_indices = np.flatnonzero(bad_edges)
            if bad_edge_indices.size == 0:
                iteration_reports.append(
                    {
                        "iteration": iteration,
                        "sample_count": len(samples),
                        "bad_edge_count": 0,
                        "max_step_speed_mps": float(np.max(step_speed)) if step_speed.size else None,
                    }
                )
                break

            adjacent_bad_counts = np.zeros(len(samples), dtype=int)
            severity = np.zeros(len(samples), dtype=float)
            ratio = step_speed / np.maximum(edge_gate, 1e-9)
            for edge_idx in bad_edge_indices:
                adjacent_bad_counts[edge_idx] += 1
                adjacent_bad_counts[edge_idx + 1] += 1
                severity[edge_idx] = max(severity[edge_idx], ratio[edge_idx])
                severity[edge_idx + 1] = max(severity[edge_idx + 1], ratio[edge_idx])

            candidates: set[int] = set()
            # Interior bridge outlier: both incoming and outgoing edges are bad.
            for i in range(1, len(samples) - 1):
                if adjacent_bad_counts[i] >= 2:
                    candidates.add(i)
            # Endpoint outlier: the endpoint is attached to one impossible edge
            # and the following/preceding local track is not also impossible.
            if bad_edges[0] and (bad_edges.size == 1 or not bad_edges[1]):
                candidates.add(0)
            if bad_edges[-1] and (bad_edges.size == 1 or not bad_edges[-2]):
                candidates.add(len(samples) - 1)

            if not candidates:
                # Ambiguous run of bad edges: remove the most severe attached
                # sample, but never remove so many samples that fitting becomes
                # impossible.  This keeps the filter from silently deleting whole
                # maneuver intervals.
                candidates.add(int(np.argmax(severity)))

            candidate_list = sorted(candidates, key=lambda idx: (-severity[idx], idx))
            removable_count = max(0, len(samples) - min_remaining)
            candidate_list = candidate_list[:removable_count]
            if not candidate_list:
                break

            removed_this_iter = []
            keep_mask = np.ones(len(samples), dtype=bool)
            for idx in candidate_list:
                keep_mask[idx] = False
                sample = samples[idx]
                removed_info = {
                    "iteration": iteration,
                    "keyframe_id": sample.keyframe_id,
                    "raw_index": sample.raw_index,
                    "t": float(sample.t),
                    "adjacent_bad_edge_count": int(adjacent_bad_counts[idx]),
                    "max_edge_speed_ratio": float(severity[idx]),
                }
                removed.append(removed_info)
                removed_this_iter.append(removed_info)
            iteration_reports.append(
                {
                    "iteration": iteration,
                    "sample_count_before": len(samples),
                    "bad_edge_count": int(bad_edge_indices.size),
                    "removed_count": len(candidate_list),
                    "max_step_speed_mps": float(np.max(step_speed)) if step_speed.size else None,
                    "max_step_speed_ratio": float(np.max(ratio)) if ratio.size else None,
                    "removed_preview": removed_this_iter[:10],
                }
            )
            samples = [s for keep, s in zip(keep_mask, samples) if keep]

        return samples, {
            "enabled": True,
            "policy": "paired_position_motion_gate",
            "absolute_speed_gate_mps": gate_abs,
            "reported_speed_factor": factor,
            "max_iterations": max_iterations,
            "input_sample_count": len(ordered_samples),
            "output_sample_count": len(samples),
            "removed_count": len(removed),
            "removed_preview": removed[:20],
            "iterations": iteration_reports,
        }

    # ---------------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------------

    def split_samples(
        self,
        ordered_samples: Sequence[PreparedVSplineSample],
    ) -> tuple[list[PreparedVSplineSegment], dict[str, Any]]:
        """Split paired samples into fit segments.

        Splitting is based only on gaps between paired observations.  Segment
        boundaries are therefore always real paired keyframes, so hard endpoint
        constraints in the core are meaningful.
        """
        if not ordered_samples:
            raise ValueError("Cannot split empty sample list")

        max_gap = self.config.max_gap_s
        groups: list[tuple[list[PreparedVSplineSample], str]] = []
        current: list[PreparedVSplineSample] = [ordered_samples[0]]

        for prev, cur in zip(ordered_samples[:-1], ordered_samples[1:]):
            gap = cur.t - prev.t
            if max_gap is not None and gap > max_gap:
                groups.append((current, f"gap>{max_gap:g}s"))
                current = [cur]
            else:
                current.append(cur)
        groups.append((current, "end_of_track"))

        segments: list[PreparedVSplineSegment] = []
        short_segments: list[dict[str, Any]] = []

        for idx, (group, reason) in enumerate(groups, start=1):
            if len(group) < self.config.min_segment_observations:
                info = {
                    "candidate_segment_id": f"seg_{idx:04d}",
                    "n_observations": len(group),
                    "t0": float(group[0].t),
                    "t1": float(group[-1].t),
                    "reason": reason,
                }
                short_segments.append(info)
                if self.config.fail_on_short_segment:
                    raise ValueError(
                        "Segment shorter than min_segment_observations. "
                        f"{info}"
                    )
                continue

            t = np.asarray([s.t for s in group], dtype=float)
            dt = np.diff(t)
            segments.append(
                PreparedVSplineSegment(
                    segment_id=f"seg_{len(segments) + 1:04d}",
                    samples=tuple(group),
                    split_reason=reason,
                    t0=float(t[0]),
                    t1=float(t[-1]),
                    dt_min_s=float(np.min(dt)) if dt.size else None,
                    dt_max_s=float(np.max(dt)) if dt.size else None,
                )
            )

        if not segments:
            raise ValueError("No usable V-Spline segments after splitting")

        diag = {
            "max_gap_s": max_gap,
            "min_segment_observations": self.config.min_segment_observations,
            "candidate_segment_count": len(groups),
            "usable_segment_count": len(segments),
            "short_segments": short_segments,
            "segment_lengths": [seg.n_observations for seg in segments],
            "segment_time_ranges": [
                {"segment_id": seg.segment_id, "t0": seg.t0, "t1": seg.t1}
                for seg in segments
            ],
        }
        return segments, diag


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _finite_float(value: Any, field_name: str) -> float:
    try:
        x = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} is not numeric: {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"{field_name} is not finite: {value!r}")
    return x


def _ints_from_any(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    try:
        return [int(value)]
    except Exception:
        return []


def _strings_from_any(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]
