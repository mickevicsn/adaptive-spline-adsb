export class ViewerState {
  constructor(payload) {
    this.payload = payload;
    this.samples = payload.samples || [];
    this.rawPoints = payload.rawPositionPoints || [];
    this.config = payload.config || {};
    this.piecewise = payload.piecewise || {};
    this.segments = Array.isArray(payload.segments) ? payload.segments : [];
    this.segmentBoundaries = Array.isArray(payload.segmentBoundaries) ? payload.segmentBoundaries : [];
    this.segmentColoring = payload.segmentColoring || {};

    this.segmentById = new Map();
    this.segments.forEach((segment, index) => {
      if (!segment || typeof segment !== "object") {
        return;
      }
      if (segment.segment_id == null) {
        segment.segment_id = `seg_${String(index).padStart(4, "0")}`;
      }
      segment.segment_index = Number.isFinite(Number(segment.segment_index))
        ? Number(segment.segment_index)
        : index;
      this.segmentById.set(String(segment.segment_id), segment);
    });

    this.playing = true;
    this.currentRelT = 0;
    this.currentIndex = 0;
    this.lastFrameMs = null;

    this.orbitHeadingDeg = this.config.cameraInitialHeadingDeg || 0;
    this.orbitPitchDeg = this.config.cameraInitialPitchDeg || -28;
    this.userCameraRangeM = this.config.cameraRangeM || 650;

    this.minSimulationSpeed = Number(this.config.minSimulationSpeed || 0.25);
    this.maxSimulationSpeed = Number(this.config.maxSimulationSpeed || 8);
    this.simulationSpeedStep = Number(this.config.simulationSpeedStep || 0.25);
    this.speed = this.clamp(
      Number(this.config.initialSpeed || 1),
      this.minSimulationSpeed,
      this.maxSimulationSpeed,
    );

    this.showVelocityVector = true;
    this.showAccelerationVector = true;
    this.showTrail = true;
    this.displaySegments = this.config.displaySegments !== false;
  }

  validate() {
    if (!this.samples.length) {
      throw new Error("No animation samples.");
    }

    if (!this.rawPoints.length) {
      throw new Error("No raw position points.");
    }
  }

  clamp(value, minValue, maxValue) {
    return Math.max(minValue, Math.min(maxValue, value));
  }

  normalizeDeg(value) {
    return ((value % 360) + 360) % 360;
  }

  setSpeed(value) {
    const rounded = Math.round(Number(value) * 100) / 100;
    this.speed = this.clamp(rounded, this.minSimulationSpeed, this.maxSimulationSpeed);
    return this.speed;
  }

  adjustSpeed(direction) {
    return this.setSpeed(this.speed + Number(direction || 0) * this.simulationSpeedStep);
  }

  speedLabel() {
    return `${Number(this.speed || 1).toFixed(2)}x`;
  }

  indexForTime(tRel) {
    let lo = 0;
    let hi = this.samples.length - 1;

    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);

      if (this.samples[mid].t_rel_s <= tRel) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    return this.clamp(hi, 0, this.samples.length - 1);
  }

  currentSample() {
    return this.samples[this.currentIndex];
  }

  maxRelativeTime() {
    return this.samples[this.samples.length - 1].t_rel_s;
  }

  resetPlayback() {
    this.currentRelT = 0;
    this.currentIndex = 0;
    this.playing = false;
  }

  _number(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  _segmentIndexBounds(segment) {
    const start = segment.raw_point_start_index
      ?? segment.start_sample_index
      ?? segment.sample_index_start
      ?? segment.render_keyframe_start_index;
    const end = segment.raw_point_end_index
      ?? segment.end_sample_index
      ?? segment.sample_index_end
      ?? segment.render_keyframe_end_index;

    const startIndex = Number.isFinite(Number(start)) ? Number(start) : null;
    const endIndex = Number.isFinite(Number(end)) ? Number(end) : null;
    return { startIndex, endIndex };
  }

  _segmentTimeBounds(segment) {
    const t0 = this._number(segment.t0 ?? segment.render_time_start ?? segment.start_time);
    const t1 = this._number(segment.t1 ?? segment.render_time_end ?? segment.end_time);
    return { t0, t1 };
  }

  segmentForRawPoint(point, pointIndex = null) {
    if (!point) {
      return null;
    }

    const segmentId = point.segment_id ?? point.segmentId;
    if (segmentId != null && this.segmentById.has(String(segmentId))) {
      return this.segmentById.get(String(segmentId));
    }

    const rawIndex = Number.isFinite(Number(point.point_index))
      ? Number(point.point_index)
      : (Number.isFinite(Number(pointIndex)) ? Number(pointIndex) : null);

    if (rawIndex != null) {
      for (const segment of this.segments) {
        const { startIndex, endIndex } = this._segmentIndexBounds(segment);
        if (startIndex == null || endIndex == null) {
          continue;
        }
        const lo = Math.min(startIndex, endIndex);
        const hi = Math.max(startIndex, endIndex);
        if (rawIndex >= lo && rawIndex <= hi) {
          return segment;
        }
      }
    }

    const t = this._number(point.t);
    if (t == null) {
      return null;
    }

    for (const segment of this.segments) {
      const { t0, t1 } = this._segmentTimeBounds(segment);
      if (t0 == null || t1 == null) {
        continue;
      }
      const lo = Math.min(t0, t1);
      const hi = Math.max(t0, t1);
      if (t >= lo - 1e-6 && t <= hi + 1e-6) {
        return segment;
      }
    }

    return null;
  }

  segmentForTime(t) {
    const time = this._number(t);
    if (time == null) {
      return null;
    }

    for (const segment of this.segments) {
      const { t0, t1 } = this._segmentTimeBounds(segment);
      if (t0 == null || t1 == null) {
        continue;
      }
      const lo = Math.min(t0, t1);
      const hi = Math.max(t0, t1);
      if (time >= lo - 1e-6 && time <= hi + 1e-6) {
        return segment;
      }
    }

    return null;
  }

  segmentColorizedRawPointCount() {
    let count = 0;
    for (let i = 0; i < this.rawPoints.length; i++) {
      if (this.segmentForRawPoint(this.rawPoints[i], i)) {
        count += 1;
      }
    }
    return count;
  }
}
