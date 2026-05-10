import {
  cartesianFromPoint,
  cartesianFromSample,
  deltaEastNorthMeters,
  heightFor,
  heightForPoint,
  shiftedPointAlongSegment,
  vectorGlyphPositionsFromComponents,
} from "./geometry.js";

export class CesiumFlightScene {
  constructor(containerId, state) {
    this.containerId = containerId;
    this.state = state;
    this.viewer = null;

    this.connectionCollection = null;
    this.connectionPrimitives = [];
    this.connectionSegmentTimes = [];
    this.connectionMaterialCache = new Map();
    this.reconstructedPathEntity = null;
    this.rawPointCollection = null;
    this.rawPointPrimitives = [];

    this.trailEntity = null;
    this.vectorCollection = null;
    this.velocityGlyph = null;
    this.accelerationGlyph = null;
    this.aircraftEntity = null;
  }

  initialize() {
    this.state.validate();
    this.viewer = this._createViewer();
    this._configureViewer();
    this._buildConnectionLines();
    this._buildRawDots();
    this._buildReconstructedPathLine();
    this._buildAnimatedEntities();
    this._buildLegend();
  }

  _createViewer() {
    const cartoProvider = new Cesium.UrlTemplateImageryProvider({
      url: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
      subdomains: ["a", "b", "c", "d"],
      minimumLevel: 0,
      maximumLevel: 19,
      credit: "Map tiles by CARTO. Data by OpenStreetMap.",
    });

    cartoProvider.errorEvent.addEventListener(error => {
      console.error("CARTO tile provider error:", error);
    });

    return new Cesium.Viewer(this.containerId, {
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      selectionIndicator: false,
      infoBox: false,
      shouldAnimate: true,
      baseLayer: new Cesium.ImageryLayer(cartoProvider),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    });
  }

  _configureViewer() {
    this.viewer.scene.globe.depthTestAgainstTerrain = false;

    if (this.viewer.scene.skyAtmosphere) {
      this.viewer.scene.skyAtmosphere.show = true;
    }

    this.viewer.scene.requestRenderMode = false;

    const controller = this.viewer.scene.screenSpaceCameraController;
    controller.enableRotate = false;
    controller.enableTranslate = false;
    controller.enableZoom = false;
    controller.enableTilt = false;
    controller.enableLook = false;
  }

  _connectionLineWidthPx() {
    const cfg = this.state.config;
    const rawPointSizePx = cfg.rawPointPixelSize || 8;

    return Math.max(
      1,
      Math.min(cfg.connectionLineWidthPx || 2, rawPointSizePx - 4),
    );
  }

  _clampUnit(value) {
    return Math.max(0, Math.min(1, Number(value)));
  }

  _futureTrackFadeAlphaForLeadSeconds(leadSeconds) {
    const cfg = this.state.config;
    const fadeStartS = Number(cfg.futureTrackFadeStartS ?? 120);
    const fadeDurationS = Math.max(1, Number(cfg.futureTrackFadeDurationS ?? 480));
    const minAlpha = this._clampUnit(cfg.futureTrackMinAlpha ?? 0.08);

    if (!Number.isFinite(leadSeconds) || leadSeconds <= fadeStartS) {
      return 1.0;
    }

    const fadeProgress = this._clampUnit((leadSeconds - fadeStartS) / fadeDurationS);
    return minAlpha + (1.0 - minAlpha) * (1.0 - fadeProgress);
  }

  _segmentPalette() {
    // Do not start with red: red is the fallback/upcoming raw colour, so using
    // it for segment 0 makes successful segment colouring look like a failure.
    return [
      "#00e5ff",
      "#ffb000",
      "#7cff4f",
      "#d25cff",
      "#00ffa6",
      "#ff6ad5",
      "#4d7cff",
      "#f6ff00",
      "#00c26e",
      "#ff8a3d",
      "#8ad7ff",
      "#ffffff",
    ];
  }

  _segmentCssColor(segment) {
    if (!segment) {
      return "#ff3c3c";
    }

    const palette = this._segmentPalette();
    const rawIndex = Number(segment.segment_color_index ?? segment.segment_index ?? 0);
    const index = Number.isFinite(rawIndex) ? Math.abs(Math.trunc(rawIndex)) : 0;
    return palette[index % palette.length];
  }

  _segmentColor(segment, alpha = 1.0) {
    if (!segment) {
      return Cesium.Color.RED.withAlpha(alpha);
    }

    return Cesium.Color.fromCssColorString(this._segmentCssColor(segment)).withAlpha(alpha);
  }

  _passedRawPointColor() {
    return Cesium.Color.fromBytes(150, 150, 150, 255);
  }

  _rawPointColor(point, currentT, passedIndex, pointIndex) {
    const pointT = Number(point.t || 0);

    // Playback state takes precedence over segment colouring.  In particular,
    // v-spline raw points are segment-coloured while still ahead of the
    // aircraft, but should turn gray once they are in the past.
    if (pointIndex <= passedIndex) {
      return this._passedRawPointColor();
    }

    const alpha = this._futureTrackFadeAlphaForLeadSeconds(pointT - currentT);

    if (this.state.displaySegments !== false) {
      const segment = this.state.segmentForRawPoint(point, pointIndex);
      if (segment) {
        return this._segmentColor(segment, alpha);
      }
    }

    return Cesium.Color.RED.withAlpha(alpha);
  }

  _connectionMaterialForAlpha(alpha) {
    const effectiveAlpha = this._clampUnit(alpha) * 0.95;
    const key = effectiveAlpha.toFixed(2);

    if (!this.connectionMaterialCache.has(key)) {
      this.connectionMaterialCache.set(
        key,
        Cesium.Material.fromType("Color", {
          color: Cesium.Color.DEEPSKYBLUE.withAlpha(effectiveAlpha),
        }),
      );
    }

    return this.connectionMaterialCache.get(key);
  }

  _buildConnectionLines() {
    const cfg = this.state.config;
    const rawPoints = this.state.rawPoints;

    if (cfg.showRawConnectionLines === false) {
      this.connectionCollection = null;
      this.connectionPrimitives = [];
      this.connectionSegmentTimes = [];
      return;
    }

    const endpointGapM = cfg.connectionLineEndpointGapM || 11;
    const connectionZOffsetM = -0.4;

    this.connectionCollection = this.viewer.scene.primitives.add(
      new Cesium.PolylineCollection(),
    );

    this.connectionPrimitives = [];
    this.connectionSegmentTimes = [];

    for (let i = 0; i < rawPoints.length - 1; i++) {
      const a = rawPoints[i];
      const b = rawPoints[i + 1];

      const delta = deltaEastNorthMeters(a, b);
      const dist = Math.hypot(delta.east, delta.north);

      if (dist < 1e-6) {
        continue;
      }

      // If points are very close, do not draw a segment at all.
      // The dots already show continuity there.
      if (dist <= endpointGapM * 2.1) {
        continue;
      }

      const ux = delta.east / dist;
      const uy = delta.north / dist;

      const zA = heightForPoint(a);
      const zB = heightForPoint(b);

      const startGap = endpointGapM;
      const endGap = endpointGapM;

      const zStart = zA + (zB - zA) * (startGap / dist) + connectionZOffsetM;
      const zEnd = zB - (zB - zA) * (endGap / dist) + connectionZOffsetM;

      const start = shiftedPointAlongSegment(a, ux * startGap, uy * startGap, zStart);
      const end = shiftedPointAlongSegment(b, -ux * endGap, -uy * endGap, zEnd);

      const primitive = this.connectionCollection.add({
        positions: [start, end],
        width: this._connectionLineWidthPx(),
        material: this._connectionMaterialForAlpha(1.0),
      });

      this.connectionPrimitives.push(primitive);
      this.connectionSegmentTimes.push({
        startT: Math.min(Number(a.t || 0), Number(b.t || 0)),
      });
    }
  }

  rebuildConnectionLines() {
    if (this.connectionCollection) {
      this.viewer.scene.primitives.remove(this.connectionCollection);
      this.connectionCollection = null;
      this.connectionPrimitives = [];
      this.connectionSegmentTimes = [];
      this.connectionMaterialCache.clear();
    }

    this._buildConnectionLines();
  }

  _buildRawDots() {
    const cfg = this.state.config;
    this.rawPointCollection = this.viewer.scene.primitives.add(
      new Cesium.PointPrimitiveCollection(),
    );

    this.rawPointPrimitives = [];

    for (let i = 0; i < this.state.rawPoints.length; i++) {
      const p = this.state.rawPoints[i];
      const primitive = this.rawPointCollection.add({
        position: cartesianFromPoint(p, 0),
        pixelSize: cfg.rawPointPixelSize || 8,
        color: this._rawPointColor(p, Number.NEGATIVE_INFINITY, -1, i),
        outlineColor: Cesium.Color.TRANSPARENT,
        outlineWidth: 0,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      });

      this.rawPointPrimitives.push(primitive);
    }
  }


  _buildReconstructedPathLine() {
    const cfg = this.state.config;
    if (cfg.showReconstructedPathLine !== true) {
      return;
    }

    const samples = this.state.samples || [];
    if (samples.length < 2) {
      return;
    }

    this.reconstructedPathEntity = this.viewer.entities.add({
      name: "Kalman RTS reconstructed path",
      polyline: {
        positions: samples.map(sample => cartesianFromSample(sample)),
        width: cfg.reconstructedPathLineWidthPx || 4,
        material: Cesium.Color.CYAN.withAlpha(0.95),
        arcType: Cesium.ArcType.NONE,
      },
    });
  }

  _buildAnimatedEntities() {
    const cfg = this.state.config;
    const firstSample = this.state.samples[0];

    this.trailEntity = this.viewer.entities.add({
      name: "Animated passed trail",
      polyline: {
        positions: this.trailPositions(0),
        width: cfg.trailWidthPx,
        material: Cesium.Color.CYAN.withAlpha(0.95),
        arcType: Cesium.ArcType.NONE,
      },
    });

    this.velocityGlyph = this._createVectorGlyph(
      "Velocity vector",
      Cesium.Color.fromCssColorString("#2b63ff"),
    );

    this.accelerationGlyph = this._createVectorGlyph(
      "Acceleration vector",
      Cesium.Color.fromCssColorString("#2bd957"),
    );

    this.aircraftEntity = this.viewer.entities.add({
      name: "Aircraft point",
      position: cartesianFromSample(firstSample),
      point: {
        pixelSize: cfg.aircraftPointPixelSize,
        color: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.TRANSPARENT,
        outlineWidth: 0,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }




  _segmentLegendLabel(segment) {
    if (!segment) {
      return "Current segment: n/a";
    }

    const id = segment.segment_id || `seg_${segment.segment_index ?? "?"}`;
    return `Current segment: ${id}${segment.regime_label ? ` (${segment.regime_label})` : ""}`;
  }

  _buildLegend(sample = null) {
    const legend = document.getElementById("legend");
    if (!legend) {
      return;
    }

    const activeSegment = this.state.displaySegments === false
      ? null
      : this.state.segmentForTime(Number((sample || this.state.currentSample() || {}).t));

    const segmentLegendItems = this.state.displaySegments === false
      ? []
      : [[
        "dot",
        activeSegment ? this._segmentCssColor(activeSegment) : "rgba(255,255,255,0.35)",
        this._segmentLegendLabel(activeSegment),
      ]];

    const items = [
      ...(segmentLegendItems.length ? segmentLegendItems : [["dot", "#ff3c3c", "Upcoming raw ADS-B point"]]),
      ...(segmentLegendItems.length ? [["dot", "rgba(255,255,255,0.35)", "Future segment points fade by alpha"]] : [["dot", "rgba(255,60,60,0.3)", ">2 min future fade"]]),
      ["dot", "#969696", "Passed raw point"],
      ["dot", "#ffffff", "Current aircraft"],
      ...(this.state.config.showRawConnectionLines === false ? [] : [["line", "#00b4ff", "Raw connection"]]),
      ...(this.state.config.showReconstructedPathLine === true ? [["line", "#00ffff", "Reconstructed path"]] : []),
      ["line", "#2b63ff", "Velocity vector"],
      ["line", "#2bd957", "Acceleration vector"],
    ];

    const legendKey = items.map(([kind, color, label]) => `${kind}:${color}:${label}`).join("|");
    if (legend.dataset.legendKey === legendKey) {
      return;
    }
    legend.dataset.legendKey = legendKey;

    legend.innerHTML = items
      .map(([kind, color, label]) => `
        <div class="legend-row">
          ${kind === "dot"
            ? `<span class="dot" style="background:${color}"></span>`
            : `<span class="legend-line" style="color:${color}"></span>`}
          <span>${label}</span>
        </div>`)
      .join("");
  }

  _createVectorGlyph(name, color) {
    if (!this.vectorCollection) {
      this.vectorCollection = this.viewer.scene.primitives.add(
        new Cesium.PolylineCollection(),
      );
    }

    const arrowMaterial = Cesium.Material.fromType("PolylineArrow", {
      color: color.withAlpha(0.98),
    });

    const zero = Cesium.Cartesian3.ZERO;

    const arrow = this.vectorCollection.add({
      id: `${name} single arrow`,
      show: false,
      positions: [zero, zero],
      width: 14,
      material: arrowMaterial,
      // Keep vector arrows visible when another primitive would normally hide them.
      // Cesium renders depth-failed polyline fragments with this material.
      depthFailMaterial: arrowMaterial,
    });

    // Some Cesium builds expose depthFailMaterial as an assignable property rather
    // than honoring it only from constructor options, so set it explicitly too.
    arrow.depthFailMaterial = arrowMaterial;

    return { arrow };
  }

  _setVectorGlyph(glyph, positions, shouldShow) {
    if (!glyph || !glyph.arrow) {
      return;
    }

    const show = Boolean(shouldShow && positions && positions.valid);
    glyph.arrow.show = show;

    if (!show || !positions) {
      return;
    }

    glyph.arrow.positions = positions.segment;
  }

  _velocityGlyphPositions(sample) {
    return vectorGlyphPositionsFromComponents(
      sample,
      sample.vel_east_mps,
      sample.vel_north_mps,
      1.90,
      96,
      240,
      0.0,
      0.0,
      0.0,
      sample.vel_up_mps,
      0.0,
    );
  }

  _velocityDisplayOffsetMeters(sample) {
    const east = Number(sample.vel_east_mps);
    const north = Number(sample.vel_north_mps);

    if (!Number.isFinite(east) || !Number.isFinite(north)) {
      return null;
    }

    const mag = Math.hypot(east, north);
    if (!Number.isFinite(mag) || mag < 1e-6) {
      return null;
    }

    const lengthM = Math.max(96, Math.min(240, mag * 1.90));

    const up = Number(sample.vel_up_mps || 0);
    const mag3d = Math.hypot(east, north, up);

    return {
      east: (east / mag) * lengthM,
      north: (north / mag) * lengthM,
      up: Number.isFinite(mag3d) && mag3d > 1e-6 ? (up / mag3d) * lengthM : 0.0,
    };
  }

  _recentAccelerationComponents(currentIndex) {
    const current = this.state.samples[currentIndex];
    if (!current) {
      return null;
    }

    const currentT = Number(current.t_rel_s || 0);
    for (let i = currentIndex; i >= 0; i--) {
      const sample = this.state.samples[i];
      if (!sample) {
        continue;
      }

      if (currentT - Number(sample.t_rel_s || 0) > 15) {
        break;
      }

      const east = Number(sample.accel_east_mps2 || 0);
      const north = Number(sample.accel_north_mps2 || 0);
      const up = Number(sample.accel_up_mps2 || 0);
      if (!Number.isFinite(east) || !Number.isFinite(north) || !Number.isFinite(up)) {
        continue;
      }

      if (Math.hypot(east, north, up) > 0.05) {
        return { east, north, up };
      }
    }

    return null;
  }

  _accelerationGlyphPositions(sample, currentIndex) {
    const velocityTip = this._velocityDisplayOffsetMeters(sample);
    if (!velocityTip) {
      return vectorGlyphPositionsFromComponents(sample, 0, 0, 1, 0, 0, 0.0);
    }

    const acceleration = this._recentAccelerationComponents(currentIndex);
    if (!acceleration) {
      return vectorGlyphPositionsFromComponents(sample, 0, 0, 1, 0, 0, 0.0);
    }

    return vectorGlyphPositionsFromComponents(
      sample,
      acceleration.east,
      acceleration.north,
      14,
      36,
      100,
      0.0,
      velocityTip.east,
      velocityTip.north,
      acceleration.up,
      velocityTip.up,
    );
  }



  trailPositions(index) {
    const sample = this.state.samples[index];
    const trailStartT = Math.max(0, sample.t_rel_s - this.state.config.trailSeconds);

    let startIndex = index;
    while (startIndex > 0 && this.state.samples[startIndex].t_rel_s >= trailStartT) {
      startIndex -= 1;
    }

    const positions = [];
    for (let i = startIndex; i <= index; i++) {
      positions.push(cartesianFromSample(this.state.samples[i]));
    }

    return positions;
  }

  lastPassedRawPointIndexForTime(currentT) {
    let lo = 0;
    let hi = this.state.rawPoints.length - 1;
    let result = -1;

    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      const pointT = Number(this.state.rawPoints[mid].t || 0);

      if (pointT <= currentT) {
        result = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    return result;
  }

  update(sample, currentIndex) {
    this.aircraftEntity.position = cartesianFromSample(sample);

    this.trailEntity.show = this.state.showTrail;

    this.trailEntity.polyline.positions = this.trailPositions(currentIndex);
    this.trailEntity.polyline.width = this.state.config.trailWidthPx;

    this._setVectorGlyph(
      this.velocityGlyph,
      this._velocityGlyphPositions(sample),
      this.state.showVelocityVector,
    );
    this._setVectorGlyph(
      this.accelerationGlyph,
      this._accelerationGlyphPositions(sample, currentIndex),
      this.state.showAccelerationVector,
    );



    return this.updatePassedPointColors(sample);
  }

  updatePassedPointColors(sample) {
    const currentT = Number(sample.t || 0);
    const passedIndex = this.lastPassedRawPointIndexForTime(currentT);
    const rawPointSizePx = this.state.config.rawPointPixelSize || 8;

    for (let i = 0; i < this.state.rawPoints.length; i++) {
      const primitive = this.rawPointPrimitives[i];

      if (!primitive) {
        continue;
      }

      primitive.color = this._rawPointColor(
        this.state.rawPoints[i],
        currentT,
        passedIndex,
        i,
      );

      primitive.outlineColor = Cesium.Color.TRANSPARENT;
      primitive.outlineWidth = 0;
      primitive.pixelSize = rawPointSizePx;
    }

    this.updateConnectionLineTransparency(currentT);

    return passedIndex;
  }

  updateConnectionLineTransparency(currentT) {
    for (let i = 0; i < this.connectionPrimitives.length; i++) {
      const primitive = this.connectionPrimitives[i];
      const segment = this.connectionSegmentTimes[i];

      if (!primitive || !segment) {
        continue;
      }

      const segmentLeadS = Math.max(0, segment.startT - currentT);
      primitive.material = this._connectionMaterialForAlpha(
        this._futureTrackFadeAlphaForLeadSeconds(segmentLeadS),
      );
    }
  }

  updateSegmentLegend() {
    this._buildLegend();
  }

  updateRawPointStyle() {
    const size = this.state.config.rawPointPixelSize || 8;
    const sample = this.state.currentSample();
    const currentT = sample ? Number(sample.t || 0) : Number.NEGATIVE_INFINITY;
    const passedIndex = Number.isFinite(currentT)
      ? this.lastPassedRawPointIndexForTime(currentT)
      : -1;

    for (let i = 0; i < this.rawPointPrimitives.length; i++) {
      const primitive = this.rawPointPrimitives[i];
      if (!primitive) {
        continue;
      }
      primitive.pixelSize = size;
      primitive.color = this._rawPointColor(
        this.state.rawPoints[i],
        currentT,
        passedIndex,
        i,
      );
      primitive.outlineColor = Cesium.Color.TRANSPARENT;
      primitive.outlineWidth = 0;
    }

    for (const primitive of this.connectionPrimitives) {
      primitive.width = this._connectionLineWidthPx();
    }
  }

  renderStatus(sample, passedRawPointIndex, cameraController) {
    this._buildLegend(sample);

    const speed = sample.ground_speed_kt == null ? "n/a" : sample.ground_speed_kt.toFixed(1) + " kt";
    const accel = sample.accel_horizontal_mps2 == null ? "n/a" : sample.accel_horizontal_mps2.toFixed(2) + " m/s²";
    const rawPointSizePx = this.state.config.rawPointPixelSize || 8;

    document.getElementById("status").textContent =
      `raw position events=${this.state.rawPoints.length} | animation sample=${this.state.currentIndex + 1}/${this.state.samples.length}\n` +
      `last passed raw point index=${passedRawPointIndex + 1}/${this.state.rawPoints.length}\n` +
      `dot size=${rawPointSizePx}px | raw connections=${this.state.config.showRawConnectionLines === false ? "off" : "on"} | reconstructed path=${this.state.config.showReconstructedPathLine === true ? "on" : "off"}\n` +
      `segments=${this.state.displaySegments === false ? "hidden" : this.state.segments.length} | colorized raw points=${this.state.displaySegments === false ? 0 : this.state.segmentColorizedRawPointCount()}/${this.state.rawPoints.length} | active=${this.state.displaySegments === false ? "n/a" : ((this.state.segmentForTime(sample.t) || {}).segment_id || "n/a")}\n` +
      `t_rel=${sample.t_rel_s.toFixed(1)} s | lat=${sample.lat.toFixed(6)} lon=${sample.lon.toFixed(6)}\n` +
      `z=${heightFor(sample).toFixed(1)} m | camera range=${cameraController.cameraRangeFor(sample).toFixed(1)} m | wheel base range=${this.state.userCameraRangeM.toFixed(1)} m\n` +
      `heading=${sample.heading_deg.toFixed(1)}° | gs=${speed} | accel=${accel}\n` +
      `camera orbit heading=${this.state.orbitHeadingDeg.toFixed(1)}° pitch=${this.state.orbitPitchDeg.toFixed(1)}°`;
  }
}
