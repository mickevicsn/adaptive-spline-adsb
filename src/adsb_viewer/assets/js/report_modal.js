const REPORT_VIEWS = {
  Position: [
    "XY Top-Down Path",
    "XZ Side View",
    "YZ Side View",
    "X over Time",
    "Y over Time",
    "Z over Time",
  ],
  Velocity: [
    "X Component over Time",
    "Y Component over Time",
    "Z Component over Time",
    "XY Vector Animation",
    "XZ Vector Animation",
    "YZ Vector Animation",
  ],
  Acceleration: [
    "X Component over Time",
    "Y Component over Time",
    "Z Component over Time",
    "XY Vector Animation",
    "XZ Vector Animation",
    "YZ Vector Animation",
  ],
};

const REPORT_COLORS = {
  background: "rgba(4, 16, 34, 1)",
  grid: "rgba(255,255,255,0.09)",
  axis: "rgba(225,240,255,0.45)",
  text: "rgba(238,247,255,0.92)",
  mutedText: "rgba(221,238,255,0.68)",
  raw: "#ffb000",
  reconstructed: "#4caeff",
  current: "#ffffff",
  rawCurrent: "#ff7f50",
};

const REPORT_WINDOW_SECONDS_OPTIONS = [5, 10, 15, 20, 30, 45, 60];
const DEFAULT_REPORT_WINDOW_SECONDS = 30;
const AUTO_AXIS_MAX_SCALE_RATIO = 24;


function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function firstFinite(row, keys) {
  if (!row) return null;
  for (const key of keys) {
    const value = finiteNumber(row[key]);
    if (value != null) return value;
  }
  return null;
}

function cleanLabel(label) {
  const text = String(label || "").trim();
  return text || null;
}

function titleCaseMethod(methodId) {
  const id = String(methodId || "").trim();
  if (!id) return "Reconstructed";
  const special = {
    v_spline: "V-Spline",
    vspline: "V-Spline",
    kalman_rts: "Kalman RTS",
    raw_adsb: "Raw ADS-B",
  };
  const lower = id.toLowerCase();
  if (special[lower]) return special[lower];
  return id.replace(/[_-]+/g, " ").replace(/\b\w/g, letter => letter.toUpperCase());
}


function formatReportWindowTime(value) {
  if (!Number.isFinite(value)) return "—";
  const sign = value < 0 ? "−" : "";
  const totalSeconds = Math.max(0, Math.floor(Math.abs(value)));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${sign}T+${minutes}:${seconds}`;
}

function zValue(row) {
  return firstFinite(row, ["z_m", "height_above_field_m", "altitude_m", "baro_altitude_m", "geo_altitude_m"]);
}

function markerPath(ctx, x, y, radius) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
}

function roundedRect(ctx, x, y, width, height, radius) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, radius);
    return;
  }
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
}

export class ReportModal {
  constructor(state) {
    this.state = state;
    this.payload = state.payload || {};
    this.mounted = false;
    this.opened = false;
    this.animationRunning = false;
    this.animationFrame = null;
    this.animationStartMs = 0;
    this.animationDurationMs = 5000;
    this.resizeObserver = null;
    this.reportWindowSeconds = DEFAULT_REPORT_WINDOW_SECONDS;
    this.reportWindowStartS = null;
    this.reportWindowEndS = null;

    const samples = Array.isArray(state.samples) ? state.samples : [];
    const rawPoints = Array.isArray(state.rawPoints)
      ? state.rawPoints
      : (Array.isArray(this.payload.rawPositionPoints) ? this.payload.rawPositionPoints : []);
    const firstSample = samples[0] || null;
    const lastSample = samples[samples.length - 1] || null;
    const firstRawPoint = rawPoints[0] || null;
    const firstSampleT = finiteNumber(firstSample?.t);
    const firstSampleRel = finiteNumber(firstSample?.t_rel_s);
    const lastSampleT = finiteNumber(lastSample?.t);
    const lastSampleRel = finiteNumber(lastSample?.t_rel_s);
    const firstRawT = firstFinite(firstRawPoint, ["t", "timestamp", "time", "ts"]);
    this.firstSampleT = firstSampleT;
    this.firstRawT = firstRawT;
    this.maxSampleRel = lastSampleRel != null ? lastSampleRel : null;
    this.sampleTimestampSpan = firstSampleT != null && lastSampleT != null ? Math.max(0, lastSampleT - firstSampleT) : null;
    this.timeOffset = firstSampleT != null && firstSampleRel != null ? firstSampleT - firstSampleRel : null;
    this.rawTimeOffset = firstRawT != null && Math.abs(firstRawT) > 1000000 && (this.timeOffset == null || firstSampleT == null || Math.abs(firstSampleT) <= 1000000) ? firstRawT : null;
  }

  mount() {
    if (this.mounted) return;

    this.backdrop = document.getElementById("reportModalBackdrop");
    this.modal = this.backdrop?.querySelector(".report-modal") || null;
    this.closeButton = document.getElementById("reportCloseButton");
    this.dataTypeSelect = document.getElementById("reportDataType");
    this.viewSelect = document.getElementById("reportView");
    this.windowSelect = document.getElementById("reportWindowSeconds");
    this.trueAxisCheckbox = document.getElementById("reportTrueAxis");
    this.animateButton = document.getElementById("reportAnimateButton");
    this.canvas = document.getElementById("reportCanvas");
    this.noteEl = document.getElementById("reportNote");
    this.windowSummaryEl = document.getElementById("reportWindowSummary");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;

    if (!this.backdrop || !this.modal || !this.closeButton || !this.dataTypeSelect || !this.viewSelect || !this.windowSelect || !this.trueAxisCheckbox || !this.animateButton || !this.canvas || !this.ctx) {
      console.warn("Report modal disabled: report modal DOM was not found.");
      return;
    }

    this.populateReportWindowOptions();
    this.updateReportViewOptions();

    this.closeButton.addEventListener("click", () => this.close());
    this.backdrop.addEventListener("click", event => {
      if (event.target === this.backdrop) this.close();
    });
    document.addEventListener("keydown", event => {
      if (this.opened && event.key === "Escape") {
        event.preventDefault();
        this.close();
      }
    });

    this.dataTypeSelect.addEventListener("change", () => {
      this.updateReportViewOptions();
      this.stopAnimation();
      this.renderReportChart();
    });
    this.viewSelect.addEventListener("change", () => {
      this.stopAnimation();
      this.renderReportChart();
    });
    this.windowSelect.addEventListener("change", () => {
      this.reportWindowSeconds = this.getReportWindowSeconds();
      this.applyReportWindowFromEnd();
      this.stopAnimation();
      this.renderReportChart();
    });
    this.trueAxisCheckbox.addEventListener("change", () => {
      this.stopAnimation();
      this.renderReportChart();
    });
    this.animateButton.addEventListener("click", () => {
      if (this.animationRunning) {
        this.stopAnimation();
        this.renderReportChart();
      } else {
        this.renderReportAnimation();
      }
    });

    const handleResize = () => {
      if (this.opened && !this.animationRunning) {
        this.resizeCanvas();
        this.renderReportChart();
      }
    };
    if (typeof ResizeObserver === "function") {
      this.resizeObserver = new ResizeObserver(handleResize);
      this.resizeObserver.observe(this.canvas.parentElement || this.canvas);
    } else {
      window.addEventListener("resize", handleResize);
    }

    this.mounted = true;
  }

  open() {
    if (!this.mounted) this.mount();
    if (!this.backdrop) return;

    this.captureReportWindow();
    this.opened = true;
    this.backdrop.classList.remove("hidden");
    this.backdrop.setAttribute("aria-hidden", "false");
    requestAnimationFrame(() => {
      this.resizeCanvas();
      this.renderReportChart();
      this.dataTypeSelect?.focus({ preventScroll: true });
    });
  }

  close() {
    if (!this.backdrop) return;
    this.stopAnimation();
    this.opened = false;
    this.backdrop.classList.add("hidden");
    this.backdrop.setAttribute("aria-hidden", "true");
  }

  getReportDataType() {
    return this.dataTypeSelect?.value || "Position";
  }

  getReportView() {
    return this.viewSelect?.value || REPORT_VIEWS.Position[0];
  }

  populateReportWindowOptions() {
    if (!this.windowSelect) return;
    const current = this.getReportWindowSeconds();
    this.windowSelect.innerHTML = "";
    for (const seconds of REPORT_WINDOW_SECONDS_OPTIONS) {
      const option = document.createElement("option");
      option.value = String(seconds);
      option.textContent = `${seconds} seconds`;
      this.windowSelect.appendChild(option);
    }
    this.windowSelect.value = String(REPORT_WINDOW_SECONDS_OPTIONS.includes(current) ? current : DEFAULT_REPORT_WINDOW_SECONDS);
    this.reportWindowSeconds = this.getReportWindowSeconds();
  }

  getReportWindowSeconds() {
    const selected = Number(this.windowSelect?.value);
    if (REPORT_WINDOW_SECONDS_OPTIONS.includes(selected)) return selected;
    return DEFAULT_REPORT_WINDOW_SECONDS;
  }

  useTrueAxisMagnitudes() {
    return Boolean(this.trueAxisCheckbox?.checked);
  }

  updateReportViewOptions() {
    if (!this.viewSelect || !this.dataTypeSelect) return;
    const dataType = this.getReportDataType();
    const views = REPORT_VIEWS[dataType] || REPORT_VIEWS.Position;
    const previous = this.viewSelect.value;
    this.viewSelect.innerHTML = "";

    for (const view of views) {
      const option = document.createElement("option");
      option.value = view;
      option.textContent = view;
      this.viewSelect.appendChild(option);
    }

    this.viewSelect.value = views.includes(previous) ? previous : views[0];
  }

  resizeCanvas() {
    if (!this.canvas || !this.ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(rect.width || 1120));
    const height = Math.max(260, Math.round(rect.height || 620));
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const scaledWidth = Math.round(width * dpr);
    const scaledHeight = Math.round(height * dpr);

    if (this.canvas.width !== scaledWidth || this.canvas.height !== scaledHeight) {
      this.canvas.width = scaledWidth;
      this.canvas.height = scaledHeight;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cssWidth = width;
    this.cssHeight = height;
  }

  methodLabel() {
    return (
      cleanLabel(this.payload.selectedMethod?.label) ||
      cleanLabel(this.payload.selectedMethod?.name) ||
      cleanLabel(this.payload.method?.methodLabel) ||
      titleCaseMethod(this.payload.selectedMethodId || this.payload.method?.methodId)
    );
  }

  currentReportEndTime() {
    const currentRelT = finiteNumber(this.state.currentRelT);
    if (currentRelT != null) return currentRelT;

    const currentIndex = Number.isFinite(Number(this.state.currentIndex)) ? Number(this.state.currentIndex) : 0;
    const currentSample = typeof this.state.currentSample === "function"
      ? this.state.currentSample()
      : this.reconstructedRows()[currentIndex];
    const currentSampleTime = this.timeFor(currentSample, currentIndex);
    if (Number.isFinite(currentSampleTime)) return currentSampleTime;

    const rows = this.reconstructedRows();
    const lastIndex = rows.length - 1;
    if (lastIndex >= 0) {
      const lastTime = this.timeFor(rows[lastIndex], lastIndex);
      if (Number.isFinite(lastTime)) return lastTime;
    }
    return 0;
  }

  captureReportWindow() {
    this.reportWindowSeconds = this.getReportWindowSeconds();
    const end = this.currentReportEndTime();
    this.reportWindowEndS = Number.isFinite(end) ? end : null;
    this.applyReportWindowFromEnd();
  }

  applyReportWindowFromEnd() {
    this.reportWindowStartS = this.reportWindowEndS == null
      ? null
      : Math.max(0, this.reportWindowEndS - this.reportWindowSeconds);
    this.updateWindowSummary();
  }

  updateWindowSummary() {
    if (this.windowSummaryEl) this.windowSummaryEl.textContent = this.reportWindowLabel();
  }

  reportWindowLabel() {
    if (this.reportWindowStartS == null || this.reportWindowEndS == null) {
      return `Last ${this.reportWindowSeconds} seconds`;
    }
    return `Last ${this.reportWindowSeconds} seconds: ${formatReportWindowTime(this.reportWindowStartS)} to ${formatReportWindowTime(this.reportWindowEndS)}`;
  }

  withinReportWindow(point) {
    if (this.reportWindowStartS == null || this.reportWindowEndS == null) return true;
    const t = finiteNumber(point?.t);
    if (t == null) return true;
    return t >= this.reportWindowStartS - 1e-6 && t <= this.reportWindowEndS + 1e-6;
  }

  timeFor(row, index = 0) {
    const rel = finiteNumber(row?.t_rel_s);
    if (rel != null) return rel;

    const timestamp = firstFinite(row, ["t", "timestamp", "time", "ts"]);
    if (timestamp != null) {
      if (this.timeOffset != null && this.firstSampleT != null) {
        const span = Math.max(0, this.sampleTimestampSpan ?? this.maxSampleRel ?? 0);
        const tolerance = Math.max(10, span * 2, 60);
        if (timestamp >= this.firstSampleT - tolerance && timestamp <= this.firstSampleT + span + tolerance) {
          return timestamp - this.timeOffset;
        }
      }
      if (this.rawTimeOffset != null && Math.abs(timestamp) > 1000000) {
        return timestamp - this.rawTimeOffset;
      }
      return timestamp;
    }

    return index;
  }

  normalizeRows(rows, mapper) {
    if (!Array.isArray(rows)) return [];
    const normalized = [];
    rows.forEach((row, index) => {
      if (!row || typeof row !== "object") return;
      const mapped = mapper(row, index);
      if (mapped && Number.isFinite(mapped.x) && Number.isFinite(mapped.y) && this.withinReportWindow(mapped)) {
        normalized.push(mapped);
      }
    });
    return normalized;
  }

  rawRowsFor(dataType) {
    if (dataType === "Position") return this.state.rawPoints || this.payload.rawPositionPoints || [];

    const fields = Object.values(this.fieldsForDataType(dataType));
    const candidates = dataType === "Velocity"
      ? [
          this.payload.rawVelocityPoints,
          this.payload.rawVelocitySamples,
          this.payload.rawVelocityVectors,
          this.payload.velocityRawPoints,
          this.state.rawVelocityPoints,
          this.state.rawPoints,
          this.payload.rawPositionPoints,
        ]
      : [
          this.payload.rawAccelerationPoints,
          this.payload.rawAccelerationSamples,
          this.payload.rawAccelerationVectors,
          this.payload.accelerationRawPoints,
          this.state.rawAccelerationPoints,
          this.state.rawPoints,
          this.payload.rawPositionPoints,
        ];

    for (const rows of candidates) {
      if (!Array.isArray(rows) || !rows.length) continue;
      if (rows.some(row => fields.some(field => finiteNumber(row?.[field]) != null))) return rows;
    }
    return [];
  }

  reconstructedRows() {
    return this.state.samples || this.payload.samples || [];
  }

  positionProjectionSpec(view) {
    if (view === "XZ Side View") {
      return {
        title: "Position XZ Side View",
        xLabel: "X position (m)",
        yLabel: "Z position / altitude (m)",
        xValue: row => finiteNumber(row.x_m),
        yValue: row => zValue(row),
        sameUnitAxes: true,
      };
    }
    if (view === "YZ Side View") {
      return {
        title: "Position YZ Side View",
        xLabel: "Y position (m)",
        yLabel: "Z position / altitude (m)",
        xValue: row => finiteNumber(row.y_m),
        yValue: row => zValue(row),
        sameUnitAxes: true,
      };
    }
    if (view === "X over Time") {
      return {
        title: "Position X over Time",
        xLabel: "Time (s)",
        yLabel: "X position (m)",
        xValue: (row, index) => this.timeFor(row, index),
        yValue: row => finiteNumber(row.x_m),
      };
    }
    if (view === "Y over Time") {
      return {
        title: "Position Y over Time",
        xLabel: "Time (s)",
        yLabel: "Y position (m)",
        xValue: (row, index) => this.timeFor(row, index),
        yValue: row => finiteNumber(row.y_m),
      };
    }
    if (view === "Z over Time") {
      return {
        title: "Position Z over Time",
        xLabel: "Time (s)",
        yLabel: "Z position / altitude (m)",
        xValue: (row, index) => this.timeFor(row, index),
        yValue: row => zValue(row),
      };
    }
    return {
      title: "Position XY Top-Down Path",
      xLabel: "X position (m)",
      yLabel: "Y position (m)",
      xValue: row => finiteNumber(row.x_m),
      yValue: row => finiteNumber(row.y_m),
      sameUnitAxes: true,
    };
  }

  getPositionProjectionData(view) {
    const spec = this.positionProjectionSpec(view);
    const raw = this.normalizeRows(this.rawRowsFor("Position"), (row, index) => ({
      x: spec.xValue(row, index),
      y: spec.yValue(row, index),
      t: this.timeFor(row, index),
      row,
      index,
    }));
    const reconstructed = this.normalizeRows(this.reconstructedRows(), (row, index) => ({
      x: spec.xValue(row, index),
      y: spec.yValue(row, index),
      t: this.timeFor(row, index),
      row,
      index,
    }));
    return { spec, raw, reconstructed };
  }

  velocityFields() {
    return { X: "vel_east_mps", Y: "vel_north_mps", Z: "vel_up_mps" };
  }

  accelerationFields() {
    return { X: "accel_east_mps2", Y: "accel_north_mps2", Z: "accel_up_mps2" };
  }

  fieldsForDataType(dataType) {
    return dataType === "Acceleration" ? this.accelerationFields() : this.velocityFields();
  }

  unitForDataType(dataType) {
    return dataType === "Acceleration" ? "m/s²" : "m/s";
  }

  componentFromView(view) {
    if (view.startsWith("Y ")) return "Y";
    if (view.startsWith("Z ")) return "Z";
    return "X";
  }

  projectionAxesFromView(view) {
    if (view.startsWith("XZ")) return ["X", "Z"];
    if (view.startsWith("YZ")) return ["Y", "Z"];
    return ["X", "Y"];
  }

  getComponentTimeSeries(dataType, component) {
    const fields = this.fieldsForDataType(dataType);
    const field = fields[component];
    const unit = this.unitForDataType(dataType);
    const raw = this.normalizeRows(this.rawRowsFor(dataType), (row, index) => ({
      x: this.timeFor(row, index),
      y: finiteNumber(row[field]),
      t: this.timeFor(row, index),
      row,
      index,
    }));
    const reconstructed = this.normalizeRows(this.reconstructedRows(), (row, index) => ({
      x: this.timeFor(row, index),
      y: finiteNumber(row[field]),
      t: this.timeFor(row, index),
      row,
      index,
    }));
    return {
      spec: {
        title: `${dataType} ${component} Component over Time`,
        xLabel: "Time (s)",
        yLabel: `${component} ${dataType.toLowerCase()} (${unit})`,
      },
      raw,
      reconstructed,
      rawUnavailableNote: raw.length ? "" : `Raw ${dataType.toLowerCase()} data unavailable in this ${this.reportWindowSeconds}-second window.`,
    };
  }

  vectorData(dataType, view) {
    const [axisA, axisB] = this.projectionAxesFromView(view);
    const fields = this.fieldsForDataType(dataType);
    const unit = this.unitForDataType(dataType);
    const raw = this.normalizeRows(this.rawRowsFor(dataType), (row, index) => ({
      x: finiteNumber(row[fields[axisA]]),
      y: finiteNumber(row[fields[axisB]]),
      t: this.timeFor(row, index),
      row,
      index,
    }));
    const reconstructed = this.normalizeRows(this.reconstructedRows(), (row, index) => ({
      x: finiteNumber(row[fields[axisA]]),
      y: finiteNumber(row[fields[axisB]]),
      t: this.timeFor(row, index),
      row,
      index,
    }));
    return {
      spec: {
        title: `${dataType} ${axisA}${axisB} Vector Animation`,
        xLabel: `${axisA} ${dataType.toLowerCase()} (${unit})`,
        yLabel: `${axisB} ${dataType.toLowerCase()} (${unit})`,
        sameUnitAxes: true,
      },
      raw,
      reconstructed,
      rawUnavailableNote: raw.length ? "" : `Raw ${dataType.toLowerCase()} data unavailable in this ${this.reportWindowSeconds}-second window.`,
    };
  }

  renderReportChart() {
    if (!this.ctx || !this.canvas) return;
    this.resizeCanvas();
    const dataType = this.getReportDataType();
    const view = this.getReportView();

    if (dataType === "Position") {
      const data = this.getPositionProjectionData(view);
      const notes = [];
      const axisNote = this.axisScaleNote(data.spec);
      if (axisNote) notes.push(axisNote);
      if (data.raw.length && data.reconstructed.length) {
        notes.push("Raw ADS-B points are shown as markers; reconstructed/interpolated data is shown as a continuous line.");
      } else {
        if (!data.raw.length) notes.push(`Raw ADS-B position data unavailable in this ${this.reportWindowSeconds}-second window.`);
        if (!data.reconstructed.length) notes.push(`Reconstructed/interpolated position data unavailable in this ${this.reportWindowSeconds}-second window.`);
      }
      this.setNote(notes.join(" "));
      this.drawSeriesChart(data.spec, data.raw, data.reconstructed, { highlightIndex: null });
      return;
    }

    if (view.includes("Vector Animation")) {
      const data = this.vectorData(dataType, view);
      const axisNote = this.axisScaleNote(data.spec);
      this.setNote([data.rawUnavailableNote || "Click Animate to step through vectors over time.", axisNote].filter(Boolean).join(" "));
      const defaultIndex = Math.max(0, Math.min(this.state.currentIndex || 0, data.reconstructed.length - 1));
      this.drawVectorFrame(data.spec, data.raw, data.reconstructed, defaultIndex);
      return;
    }

    const component = this.componentFromView(view);
    const data = this.getComponentTimeSeries(dataType, component);
    this.setNote(data.rawUnavailableNote || "Raw data is shown as markers when available; reconstructed/method data is shown as a line.");
    this.drawSeriesChart(data.spec, data.raw, data.reconstructed, { highlightIndex: null });
  }

  renderReportAnimation() {
    if (!this.ctx || !this.canvas) return;
    this.stopAnimation(false);
    const dataType = this.getReportDataType();
    const view = this.getReportView();
    const frameCount = this.animationFrameCount(dataType, view);

    if (!frameCount) {
      this.stopAnimation();
      this.setNote("No data available to animate for this view.");
      this.renderReportChart();
      return;
    }

    this.animationRunning = true;
    this.animationStartMs = performance.now();
    this.animationDurationMs = Math.min(12000, Math.max(2800, frameCount * 55));
    this.animateButton.textContent = "Stop";

    const tick = now => {
      if (!this.animationRunning) return;
      const elapsed = now - this.animationStartMs;
      const progress = (elapsed % this.animationDurationMs) / this.animationDurationMs;
      const index = Math.min(frameCount - 1, Math.floor(progress * frameCount));
      this.renderAnimationFrame(index);
      this.animationFrame = requestAnimationFrame(tick);
    };
    this.animationFrame = requestAnimationFrame(tick);
  }

  animationFrameCount(dataType, view) {
    if (dataType === "Position") return this.getPositionProjectionData(view).reconstructed.length;
    if (view.includes("Vector Animation")) return this.vectorData(dataType, view).reconstructed.length;
    const component = this.componentFromView(view);
    return this.getComponentTimeSeries(dataType, component).reconstructed.length;
  }

  renderAnimationFrame(index) {
    const dataType = this.getReportDataType();
    const view = this.getReportView();

    if (dataType === "Position") {
      const data = this.getPositionProjectionData(view);
      const axisNote = this.axisScaleNote(data.spec);
      this.setNote(["Animating reconstructed position. The nearest raw ADS-B point is highlighted when available.", axisNote].filter(Boolean).join(" "));
      this.drawSeriesChart(data.spec, data.raw, data.reconstructed, { highlightIndex: index });
      return;
    }

    if (view.includes("Vector Animation")) {
      const data = this.vectorData(dataType, view);
      const axisNote = this.axisScaleNote(data.spec);
      this.setNote([data.rawUnavailableNote || "Animating raw and reconstructed vectors over time.", axisNote].filter(Boolean).join(" "));
      this.drawVectorFrame(data.spec, data.raw, data.reconstructed, index);
      return;
    }

    const component = this.componentFromView(view);
    const data = this.getComponentTimeSeries(dataType, component);
    this.setNote(data.rawUnavailableNote || "Animating component value over time.");
    this.drawSeriesChart(data.spec, data.raw, data.reconstructed, { highlightIndex: index });
  }

  stopAnimation(resetButton = true) {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    this.animationRunning = false;
    if (resetButton && this.animateButton) this.animateButton.textContent = "Animate";
  }

  setNote(text) {
    if (!this.noteEl) return;
    const message = text || "";
    this.noteEl.textContent = message ? `${this.reportWindowLabel()} — ${message}` : this.reportWindowLabel();
  }

  clearCanvas() {
    const ctx = this.ctx;
    const width = this.cssWidth || this.canvas.clientWidth || 1120;
    const height = this.cssHeight || this.canvas.clientHeight || 620;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = REPORT_COLORS.background;
    ctx.fillRect(0, 0, width, height);
    return { width, height };
  }

  plotArea(width, height) {
    return {
      left: Math.min(92, Math.max(64, width * 0.085)),
      right: Math.min(38, Math.max(22, width * 0.03)),
      top: 62,
      bottom: 70,
      width: 0,
      height: 0,
    };
  }

  paddedBounds(points, options = {}) {
    const xs = points.map(point => point.x).filter(Number.isFinite);
    const ys = points.map(point => point.y).filter(Number.isFinite);
    if (!xs.length || !ys.length) return null;

    let minX = Math.min(...xs);
    let maxX = Math.max(...xs);
    let minY = Math.min(...ys);
    let maxY = Math.max(...ys);

    if (Math.abs(maxX - minX) < 1e-9) {
      minX -= 1;
      maxX += 1;
    }
    if (Math.abs(maxY - minY) < 1e-9) {
      minY -= 1;
      maxY += 1;
    }

    const padX = (maxX - minX) * 0.08;
    const padY = (maxY - minY) * 0.12;
    const bounds = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
    return this.adjustAxisBounds(bounds, options);
  }

  axisPlotDimensions() {
    const width = this.cssWidth || this.canvas?.clientWidth || 1120;
    const height = this.cssHeight || this.canvas?.clientHeight || 620;
    const area = this.plotArea(width, height);
    return {
      width: Math.max(1, width - area.left - area.right),
      height: Math.max(1, height - area.top - area.bottom),
    };
  }

  expandRange(bounds, axis, newRange) {
    const minKey = axis === "x" ? "minX" : "minY";
    const maxKey = axis === "x" ? "maxX" : "maxY";
    const currentRange = bounds[maxKey] - bounds[minKey];
    if (!Number.isFinite(newRange) || newRange <= currentRange) return bounds;
    const center = (bounds[minKey] + bounds[maxKey]) / 2;
    bounds[minKey] = center - newRange / 2;
    bounds[maxKey] = center + newRange / 2;
    return bounds;
  }

  adjustAxisBounds(inputBounds, options = {}) {
    const bounds = { ...inputBounds };
    if (!options.sameUnitAxes) return bounds;

    const dims = this.axisPlotDimensions();
    let xRange = bounds.maxX - bounds.minX;
    let yRange = bounds.maxY - bounds.minY;
    if (!Number.isFinite(xRange) || !Number.isFinite(yRange) || xRange <= 0 || yRange <= 0) return bounds;

    if (options.trueScale) {
      const pixelsPerUnit = Math.min(dims.width / xRange, dims.height / yRange);
      if (pixelsPerUnit > 0 && Number.isFinite(pixelsPerUnit)) {
        this.expandRange(bounds, "x", dims.width / pixelsPerUnit);
        this.expandRange(bounds, "y", dims.height / pixelsPerUnit);
      }
      return bounds;
    }

    const maxScaleRatio = Number(options.maxScaleRatio || AUTO_AXIS_MAX_SCALE_RATIO);
    if (!Number.isFinite(maxScaleRatio) || maxScaleRatio <= 1) return bounds;

    const xScale = dims.width / xRange;
    const yScale = dims.height / yRange;
    if (xScale > yScale * maxScaleRatio) {
      this.expandRange(bounds, "x", dims.width / (yScale * maxScaleRatio));
    } else if (yScale > xScale * maxScaleRatio) {
      this.expandRange(bounds, "y", dims.height / (xScale * maxScaleRatio));
    }
    return bounds;
  }

  axisScaleOptions(spec = {}) {
    if (!spec.sameUnitAxes) return {};
    return {
      sameUnitAxes: true,
      trueScale: this.useTrueAxisMagnitudes(),
      maxScaleRatio: AUTO_AXIS_MAX_SCALE_RATIO,
    };
  }

  axisScaleNote(spec = {}) {
    if (!spec.sameUnitAxes) return "";
    if (this.useTrueAxisMagnitudes()) {
      return "True axis magnitudes enabled: equal units use equal visual scale.";
    }
    return `Axis scale is auto-limited to avoid more than ${AUTO_AXIS_MAX_SCALE_RATIO}:1 visual exaggeration; enable true axis magnitudes for 1:1 scale.`;
  }

  formatTick(value, step = null) {
    if (!Number.isFinite(value)) return "";
    const absStep = Math.abs(Number(step));
    if (Number.isFinite(absStep) && absStep > 0) {
      let decimals;
      if (absStep >= 100) decimals = 0;
      else if (absStep >= 10) decimals = 1;
      else decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(absStep)) + 1));
      return value.toFixed(decimals).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
    }

    const abs = Math.abs(value);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    return value.toFixed(2).replace(/\.00$/, "");
  }

  drawBaseChart(spec, bounds, options = {}) {
    const ctx = this.ctx;
    const { width, height } = this.clearCanvas();
    const area = this.plotArea(width, height);
    area.width = width - area.left - area.right;
    area.height = height - area.top - area.bottom;

    ctx.save();
    ctx.fillStyle = REPORT_COLORS.text;
    ctx.font = "700 18px Inter, Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(spec.title || "Report", area.left, 20);

    if (options.subtitle) {
      ctx.fillStyle = REPORT_COLORS.mutedText;
      ctx.font = "12px Inter, Arial, sans-serif";
      ctx.fillText(options.subtitle, area.left, 43);
    }

    ctx.strokeStyle = REPORT_COLORS.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(area.left, area.top, area.width, area.height);

    ctx.strokeStyle = REPORT_COLORS.grid;
    ctx.font = "12px Inter, Arial, sans-serif";
    for (let i = 0; i <= 5; i++) {
      const x = area.left + (area.width / 5) * i;
      const value = bounds.minX + ((bounds.maxX - bounds.minX) / 5) * i;
      ctx.beginPath();
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, area.top + area.height);
      ctx.stroke();
      ctx.fillStyle = REPORT_COLORS.mutedText;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(this.formatTick(value, (bounds.maxX - bounds.minX) / 5), x, area.top + area.height + 8);
    }

    for (let i = 0; i <= 5; i++) {
      const y = area.top + (area.height / 5) * i;
      const value = bounds.maxY - ((bounds.maxY - bounds.minY) / 5) * i;
      ctx.beginPath();
      ctx.moveTo(area.left, y);
      ctx.lineTo(area.left + area.width, y);
      ctx.stroke();
      ctx.fillStyle = REPORT_COLORS.mutedText;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(this.formatTick(value, (bounds.maxY - bounds.minY) / 5), area.left - 10, y);
    }

    ctx.fillStyle = REPORT_COLORS.text;
    ctx.font = "13px Inter, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(spec.xLabel || "X", area.left + area.width / 2, height - 18);

    ctx.save();
    ctx.translate(18, area.top + area.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(spec.yLabel || "Y", 0, 0);
    ctx.restore();
    ctx.restore();

    const xFor = value => area.left + ((value - bounds.minX) / (bounds.maxX - bounds.minX)) * area.width;
    const yFor = value => area.top + ((bounds.maxY - value) / (bounds.maxY - bounds.minY)) * area.height;
    return { width, height, area, xFor, yFor };
  }

  drawSeriesChart(spec, raw, reconstructed, options = {}) {
    const all = [...raw, ...reconstructed];
    const bounds = this.paddedBounds(all, this.axisScaleOptions(spec));
    if (!bounds) {
      this.drawEmpty(spec?.title || "Report", "No usable data for this view.");
      return;
    }

    const plot = this.drawBaseChart(spec, bounds, { subtitle: this.reportWindowLabel() });
    const ctx = this.ctx;

    if (reconstructed.length >= 2) {
      ctx.save();
      ctx.strokeStyle = REPORT_COLORS.reconstructed;
      ctx.lineWidth = 2.5;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      reconstructed.forEach((point, index) => {
        const x = plot.xFor(point.x);
        const y = plot.yFor(point.y);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    } else if (reconstructed.length === 1) {
      ctx.save();
      ctx.fillStyle = REPORT_COLORS.reconstructed;
      markerPath(ctx, plot.xFor(reconstructed[0].x), plot.yFor(reconstructed[0].y), 4);
      ctx.fill();
      ctx.restore();
    }

    if (raw.length) {
      ctx.save();
      ctx.fillStyle = REPORT_COLORS.raw;
      ctx.globalAlpha = raw.length > 250 ? 0.75 : 0.92;
      const radius = raw.length > 500 ? 2.2 : 3.2;
      for (const point of raw) {
        markerPath(ctx, plot.xFor(point.x), plot.yFor(point.y), radius);
        ctx.fill();
      }
      ctx.restore();
    }

    let highlighted = null;
    if (Number.isInteger(options.highlightIndex) && reconstructed.length) {
      const idx = Math.max(0, Math.min(reconstructed.length - 1, options.highlightIndex));
      highlighted = reconstructed[idx];
      ctx.save();
      ctx.fillStyle = REPORT_COLORS.current;
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.lineWidth = 2;
      markerPath(ctx, plot.xFor(highlighted.x), plot.yFor(highlighted.y), 6);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      const nearestRaw = this.nearestByTime(raw, highlighted.t);
      if (nearestRaw) {
        ctx.save();
        ctx.fillStyle = REPORT_COLORS.rawCurrent;
        ctx.strokeStyle = "rgba(255,255,255,0.82)";
        ctx.lineWidth = 2;
        markerPath(ctx, plot.xFor(nearestRaw.x), plot.yFor(nearestRaw.y), 5.5);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }

    this.drawLegend(plot.area, [
      raw.length ? { kind: "marker", color: REPORT_COLORS.raw, label: "Raw" } : null,
      reconstructed.length ? { kind: "line", color: REPORT_COLORS.reconstructed, label: this.methodLabel() } : null,
      highlighted ? { kind: "marker", color: REPORT_COLORS.current, label: "Current" } : null,
    ].filter(Boolean));
  }

  drawVectorFrame(spec, raw, reconstructed, index) {
    const selected = reconstructed.length ? reconstructed[Math.max(0, Math.min(reconstructed.length - 1, index))] : null;
    const nearestRaw = selected ? this.nearestByTime(raw, selected.t) : null;
    const candidates = [...reconstructed, ...raw];
    const bounds = this.paddedBounds(
      candidates.length ? [...candidates, { x: 0, y: 0 }] : [{ x: -1, y: -1 }, { x: 1, y: 1 }],
      this.axisScaleOptions(spec),
    );
    const vectorSubtitle = selected
      ? `${this.reportWindowLabel()} · t = ${this.formatTick(selected.t)} s`
      : `${this.reportWindowLabel()} · No reconstructed vector data`;
    const plot = this.drawBaseChart(spec, bounds, { subtitle: vectorSubtitle });
    const ctx = this.ctx;

    const zeroX = plot.xFor(0);
    const zeroY = plot.yFor(0);
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(plot.area.left, zeroY);
    ctx.lineTo(plot.area.left + plot.area.width, zeroY);
    ctx.moveTo(zeroX, plot.area.top);
    ctx.lineTo(zeroX, plot.area.top + plot.area.height);
    ctx.stroke();
    ctx.restore();

    if (nearestRaw) this.drawArrow(zeroX, zeroY, plot.xFor(nearestRaw.x), plot.yFor(nearestRaw.y), REPORT_COLORS.raw);
    if (selected) this.drawArrow(zeroX, zeroY, plot.xFor(selected.x), plot.yFor(selected.y), REPORT_COLORS.reconstructed);

    this.drawLegend(plot.area, [
      nearestRaw ? { kind: "arrow", color: REPORT_COLORS.raw, label: "Raw" } : null,
      selected ? { kind: "arrow", color: REPORT_COLORS.reconstructed, label: this.methodLabel() } : null,
    ].filter(Boolean));
  }

  drawArrow(x0, y0, x1, y1, color) {
    const ctx = this.ctx;
    const angle = Math.atan2(y1 - y0, x1 - x0);
    const head = 12;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(angle - Math.PI / 6), y1 - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x1 - head * Math.cos(angle + Math.PI / 6), y1 - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  drawLegend(area, items) {
    if (!items.length) return;
    const ctx = this.ctx;
    const rowH = 20;
    const padding = 10;
    const width = Math.min(260, Math.max(150, ...items.map(item => 80 + String(item.label).length * 7)));
    const height = padding * 2 + rowH * items.length;
    const x = area.left + area.width - width - 12;
    const y = area.top + 12;

    ctx.save();
    ctx.fillStyle = "rgba(8, 27, 51, 0.82)";
    ctx.strokeStyle = "rgba(196, 214, 232, 0.18)";
    ctx.lineWidth = 1;
    roundedRect(ctx, x, y, width, height, 9);
    ctx.fill();
    ctx.stroke();

    ctx.font = "12px Inter, Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    items.forEach((item, index) => {
      const rowY = y + padding + rowH * index + rowH / 2;
      ctx.strokeStyle = item.color;
      ctx.fillStyle = item.color;
      ctx.lineWidth = 2.5;
      if (item.kind === "line") {
        ctx.beginPath();
        ctx.moveTo(x + 13, rowY);
        ctx.lineTo(x + 33, rowY);
        ctx.stroke();
      } else if (item.kind === "arrow") {
        ctx.beginPath();
        ctx.moveTo(x + 12, rowY);
        ctx.lineTo(x + 33, rowY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x + 33, rowY);
        ctx.lineTo(x + 28, rowY - 4);
        ctx.lineTo(x + 28, rowY + 4);
        ctx.closePath();
        ctx.fill();
      } else {
        markerPath(ctx, x + 23, rowY, 4);
        ctx.fill();
      }
      ctx.fillStyle = REPORT_COLORS.text;
      ctx.fillText(item.label, x + 44, rowY);
    });
    ctx.restore();
  }

  nearestByTime(rows, t) {
    if (!rows.length || !Number.isFinite(t)) return null;
    let best = null;
    let bestDt = Infinity;
    for (const row of rows) {
      const dt = Math.abs(row.t - t);
      if (dt < bestDt) {
        best = row;
        bestDt = dt;
      }
    }
    return best;
  }

  drawEmpty(title, message) {
    const ctx = this.ctx;
    const { width, height } = this.clearCanvas();
    ctx.save();
    ctx.fillStyle = REPORT_COLORS.text;
    ctx.font = "700 18px Inter, Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(title || "Report", 42, 32);

    ctx.fillStyle = REPORT_COLORS.mutedText;
    ctx.font = "14px Inter, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message || "No data available.", width / 2, height / 2);
    ctx.restore();
  }
}
