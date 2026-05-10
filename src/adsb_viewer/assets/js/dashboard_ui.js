import { heightFor } from "./geometry.js";
import { ReportModal } from "./report_modal.js";

const KNOTS_PER_MPS = 1.9438444924406046;

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const secs = String(seconds % 60).padStart(2, "0");
  return `${hours}:${minutes}:${secs}`;
}

function formatLatLon(lat, lon) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(5)}° ${ns}, ${Math.abs(lon).toFixed(5)}° ${ew}`;
}

function speedKnots(sample) {
  if (sample.ground_speed_kt != null) {
    return sample.ground_speed_kt;
  }
  if (sample.vel_east_mps != null && sample.vel_north_mps != null) {
    return Math.hypot(sample.vel_east_mps, sample.vel_north_mps) * KNOTS_PER_MPS;
  }
  return null;
}

function sampleTimeLabel(sample) {
  const t = Number(sample.t || 0);
  if (Number.isFinite(t) && t > 1000000000) {
    return new Date(t * 1000).toISOString().slice(11, 19) + " UTC";
  }
  return `T+ ${formatDuration(sample.t_rel_s || 0)}`;
}

export class DashboardUI {
  constructor(state, options = {}) {
    this.state = state;
    this.onReset = options.onReset || null;
    this.chartMode = "velocity";
    this.chartWindowS = 90;
    this.reportModal = new ReportModal(state);

    this.pauseButton = document.getElementById("pauseButton");
    this.generateReportButton = document.getElementById("generateReportButton");
    this.resetButton = document.getElementById("resetButton");
    this.speedDownButton = document.getElementById("speedDownButton");
    this.speedUpButton = document.getElementById("speedUpButton");
    this.speedDisplay = document.getElementById("speedDisplay");
    this.settingsButton = document.getElementById("settingsButton");
    this.settingsPanel = document.getElementById("settingsPanel");
    this.chartModeSelect = document.getElementById("chartMode");
    this.chartNote = document.getElementById("chartNote");
    this.canvas = document.getElementById("chartCanvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.headingCompassRose = document.getElementById("headingCompassRose");
    this.headingCompassValue = document.getElementById("headingCompassValue");
    this.enabled = Boolean(
      this.pauseButton &&
      this.resetButton &&
      this.speedDownButton &&
      this.speedUpButton &&
      this.speedDisplay &&
      this.settingsButton &&
      this.settingsPanel &&
      this.chartModeSelect &&
      this.canvas &&
      this.ctx,
    );
  }

  mount() {
    if (!this.enabled) {
      console.warn("DashboardUI disabled: new dashboard DOM was not found. Replace templates/viewer.html and hard-refresh the browser.");
      return;
    }
    this.pauseButton.addEventListener("click", () => {
      this.state.playing = !this.state.playing;
      this.syncButtons();
    });

    this.reportModal.mount();
    if (this.generateReportButton) {
      this.generateReportButton.addEventListener("click", () => {
        this.reportModal.open();
      });
    }

    this.resetButton.addEventListener("click", () => {
      if (this.onReset) {
        this.state.playing = false;
        this.onReset();
      } else {
        this.state.resetPlayback();
      }
      this.syncButtons();
    });

    this.speedDownButton.addEventListener("click", () => {
      this.state.adjustSpeed(-1);
      this.syncButtons();
    });

    this.speedUpButton.addEventListener("click", () => {
      this.state.adjustSpeed(1);
      this.syncButtons();
    });

    window.addEventListener("keydown", event => {
      const target = event.target;
      const tag = target?.tagName?.toLowerCase();
      if (["input", "select", "textarea"].includes(tag)) {
        return;
      }

      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        this.state.adjustSpeed(1);
        this.syncButtons();
      } else if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        this.state.adjustSpeed(-1);
        this.syncButtons();
      }
    });

    this.settingsButton.addEventListener("click", event => {
      event.stopPropagation();
      this.settingsPanel.classList.toggle("hidden");
    });

    document.addEventListener("click", event => {
      if (
        !this.settingsPanel.contains(event.target) &&
        !this.settingsButton.contains(event.target)
      ) {
        this.settingsPanel.classList.add("hidden");
      }
    });

    this.chartModeSelect.addEventListener("change", event => {
      this.chartMode = event.target.value;
    });

    this.syncButtons();
    this.setStaticValues();
  }

  syncButtons() {
    if (!this.enabled) {
      return;
    }
    this.pauseButton.textContent = this.state.playing ? "Pause" : "Play";
    this.speedDisplay.textContent = `⚡ ${this.state.speedLabel()}`;
    this.speedDownButton.disabled = this.state.speed <= this.state.minSimulationSpeed + 1e-9;
    this.speedUpButton.disabled = this.state.speed >= this.state.maxSimulationSpeed - 1e-9;
  }

  setStaticValues() {
    const flightEl = document.getElementById("metricFlight");
    if (flightEl) {
      const selectedFlight = this.state.payload.selectedFlight || {};
      const selectedMethod = this.state.payload.selectedMethod || {};
      const flightLabel = selectedFlight.label || selectedFlight.icao || this.state.payload.track_id || "track";
      const methodLabel = selectedMethod.label || selectedMethod.methodId || this.state.payload.selectedMethodId || "";
      flightEl.textContent = methodLabel ? `${flightLabel} · ${methodLabel}` : flightLabel;
    }
  }

  update(sample) {
    if (!this.enabled) {
      return;
    }
    document.getElementById("metricElapsed").textContent = formatDuration(sample.t_rel_s || 0);
    document.getElementById("metricAltitude").textContent = `${heightFor(sample).toFixed(0)} m`;

    const gs = speedKnots(sample);
    document.getElementById("metricSpeed").textContent = gs == null ? "—" : `${gs.toFixed(0)} kt`;
    document.getElementById("metricHeading").textContent = `${Number(sample.heading_deg || 0).toFixed(0).padStart(3, "0")}°`;
    this.updateHeadingCompass(sample);
    document.getElementById("metricPosition").textContent = formatLatLon(sample.lat, sample.lon);
    document.getElementById("metricPositionSub").textContent = sampleTimeLabel(sample);

    this.drawChart(sample);
    this.syncButtons();
  }

  updateHeadingCompass(sample) {
    const heading = ((Number(sample.heading_deg || 0) % 360) + 360) % 360;

    if (this.headingCompassRose) {
      this.headingCompassRose.style.transform = `rotate(${-heading.toFixed(1)}deg)`;
    }

    if (this.headingCompassValue) {
      this.headingCompassValue.textContent = `${heading.toFixed(0).padStart(3, "0")}°`;
    }
  }

  drawChart(currentSample) {
    if (!this.enabled || !this.ctx) {
      return;
    }
    const ctx = this.ctx;
    const canvas = this.canvas;
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(4, 16, 34, 1)";
    ctx.fillRect(0, 0, width, height);

    const padding = { left: 46, right: 14, top: 16, bottom: 28 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const endT = currentSample.t_rel_s || 0;
    const startT = Math.max(0, endT - this.chartWindowS);
    const samples = this.state.samples.filter(s => s.t_rel_s >= startT && s.t_rel_s <= endT);
    if (!samples.length) {
      return;
    }

    let series;
    if (this.chartMode === "acceleration") {
      this.chartNote.textContent = "Ax / Ay / |A|";
      series = [
        { key: "accel_east_mps2", label: "Ax", color: "#1ec7ff" },
        { key: "accel_north_mps2", label: "Ay", color: "#7cf03a" },
        { key: "accel_horizontal_mps2", label: "|A|", color: "#b665ff" },
      ];
    } else {
      this.chartNote.textContent = "Vx / Vy / |V|";
      series = [
        { key: "vel_east_mps", label: "Vx", color: "#1ec7ff" },
        { key: "vel_north_mps", label: "Vy", color: "#7cf03a" },
        {
          key: null,
          label: "|V|",
          color: "#b665ff",
          value: sample => {
            if (sample.vel_east_mps == null || sample.vel_north_mps == null) {
              return null;
            }
            return Math.hypot(sample.vel_east_mps, sample.vel_north_mps);
          },
        },
      ];
    }

    const values = [];
    for (const s of samples) {
      for (const spec of series) {
        const value = spec.value ? spec.value(s) : s[spec.key];
        if (Number.isFinite(value)) {
          values.push(value);
        }
      }
    }
    if (!values.length) {
      return;
    }

    let minV = Math.min(...values);
    let maxV = Math.max(...values);
    if (Math.abs(maxV - minV) < 1e-6) {
      minV -= 1;
      maxV += 1;
    }
    const padV = (maxV - minV) * 0.12;
    minV -= padV;
    maxV += padV;

    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (plotHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }
    for (let i = 0; i <= 6; i++) {
      const x = padding.left + (plotWidth / 6) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(225, 240, 255, 0.75)";
    ctx.font = "12px Arial";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const v = maxV - ((maxV - minV) / 4) * i;
      const y = padding.top + (plotHeight / 4) * i + 4;
      ctx.fillText(v.toFixed(0), padding.left - 6, y);
    }

    ctx.textAlign = "center";
    const tickValues = [startT, startT + this.chartWindowS / 3, startT + (this.chartWindowS * 2) / 3, endT];
    const tickLabels = tickValues.map((t, idx) => (idx === tickValues.length - 1 ? "Now" : `${Math.round(t - endT)} s`));
    tickValues.forEach((t, idx) => {
      const x = padding.left + ((t - startT) / Math.max(1e-9, endT - startT || 1)) * plotWidth;
      ctx.fillText(tickLabels[idx], x, height - 8);
    });

    const xFor = t => padding.left + ((t - startT) / Math.max(1e-9, endT - startT || 1)) * plotWidth;
    const yFor = v => padding.top + ((maxV - v) / (maxV - minV)) * plotHeight;

    series.forEach((spec, index) => {
      ctx.strokeStyle = spec.color;
      ctx.fillStyle = spec.color;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      let started = false;
      for (const s of samples) {
        const value = spec.value ? spec.value(s) : s[spec.key];
        if (!Number.isFinite(value)) {
          continue;
        }
        const x = xFor(s.t_rel_s);
        const y = yFor(value);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      const last = samples[samples.length - 1];
      const lastValue = spec.value ? spec.value(last) : last[spec.key];
      if (Number.isFinite(lastValue)) {
        const lx = xFor(last.t_rel_s);
        const ly = yFor(lastValue);
        ctx.beginPath();
        ctx.arc(lx, ly, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(spec.label, width - 28, padding.top + 18 + index * 18);
      }
    });
  }
}
