export class ControlPanel {
  constructor(state, scene) {
    this.state = state;
    this.scene = scene;
    this.container = document.getElementById("controls");
  }

  mount() {
    this.container.innerHTML = "";

    this.addRangeControl({
      label: "Trail sec",
      value: this.state.config.trailSeconds,
      min: 5,
      max: 600,
      step: 5,
      format: value => `${Number(value).toFixed(0)} s`,
      onInput: value => {
        this.state.config.trailSeconds = Number(value);
      },
    });

    this.addRangeControl({
      label: "Dot size",
      value: this.state.config.rawPointPixelSize,
      min: 2,
      max: 24,
      step: 1,
      format: value => `${Number(value).toFixed(0)} px`,
      onInput: value => {
        this.state.config.rawPointPixelSize = Number(value);
        this.scene.updateRawPointStyle();
      },
    });

    this.addRangeControl({
      label: "Line width",
      value: this.state.config.connectionLineWidthPx,
      min: 1,
      max: 10,
      step: 1,
      format: value => `${Number(value).toFixed(0)} px`,
      onInput: value => {
        this.state.config.connectionLineWidthPx = Number(value);
        this.scene.updateRawPointStyle();
      },
    });

    this.addRangeControl({
      label: "Endpoint gap",
      value: this.state.config.connectionLineEndpointGapM,
      min: 0,
      max: 40,
      step: 1,
      format: value => `${Number(value).toFixed(0)} m`,
      onInput: value => {
        this.state.config.connectionLineEndpointGapM = Number(value);
        this.scene.rebuildConnectionLines();
      },
    });

    this.addRangeControl({
      label: "Camera",
      value: this.state.userCameraRangeM,
      min: this.state.config.cameraMinRangeM || 80,
      max: 5000,
      step: 10,
      format: value => `${Number(value).toFixed(0)} m`,
      onInput: value => {
        this.state.userCameraRangeM = Number(value);
      },
    });

    this.addCheckboxControl({
      label: "Display segments",
      checked: this.state.displaySegments,
      onChange: checked => {
        this.state.displaySegments = checked;
        this.state.config.displaySegments = checked;
        this.scene.updateRawPointStyle();
        this.scene.updateSegmentLegend();
      },
    });

    this.addCheckboxControl({
      label: "Show trail",
      checked: this.state.showTrail,
      onChange: checked => {
        this.state.showTrail = checked;
      },
    });

    this.addCheckboxControl({
      label: "Show velocity",
      checked: this.state.showVelocityVector,
      onChange: checked => {
        this.state.showVelocityVector = checked;
      },
    });

    this.addCheckboxControl({
      label: "Show accel",
      checked: this.state.showAccelerationVector,
      onChange: checked => {
        this.state.showAccelerationVector = checked;
      },
    });
  }

  addRangeControl({ label, value, min, max, step, format, onInput }) {
    const row = document.createElement("div");
    row.className = "row";

    const labelEl = document.createElement("label");
    labelEl.className = "control-label";
    labelEl.textContent = label;

    const input = document.createElement("input");
    input.type = "range";
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;

    const valueEl = document.createElement("span");
    valueEl.className = "value-label";
    valueEl.textContent = format(value);

    input.addEventListener("input", event => {
      const newValue = Number(event.target.value);
      valueEl.textContent = format(newValue);
      onInput(newValue);
    });

    row.appendChild(labelEl);
    row.appendChild(input);
    row.appendChild(valueEl);
    this.container.appendChild(row);
  }

  addCheckboxControl({ label, checked, onChange }) {
    const row = document.createElement("div");
    row.className = "row";

    const labelEl = document.createElement("label");
    labelEl.className = "control-label";
    labelEl.textContent = label;

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;

    input.addEventListener("change", event => {
      onChange(event.target.checked);
    });

    row.appendChild(labelEl);
    row.appendChild(input);
    this.container.appendChild(row);
  }
}
