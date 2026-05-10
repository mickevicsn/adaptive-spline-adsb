import { cartesianFromSample, heightFor } from "./geometry.js";

export class CameraController {
  constructor(viewer, state) {
    this.viewer = viewer;
    this.state = state;
  }

  installKeyboardHandlers() {
    window.addEventListener("keydown", event => {
      const handledKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];

      if (!handledKeys.includes(event.key)) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();

      const cfg = this.state.config;

      if (event.key === "ArrowLeft") {
        this.state.orbitHeadingDeg = this.state.normalizeDeg(
          this.state.orbitHeadingDeg - cfg.cameraHeadingStepDeg,
        );
      } else if (event.key === "ArrowRight") {
        this.state.orbitHeadingDeg = this.state.normalizeDeg(
          this.state.orbitHeadingDeg + cfg.cameraHeadingStepDeg,
        );
      } else if (event.key === "ArrowUp") {
        this.state.orbitPitchDeg = this.state.clamp(
          this.state.orbitPitchDeg - cfg.cameraPitchStepDeg,
          cfg.cameraMinPitchDeg,
          cfg.cameraMaxPitchDeg,
        );
      } else if (event.key === "ArrowDown") {
        this.state.orbitPitchDeg = this.state.clamp(
          this.state.orbitPitchDeg + cfg.cameraPitchStepDeg,
          cfg.cameraMinPitchDeg,
          cfg.cameraMaxPitchDeg,
        );
      }
    }, true);
  }

  installWheelHandler(element) {
    element.addEventListener("wheel", event => {
      event.preventDefault();
      event.stopPropagation();

      const cfg = this.state.config;
      const zoomFactor = cfg.cameraWheelZoomFactor || 1.12;

      if (event.deltaY > 0) {
        this.state.userCameraRangeM *= zoomFactor;
      } else if (event.deltaY < 0) {
        this.state.userCameraRangeM /= zoomFactor;
      }

      this.state.userCameraRangeM = this.state.clamp(
        this.state.userCameraRangeM,
        cfg.cameraMinRangeM || 80,
        cfg.cameraMaxRangeM || 50000,
      );
    }, { passive: false });
  }

  cameraRangeFor(sample) {
    const cfg = this.state.config;
    const z = Math.max(0, heightFor(sample));
    const range = this.state.userCameraRangeM + z * cfg.cameraAltitudeRangeFactor;

    return this.state.clamp(
      range,
      cfg.cameraMinRangeM || 80,
      cfg.cameraMaxRangeM || 50000,
    );
  }

  update(sample) {
    const target = cartesianFromSample(sample);

    const headingRad = Cesium.Math.toRadians(this.state.orbitHeadingDeg);
    const pitchRad = Cesium.Math.toRadians(this.state.orbitPitchDeg);
    const range = this.cameraRangeFor(sample);

    this.viewer.camera.lookAt(
      target,
      new Cesium.HeadingPitchRange(headingRad, pitchRad, range),
    );

    const cfg = this.state.config;
    const rightRatio = Number(cfg.cameraTargetScreenOffsetRightRatio || 0);
    const downRatio = Number(cfg.cameraTargetScreenOffsetDownRatio || 0);

    if (rightRatio !== 0) {
      this.viewer.camera.moveLeft(range * rightRatio);
    }

    if (downRatio !== 0) {
      this.viewer.camera.moveUp(range * downRatio);
    }
  }
}
