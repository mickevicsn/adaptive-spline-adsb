export class AnimationController {
  constructor(state, scene, camera, hud = null) {
    this.state = state;
    this.scene = scene;
    this.camera = camera;
    this.hud = hud;
    this._tick = this._tick.bind(this);
  }

  start() {
    this.running = true;
    this.renderOnce();
    requestAnimationFrame(this._tick);
  }

  renderOnce() {
    const sample = this.state.currentSample();
    const passedRawPointIndex = this.scene.update(sample, this.state.currentIndex);
    this.camera.update(sample);
    this.scene.renderStatus(sample, passedRawPointIndex, this.camera);
    if (this.hud) {
      this.hud.update(sample);
    }
  }

  _tick(nowMs) {
    if (!this.running) {
      return;
    }

    if (this.state.lastFrameMs === null) {
      this.state.lastFrameMs = nowMs;
    }

    const dt = (nowMs - this.state.lastFrameMs) / 1000;
    this.state.lastFrameMs = nowMs;

    if (this.state.playing) {
      this.state.currentRelT += dt * this.state.speed;

      const maxT = this.state.maxRelativeTime();
      if (this.state.currentRelT >= maxT) {
        this.state.currentRelT = maxT;
        this.state.playing = false;
      }

      this.state.currentIndex = this.state.indexForTime(this.state.currentRelT);
    }

    this.renderOnce();
    requestAnimationFrame(this._tick);
  }
}
