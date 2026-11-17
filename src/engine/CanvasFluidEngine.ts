import {
  FluidEngine,
  FluidMode,
  FluidParams,
  KarmanPreset,
  MagnetPole,
  TelemetrySnapshot,
} from './types';

const GRID_W = 128;
const GRID_H = 72;
const N = GRID_W * GRID_H;
const PARTICLE_COUNT = 6400;

export class CanvasFluidEngine implements FluidEngine {
  readonly backendLabel = 'Canvas 2D TypedArray Fallback';
  readonly particleCount = PARTICLE_COUNT;

  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private offscreen: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;
  private imgData: ImageData;

  private u = new Float32Array(N);
  private v = new Float32Array(N);
  private uPrev = new Float32Array(N);
  private vPrev = new Float32Array(N);
  private p = new Float32Array(N);
  private div = new Float32Array(N);
  private curl = new Float32Array(N);

  private dyeR = new Float32Array(N);
  private dyeG = new Float32Array(N);
  private dyeB = new Float32Array(N);
  private dyeR0 = new Float32Array(N);
  private dyeG0 = new Float32Array(N);
  private dyeB0 = new Float32Array(N);

  private obstacles = new Uint8Array(N);

  // 6,400 Lagrangian particles: x, y, energy, life
  private px = new Float32Array(PARTICLE_COUNT);
  private py = new Float32Array(PARTICLE_COUNT);
  private pEnergy = new Float32Array(PARTICLE_COUNT);
  private pLife = new Float32Array(PARTICLE_COUNT);

  private time = 0;
  private fps = 60;
  private lastFpsTime = performance.now();
  private fpsFrames = 0;
  private telemetry: TelemetrySnapshot = {
    fps: 60,
    reynoldsNumber: 1200,
    peakVorticity: 0,
    kineticEnergy: 0,
    turbulenceRatio: 0.1,
    particleCount: PARTICLE_COUNT,
    backendLabel: this.backendLabel,
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('2D Canvas context unavailable');
    }
    this.ctx = ctx;
    this.offscreen = document.createElement('canvas');
    this.offscreen.width = GRID_W;
    this.offscreen.height = GRID_H;
    this.offCtx = this.offscreen.getContext('2d')!;
    this.imgData = this.offCtx.createImageData(GRID_W, GRID_H);
    this.seedParticles();
  }

  private idx(x: number, y: number): number {
    const cx = x < 0 ? 0 : x >= GRID_W ? GRID_W - 1 : x;
    const cy = y < 0 ? 0 : y >= GRID_H ? GRID_H - 1 : y;
    return cy * GRID_W + cx;
  }

  private seedParticles(): void {
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      this.px[i] = Math.random();
      this.py[i] = Math.random();
      this.pEnergy[i] = 0.1;
      this.pLife[i] = 0.2 + Math.random() * 0.8;
    }
  }

  resize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return;
    this.canvas.width = width;
    this.canvas.height = height;
  }

  splat(
    nx: number,
    ny: number,
    dx: number,
    dy: number,
    color: [number, number, number],
    radius: number
  ): void {
    const cx = nx * GRID_W;
    const cy = (1 - ny) * GRID_H;
    const rCells = Math.max(2, Math.sqrt(radius) * GRID_W * 1.4);
    const r2 = rCells * rCells;
    const x0 = Math.max(1, Math.floor(cx - rCells));
    const x1 = Math.min(GRID_W - 2, Math.ceil(cx + rCells));
    const y0 = Math.max(1, Math.floor(cy - rCells));
    const y1 = Math.min(GRID_H - 2, Math.ceil(cy + rCells));

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * GRID_W + x;
        if (this.obstacles[i]) continue;
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (d2 > r2) continue;
        const w = Math.exp(-d2 / (r2 * 0.35));
        this.u[i] += dx * w * 1.8;
        this.v[i] -= dy * w * 1.8; // invert Y for top-down canvas
        this.dyeR[i] = Math.min(3, this.dyeR[i] + color[0] * w);
        this.dyeG[i] = Math.min(3, this.dyeG[i] + color[1] * w);
        this.dyeB[i] = Math.min(3, this.dyeB[i] + color[2] * w);
      }
    }
  }

  drawObstacle(nx: number, ny: number, radius: number, erase: boolean): void {
    const cx = nx * GRID_W;
    const cy = (1 - ny) * GRID_H;
    const rCells = Math.max(1.5, radius * GRID_W);
    const r2 = rCells * rCells;
    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (d2 <= r2) {
          const i = y * GRID_W + x;
          this.obstacles[i] = erase ? 0 : 1;
          if (!erase) {
            this.u[i] = 0;
            this.v[i] = 0;
            this.dyeR[i] = 0;
            this.dyeG[i] = 0;
            this.dyeB[i] = 0;
          }
        }
      }
    }
  }

  clearObstacles(): void {
    this.obstacles.fill(0);
  }

  applyKarmanPreset(preset: KarmanPreset): void {
    this.clearObstacles();
    const cx = 0.28 * GRID_W;
    const cy = 0.5 * GRID_H;
    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        let solid = false;
        if (preset === 'cylinder') {
          solid = (x - cx) ** 2 + (y - cy) ** 2 < (0.07 * GRID_H * 1.4) ** 2;
        } else if (preset === 'airfoil') {
          const dx = (x - cx) / (0.22 * GRID_W);
          const dy = (y - cy) / (0.09 * GRID_H);
          solid = dx > -0.4 && dx < 0.6 && Math.abs(dy) < Math.max(0, 0.55 * (1 - dx));
        } else {
          const d1 = (x - cx) ** 2 + (y - 0.38 * GRID_H) ** 2;
          const d2 = (x - cx) ** 2 + (y - 0.62 * GRID_H) ** 2;
          const rad2 = (0.055 * GRID_H * 1.3) ** 2;
          solid = d1 < rad2 || d2 < rad2;
        }
        if (solid) this.obstacles[y * GRID_W + x] = 1;
      }
    }
  }

  burstPulse(nx: number, ny: number, mode: FluidMode): void {
    const count = 12;
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2;
      const col: [number, number, number] =
        mode === 'ferrofluid'
          ? [0.7, 0.85, 1.0]
          : mode === 'plankton'
          ? [0.05, 0.9, 1.0]
          : i % 2 === 0
          ? [0.1, 0.9, 1.0]
          : [0.85, 0.3, 1.0];
      this.splat(nx + Math.cos(a) * 0.02, ny + Math.sin(a) * 0.02, Math.cos(a) * 4, Math.sin(a) * 4, col, 0.002);
    }
  }

  resetField(mode: FluidMode): void {
    this.u.fill(0);
    this.v.fill(0);
    this.p.fill(0);
    this.dyeR.fill(0);
    this.dyeG.fill(0);
    this.dyeB.fill(0);
    if (mode !== 'karman') {
      this.clearObstacles();
    }
    this.seedParticles();
  }

  private sampleField(arr: Float32Array, fx: number, fy: number): number {
    const x = Math.max(0.5, Math.min(GRID_W - 1.5, fx));
    const y = Math.max(0.5, Math.min(GRID_H - 1.5, fy));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const sx = x - x0;
    const sy = y - y0;
    const i00 = y0 * GRID_W + x0;
    const i10 = i00 + 1;
    const i01 = i00 + GRID_W;
    const i11 = i01 + 1;
    const top = arr[i00] * (1 - sx) + arr[i10] * sx;
    const bot = arr[i01] * (1 - sx) + arr[i11] * sx;
    return top * (1 - sy) + bot * sy;
  }

  private advect(
    src: Float32Array,
    dst: Float32Array,
    dt: number,
    dissipation: number
  ): void {
    const scale = dt * 36;
    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = y * GRID_W + x;
        if (this.obstacles[i]) {
          dst[i] = 0;
          continue;
        }
        const bx = x - scale * this.u[i];
        const by = y - scale * this.v[i];
        dst[i] = dissipation * this.sampleField(src, bx, by);
      }
    }
  }

  private project(iterations: number): void {
    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = y * GRID_W + x;
        if (this.obstacles[i]) {
          this.div[i] = 0;
          this.p[i] = 0;
          continue;
        }
        this.div[i] = 0.5 * (this.u[i + 1] - this.u[i - 1] + this.v[i + GRID_W] - this.v[i - GRID_W]);
        this.p[i] *= 0.8;
      }
    }

    for (let k = 0; k < iterations; k++) {
      for (let y = 1; y < GRID_H - 1; y++) {
        for (let x = 1; x < GRID_W - 1; x++) {
          const i = y * GRID_W + x;
          if (this.obstacles[i]) continue;
          const pC = this.p[i];
          const pL = this.obstacles[i - 1] ? pC : this.p[i - 1];
          const pR = this.obstacles[i + 1] ? pC : this.p[i + 1];
          const pT = this.obstacles[i - GRID_W] ? pC : this.p[i - GRID_W];
          const pB = this.obstacles[i + GRID_W] ? pC : this.p[i + GRID_W];
          this.p[i] = (pL + pR + pT + pB - this.div[i]) * 0.25;
        }
      }
    }

    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = y * GRID_W + x;
        if (this.obstacles[i]) {
          this.u[i] = 0;
          this.v[i] = 0;
          continue;
        }
        const pC = this.p[i];
        const pL = this.obstacles[i - 1] ? pC : this.p[i - 1];
        const pR = this.obstacles[i + 1] ? pC : this.p[i + 1];
        const pT = this.obstacles[i - GRID_W] ? pC : this.p[i - GRID_W];
        const pB = this.obstacles[i + GRID_W] ? pC : this.p[i + GRID_W];
        this.u[i] -= 0.5 * (pR - pL);
        this.v[i] -= 0.5 * (pB - pT);
      }
    }
  }

  step(dt: number, params: FluidParams, magnets: MagnetPole[]): void {
    const cdt = Math.min(dt, 0.025);
    this.time += cdt;

    // 1. Curl & Vorticity Confinement
    let maxCurl = 0;
    let sumKE = 0;
    let sumSpeed = 0;
    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = y * GRID_W + x;
        this.curl[i] = 0.5 * (this.v[i + 1] - this.v[i - 1] - (this.u[i + GRID_W] - this.u[i - GRID_W]));
        const ac = Math.abs(this.curl[i]);
        if (ac > maxCurl) maxCurl = ac;
      }
    }

    const vc = params.vorticity * cdt * 0.06;
    for (let y = 1; y < GRID_H - 1; y++) {
      for (let x = 1; x < GRID_W - 1; x++) {
        const i = y * GRID_W + x;
        if (this.obstacles[i]) continue;
        const dx = Math.abs(this.curl[i + 1]) - Math.abs(this.curl[i - 1]);
        const dy = Math.abs(this.curl[i + GRID_W]) - Math.abs(this.curl[i - GRID_W]);
        const len = Math.hypot(dx, dy) + 1e-4;
        this.u[i] += ((dy / len) * this.curl[i]) * vc;
        this.v[i] -= ((dx / len) * this.curl[i]) * vc;
        this.u[i] += params.windJet * 1.4 * cdt;

        if (params.mode === 'karman' && x < 4) {
          this.u[i] = 1.9 + Math.max(params.windJet, -0.8);
          this.v[i] = 0.05 * Math.sin(this.time * 3 + y * 0.4);
          if (Math.sin(y * 0.8) > 0.25) {
            this.dyeR[i] = 0.1;
            this.dyeG[i] = 0.85;
            this.dyeB[i] = 0.98;
          }
        }
      }
    }

    // Magnetic forces in ferrofluid mode
    if (params.mode === 'ferrofluid') {
      for (const m of magnets) {
        const mx = m.x * GRID_W;
        const my = (1 - m.y) * GRID_H;
        const rad = Math.max(4, m.radius * GRID_W);
        for (let y = 1; y < GRID_H - 1; y++) {
          for (let x = 1; x < GRID_W - 1; x++) {
            const i = y * GRID_W + x;
            const dx = mx - x;
            const dy = my - y;
            const d2 = dx * dx + dy * dy + 2;
            const w = Math.exp(-d2 / (rad * rad * 1.4));
            const dist = Math.sqrt(d2);
            this.u[i] += (dx / dist) * m.strength * 2.5 * w * cdt;
            this.v[i] += (dy / dist) * m.strength * 2.5 * w * cdt;
            this.dyeR[i] = Math.max(this.dyeR[i], 0.7 * w);
            this.dyeG[i] = Math.max(this.dyeG[i], 0.8 * w);
            this.dyeB[i] = Math.max(this.dyeB[i], 0.9 * w);
          }
        }
      }
    }

    if (params.autoDemo && (params.mode === 'plasma' || params.mode === 'plankton')) {
      const nx = 0.5 + 0.25 * Math.cos(this.time * 0.85);
      const ny = 0.5 + 0.22 * Math.sin(this.time * 1.3);
      const col: [number, number, number] =
        params.mode === 'plasma' ? [0.1, 0.85, 1.0] : [0.02, 0.35, 0.55];
      this.splat(nx, ny, -Math.sin(this.time * 1.3) * 1.8, Math.cos(this.time * 0.85) * 1.8, col, 0.0012);
    }

    // 2. Project & Advect
    this.project(params.jacobiIterations);
    this.uPrev.set(this.u);
    this.vPrev.set(this.v);
    this.advect(this.uPrev, this.u, cdt, 0.994);
    this.advect(this.vPrev, this.v, cdt, 0.994);
    this.project(params.jacobiIterations);

    this.dyeR0.set(this.dyeR);
    this.dyeG0.set(this.dyeG);
    this.dyeB0.set(this.dyeB);
    this.advect(this.dyeR0, this.dyeR, cdt, params.dissipation);
    this.advect(this.dyeG0, this.dyeG, cdt, params.dissipation);
    this.advect(this.dyeB0, this.dyeB, cdt, params.dissipation);

    // 3. Advect 6,400 particles + gather telemetry
    for (let i = 0; i < N; i++) {
      const s2 = this.u[i] * this.u[i] + this.v[i] * this.v[i];
      sumKE += 0.5 * s2;
      sumSpeed += Math.sqrt(s2);
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const fx = this.px[i] * GRID_W;
      const fy = (1 - this.py[i]) * GRID_H;
      const vx = this.sampleField(this.u, fx, fy);
      const vy = this.sampleField(this.v, fx, fy);
      const c = Math.abs(this.sampleField(this.curl, fx, fy));
      this.px[i] += vx * cdt * 0.28;
      this.py[i] -= vy * cdt * 0.28;
      this.pEnergy[i] = this.pEnergy[i] * 0.9 + Math.min(1.5, (c * 1.4 + Math.hypot(vx, vy) * 0.35) * params.biolumSensitivity) * 0.1;
      this.pLife[i] -= cdt * 0.15;

      const gx = Math.floor(this.px[i] * GRID_W);
      const gy = Math.floor((1 - this.py[i]) * GRID_H);
      const hitObs = gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H && this.obstacles[this.idx(gx, gy)] === 1;
      if (this.pLife[i] <= 0 || this.px[i] < 0 || this.px[i] > 1 || this.py[i] < 0 || this.py[i] > 1 || hitObs) {
        this.px[i] = params.mode === 'karman' ? Math.random() * 0.03 : Math.random();
        this.py[i] = Math.random();
        this.pEnergy[i] = 0.1;
        this.pLife[i] = 0.5 + Math.random() * 0.5;
      }
    }

    const avgSpeed = sumSpeed / N;
    const avgKE = sumKE / N;
    const nu = Math.max(params.viscosity, 0.00004);
    this.telemetry = {
      fps: this.fps,
      reynoldsNumber: Math.round((avgSpeed * 0.25) / nu),
      peakVorticity: maxCurl * 18.5,
      kineticEnergy: avgKE,
      turbulenceRatio: Math.min(1, avgKE / 2.2),
      particleCount: PARTICLE_COUNT,
      backendLabel: this.backendLabel,
    };
  }

  render(params: FluidParams, magnets: MagnetPole[]): void {
    this.fpsFrames++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 450) {
      this.fps = Math.round((this.fpsFrames * 1000) / (now - this.lastFpsTime));
      this.telemetry.fps = this.fps;
      this.fpsFrames = 0;
      this.lastFpsTime = now;
    }

    const d = this.imgData.data;
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      if (this.obstacles[i]) {
        d[o] = 22;
        d[o + 1] = 36;
        d[o + 2] = 60;
        d[o + 3] = 255;
      } else if (params.mode === 'ferrofluid') {
        const dens = Math.min(1, (this.dyeR[i] + this.dyeG[i] + this.dyeB[i]) * 0.33);
        const spec = Math.min(255, Math.round(dens * 210 + Math.abs(this.curl[i]) * 180));
        d[o] = Math.round(10 + spec * 0.7);
        d[o + 1] = Math.round(14 + spec * 0.85);
        d[o + 2] = Math.round(24 + spec);
        d[o + 3] = 255;
      } else {
        d[o] = Math.min(255, Math.round(8 + this.dyeR[i] * 220));
        d[o + 1] = Math.min(255, Math.round(12 + this.dyeG[i] * 220));
        d[o + 2] = Math.min(255, Math.round(24 + this.dyeB[i] * 220));
        d[o + 3] = 255;
      }
    }
    this.offCtx.putImageData(this.imgData, 0, 0);

    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(this.offscreen, 0, 0, w, h);

    // Draw 6,400 Lagrangian particles
    this.ctx.save();
    this.ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const x = this.px[i] * w;
      const y = (1 - this.py[i]) * h;
      const e = Math.min(1, this.pEnergy[i]);
      const sz = params.mode === 'plankton' ? 1.5 + e * 2.2 : 1.2 + e * 1.4;
      this.ctx.fillStyle =
        params.mode === 'plankton'
          ? `rgba(6, 220, 255, ${0.22 + e * 0.72})`
          : `rgba(120, 210, 255, ${0.15 + e * 0.45})`;
      this.ctx.fillRect(x, y, sz, sz);
    }
    this.ctx.restore();

    if (params.mode === 'ferrofluid') {
      for (const m of magnets) {
        const mx = m.x * w;
        const my = (1 - m.y) * h;
        this.ctx.beginPath();
        this.ctx.arc(mx, my, 12, 0, Math.PI * 2);
        this.ctx.strokeStyle = m.strength >= 0 ? '#06b6d4' : '#f59e0b';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
      }
    }
  }

  getTelemetry(): TelemetrySnapshot {
    return this.telemetry;
  }

  export4KSnapshot(params: FluidParams, magnets: MagnetPole[]): string {
    const out = document.createElement('canvas');
    out.width = 3840;
    out.height = 2160;
    const octx = out.getContext('2d')!;
    this.render(params, magnets);
    octx.drawImage(this.canvas, 0, 0, 3840, 2160);
    return out.toDataURL('image/png');
  }

  dispose(): void {
    // Nothing to release for 2D arrays
  }
}
