import { TelemetrySnapshot } from '../engine/types';

export class TelemetryScope {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private keHistory: number[] = [];
  private vortHistory: number[] = [];
  private readonly maxSamples = 96;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.resize();
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(200, Math.round(rect.width * dpr));
    const h = Math.max(100, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  pushAndRender(telemetry: TelemetrySnapshot): void {
    this.keHistory.push(Math.min(1, telemetry.kineticEnergy / 2.5));
    this.vortHistory.push(Math.min(1, telemetry.peakVorticity / 55));
    if (this.keHistory.length > this.maxSamples) this.keHistory.shift();
    if (this.vortHistory.length > this.maxSamples) this.vortHistory.shift();

    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#040711';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.14)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (h * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    this.drawSeries(this.keHistory, '#06b6d4', 'rgba(6, 182, 212, 0.18)', w, h);
    this.drawSeries(this.vortHistory, '#f59e0b', 'rgba(245, 158, 11, 0.10)', w, h);

    ctx.font = `${Math.max(10, Math.round(h * 0.1))}px "Fira Code", monospace`;
    ctx.fillStyle = '#06b6d4';
    ctx.fillText('Ek (Cyan)', 8, 16);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('ω_max (Gold)', 92, 16);
  }

  private drawSeries(
    data: number[],
    stroke: string,
    fill: string,
    w: number,
    h: number
  ): void {
    if (data.length < 2) return;
    const ctx = this.ctx;
    const stepX = w / (this.maxSamples - 1);

    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = i * stepX;
      const y = h - 8 - data[i] * (h - 22);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.lineTo((data.length - 1) * stepX, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }
}
