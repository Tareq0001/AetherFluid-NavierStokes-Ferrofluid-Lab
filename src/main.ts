import './style.css';
import { WebGLFluidEngine } from './engine/WebGLFluidEngine';
import { CanvasFluidEngine } from './engine/CanvasFluidEngine';
import { FluidAudio } from './engine/FluidAudio';
import {
  FluidEngine,
  FluidMode,
  FluidParams,
  InteractionTool,
  KarmanPreset,
  MagnetPole,
} from './engine/types';
import { TelemetryScope } from './components/TelemetryScope';
import { MathDossierModal } from './components/MathDossierModal';

type PlasmaPalette = 'cyan_violet' | 'solar' | 'aurora';

const MODE_META: Record<FluidMode, { title: string; hint: string }> = {
  plasma: {
    title: 'Neon Plasma Advection',
    hint: 'Drag across canvas to inject momentum & thermal plasma filaments',
  },
  ferrofluid: {
    title: 'Ferrofluid Magnetic Spikes',
    hint: 'Drag magnetic poles or switch tool to add attract/repel dipoles',
  },
  karman: {
    title: 'Kármán Vortex Street & Wind Tunnel',
    hint: 'Draw obstacles directly in the wind tunnel to observe alternating vortex shedding',
  },
  plankton: {
    title: 'Bioluminescent Plankton',
    hint: 'Stir the pelagic abyss — 8,192+ plankton ignite with local shear rate & curl',
  },
};

function createFluidEngine(canvas: HTMLCanvasElement): { engine: FluidEngine; canvas: HTMLCanvasElement } {
  try {
    return { engine: new WebGLFluidEngine(canvas), canvas };
  } catch (err) {
    console.warn('WebGL2 FBO solver unavailable, activating Canvas 2D TypedArray fallback:', err);
    // Replace canvas because calling getContext('webgl2') permanently locks it against '2d'
    const freshCanvas = canvas.cloneNode(false) as HTMLCanvasElement;
    canvas.parentNode?.replaceChild(freshCanvas, canvas);
    return { engine: new CanvasFluidEngine(freshCanvas), canvas: freshCanvas };
  }
}

class AetherFluidApp {
  private canvas: HTMLCanvasElement;
  private engine: FluidEngine;
  private audio = new FluidAudio();
  private scope: TelemetryScope;
  private dossier: MathDossierModal;

  private params: FluidParams = {
    mode: 'plasma',
    viscosity: 0.0002,
    vorticity: 28,
    dissipation: 0.985,
    windJet: 0.0,
    jacobiIterations: 20,
    autoDemo: true,
    magneticGain: 0.95,
    biolumSensitivity: 1.15,
    brushRadius: 0.032,
  };

  private activeTool: InteractionTool = 'fluid';
  private plasmaPalette: PlasmaPalette = 'cyan_violet';
  private newMagnetStrength = 1.15;
  private nextMagnetId = 1;
  private magnets: MagnetPole[] = [
    { id: 1, x: 0.36, y: 0.52, strength: 1.2, radius: 0.14 },
    { id: 2, x: 0.64, y: 0.48, strength: 1.2, radius: 0.14 },
  ];

  private isPaused = false;
  private isPointerDown = false;
  private draggingMagnetId: number | null = null;
  private lastNx = 0.5;
  private lastNy = 0.5;
  private lastTime = performance.now();
  private hueClock = 0;

  constructor() {
    const initialCanvas = document.getElementById('fluid-canvas') as HTMLCanvasElement;
    this.canvas = initialCanvas;
    this.resizeArenaCanvas();
    const created = createFluidEngine(this.canvas);
    this.engine = created.engine;
    this.canvas = created.canvas;
    this.resizeArenaCanvas();
    this.scope = new TelemetryScope(document.getElementById('telemetry-canvas') as HTMLCanvasElement);
    this.dossier = new MathDossierModal(document.getElementById('math-modal') as HTMLElement);

    this.nextMagnetId = 3;
    this.updateBackendBadge();
    this.bindUI();
    this.bindPointerEvents();
    this.renderModeToolsPanel();
    this.syncModeUI();

    // Seed initial energetic swirl so Frame 1 looks alive
    this.engine.burstPulse(0.5, 0.5, this.params.mode);
    requestAnimationFrame((t) => this.loop(t));
  }

  private resizeArenaCanvas(): void {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.round(rect.width * dpr));
    const h = Math.max(240, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.engine?.resize(w, h);
    }
  }

  private updateBackendBadge(): void {
    const backendEl = document.getElementById('hud-backend');
    const countEl = document.getElementById('val-particle-count');
    if (backendEl) backendEl.textContent = this.engine.backendLabel;
    if (countEl) countEl.textContent = this.engine.particleCount.toLocaleString();
  }

  private bindUI(): void {
    window.addEventListener('resize', () => {
      this.resizeArenaCanvas();
      this.scope.resize();
    });

    // Mode tabs
    document.querySelectorAll<HTMLButtonElement>('.system-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode as FluidMode;
        if (!mode) return;
        this.switchMode(mode);
      });
    });

    // Header actions
    const audioBtn = document.getElementById('audio-toggle-btn')!;
    const audioTxt = document.getElementById('audio-status-text')!;
    const audioIcon = document.getElementById('audio-icon')!;
    audioBtn.addEventListener('click', () => {
      const on = this.audio.toggle();
      audioBtn.classList.toggle('active', on);
      audioTxt.textContent = on ? 'Audio: ON' : 'Audio: OFF';
      audioIcon.innerHTML = on
        ? '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>'
        : '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>';
    });

    document.getElementById('open-math-btn')?.addEventListener('click', () => this.dossier.open());
    document.getElementById('btn-export-4k')?.addEventListener('click', () => this.export4K());

    // Bottom simulation controls
    const playBtn = document.getElementById('btn-play-pause')!;
    playBtn.addEventListener('click', () => this.togglePause());

    document.getElementById('btn-burst-pulse')?.addEventListener('click', () => {
      this.engine.burstPulse(0.5, 0.5, this.params.mode);
    });

    const autoBtn = document.getElementById('btn-auto-demo')!;
    autoBtn.addEventListener('click', () => {
      this.params.autoDemo = !this.params.autoDemo;
      autoBtn.classList.toggle('active', this.params.autoDemo);
    });

    document.getElementById('btn-clear-obstacles')?.addEventListener('click', () => {
      this.engine.clearObstacles();
      if (this.params.mode === 'ferrofluid') {
        this.magnets = [];
        this.renderModeToolsPanel();
      }
    });

    document.getElementById('btn-reset-fluid')?.addEventListener('click', () => {
      this.engine.resetField(this.params.mode);
      if (this.params.mode === 'karman') {
        this.engine.applyKarmanPreset('cylinder');
      } else {
        this.engine.burstPulse(0.5, 0.5, this.params.mode);
      }
    });

    // Global Navier-Stokes sliders
    this.bindSlider('slider-viscosity', 'disp-viscosity', (v) => {
      this.params.viscosity = v;
      return v.toFixed(4);
    });
    this.bindSlider('slider-vorticity', 'disp-vorticity', (v) => {
      this.params.vorticity = v;
      return v.toFixed(1);
    });
    this.bindSlider('slider-dissipation', 'disp-dissipation', (v) => {
      this.params.dissipation = v;
      return v.toFixed(3);
    });
    this.bindSlider('slider-wind-jet', 'disp-wind-jet', (v) => {
      this.params.windJet = v;
      return v.toFixed(2);
    });

    document.querySelectorAll<HTMLButtonElement>('[data-iters]').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-iters]').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const iters = Number(btn.dataset.iters || 20);
        this.params.jacobiIterations = iters;
        const disp = document.getElementById('disp-iterations');
        if (disp) disp.textContent = `${iters} iters`;
      });
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === 'Space') {
        e.preventDefault();
        this.togglePause();
      } else if (e.key.toLowerCase() === 'b') {
        this.engine.burstPulse(0.5, 0.5, this.params.mode);
      } else if (e.key.toLowerCase() === 'c') {
        this.engine.clearObstacles();
      } else if (e.key.toLowerCase() === 'm') {
        audioBtn.click();
      }
    });
  }

  private bindSlider(
    sliderId: string,
    dispId: string,
    onUpdate: (val: number) => string
  ): void {
    const slider = document.getElementById(sliderId) as HTMLInputElement | null;
    const disp = document.getElementById(dispId);
    if (!slider || !disp) return;
    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      disp.textContent = onUpdate(val);
    });
  }

  private togglePause(): void {
    this.isPaused = !this.isPaused;
    const txt = document.getElementById('txt-play-pause');
    const icon = document.getElementById('icon-play-pause');
    if (txt) txt.textContent = this.isPaused ? 'Resume' : 'Pause';
    if (icon) {
      icon.innerHTML = this.isPaused
        ? '<polygon points="5 3 19 12 5 21 5 3"></polygon>'
        : '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    }
  }

  private switchMode(mode: FluidMode): void {
    this.params.mode = mode;
    document.querySelectorAll<HTMLButtonElement>('.system-tab').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    if (mode === 'karman') {
      this.activeTool = 'obstacle_draw';
      this.engine.resetField('karman');
      this.engine.applyKarmanPreset('cylinder');
    } else if (mode === 'ferrofluid') {
      this.activeTool = 'fluid';
      this.engine.clearObstacles();
      if (this.magnets.length === 0) {
        this.magnets = [
          { id: this.nextMagnetId++, x: 0.36, y: 0.52, strength: 1.2, radius: 0.14 },
          { id: this.nextMagnetId++, x: 0.64, y: 0.48, strength: 1.2, radius: 0.14 },
        ];
      }
    } else {
      this.activeTool = 'fluid';
      this.engine.clearObstacles();
      this.engine.burstPulse(0.5, 0.5, mode);
    }

    this.syncModeUI();
    this.renderModeToolsPanel();
  }

  private syncModeUI(): void {
    const meta = MODE_META[this.params.mode];
    const titleEl = document.getElementById('hud-mode-title');
    const hintEl = document.getElementById('hud-interaction-hint');
    if (titleEl) titleEl.textContent = meta.title;
    if (hintEl) hintEl.innerHTML = `<span>${meta.hint}</span>`;
  }

  private renderModeToolsPanel(): void {
    const panel = document.getElementById('panel-mode-tools');
    if (!panel) return;

    if (this.params.mode === 'plasma') {
      panel.innerHTML = `
        <div class="panel-header">
          <span>Thermal Plasma Controls</span>
          <span class="badge">MODE 01</span>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Plasma Chromatic Spectrum</span>
          </div>
          <div class="preset-btn-group">
            <button class="sub-btn ${this.plasmaPalette === 'cyan_violet' ? 'active' : ''}" data-pal="cyan_violet">Cyan / Violet</button>
            <button class="sub-btn ${this.plasmaPalette === 'solar' ? 'active' : ''}" data-pal="solar">Solar Ember</button>
            <button class="sub-btn ${this.plasmaPalette === 'aurora' ? 'active' : ''}" data-pal="aurora">Aurora Bio</button>
          </div>
        </div>
        <div class="helper-note">
          Drag pointer anywhere to inject high-momentum plasma filaments amplified by Vorticity Confinement.
        </div>
      `;
      panel.querySelectorAll<HTMLButtonElement>('[data-pal]').forEach((b) => {
        b.addEventListener('click', () => {
          this.plasmaPalette = b.dataset.pal as PlasmaPalette;
          this.renderModeToolsPanel();
        });
      });
    } else if (this.params.mode === 'ferrofluid') {
      panel.innerHTML = `
        <div class="panel-header">
          <span>Ferrofluid Dipole Studio</span>
          <span class="badge">${this.magnets.length} POLES</span>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Pointer Tool</span>
          </div>
          <div class="preset-btn-group">
            <button class="sub-btn ${this.activeTool === 'fluid' ? 'active' : ''}" data-tool="fluid">Drag Poles / Stir</button>
            <button class="sub-btn ${this.activeTool === 'magnet_place' ? 'active' : ''}" data-tool="magnet_place">Place New Pole</button>
          </div>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>New Pole Polarity</span>
            <span class="control-val">${this.newMagnetStrength >= 0 ? 'North (Attract)' : 'South (Repel)'}</span>
          </div>
          <div class="preset-btn-group">
            <button class="sub-btn ${this.newMagnetStrength >= 0 ? 'active' : ''}" id="btn-pol-attract">Attract (+N)</button>
            <button class="sub-btn ${this.newMagnetStrength < 0 ? 'active' : ''}" id="btn-pol-repel">Repel (-S)</button>
            <button class="sub-btn" id="btn-flip-all">Invert All</button>
          </div>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Rosensweig Spike Gain</span>
            <span class="control-val" id="disp-mag-gain">${this.params.magneticGain.toFixed(2)}</span>
          </div>
          <input type="range" id="slider-mag-gain" min="0.2" max="2.0" step="0.05" value="${this.params.magneticGain}" />
        </div>
        <div class="helper-note">
          Drag cyan/orange dipole rings directly to sculpt liquid obsidian Rosensweig magnetic spikes in real time.
        </div>
      `;
      panel.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((b) => {
        b.addEventListener('click', () => {
          this.activeTool = b.dataset.tool as InteractionTool;
          this.renderModeToolsPanel();
        });
      });
      panel.querySelector('#btn-pol-attract')?.addEventListener('click', () => {
        this.newMagnetStrength = 1.2;
        this.renderModeToolsPanel();
      });
      panel.querySelector('#btn-pol-repel')?.addEventListener('click', () => {
        this.newMagnetStrength = -1.2;
        this.renderModeToolsPanel();
      });
      panel.querySelector('#btn-flip-all')?.addEventListener('click', () => {
        this.magnets = this.magnets.map((m) => ({ ...m, strength: -m.strength }));
        this.renderModeToolsPanel();
      });
      this.bindSlider('slider-mag-gain', 'disp-mag-gain', (v) => {
        this.params.magneticGain = v;
        return v.toFixed(2);
      });
    } else if (this.params.mode === 'karman') {
      panel.innerHTML = `
        <div class="panel-header">
          <span>Wind Tunnel &amp; Obstacles</span>
          <span class="badge">BOUNDARY CFD</span>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Mouse Interaction Tool</span>
          </div>
          <div class="preset-btn-group">
            <button class="sub-btn ${this.activeTool === 'obstacle_draw' ? 'active' : ''}" data-tool="obstacle_draw">Draw Obstacles</button>
            <button class="sub-btn ${this.activeTool === 'obstacle_erase' ? 'active' : ''}" data-tool="obstacle_erase">Erase</button>
            <button class="sub-btn ${this.activeTool === 'fluid' ? 'active' : ''}" data-tool="fluid">Stir Flow</button>
          </div>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Aerodynamic Obstacle Presets</span>
          </div>
          <div class="preset-btn-group">
            <button class="sub-btn" data-preset="cylinder">Cylinder</button>
            <button class="sub-btn" data-preset="airfoil">NACA Airfoil</button>
            <button class="sub-btn" data-preset="twin_cylinders">Twin Cylinders</button>
          </div>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Obstacle Brush Radius</span>
            <span class="control-val" id="disp-brush-rad">${(this.params.brushRadius * 100).toFixed(1)}%</span>
          </div>
          <input type="range" id="slider-brush-rad" min="0.012" max="0.075" step="0.002" value="${this.params.brushRadius}" />
        </div>
        <div class="helper-note">
          Paint solid bluff bodies or wings directly in the wind tunnel to induce alternating von Kármán vortex streets.
        </div>
      `;
      panel.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach((b) => {
        b.addEventListener('click', () => {
          this.activeTool = b.dataset.tool as InteractionTool;
          this.renderModeToolsPanel();
        });
      });
      panel.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach((b) => {
        b.addEventListener('click', () => {
          this.engine.applyKarmanPreset(b.dataset.preset as KarmanPreset);
        });
      });
      this.bindSlider('slider-brush-rad', 'disp-brush-rad', (v) => {
        this.params.brushRadius = v;
        return `${(v * 100).toFixed(1)}%`;
      });
    } else {
      panel.innerHTML = `
        <div class="panel-header">
          <span>Bioluminescent Ocean Lab</span>
          <span class="badge">${this.engine.particleCount.toLocaleString()} PLANKTON</span>
        </div>
        <div class="control-group">
          <div class="control-label-row">
            <span>Shear Bioluminescence Gain</span>
            <span class="control-val" id="disp-biolum">${this.params.biolumSensitivity.toFixed(2)}×</span>
          </div>
          <input type="range" id="slider-biolum" min="0.3" max="2.5" step="0.05" value="${this.params.biolumSensitivity}" />
        </div>
        <div class="preset-btn-group">
          <button class="sub-btn" id="btn-plankton-wave">Trigger Pelagic Current Wave</button>
        </div>
        <div class="helper-note">
          Over 8,192 Lagrangian bioluminescent plankton follow RK2 streamlines and glow cyan/electric blue with fluid shear &amp; curl.
        </div>
      `;
      this.bindSlider('slider-biolum', 'disp-biolum', (v) => {
        this.params.biolumSensitivity = v;
        return `${v.toFixed(2)}×`;
      });
      panel.querySelector('#btn-plankton-wave')?.addEventListener('click', () => {
        this.engine.burstPulse(0.32, 0.5, 'plankton');
        this.engine.burstPulse(0.68, 0.5, 'plankton');
      });
    }
  }

  private getNormalizedCoords(e: PointerEvent): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(rect.width, 1)));
    const ny = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top) / Math.max(rect.height, 1)));
    return [nx, ny];
  }

  private getSplatColor(): [number, number, number] {
    if (this.params.mode === 'ferrofluid') return [0.75, 0.85, 0.96];
    if (this.params.mode === 'plankton') return [0.02, 0.45, 0.72];
    if (this.params.mode === 'karman') return [0.08, 0.85, 0.96];

    this.hueClock = (this.hueClock + 0.015) % 1;
    if (this.plasmaPalette === 'solar') {
      return [1.0, 0.35 + 0.35 * Math.sin(this.hueClock * 6.28), 0.06];
    }
    if (this.plasmaPalette === 'aurora') {
      return [0.06, 0.92, 0.45 + 0.4 * Math.cos(this.hueClock * 6.28)];
    }
    // cyan_violet
    const t = 0.5 + 0.5 * Math.sin(this.hueClock * Math.PI * 2);
    return [0.05 + t * 0.65, 0.82 - t * 0.45, 1.0];
  }

  private findNearestMagnet(nx: number, ny: number): MagnetPole | null {
    let best: MagnetPole | null = null;
    let bestDist = 0.045;
    for (const m of this.magnets) {
      const d = Math.hypot(m.x - nx, m.y - ny);
      if (d < bestDist) {
        bestDist = d;
        best = m;
      }
    }
    return best;
  }

  private bindPointerEvents(): void {
    this.canvas.addEventListener('pointerdown', (e) => {
      this.canvas.setPointerCapture(e.pointerId);
      const [nx, ny] = this.getNormalizedCoords(e);
      this.isPointerDown = true;
      this.lastNx = nx;
      this.lastNy = ny;

      if (this.params.mode === 'ferrofluid') {
        const hit = this.findNearestMagnet(nx, ny);
        if (hit) {
          this.draggingMagnetId = hit.id;
          return;
        }
        if (this.activeTool === 'magnet_place') {
          if (this.magnets.length >= 12) this.magnets.shift();
          const pole: MagnetPole = {
            id: this.nextMagnetId++,
            x: nx,
            y: ny,
            strength: this.newMagnetStrength,
            radius: 0.14,
          };
          this.magnets.push(pole);
          this.draggingMagnetId = pole.id;
          this.renderModeToolsPanel();
          return;
        }
      }

      if (this.activeTool === 'obstacle_draw' || this.activeTool === 'obstacle_erase') {
        this.engine.drawObstacle(nx, ny, this.params.brushRadius, this.activeTool === 'obstacle_erase');
      }
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.isPointerDown) return;
      const [nx, ny] = this.getNormalizedCoords(e);

      if (this.draggingMagnetId !== null) {
        const mag = this.magnets.find((m) => m.id === this.draggingMagnetId);
        if (mag) {
          mag.x = nx;
          mag.y = ny;
        }
        this.lastNx = nx;
        this.lastNy = ny;
        return;
      }

      if (this.activeTool === 'obstacle_draw' || this.activeTool === 'obstacle_erase') {
        const erase = this.activeTool === 'obstacle_erase';
        const steps = 6;
        for (let i = 1; i <= steps; i++) {
          const ix = this.lastNx + ((nx - this.lastNx) * i) / steps;
          const iy = this.lastNy + ((ny - this.lastNy) * i) / steps;
          this.engine.drawObstacle(ix, iy, this.params.brushRadius, erase);
        }
      } else {
        const dx = (nx - this.lastNx) * 32;
        const dy = (ny - this.lastNy) * 32;
        this.engine.splat(nx, ny, dx, dy, this.getSplatColor(), 0.0014);
      }

      this.lastNx = nx;
      this.lastNy = ny;
    });

    const endPointer = (e: PointerEvent) => {
      if (this.canvas.hasPointerCapture(e.pointerId)) {
        this.canvas.releasePointerCapture(e.pointerId);
      }
      this.isPointerDown = false;
      this.draggingMagnetId = null;
    };

    this.canvas.addEventListener('pointerup', endPointer);
    this.canvas.addEventListener('pointercancel', endPointer);
  }

  private updateTelemetryUI(): void {
    const t = this.engine.getTelemetry();
    const fpsStr = `${t.fps} FPS`;
    const reStr = t.reynoldsNumber.toLocaleString();
    const vortStr = `${t.peakVorticity.toFixed(1)} s⁻¹`;
    const keStr = t.kineticEnergy.toFixed(2);
    const pct = Math.round(t.turbulenceRatio * 100);

    const set = (id: string, html: string) => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = html;
    };

    set('hud-fps', fpsStr);
    set('hud-re', reStr);
    set('hud-vort', vortStr);
    set('hud-ke', keStr);

    set('val-fps', `${t.fps}<span class="unit">FPS</span>`);
    set('val-reynolds', reStr);
    set('val-vorticity', `${t.peakVorticity.toFixed(1)}<span class="unit">s⁻¹</span>`);
    set('val-ke', `${keStr}<span class="unit">J/m²</span>`);
    set('val-turb-pct', `${pct}%`);

    const bar = document.getElementById('bar-turbulence');
    if (bar) bar.style.width = `${Math.max(4, pct)}%`;

    const badge = document.getElementById('flow-regime-badge');
    if (badge) {
      if (t.reynoldsNumber < 600) {
        badge.textContent = 'LAMINAR';
        badge.style.color = '#10b981';
      } else if (t.reynoldsNumber < 2400) {
        badge.textContent = 'TRANSITIONAL';
        badge.style.color = '#06b6d4';
      } else {
        badge.textContent = 'TURBULENT';
        badge.style.color = '#f59e0b';
      }
    }

    this.scope.pushAndRender(t);
    this.audio.update(t, this.params.mode);
  }

  private export4K(): void {
    const url = this.engine.export4KSnapshot(this.params, this.magnets);
    const link = document.createElement('a');
    link.download = `aether-fluid-4k-${this.params.mode}.png`;
    link.href = url;
    link.click();
  }

  private loop(now: number): void {
    const dt = Math.min(0.033, Math.max(0.004, (now - this.lastTime) / 1000));
    this.lastTime = now;

    if (!this.isPaused) {
      this.engine.step(dt, this.params, this.magnets);
      this.updateTelemetryUI();
    }
    this.engine.render(this.params, this.magnets);
    requestAnimationFrame((t) => this.loop(t));
  }
}

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', () => new AetherFluidApp());
} else {
  new AetherFluidApp();
}
