export type FluidMode = 'plasma' | 'ferrofluid' | 'karman' | 'plankton';

export type InteractionTool = 'fluid' | 'obstacle_draw' | 'obstacle_erase' | 'magnet_place';

export type KarmanPreset = 'cylinder' | 'airfoil' | 'twin_cylinders';

export interface MagnetPole {
  id: number;
  x: number; // normalized [0, 1], origin bottom-left in GL or mapped consistently
  y: number; // normalized [0, 1], 0 = bottom, 1 = top
  strength: number; // positive = attract (North), negative = repel (South)
  radius: number;
}

export interface FluidParams {
  mode: FluidMode;
  viscosity: number;        // 0 .. 0.003
  vorticity: number;        // 0 .. 60
  dissipation: number;      // 0.90 .. 0.999
  windJet: number;          // -1.0 .. 1.5
  jacobiIterations: number; // 12 | 20 | 32
  autoDemo: boolean;
  magneticGain: number;     // Ferrofluid Rosensweig spike gain
  biolumSensitivity: number;// Plankton shear excitation gain
  brushRadius: number;      // Normalized obstacle / splat radius
}

export interface TelemetrySnapshot {
  fps: number;
  reynoldsNumber: number;
  peakVorticity: number;
  kineticEnergy: number;
  turbulenceRatio: number; // 0..1
  particleCount: number;
  backendLabel: string;
}

export interface FluidEngine {
  readonly backendLabel: string;
  readonly particleCount: number;
  resize(width: number, height: number): void;
  step(dt: number, params: FluidParams, magnets: MagnetPole[]): void;
  render(params: FluidParams, magnets: MagnetPole[]): void;
  splat(
    nx: number,
    ny: number,
    dx: number,
    dy: number,
    color: [number, number, number],
    radius: number
  ): void;
  drawObstacle(nx: number, ny: number, radius: number, erase: boolean): void;
  clearObstacles(): void;
  applyKarmanPreset(preset: KarmanPreset): void;
  burstPulse(nx: number, ny: number, mode: FluidMode): void;
  resetField(mode: FluidMode): void;
  getTelemetry(): TelemetrySnapshot;
  export4KSnapshot(params: FluidParams, magnets: MagnetPole[]): string;
  dispose(): void;
}
