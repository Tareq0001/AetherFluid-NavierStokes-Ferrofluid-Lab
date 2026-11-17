import { FluidMode, TelemetrySnapshot } from './types';

export class FluidAudio {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private noiseFilter: BiquadFilterNode | null = null;
  private noiseGain: GainNode | null = null;
  private droneOsc1: OscillatorNode | null = null;
  private droneOsc2: OscillatorNode | null = null;
  private droneFilter: BiquadFilterNode | null = null;
  private droneGain: GainNode | null = null;
  private enabled = false;

  get isEnabled(): boolean {
    return this.enabled;
  }

  toggle(): boolean {
    if (!this.ctx) {
      this.init();
    }
    this.enabled = !this.enabled;
    if (this.ctx && this.ctx.state === 'suspended' && this.enabled) {
      void this.ctx.resume();
    }
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setTargetAtTime(this.enabled ? 0.22 : 0.0, this.ctx.currentTime, 0.08);
    }
    return this.enabled;
  }

  private init(): void {
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;

    const ctx = new AudioCtx();
    this.ctx = ctx;

    const master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination);
    this.masterGain = master;

    // 1. Pink-ish Filtered Fluid Turbulence Noise
    const bufferSize = ctx.sampleRate * 2;
    const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    let b0 = 0;
    let b1 = 0;
    let b2 = 0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99765 * b0 + white * 0.099046;
      b1 = 0.963 * b1 + white * 0.2965164;
      b2 = 0.57 * b2 + white * 1.0526913;
      output[i] = (b0 + b1 + b2 + white * 0.1848) * 0.12;
    }

    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = noiseBuffer;
    noiseSrc.loop = true;

    const noiseFilter = ctx.createBiquadFilter();
    noiseFilter.type = 'bandpass';
    noiseFilter.frequency.value = 320;
    noiseFilter.Q.value = 1.6;
    this.noiseFilter = noiseFilter;

    const noiseGain = ctx.createGain();
    noiseGain.gain.value = 0.12;
    this.noiseGain = noiseGain;

    noiseSrc.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(master);
    noiseSrc.start();

    // 2. Harmonic Ambient Drone Pads
    const droneFilter = ctx.createBiquadFilter();
    droneFilter.type = 'lowpass';
    droneFilter.frequency.value = 420;
    this.droneFilter = droneFilter;

    const droneGain = ctx.createGain();
    droneGain.gain.value = 0.15;
    this.droneGain = droneGain;

    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.value = 110; // A2
    this.droneOsc1 = osc1;

    const osc2 = ctx.createOscillator();
    osc2.type = 'triangle';
    osc2.frequency.value = 164.81; // E3 fifth harmonic
    this.droneOsc2 = osc2;

    osc1.connect(droneFilter);
    osc2.connect(droneFilter);
    droneFilter.connect(droneGain);
    droneGain.connect(master);

    osc1.start();
    osc2.start();
  }

  update(telemetry: TelemetrySnapshot, mode: FluidMode): void {
    if (!this.enabled || !this.ctx || !this.noiseFilter || !this.noiseGain || !this.droneFilter || !this.droneGain || !this.droneOsc1 || !this.droneOsc2) {
      return;
    }

    const t = this.ctx.currentTime;
    const turb = Math.min(1, telemetry.turbulenceRatio);
    const vort = Math.min(1, telemetry.peakVorticity / 40);

    // Filtered noise responds to turbulence & shear
    const baseFreq = mode === 'plankton' ? 420 : mode === 'ferrofluid' ? 180 : 280;
    const targetNoiseFreq = baseFreq + turb * 1650 + vort * 650;
    this.noiseFilter.frequency.setTargetAtTime(targetNoiseFreq, t, 0.12);
    this.noiseGain.gain.setTargetAtTime(0.05 + turb * 0.28, t, 0.12);

    // Harmonic Drone Pads respond gently to kinetic energy & mode
    const rootHz = mode === 'ferrofluid' ? 82.41 : mode === 'plankton' ? 130.81 : 110.0;
    this.droneOsc1.frequency.setTargetAtTime(rootHz + turb * 8.0, t, 0.25);
    this.droneOsc2.frequency.setTargetAtTime(rootHz * 1.498 + vort * 14.0, t, 0.25);
    this.droneFilter.frequency.setTargetAtTime(240 + turb * 720, t, 0.2);
    this.droneGain.gain.setTargetAtTime(0.11 + (1 - turb * 0.3) * 0.08, t, 0.2);
  }

  dispose(): void {
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
    }
  }
}
