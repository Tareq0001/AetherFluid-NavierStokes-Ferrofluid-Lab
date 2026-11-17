import {
  FluidEngine,
  FluidMode,
  FluidParams,
  KarmanPreset,
  MagnetPole,
  TelemetrySnapshot,
} from './types';

interface FBO {
  texture: WebGLTexture;
  fbo: WebGLFramebuffer;
  width: number;
  height: number;
  attach(unit: number): number;
}

interface DoubleFBO {
  width: number;
  height: number;
  read: FBO;
  write: FBO;
  swap(): void;
}

const PARTICLE_TEX_W = 128;
const PARTICLE_TEX_H = 64;
const PARTICLE_COUNT = PARTICLE_TEX_W * PARTICLE_TEX_H; // 8,192 GPU particles (> 6,000+)
const MAX_MAGNETS = 12;

const BASE_VERTEX = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 pos = vec2(
    float((gl_VertexID & 1) << 2) - 1.0,
    float((gl_VertexID & 2) << 1) - 1.0
  );
  vUv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

const CLEAR_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTexture;
uniform float uValue;
out vec4 outColor;
void main() {
  outColor = uValue * texture(uTexture, vUv);
}
`;

const SPLAT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTarget;
uniform sampler2D uObstacles;
uniform float uAspect;
uniform vec2 uPoint;
uniform vec3 uValue;
uniform float uRadius;
uniform int uCheckObstacle;
out vec4 outColor;
void main() {
  if (uCheckObstacle == 1 && texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  vec2 p = vUv - uPoint;
  p.x *= uAspect;
  float splat = exp(-dot(p, p) / max(uRadius, 1e-6));
  vec3 base = texture(uTarget, vUv).xyz;
  outColor = vec4(base + splat * uValue, 1.0);
}
`;

const OBSTACLE_BRUSH_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uObstacles;
uniform float uAspect;
uniform vec2 uPoint;
uniform float uRadius;
uniform float uErase;
out vec4 outColor;
void main() {
  float current = texture(uObstacles, vUv).r;
  vec2 p = vUv - uPoint;
  p.x *= uAspect;
  float dist = length(p);
  if (dist < uRadius) {
    current = uErase > 0.5 ? 0.0 : 1.0;
  }
  outColor = vec4(current, 0.0, 0.0, 1.0);
}
`;

const OBSTACLE_PRESET_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform float uAspect;
uniform int uPreset; // 0 = cylinder, 1 = airfoil, 2 = twin_cylinders
out vec4 outColor;

float sdAirfoil(vec2 p) {
  // Rotate -9 degrees angle of attack
  float a = 0.16;
  mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
  p = rot * p;
  float chord = 0.24;
  float x = (p.x + chord * 0.45) / chord;
  if (x < 0.0 || x > 1.0) return 1.0;
  // NACA 0018 symmetric thickness distribution
  float yt = 0.9 * 0.18 * (0.2969 * sqrt(max(x, 0.0)) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);
  return abs(p.y) - yt * chord * 1.4;
}

void main() {
  vec2 p = vUv - vec2(0.28, 0.5);
  p.x *= uAspect;
  float solid = 0.0;
  if (uPreset == 0) {
    if (length(p) < 0.068) solid = 1.0;
  } else if (uPreset == 1) {
    if (sdAirfoil(p) < 0.0) solid = 1.0;
  } else {
    vec2 p1 = vUv - vec2(0.27, 0.38);
    p1.x *= uAspect;
    vec2 p2 = vUv - vec2(0.27, 0.62);
    p2.x *= uAspect;
    if (length(p1) < 0.052 || length(p2) < 0.052) solid = 1.0;
  }
  outColor = vec4(solid, 0.0, 0.0, 1.0);
}
`;

const CURL_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform vec2 uTexelSize;
out vec4 outColor;
void main() {
  float L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).y;
  float R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).y;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).x;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).x;
  float vorticity = 0.5 * ((R - L) - (T - B));
  outColor = vec4(vorticity, 0.0, 0.0, 1.0);
}
`;

const VORTICITY_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uCurlStrength;
uniform float uDt;
out vec4 outColor;
void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  float L = abs(texture(uCurl, vUv - vec2(uTexelSize.x, 0.0)).x);
  float R = abs(texture(uCurl, vUv + vec2(uTexelSize.x, 0.0)).x);
  float T = abs(texture(uCurl, vUv + vec2(0.0, uTexelSize.y)).x);
  float B = abs(texture(uCurl, vUv - vec2(0.0, uTexelSize.y)).x);
  float C = texture(uCurl, vUv).x;

  vec2 force = 0.5 * vec2(T - B, L - R);
  force /= (length(force) + 1e-4);
  force *= uCurlStrength * C;

  vec2 vel = texture(uVelocity, vUv).xy;
  vel += force * uDt;
  outColor = vec4(vel, 0.0, 1.0);
}
`;

const FORCES_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform int uMode; // 0=plasma, 1=ferrofluid, 2=karman, 3=plankton
uniform float uDt;
uniform float uTime;
uniform float uWindJet;
uniform float uAutoDemo;
uniform int uMagnetCount;
uniform vec4 uMagnets[${MAX_MAGNETS}];
out vec4 outColor;

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }

  vec2 vel = texture(uVelocity, vUv).xy;

  // Global Wind / Gravity Jet
  vel.x += uWindJet * 1.4 * uDt;

  // Mode 2: Karman Wind Tunnel Left Boundary Inflow
  if (uMode == 2) {
    if (vUv.x < 0.045 && vUv.y > 0.03 && vUv.y < 0.97) {
      float targetVx = 1.95 + max(uWindJet, -0.8) * 1.1;
      float perturbation = 0.05 * sin(uTime * 3.1 + vUv.y * 24.0);
      vel = mix(vel, vec2(targetVx, perturbation), 0.72);
    }
    // Top/bottom no-slip channel walls
    if (vUv.y < 0.02 || vUv.y > 0.98) {
      vel *= 0.2;
    }
  }

  // Mode 1: Ferrofluid Magnetic Dipole Kelvin & Lorentz Forces
  if (uMode == 1) {
    for (int i = 0; i < ${MAX_MAGNETS}; i++) {
      if (i >= uMagnetCount) break;
      vec4 m = uMagnets[i];
      vec2 d = m.xy - vUv;
      float dist2 = dot(d, d) + 0.0028;
      float dist = sqrt(dist2);
      // Kelvin magnetic gradient pull/push + swirling dipole instability
      float falloff = exp(-dist2 / max(m.w * m.w * 2.2, 0.003));
      vec2 radial = d / dist;
      vec2 tangent = vec2(-radial.y, radial.x);
      float spikeWave = sin(atan(d.y, d.x) * 9.0 + uTime * 3.0) * 0.45;
      vel += (radial * (m.z * 3.6 + spikeWave * abs(m.z)) + tangent * (m.z * 0.95)) * falloff * uDt;
    }
  }

  // Auto-Turbulence Demo stirrers for Plasma / Plankton
  if (uAutoDemo > 0.5 && (uMode == 0 || uMode == 3)) {
    vec2 c1 = vec2(0.5 + 0.26 * cos(uTime * 0.85), 0.5 + 0.22 * sin(uTime * 1.3));
    vec2 c2 = vec2(0.5 + 0.28 * sin(uTime * 0.72 + 2.1), 0.5 + 0.24 * cos(uTime * 1.05));
    vec2 d1 = vUv - c1;
    vec2 d2 = vUv - c2;
    float e1 = exp(-dot(d1, d1) / 0.0022);
    float e2 = exp(-dot(d2, d2) / 0.0022);
    vel += vec2(-sin(uTime * 1.3), cos(uTime * 0.85)) * 2.8 * e1 * uDt;
    vel += vec2(cos(uTime * 1.05), -sin(uTime * 0.72 + 2.1)) * 2.8 * e2 * uDt;
  }

  outColor = vec4(clamp(vel, vec2(-12.0), vec2(12.0)), 0.0, 1.0);
}
`;

const DYE_INJECT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uDye;
uniform sampler2D uObstacles;
uniform int uMode;
uniform float uTime;
uniform float uAutoDemo;
uniform int uMagnetCount;
uniform vec4 uMagnets[${MAX_MAGNETS}];
out vec4 outColor;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  vec3 dye = texture(uDye, vUv).rgb;

  // Karman Wind Tunnel striped inflow ribbons
  if (uMode == 2 && vUv.x < 0.032 && vUv.y > 0.05 && vUv.y < 0.95) {
    float band = sin(vUv.y * 54.0);
    if (band > 0.25) {
      vec3 c1 = vec3(0.05, 0.82, 0.95);
      vec3 c2 = vec3(0.58, 0.35, 0.98);
      vec3 c3 = vec3(0.98, 0.65, 0.12);
      float sel = fract(vUv.y * 4.5);
      vec3 ribbon = sel < 0.33 ? c1 : (sel < 0.66 ? c2 : c3);
      dye = max(dye, ribbon * 0.95);
    }
  }

  // Ferrofluid carrier pool near magnetic poles so spikes stay lush
  if (uMode == 1) {
    for (int i = 0; i < ${MAX_MAGNETS}; i++) {
      if (i >= uMagnetCount) break;
      vec4 m = uMagnets[i];
      vec2 d = vUv - m.xy;
      float pool = exp(-dot(d, d) / max(m.w * m.w * 0.65, 0.003));
      dye = max(dye, vec3(0.75, 0.82, 0.95) * pool * 0.65);
    }
  }

  // Auto-Demo Plasma / Plankton dye injection
  if (uAutoDemo > 0.5 && (uMode == 0 || uMode == 3)) {
    vec2 c1 = vec2(0.5 + 0.26 * cos(uTime * 0.85), 0.5 + 0.22 * sin(uTime * 1.3));
    vec2 c2 = vec2(0.5 + 0.28 * sin(uTime * 0.72 + 2.1), 0.5 + 0.24 * cos(uTime * 1.05));
    float s1 = exp(-dot(vUv - c1, vUv - c1) / 0.00055);
    float s2 = exp(-dot(vUv - c2, vUv - c2) / 0.00055);
    if (uMode == 0) {
      vec3 col1 = hsv2rgb(vec3(fract(uTime * 0.08), 0.88, 1.0));
      vec3 col2 = hsv2rgb(vec3(fract(uTime * 0.08 + 0.5), 0.88, 1.0));
      dye += col1 * s1 * 0.45 + col2 * s2 * 0.45;
    } else {
      dye += vec3(0.02, 0.35, 0.55) * (s1 + s2) * 0.28;
    }
  }

  outColor = vec4(min(dye, vec3(3.5)), 1.0);
}
`;

const ADVECT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uDt;
uniform float uDissipation;
uniform int uMode;
out vec4 outColor;
void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  vec2 vel = texture(uVelocity, vUv).xy;
  vec2 coord = vUv - uDt * vel * uTexelSize * 85.0;
  if (uMode == 2) {
    coord.x = max(coord.x, 0.0);
  }
  vec4 result = uDissipation * texture(uSource, clamp(coord, vec2(0.001), vec2(0.999)));
  outColor = result;
}
`;

const DIVERGENCE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
out vec4 outColor;
void main() {
  float L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).x;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).y;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).y;

  vec2 C = texture(uVelocity, vUv).xy;
  if (texture(uObstacles, vUv - vec2(uTexelSize.x, 0.0)).r > 0.5) L = -C.x;
  if (texture(uObstacles, vUv + vec2(uTexelSize.x, 0.0)).r > 0.5) R = -C.x;
  if (texture(uObstacles, vUv + vec2(0.0, uTexelSize.y)).r > 0.5) T = -C.y;
  if (texture(uObstacles, vUv - vec2(0.0, uTexelSize.y)).r > 0.5) B = -C.y;

  float div = 0.5 * (R - L + T - B);
  outColor = vec4(div, 0.0, 0.0, 1.0);
}
`;

const JACOBI_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
out vec4 outColor;
void main() {
  float C = texture(uPressure, vUv).x;
  float L = texture(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;

  // Neumann boundary condition on obstacle faces
  if (texture(uObstacles, vUv - vec2(uTexelSize.x, 0.0)).r > 0.5) L = C;
  if (texture(uObstacles, vUv + vec2(uTexelSize.x, 0.0)).r > 0.5) R = C;
  if (texture(uObstacles, vUv + vec2(0.0, uTexelSize.y)).r > 0.5) T = C;
  if (texture(uObstacles, vUv - vec2(0.0, uTexelSize.y)).r > 0.5) B = C;

  float div = texture(uDivergence, vUv).x;
  float pressure = (L + R + B + T - div) * 0.25;
  outColor = vec4(pressure, 0.0, 0.0, 1.0);
}
`;

const VISCOUS_JACOBI_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
uniform float uAlpha;
uniform float uRbeta;
out vec4 outColor;
void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  vec2 C = texture(uVelocity, vUv).xy;
  vec2 L = texture(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).xy;
  vec2 R = texture(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).xy;
  vec2 T = texture(uVelocity, vUv + vec2(0.0, uTexelSize.y)).xy;
  vec2 B = texture(uVelocity, vUv - vec2(0.0, uTexelSize.y)).xy;
  vec2 diffused = (L + R + B + T + uAlpha * C) * uRbeta;
  outColor = vec4(diffused, 0.0, 1.0);
}
`;

const GRADIENT_SUB_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform sampler2D uObstacles;
uniform vec2 uTexelSize;
out vec4 outColor;
void main() {
  if (texture(uObstacles, vUv).r > 0.5) {
    outColor = vec4(0.0);
    return;
  }
  float C = texture(uPressure, vUv).x;
  float L = texture(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;

  if (texture(uObstacles, vUv - vec2(uTexelSize.x, 0.0)).r > 0.5) L = C;
  if (texture(uObstacles, vUv + vec2(uTexelSize.x, 0.0)).r > 0.5) R = C;
  if (texture(uObstacles, vUv + vec2(0.0, uTexelSize.y)).r > 0.5) T = C;
  if (texture(uObstacles, vUv - vec2(0.0, uTexelSize.y)).r > 0.5) B = C;

  vec2 vel = texture(uVelocity, vUv).xy;
  vel -= 0.5 * vec2(R - L, T - B);
  outColor = vec4(vel, 0.0, 1.0);
}
`;

const PARTICLE_ADVECT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uParticles;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform sampler2D uObstacles;
uniform vec2 uSimTexel;
uniform float uDt;
uniform float uTime;
uniform int uMode;
uniform float uBiolumSens;
out vec4 outColor;

float hash12(vec2 p) {
  vec3 p3  = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec4 p = texelFetch(uParticles, ivec2(gl_FragCoord.xy), 0);
  vec2 pos = p.xy;
  float energy = p.z;
  float life = p.w;

  // RK2 Midpoint Advection
  vec2 v1 = texture(uVelocity, pos).xy;
  vec2 scale = uSimTexel * 85.0;
  vec2 mid = clamp(pos + 0.5 * uDt * v1 * scale, vec2(0.002), vec2(0.998));
  vec2 v2 = texture(uVelocity, mid).xy;
  pos += uDt * v2 * scale;

  // Compute local Shear Rate + Curl for bioluminescent excitation
  vec2 vL = texture(uVelocity, pos - vec2(uSimTexel.x, 0.0)).xy;
  vec2 vR = texture(uVelocity, pos + vec2(uSimTexel.x, 0.0)).xy;
  vec2 vB = texture(uVelocity, pos - vec2(0.0, uSimTexel.y)).xy;
  vec2 vT = texture(uVelocity, pos + vec2(0.0, uSimTexel.y)).xy;
  float shear = length(vR - vL) + length(vT - vB);
  float curlVal = abs(texture(uCurl, pos).x);
  float speed = length(v2);

  float targetEnergy = clamp((shear * 0.9 + curlVal * 0.65 + speed * 0.25) * uBiolumSens, 0.0, 2.0);
  energy = mix(energy * 0.94, targetEnergy, targetEnergy > energy ? 0.42 : 0.08);

  life -= uDt * 0.16;

  bool hitObstacle = texture(uObstacles, pos).r > 0.5;
  bool outOfBounds = pos.x < 0.002 || pos.x > 0.998 || pos.y < 0.002 || pos.y > 0.998;

  if (life <= 0.0 || outOfBounds || hitObstacle) {
    float r1 = hash12(vUv * 123.4 + uTime * 1.13);
    float r2 = hash12(vUv * 456.7 + uTime * 2.71);
    if (uMode == 2) {
      // Wind tunnel respawns cleanly at inlet
      pos = vec2(0.008 + r1 * 0.03, 0.04 + r2 * 0.92);
    } else {
      pos = vec2(0.02 + r1 * 0.96, 0.02 + r2 * 0.96);
    }
    energy = 0.08;
    life = 0.5 + 0.5 * hash12(vUv * 789.1 + uTime);
  }

  outColor = vec4(pos, energy, life);
}
`;

const TELEMETRY_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
out vec4 outColor;
void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  float speed = length(vel);
  float curl = abs(texture(uCurl, vUv).x);
  float ke = 0.5 * dot(vel, vel);
  outColor = vec4(
    clamp(speed / 6.0, 0.0, 1.0),
    clamp(curl / 4.0, 0.0, 1.0),
    clamp(ke / 12.0, 0.0, 1.0),
    1.0
  );
}
`;

const DISPLAY_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uDye;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform sampler2D uObstacles;
uniform vec2 uDyeTexel;
uniform vec2 uSimTexel;
uniform float uAspect;
uniform float uTime;
uniform int uMode;
uniform float uMagneticGain;
uniform int uMagnetCount;
uniform vec4 uMagnets[${MAX_MAGNETS}];
out vec4 outColor;

float ferroHeight(vec2 uv) {
  vec3 d = texture(uDye, uv).rgb;
  float density = clamp(dot(d, vec3(0.333)), 0.0, 1.4);
  float field = 0.0;
  float spikePattern = 0.0;

  for (int i = 0; i < ${MAX_MAGNETS}; i++) {
    if (i >= uMagnetCount) break;
    vec4 m = uMagnets[i];
    vec2 diff = uv - m.xy;
    diff.x *= uAspect;
    float dist = length(diff);
    float infl = exp(-(dist * dist) / max(m.w * m.w * 0.9, 0.0025));
    field += abs(m.z) * infl;

    // Rosensweig hexagonal normal-field instability spikes
    float angle = atan(diff.y, diff.x);
    float freq = 115.0 + 35.0 * abs(m.z);
    float ring = cos(dist * freq - uTime * 2.2 * sign(m.z));
    float petals = cos(angle * 10.0 + sin(dist * 40.0 + uTime));
    float hex = 0.5 + 0.5 * ring * (0.65 + 0.35 * petals);
    spikePattern += pow(hex, 2.2) * infl * abs(m.z);
  }

  float curlMod = abs(texture(uCurl, uv).x) * 0.14;
  return density * 0.45 + field * 0.5 + spikePattern * uMagneticGain * (0.35 + density * 0.9) + curlMod;
}

void main() {
  float obs = texture(uObstacles, vUv).r;
  vec3 dye = texture(uDye, vUv).rgb;
  vec2 vel = texture(uVelocity, vUv).xy;
  float speed = length(vel);
  float curl = texture(uCurl, vUv).x;

  vec3 col = vec3(0.015, 0.025, 0.052);

  if (uMode == 0) {
    // Mode 0: Neon Plasma Advection
    float glow = dot(dye, vec3(0.333));
    vec3 plasmaBoost = vec3(0.04, 0.18, 0.32) * smoothstep(0.0, 2.2, speed) * 0.45;
    col += dye * 1.15 + plasmaBoost + vec3(0.18, 0.06, 0.35) * smoothstep(0.5, 2.5, abs(curl)) * glow * 0.4;
  } else if (uMode == 1) {
    // Mode 1: Ferrofluid Magnetic Spikes (PBR Liquid Obsidian + Chrome Specular)
    vec2 eps = uDyeTexel * 2.2;
    float hL = ferroHeight(vUv - vec2(eps.x, 0.0));
    float hR = ferroHeight(vUv + vec2(eps.x, 0.0));
    float hB = ferroHeight(vUv - vec2(0.0, eps.y));
    float hT = ferroHeight(vUv + vec2(0.0, eps.y));
    float hC = ferroHeight(vUv);

    vec3 normal = normalize(vec3((hL - hR) * 3.4, (hB - hT) * 3.4, 0.18));
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 light1 = normalize(vec3(-0.6, 0.7, 0.85));
    vec3 light2 = normalize(vec3(0.75, -0.45, 0.6));

    vec3 half1 = normalize(light1 + viewDir);
    vec3 half2 = normalize(light2 + viewDir);
    float spec1 = pow(max(dot(normal, half1), 0.0), 84.0);
    float spec2 = pow(max(dot(normal, half2), 0.0), 48.0);
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.2);

    float metalMask = smoothstep(0.06, 0.42, hC);
    vec3 obsidian = vec3(0.015, 0.02, 0.032);
    vec3 silverRim = vec3(0.78, 0.9, 1.0) * fresnel * 0.85;
    vec3 ferroCol = obsidian
      + vec3(0.9, 0.96, 1.0) * spec1 * 1.65
      + vec3(0.35, 0.72, 0.98) * spec2 * 0.75
      + silverRim;

    // Subtle metallic studio floor
    vec3 studioBg = vec3(0.035, 0.045, 0.075) + vec3(0.02, 0.03, 0.05) * (1.0 - length(vUv - 0.5));
    col = mix(studioBg, ferroCol, metalMask);

    // Render magnetic dipole indicators
    for (int i = 0; i < ${MAX_MAGNETS}; i++) {
      if (i >= uMagnetCount) break;
      vec4 m = uMagnets[i];
      vec2 d = vUv - m.xy;
      d.x *= uAspect;
      float dist = length(d);
      float ring = smoothstep(0.004, 0.0, abs(dist - 0.024));
      float core = smoothstep(0.01, 0.0, dist);
      vec3 poleCol = m.z >= 0.0 ? vec3(0.04, 0.85, 0.98) : vec3(0.95, 0.45, 0.12);
      col += poleCol * (ring * 0.9 + core * 1.1);
    }
  } else if (uMode == 2) {
    // Mode 2: Karman Vortex Street & Wind Tunnel
    float vortPos = clamp(curl * 0.35, 0.0, 1.0);
    float vortNeg = clamp(-curl * 0.35, 0.0, 1.0);
    vec3 vortTint = vec3(0.02, 0.65, 0.95) * vortPos + vec3(0.96, 0.36, 0.45) * vortNeg;
    col += dye * 0.95 + vortTint * 0.55;
    // Subtle wind-tunnel grid lines
    float grid = step(0.985, fract(vUv.y * 12.0)) * 0.05;
    col += vec3(0.08, 0.2, 0.35) * grid;
  } else {
    // Mode 3: Bioluminescent Plankton Abyss
    vec3 abyss = vec3(0.006, 0.015, 0.038);
    float currentGlow = smoothstep(0.05, 1.8, abs(curl) + speed * 0.35);
    col = abyss + dye * 0.6 + vec3(0.01, 0.14, 0.28) * currentGlow;
  }

  // Render crisp solid obstacles
  if (obs > 0.5) {
    float edge = 0.0;
    edge += 1.0 - texture(uObstacles, vUv + vec2(uDyeTexel.x * 2.0, 0.0)).r;
    edge += 1.0 - texture(uObstacles, vUv - vec2(uDyeTexel.x * 2.0, 0.0)).r;
    edge += 1.0 - texture(uObstacles, vUv + vec2(0.0, uDyeTexel.y * 2.0)).r;
    edge += 1.0 - texture(uObstacles, vUv - vec2(0.0, uDyeTexel.y * 2.0)).r;
    vec3 solidBody = vec3(0.07, 0.1, 0.17);
    vec3 solidRim = vec3(0.22, 0.85, 0.98) * clamp(edge, 0.0, 1.0) * 0.85;
    col = solidBody + solidRim;
  }

  // Tone mapping
  col = col / (1.0 + col * 0.25);
  outColor = vec4(col, 1.0);
}
`;

const PARTICLE_VERT = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uParticles;
uniform float uPointScale;
uniform int uMode;
out float vEnergy;
out float vLife;

void main() {
  int id = gl_VertexID;
  ivec2 coord = ivec2(id % ${PARTICLE_TEX_W}, id / ${PARTICLE_TEX_W});
  vec4 p = texelFetch(uParticles, coord, 0);
  vEnergy = p.z;
  vLife = p.w;

  gl_Position = vec4(p.xy * 2.0 - 1.0, 0.0, 1.0);
  float baseSize = (uMode == 3) ? 2.6 : 1.9;
  gl_PointSize = (baseSize + clamp(p.z, 0.0, 1.5) * 3.2) * uPointScale;
}
`;

const PARTICLE_FRAG = `#version 300 es
precision highp float;
precision highp int;
in float vEnergy;
in float vLife;
uniform int uMode;
out vec4 outColor;

void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float alpha = exp(-r2 * 14.0) * smoothstep(0.0, 0.15, vLife);

  vec3 col;
  if (uMode == 3) {
    // Bioluminescent Plankton: Deep pelagic blue -> Electric Cyan -> Bright Aqua Spark
    float e = clamp(vEnergy, 0.0, 1.4);
    vec3 calm = vec3(0.03, 0.18, 0.55);
    vec3 mid = vec3(0.02, 0.72, 0.92);
    vec3 peak = vec3(0.45, 0.98, 1.0);
    col = e < 0.5 ? mix(calm, mid, e * 2.0) : mix(mid, peak, (e - 0.5) * 1.1);
    alpha *= (0.28 + 0.72 * clamp(e, 0.0, 1.0));
  } else if (uMode == 1) {
    // Ferrofluid subtle silver-cyan flux tracers
    col = vec3(0.45, 0.85, 1.0);
    alpha *= 0.22 * clamp(vEnergy + 0.2, 0.0, 1.0);
  } else if (uMode == 2) {
    // Wind tunnel streamlines
    col = mix(vec3(0.25, 0.85, 0.98), vec3(0.98, 0.75, 0.25), clamp(vEnergy, 0.0, 1.0));
    alpha *= 0.45;
  } else {
    // Neon Plasma tracer sparks
    col = mix(vec3(0.35, 0.75, 1.0), vec3(0.95, 0.45, 1.0), clamp(vEnergy * 0.7, 0.0, 1.0));
    alpha *= 0.38;
  }

  outColor = vec4(col * alpha, alpha);
}
`;

export class WebGLFluidEngine implements FluidEngine {
  readonly backendLabel = 'WebGL2 Ping-Pong FBO';
  readonly particleCount = PARTICLE_COUNT;

  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private simW = 256;
  private simH = 144;
  private dyeW = 1024;
  private dyeH = 576;

  private velocity!: DoubleFBO;
  private dye!: DoubleFBO;
  private pressure!: DoubleFBO;
  private obstacles!: DoubleFBO;
  private particles!: DoubleFBO;
  private divergence!: FBO;
  private curl!: FBO;
  private telemetryFBO!: FBO;

  private programs: Record<string, WebGLProgram> = {};
  private uniforms: Record<string, Record<string, WebGLUniformLocation | null>> = {};
  private emptyVAO!: WebGLVertexArrayObject;

  private time = 0;
  private frameCounter = 0;
  private fps = 60;
  private lastFpsTime = performance.now();
  private fpsFrames = 0;
  private telemetryBuf = new Uint8Array(32 * 32 * 4);
  private telemetryPBOs: [WebGLBuffer, WebGLBuffer] = [null!, null!];
  private telemetryPboIdx = 0;
  private telemetrySync: WebGLSync | null = null;
  private telemetryPendingIdx = 0;
  private lastTelemetry: TelemetrySnapshot = {
    fps: 60,
    reynoldsNumber: 1200,
    peakVorticity: 0,
    kineticEnergy: 0,
    turbulenceRatio: 0.15,
    particleCount: PARTICLE_COUNT,
    backendLabel: this.backendLabel,
  };

  private useFloat32Particles = false;
  private floatTexType: number;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) {
      throw new Error('WebGL2 unavailable');
    }
    this.gl = gl;

    const extFloat = gl.getExtension('EXT_color_buffer_float');
    const extHalf = gl.getExtension('EXT_color_buffer_half_float');
    gl.getExtension('OES_texture_float_linear');
    if (!extFloat && !extHalf) {
      throw new Error('Floating-point FBO rendering unsupported');
    }

    this.useFloat32Particles = Boolean(extFloat);
    this.floatTexType = gl.HALF_FLOAT;

    this.emptyVAO = gl.createVertexArray()!;
    for (let i = 0; i < 2; i++) {
      const pbo = gl.createBuffer()!;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, 32 * 32 * 4, gl.DYNAMIC_READ);
      this.telemetryPBOs[i] = pbo;
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.compileAllPrograms();
    this.initDimensions(canvas.width || 1280, canvas.height || 720);
    this.initFramebuffers();
    this.seedParticles();
  }

  private compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${info}`);
    }
    return shader;
  }

  private createProgram(name: string, vertSrc: string, fragSrc: string, uniformNames: string[]): void {
    const gl = this.gl;
    const vs = this.compileShader(gl.VERTEX_SHADER, vertSrc);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Program link error (${name}): ${gl.getProgramInfoLog(prog)}`);
    }
    this.programs[name] = prog;
    const locs: Record<string, WebGLUniformLocation | null> = {};
    for (const u of uniformNames) {
      locs[u] = gl.getUniformLocation(prog, u);
    }
    this.uniforms[name] = locs;
  }

  private compileAllPrograms(): void {
    const gl = this.gl;
    const magnetUniforms = Array.from({ length: MAX_MAGNETS }, (_, i) => `uMagnets[${i}]`);

    this.createProgram('clear', BASE_VERTEX, CLEAR_FRAG, ['uTexture', 'uValue']);
    this.createProgram('splat', BASE_VERTEX, SPLAT_FRAG, [
      'uTarget',
      'uObstacles',
      'uAspect',
      'uPoint',
      'uValue',
      'uRadius',
      'uCheckObstacle',
    ]);
    this.createProgram('obstacleBrush', BASE_VERTEX, OBSTACLE_BRUSH_FRAG, [
      'uObstacles',
      'uAspect',
      'uPoint',
      'uRadius',
      'uErase',
    ]);
    this.createProgram('obstaclePreset', BASE_VERTEX, OBSTACLE_PRESET_FRAG, ['uAspect', 'uPreset']);
    this.createProgram('curl', BASE_VERTEX, CURL_FRAG, ['uVelocity', 'uTexelSize']);
    this.createProgram('vorticity', BASE_VERTEX, VORTICITY_FRAG, [
      'uVelocity',
      'uCurl',
      'uObstacles',
      'uTexelSize',
      'uCurlStrength',
      'uDt',
    ]);
    this.createProgram('forces', BASE_VERTEX, FORCES_FRAG, [
      'uVelocity',
      'uObstacles',
      'uMode',
      'uDt',
      'uTime',
      'uWindJet',
      'uAutoDemo',
      'uMagnetCount',
      ...magnetUniforms,
    ]);
    this.createProgram('dyeInject', BASE_VERTEX, DYE_INJECT_FRAG, [
      'uDye',
      'uObstacles',
      'uMode',
      'uTime',
      'uAutoDemo',
      'uMagnetCount',
      ...magnetUniforms,
    ]);
    this.createProgram('advect', BASE_VERTEX, ADVECT_FRAG, [
      'uVelocity',
      'uSource',
      'uObstacles',
      'uTexelSize',
      'uDt',
      'uDissipation',
      'uMode',
    ]);
    this.createProgram('divergence', BASE_VERTEX, DIVERGENCE_FRAG, [
      'uVelocity',
      'uObstacles',
      'uTexelSize',
    ]);
    this.createProgram('jacobi', BASE_VERTEX, JACOBI_FRAG, [
      'uPressure',
      'uDivergence',
      'uObstacles',
      'uTexelSize',
    ]);
    this.createProgram('viscous', BASE_VERTEX, VISCOUS_JACOBI_FRAG, [
      'uVelocity',
      'uObstacles',
      'uTexelSize',
      'uAlpha',
      'uRbeta',
    ]);
    this.createProgram('gradSub', BASE_VERTEX, GRADIENT_SUB_FRAG, [
      'uPressure',
      'uVelocity',
      'uObstacles',
      'uTexelSize',
    ]);
    this.createProgram('particleAdvect', BASE_VERTEX, PARTICLE_ADVECT_FRAG, [
      'uParticles',
      'uVelocity',
      'uCurl',
      'uObstacles',
      'uSimTexel',
      'uDt',
      'uTime',
      'uMode',
      'uBiolumSens',
    ]);
    this.createProgram('telemetry', BASE_VERTEX, TELEMETRY_FRAG, ['uVelocity', 'uCurl']);
    this.createProgram('display', BASE_VERTEX, DISPLAY_FRAG, [
      'uDye',
      'uVelocity',
      'uCurl',
      'uObstacles',
      'uDyeTexel',
      'uSimTexel',
      'uAspect',
      'uTime',
      'uMode',
      'uMagneticGain',
      'uMagnetCount',
      ...magnetUniforms,
    ]);
    this.createProgram('particleRender', PARTICLE_VERT, PARTICLE_FRAG, [
      'uParticles',
      'uPointScale',
      'uMode',
    ]);
    void gl;
  }

  private initDimensions(w: number, h: number): void {
    const aspect = Math.max(0.5, Math.min(2.5, w / Math.max(h, 1)));
    if (aspect >= 1) {
      this.simW = 256;
      this.simH = Math.max(128, Math.round(256 / aspect));
      this.dyeW = 1024;
      this.dyeH = Math.max(512, Math.round(1024 / aspect));
    } else {
      this.simH = 256;
      this.simW = Math.max(128, Math.round(256 * aspect));
      this.dyeH = 1024;
      this.dyeW = Math.max(512, Math.round(1024 * aspect));
    }
  }

  private createFBO(
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    filter: number,
    initialData: ArrayBufferView | null = null
  ): FBO {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, initialData);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Incomplete FBO status: ${status}`);
    }
    gl.viewport(0, 0, w, h);
    if (!initialData) {
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    return {
      texture,
      fbo,
      width: w,
      height: h,
      attach(unit: number): number {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return unit;
      },
    };
  }

  private createDoubleFBO(
    w: number,
    h: number,
    internalFormat: number,
    format: number,
    type: number,
    filter: number
  ): DoubleFBO {
    let fbo1 = this.createFBO(w, h, internalFormat, format, type, filter);
    let fbo2 = this.createFBO(w, h, internalFormat, format, type, filter);
    return {
      width: w,
      height: h,
      get read() {
        return fbo1;
      },
      get write() {
        return fbo2;
      },
      swap() {
        const tmp = fbo1;
        fbo1 = fbo2;
        fbo2 = tmp;
      },
    };
  }

  private initFramebuffers(): void {
    const gl = this.gl;
    this.velocity = this.createDoubleFBO(this.simW, this.simH, gl.RGBA16F, gl.RGBA, this.floatTexType, gl.LINEAR);
    this.pressure = this.createDoubleFBO(this.simW, this.simH, gl.RGBA16F, gl.RGBA, this.floatTexType, gl.NEAREST);
    this.divergence = this.createFBO(this.simW, this.simH, gl.RGBA16F, gl.RGBA, this.floatTexType, gl.NEAREST);
    this.curl = this.createFBO(this.simW, this.simH, gl.RGBA16F, gl.RGBA, this.floatTexType, gl.LINEAR);
    this.dye = this.createDoubleFBO(this.dyeW, this.dyeH, gl.RGBA16F, gl.RGBA, this.floatTexType, gl.LINEAR);
    this.obstacles = this.createDoubleFBO(this.dyeW, this.dyeH, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);
    this.telemetryFBO = this.createFBO(32, 32, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);

    if (this.useFloat32Particles) {
      this.particles = this.createDoubleFBO(
        PARTICLE_TEX_W,
        PARTICLE_TEX_H,
        gl.RGBA32F,
        gl.RGBA,
        gl.FLOAT,
        gl.NEAREST
      );
    } else {
      this.particles = this.createDoubleFBO(
        PARTICLE_TEX_W,
        PARTICLE_TEX_H,
        gl.RGBA16F,
        gl.RGBA,
        this.floatTexType,
        gl.NEAREST
      );
    }
  }

  private seedParticles(): void {
    const gl = this.gl;
    const count = PARTICLE_COUNT;
    if (this.useFloat32Particles) {
      const data = new Float32Array(count * 4);
      for (let i = 0; i < count; i++) {
        data[i * 4 + 0] = Math.random();
        data[i * 4 + 1] = Math.random();
        data[i * 4 + 2] = 0.08 + Math.random() * 0.15;
        data[i * 4 + 3] = 0.2 + Math.random() * 0.8;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.particles.read.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, PARTICLE_TEX_W, PARTICLE_TEX_H, gl.RGBA, gl.FLOAT, data);
    } else {
      // Fallback: write via shader by setting life = 0, so first advection pass scatters randomly
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.particles.read.fbo);
      gl.clearColor(0.5, 0.5, 0.1, 0.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
  }

  private blit(target: FBO | null, width?: number, height?: number): void {
    const gl = this.gl;
    if (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.width, target.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width ?? this.canvas.width, height ?? this.canvas.height);
    }
    gl.bindVertexArray(this.emptyVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private modeToInt(mode: FluidMode): number {
    if (mode === 'plasma') return 0;
    if (mode === 'ferrofluid') return 1;
    if (mode === 'karman') return 2;
    return 3;
  }

  private uploadMagnets(progName: string, magnets: MagnetPole[]): void {
    const gl = this.gl;
    const u = this.uniforms[progName];
    const count = Math.min(magnets.length, MAX_MAGNETS);
    gl.uniform1i(u.uMagnetCount, count);
    for (let i = 0; i < count; i++) {
      const m = magnets[i];
      gl.uniform4f(u[`uMagnets[${i}]`], m.x, m.y, m.strength, m.radius);
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
    const gl = this.gl;
    const aspect = this.canvas.width / Math.max(this.canvas.height, 1);

    gl.useProgram(this.programs.splat);
    const u = this.uniforms.splat;
    gl.uniform1f(u.uAspect, aspect);
    gl.uniform2f(u.uPoint, nx, ny);
    gl.uniform1f(u.uRadius, radius);
    gl.uniform1i(u.uObstacles, this.obstacles.read.attach(1));
    gl.uniform1i(u.uCheckObstacle, 1);

    // Splat velocity
    gl.uniform1i(u.uTarget, this.velocity.read.attach(0));
    gl.uniform3f(u.uValue, dx, dy, 0);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // Splat dye
    gl.uniform1i(u.uTarget, this.dye.read.attach(0));
    gl.uniform3f(u.uValue, color[0], color[1], color[2]);
    this.blit(this.dye.write);
    this.dye.swap();
  }

  drawObstacle(nx: number, ny: number, radius: number, erase: boolean): void {
    const gl = this.gl;
    const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
    gl.useProgram(this.programs.obstacleBrush);
    const u = this.uniforms.obstacleBrush;
    gl.uniform1i(u.uObstacles, this.obstacles.read.attach(0));
    gl.uniform1f(u.uAspect, aspect);
    gl.uniform2f(u.uPoint, nx, ny);
    gl.uniform1f(u.uRadius, radius);
    gl.uniform1f(u.uErase, erase ? 1.0 : 0.0);
    this.blit(this.obstacles.write);
    this.obstacles.swap();
  }

  clearObstacles(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.obstacles.read.fbo);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.obstacles.write.fbo);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  applyKarmanPreset(preset: KarmanPreset): void {
    const gl = this.gl;
    const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
    const idx = preset === 'cylinder' ? 0 : preset === 'airfoil' ? 1 : 2;
    gl.useProgram(this.programs.obstaclePreset);
    const u = this.uniforms.obstaclePreset;
    gl.uniform1f(u.uAspect, aspect);
    gl.uniform1i(u.uPreset, idx);
    this.blit(this.obstacles.write);
    this.obstacles.swap();
  }

  burstPulse(nx: number, ny: number, mode: FluidMode): void {
    const count = 12;
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const vx = Math.cos(angle) * 5.2;
      const vy = Math.sin(angle) * 5.2;
      const ox = nx + Math.cos(angle) * 0.025;
      const oy = ny + Math.sin(angle) * 0.025;
      const col: [number, number, number] =
        mode === 'ferrofluid'
          ? [0.7, 0.85, 1.0]
          : mode === 'plankton'
          ? [0.05, 0.9, 1.0]
          : i % 2 === 0
          ? [0.1, 0.9, 1.0]
          : [0.85, 0.3, 1.0];
      this.splat(ox, oy, vx, vy, col, 0.0018);
    }
  }

  resetField(mode: FluidMode): void {
    const gl = this.gl;
    for (const fbo of [
      this.velocity.read,
      this.velocity.write,
      this.dye.read,
      this.dye.write,
      this.pressure.read,
      this.pressure.write,
    ]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fbo);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    if (mode !== 'karman') {
      this.clearObstacles();
    }
    this.seedParticles();
  }

  step(dt: number, params: FluidParams, magnets: MagnetPole[]): void {
    const gl = this.gl;
    const clampedDt = Math.min(dt, 0.025);
    this.time += clampedDt;
    this.frameCounter++;

    const modeInt = this.modeToInt(params.mode);
    const simTexelX = 1.0 / this.simW;
    const simTexelY = 1.0 / this.simH;

    // 1. Curl computation
    gl.useProgram(this.programs.curl);
    gl.uniform1i(this.uniforms.curl.uVelocity, this.velocity.read.attach(0));
    gl.uniform2f(this.uniforms.curl.uTexelSize, simTexelX, simTexelY);
    this.blit(this.curl);

    // 2. Vorticity Confinement
    gl.useProgram(this.programs.vorticity);
    gl.uniform1i(this.uniforms.vorticity.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(this.uniforms.vorticity.uCurl, this.curl.attach(1));
    gl.uniform1i(this.uniforms.vorticity.uObstacles, this.obstacles.read.attach(2));
    gl.uniform2f(this.uniforms.vorticity.uTexelSize, simTexelX, simTexelY);
    gl.uniform1f(this.uniforms.vorticity.uCurlStrength, params.vorticity);
    gl.uniform1f(this.uniforms.vorticity.uDt, clampedDt);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // 3. External Forces (Wind Tunnel, Magnetic Kelvin Forces, Auto-Turbulence)
    gl.useProgram(this.programs.forces);
    const uf = this.uniforms.forces;
    gl.uniform1i(uf.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(uf.uObstacles, this.obstacles.read.attach(1));
    gl.uniform1i(uf.uMode, modeInt);
    gl.uniform1f(uf.uDt, clampedDt);
    gl.uniform1f(uf.uTime, this.time);
    gl.uniform1f(uf.uWindJet, params.windJet);
    gl.uniform1f(uf.uAutoDemo, params.autoDemo ? 1.0 : 0.0);
    this.uploadMagnets('forces', magnets);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // 4. Viscous Diffusion (when viscosity > 0)
    if (params.viscosity > 0.00001) {
      const viscRate = params.viscosity * 1800.0;
      const alpha = 1.0 / Math.max(viscRate * clampedDt, 1e-4);
      const rBeta = 1.0 / (4.0 + alpha);
      gl.useProgram(this.programs.viscous);
      gl.uniform1i(this.uniforms.viscous.uObstacles, this.obstacles.read.attach(1));
      gl.uniform2f(this.uniforms.viscous.uTexelSize, simTexelX, simTexelY);
      gl.uniform1f(this.uniforms.viscous.uAlpha, alpha);
      gl.uniform1f(this.uniforms.viscous.uRbeta, rBeta);
      for (let i = 0; i < 6; i++) {
        gl.uniform1i(this.uniforms.viscous.uVelocity, this.velocity.read.attach(0));
        this.blit(this.velocity.write);
        this.velocity.swap();
      }
    }

    // 5. Divergence
    gl.useProgram(this.programs.divergence);
    gl.uniform1i(this.uniforms.divergence.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(this.uniforms.divergence.uObstacles, this.obstacles.read.attach(1));
    gl.uniform2f(this.uniforms.divergence.uTexelSize, simTexelX, simTexelY);
    this.blit(this.divergence);

    // 6. Pressure Decay + Jacobi Projection
    gl.useProgram(this.programs.clear);
    gl.uniform1i(this.uniforms.clear.uTexture, this.pressure.read.attach(0));
    gl.uniform1f(this.uniforms.clear.uValue, 0.8);
    this.blit(this.pressure.write);
    this.pressure.swap();

    gl.useProgram(this.programs.jacobi);
    gl.uniform1i(this.uniforms.jacobi.uDivergence, this.divergence.attach(1));
    gl.uniform1i(this.uniforms.jacobi.uObstacles, this.obstacles.read.attach(2));
    gl.uniform2f(this.uniforms.jacobi.uTexelSize, simTexelX, simTexelY);
    for (let i = 0; i < params.jacobiIterations; i++) {
      gl.uniform1i(this.uniforms.jacobi.uPressure, this.pressure.read.attach(0));
      this.blit(this.pressure.write);
      this.pressure.swap();
    }

    // 7. Gradient Subtraction (Helmholtz-Hodge projection)
    gl.useProgram(this.programs.gradSub);
    gl.uniform1i(this.uniforms.gradSub.uPressure, this.pressure.read.attach(0));
    gl.uniform1i(this.uniforms.gradSub.uVelocity, this.velocity.read.attach(1));
    gl.uniform1i(this.uniforms.gradSub.uObstacles, this.obstacles.read.attach(2));
    gl.uniform2f(this.uniforms.gradSub.uTexelSize, simTexelX, simTexelY);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // 8. Self-Advect Velocity
    gl.useProgram(this.programs.advect);
    gl.uniform1i(this.uniforms.advect.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(this.uniforms.advect.uSource, this.velocity.read.attach(0));
    gl.uniform1i(this.uniforms.advect.uObstacles, this.obstacles.read.attach(1));
    gl.uniform2f(this.uniforms.advect.uTexelSize, simTexelX, simTexelY);
    gl.uniform1f(this.uniforms.advect.uDt, clampedDt);
    gl.uniform1f(this.uniforms.advect.uDissipation, 0.996);
    gl.uniform1i(this.uniforms.advect.uMode, modeInt);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // 9. Inject Mode Dye & Advect Dye
    gl.useProgram(this.programs.dyeInject);
    const udi = this.uniforms.dyeInject;
    gl.uniform1i(udi.uDye, this.dye.read.attach(0));
    gl.uniform1i(udi.uObstacles, this.obstacles.read.attach(1));
    gl.uniform1i(udi.uMode, modeInt);
    gl.uniform1f(udi.uTime, this.time);
    gl.uniform1f(udi.uAutoDemo, params.autoDemo ? 1.0 : 0.0);
    this.uploadMagnets('dyeInject', magnets);
    this.blit(this.dye.write);
    this.dye.swap();

    gl.useProgram(this.programs.advect);
    gl.uniform1i(this.uniforms.advect.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(this.uniforms.advect.uSource, this.dye.read.attach(1));
    gl.uniform1i(this.uniforms.advect.uObstacles, this.obstacles.read.attach(2));
    gl.uniform2f(this.uniforms.advect.uTexelSize, simTexelX, simTexelY);
    gl.uniform1f(this.uniforms.advect.uDt, clampedDt);
    gl.uniform1f(this.uniforms.advect.uDissipation, params.dissipation);
    gl.uniform1i(this.uniforms.advect.uMode, modeInt);
    this.blit(this.dye.write);
    this.dye.swap();

    // 10. GPU Lagrangian Particle Advection (8,192 particles)
    gl.useProgram(this.programs.particleAdvect);
    const upa = this.uniforms.particleAdvect;
    gl.uniform1i(upa.uParticles, this.particles.read.attach(0));
    gl.uniform1i(upa.uVelocity, this.velocity.read.attach(1));
    gl.uniform1i(upa.uCurl, this.curl.attach(2));
    gl.uniform1i(upa.uObstacles, this.obstacles.read.attach(3));
    gl.uniform2f(upa.uSimTexel, simTexelX, simTexelY);
    gl.uniform1f(upa.uDt, clampedDt);
    gl.uniform1f(upa.uTime, this.time);
    gl.uniform1i(upa.uMode, modeInt);
    gl.uniform1f(upa.uBiolumSens, params.biolumSensitivity);
    this.blit(this.particles.write);
    this.particles.swap();

    // 11. Non-blocking PBO Telemetry Readback
    this.pollOrRequestTelemetry(params);
  }

  private pollOrRequestTelemetry(params: FluidParams): void {
    const gl = this.gl;
    if (this.telemetrySync) {
      const status = gl.clientWaitSync(this.telemetrySync, 0, 0);
      if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.telemetryPBOs[this.telemetryPendingIdx]);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.telemetryBuf);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        gl.deleteSync(this.telemetrySync);
        this.telemetrySync = null;
        this.computeTelemetryFromBuffer(params);
      }
      return;
    }

    if (this.frameCounter % 10 === 0) {
      const idx = this.telemetryPboIdx;
      this.telemetryPboIdx = 1 - this.telemetryPboIdx;
      this.telemetryPendingIdx = idx;
      gl.useProgram(this.programs.telemetry);
      gl.uniform1i(this.uniforms.telemetry.uVelocity, this.velocity.read.attach(0));
      gl.uniform1i(this.uniforms.telemetry.uCurl, this.curl.attach(1));
      this.blit(this.telemetryFBO);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.telemetryPBOs[idx]);
      gl.readPixels(0, 0, 32, 32, gl.RGBA, gl.UNSIGNED_BYTE, 0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this.telemetrySync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
  }

  private computeTelemetryFromBuffer(params: FluidParams): void {

    let sumSpeed = 0;
    let maxCurl = 0;
    let sumKE = 0;
    const n = 32 * 32;
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      const speed = (this.telemetryBuf[idx] / 255) * 6.0;
      const curl = (this.telemetryBuf[idx + 1] / 255) * 4.0;
      const ke = (this.telemetryBuf[idx + 2] / 255) * 12.0;
      sumSpeed += speed;
      sumKE += ke;
      if (curl > maxCurl) maxCurl = curl;
    }

    const avgSpeed = sumSpeed / n;
    const avgKE = sumKE / n;
    const peakVorticity = maxCurl * 18.5; // scaled to physical s^-1
    const nu = Math.max(params.viscosity, 0.00004);
    const re = Math.round((avgSpeed * 0.25) / nu);
    const turb = Math.min(1, avgKE / 2.2);

    this.lastTelemetry = {
      fps: this.fps,
      reynoldsNumber: re,
      peakVorticity,
      kineticEnergy: avgKE,
      turbulenceRatio: turb,
      particleCount: PARTICLE_COUNT,
      backendLabel: this.backendLabel,
    };
  }

  render(params: FluidParams, magnets: MagnetPole[]): void {
    this.updateFps();
    this.renderToScreen(params, magnets, this.canvas.width, this.canvas.height, 1.0);
  }

  private renderToScreen(
    params: FluidParams,
    magnets: MagnetPole[],
    width: number,
    height: number,
    pointScale: number
  ): void {
    const gl = this.gl;
    const modeInt = this.modeToInt(params.mode);
    const aspect = width / Math.max(height, 1);

    // 1. Display Fluid / Ferrofluid / Obstacles
    gl.disable(gl.BLEND);
    gl.useProgram(this.programs.display);
    const ud = this.uniforms.display;
    gl.uniform1i(ud.uDye, this.dye.read.attach(0));
    gl.uniform1i(ud.uVelocity, this.velocity.read.attach(1));
    gl.uniform1i(ud.uCurl, this.curl.attach(2));
    gl.uniform1i(ud.uObstacles, this.obstacles.read.attach(3));
    gl.uniform2f(ud.uDyeTexel, 1.0 / this.dyeW, 1.0 / this.dyeH);
    gl.uniform2f(ud.uSimTexel, 1.0 / this.simW, 1.0 / this.simH);
    gl.uniform1f(ud.uAspect, aspect);
    gl.uniform1f(ud.uTime, this.time);
    gl.uniform1i(ud.uMode, modeInt);
    gl.uniform1f(ud.uMagneticGain, params.magneticGain);
    this.uploadMagnets('display', magnets);
    this.blit(null, width, height);

    // 2. Render 8,192 GPU Particles with Additive Blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.useProgram(this.programs.particleRender);
    const upr = this.uniforms.particleRender;
    gl.uniform1i(upr.uParticles, this.particles.read.attach(0));
    gl.uniform1f(upr.uPointScale, pointScale);
    gl.uniform1i(upr.uMode, modeInt);
    gl.bindVertexArray(this.emptyVAO);
    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);
    gl.disable(gl.BLEND);
  }

  private updateFps(): void {
    this.fpsFrames++;
    const now = performance.now();
    const elapsed = now - this.lastFpsTime;
    if (elapsed >= 450) {
      this.fps = Math.round((this.fpsFrames * 1000) / elapsed);
      this.lastTelemetry.fps = this.fps;
      this.fpsFrames = 0;
      this.lastFpsTime = now;
    }
  }

  getTelemetry(): TelemetrySnapshot {
    return this.lastTelemetry;
  }

  export4KSnapshot(params: FluidParams, magnets: MagnetPole[]): string {
    const prevW = this.canvas.width;
    const prevH = this.canvas.height;
    const targetW = 3840;
    const targetH = 2160;

    this.canvas.width = targetW;
    this.canvas.height = targetH;
    this.renderToScreen(params, magnets, targetW, targetH, 2.5);

    const outCanvas = document.createElement('canvas');
    outCanvas.width = targetW;
    outCanvas.height = targetH;
    const ctx = outCanvas.getContext('2d')!;
    ctx.drawImage(this.canvas, 0, 0, targetW, targetH);

    // Crisp studio watermark stamp
    ctx.save();
    ctx.fillStyle = 'rgba(6, 9, 19, 0.72)';
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.55)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(targetW - 980, targetH - 130, 930, 86, 14);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#06b6d4';
    ctx.font = 'bold 28px "Fira Code", monospace';
    ctx.fillText('AETHER-FLUID // NAVIER-STOKES CFD STUDIO (4K)', targetW - 950, targetH - 86);
    ctx.fillStyle = '#f59e0b';
    ctx.font = '600 22px "Outfit", sans-serif';
    ctx.fillText('أ. طارق ابوعشي | Tareq Abuashi — Lead Creative Technologist & Systems Architect', targetW - 950, targetH - 56);
    ctx.restore();

    const dataUrl = outCanvas.toDataURL('image/png');
    this.canvas.width = prevW;
    this.canvas.height = prevH;
    this.renderToScreen(params, magnets, prevW, prevH, 1.0);
    return dataUrl;
  }

  dispose(): void {
    const gl = this.gl;
    for (const key of Object.keys(this.programs)) {
      gl.deleteProgram(this.programs[key]);
    }
    if (this.telemetrySync) gl.deleteSync(this.telemetrySync);
    gl.deleteBuffer(this.telemetryPBOs[0]);
    gl.deleteBuffer(this.telemetryPBOs[1]);
    gl.deleteVertexArray(this.emptyVAO);
  }
}
