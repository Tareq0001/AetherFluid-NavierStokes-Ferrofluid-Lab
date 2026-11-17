# AETHER-FLUID: Real-Time Navier-Stokes Fluid, Ferrofluid & Bioluminescent Particle Studio

> **Lead Creative Technologist & Systems Architect:** أ. طارق ابوعشي | Tareq Abuashi

A high-performance, real-time 2D Eulerian Computational Fluid Dynamics (CFD) & Lagrangian Particle laboratory powered by **WebGL2 Ping-Pong Framebuffer Shaders** (with an automatic **Canvas 2D TypedArray Fallback** for universal 60 FPS compatibility).

---

## Core Physics & Visual Modes

1. **Neon Plasma Advection (`CFD 01`)**
   - Unconditionally stable Jos Stam *Stable Fluids* solver with semi-Lagrangian advection, Helmholtz-Hodge pressure projection, and **Vorticity Confinement** ($\mathbf{f}_{\text{conf}} = \varepsilon h (\mathbf{N} \times \boldsymbol{\omega})$) to preserve crisp micro-eddies.
2. **Ferrofluid Magnetic Spikes (`CFD 02`)**
   - Interactive magnetic dipoles (North attract / South repel) exercising **Kelvin magnetic body forces** and **Rosensweig normal-field instability** heightfield spikes rendered with obsidian-chrome PBR specular & Fresnel shading.
3. **Kármán Vortex Street & Wind Tunnel (`CFD 03`)**
   - Continuous inlet wind tunnel with no-slip velocity & Neumann pressure boundary enforcement around solid obstacles. Includes **Draw/Erase Obstacles** brush tools and aerodynamic presets (Cylinder, NACA Airfoil, Twin Cylinders).
4. **Bioluminescent Plankton (`CFD 04`)**
   - **8,192+ GPU Lagrangian particles** (`6,400` in Canvas 2D fallback) advected with RK2 midpoint integration that ignite in cyan and electric blue proportionally to local fluid **Shear Rate & Curl**.

---

## Real-Time Telemetry, Audio & Export

- **Live CFD Telemetry HUD & Scope:** Real-time FPS, Reynolds Number ($\text{Re}$), Peak Vorticity ($\omega_{\max}$ in $\text{s}^{-1}$), Kinetic Energy ($E_k$), Flow Regime classification (`LAMINAR` / `TRANSITIONAL` / `TURBULENT`), and dual-channel waveform oscilloscope.
- **Web Audio Fluid Sonification:** Synthesized filtered turbulence noise + harmonic ambient drone pads dynamically modulated by kinetic energy and vorticity.
- **CFD & Navier-Stokes Dossier Modal:** Interactive mathematical reference covering the incompressible Navier-Stokes PDEs, operator splitting, vorticity confinement, and magnetohydrodynamics.
- **4K PNG Snapshot Export:** Renders native `3840 × 2160` high-resolution fluid artwork directly from the GPU pipeline.

---

## Local Development & Build

```powershell
# Run dev server
npm run dev

# Type-check & production build
npm run build
```

*Built with TypeScript, WebGL2 GLSL ES 3.00, Web Audio API, and Vite.*
