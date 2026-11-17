export class MathDossierModal {
  private modalEl: HTMLElement;

  constructor(modalEl: HTMLElement) {
    this.modalEl = modalEl;
    this.render();
    this.bindEvents();
  }

  private render(): void {
    this.modalEl.innerHTML = `
      <div class="modal-dialog">
        <div class="modal-header">
          <div class="modal-title-wrap">
            <span class="modal-badge">CFD &amp; CONTINUUM MECHANICS DOSSIER</span>
            <h2 id="dossier-title">AETHER-FLUID: Incompressible Navier-Stokes &amp; Magnetohydrodynamics</h2>
            <p class="modal-author">
              Exclusive Lead Architect: <strong>أ. طارق ابوعشي | Tareq Abuashi — Lead Creative Technologist &amp; Systems Architect</strong>
            </p>
          </div>
          <button id="close-dossier-x" class="close-btn" aria-label="Close CFD Dossier">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div class="modal-body custom-scroll">
          <section class="math-section">
            <span class="section-tag">01 // EULERIAN MOMENTUM CONSERVATION</span>
            <h3>Incompressible Navier-Stokes Equations</h3>
            <p>
              The fluid continuum evolves on an Eulerian MAC/collocated grid by enforcing conservation of linear momentum and
              zero mass divergence (incompressibility):
            </p>
            <div class="formula-box highlight">
              <div>∂<strong>u</strong>/∂t + (<strong>u</strong> · ∇)<strong>u</strong> = &minus;(1/ρ)∇p + ν∇²<strong>u</strong> + <strong>f</strong><sub>ext</sub></div>
              <div>∇ · <strong>u</strong> = 0 &nbsp;&nbsp;(Incompressibility Constraint)</div>
            </div>
            <p>
              Here <strong>u</strong> = (u, v) is the velocity field, p is scalar pressure, ρ is fluid density, ν is kinematic viscosity,
              and <strong>f</strong><sub>ext</sub> encapsulates wind-tunnel jets, Kelvin magnetic forces, and vorticity confinement.
            </p>
          </section>

          <section class="math-section">
            <span class="section-tag">02 // JOS STAM STABLE FLUIDS PIPELINE</span>
            <h3>Unconditionally Stable Operator Splitting</h3>
            <p>
              Each time-step Δt splits the PDE into four sequential GPU ping-pong framebuffer passes:
            </p>
            <div class="formula-box">
              <div>1. Force Injection: &nbsp;<strong>u</strong>₁ = <strong>u</strong>₀ + Δt · <strong>f</strong><sub>ext</sub></div>
              <div>2. Viscous Diffusion: (I &minus; νΔt∇²)<strong>u</strong>₂ = <strong>u</strong>₁ &nbsp;(Jacobi Relaxation)</div>
              <div>3. Helmholtz-Hodge Projection: ∇²p = ∇·<strong>u</strong>₂ &nbsp;⇒&nbsp; <strong>u</strong>₃ = <strong>u</strong>₂ &minus; ∇p</div>
              <div>4. Semi-Lagrangian Advection: <strong>u</strong><sub>next</sub>(<strong>x</strong>) = <strong>u</strong>₃(<strong>x</strong> &minus; Δt·<strong>u</strong>₃(<strong>x</strong>))</div>
            </div>
          </section>

          <section class="math-section">
            <span class="section-tag">03 // VORTICITY CONFINEMENT &amp; KÁRMÁN SHEDDING</span>
            <h3>Restoring Sub-Grid Rotational Energy</h3>
            <p>
              First-order semi-Lagrangian advection introduces numerical dissipation that damps fine eddies. Vorticity confinement
              re-injects rotational energy along normalized curl gradients:
            </p>
            <div class="formula-box">
              <div>ω = (∇ × <strong>u</strong>)<sub>z</sub> = ∂v/∂x &minus; ∂u/∂y</div>
              <div><strong>N</strong> = ∇|ω| / (|∇|ω|| + ε₀), &nbsp;&nbsp;<span class="highlight-text"><strong>f</strong><sub>conf</sub> = ε · h · (<strong>N</strong> × ω<strong>k̂</strong>)</span></div>
            </div>
            <p>
              In Wind Tunnel mode, solid obstacle cells enforce no-slip velocity (<strong>u</strong> = 0) and Neumann pressure boundaries
              (∂p/∂n = 0), triggering periodic alternating vortex shedding known as a <strong>Kármán Vortex Street</strong>.
            </p>
          </section>

          <section class="math-section">
            <span class="section-tag">04 // FERROFLUID &amp; BIOLUMINESCENT COUPLING</span>
            <h3>Rosensweig Magnetic Spikes &amp; Lagrangian Plankton</h3>
            <p>
              In Ferrofluid mode, magnetic dipoles exert a Kelvin body force <strong>F</strong><sub>m</sub> = μ₀(<strong>M</strong> · ∇)<strong>H</strong>
              while Rosensweig normal-field instability sculpts specular obsidian-chrome peaks. In Bioluminescent Plankton mode,
              8,192+ GPU Lagrangian tracers advect via RK2 midpoint integration and emit cyan-blue bioluminescence proportional to
              local shear rate ‖∇<strong>u</strong>‖ and vorticity |ω|.
            </p>
          </section>
        </div>

        <div class="modal-footer">
          <span class="footer-note">AETHER-FLUID // Engineered by أ. طارق ابوعشي | Tareq Abuashi</span>
          <button id="close-dossier-btn" class="action-btn">Return to CFD Studio</button>
        </div>
      </div>
    `;
  }

  private bindEvents(): void {
    const closeX = this.modalEl.querySelector('#close-dossier-x');
    const closeBtn = this.modalEl.querySelector('#close-dossier-btn');
    closeX?.addEventListener('click', () => this.close());
    closeBtn?.addEventListener('click', () => this.close());
    this.modalEl.addEventListener('click', (e) => {
      if (e.target === this.modalEl) this.close();
    });
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !this.modalEl.classList.contains('hidden')) {
        this.close();
      }
    });
  }

  open(): void {
    this.modalEl.classList.remove('hidden');
  }

  close(): void {
    this.modalEl.classList.add('hidden');
  }
}
