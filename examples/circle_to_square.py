#!/usr/bin/env python3
"""
Section 4.1: Circle (constant intensity) -> Square (constant intensity)
using rectangular mesh topology.

Boundary nodes placed ON the circle/square shapes.
All mesh cells are INSIDE the shapes — no cells over the black region.
4-point interpolation for interior nodes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffractive_fem.mesh import RectMesh
from diffractive_fem.phase import recover_phase_polynomial, evaluate_phase
from diffractive_fem.propagation import propagate, apply_dpe, make_input_field
from diffractive_fem.metrics import compute_snr, compute_efficiency
from diffractive_fem.ifta import ifta
from diffractive_fem.visualization import plot_mesh


def make_circle(N, r=0.15):
    x = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(x, x)
    amp = np.zeros((N, N)); amp[xx**2 + yy**2 <= r**2] = 1.0
    return amp

def make_square(N, h=0.12):
    x = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(x, x)
    amp = np.zeros((N, N)); amp[(np.abs(xx) <= h) & (np.abs(yy) <= h)] = 1.0
    return amp


def main():
    print("=" * 65)
    print("  Section 4.1: Circle -> Square (constant intensity)")
    print("  Rectangular topology, boundary ON the shapes")
    print("=" * 65)

    N = 128; s = 15; D = 6

    # Input mesh: boundary on circle, 4-point interpolation
    print(f"\n[1] Input mesh (s={s}): boundary on circle, 4-pt interpolation...")
    mesh_in = RectMesh(s)
    mesh_in.init_uniform_circle(radius=0.25, cx=0.5, cy=0.5)
    areas = mesh_in.cell_areas()
    print(f"    Cell area sigma/mu = {np.std(areas)/np.mean(areas):.4f}")

    # Output mesh: uniform grid on square
    print("[1] Output mesh: uniform grid on square...")
    mesh_out = RectMesh(s)
    mesh_out.init_uniform_square()
    mesh_out.x = mesh_out.x * 0.5 + 0.25
    mesh_out.y = mesh_out.y * 0.5 + 0.25

    # Phase recovery
    print(f"\n[2] Phase recovery (degree {D-1}, grid_size={N})...")
    coeffs = recover_phase_polynomial(
        mesh_in.x, mesh_in.y, mesh_out.x, mesh_out.y, D, grid_size=N
    )
    t = np.linspace(0, 1, N); xx, yy = np.meshgrid(t, t)
    phase_fem = evaluate_phase(coeffs, D, xx, yy)

    # Evaluate
    input_amp = make_circle(N); desired_amp = make_square(N)
    signal_window = desired_amp > 0.5
    field_in = make_input_field(input_amp)
    output_fem = propagate(apply_dpe(field_in, phase_fem))
    snr_fem = compute_snr(desired_amp, output_fem, signal_window)
    eff_fem = compute_efficiency(field_in, output_fem, signal_window)
    print(f"\n[3] FEM:  SNR = {snr_fem:.2f} dB, eta = {eff_fem*100:.1f}%")

    # IFTA
    print("\n[4] IFTA refinement...")
    phase_opt, hist_f = ifta(input_amp, desired_amp, phase_fem,
        n_iter_efficiency=10, n_iter_snr=20, signal_window=signal_window, verbose=True)
    output_opt = propagate(apply_dpe(field_in, phase_opt))
    snr_opt = compute_snr(desired_amp, output_opt, signal_window)
    eff_opt = compute_efficiency(field_in, output_opt, signal_window)

    # Random baseline
    print("\n[5] Baseline: random + IFTA...")
    phase_r = np.random.uniform(0, 2*np.pi, (N, N))
    phase_r_opt, hist_r = ifta(input_amp, desired_amp, phase_r,
        n_iter_efficiency=10, n_iter_snr=20, signal_window=signal_window, verbose=False)
    output_r = propagate(apply_dpe(field_in, phase_r_opt))
    snr_r = compute_snr(desired_amp, output_r, signal_window)
    eff_r = compute_efficiency(field_in, output_r, signal_window)

    print("\n" + "=" * 55)
    print(f"  {'Method':<22} {'SNR (dB)':>10} {'eta (%)':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'FEM (no IFTA)':<22} {snr_fem:>10.2f} {eff_fem*100:>10.1f}")
    print(f"  {'FEM + IFTA':<22} {snr_opt:>10.2f} {eff_opt*100:>10.1f}")
    print(f"  {'Random + IFTA':<22} {snr_r:>10.2f} {eff_r*100:>10.1f}")
    print("=" * 55)

    # Figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('§4.1: Circle → Square (Constant Intensity, Rectangular Topology)',
                 fontsize=13, fontweight='bold')
    axes[0,0].imshow(input_amp, cmap='gray', origin='lower')
    axes[0,0].set_title('Input: Circle')
    axes[0,1].imshow(desired_amp, cmap='gray', origin='lower')
    axes[0,1].set_title('Desired: Square')
    plot_mesh(mesh_in, ax=axes[0,2], title=f'Input mesh (s={s})')
    plot_mesh(mesh_out, ax=axes[0,3], title='Output mesh', color='firebrick')

    im = axes[1,0].imshow(np.mod(phase_fem, 2*np.pi), cmap='twilight', origin='lower')
    axes[1,0].set_title('FEM phase mod 2π'); plt.colorbar(im, ax=axes[1,0], shrink=0.8)
    axes[1,1].imshow(np.abs(output_fem)**2, cmap='inferno', origin='lower')
    axes[1,1].set_title(f'FEM output\nSNR={snr_fem:.1f} dB')
    axes[1,2].imshow(np.abs(output_opt)**2, cmap='inferno', origin='lower')
    axes[1,2].set_title(f'FEM+IFTA\nSNR={snr_opt:.1f} dB')
    axes[1,3].imshow(np.abs(output_r)**2, cmap='inferno', origin='lower')
    axes[1,3].set_title(f'Random+IFTA\nSNR={snr_r:.1f} dB')

    plt.tight_layout()
    fig.savefig('circle_to_square.png', dpi=150, bbox_inches='tight')
    print("\n  Saved circle_to_square.png\nDone!")

if __name__ == '__main__':
    main()
