#!/usr/bin/env python3
"""
Example: Circle (constant intensity) -> Equilateral Triangle (constant intensity)
using the general energy-weighted rectangular mesh method (Section 4.3).

The triangular topology from Section 4.2 is the natural choice for this geometry,
but the general method from Section 4.3 also works — the energy-weighted mesh
optimization adapts rectangular cells to approximate the triangular region.
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
from diffractive_fem.visualization import plot_mesh, plot_area_histogram


def make_circle(N, radius=0.15):
    """Circular aperture with constant amplitude."""
    x = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(x, x)
    amp = np.zeros((N, N))
    amp[xx**2 + yy**2 <= radius**2] = 1.0
    return amp


def make_equilateral_triangle(N, side=0.3):
    """Equilateral triangle with constant amplitude, centered at origin."""
    x = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(x, x)
    h = side * np.sqrt(3) / 2

    # Vertices of equilateral triangle centered at origin
    # V0 = (0, 2h/3), V1 = (-side/2, -h/3), V2 = (side/2, -h/3)
    y_top = 2 * h / 3
    y_bot = -h / 3

    amp = np.zeros((N, N))
    # Point (x,y) is inside if it satisfies all three half-plane conditions
    inside = (
        (yy >= y_bot) &                                       # above bottom edge
        (yy <= y_top - (y_top - y_bot) / (side / 2) * np.abs(xx))  # below the two slanted edges
    )
    amp[inside] = 1.0
    return amp


def main():
    print("=" * 65)
    print("  Circle -> Triangle (constant intensity)")
    print("  General energy-weighted method (Section 4.3)")
    print("=" * 65)

    N = 128;  s = 12;  D = 5;  sig_size = 64
    print(f"\nGrid: {N}x{N}, Mesh: {s}x{s}, Poly degree: {D-1}")

    # Signals (low-res for mesh, high-res for propagation)
    input_amp = make_circle(N)
    input_lo = make_circle(sig_size)
    desired_amp = make_equilateral_triangle(N)
    desired_lo = make_equilateral_triangle(sig_size)
    signal_window = desired_amp > 0.5

    # --- Meshes ---
    print("\n[1] Input mesh (energy-weighted for circle)...")
    mesh_in = RectMesh(s)
    mesh_in.init_uniform_square()
    mesh_in.optimize_equal_energy(input_lo, n_iter=400, alpha=0.3)
    e_in = mesh_in.cell_energies(input_lo)
    print(f"    sigma/mu = {np.std(e_in)/np.mean(e_in):.4f}")

    print("[1] Output mesh (energy-weighted for triangle)...")
    mesh_out = RectMesh(s)
    mesh_out.init_uniform_square()
    mesh_out.optimize_equal_energy(desired_lo, n_iter=400, alpha=0.3)
    e_out = mesh_out.cell_energies(desired_lo)
    print(f"    sigma/mu = {np.std(e_out)/np.mean(e_out):.4f}")

    # --- Phase recovery ---
    print(f"\n[2] Phase recovery (degree {D-1} polynomial)...")
    coeffs = recover_phase_polynomial(
        mesh_in.x, mesh_in.y, mesh_out.x, mesh_out.y, D, grid_size=N
    )
    t = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(t, t)
    phase_fem = evaluate_phase(coeffs, D, xx, yy)

    # --- Evaluate ---
    print("\n[3] Evaluating DPE...")
    field_in = make_input_field(input_amp)
    output_fem = propagate(apply_dpe(field_in, phase_fem))
    snr_fem = compute_snr(desired_amp, output_fem, signal_window)
    eff_fem = compute_efficiency(field_in, output_fem, signal_window)
    print(f"    FEM:  SNR = {snr_fem:.2f} dB, eta = {eff_fem*100:.1f}%")

    # --- IFTA ---
    print("\n[4] IFTA refinement...")
    phase_opt, hist_fem = ifta(input_amp, desired_amp, phase_fem,
                               n_iter_efficiency=10, n_iter_snr=20,
                               signal_window=signal_window, verbose=True)
    output_opt = propagate(apply_dpe(field_in, phase_opt))
    snr_opt = compute_snr(desired_amp, output_opt, signal_window)
    eff_opt = compute_efficiency(field_in, output_opt, signal_window)

    # --- Random baseline ---
    print("\n[5] Baseline: random phase + IFTA...")
    phase_rand = np.random.uniform(0, 2*np.pi, (N, N))
    phase_rand_opt, hist_rand = ifta(input_amp, desired_amp, phase_rand,
                                     n_iter_efficiency=10, n_iter_snr=20,
                                     signal_window=signal_window, verbose=False)
    output_rand = propagate(apply_dpe(field_in, phase_rand_opt))
    snr_rand = compute_snr(desired_amp, output_rand, signal_window)
    eff_rand = compute_efficiency(field_in, output_rand, signal_window)

    # Results
    print("\n" + "=" * 55)
    print("  Results: Circle -> Triangle (constant intensity)")
    print("=" * 55)
    print(f"  {'Method':<22} {'SNR (dB)':>10} {'eta (%)':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'FEM (no IFTA)':<22} {snr_fem:>10.2f} {eff_fem*100:>10.1f}")
    print(f"  {'FEM + IFTA':<22} {snr_opt:>10.2f} {eff_opt*100:>10.1f}")
    print(f"  {'Random + IFTA':<22} {snr_rand:>10.2f} {eff_rand*100:>10.1f}")
    print("=" * 55)

    # --- Figure ---
    print("\nGenerating figure...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Circle → Triangle (Constant Intensity, Energy-Weighted Method §4.3)',
                 fontsize=13, fontweight='bold')

    axes[0,0].imshow(input_amp, cmap='gray', origin='lower')
    axes[0,0].set_title('Input: Circle (const)')
    axes[0,1].imshow(desired_amp, cmap='gray', origin='lower')
    axes[0,1].set_title('Desired: Triangle (const)')
    plot_mesh(mesh_in, ax=axes[0,2], title=f'Input mesh (s={s})')
    plot_mesh(mesh_out, ax=axes[0,3], title='Output mesh', color='firebrick')

    im = axes[1,0].imshow(np.mod(phase_fem, 2*np.pi), cmap='twilight', origin='lower')
    axes[1,0].set_title('FEM phase mod 2π')
    plt.colorbar(im, ax=axes[1,0], shrink=0.8)
    axes[1,1].imshow(np.abs(output_fem)**2, cmap='inferno', origin='lower')
    axes[1,1].set_title(f'FEM output\nSNR={snr_fem:.1f} dB')
    axes[1,2].imshow(np.abs(output_opt)**2, cmap='inferno', origin='lower')
    axes[1,2].set_title(f'FEM+IFTA\nSNR={snr_opt:.1f} dB')
    axes[1,3].imshow(np.abs(output_rand)**2, cmap='inferno', origin='lower')
    axes[1,3].set_title(f'Random+IFTA\nSNR={snr_rand:.1f} dB')

    plt.tight_layout()
    fig.savefig('circle_to_triangle.png', dpi=150, bbox_inches='tight')
    print("  Saved circle_to_triangle.png")
    print("\nDone!")


if __name__ == '__main__':
    main()
