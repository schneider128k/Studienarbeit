#!/usr/bin/env python3
"""
Demo: Transform a Gaussian beam into a non-trivial intensity pattern
using energy-weighted mesh optimization.

This demonstrates the generalization from Section 4.3 of the Studienarbeit,
where the mesh optimization accounts for non-uniform intensity distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffractive_fem.mesh import RectMesh
from diffractive_fem.phase import recover_phase_polynomial, evaluate_phase
from diffractive_fem.propagation import propagate, apply_dpe, make_input_field
from diffractive_fem.metrics import compute_snr, compute_efficiency
from diffractive_fem.ifta import ifta
from diffractive_fem.visualization import plot_mesh, plot_energy_histogram


def make_gaussian_signal(size, sigma=0.15):
    """2D Gaussian amplitude on [0, 1] x [0, 1]."""
    x = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, x)
    return np.exp(-((xx - 0.5)**2 + (yy - 0.5)**2) / (2 * sigma**2))


def make_cross_signal(size, arm_width=0.08):
    """A cross-shaped intensity pattern."""
    x = np.linspace(-0.5, 0.5, size)
    xx, yy = np.meshgrid(x, x)
    cross = ((np.abs(xx) < arm_width) | (np.abs(yy) < arm_width)) & \
            ((np.abs(xx) < 0.3) & (np.abs(yy) < 0.3))
    amp = np.zeros((size, size))
    amp[cross] = 1.0
    return amp


def main():
    print("=" * 70)
    print("Gaussian → Cross: Energy-weighted FEM beam shaping")
    print("=" * 70)

    N = 256
    s = 12
    D = 6
    sig_size = 64  # Signal array size for mesh energy computation

    print(f"\nParameters: N={N}, s={s}, D={D}, signal grid={sig_size}")

    # Create signals for mesh optimization (low-res for speed)
    input_signal = make_gaussian_signal(sig_size, sigma=0.15)
    output_signal = make_cross_signal(sig_size, arm_width=0.08)

    # ---- Generate and optimize input mesh ----
    print("\n--- Generating input mesh (energy-weighted optimization) ---")
    mesh_in = RectMesh(s)
    mesh_in.init_uniform_square()
    mesh_in.laplace_smooth(n_iter=20, alpha=0.5)

    print("  Running energy-weighted optimization (150 iterations)...")
    mesh_in.optimize_equal_energy(input_signal, n_iter=150, alpha=0.25)

    energies_in = mesh_in.cell_energies(input_signal)
    mean_e = np.mean(energies_in)
    std_e = np.std(energies_in)
    print(f"  Input mesh energy stats: σ/μ = {std_e/mean_e:.4f}")

    # ---- Generate and optimize output mesh ----
    print("\n--- Generating output mesh (energy-weighted optimization) ---")
    mesh_out = RectMesh(s)
    mesh_out.init_uniform_square()
    mesh_out.laplace_smooth(n_iter=20, alpha=0.5)

    print("  Running energy-weighted optimization (150 iterations)...")
    mesh_out.optimize_equal_energy(output_signal, n_iter=150, alpha=0.25)

    # ---- Recover phase ----
    print(f"\n--- Phase recovery (polynomial degree {D-1}) ---")
    coeffs = recover_phase_polynomial(
        mesh_in.x, mesh_in.y,
        mesh_out.x, mesh_out.y,
        D
    )

    # ---- Propagate ----
    print("\n--- Evaluating DPE ---")
    x_grid = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x_grid, x_grid)
    phase = evaluate_phase(coeffs, D, xx, yy)

    input_amp_hr = make_gaussian_signal(N, sigma=0.15)
    desired_amp_hr = make_cross_signal(N, arm_width=0.08)

    input_field = make_input_field(input_amp_hr)
    output_field = propagate(apply_dpe(input_field, phase))

    signal_window = desired_amp_hr > 0.5
    snr = compute_snr(desired_amp_hr, output_field, signal_window)
    eff = compute_efficiency(input_field, output_field, signal_window)
    print(f"  FEM: SNR = {snr:.2f} dB, η = {eff*100:.1f}%")

    # ---- IFTA refinement ----
    print("\n--- IFTA refinement ---")
    refined_phase, history = ifta(
        input_amp_hr, desired_amp_hr, phase,
        n_iter_efficiency=10, n_iter_snr=20,
        signal_window=signal_window, verbose=True
    )
    output_refined = propagate(apply_dpe(input_field, refined_phase))
    snr_final = compute_snr(desired_amp_hr, output_refined, signal_window)
    eff_final = compute_efficiency(input_field, output_refined, signal_window)
    print(f"\n  FEM+IFTA: SNR = {snr_final:.2f} dB, η = {eff_final*100:.1f}%")

    # ---- Visualization ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(input_signal, cmap='hot', origin='lower')
    axes[0, 0].set_title('Input: Gaussian')

    axes[0, 1].imshow(output_signal, cmap='hot', origin='lower')
    axes[0, 1].set_title('Desired: Cross')

    plot_mesh(mesh_in, ax=axes[0, 2], title=f'Input mesh (s={s})')
    plot_mesh(mesh_out, ax=axes[0, 3], title='Output mesh', color='firebrick')

    im = axes[1, 0].imshow(np.mod(phase, 2*np.pi), cmap='twilight', origin='lower')
    axes[1, 0].set_title('Phase (FEM)')
    plt.colorbar(im, ax=axes[1, 0])

    axes[1, 1].imshow(np.abs(output_field)**2, cmap='inferno', origin='lower')
    axes[1, 1].set_title(f'FEM output (SNR={snr:.1f} dB)')

    axes[1, 2].imshow(np.abs(output_refined)**2, cmap='inferno', origin='lower')
    axes[1, 2].set_title(f'FEM+IFTA (SNR={snr_final:.1f} dB)')

    iters = [h['iteration'] for h in history]
    snrs = [h['snr_db'] for h in history]
    axes[1, 3].plot(iters, snrs, 'o-', markersize=3)
    axes[1, 3].set_xlabel('Iteration')
    axes[1, 3].set_ylabel('SNR (dB)')
    axes[1, 3].set_title('IFTA convergence')
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('gaussian_to_cross.png', dpi=150, bbox_inches='tight')
    print("\n  Saved gaussian_to_cross.png")
    plt.show()


if __name__ == '__main__':
    main()
