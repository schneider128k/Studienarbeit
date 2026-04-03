#!/usr/bin/env python3
"""
Demo: Diffractive beam shaping via the Finite Element Method.

Reproduces the core pipeline from Wocjan, Studienarbeit, Univ. Karlsruhe, 1997.

Gaussian -> Square flat-top, comparing FEM start vs random start for IFTA.
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


def make_gaussian(N, sigma=0.1):
    x = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(x, x)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def make_square(N, half_side=0.12):
    x = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(x, x)
    amp = np.zeros((N, N))
    amp[(np.abs(xx) <= half_side) & (np.abs(yy) <= half_side)] = 1.0
    return amp


def generate_energy_mesh(signal_lowres, s, n_iter=300, alpha=0.2):
    """Generate energy-equalized mesh for a signal."""
    mesh = RectMesh(s)
    mesh.init_uniform_square()
    mesh.optimize_equal_energy(signal_lowres, n_iter=n_iter, alpha=alpha)
    return mesh


def main():
    print("=" * 65)
    print("  Diffractive Beam Shaper via Finite Element Method")
    print("  Wocjan, Studienarbeit, Univ. Karlsruhe (TH), 1997")
    print("=" * 65)

    N = 128;  s = 10;  D = 5;  sig_size = 48
    print(f"\nGrid: {N}x{N}, Mesh: {s}x{s}, Poly degree: {D-1}")

    # Signals
    input_amp = make_gaussian(N, sigma=0.12)
    input_lo = make_gaussian(sig_size, sigma=0.12)
    desired_amp = make_square(N, half_side=0.1)
    desired_lo = make_square(sig_size, half_side=0.1)
    signal_window = desired_amp > 0.5

    # Step 1: Energy-equalized meshes
    print("\n[1] Generating input mesh (energy-weighted)...")
    mesh_in = generate_energy_mesh(input_lo, s)
    e_in = mesh_in.cell_energies(input_lo)
    print(f"    Energy uniformity: sigma/mu = {np.std(e_in)/np.mean(e_in):.4f}")

    print("[1] Generating output mesh (energy-weighted)...")
    mesh_out = generate_energy_mesh(desired_lo, s)
    e_out = mesh_out.cell_energies(desired_lo)
    print(f"    Energy uniformity: sigma/mu = {np.std(e_out)/np.mean(e_out):.4f}")

    # Step 2: Phase recovery
    print(f"\n[2] Phase recovery (degree {D-1} polynomial, {D*(D+1)//2} coefficients)...")
    coeffs = recover_phase_polynomial(mesh_in.x, mesh_in.y, mesh_out.x, mesh_out.y, D)

    t = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(t, t)
    phase_fem = evaluate_phase(coeffs, D, xx, yy)

    # Step 3: Evaluate
    print("\n[3] Evaluating DPE...")
    field_in = make_input_field(input_amp)
    output_fem = propagate(apply_dpe(field_in, phase_fem))
    snr_fem = compute_snr(desired_amp, output_fem, signal_window)
    eff_fem = compute_efficiency(field_in, output_fem, signal_window)
    print(f"    FEM only:  SNR = {snr_fem:.2f} dB,  eta = {eff_fem*100:.1f}%")

    # Step 4: IFTA refinement
    print("\n[4] IFTA refinement...")
    phase_opt, hist_fem = ifta(input_amp, desired_amp, phase_fem,
                               n_iter_efficiency=10, n_iter_snr=20,
                               signal_window=signal_window, verbose=True)
    output_opt = propagate(apply_dpe(field_in, phase_opt))
    snr_opt = compute_snr(desired_amp, output_opt, signal_window)
    eff_opt = compute_efficiency(field_in, output_opt, signal_window)

    # Step 5: Random baseline
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
    print("  Results: Gaussian -> Square flat-top")
    print("=" * 55)
    print(f"  {'Method':<22} {'SNR (dB)':>10} {'eta (%)':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'FEM (no IFTA)':<22} {snr_fem:>10.2f} {eff_fem*100:>10.1f}")
    print(f"  {'FEM + IFTA':<22} {snr_opt:>10.2f} {eff_opt*100:>10.1f}")
    print(f"  {'Random + IFTA':<22} {snr_rand:>10.2f} {eff_rand*100:>10.1f}")
    print("=" * 55)

    # Plots
    print("\nGenerating figure...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle('Diffractive Beam Shaper via Finite Element Method\n'
                 'Wocjan, Studienarbeit, Univ. Karlsruhe (TH), 1997',
                 fontsize=13, fontweight='bold')

    axes[0,0].imshow(input_amp**2, cmap='hot', origin='lower')
    axes[0,0].set_title('Input: Gaussian |f1|^2')
    axes[0,1].imshow(desired_amp**2, cmap='hot', origin='lower')
    axes[0,1].set_title('Desired: Square |f2|^2')
    plot_mesh(mesh_in, ax=axes[0,2], title=f'Input mesh (s={s})')
    plot_mesh(mesh_out, ax=axes[0,3], title='Output mesh', color='firebrick')

    im = axes[1,0].imshow(np.mod(phase_fem, 2*np.pi), cmap='twilight', origin='lower')
    axes[1,0].set_title('FEM phase mod 2pi')
    plt.colorbar(im, ax=axes[1,0], shrink=0.8)
    axes[1,1].imshow(np.abs(output_fem)**2, cmap='inferno', origin='lower')
    axes[1,1].set_title(f'FEM output\nSNR={snr_fem:.1f} dB')
    axes[1,2].imshow(np.abs(output_opt)**2, cmap='inferno', origin='lower')
    axes[1,2].set_title(f'FEM+IFTA\nSNR={snr_opt:.1f} dB')
    axes[1,3].imshow(np.abs(output_rand)**2, cmap='inferno', origin='lower')
    axes[1,3].set_title(f'Random+IFTA\nSNR={snr_rand:.1f} dB')

    iters_f = [h['iteration'] for h in hist_fem]
    snrs_f = [h['snr_db'] for h in hist_fem]
    iters_r = [h['iteration'] for h in hist_rand]
    snrs_r = [h['snr_db'] for h in hist_rand]
    axes[2,0].plot(iters_f, snrs_f, 'b-o', ms=3, label='FEM start')
    axes[2,0].plot(iters_r, snrs_r, 'r-s', ms=3, label='Random start')
    axes[2,0].set_xlabel('Iteration'); axes[2,0].set_ylabel('SNR (dB)')
    axes[2,0].set_title('IFTA convergence'); axes[2,0].legend(fontsize=9)
    axes[2,0].grid(True, alpha=0.3)

    effs_f = [h['efficiency']*100 for h in hist_fem]
    effs_r = [h['efficiency']*100 for h in hist_rand]
    axes[2,1].plot(iters_f, effs_f, 'b-o', ms=3, label='FEM start')
    axes[2,1].plot(iters_r, effs_r, 'r-s', ms=3, label='Random start')
    axes[2,1].set_xlabel('Iteration'); axes[2,1].set_ylabel('eta (%)')
    axes[2,1].set_title('Diffraction efficiency'); axes[2,1].legend(fontsize=9)
    axes[2,1].grid(True, alpha=0.3)

    mid = N // 2
    axes[2,2].plot(np.abs(desired_amp[mid,:])**2, 'k-', lw=2, label='Desired')
    axes[2,2].plot(np.abs(output_opt[mid,:])**2, 'b--', label='FEM+IFTA')
    axes[2,2].plot(np.abs(output_rand[mid,:])**2, 'r:', label='Rand+IFTA')
    axes[2,2].set_title('Cross-section (y=0)'); axes[2,2].legend(fontsize=9)

    plot_area_histogram(mesh_in, ax=axes[2,3], title='Input mesh cell areas')

    plt.tight_layout()
    fig.savefig('fem_beam_shaping.png', dpi=150, bbox_inches='tight')
    print("  Saved fem_beam_shaping.png")
    print("\nDone!")


if __name__ == '__main__':
    main()
