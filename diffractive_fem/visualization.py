"""
Visualization utilities for diffractive element design.

Provides plotting functions for meshes, phase functions, intensity patterns,
and convergence diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Optional


def plot_mesh(mesh, ax=None, title="Mesh", color='steelblue', linewidth=0.5):
    """
    Plot a rectangular mesh showing all grid lines.

    Parameters
    ----------
    mesh : RectMesh
        The mesh to plot.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str
        Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    s = mesh.s

    # Draw horizontal lines (constant j)
    for j in range(s):
        ax.plot(mesh.x[:, j], mesh.y[:, j], '-', color=color, linewidth=linewidth)

    # Draw vertical lines (constant i)
    for i in range(s):
        ax.plot(mesh.x[i, :], mesh.y[i, :], '-', color=color, linewidth=linewidth)

    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


def plot_mesh_comparison(mesh_in, mesh_out, titles=("Input mesh (E)", "Output mesh (A)")):
    """Plot input and output meshes side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_mesh(mesh_in, ax=ax1, title=titles[0])
    plot_mesh(mesh_out, ax=ax2, title=titles[1], color='firebrick')
    plt.tight_layout()
    return fig


def plot_area_histogram(mesh, ax=None, title="Cell area distribution"):
    """Plot histogram of mesh cell areas."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    areas = mesh.cell_areas()
    mean_area = np.mean(areas)

    ax.hist(areas.ravel(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(mean_area, color='red', linestyle='--', label=f'Mean = {mean_area:.6f}')
    ax.set_xlabel('Cell area')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    # Add statistics
    std = np.std(areas)
    max_dev = np.max(np.abs(areas - mean_area))
    ax.text(0.95, 0.95,
            f'σ/μ = {std/mean_area:.4f}\nmax dev = {max_dev/mean_area:.4f}',
            transform=ax.transAxes, va='top', ha='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax


def plot_energy_histogram(mesh, signal, ax=None, title="Cell energy distribution"):
    """Plot histogram of mesh cell energies."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    energies = mesh.cell_energies(signal)
    mean_e = np.mean(energies)

    ax.hist(energies.ravel(), bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(mean_e, color='red', linestyle='--', label=f'Mean = {mean_e:.2f}')
    ax.set_xlabel('Cell energy')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    std = np.std(energies)
    rel_std = std / mean_e if mean_e > 0 else 0
    ax.text(0.95, 0.95,
            f'σ/μ = {rel_std:.4f}',
            transform=ax.transAxes, va='top', ha='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax


def plot_phase(phase, ax=None, title="Phase function φ(x,y)"):
    """Plot the 2D phase function."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(phase, cmap='twilight', origin='lower')
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    ax.set_title(title)
    return ax


def plot_intensity(field, ax=None, title="Intensity |f|²", log_scale=False):
    """Plot intensity of a complex field."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    intensity = np.abs(field) ** 2
    if log_scale:
        intensity = np.log10(intensity + 1e-10)

    im = ax.imshow(intensity, cmap='inferno', origin='lower')
    plt.colorbar(im, ax=ax, label='Intensity' + (' (log)' if log_scale else ''))
    ax.set_title(title)
    return ax


def plot_convergence(history, ax=None):
    """Plot IFTA convergence history."""
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        ax1, ax2 = ax

    iters = [h['iteration'] for h in history]
    snrs = [h['snr_db'] for h in history]
    effs = [h['efficiency'] * 100 for h in history]

    # Color by mode
    colors = ['steelblue' if h['mode'] == 'efficiency' else 'firebrick' for h in history]

    ax1.scatter(iters, snrs, c=colors, s=10)
    ax1.plot(iters, snrs, 'k-', alpha=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Signal-to-Noise Ratio')

    ax2.scatter(iters, effs, c=colors, s=10)
    ax2.plot(iters, effs, 'k-', alpha=0.3)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('η (%)')
    ax2.set_title('Diffraction Efficiency')

    return ax1, ax2


def plot_results_summary(input_amp, desired, achieved, phase,
                         mesh_in=None, mesh_out=None):
    """
    Create a comprehensive summary figure showing all results.
    """
    n_rows = 2 if mesh_in is None else 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))

    # Row 1: Input, Desired output, Achieved output
    axes[0, 0].imshow(input_amp ** 2, cmap='gray', origin='lower')
    axes[0, 0].set_title('Input |f₁|²')

    axes[0, 1].imshow(np.abs(desired) ** 2, cmap='gray', origin='lower')
    axes[0, 1].set_title('Desired |f₂|²')

    axes[0, 2].imshow(np.abs(achieved) ** 2, cmap='gray', origin='lower')
    axes[0, 2].set_title('Achieved |f|²')

    # Row 2: Phase, Phase (wrapped), Comparison cross-section
    axes[1, 0].imshow(phase, cmap='twilight', origin='lower')
    axes[1, 0].set_title('Phase φ(x,y)')

    axes[1, 1].imshow(np.mod(phase, 2 * np.pi), cmap='twilight', origin='lower')
    axes[1, 1].set_title('Wrapped phase mod 2π')

    # Cross-section comparison
    mid = desired.shape[0] // 2
    axes[1, 2].plot(np.abs(desired[mid, :]) ** 2, 'b-', label='Desired', alpha=0.7)
    axes[1, 2].plot(np.abs(achieved[mid, :]) ** 2, 'r--', label='Achieved', alpha=0.7)
    axes[1, 2].set_title('Cross-section (middle row)')
    axes[1, 2].legend()

    # Row 3: Meshes (if provided)
    if mesh_in is not None and n_rows > 2:
        plot_mesh(mesh_in, ax=axes[2, 0], title='Input mesh')
        if mesh_out is not None:
            plot_mesh(mesh_out, ax=axes[2, 1], title='Output mesh', color='firebrick')
        axes[2, 2].axis('off')

    plt.tight_layout()
    return fig
