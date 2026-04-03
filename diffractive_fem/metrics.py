"""
Quality metrics for diffractive phase elements.

Implements the signal-to-noise ratio (SNR) and diffraction efficiency (η)
from Section 5 of the Studienarbeit.

The SNR measures how well the generated output matches the desired signal
within the signal window W_Signal. The diffraction efficiency measures
how much of the input energy reaches the signal window.
"""

import numpy as np
from typing import Optional


def compute_snr(
    desired: np.ndarray,
    achieved: np.ndarray,
    signal_window: Optional[np.ndarray] = None
) -> float:
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB.

    SNR_I(f₂, f) = ||f||² / ||‖f₂| - α|f||²

    where α = <|f|, |f₂|> / ||f||² is the optimal scaling factor.

    Both signals are treated as intensity signals (we compare |f₂| with |f|).

    Parameters
    ----------
    desired : ndarray
        Desired output signal f₂ (complex or real amplitude).
    achieved : ndarray
        Actually achieved signal f (complex or real amplitude).
    signal_window : ndarray, optional
        Boolean mask defining the signal window W_Signal.
        If None, the entire array is used.

    Returns
    -------
    snr_db : float
        SNR in decibels.
    """
    if signal_window is None:
        signal_window = np.ones(desired.shape, dtype=bool)

    # Work with intensities (magnitudes) within the signal window
    f2 = np.abs(desired[signal_window])
    f = np.abs(achieved[signal_window])

    # Norms within the signal window
    norm_f_sq = np.sum(f ** 2)
    if norm_f_sq < 1e-30:
        return -np.inf

    # Optimal scaling factor
    alpha = np.sum(f * f2) / norm_f_sq

    # Noise = |f₂| - α|f|
    noise = f2 - alpha * f
    noise_energy = np.sum(noise ** 2)

    if noise_energy < 1e-30:
        return np.inf  # perfect match

    snr = norm_f_sq / noise_energy
    return 10 * np.log10(snr)


def compute_efficiency(
    input_field: np.ndarray,
    output_field: np.ndarray,
    signal_window: Optional[np.ndarray] = None,
    de_window: Optional[np.ndarray] = None
) -> float:
    """
    Compute the diffraction efficiency η.

    η = ||f||²_{W_Signal} / ||f₁||²_{W_DE}

    This is the fraction of input energy that reaches the signal window.

    Parameters
    ----------
    input_field : ndarray
        Input wavefront f₁.
    output_field : ndarray
        Output wavefront f (produced by the DPE).
    signal_window : ndarray, optional
        Boolean mask for the signal window in the output plane.
    de_window : ndarray, optional
        Boolean mask for the DE window in the input plane.

    Returns
    -------
    eta : float
        Diffraction efficiency as a fraction (0 to 1).
    """
    if signal_window is None:
        signal_window = np.ones(output_field.shape, dtype=bool)
    if de_window is None:
        de_window = np.ones(input_field.shape, dtype=bool)

    energy_out = np.sum(np.abs(output_field[signal_window]) ** 2)
    energy_in = np.sum(np.abs(input_field[de_window]) ** 2)

    if energy_in < 1e-30:
        return 0.0

    return energy_out / energy_in
