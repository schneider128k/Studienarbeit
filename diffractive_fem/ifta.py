"""
Iterative Fourier Transform Algorithm (IFTA).

The IFTA refines a diffractive phase element starting from an initial guess.
It alternates between the input and output planes, applying constraints in each:

  Input plane:  Keep the phase, replace amplitude with the input signal
  Output plane: Keep the phase (phase freedom), replace amplitude with
                the desired output signal (for SNR) or keep amplitude (for η)

This implements the Gerchberg-Saxton / Fienup type algorithm referenced in
the Studienarbeit (citations [Fie80], [Ger72], [Wyr88]).
"""

import numpy as np
from .propagation import propagate, inverse_propagate, apply_dpe
from .metrics import compute_snr, compute_efficiency
from typing import Optional, List, Tuple


def ifta(
    input_amplitude: np.ndarray,
    desired_output: np.ndarray,
    initial_phase: np.ndarray,
    n_iter_efficiency: int = 10,
    n_iter_snr: int = 20,
    signal_window: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, List[dict]]:
    """
    Run the Iterative Fourier Transform Algorithm.

    This follows the procedure described in Section 5 of the Studienarbeit:
    first run iterations to improve diffraction efficiency, then iterations
    to improve SNR.

    Parameters
    ----------
    input_amplitude : ndarray, shape (N, N)
        Amplitude of the input wavefront |f₁|.
    desired_output : ndarray, shape (N, N)
        Desired output intensity pattern |f₂|.
    initial_phase : ndarray, shape (N, N)
        Initial phase function φ(x,y), e.g. from the FEM method.
    n_iter_efficiency : int
        Number of iterations for efficiency improvement.
    n_iter_snr : int
        Number of iterations for SNR improvement.
    signal_window : ndarray, optional
        Boolean mask for the signal window. If None, uses nonzero region of desired_output.
    verbose : bool
        Print progress information.

    Returns
    -------
    optimized_phase : ndarray, shape (N, N)
        The optimized phase function.
    history : list of dict
        Metrics at each iteration: {'snr_db', 'efficiency', 'iteration', 'mode'}.
    """
    N = input_amplitude.shape[0]

    if signal_window is None:
        signal_window = np.abs(desired_output) > 1e-10 * np.max(np.abs(desired_output))

    phase = initial_phase.copy()
    history = []

    # Phase 1: Optimize for diffraction efficiency
    # Constraint: replace output amplitude with desired, keep output phase
    for it in range(n_iter_efficiency):
        # Forward: apply DPE and propagate
        field_in = input_amplitude * np.exp(1j * phase)
        field_out = propagate(field_in)

        # Record metrics
        snr = compute_snr(desired_output, field_out, signal_window)
        eff = compute_efficiency(
            input_amplitude.astype(complex), field_out, signal_window
        )
        history.append({
            'iteration': it,
            'mode': 'efficiency',
            'snr_db': snr,
            'efficiency': eff,
        })

        if verbose and it % 5 == 0:
            print(f"  [efficiency iter {it:3d}] SNR={snr:.2f} dB, η={eff*100:.1f}%")

        # Output constraint: replace amplitude with desired, keep phase
        output_phase = np.angle(field_out)
        field_out_constrained = desired_output * np.exp(1j * output_phase)

        # Inverse propagate
        field_in_back = inverse_propagate(field_out_constrained)

        # Input constraint: replace amplitude with input, keep phase
        phase = np.angle(field_in_back)

    # Phase 2: Optimize for SNR
    # Constraint: in signal window replace amplitude; outside keep as-is
    for it in range(n_iter_snr):
        field_in = input_amplitude * np.exp(1j * phase)
        field_out = propagate(field_in)

        snr = compute_snr(desired_output, field_out, signal_window)
        eff = compute_efficiency(
            input_amplitude.astype(complex), field_out, signal_window
        )
        history.append({
            'iteration': n_iter_efficiency + it,
            'mode': 'snr',
            'snr_db': snr,
            'efficiency': eff,
        })

        if verbose and it % 5 == 0:
            print(f"  [SNR iter {it:3d}]        SNR={snr:.2f} dB, η={eff*100:.1f}%")

        # Output constraint: inside signal window replace amplitude, keep phase;
        # outside signal window keep everything (allows energy redistribution)
        output_phase = np.angle(field_out)
        field_out_constrained = field_out.copy()
        field_out_constrained[signal_window] = (
            desired_output[signal_window] * np.exp(1j * output_phase[signal_window])
        )

        field_in_back = inverse_propagate(field_out_constrained)
        phase = np.angle(field_in_back)

    return phase, history
