"""
Fourier propagation for diffractive optics.

In the optical setup described in Section 1 of the Studienarbeit, input and
output planes are at focal distance f from a lens. Wave propagation between
these planes is described by the 2D Fourier transform:

    f₂ = F{ T_DPE · f₁ }

where T_DPE = exp(i·φ(x,y)) is the diffractive phase element transmission,
f₁ is the input wavefront, and f₂ is the output wavefront.

We use the FFT to compute this numerically on a discrete grid.
"""

import numpy as np


def propagate(field: np.ndarray) -> np.ndarray:
    """
    Propagate a complex wavefront through a Fourier-transforming system.

    This computes the 2D FFT with proper shifting, modeling the
    Fraunhofer diffraction / lens Fourier transform:
        output = F{input}

    Parameters
    ----------
    field : ndarray, shape (N, N)
        Complex input field (after applying the DPE).

    Returns
    -------
    output : ndarray, shape (N, N)
        Complex output field in the Fourier (output) plane.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm='ortho'))


def inverse_propagate(field: np.ndarray) -> np.ndarray:
    """
    Inverse propagation (from output plane back to input plane).

    Parameters
    ----------
    field : ndarray, shape (N, N)
        Complex field in the output plane.

    Returns
    -------
    output : ndarray, shape (N, N)
        Complex field in the input plane.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field), norm='ortho'))


def apply_dpe(input_field: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Apply a diffractive phase element to an input wavefront.

    T_DPE(x,y) = exp(i·φ(x,y))

    Parameters
    ----------
    input_field : ndarray, shape (N, N)
        Complex input wavefront f₁.
    phase : ndarray, shape (N, N)
        Phase function φ(x,y) of the DPE.

    Returns
    -------
    modulated : ndarray, shape (N, N)
        The modulated field T_DPE · f₁.
    """
    return input_field * np.exp(1j * phase)


def make_input_field(amplitude: np.ndarray, phase: np.ndarray = None) -> np.ndarray:
    """
    Create a complex input field from amplitude and (optional) phase.

    Parameters
    ----------
    amplitude : ndarray
        Real-valued amplitude distribution |f₁|.
    phase : ndarray, optional
        Phase distribution. If None, constant phase (= 0) is assumed.

    Returns
    -------
    field : ndarray
        Complex wavefront f₁ = |f₁| · exp(i·arg(f₁)).
    """
    if phase is None:
        return amplitude.astype(complex)
    return amplitude * np.exp(1j * phase)
