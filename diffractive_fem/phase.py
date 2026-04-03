"""
Phase function recovery via bivariate polynomial least squares.

This implements Section 4.4 of the Studienarbeit. Given mesh correspondences
between input and output planes, we recover the phase function φ(x,y) of the
diffractive phase element using the stationary phase approximation:

    ∇φ(x,y) = c · T(x,y)

where T is the geometric transformation defined by the mesh correspondences.

The phase is approximated as a bivariate polynomial:

    φ(x,y) = Σ_{k,l} a_{kl} · x^k · y^l,   k+l ≤ D-1

and the coefficients a_{kl} are found by least squares on the gradient conditions.
"""

import numpy as np
from typing import Tuple


def recover_phase_polynomial(
    mesh_in_x: np.ndarray, mesh_in_y: np.ndarray,
    mesh_out_x: np.ndarray, mesh_out_y: np.ndarray,
    D: int,
    grid_size: int = None
) -> np.ndarray:
    """
    Recover the phase polynomial coefficients from mesh correspondences.

    The mesh nodes define the geometric transformation:
      in(i,j) = (x_ij, y_ij)  →  out(i,j) = (φx_ij, φy_ij)

    The stationary phase condition is:
        ∇φ(x_ij, y_ij) = c · (φx_ij - 0.5, φy_ij - 0.5)

    where the output coordinates are centered (subtract 0.5) because
    the FFT places DC at the center. The constant c = 2π·N_grid
    converts from normalized coordinates to the phase scale required
    by the discrete Fourier transform.

    Parameters
    ----------
    mesh_in_x, mesh_in_y : ndarray, shape (s, s)
        Input mesh node coordinates in [0, 1].
    mesh_out_x, mesh_out_y : ndarray, shape (s, s)
        Output mesh node coordinates in [0, 1].
    D : int
        Polynomial degree parameter. The polynomial has degree D-1,
        with D*(D+1)/2 coefficients.
    grid_size : int, optional
        Size of the FFT propagation grid. If provided, the output
        coordinates are centered and scaled by 2π·grid_size.
        If None, no scaling is applied (unit constant c=1).

    Returns
    -------
    coeffs : ndarray, shape (D*(D+1)//2,)
        Polynomial coefficients a_{kl}, ordered by the indexing
        row = l*(2D-l+1)//2 + k.
    """
    s = mesh_in_x.shape[0]
    n_coeffs = D * (D + 1) // 2

    # Center and scale output coordinates for FFT compatibility
    if grid_size is not None:
        c = 2 * np.pi * grid_size
        target_x = c * (mesh_out_x - 0.5)
        target_y = c * (mesh_out_y - 0.5)
    else:
        target_x = mesh_out_x.copy()
        target_y = mesh_out_y.copy()

    # Build the matrix and RHS for the normal equations
    # From the Studienarbeit (Equations 11 and 12):
    #
    # M[row, col] = Σ_{i,j} [m·k · x_{ij}^{m-1+k-1} · y_{ij}^{n+l}
    #                       + n·l · x_{ij}^{m+k}     · y_{ij}^{n-1+l-1}]
    #
    # b[row] = Σ_{i,j} [k · φx^{ij} · x_{ij}^{k-1} · y_{ij}^l
    #                  + l · φy^{ij} · x_{ij}^k     · y_{ij}^{l-1}]

    M = np.zeros((n_coeffs, n_coeffs))
    b = np.zeros(n_coeffs)

    for row_l in range(D):
        for row_k in range(D - row_l):
            row = _coeff_index(row_k, row_l, D)

            # Build RHS
            rhs = 0.0
            for i in range(s):
                for j in range(s):
                    xij = mesh_in_x[i, j]
                    yij = mesh_in_y[i, j]
                    phix = target_x[i, j]
                    phiy = target_y[i, j]

                    if row_k > 0:
                        rhs += row_k * phix * _safe_pow(xij, row_k - 1) * _safe_pow(yij, row_l)
                    if row_l > 0:
                        rhs += row_l * phiy * _safe_pow(xij, row_k) * _safe_pow(yij, row_l - 1)
            b[row] = rhs

            # Build matrix row
            for col_n in range(D):
                for col_m in range(D - col_n):
                    col = _coeff_index(col_m, col_n, D)

                    val = 0.0
                    for i in range(s):
                        for j in range(s):
                            xij = mesh_in_x[i, j]
                            yij = mesh_in_y[i, j]

                            if col_m > 0 and row_k > 0:
                                val += (col_m * row_k *
                                        _safe_pow(xij, col_m - 1 + row_k - 1) *
                                        _safe_pow(yij, col_n + row_l))
                            if col_n > 0 and row_l > 0:
                                val += (col_n * row_l *
                                        _safe_pow(xij, col_m + row_k) *
                                        _safe_pow(yij, col_n - 1 + row_l - 1))

                    M[row, col] = val

    # Solve the system (with Tikhonov regularization for stability)
    # Add small regularization to diagonal to handle near-singular systems
    reg = 1e-10 * np.trace(M) / n_coeffs if np.isfinite(np.trace(M)) else 1e-10
    M += reg * np.eye(n_coeffs)

    # Replace any NaN/inf values
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        coeffs = np.linalg.lstsq(M, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        coeffs = np.linalg.solve(M + 1e-6 * np.eye(n_coeffs), b)

    return coeffs


def evaluate_phase(coeffs: np.ndarray, D: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Evaluate the phase polynomial φ(x,y) = Σ a_{kl} x^k y^l.

    Parameters
    ----------
    coeffs : ndarray
        Polynomial coefficients from recover_phase_polynomial.
    D : int
        Polynomial degree parameter (same as used in recovery).
    x, y : ndarray
        Coordinates at which to evaluate. Can be any shape; will be broadcast.

    Returns
    -------
    phi : ndarray
        Phase values, same shape as x and y.
    """
    phi = np.zeros_like(x, dtype=float)
    for l in range(D):
        for k in range(D - l):
            idx = _coeff_index(k, l, D)
            if idx < len(coeffs):
                phi += coeffs[idx] * _safe_pow(x, k) * _safe_pow(y, l)
    return phi


def evaluate_phase_gradient(
    coeffs: np.ndarray, D: int, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the gradient ∇φ = (∂φ/∂x, ∂φ/∂y).

    Returns
    -------
    dphi_dx, dphi_dy : ndarray
        Partial derivatives, same shape as x and y.
    """
    dphi_dx = np.zeros_like(x, dtype=float)
    dphi_dy = np.zeros_like(x, dtype=float)

    for l in range(D):
        for k in range(D - l):
            idx = _coeff_index(k, l, D)
            if idx < len(coeffs):
                if k > 0:
                    dphi_dx += k * coeffs[idx] * _safe_pow(x, k - 1) * _safe_pow(y, l)
                if l > 0:
                    dphi_dy += l * coeffs[idx] * _safe_pow(x, k) * _safe_pow(y, l - 1)

    return dphi_dx, dphi_dy


def _coeff_index(k: int, l: int, D: int) -> int:
    """
    Map polynomial indices (k, l) to a linear index.
    Uses the formula from the Studienarbeit:
        row = l * (2*D - l + 1) // 2 + k
    """
    return l * (2 * D - l + 1) // 2 + k


def _safe_pow(x, n):
    """Compute x^n safely, handling 0^0 = 1 and negative base with integer exponents."""
    if n == 0:
        return np.ones_like(x)
    return np.power(x, n)
