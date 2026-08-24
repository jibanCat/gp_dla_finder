"""Gaussian-process likelihood for a quasar spectrum.

The null (no-absorber) model for the observed flux is

.. math::

    p(y \\mid \\lambda, \\sigma^2) = \\mathcal{N}(y;\\;
        \\mu \\cdot a_{\\mathrm{Ly}\\alpha},\\;
        M M^{\\mathsf{T}} + \\Omega^2 + V)

where ``mu`` and ``M`` come from the trained model, ``a_lya`` is the Lyman-series
mean-flux suppression, ``Omega^2`` is the Lyman-forest absorption noise, and ``V``
is the pipeline's own per-pixel noise variance.

The covariance is a rank-``k`` update to a diagonal, so the Woodbury identity
evaluates the log-likelihood in ``O(n k^2)`` rather than ``O(n^3)`` — the single
hot operation in the whole finder.

The numerics follow the reference implementation
(`gpy_dla_detection/effective_optical_depth.py` and
`gpy_dla_detection/null_gp.py`). Expression order, summation order and the LAPACK
routine all affect bitwise reference fidelity, even when an alternative would be
mathematically equivalent.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lapack

from ..voigt import OSCILLATOR_STRENGTHS, TRANSITION_WAVELENGTHS

__all__ = [
    "effective_optical_depth",
    "log_mvnpdf_low_rank",
]

#: log(2 pi), to the precision the reference hard-codes it.
_LOG_2PI: float = 1.83787706640934534


def effective_optical_depth(
    wavelengths: np.ndarray,
    beta: float,
    tau_0: float,
    z_qso: float,
    num_forest_lines: int,
) -> np.ndarray:
    """Per-pixel, per-line Lyman-series effective optical depth.

    .. math::

        \\tau_i(\\lambda) = \\tau_0 \\frac{f_i \\lambda_i}{f_1 \\lambda_1}
                            (1 + z_i)^{\\beta},
        \\qquad 1 + z_i = \\lambda / \\lambda_i

    Absorbers beyond the quasar are switched off by an indicator ``z_i <= z_qso``.
    The reference multiplies by the indicator rather than masking, so pixels
    outside a line's forest contribute exactly zero rather than NaN; that choice
    is preserved because ``exp(-sum)`` then stays finite everywhere.

    Parameters
    ----------
    wavelengths
        Observed-frame wavelengths, angstroms.
    beta, tau_0
        Effective-optical-depth power law, ``tau_eff = tau_0 (1 + z)^beta``.
    z_qso
        Quasar redshift.
    num_forest_lines
        Number of Lyman-series members to include.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_pixels, num_forest_lines)``. Sum over axis 1 for the total.
    """
    if not 1 <= num_forest_lines <= len(TRANSITION_WAVELENGTHS):
        raise ValueError(
            f"num_forest_lines must be in [1, {len(TRANSITION_WAVELENGTHS)}], "
            f"got {num_forest_lines}"
        )

    # Transition wavelengths are stored in cm; the spectrum is in angstroms.
    transitions = TRANSITION_WAVELENGTHS * 1e8
    lya_wavelength = transitions[0]
    lya_oscillator_strength = OSCILLATOR_STRENGTHS[0]

    total_optical_depth = np.empty((wavelengths.shape[0], num_forest_lines))
    for i in range(num_forest_lines):
        # Absorber redshift at which line i lands on this pixel.
        this_z = (wavelengths - transitions[i]) / transitions[i]
        this_tau_0 = (
            tau_0
            * OSCILLATOR_STRENGTHS[i]
            / lya_oscillator_strength
            * transitions[i]
            / lya_wavelength
        )
        total_optical_depth[:, i] = this_tau_0 * (1 + this_z) ** beta
        total_optical_depth[:, i] *= this_z <= z_qso

    return total_optical_depth


def log_mvnpdf_low_rank(
    y: np.ndarray,
    mu: np.ndarray,
    M: np.ndarray,
    d: np.ndarray,
) -> float:
    """``log N(y; mu, M M^T + diag(d))`` via the Woodbury identity.

    With ``D = diag(d)`` and ``B = I + M^T D^-1 M``,

    .. math::

        K^{-1} = D^{-1} - D^{-1} M B^{-1} M^{\\mathsf{T}} D^{-1},
        \\qquad \\log\\det K = \\log\\det D + \\log\\det B

    so the cost is ``O(n k^2)`` instead of ``O(n^3)``.

    Ported verbatim, including two choices that look incidental and are not:
    ``B`` gets its identity added by strided in-place addition on the raveled
    array, and ``B^-1`` is applied via two ``lapack.dtrtri`` triangular inversions
    rather than triangular solves. Both affect the floating-point result at the
    last bits, which matters because the reference-equivalence tests are bitwise.

    Parameters
    ----------
    y
        Observed values, shape ``(n,)``.
    mu
        Mean, shape ``(n,)``.
    M
        Low-rank covariance factor, shape ``(n, k)``.
    d
        Diagonal of the noise covariance, shape ``(n,)``. Must be positive.

    Returns
    -------
    float
        The log density.
    """
    n, k = M.shape

    y = y[:, None] - mu[:, None]

    d_inv = 1 / d[:, None]
    D_inv_y = d_inv * y
    D_inv_M = d_inv * M

    # B = I + M' D^-1 M, with the identity added by strided in-place addition.
    B = np.matmul(M.T, D_inv_M)
    B.ravel()[0 :: (k + 1)] = B.ravel()[0 :: (k + 1)] + 1
    # NumPy's Cholesky is lower-triangular where MATLAB's is upper; the reference
    # accounts for that and so does this.
    L = np.linalg.cholesky(B)

    # C = B^-1 M' D^-1, formed from explicit triangular inverses.
    tmp = np.matmul(lapack.dtrtri(np.asfortranarray(L), lower=1)[0], D_inv_M.T)
    C = np.matmul(lapack.dtrtri(np.asfortranarray(L.T), lower=0)[0], tmp)

    K_inv_y = D_inv_y - np.matmul(D_inv_M, np.matmul(C, y))

    log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))

    return -0.5 * (np.matmul(y.T, K_inv_y).sum() + log_det_K + n * _LOG_2PI)
