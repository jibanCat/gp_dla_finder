# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Compiled Lyman-series absorption profile, evaluated with libcerf.

Why this exists
---------------
Not for speed. NumPy's
``scipy.special.wofz`` ufunc is already a tight C loop, the Voigt evaluation is
about a third of the per-sample cost, and libcerf is roughly 1.2x faster than
scipy at the Faddeeva function itself. The end-to-end gain is a few percent, not
an order of magnitude, and the NumPy backend remains the official one.

What this buys is **fidelity**. The deployed DESI catalogues were produced with
the reference's compiled extension, which obtains its Voigt function from
libcerf. libcerf and scipy do not agree bitwise -- measured at up to 1.8e-15
relative on the Lyman-alpha profile at production parameters -- so reproducing a
published catalogue exactly requires the same Faddeeva implementation it used.
This backend is that path, made available without the reference's manual
build-libcerf-from-source-and-set-LD_LIBRARY_PATH procedure.

Scope
-----
The extension owns the inner loop and nothing else. It carries no atomic data:
transition constants, the Doppler width and the number of Lyman lines are all
passed in from :mod:`gp_dla_finder.voigt`, which stays the single source of
truth. The line-spread convolution is deliberately **not** done here -- it is
done once, in NumPy, for every backend, so that a backend switch changes the
Voigt function and nothing else.

Origin
------
Ported from ``gpy_dla_detection/ctypes_voigt.c`` in the reference implementation
(MIT, Copyright (c) 2016 Roman Garnett). Differences from that file, all
deliberate: the atomic data are parameters rather than static tables, the LSF is
handled by the caller rather than hard-coded as a 7-tap BOSS kernel, and the
number of lines is not capped at a compile-time constant.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport exp

cnp.import_array()

cdef extern from "cerf.h" nogil:
    double voigt(double x, double sigma, double gamma)


#: Reported by :func:`gp_dla_finder.voigt.backend_provenance`.
FADDEEVA_SOURCE = "libcerf"


def raw_absorption(
    const double[::1] wavelengths,
    double nhi,
    double z_dla,
    const double[::1] transition_wavelengths,
    const double[::1] leading_constants,
    const double[::1] gammas,
    double sigma,
    double c_cgs,
):
    """``exp(-tau)`` before instrumental broadening.

    ``transition_wavelengths``, ``leading_constants`` and ``gammas`` must already
    be sliced to the number of Lyman lines wanted and have equal length.

    The accumulation order matches the NumPy backend's: velocities are
    ``lambda * multiplier - c``, and line contributions are summed in ascending
    line order, which is what ``np.nansum(..., axis=0)`` does over a stack of
    lines.
    """
    cdef Py_ssize_t n = wavelengths.shape[0]
    cdef Py_ssize_t num_lines = transition_wavelengths.shape[0]

    if num_lines != leading_constants.shape[0] or num_lines != gammas.shape[0]:
        raise ValueError("atomic-data arrays must have matching lengths")
    if num_lines == 0:
        raise ValueError("at least one Lyman line is required")

    out_array = np.empty(n, dtype=np.float64)
    cdef double[::1] out = out_array
    cdef double[::1] multipliers = np.empty(num_lines, dtype=np.float64)

    cdef Py_ssize_t i, j
    cdef double total, velocity

    with nogil:
        for j in range(num_lines):
            multipliers[j] = c_cgs / (transition_wavelengths[j] * (1.0 + z_dla)) / 1e8

        for i in range(n):
            total = 0.0
            for j in range(num_lines):
                velocity = wavelengths[i] * multipliers[j] - c_cgs
                total = total - leading_constants[j] * voigt(velocity, sigma, gammas[j])
            out[i] = exp(nhi * total)

    return out_array


def voigt_function(const double[::1] x, double sigma, double gamma):
    """libcerf's Voigt function on an array, for backend-comparison tests."""
    cdef Py_ssize_t i, n = x.shape[0]
    out_array = np.empty(n, dtype=np.float64)
    cdef double[::1] out = out_array
    with nogil:
        for i in range(n):
            out[i] = voigt(x[i], sigma, gamma)
    return out_array
