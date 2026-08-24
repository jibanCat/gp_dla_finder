"""Voigt absorption profiles for the Lyman series.

The absorption profile of an absorber at redshift ``z_dla`` with column density
``nhi`` is

.. math::

    \\tau(\\lambda) = N_\\mathrm{HI} \\sum_j a_j\\, V(v_j(\\lambda); \\sigma, \\gamma_j)

evaluated as ``exp(-tau)`` and convolved with a discrete instrumental line-spread
function (LSF). ``V`` is the Voigt function, computed from the Faddeeva function
``scipy.special.wofz`` — the same function the reference C implementation obtains
from ``libcerf``.

Backends
--------
The numerics live behind :class:`VoigtBackend`. Two backends exist:

``numpy``
    :class:`NumpyVoigtBackend`, using ``scipy.special.wofz``. Always present, and
    the official v0.1 backend.
``libcerf``
    :class:`LibcerfVoigtBackend`, a compiled extension using libcerf's Faddeeva
    function. Present only where the optional extension was built. It exists for
    **fidelity, not speed**: it is the Faddeeva implementation behind the
    deployed DESI catalogs, and it is measurably not the same function as
    SciPy's. End to end it is about 3% *slower* than the NumPy backend
    (measured; the Voigt evaluation is roughly a third of the per-sample cost and
    NumPy's ``wofz`` ufunc is already a tight C loop).

Backend selection is explicit. Requesting a backend that was not built raises
an error. A compiled
backend that disagrees with the NumPy backend beyond
:data:`BACKEND_AGREEMENT_ATOL` is not registered at all.

Instrumental line-spread function
---------------------------------
The LSF is a named, tabulated kernel, **not** a global default, because the
kernel is a property of the instrument/model configuration and choosing the wrong
one changes the inferred profile shape. The production kernel is
``"desi-r3000-7tap"``.

.. warning::

   The historical pure-Python module in the reference implementation
   (``gpy_dla_detection/voigt.py``) carries a *BOSS* R=2000 kernel while the
   compiled extension it stands in for uses the DESI R≈3000 kernel above. Those
   are different forward models: the profile shapes differ by up to ~4e-2 at
   ``log10 N_HI = 19`` on a DESI grid, largest in the LLS/sub-DLA regime.
   Users must select a kernel that describes their instrument.

References
----------
Garnett et al. (2017), arXiv:1605.04460 -- note this is the correct identifier;
some upstream docstrings cite 1605.04538, which is an unrelated paper.
Ho, Bird & Garnett (2020), arXiv:2003.11036.
Ho, Bird & Garnett (2021), arXiv:2103.10964.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

import numpy as np
from scipy.special import wofz

from ._immutable import frozen_array

__all__ = [
    "BACKEND_AGREEMENT_ATOL",
    "BACKEND_DECREMENT_ATOL",
    "BACKEND_DECREMENT_RTOL",
    "BackendRejected",
    "BOSS_KERNEL",
    "LSF_KERNELS",
    "PRODUCTION_KERNEL",
    "LibcerfVoigtBackend",
    "NumpyVoigtBackend",
    "VoigtBackend",
    "available_backends",
    "backend_provenance",
    "backend_local_diagnostics",
    "backend_rejections",
    "gaussian_lsf_kernel",
    "get_backend",
    "voigt_absorption",
]

# --------------------------------------------------------------------------------
# Physical constants and Lyman-series atomic data.
#
# Transcribed verbatim from the reference implementation
# (gpy_dla_detection/ctypes_voigt.c and gpy_dla_detection/voigt.py, which agree
# element-for-element). CGS units throughout. Do not "tidy" these numbers.
# --------------------------------------------------------------------------------

#: Speed of light, cm/s.
C_CGS: float = 2.99792458e10

#: Doppler width at the assumed gas temperature T = 1e4 K, cm/s.
#: Garnett et al. (2017) fix the thermal broadening at 13 km/s.
SIGMA_CGS: float = 9.08537121627923800e05

#: Lyman-series transition wavelengths, cm (Lyman-alpha first).
TRANSITION_WAVELENGTHS: np.ndarray = np.array(
    [
        1.2156701e-05,
        1.0257223e-05,
        9.725368e-06,
        9.497431e-06,
        9.378035e-06,
        9.307483e-06,
        9.262257e-06,
        9.231504e-06,
        9.209631e-06,
        9.193514e-06,
        9.181294e-06,
        9.171806e-06,
        9.16429e-06,
        9.15824e-06,
        9.15329e-06,
        9.14919e-06,
        9.14576e-06,
        9.14286e-06,
        9.14039e-06,
        9.13826e-06,
        9.13641e-06,
        9.13480e-06,
        9.13339e-06,
        9.13215e-06,
        9.13104e-06,
        9.13006e-06,
        9.12918e-06,
        9.12839e-06,
        9.12768e-06,
        9.12703e-06,
        9.12645e-06,
    ]
)

#: Oscillator strengths f_ul, dimensionless.
OSCILLATOR_STRENGTHS: np.ndarray = np.array(
    [
        0.416400,
        0.079120,
        0.029000,
        0.013940,
        0.007799,
        0.004814,
        0.003183,
        0.002216,
        0.001605,
        0.00120,
        0.000921,
        0.0007226,
        0.000577,
        0.000469,
        0.000386,
        0.000321,
        0.000270,
        0.000230,
        0.000197,
        0.000170,
        0.000148,
        0.000129,
        0.000114,
        0.000101,
        0.000089,
        0.000080,
        0.000071,
        0.000064,
        0.000058,
        0.000053,
        0.000048,
    ]
)

#: leading_constants[i] = pi e^2 f_i lambda_i / (m_e c), cm^2.
LEADING_CONSTANTS: np.ndarray = np.array(
    [
        1.34347262962625339e-07,
        2.15386482180851912e-08,
        7.48525170087141461e-09,
        3.51375347286007472e-09,
        1.94112336271172934e-09,
        1.18916112899713152e-09,
        7.82448627128742997e-10,
        5.42930932279390593e-10,
        3.92301197282493829e-10,
        2.92796010451409027e-10,
        2.24422239410389782e-10,
        1.75895684469038289e-10,
        1.40338556137474778e-10,
        1.13995374637743197e-10,
        9.37706429662300083e-11,
        7.79453203101192392e-11,
        6.55369055970184901e-11,
        5.58100321584169051e-11,
        4.77895916635794548e-11,
        4.12301389852588843e-11,
        3.58872072638707592e-11,
        3.12745536798214080e-11,
        2.76337116167110415e-11,
        2.44791750078032772e-11,
        2.15681362798480253e-11,
        1.93850080479346101e-11,
        1.72025364178111889e-11,
        1.55051698336865945e-11,
        1.40504672409331934e-11,
        1.28383057589411395e-11,
        1.16264059622218997e-11,
    ]
)

#: Lorentzian widths gamma_i = Gamma_i lambda_i / (4 pi), cm/s.
GAMMAS_CGS: np.ndarray = np.array(
    [
        6.06075804241938613e02,
        1.54841462408931704e02,
        6.28964942715328164e01,
        3.17730561586147395e01,
        1.82838676775503330e01,
        9.15463131005758157e00,
        6.08448802613156925e00,
        4.24977523573725779e00,
        3.08542121666345803e00,
        2.31184525202557767e00,
        1.77687796208123139e00,
        1.39477990932179852e00,
        1.11505539984541979e00,
        9.05885451682623022e-01,
        7.45877170715450677e-01,
        6.21261624902197052e-01,
        5.22994533400935269e-01,
        4.44469874827484512e-01,
        3.80923210837841919e-01,
        3.28912390446060132e-01,
        2.85949711597237033e-01,
        2.50280032040928802e-01,
        2.20224061101442048e-01,
        1.94686521675913549e-01,
        1.73082093051965591e-01,
        1.54536566013816490e-01,
        1.38539175663870029e-01,
        1.24652675945279762e-01,
        1.12585442799479921e-01,
        1.02045988802423507e-01,
        9.27433783998286437e-02,
    ]
)

# The atomic data define the forward model just as the LSF kernel does, so they
# get the same protection: shared, module-level, and genuinely immutable -- not
# merely flagged read-only, which a caller can undo with setflags(write=True).
TRANSITION_WAVELENGTHS = frozen_array(TRANSITION_WAVELENGTHS)
OSCILLATOR_STRENGTHS = frozen_array(OSCILLATOR_STRENGTHS)
LEADING_CONSTANTS = frozen_array(LEADING_CONSTANTS)
GAMMAS_CGS = frozen_array(GAMMAS_CGS)

MAX_LYMAN_LINES: int = len(TRANSITION_WAVELENGTHS)

#: Named LSF kernels. Each is a normalised, symmetric, odd-length pixel kernel.
#:
#: ``desi-r3000-7tap`` is the production kernel, transcribed verbatim from the
#: ``instrument_profile`` array in the reference ``ctypes_voigt.c``. It was built
#: for a representative DESI resolving power R = 3000; applied to the DESI 0.8 A
#: linear coadd grid it corresponds to sigma ~ 0.49 A, i.e. an effective
#: R ~ 3300-3900 across the Lyman-alpha search window.
#:
#: The registry and its arrays are **immutable**. The kernel defines the forward
#: model, so a caller that could edit it in place would silently change every
#: subsequent profile in the process.
_LSF_KERNELS: dict[str, np.ndarray] = {
    # BOSS/eBOSS convention: the 7-tap kernel hard-coded in the reference's
    # pure-Python voigt module, built for R = 2000 on the BOSS log-lambda pixel
    # grid. Use it for SDSS/eBOSS input spectra.
    "boss-r2000-7tap": np.array(
        [
            2.17460992138080811e-03,
            4.11623059580451742e-02,
            2.40309364651846963e-01,
            4.32707438937454059e-01,  # centre pixel
            2.40309364651846963e-01,
            4.11623059580451742e-02,
            2.17460992138080811e-03,
        ]
    ),
    "desi-r3000-7tap": np.array(
        [
            4.359382001258239556e-06,
            3.257925674795976966e-03,
            1.726040252342891379e-01,
            6.482673794178271942e-01,  # centre pixel
            1.726040252342891379e-01,
            3.257925674795976966e-03,
            4.359382001258239556e-06,
        ]
    ),
}

_LSF_KERNELS = {name: frozen_array(k) for name, k in _LSF_KERNELS.items()}

#: Read-only view of the kernel registry. Entries cannot be added, replaced, or
#: edited in place.
LSF_KERNELS: Mapping[str, np.ndarray] = MappingProxyType(_LSF_KERNELS)

#: The kernel used by every DESI production catalogue this package reproduces.
PRODUCTION_KERNEL: str = "desi-r3000-7tap"

#: The historical BOSS/eBOSS kernel, for SDSS-era input spectra.
BOSS_KERNEL: str = "boss-r2000-7tap"

#: Full width at half maximum of a Gaussian, in units of its standard deviation.
_FWHM_PER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian_lsf_kernel(
    resolving_power: float,
    pixel_scale: float,
    wavelength: float,
    *,
    truncate_sigma: float = 4.0,
) -> np.ndarray:
    """Build a Gaussian LSF kernel from a resolving power.

    Conventions, all of which are testable and none of which are implicit:

    * ``FWHM = wavelength / resolving_power`` in angstroms, so
      ``sigma = FWHM / (2 sqrt(2 ln 2))``;
    * the kernel is sampled on the *pixel* grid, ``sigma_pixels = sigma /
      pixel_scale``, at integer offsets from the centre;
    * it is truncated at ``+/- ceil(truncate_sigma * sigma_pixels)`` pixels, so it
      always has odd length and a well-defined centre;
    * it is normalised to sum to exactly 1 after truncation.

    .. warning::

       This is an **approximation**: a Gaussian of constant resolving power. Real
       instruments have a line-spread function that varies with wavelength and
       across spectrograph arms, and is not exactly Gaussian. Use it for
       exploration and for instruments this package has no named kernel for. It is
       **not** a reproduction path -- for that use a named historical kernel
       (:data:`PRODUCTION_KERNEL`, :data:`BOSS_KERNEL`), whose values are pinned
       and parity-tested.

    Parameters
    ----------
    resolving_power
        ``R = lambda / FWHM``, dimensionless. Must be finite and positive.
    pixel_scale
        Wavelength step per pixel, angstroms. Must be finite and positive.
    wavelength
        Wavelength at which to evaluate ``R``, angstroms. For a constant-``R``
        instrument use a representative wavelength of the search window.
    truncate_sigma
        Half-width of the kernel in standard deviations.

    Returns
    -------
    numpy.ndarray
        Normalised, symmetric, odd-length, read-only kernel.

    Raises
    ------
    ValueError
        On non-finite or non-positive inputs, or if the resulting kernel would be
        narrower than one pixel -- an under-sampled LSF is silently wrong, so it
        is refused rather than rounded up.
    """
    for name, value in (
        ("resolving_power", resolving_power),
        ("pixel_scale", pixel_scale),
        ("wavelength", wavelength),
        ("truncate_sigma", truncate_sigma),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")

    sigma_angstrom = wavelength / (resolving_power * _FWHM_PER_SIGMA)
    sigma_pixels = sigma_angstrom / pixel_scale
    if sigma_pixels < 0.5:
        raise ValueError(
            f"resolving power {resolving_power:g} at {wavelength:g} A on a "
            f"{pixel_scale:g} A pixel grid gives sigma = {sigma_pixels:.3f} pixels, "
            "which under-samples the line-spread function. Supply a finer pixel "
            "scale or a lower resolving power."
        )

    half_width = int(np.ceil(truncate_sigma * sigma_pixels))
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
    return frozen_array(kernel / kernel.sum())


def lsf_kernel(name: str) -> np.ndarray:
    """Return a named LSF kernel.

    The returned array is read-only and shared; copy it if you need to modify one
    (``lsf_kernel(name).copy()``).

    Raises
    ------
    KeyError
        If ``name`` is not a known kernel. There is deliberately no default and
        no nearest-match behaviour: silently substituting a kernel changes the
        forward model.
    """
    try:
        return _LSF_KERNELS[name]
    except KeyError:
        known = ", ".join(sorted(_LSF_KERNELS))
        raise KeyError(f"unknown LSF kernel {name!r}; known kernels: {known}") from None


def kernel_half_width(name: str) -> int:
    """Number of pixels trimmed from each end of a profile by ``name``'s kernel."""
    return (len(lsf_kernel(name)) - 1) // 2


def voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """Voigt line profile ``Re[w(z)] / sqrt(2 pi sigma^2)``.

    with ``z = (x + i gamma) / (sqrt(2) sigma)``.
    """
    z = (x + 1j * gamma) / (np.sqrt(2) * sigma)
    return np.real(wofz(z)) / (np.sqrt(2 * np.pi) * sigma)


class VoigtBackend(Protocol):
    """Computes instrument-convolved Lyman-series absorption profiles.

    Implementations must be pure functions of their arguments: the same inputs
    must give bit-identical outputs within and across processes so the finder
    remains deterministic.
    """

    name: str

    def absorption(
        self,
        wavelengths: np.ndarray,
        nhi: float,
        z_dla: float,
        num_lines: int,
        kernel: str,
        broadening: bool,
    ) -> np.ndarray:
        """Return ``exp(-tau)`` convolved with the LSF.

        ``wavelengths`` are observed-frame angstroms and must already be padded by
        ``kernel_half_width(kernel)`` pixels at each end when ``broadening`` is
        true; the returned array is shorter by twice that amount.
        """
        ...


class NumpyVoigtBackend:
    """Vectorized NumPy/SciPy backend and the official v0.1 implementation."""

    name = "numpy"

    def absorption(
        self,
        wavelengths: np.ndarray,
        nhi: float,
        z_dla: float,
        num_lines: int = 3,
        kernel: str = PRODUCTION_KERNEL,
        broadening: bool = True,
    ) -> np.ndarray:
        if not 1 <= num_lines <= MAX_LYMAN_LINES:
            raise ValueError(
                f"num_lines must be in [1, {MAX_LYMAN_LINES}], got {num_lines}"
            )
        wavelengths = np.asarray(wavelengths, dtype=np.float64)

        # velocity_j(lambda) = c * (lambda / (lambda_j (1+z)) - 1), with the 1e8
        # factor converting the transition wavelengths from cm to angstrom.
        multipliers = C_CGS / (TRANSITION_WAVELENGTHS[:num_lines] * (1 + z_dla)) / 1e8

        total = np.empty((num_lines, wavelengths.shape[0]))
        for line in range(num_lines):
            velocity = wavelengths * multipliers[line] - C_CGS
            total[line, :] = -LEADING_CONSTANTS[line] * voigt(
                velocity, SIGMA_CGS, GAMMAS_CGS[line]
            )

        raw_profile = np.exp(np.float64(nhi) * np.nansum(total, axis=0))

        if not broadening:
            return raw_profile

        # The kernels are symmetric, so correlation (the reference C loop) and
        # convolution (np.convolve, which flips the kernel) agree exactly.
        return np.convolve(raw_profile, lsf_kernel(kernel), "valid")


class LibcerfVoigtBackend:
    """Compiled backend using libcerf's Faddeeva function.

    This is the Faddeeva implementation the reference's compiled extension uses,
    and therefore the one behind the deployed DESI catalogues. It is **not**
    numerically identical to :class:`NumpyVoigtBackend`: libcerf and SciPy differ
    by up to ~2e-15 relative on the Lyman-alpha Voigt function at production
    parameters (measured; see ``tests/test_voigt_backends.py``). That is far below
    anything that moves a detection, but it is not zero, so the backend name is
    recorded in result provenance and the two are never substituted for each
    other silently.

    It is also not meaningfully faster end to end. See
    :func:`backend_provenance` for the recorded implementation details.
    """

    name = "libcerf"

    def __init__(self, extension):
        self._ext = extension

    def absorption(
        self,
        wavelengths: np.ndarray,
        nhi: float,
        z_dla: float,
        num_lines: int = 3,
        kernel: str = PRODUCTION_KERNEL,
        broadening: bool = True,
    ) -> np.ndarray:
        if not 1 <= num_lines <= MAX_LYMAN_LINES:
            raise ValueError(
                f"num_lines must be in [1, {MAX_LYMAN_LINES}], got {num_lines}"
            )
        wavelengths = np.ascontiguousarray(wavelengths, dtype=np.float64)

        raw_profile = self._ext.raw_absorption(
            wavelengths,
            float(nhi),
            float(z_dla),
            TRANSITION_WAVELENGTHS[:num_lines],
            LEADING_CONSTANTS[:num_lines],
            GAMMAS_CGS[:num_lines],
            SIGMA_CGS,
            C_CGS,
        )

        if not broadening:
            return raw_profile

        # Deliberately the same NumPy convolution every backend uses: switching
        # backend must change the Voigt function and nothing else.
        return np.convolve(raw_profile, lsf_kernel(kernel), "valid")


_BACKENDS: dict[str, VoigtBackend] = {"numpy": NumpyVoigtBackend()}

#: How far a compiled backend's un-broadened profile may sit from the NumPy
#: backend's before the package refuses to register it.
#:
#: The tolerance is **absolute**, deliberately. The profile lies in [0, 1] and
#: enters the likelihood by multiplying the model mean, so an absolute bound is
#: the one with a direct interpretation. A relative bound would be misleading:
#: relative error grows without limit in the saturated core of a strong absorber,
#: where ``exp(-tau)`` underflows towards zero and a 1e-15 error in ``tau`` of
#: order 1e4 shows up as a 1e-11 relative error on a number that is already 1e-23.
#:
#: Measured libcerf-versus-SciPy worst case over redshift 2.0-3.4 and
#: log10 N_HI 17.2-23.0: 9.4e-14 absolute. The gate sits two orders above that.
BACKEND_AGREEMENT_ATOL: float = 1e-11

_BACKEND_PROVENANCE: dict[str, dict[str, object]] = {
    "numpy": {
        "backend": "numpy",
        "faddeeva_source": "scipy.special.wofz",
        "compiled": False,
    }
}

#: Why a compiled backend that was *built* is nonetheless unavailable. An empty
#: entry here and a missing backend mean different things -- "not built" versus
#: "built and refused" -- and a user chasing a missing backend needs to know which.
_BACKEND_REJECTIONS: dict[str, str] = {}


def backend_rejections() -> Mapping[str, str]:
    """Backends that were built but failed their agreement checks, and why."""
    return MappingProxyType(dict(_BACKEND_REJECTIONS))


#: Relative tolerance on the absorption *decrement* ``1 - profile`` in the weak
#: regime. Absolute error on the profile itself says nothing there -- a profile of
#: 0.999999 agrees absolutely with 0.9999985 to 5e-7 while the physical signal,
#: the decrement, is wrong by 50%. The agreement gate therefore checks both.
BACKEND_DECREMENT_RTOL: float = 1e-9

#: Absolute floor on the decrement difference, below which the relative test is
#: not applied.
#:
#: The relative test divides by ``1 - profile``, which goes to zero in the far
#: wings. A difference of 1e-16 -- pure round-off -- against a decrement of
#: 1e-8 reads as a relative error of 1e-8 and rejects a perfectly good backend.
#: That is what refused the compiled backend on Linux at 9.32e-9 while the
#: absolute profile agreement was ~1e-14.
#:
#: The floor is chosen from measurement, not to clear a number that failed.
#: Across both audited platforms the largest *absolute* decrement difference
#: between the NumPy and libcerf backends is 4.8e-14 (macOS/arm64) and 3.3e-16
#: (Linux/x86-64), and every probe point with a decrement below 1e-5 agrees
#: exactly. A deliberately biased backend that shifts weak profiles by 1e-9 --
#: a 0.1% error on a decrement that carries signal -- sits three orders of
#: magnitude above this floor and is still rejected, which the test suite pins.
#:
#: So: two orders of magnitude above the worst real disagreement, three below
#: the smallest bias worth catching.
BACKEND_DECREMENT_ATOL: float = 1e-12


class BackendRejected(Exception):
    """A compiled backend failed its agreement checks and was not registered."""


def _verify_against_numpy(backend: VoigtBackend) -> dict[str, float]:
    """Check a backend against the NumPy backend, or raise.

    Absolute agreement is the primary gate, but it is not sufficient on its own.
    The checks run in the following order, and each fails closed:

    1. **shape** -- a backend returning the wrong length is not "slightly off";
    2. **finiteness** -- explicitly, before any maximum is taken. This closes a
       real hole: ``max(previous, np.nan)`` in Python returns ``previous``, so a
       backend emitting NaN could sail through a running-maximum gate with its
       error recorded as zero;
    3. **physical bounds** -- ``exp(-tau)`` must lie in [0, 1];
    4. **absolute agreement** on the un-broadened profile, the primary gate;
    5. **absolute agreement on the broadened profile**, because the convolution is
       where a length or ordering mistake would show;
    6. **relative agreement on the decrement** ``1 - profile`` in the weak regime,
       where absolute error on the profile is uninformative.

    An end-to-end evidence comparison is deliberately *not* here -- it costs
    seconds and this runs at import. It is a test
    (``test_voigt_backends.py``), which is the right place for it.

    Returns the measured quantities, so provenance records numbers rather than a
    verdict.

    Raises
    ------
    BackendRejected
        On any failed check, naming which one.
    """
    reference = _BACKENDS["numpy"]
    probe = np.linspace(3600.0, 6000.0, 512)
    padded = np.linspace(3600.0, 6000.0, 512 + 2 * kernel_half_width(PRODUCTION_KERNEL))

    worst_absolute = 0.0
    worst_broadened = 0.0
    worst_decrement_relative = 0.0
    worst_decrement_absolute = 0.0

    for z_dla in (2.0, 2.6, 3.4):
        for log_nhi in (17.2, 19.0, 20.3, 22.0, 23.0):
            nhi = 10.0**log_nhi
            expected = reference.absorption(
                probe, nhi, z_dla, 3, PRODUCTION_KERNEL, False
            )
            actual = backend.absorption(probe, nhi, z_dla, 3, PRODUCTION_KERNEL, False)

            if actual.shape != expected.shape:
                raise BackendRejected(
                    f"shape {actual.shape} != {expected.shape} at "
                    f"z={z_dla}, log N_HI={log_nhi}"
                )
            if not np.all(np.isfinite(actual)):
                raise BackendRejected(
                    f"non-finite profile values at z={z_dla}, log N_HI={log_nhi}"
                )
            if np.any(actual < 0.0) or np.any(actual > 1.0):
                raise BackendRejected(
                    f"profile outside [0, 1] at z={z_dla}, log N_HI={log_nhi}: "
                    f"range [{actual.min():.3e}, {actual.max():.3e}]"
                )

            worst_absolute = max(
                worst_absolute, float(np.max(np.abs(actual - expected)))
            )

            # The decrement, where the profile is weak enough for it to be the
            # quantity that carries the signal.
            weak = expected > 0.5
            if np.any(weak):
                expected_decrement = 1.0 - expected[weak]
                actual_decrement = 1.0 - actual[weak]
                difference = np.abs(actual_decrement - expected_decrement)
                # Hybrid: relative error is only meaningful where the difference
                # is itself larger than round-off. Dividing a 1e-16 difference by
                # a 1e-8 decrement produces a number that describes the floating
                # point format, not the backend.
                worst_decrement_absolute = max(
                    worst_decrement_absolute, float(np.max(difference))
                )
                significant = (expected_decrement > 0) & (
                    difference > BACKEND_DECREMENT_ATOL
                )
                if np.any(significant):
                    worst_decrement_relative = max(
                        worst_decrement_relative,
                        float(
                            np.max(
                                difference[significant]
                                / expected_decrement[significant]
                            )
                        ),
                    )

            expected_broadened = reference.absorption(
                padded, nhi, z_dla, 3, PRODUCTION_KERNEL, True
            )
            actual_broadened = backend.absorption(
                padded, nhi, z_dla, 3, PRODUCTION_KERNEL, True
            )
            if actual_broadened.shape != expected_broadened.shape:
                raise BackendRejected(
                    f"broadened shape {actual_broadened.shape} != "
                    f"{expected_broadened.shape}"
                )
            if not np.all(np.isfinite(actual_broadened)):
                raise BackendRejected("non-finite broadened profile values")
            worst_broadened = max(
                worst_broadened,
                float(np.max(np.abs(actual_broadened - expected_broadened))),
            )

    measured = {
        "max_absolute_difference_from_numpy": worst_absolute,
        "max_absolute_difference_broadened": worst_broadened,
        # Reported alongside the relative figure so a zero is readable: it means
        # "nothing exceeded BACKEND_DECREMENT_ATOL", and this says by how much.
        "max_absolute_decrement_difference": worst_decrement_absolute,
        "max_relative_decrement_difference": worst_decrement_relative,
    }

    for key in (
        "max_absolute_difference_from_numpy",
        "max_absolute_difference_broadened",
    ):
        value = measured[key]
        if not np.isfinite(value) or value > BACKEND_AGREEMENT_ATOL:
            raise BackendRejected(
                f"{key} = {value!r} exceeds BACKEND_AGREEMENT_ATOL "
                f"({BACKEND_AGREEMENT_ATOL:g})"
            )
    if (
        not np.isfinite(worst_decrement_relative)
        or worst_decrement_relative > BACKEND_DECREMENT_RTOL
    ):
        raise BackendRejected(
            f"max_relative_decrement_difference = {worst_decrement_relative!r} "
            f"exceeds BACKEND_DECREMENT_RTOL ({BACKEND_DECREMENT_RTOL:g})"
        )

    return measured


def _build_provenance() -> dict[str, object]:
    """The *shareable* half of what the optional extension linked against.

    This record excludes local paths because provenance can be shared outside the
    machine on which the extension was built. The library version, content hash,
    compiler family, and optimization flags identify the relevant calculation.
    Paths remain available only through :func:`backend_local_diagnostics`.
    """
    try:
        from ._build_info import SHAREABLE
    except ImportError:
        return {"build_info": "unavailable"}
    return dict(SHAREABLE)


def backend_local_diagnostics(name: str) -> Mapping[str, object]:
    """Machine-local build detail, for diagnosing a build in place.

    Contains absolute paths. **Never** merge this into a result, a catalogue, or
    anything else that leaves the machine -- that separation is the whole reason
    it is a separate function.
    """
    if name not in _BACKENDS:
        raise ValueError(f"unknown or unavailable Voigt backend {name!r}")
    try:
        from ._build_info import LOCAL
    except ImportError:
        return MappingProxyType({})
    return MappingProxyType(dict(LOCAL))


def _register_compiled_backends() -> None:
    """Register the compiled backend, if it was built *and* it agrees.

    A compiled extension that fails any check in :func:`_verify_against_numpy` is
    not registered. A build that went wrong -- a mismatched libcerf, a miscompiled
    inner loop, a NaN-emitting Faddeeva -- must not become a silently different
    forward model; refusing to register it leaves the installation on the NumPy
    backend, which is the official one anyway.
    """
    try:
        from . import _voigt_ext
    except ImportError:
        return

    backend = LibcerfVoigtBackend(_voigt_ext)
    try:
        measured = _verify_against_numpy(backend)
    except BackendRejected as exc:
        _BACKEND_REJECTIONS[backend.name] = str(exc)
        return
    except Exception as exc:  # pragma: no cover - a broken build, not a code path
        _BACKEND_REJECTIONS[backend.name] = f"agreement check raised: {exc!r}"
        return

    _BACKENDS[backend.name] = backend
    _BACKEND_PROVENANCE[backend.name] = {
        "backend": backend.name,
        "faddeeva_source": getattr(_voigt_ext, "FADDEEVA_SOURCE", "libcerf"),
        "compiled": True,
        **measured,
        **_build_provenance(),
    }


_register_compiled_backends()


def available_backends() -> tuple[str, ...]:
    """Names of Voigt backends available in this installation.

    Always contains ``"numpy"``. Contains ``"libcerf"`` only where the optional
    compiled extension was built and passed its agreement check at import.
    """
    return tuple(sorted(_BACKENDS))


def backend_provenance(name: str) -> Mapping[str, object]:
    """What a backend is, for the record a result carries.

    Includes the Faddeeva implementation, whether the backend is compiled, and --
    for a compiled backend -- its measured agreement with the NumPy backend.
    """
    return MappingProxyType(dict(_BACKEND_PROVENANCE[name]))


def get_backend(name: str = "numpy") -> VoigtBackend:
    """Return the named Voigt backend.

    Raises
    ------
    ValueError
        If the backend is unknown or not built in this installation. A future
        compiled backend that is requested but unavailable raises here rather
        than falling back, so a run never silently changes forward model.
    """
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown or unavailable Voigt backend {name!r}; "
            f"available: {', '.join(available_backends())}"
        ) from None


def voigt_absorption(
    wavelengths: np.ndarray,
    nhi: float,
    z_dla: float,
    num_lines: int = 3,
    *,
    kernel: str = PRODUCTION_KERNEL,
    broadening: bool = True,
    backend: str = "numpy",
) -> np.ndarray:
    """Instrument-convolved Lyman-series absorption profile ``exp(-tau)``.

    Parameters
    ----------
    wavelengths
        Observed-frame wavelengths, angstroms. When ``broadening`` is true these
        must be padded by ``kernel_half_width(kernel)`` pixels at each end.
    nhi
        Neutral hydrogen column density, cm^-2 (linear, not log10).
    z_dla
        Absorber redshift.
    num_lines
        Number of Lyman-series members to include, starting at Lyman-alpha.
    kernel
        Named LSF kernel. Defaults to the production DESI kernel.
    broadening
        Convolve with the LSF. When false the bare profile is returned at full
        length.
    backend
        Voigt backend name.

    Returns
    -------
    numpy.ndarray
        ``exp(-tau)`` in [0, 1]; length ``len(wavelengths) - 2 * half_width``
        when broadening, else ``len(wavelengths)``.
    """
    return get_backend(backend).absorption(
        wavelengths,
        nhi=nhi,
        z_dla=z_dla,
        num_lines=num_lines,
        kernel=kernel,
        broadening=broadening,
    )
