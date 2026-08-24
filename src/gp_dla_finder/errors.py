"""Exception taxonomy for package and spectrum-level failures.

Callers need to distinguish four situations reliably:

* the run is misconfigured or its assets are wrong — nothing will work, fail the
  batch (:class:`ConfigurationError`, :class:`AssetError`);
* this particular spectrum is structurally invalid — the caller passed something
  that is not a usable spectrum (:class:`SpectrumError`);
* this spectrum is well formed but cannot support inference — fully masked, no
  normalisation coverage, too few usable pixels. That is **not** an error and not
  a non-detection; see :class:`~gp_dla_finder.gp.spectrum.InsufficientData`;
* the numerics failed on otherwise valid input (:class:`NumericalError`).

A processing failure must never be reported as a no-absorber result. Those are
different states and downstream population statistics depend on telling them
apart.
"""

from __future__ import annotations

__all__ = [
    "AssetError",
    "ConfigurationError",
    "GPDLAError",
    "NumericalError",
    "SpectrumError",
]


class GPDLAError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigurationError(GPDLAError):
    """The configuration is invalid, or incompatible with the chosen assets.

    Global: it invalidates the whole run, not one spectrum.
    """


class AssetError(GPDLAError):
    """A packaged or supplied asset is missing, malformed, or inconsistent.

    Global, like :class:`ConfigurationError`.
    """


class SpectrumError(GPDLAError):
    """The input is not a structurally valid spectrum.

    Spectrum-local: a batch layer may record the failure and continue. Wrong
    shapes, non-monotonic wavelengths, negative inverse variance, a non-finite
    redshift. Distinct from a *valid* spectrum that merely cannot support
    inference.
    """


class NumericalError(GPDLAError):
    """The numerics failed on structurally valid input.

    Spectrum-local. A non-positive-definite covariance, an all-NaN likelihood
    slice. Raised rather than returned as a result, because a numerical failure
    is not a scientific conclusion.
    """
