"""Validate and prepare one spectrum for the GP likelihood.

Turns a raw quasar spectrum into the arrays the Gaussian-process likelihood needs:

1. **structural validation** — is this a spectrum at all;
2. **normalisation and masking** — divide by the median flux over the model's
   own training band, then keep only unmasked pixels inside the search window;
3. **padded grid** — extend the window by the line-spread function's half-width so
   absorber profiles can be convolved without edge effects.

The arithmetic follows the reference implementation's ``NullGP.set_data``.
Two details affect numerical fidelity and are preserved exactly:

* the padding is built with ``np.logspace`` at a fixed spacing in *dex*. On a
  linear-Å grid that is not the local pixel scale, so the padded pixels are not
  evenly spaced with their neighbours. It affects only the outermost
  ``half_width`` pixels of the convolution, but it is part of the reference
  arithmetic;
* the median is ``nanmedian`` over pixels that are inside the normalisation band
  **and** unmasked, computed *before* the search-window cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._immutable import frozen_array
from ..config import Config
from ..errors import SpectrumError
from ..model import GPModel

__all__ = ["InsufficientData", "PreparedSpectrum", "Spectrum", "prepare_spectrum"]


@dataclass(frozen=True)
class InsufficientData(Exception):
    """A valid spectrum that cannot support inference.

    Not an error in the input and **not** a non-detection. Carries a stable
    ``reason`` code so a batch layer can aggregate causes.
    """

    reason: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


@dataclass(frozen=True)
class Spectrum:
    """One quasar spectrum, in observed-frame arrays.

    Parameters
    ----------
    wavelength
        Observed-frame wavelengths, angstroms, strictly increasing.
    flux
        Calibrated flux. Units are irrelevant: the model self-normalises.
    ivar
        Inverse variance. Zero marks a pixel with no information; such pixels are
        folded into the mask, matching the reference pipeline.
    mask
        Optional additional bad-pixel mask, ``True`` meaning *bad*.
    z_qso
        Quasar emission redshift.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    z_qso: float
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength, dtype=np.float64)
        flux = np.asarray(self.flux, dtype=np.float64)
        ivar = np.asarray(self.ivar, dtype=np.float64)

        if wavelength.ndim != 1:
            raise SpectrumError(f"wavelength must be 1-D, got shape {wavelength.shape}")
        for name, array in (("flux", flux), ("ivar", ivar)):
            if array.shape != wavelength.shape:
                raise SpectrumError(
                    f"{name} has shape {array.shape}, expected {wavelength.shape}"
                )
        if wavelength.size == 0:
            raise SpectrumError("spectrum is empty")
        if not np.all(np.isfinite(wavelength)):
            raise SpectrumError("wavelength contains non-finite values")
        if not np.all(np.diff(wavelength) > 0):
            raise SpectrumError("wavelength must be strictly increasing")
        if not np.isfinite(self.z_qso):
            raise SpectrumError(f"z_qso must be finite, got {self.z_qso!r}")
        if self.z_qso <= 0:
            raise SpectrumError(f"z_qso must be positive, got {self.z_qso!r}")
        if np.any(ivar < 0):
            raise SpectrumError("ivar must be non-negative")
        if not np.all(np.isfinite(ivar)):
            raise SpectrumError("ivar contains non-finite values")

        mask = (
            np.zeros(wavelength.shape, dtype=bool)
            if self.mask is None
            else np.asarray(self.mask, dtype=bool)
        )
        if mask.shape != wavelength.shape:
            raise SpectrumError(
                f"mask has shape {mask.shape}, expected {wavelength.shape}"
            )
        # Zero inverse variance carries no information; the reference pipeline
        # folds it into the mask, and doing it here means callers cannot forget.
        mask = mask | (ivar == 0)

        if np.any(~np.isfinite(flux[~mask])):
            raise SpectrumError("flux contains non-finite values at unmasked pixels")

        object.__setattr__(self, "wavelength", frozen_array(wavelength))
        object.__setattr__(self, "flux", frozen_array(flux))
        object.__setattr__(self, "ivar", frozen_array(ivar))
        object.__setattr__(self, "mask", frozen_array(mask, dtype=bool))

    @property
    def rest_wavelength(self) -> np.ndarray:
        """Rest-frame wavelengths, ``lambda / (1 + z_qso)``."""
        return self.wavelength / (1.0 + self.z_qso)

    @property
    def noise_variance(self) -> np.ndarray:
        """``1 / ivar``, with NaN where ``ivar == 0``, as the reference expects."""
        with np.errstate(divide="ignore"):
            variance = np.where(
                self.ivar > 0, 1.0 / np.where(self.ivar > 0, self.ivar, 1.0), np.nan
            )
        return variance


@dataclass(frozen=True)
class PreparedSpectrum:
    """A spectrum reduced to the pixels the likelihood will actually use."""

    #: Rest-frame wavelengths of the kept pixels.
    rest_wavelength: np.ndarray
    #: Observed-frame wavelengths of the kept pixels.
    wavelength: np.ndarray
    #: Normalised flux at the kept pixels.
    flux: np.ndarray
    #: Normalised noise variance at the kept pixels.
    noise_variance: np.ndarray
    #: Observed wavelengths inside the search window *before* masking. The
    #: absorber profile is evaluated here, then reduced by ``mask_in_window``.
    window_wavelength: np.ndarray
    #: Which pixels of ``window_wavelength`` survive the mask.
    mask_in_window: np.ndarray
    #: ``window_wavelength`` extended by the LSF half-width at each end.
    padded_wavelength: np.ndarray
    #: The median used to normalise, in the input's flux units.
    normalization_median: float
    z_qso: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_pixels(self) -> int:
        return int(self.flux.size)


def prepare_spectrum(
    spectrum: Spectrum,
    model: GPModel,
    config: Config,
    *,
    min_usable_pixels: int = 1,
) -> PreparedSpectrum:
    """Validate, normalise, mask, window and pad a spectrum.

    Parameters
    ----------
    min_usable_pixels
        Below this many unmasked pixels in the search window the spectrum cannot
        support inference. The default of 1 is a *structural* floor only: it is
        deliberately not a science quality cut, because the reference pipeline's
        selection (>20 % unmasked in a fixed rest-frame window) lives in its DESI
        I/O layer, not in the inference, and inventing a threshold here would
        silently change which spectra are searched.

    Raises
    ------
    InsufficientData
        The spectrum is valid but unusable: no normalisation coverage, nothing
        left after masking, or too few usable pixels.
    """
    rest = spectrum.rest_wavelength
    # The reference derives observed wavelengths by multiplying the rest-frame
    # array back up, i.e. `wave / (1 + z) * (1 + z)`, rather than reusing the
    # input. That round trip is not the identity in floating point -- it shifts
    # results by ~1e-13 A, which propagates into the optical depth and the
    # evidence. It is a named, versioned compatibility behaviour, not a modelling
    # choice: see :mod:`gp_dla_finder.compat`.
    if config.compatibility_profile.rest_frame_round_trip:
        observed = rest * (1.0 + spectrum.z_qso)
    else:
        observed = np.array(spectrum.wavelength, dtype=np.float64)
    flux = np.array(spectrum.flux, dtype=np.float64)
    variance = spectrum.noise_variance
    mask = spectrum.mask

    # --- stage 2a: normalise over the model's own training band ---------------
    lo_norm = model.normalization_min_lambda
    hi_norm = model.normalization_max_lambda
    in_band = (rest >= lo_norm) & (rest <= hi_norm) & (~mask)
    if not np.any(in_band):
        raise InsufficientData(
            "no_normalisation_coverage",
            f"no unmasked pixels in the model's normalisation band "
            f"[{lo_norm:.1f}, {hi_norm:.1f}] A rest-frame; the spectrum covers "
            f"[{rest[0]:.1f}, {rest[-1]:.1f}] A",
        )
    median = float(np.nanmedian(flux[in_band]))
    if not np.isfinite(median) or median == 0.0:
        raise InsufficientData(
            "degenerate_normalisation",
            f"median flux over the normalisation band is {median!r}",
        )
    flux = flux / median
    variance = variance / median**2

    # --- stage 2b: search window, then the mask ------------------------------
    in_window = (rest >= config.min_lambda) & (rest <= config.max_lambda)
    if not np.any(in_window):
        raise InsufficientData(
            "no_search_window_coverage",
            f"the spectrum has no pixels in rest-frame "
            f"[{config.min_lambda:.2f}, {config.max_lambda:.2f}] A",
        )
    window_wavelength = observed[in_window]
    mask_in_window = ~mask[in_window]

    keep = in_window & (~mask)
    n_keep = int(np.count_nonzero(keep))
    if n_keep < max(1, min_usable_pixels):
        raise InsufficientData(
            "too_few_usable_pixels",
            f"{n_keep} unmasked pixels in the search window, need at least "
            f"{max(1, min_usable_pixels)}",
        )

    # --- stage 3: pad for the LSF convolution --------------------------------
    # Verbatim from the reference: logspace at a fixed dex spacing, which is not
    # the local pixel scale on a linear-A grid. Reproducing the reference means
    # reproducing this.
    half_width = config.convolution_half_width
    spacing = config.pixel_spacing_dex
    lo, hi = window_wavelength.min(), window_wavelength.max()
    padded_wavelength = np.concatenate(
        [
            np.logspace(
                np.log10(lo) - half_width * spacing,
                np.log10(lo) - spacing,
                half_width,
            ),
            window_wavelength,
            np.logspace(
                np.log10(hi) + spacing,
                np.log10(hi) + half_width * spacing,
                half_width,
            ),
        ]
    )

    return PreparedSpectrum(
        rest_wavelength=frozen_array(rest[keep]),
        wavelength=frozen_array(observed[keep]),
        flux=frozen_array(flux[keep]),
        noise_variance=frozen_array(variance[keep]),
        window_wavelength=frozen_array(window_wavelength),
        mask_in_window=frozen_array(mask_in_window, dtype=bool),
        padded_wavelength=frozen_array(padded_wavelength),
        normalization_median=median,
        z_qso=float(spectrum.z_qso),
        diagnostics={
            "n_input_pixels": int(spectrum.wavelength.size),
            "n_masked": int(np.count_nonzero(mask)),
            "n_in_window": int(np.count_nonzero(in_window)),
            "n_usable": n_keep,
            "normalization_band": (lo_norm, hi_norm),
            "n_normalisation_pixels": int(np.count_nonzero(in_band)),
        },
    )
