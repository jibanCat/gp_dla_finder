"""Deterministic generated spectra shared by the evidence and parity tests.

Nothing here comes from a survey or a mock catalogue: every spectrum is produced
from a named seed by the code in this file, so the fixtures can ship publicly
(PI ruling A4 — no 2LPT mock spectrum, truth catalogue, cutout, or private
identifier is distributed with this repository).

The generator is deliberately simple and *not* a physical simulation. Its job is
to exercise the inference path over a controlled range of redshift, signal to
noise, masking pattern and pixel grid, so that equivalence and approximation
claims rest on more than one anecdote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gp_dla_finder.gp.spectrum import Spectrum
from gp_dla_finder.voigt import PRODUCTION_KERNEL, kernel_half_width, voigt_absorption

#: The single spectrum the increment-7 milestone was measured on. Kept exactly
#: as it was so the recorded evidences remain comparable across increments.
Z_QSO = 2.6
WAVE = np.arange(3600.0, 5600.0, 0.8)


def make_spectrum(*, mask_slice=slice(500, 520), seed=20260819, ivar=25.0):
    """The milestone spectrum: deterministic, with structure and a masked region."""
    rng = np.random.default_rng(seed)
    flux = 1.0 + 0.3 * np.sin(WAVE / 180.0) + rng.normal(0, 0.2, WAVE.size)
    mask = np.zeros_like(WAVE, dtype=bool)
    if mask_slice is not None:
        mask[mask_slice] = True
    return Spectrum(
        wavelength=WAVE,
        flux=flux,
        ivar=np.full_like(WAVE, ivar),
        z_qso=Z_QSO,
        mask=mask,
    )


@dataclass(frozen=True)
class Case:
    """One named point in the comparison corpus."""

    name: str
    z_qso: float
    #: Observed-frame grid, angstroms.
    wave_start: float
    wave_stop: float
    pixel_scale: float
    #: Inverse variance per pixel; larger is higher signal to noise.
    ivar: float
    #: Fraction of pixels masked, spread over several contiguous chunks.
    masked_fraction: float
    #: ``None`` for an absorber-free spectrum, one ``(z_abs, log10 N_HI)`` pair,
    #: or several for a multiple-absorber case.
    absorber: tuple[float, float] | tuple[tuple[float, float], ...] | None
    seed: int

    @property
    def label(self) -> str:  # pragma: no cover - display only
        return self.name


#: A corpus spanning the axes the FILTER comparison has to cover: redshift,
#: signal to noise, usable-pixel fraction, masking pattern, pixel grid, and
#: absorber-free / weak / strong / high-column regimes.
#:
#: These are *generated* spectra. They exercise the machinery over a controlled
#: range; they are not a substitute for licence-cleared real spectra, which
#: remain the approved A+B+D fixture work.
CORPUS: tuple[Case, ...] = (
    Case(
        name="absorber-free-desi-grid",
        z_qso=2.6,
        wave_start=3600.0,
        wave_stop=5600.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.01,
        absorber=None,
        seed=20260819,
    ),
    Case(
        name="weak-lls-low-z",
        z_qso=2.25,
        wave_start=3600.0,
        wave_stop=5200.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.0,
        absorber=(2.05, 19.0),
        seed=11,
    ),
    Case(
        name="classical-dla-mid-z",
        z_qso=2.9,
        wave_start=3600.0,
        wave_stop=6000.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.05,
        absorber=(2.55, 20.5),
        seed=12,
    ),
    Case(
        name="high-column-dla",
        z_qso=3.2,
        wave_start=3600.0,
        wave_stop=6400.0,
        pixel_scale=0.8,
        ivar=100.0,
        masked_fraction=0.02,
        absorber=(2.85, 21.8),
        seed=13,
    ),
    Case(
        name="low-snr-dla",
        z_qso=2.7,
        wave_start=3600.0,
        wave_stop=5700.0,
        pixel_scale=0.8,
        ivar=1.0,
        masked_fraction=0.03,
        absorber=(2.35, 20.6),
        seed=14,
    ),
    Case(
        name="heavily-masked-dla",
        z_qso=2.8,
        wave_start=3600.0,
        wave_stop=5900.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.35,
        absorber=(2.45, 20.4),
        seed=15,
    ),
    Case(
        name="coarse-grid-dla",
        z_qso=2.6,
        wave_start=3600.0,
        wave_stop=5600.0,
        pixel_scale=1.6,
        ivar=25.0,
        masked_fraction=0.02,
        absorber=(2.3, 20.7),
        seed=16,
    ),
    # --- marginal detections -------------------------------------------------
    # Column densities tuned by bisection until the EXACT posterior lands on a
    # decision threshold, so a FILTER comparison can actually flip. The previous
    # corpus was all saturated at p = 1.000000 either way, which made "no flips"
    # a statement about the corpus rather than about FILTER.
    Case(
        name="marginal-p050",
        z_qso=2.6,
        wave_start=3600.0,
        wave_stop=5600.0,
        pixel_scale=0.8,
        ivar=1.0,
        masked_fraction=0.02,
        absorber=(2.30, 19.5329),
        seed=101,
    ),
    Case(
        name="marginal-p090",
        z_qso=2.6,
        wave_start=3600.0,
        wave_stop=5600.0,
        pixel_scale=0.8,
        ivar=1.0,
        masked_fraction=0.02,
        absorber=(2.30, 19.8099),
        seed=102,
    ),
    Case(
        name="marginal-p098",
        z_qso=2.6,
        wave_start=3600.0,
        wave_stop=5600.0,
        pixel_scale=0.8,
        ivar=1.0,
        masked_fraction=0.02,
        absorber=(2.30, 20.5319),
        seed=103,
    ),
    # --- near-null: no absorber, low signal, where a false positive would come
    # from if the approximation created one ------------------------------------
    Case(
        name="near-null-low-snr",
        z_qso=2.6,
        wave_start=3600.0,
        wave_stop=5600.0,
        pixel_scale=0.8,
        ivar=1.0,
        masked_fraction=0.02,
        absorber=None,
        seed=104,
    ),
    # --- multiple absorbers ---------------------------------------------------
    Case(
        name="two-separated-dlas",
        z_qso=2.9,
        wave_start=3600.0,
        wave_stop=6000.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.02,
        absorber=((2.20, 20.5), (2.70, 20.6)),
        seed=105,
    ),
    Case(
        name="two-blended-dlas",
        z_qso=2.9,
        wave_start=3600.0,
        wave_stop=6000.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.02,
        # Separated by ~0.02 in z: the damping wings overlap, so the one-absorber
        # model must try to explain both at once.
        absorber=((2.45, 20.5), (2.47, 20.4)),
        seed=106,
    ),
    Case(
        name="strong-plus-weak",
        z_qso=2.9,
        wave_start=3600.0,
        wave_stop=6000.0,
        pixel_scale=0.8,
        ivar=25.0,
        masked_fraction=0.02,
        absorber=((2.30, 21.0), (2.65, 19.2)),
        seed=107,
    ),
    Case(
        name="fine-grid-absorber-free",
        z_qso=2.5,
        wave_start=3600.0,
        wave_stop=5400.0,
        pixel_scale=0.4,
        ivar=25.0,
        masked_fraction=0.02,
        absorber=None,
        seed=17,
    ),
)


def absorbers_of(case: Case) -> tuple[tuple[float, float], ...]:
    """Normalise the ``absorber`` field to a tuple of ``(z_abs, log10 N_HI)``."""
    if case.absorber is None:
        return ()
    first = case.absorber[0]
    if isinstance(first, (int, float)):
        return (case.absorber,)  # type: ignore[return-value]
    return tuple(case.absorber)  # type: ignore[arg-type]


def build(case: Case, *, kernel: str = PRODUCTION_KERNEL) -> Spectrum:
    """Realise one corpus case as a :class:`Spectrum`.

    The absorber, when present, is imprinted with the same forward model the
    inference uses, evaluated on a grid padded by the kernel half-width so the
    convolution has no edge effect.
    """
    rng = np.random.default_rng(case.seed)
    wave = np.arange(case.wave_start, case.wave_stop, case.pixel_scale)

    continuum = 1.0 + 0.3 * np.sin(wave / 180.0) + 0.15 * np.cos(wave / 47.0)

    if case.absorber is not None:
        half = kernel_half_width(kernel)
        step = case.pixel_scale
        padded = np.concatenate(
            [
                wave[0] - step * np.arange(half, 0, -1),
                wave,
                wave[-1] + step * np.arange(1, half + 1),
            ]
        )
        for z_abs, log_nhi in absorbers_of(case):
            continuum = continuum * voigt_absorption(
                padded, nhi=10.0**log_nhi, z_dla=z_abs, num_lines=3, kernel=kernel
            )

    noise = rng.normal(0.0, 1.0 / np.sqrt(case.ivar), wave.size)
    mask = np.zeros(wave.size, dtype=bool)
    n_masked = int(round(case.masked_fraction * wave.size))
    if n_masked:
        # Several contiguous chunks rather than scattered pixels: masking in real
        # spectra comes from sky lines and bad columns, which are contiguous, and
        # contiguity is what stresses the window/padding arithmetic.
        #
        # The chunks are spread evenly rather than placed at random. Random
        # placement is not more realistic here and it has a failure mode that
        # defeats the purpose: at a high masked fraction one chunk can land on
        # the whole 1425-1475 A normalisation band, and the case then exercises
        # "no normalisation coverage" instead of "heavily masked".
        n_chunks = 8
        chunk = max(1, n_masked // n_chunks)
        for index in range(n_chunks):
            start = int((index + 0.5) * wave.size / n_chunks) - chunk // 2
            start = max(0, min(start, wave.size - chunk))
            mask[start : start + chunk] = True

    return Spectrum(
        wavelength=wave,
        flux=continuum + noise,
        ivar=np.full(wave.size, case.ivar),
        z_qso=case.z_qso,
        mask=mask,
    )
