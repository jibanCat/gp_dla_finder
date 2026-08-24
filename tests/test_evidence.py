"""End-to-end evidence path — the increment-7 milestone.

A generated spectrum goes through all eight stages and the null and one-absorber
log evidences are compared against the reference implementation.

The comparison is **bitwise**. Everything the reference does is reproduced,
including two pieces of arithmetic that are mathematically no-ops and
numerically are not:

* observed wavelengths are re-derived as ``rest * (1 + z)`` rather than reused
  from the input, because the reference round-trips them through the rest frame;
* each per-sample log-likelihood carries ``- log N`` and the estimator adds
  ``+ log N`` back.

Getting either wrong leaves agreement at ~1e-12 instead of exact, which is how
both were found.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import load_model, load_sample_grid
from gp_dla_finder.config import Config
from gp_dla_finder.errors import SpectrumError
from gp_dla_finder.gp.evidence import (
    absorber_search_window,
    assemble_model,
    null_log_evidence,
    one_absorber_log_evidence,
)
from gp_dla_finder.gp.spectrum import InsufficientData, Spectrum, prepare_spectrum
from synthetic import WAVE, Z_QSO, make_spectrum


@pytest.fixture(scope="module")
def pipeline():
    model = load_model()
    config = Config.desi_y3_fast()
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(make_spectrum(), model, config)
    assembled = assemble_model(prepared, model, config)
    return model, config, grid, prepared, assembled


# --------------------------------------------------------------------------
# Stages 1-3: validation, normalisation, masking, padding
# --------------------------------------------------------------------------


def test_spectrum_rejects_structurally_invalid_input():
    good = dict(
        wavelength=WAVE, flux=np.ones_like(WAVE), ivar=np.ones_like(WAVE), z_qso=2.6
    )
    with pytest.raises(SpectrumError, match="strictly increasing"):
        Spectrum(**{**good, "wavelength": WAVE[::-1]})
    with pytest.raises(SpectrumError, match="expected"):
        Spectrum(**{**good, "flux": np.ones(5)})
    with pytest.raises(SpectrumError, match="z_qso must be finite"):
        Spectrum(**{**good, "z_qso": float("nan")})
    with pytest.raises(SpectrumError, match="ivar must be non-negative"):
        Spectrum(**{**good, "ivar": -np.ones_like(WAVE)})
    with pytest.raises(SpectrumError, match="non-finite values at unmasked"):
        Spectrum(**{**good, "flux": np.full_like(WAVE, np.nan)})


def test_zero_ivar_is_folded_into_the_mask():
    ivar = np.full_like(WAVE, 25.0)
    ivar[100:110] = 0.0
    spectrum = Spectrum(
        wavelength=WAVE, flux=np.ones_like(WAVE), ivar=ivar, z_qso=Z_QSO
    )
    assert spectrum.mask[100:110].all()
    assert not spectrum.mask[:100].any()


def test_preparation_keeps_only_unmasked_window_pixels(pipeline):
    _, config, _, prepared, _ = pipeline
    rest = prepared.rest_wavelength
    assert rest.min() >= config.min_lambda
    assert rest.max() <= config.max_lambda
    assert prepared.n_pixels == prepared.diagnostics["n_usable"]
    assert prepared.diagnostics["n_masked"] == 20


def test_padding_adds_exactly_the_lsf_half_width(pipeline):
    _, config, _, prepared, _ = pipeline
    assert prepared.padded_wavelength.size == prepared.window_wavelength.size + 2 * (
        config.convolution_half_width
    )


def test_a_fully_masked_spectrum_is_insufficient_data_not_an_error():
    """Processing failure and non-detection are different states."""
    model, config = load_model(), Config.desi_y3_fast()
    spectrum = Spectrum(
        wavelength=WAVE,
        flux=np.ones_like(WAVE),
        ivar=np.zeros_like(WAVE),  # everything masked
        z_qso=Z_QSO,
    )
    with pytest.raises(InsufficientData) as excinfo:
        prepare_spectrum(spectrum, model, config)
    assert excinfo.value.reason == "no_normalisation_coverage"


def test_a_spectrum_missing_the_normalisation_band_is_insufficient_data():
    model, config = load_model(), Config.desi_y3_fast()
    # Blue-only coverage: no pixels in the 1425-1475 A rest band at this redshift.
    short = np.arange(3600.0, 4200.0, 0.8)
    spectrum = Spectrum(
        wavelength=short,
        flux=np.ones_like(short),
        ivar=np.ones_like(short),
        z_qso=Z_QSO,
    )
    with pytest.raises(InsufficientData) as excinfo:
        prepare_spectrum(spectrum, model, config)
    assert excinfo.value.reason == "no_normalisation_coverage"


# --------------------------------------------------------------------------
# Stages 4-7: assembly and evidence
# --------------------------------------------------------------------------


def test_assembled_model_shapes_and_suppression(pipeline):
    model, _, _, prepared, assembled = pipeline
    assert assembled.mean.shape == (prepared.n_pixels,)
    assert assembled.factor.shape == (prepared.n_pixels, model.rank)
    assert assembled.absorption_variance.shape == (prepared.n_pixels,)
    lo, hi = assembled.diagnostics["mean_flux_suppression_range"]
    assert 0.0 < lo <= hi <= 1.0  # suppression only ever removes flux


def test_search_window_lies_inside_the_quasar_redshift(pipeline):
    _, config, _, prepared, _ = pipeline
    z_min, z_max = absorber_search_window(prepared, config)
    assert z_min < z_max < Z_QSO


def test_both_evidences_are_finite(pipeline):
    _, config, grid, prepared, assembled = pipeline
    assert np.isfinite(null_log_evidence(prepared, assembled))
    assert np.isfinite(one_absorber_log_evidence(prepared, assembled, grid, config))


@pytest.mark.slow
def test_the_production_sample_count_also_runs(pipeline):
    """The 50k default is exercised somewhere, not only the 10k fast preset.

    Marked slow: ~16 s single-core on the benchmark machine, versus ~3 s at 10k.
    """
    model = load_model()
    config = Config.desi_y3()
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(make_spectrum(), model, config)
    assembled = assemble_model(prepared, model, config)
    assert np.isfinite(one_absorber_log_evidence(prepared, assembled, grid, config))


def test_evidence_is_deterministic(pipeline):
    _, config, grid, prepared, assembled = pipeline
    first = one_absorber_log_evidence(prepared, assembled, grid, config)
    second = one_absorber_log_evidence(prepared, assembled, grid, config)
    assert first == second


def test_an_injected_absorber_raises_the_one_absorber_evidence():
    """The evidence must respond to the thing it is meant to detect."""
    from gp_dla_finder.voigt import voigt_absorption

    model, config = load_model(), Config.desi_y3_fast()
    grid = load_sample_grid(config.sample_grid)

    clean = make_spectrum(mask_slice=None)
    absorbed_flux = np.array(clean.flux) * voigt_absorption(
        np.concatenate([WAVE[:3], WAVE, WAVE[-3:]]),
        nhi=10**21.0,
        z_dla=2.3,
        num_lines=3,
    )
    absorbed = Spectrum(
        wavelength=WAVE, flux=absorbed_flux, ivar=np.array(clean.ivar), z_qso=Z_QSO
    )

    def bayes_factor(spectrum):
        prepared = prepare_spectrum(spectrum, model, config)
        assembled = assemble_model(prepared, model, config)
        return one_absorber_log_evidence(
            prepared, assembled, grid, config
        ) - null_log_evidence(prepared, assembled)

    assert bayes_factor(absorbed) > bayes_factor(clean)


def test_sample_count_mismatch_is_rejected(pipeline):
    _, config, grid, prepared, assembled = pipeline
    mismatched = config.replace(num_samples=123)
    with pytest.raises(ValueError, match="exact match"):
        one_absorber_log_evidence(prepared, assembled, grid, mismatched)


# Stage 8 -- bitwise equivalence with the reference implementation -- lives in
# tests/test_reference_parity.py, which runs the comparison under *both* named
# LSF kernels (PI ruling N34) and owns the reference-module patching.
