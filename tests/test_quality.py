"""Named data-quality policies (PI ruling N33).

Two things must stay true, and both are easy to lose by accident:

* the inference path never applies a survey cut on its own;
* rejecting a spectrum on quality is a *decision*, recorded as such, and never
  reported as an inference failure or as a non-detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import load_model
from gp_dla_finder.config import Config
from gp_dla_finder.gp.spectrum import Spectrum, prepare_spectrum
from gp_dla_finder.quality import (
    DESI_Y3_REFERENCE,
    QUALITY_POLICIES,
    QualityPolicy,
    quality_policy,
)
from synthetic import WAVE, Z_QSO, make_spectrum


def spectrum_with_masked_fraction(fraction: float) -> Spectrum:
    """A spectrum whose 900-1230 A rest window is masked to ``fraction``."""
    rest = WAVE / (1 + Z_QSO)
    in_window = (rest > 900.0) & (rest < 1230.0)
    indices = np.flatnonzero(in_window)
    n_mask = int(round(fraction * indices.size))
    mask = np.zeros_like(WAVE, dtype=bool)
    mask[indices[:n_mask]] = True
    return Spectrum(
        wavelength=WAVE,
        flux=np.ones_like(WAVE),
        ivar=np.full_like(WAVE, 25.0),
        z_qso=Z_QSO,
        mask=mask,
    )


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_the_reference_policy_matches_the_deployed_requirement():
    """Transcribed from the reference's dlasearch, including its quirks.

    The window is 900-1230 A rest-frame, which is deliberately *not* the GP
    search window -- the reference carries a TODO saying exactly that -- and the
    threshold is 20 %.
    """
    assert DESI_Y3_REFERENCE.min_usable_fraction == 0.2
    assert DESI_Y3_REFERENCE.rest_lambda_min == 900.0
    assert DESI_Y3_REFERENCE.rest_lambda_max == 1230.0
    assert quality_policy("desi-y3-reference") is DESI_Y3_REFERENCE


def test_an_unknown_policy_raises_rather_than_falling_back():
    with pytest.raises(KeyError, match="unknown quality policy"):
        quality_policy("desi-y5")


def test_the_registry_is_read_only():
    with pytest.raises(TypeError):
        QUALITY_POLICIES["mine"] = DESI_Y3_REFERENCE


# --------------------------------------------------------------------------
# Selection is explicit, never inferred
# --------------------------------------------------------------------------


def test_production_presets_name_the_reference_policy():
    for config in (
        Config.desi_y3(),
        Config.desi_y3_fast(),
        Config.desi_y3_refined(),
    ):
        assert config.quality_policy == "desi-y3-reference"
        assert config.selected_quality_policy is DESI_Y3_REFERENCE


def test_a_custom_configuration_gets_no_policy_by_default():
    """No policy is inferred from an instrument label or a kernel name."""
    config = Config(preset="custom")
    assert config.quality_policy is None
    assert config.selected_quality_policy is None


def test_a_bad_policy_name_is_rejected_at_configuration_time():
    with pytest.raises(KeyError, match="unknown quality policy"):
        Config(preset="custom", quality_policy="not-a-policy")


def test_the_inference_path_does_not_apply_the_policy(monkeypatch):
    """The load-bearing separation.

    ``prepare_spectrum`` must not consult the quality policy. A spectrum that
    would fail the survey cut still goes through the low-level path, because
    deciding what belongs in a catalogue is not the inference's job.
    """
    model = load_model()
    config = Config.desi_y3_fast()
    assert config.selected_quality_policy is not None

    barely_usable = spectrum_with_masked_fraction(0.95)
    assessment = config.selected_quality_policy.assess(barely_usable)
    assert not assessment.passed

    prepared = prepare_spectrum(barely_usable, model, config)
    assert prepared.n_pixels > 0


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("masked", "expected_pass"),
    [(0.0, True), (0.5, True), (0.79, True), (0.81, False), (1.0, False)],
)
def test_the_threshold_is_where_it_says_it_is(masked, expected_pass):
    assessment = DESI_Y3_REFERENCE.assess(spectrum_with_masked_fraction(masked))
    assert assessment.passed is expected_pass
    assert assessment.usable_fraction == pytest.approx(1.0 - masked, abs=0.01)


def test_no_coverage_is_a_rejection_not_a_division_by_zero():
    # A quasar whose 900-1230 A rest window falls entirely outside the spectrum.
    wave = np.arange(9000.0, 9500.0, 0.8)
    spectrum = Spectrum(
        wavelength=wave,
        flux=np.ones_like(wave),
        ivar=np.ones_like(wave),
        z_qso=Z_QSO,
    )
    assessment = DESI_Y3_REFERENCE.assess(spectrum)
    assert assessment.n_in_window == 0
    assert not assessment.passed
    assert assessment.usable_fraction == 0.0


def test_zero_ivar_counts_as_masked():
    """The reference counts ``ivar != 0``; Spectrum folds zero ivar into the mask."""
    ivar = np.full_like(WAVE, 25.0)
    rest = WAVE / (1 + Z_QSO)
    in_window = (rest > 900.0) & (rest < 1230.0)
    indices = np.flatnonzero(in_window)
    ivar[indices[: int(0.9 * indices.size)]] = 0.0

    spectrum = Spectrum(
        wavelength=WAVE, flux=np.ones_like(WAVE), ivar=ivar, z_qso=Z_QSO
    )
    assessment = DESI_Y3_REFERENCE.assess(spectrum)
    assert not assessment.passed
    assert assessment.usable_fraction == pytest.approx(0.1, abs=0.01)


# --------------------------------------------------------------------------
# A rejection is a decision, and it is recorded
# --------------------------------------------------------------------------


def test_a_rejection_has_its_own_reason_code():
    """Distinct from an inability to compute, which is InsufficientData."""
    rejected = DESI_Y3_REFERENCE.assess(spectrum_with_masked_fraction(0.95))
    assert rejected.reason == "quality_policy_rejected"
    accepted = DESI_Y3_REFERENCE.assess(make_spectrum())
    assert accepted.reason is None


def test_provenance_records_everything_needed_to_rerun_the_decision():
    record = dict(
        DESI_Y3_REFERENCE.assess(spectrum_with_masked_fraction(0.5)).provenance()
    )
    assert record["quality_policy"] == "desi-y3-reference"
    assert record["quality_policy_version"] == "1"
    assert record["quality_passed"] is True
    assert record["quality_threshold"] == 0.2
    assert record["quality_rest_lambda_min"] == 900.0
    assert record["quality_rest_lambda_max"] == 1230.0
    assert record["quality_n_usable"] < record["quality_n_in_window"]
    assert record["quality_usable_fraction"] == pytest.approx(0.5, abs=0.01)


def test_a_policy_never_raises_on_a_well_formed_spectrum():
    """ "Not for the catalogue" is a result, not an error."""
    strict = QualityPolicy(
        name="strict",
        version="1",
        summary="test policy",
        min_usable_fraction=1.0,
        rest_lambda_min=900.0,
        rest_lambda_max=1230.0,
    )
    assessment = strict.assess(spectrum_with_masked_fraction(0.01))
    assert not assessment.passed
