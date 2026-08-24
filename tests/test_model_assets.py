"""Trained-model asset tests.

The load-bearing one is :func:`test_packaged_model_reproduces_source_bitwise`,
which re-proves the PI ruling D2 precision rule against the actual source file
rather than trusting the conversion audit written at build time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gp_dla_finder import model as m

_SOURCE_ENV = "GP_DLA_FINDER_MODEL_SOURCE"


@pytest.fixture(scope="session")
def default_model():
    return m.load_model()


# --------------------------------------------------------------------------
# Asset integrity
# --------------------------------------------------------------------------


def test_default_model_is_bundled():
    assert m.DEFAULT_MODEL in m.available_models()


def test_model_loads_as_float64_regardless_of_stored_dtype(default_model):
    """Storage may be float32; inference arithmetic must see float64."""
    for name in ("rest_wavelengths", "mu", "M", "log_omega"):
        assert getattr(default_model, name).dtype == np.float64


def test_deployed_model_geometry(default_model):
    assert default_model.rank == 30
    lo, hi = default_model.rest_wavelength_range
    assert (lo, hi) == pytest.approx((850.75, 1700.0), abs=1e-6)
    assert default_model.rest_wavelengths.size == 5662
    assert default_model.normalization_min_lambda == 1425.0
    assert default_model.normalization_max_lambda == 1475.0


def test_deployed_model_covers_the_production_search_window(default_model):
    assert default_model.covers(911.75, 1250.0)
    assert not default_model.covers(800.0, 1250.0)


def test_provenance_records_source_checksum_and_unused_datasets():
    prov = m.model_provenance()
    assert len(prov["source"]["sha256"]) == 64
    # Training history must not have been carried into the package.
    assert "loss_history" in prov["source"]["datasets_unused"]
    assert "initial_M" in prov["source"]["datasets_unused"]
    assert prov["model"]["rank"] == 30


def test_provenance_matches_the_loaded_arrays(default_model):
    prov = m.model_provenance()
    assert prov["model"]["rank"] == default_model.rank
    assert prov["model"]["learned_tau_0"] == pytest.approx(default_model.learned_tau_0)
    assert prov["model"]["learned_beta"] == pytest.approx(default_model.learned_beta)


def test_stored_arrays_round_trip_bitwise_through_float32(default_model):
    """The float32 storage claim, re-checked from the loaded arrays.

    This is what makes the size reduction lossless rather than a precision cut.
    """
    for name in ("rest_wavelengths", "mu", "M", "log_omega"):
        a = getattr(default_model, name)
        assert np.array_equal(a, a.astype(np.float32).astype(np.float64))


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_unknown_model_lists_what_is_available():
    with pytest.raises(ValueError, match="unknown model"):
        m.load_model("latest")


def test_model_validates_array_shapes():
    with pytest.raises(ValueError, match="mu has shape"):
        m.GPModel(
            name="bad",
            rest_wavelengths=np.arange(10.0),
            mu=np.zeros(9),
            M=np.zeros((10, 2)),
            log_omega=np.zeros(10),
            log_c_0=0.0,
            log_tau_0=0.0,
            log_beta=0.0,
        )


def test_model_rejects_a_non_monotonic_grid():
    with pytest.raises(ValueError, match="strictly increasing"):
        m.GPModel(
            name="bad",
            rest_wavelengths=np.array([3.0, 1.0, 2.0]),
            mu=np.zeros(3),
            M=np.zeros((3, 2)),
            log_omega=np.zeros(3),
            log_c_0=0.0,
            log_tau_0=0.0,
            log_beta=0.0,
        )


# --------------------------------------------------------------------------
# Equivalence with the source artifact
# --------------------------------------------------------------------------


@pytest.mark.needs_private_data
def test_packaged_model_reproduces_source_bitwise(default_model):
    """Packaged asset vs the original trained file, element for element.

    Set ``GP_DLA_FINDER_MODEL_SOURCE`` to the trained ``phase2_result.h5``.
    """
    raw = os.environ.get(_SOURCE_ENV)
    if not raw:
        pytest.skip(f"set {_SOURCE_ENV} to the source trained model")
    source = Path(raw).expanduser()
    if not source.is_file():
        pytest.skip(f"{source} not found")

    pytest.importorskip("h5py")
    reference = m.load_model(path=source)

    for name in ("rest_wavelengths", "mu", "M", "log_omega"):
        assert np.array_equal(getattr(default_model, name), getattr(reference, name)), (
            f"{name} differs from the source artifact"
        )
    for name in ("log_c_0", "log_tau_0", "log_beta"):
        assert getattr(default_model, name) == getattr(reference, name)
    assert default_model.normalization_min_lambda == reference.normalization_min_lambda
    assert default_model.normalization_max_lambda == reference.normalization_max_lambda


# --------------------------------------------------------------------------
# Legacy eBOSS/DR16Q asset (PI ruling N16)
# --------------------------------------------------------------------------

LEGACY_MODEL = "eboss_dr16q_minus_dr12q"


def test_legacy_model_is_bundled():
    assert LEGACY_MODEL in m.available_models()


def test_legacy_model_geometry():
    legacy = m.load_model(LEGACY_MODEL)
    assert legacy.rank == 20
    lo, hi = legacy.rest_wavelength_range
    assert (lo, hi) == pytest.approx((850.75, 1420.75), abs=1e-6)
    assert legacy.rest_wavelengths.size == 2281


def test_legacy_normalisation_band_is_stamped_not_claimed_as_extracted():
    """The value is a supplied convention, and the record must say so.

    The source artifact embeds no normalisation metadata. Provenance that
    implied otherwise would misrepresent where a scientifically consequential
    number came from.

    Asserted on the substance rather than on a decision reference: this record
    ships to users, and it has to be readable by someone with no access to the
    project's internal ledger.
    """
    prov = m.model_provenance(LEGACY_MODEL)
    norm = prov["model"]["normalization_provenance"]
    assert norm["embedded_in_source"] is False
    assert norm["stamped_at_conversion"] is True
    assert tuple(norm["value"]) == (1425.0, 1475.0)
    attributed = norm["attributed_to"]
    assert "conversion time" in attributed
    assert "NOT extracted" in attributed
    assert "NOT extracted" in norm["note"]
    assert prov["source"]["normalization_embedded"] is False


def test_deployed_model_band_is_embedded_not_stamped():
    """The converse, for the DESI model: its band really is in the file."""
    prov = m.model_provenance()
    norm = prov["model"]["normalization_provenance"]
    assert norm["embedded_in_source"] is True
    assert norm["stamped_at_conversion"] is False
    assert prov["source"]["normalization_embedded"] is True


def test_both_models_use_the_same_normalisation_band():
    """Records the N16 Option A resolution.

    An earlier ruling premised that these differed; the traced evidence showed
    they do not. This test exists so that premise cannot quietly return.
    """
    assert m.load_model().normalization_min_lambda == 1425.0
    assert m.load_model(LEGACY_MODEL).normalization_min_lambda == 1425.0
    assert m.load_model().normalization_max_lambda == 1475.0
    assert m.load_model(LEGACY_MODEL).normalization_max_lambda == 1475.0


def test_legacy_model_keeps_float64_where_float32_would_lose_precision():
    """The precision rule is per-array, and this artifact exercises the other branch.

    The DESI model is float32-exact throughout; the legacy MATLAB export is not,
    so its arrays must be stored at full precision rather than downcast.
    """
    audit = m.model_provenance(LEGACY_MODEL)["conversion"]["arrays"]
    assert audit["mu"]["float32_lossless"] is False
    assert audit["mu"]["stored_dtype"] == "float64"
    assert audit["M"]["stored_dtype"] == "float64"
