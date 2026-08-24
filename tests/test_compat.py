"""Compatibility profiles: the reference's two floating-point no-ops, isolated.

PI ruling (increment 7): the no-ops may be retained for the pinned bitwise gate,
but must be a *named, versioned* behaviour recorded in provenance, and a clean
arithmetic mode must be comparable rather than a silent replacement.

These tests do three things:

* pin the registry, so a profile's meaning cannot drift under a fixed name;
* prove each flag is load-bearing -- that switching it actually changes numbers,
  which is the only reason the flags exist;
* measure the size of the difference, so the cost of bug-compatibility is a
  recorded quantity rather than a belief.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import load_model, load_sample_grid
from gp_dla_finder.compat import (
    CLEAN,
    COMPATIBILITY_PROFILES,
    DEFAULT_COMPATIBILITY,
    REFERENCE_D5B306E6,
    compatibility_profile,
)
from gp_dla_finder.config import Config
from gp_dla_finder.gp.evidence import (
    assemble_model,
    null_log_evidence,
    one_absorber_log_evidence,
)
from gp_dla_finder.gp.spectrum import prepare_spectrum
from synthetic import make_spectrum


def run(compatibility: str):
    model = load_model()
    config = Config.desi_y3_fast().replace(compatibility=compatibility)
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(make_spectrum(), model, config)
    assembled = assemble_model(prepared, model, config)
    return (
        prepared,
        null_log_evidence(prepared, assembled),
        one_absorber_log_evidence(prepared, assembled, grid, config, mode="exact"),
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_the_default_profile_is_the_pinned_reference():
    assert DEFAULT_COMPATIBILITY == REFERENCE_D5B306E6.name == "reference-d5b306e6"
    assert Config.desi_y3().compatibility == DEFAULT_COMPATIBILITY
    assert Config.desi_y3_fast().compatibility == DEFAULT_COMPATIBILITY
    assert Config.desi_y3_refined().compatibility == DEFAULT_COMPATIBILITY


def test_the_reference_profile_pins_the_commit_it_was_measured_against():
    assert (
        REFERENCE_D5B306E6.reference_commit
        == "d5b306e6e2c8d89cdb38a6201b690557f2798f28"
    )
    assert REFERENCE_D5B306E6.rest_frame_round_trip
    assert REFERENCE_D5B306E6.log_norm_round_trip


def test_the_clean_profile_drops_both_no_ops_and_claims_no_reference():
    assert not CLEAN.rest_frame_round_trip
    assert not CLEAN.log_norm_round_trip
    assert CLEAN.reference_commit is None


def test_an_unknown_profile_raises_rather_than_falling_back():
    with pytest.raises(KeyError, match="unknown compatibility profile"):
        compatibility_profile("reference-d5b306e7")
    with pytest.raises(KeyError, match="unknown compatibility profile"):
        Config.desi_y3_fast().replace(compatibility="nope")


def test_the_registry_is_read_only():
    with pytest.raises(TypeError):
        COMPATIBILITY_PROFILES["clean"] = REFERENCE_D5B306E6


def test_provenance_is_flat_and_names_both_flags():
    record = dict(REFERENCE_D5B306E6.provenance())
    assert record["compatibility_profile"] == "reference-d5b306e6"
    assert record["compatibility_version"] == "1"
    assert record["rest_frame_round_trip"] is True
    assert record["log_norm_round_trip"] is True
    assert record["compatibility_reference_commit"].startswith("d5b306e6")


# --------------------------------------------------------------------------
# The flags are load-bearing
# --------------------------------------------------------------------------


def test_the_rest_frame_round_trip_actually_moves_wavelengths():
    """If the round trip were the identity there would be nothing to isolate."""
    reference_prepared, _, _ = run("reference-d5b306e6")
    clean_prepared, _, _ = run("clean")

    assert not np.array_equal(
        reference_prepared.wavelength, clean_prepared.wavelength
    ), "the round trip changed nothing; the compatibility flag would be pointless"

    shift = np.max(np.abs(reference_prepared.wavelength - clean_prepared.wavelength))
    # Tiny, and not zero. Recorded so the claim is a measurement.
    assert 0.0 < shift < 1e-11


def test_clean_arithmetic_changes_the_evidences_but_not_the_science():
    """Measured, not asserted: how much bug-compatibility is worth in nats."""
    _, reference_null, reference_one = run("reference-d5b306e6")
    _, clean_null, clean_one = run("clean")

    assert reference_null != clean_null or reference_one != clean_one

    # The difference is a floating-point artefact, not a modelling difference:
    # it must be far below anything that could move a detection.
    assert abs(reference_null - clean_null) < 1e-8
    assert abs(reference_one - clean_one) < 1e-8

    reference_bayes = reference_one - reference_null
    clean_bayes = clean_one - clean_null
    assert abs(reference_bayes - clean_bayes) < 1e-8


def test_switching_profiles_is_explicit_in_the_config_label():
    config = Config.desi_y3_fast().replace(compatibility="clean")
    assert config.compatibility_profile is CLEAN
    # `replace` marks the preset as modified, so a result can never claim to be a
    # plain named preset while running non-default arithmetic.
    assert config.preset != "desi_y3_fast"
