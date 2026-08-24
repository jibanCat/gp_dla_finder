"""Absorber-existence prior tests.

The compact table replaces ~115 MB of catalogues, so the load-bearing tests are
the ones showing it reproduces the reference counting *exactly* rather than
approximately.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from gp_dla_finder import prior as pr
from gp_dla_finder.config import Config

_CATALOG_ENV = "GP_DLA_FINDER_PRIOR_SOURCES"
DELTA = Config.kms_to_z(30000.0)


@pytest.fixture(scope="session")
def default_prior():
    return pr.load_prior()


# --------------------------------------------------------------------------
# Table integrity
# --------------------------------------------------------------------------


def test_default_prior_is_bundled():
    assert pr.DEFAULT_PRIOR in pr.available_priors()


def test_table_matches_its_recorded_provenance(default_prior):
    prov = pr.prior_provenance()
    assert default_prior.n_sightlines == prov["n_sightlines"]
    assert int(default_prior.cumulative_absorbers[-1]) == prov["n_with_absorber"]
    assert prov["equivalence_proof"]["result"] == "exact"


def test_provenance_records_both_public_sources_with_licences():
    sources = pr.prior_provenance()["sources"]
    assert len(sources) == 2
    for source in sources:
        assert len(source["sha256"]) == 64
        assert source["origin"].startswith("http")
        assert source["licence"]
        assert source["attribution"]


def test_table_is_sorted_and_cumulative(default_prior):
    assert np.all(np.diff(default_prior.z_qsos) >= 0)
    assert np.all(np.diff(default_prior.cumulative_absorbers) >= 0)
    assert default_prior.cumulative_absorbers[-1] <= default_prior.n_sightlines


def test_table_arrays_are_read_only(default_prior):
    with pytest.raises(ValueError, match="read-only"):
        default_prior.z_qsos[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        default_prior.cumulative_absorbers[0] = 0


def test_malformed_tables_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        pr.AbsorberPrior("x", np.arange(3.0), np.arange(2))
    with pytest.raises(ValueError, match="sorted ascending"):
        pr.AbsorberPrior("x", np.array([3.0, 1.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="non-decreasing"):
        pr.AbsorberPrior("x", np.array([1.0, 2.0]), np.array([5, 1]))
    with pytest.raises(ValueError, match="empty"):
        pr.AbsorberPrior("x", np.array([]), np.array([]))


def test_unknown_prior_lists_what_is_available():
    with pytest.raises(ValueError, match="unknown prior"):
        pr.load_prior("dr16q_made_up")


# --------------------------------------------------------------------------
# Counting behaviour
# --------------------------------------------------------------------------


def test_counts_are_monotone_in_redshift(default_prior):
    previous = (0, 0)
    for z in np.linspace(2.2, 5.4, 400):  # inside support: no warnings expected
        current = default_prior.counts(z, DELTA)
        assert current[0] >= previous[0]
        assert current[1] >= previous[1]
        previous = current


def test_counts_reproduce_a_brute_force_evaluation(default_prior):
    """The table against the definition it encodes, at breakpoints and between."""
    z_qsos = default_prior.z_qsos
    is_absorber = np.diff(default_prior.cumulative_absorbers, prepend=0) > 0

    probes = np.unique(
        np.concatenate(
            [
                np.linspace(2.0, 6.0, 5001),
                z_qsos[::97],
                np.nextafter(z_qsos[::97], np.inf),
                np.nextafter(z_qsos[::97], -np.inf),
            ]
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", UserWarning
        )  # out-of-support probes are intentional
        for z in probes:
            clamped = max(z, float(z_qsos[0]))
            mask = z_qsos < (clamped + DELTA)
            assert default_prior.counts(z, DELTA) == (
                int(is_absorber[mask].sum()),
                int(mask.sum()),
            )


def test_below_floor_clamps_but_warns_rather_than_substituting_silently(default_prior):
    """PI ruling N12: parity is retained at this level, but never silently."""
    floor = default_prior.z_qsos[0]
    with pytest.warns(UserWarning, match="below this prior's catalogue floor"):
        clamped = default_prior.counts(0.5, DELTA)
    assert clamped == default_prior.counts(floor, DELTA)


def test_above_the_catalogue_top_warns_too(default_prior):
    """Saturating above the highest catalogued sightline is also extrapolation."""
    with pytest.warns(UserWarning, match="above this prior's highest"):
        default_prior.counts(9.0, DELTA)


def test_supports_declares_the_valid_range(default_prior):
    lo, hi = default_prior.z_qso_range
    assert default_prior.supports(lo) and default_prior.supports(hi)
    assert not default_prior.supports(lo - 0.01)
    assert not default_prior.supports(hi + 0.01)


def test_external_prior_tables_are_validated():
    """Hardening required by the increment-2 ruling."""
    with pytest.raises(ValueError, match="finite"):
        pr.AbsorberPrior("x", np.array([1.0, np.nan]), np.array([0, 1]))
    with pytest.raises(ValueError, match="non-negative"):
        pr.AbsorberPrior("x", np.array([1.0, 2.0]), np.array([-1, 0]))
    with pytest.raises(ValueError, match="exceeds the number of sightlines"):
        pr.AbsorberPrior("x", np.array([1.0, 2.0]), np.array([2, 3]))


def test_absorber_fraction_is_a_probability(default_prior):
    for z in np.linspace(2.2, 5.4, 200):
        assert 0.0 < default_prior.absorber_fraction(z, DELTA) < 1.0


# --------------------------------------------------------------------------
# Prior formulas
# --------------------------------------------------------------------------


def test_log_priors_partition_the_probability_space(default_prior):
    """P(null) + sum_k P(k) = 1, with the top model absorbing the tail."""
    for z in (2.2, 3.0, 4.0, 5.0):
        null = np.exp(default_prior.log_prior_no_absorber(z, DELTA))
        ks = np.exp(default_prior.log_priors(z, 4, DELTA))
        assert null + ks.sum() == pytest.approx(1.0, abs=1e-12)


def test_log_priors_decrease_with_absorber_count(default_prior):
    ks = default_prior.log_priors(3.0, 4, DELTA)
    assert np.all(np.diff(ks) < 0)


def test_log_priors_follow_the_independence_construction(default_prior):
    z = 3.2
    fraction = default_prior.absorber_fraction(z, DELTA)
    expected = fraction ** np.arange(1, 5)
    for i in range(3):
        expected[i] = expected[i] - expected[i + 1]
    assert np.array_equal(default_prior.log_priors(z, 4, DELTA), np.log(expected))


def test_max_absorbers_must_be_positive(default_prior):
    with pytest.raises(ValueError, match="max_absorbers"):
        default_prior.log_priors(3.0, 0, DELTA)


# --------------------------------------------------------------------------
# Equivalence with the reference implementation
# --------------------------------------------------------------------------


@pytest.mark.needs_reference
@pytest.mark.needs_private_data
def test_matches_reference_prior_catalogue_exactly(default_prior, reference_repo):
    """The packaged table vs the reference ``PriorCatalog``, counting for counting.

    Set ``GP_DLA_FINDER_PRIOR_SOURCES`` to a directory holding ``catalog.mat``,
    ``los_catalog`` and ``dla_catalog``. The catalogues are public but are not
    redistributed with this package.
    """
    raw = os.environ.get(_CATALOG_ENV)
    if not raw:
        pytest.skip(f"set {_CATALOG_ENV} to a directory of source catalogues")
    root = Path(raw).expanduser()
    needed = ["catalog.mat", "los_catalog", "dla_catalog"]
    if not all((root / n).exists() for n in needed):
        pytest.skip(f"{root} must contain {needed}")

    pytest.importorskip("h5py")
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.set_parameters import Parameters

    params = Parameters()
    reference = PriorCatalog(
        params,
        str(root / "catalog.mat"),
        str(root / "los_catalog"),
        str(root / "dla_catalog"),
    )

    assert reference.z_qsos.size == default_prior.n_sightlines
    assert int(reference.dla_ind.sum()) == int(default_prior.cumulative_absorbers[-1])

    delta = params.prior_z_qso_increase
    probes = np.unique(
        np.concatenate(
            [
                np.linspace(2.0, 6.0, 20001),
                reference.z_qsos,
                np.nextafter(reference.z_qsos, np.inf),
            ]
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", UserWarning
        )  # parity is checked across the full sweep
        for z in probes:
            expected = reference.less_ind(z)
            assert default_prior.counts(z, delta) == (
                int(expected[0]),
                int(expected[1]),
            )
