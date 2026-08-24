"""GP likelihood primitives.

Three kinds of check:

* **bitwise reference equivalence** — the port must not move the numerics;
* **independent verification** — the Woodbury log-density is checked against a
  dense ``multivariate_normal.logpdf``, so a shared error in the fast path and
  the reference would still be caught;
* **properties** — physical behaviour that should hold regardless.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gp_dla_finder.gp.likelihood import effective_optical_depth, log_mvnpdf_low_rank

DESI_GRID = np.arange(3600.0, 6000.0, 0.8)

# (tau_0, beta): Turner+2024 production, Kamble+2020 legacy, and the values the
# deployed model itself learned.
TAU_BETA = [(0.00246, 3.62), (0.00554, 3.182), (0.0017514521, 2.9694497)]


def _random_low_rank(rng, n, k):
    return (
        rng.normal(size=n),
        rng.normal(size=n),
        rng.normal(size=(n, k)) * 0.1,
        rng.uniform(0.05, 2.0, size=n),
    )


# --------------------------------------------------------------------------
# effective_optical_depth
# --------------------------------------------------------------------------


def test_optical_depth_shape_and_finiteness():
    tau = effective_optical_depth(DESI_GRID, 3.62, 0.00246, 2.6, 31)
    assert tau.shape == (DESI_GRID.size, 31)
    assert np.all(np.isfinite(tau))
    assert np.all(tau >= 0.0)


def test_absorbers_beyond_the_quasar_contribute_nothing():
    """The indicator is what keeps exp(-sum) finite redward of the forest."""
    z_qso = 2.6
    tau = effective_optical_depth(DESI_GRID, 3.62, 0.00246, z_qso, 1)
    lya = 1215.6701
    beyond = DESI_GRID > lya * (1 + z_qso)
    assert np.all(tau[beyond, 0] == 0.0)
    assert np.any(tau[~beyond, 0] > 0.0)


def test_optical_depth_grows_with_redshift_within_the_forest():
    z_qso = 3.5
    tau = effective_optical_depth(DESI_GRID, 3.62, 0.00246, z_qso, 1)[:, 0]
    inside = tau > 0
    assert np.all(np.diff(tau[inside]) > 0)


def test_optical_depth_scales_linearly_with_tau_0():
    a = effective_optical_depth(DESI_GRID, 3.62, 0.00246, 2.6, 3)
    b = effective_optical_depth(DESI_GRID, 3.62, 0.00492, 2.6, 3)
    assert np.allclose(b, 2.0 * a, rtol=0, atol=0)


def test_more_forest_lines_only_add_columns_not_change_existing_ones():
    three = effective_optical_depth(DESI_GRID, 3.62, 0.00246, 3.0, 3)
    thirty_one = effective_optical_depth(DESI_GRID, 3.62, 0.00246, 3.0, 31)
    assert np.array_equal(thirty_one[:, :3], three)


@pytest.mark.parametrize("num_forest_lines", [0, -1, 32])
def test_optical_depth_rejects_out_of_range_line_counts(num_forest_lines):
    with pytest.raises(ValueError, match="num_forest_lines"):
        effective_optical_depth(DESI_GRID, 3.62, 0.00246, 2.6, num_forest_lines)


# --------------------------------------------------------------------------
# log_mvnpdf_low_rank
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "k"), [(50, 3), (200, 5), (1263, 30)])
def test_woodbury_matches_a_dense_multivariate_normal(n, k):
    """Independent check: the fast path against the textbook formula.

    A shared mistake in this port and the reference would survive the bitwise
    test below but not this one.
    """
    rng = np.random.default_rng(11)
    y, mu, M, d = _random_low_rank(rng, n, k)
    dense = multivariate_normal.logpdf(y, mean=mu, cov=M @ M.T + np.diag(d))
    assert log_mvnpdf_low_rank(y, mu, M, d) == pytest.approx(dense, rel=1e-10)


def test_density_is_maximised_at_the_mean():
    rng = np.random.default_rng(3)
    _, mu, M, d = _random_low_rank(rng, 120, 4)
    at_mean = log_mvnpdf_low_rank(mu, mu, M, d)
    for _ in range(5):
        offset = mu + rng.normal(size=mu.size) * 0.3
        assert log_mvnpdf_low_rank(offset, mu, M, d) < at_mean


def test_larger_noise_lowers_the_peak_density():
    rng = np.random.default_rng(4)
    _, mu, M, d = _random_low_rank(rng, 120, 4)
    assert log_mvnpdf_low_rank(mu, mu, M, 4.0 * d) < log_mvnpdf_low_rank(mu, mu, M, d)


def test_zero_rank_reduces_to_independent_gaussians():
    """With no low-rank term the answer is a plain diagonal Gaussian."""
    rng = np.random.default_rng(5)
    n = 64
    y, mu = rng.normal(size=n), rng.normal(size=n)
    d = rng.uniform(0.2, 1.5, size=n)
    expected = -0.5 * np.sum((y - mu) ** 2 / d + np.log(d) + np.log(2 * np.pi))
    assert log_mvnpdf_low_rank(y, mu, np.zeros((n, 1)), d) == pytest.approx(
        expected, rel=1e-12
    )


def test_result_is_deterministic():
    rng = np.random.default_rng(6)
    args = _random_low_rank(rng, 300, 10)
    assert log_mvnpdf_low_rank(*args) == log_mvnpdf_low_rank(*args)


# --------------------------------------------------------------------------
# Bitwise equivalence with the reference implementation
# --------------------------------------------------------------------------


@pytest.mark.needs_reference
@pytest.mark.parametrize("z_qso", [2.15, 2.6, 3.4, 4.5])
@pytest.mark.parametrize("num_forest_lines", [1, 3, 31])
@pytest.mark.parametrize(("tau_0", "beta"), TAU_BETA)
def test_optical_depth_is_bit_identical_to_reference(
    reference_repo, z_qso, num_forest_lines, tau_0, beta
):
    from gpy_dla_detection.effective_optical_depth import (
        effective_optical_depth as reference,
    )

    expected = reference(DESI_GRID, beta, tau_0, z_qso, num_forest_lines)
    actual = effective_optical_depth(DESI_GRID, beta, tau_0, z_qso, num_forest_lines)
    assert np.array_equal(actual, expected)


@pytest.mark.needs_reference
@pytest.mark.parametrize(("n", "k"), [(50, 3), (1125, 20), (1263, 30), (2000, 30)])
@pytest.mark.parametrize("trial", range(3))
def test_woodbury_is_bit_identical_to_reference(reference_repo, n, k, trial):
    from gpy_dla_detection.null_gp import NullGP

    rng = np.random.default_rng(100 + trial)
    y, mu, M, d = _random_low_rank(rng, n, k)
    assert log_mvnpdf_low_rank(y, mu, M, d) == NullGP.log_mvnpdf_low_rank(y, mu, M, d)
