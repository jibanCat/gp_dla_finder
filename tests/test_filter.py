"""The FILTER approximation, and what selecting it means.

FILTER is **opt-in**: no production preset selects it, and the v0.1 default
everywhere is the full configured grid. (An earlier ruling had it provisionally
on; that was reversed, and this module said otherwise long after.) These tests
cover the mechanism and the bookkeeping. The *scientific* comparison -- evidence,
posterior, classification differences against the adopted 100k full-grid
reference -- is ``tools/compare_filter.py``, which is retained so the numbers
can be re-measured rather than quoted.

The equivalence of ``mode="filter"`` with the reference's own FILTER=1 path is
established bitwise in ``test_reference_parity.py``; here we only pin the local
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import load_model, load_sample_grid
from gp_dla_finder.config import Config
from gp_dla_finder.gp.evidence import (
    assemble_model,
    coarse_scan_size,
    one_absorber_log_evidence,
)
from gp_dla_finder.gp.spectrum import prepare_spectrum
from synthetic import make_spectrum


@pytest.fixture(scope="module")
def pipeline():
    model = load_model()
    config = Config.desi_y3_fast()
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(make_spectrum(), model, config)
    return config, grid, prepared, assemble_model(prepared, model, config)


# --------------------------------------------------------------------------
# The coarse-scan size, which is not what "20x" suggests
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [
        (10_000, 5_000),  # the floor bites: 2x, not 20x
        (50_000, 5_000),  # still the floor: 10x
        (100_000, 5_000),  # num_samples // 20 == the floor exactly: 20x
    ],
)
def test_coarse_scan_size_follows_the_reference_formula(num_samples, expected):
    """``max(N // 20, 5000)``, verbatim from the reference.

    Worth pinning because the "reduces samples by a factor of ~100" comment in
    the reference is about the region-A mask, not about this; the actual
    one-absorber saving ranges from 2x to 20x depending on the operating point.
    """
    config = Config.desi_y3().replace(num_samples=num_samples)
    assert coarse_scan_size(config) == expected


def test_the_coarse_scan_never_exceeds_the_grid():
    config = Config.desi_y3().replace(num_samples=1_000)
    assert coarse_scan_size(config) == 1_000


# --------------------------------------------------------------------------
# Mode selection is explicit and recorded
# --------------------------------------------------------------------------


def test_no_production_preset_selects_filter():
    """PI ruling N60: the full-grid path is the v0.1 default, everywhere.

    This is the guard behind "no production preset silently selects FILTER". It
    walks every preset the class exposes rather than a hand-written list, so a
    preset added later cannot quietly opt its users into a screening
    approximation.
    """
    presets = [
        name
        for name in dir(Config)
        if name.startswith("desi_y3") or name.startswith("eboss")
    ]
    assert presets, "no presets discovered -- the naming convention changed"
    for name in presets:
        config = getattr(Config, name)()
        assert config.evidence_mode == "exact", f"{name} selects FILTER"
        assert config.filter_low_likelihood is False, f"{name} selects FILTER"


def test_the_bare_field_default_is_also_full_grid():
    """Not just the presets: the field itself, so a custom config is safe too."""
    assert Config(preset="custom").evidence_mode == "exact"


def test_filter_must_be_asked_for_by_name():
    config = Config.desi_y3(filter_low_likelihood=True)
    assert config.evidence_mode == "filter"
    # Opting in RELABELS the effective configuration: this is not the canonical
    # desi_y3 operating point and must not claim that name. The starting point
    # is still recorded, so provenance answers both questions.
    assert config.preset == "desi_y3+modified"
    assert config.base_preset == "desi_y3"
    assert config.is_modified
    # The classmethod override path and .replace() agree.
    assert config.digest == Config.desi_y3().replace(filter_low_likelihood=True).digest


def test_an_unknown_mode_is_rejected(pipeline):
    config, grid, prepared, assembled = pipeline
    with pytest.raises(ValueError, match="mode must be"):
        one_absorber_log_evidence(prepared, assembled, grid, config, mode="fast")


# --------------------------------------------------------------------------
# The estimator itself
# --------------------------------------------------------------------------


def test_filter_evaluates_a_prefix_and_says_which_samples_it_skipped(pipeline):
    config, grid, prepared, assembled = pipeline
    n_coarse = coarse_scan_size(config)

    _, exact_samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="exact", return_samples=True
    )
    _, filter_samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="filter", return_samples=True
    )

    # A prefix, not a resampling: the evaluated points are the same points.
    assert np.array_equal(exact_samples[:n_coarse], filter_samples[:n_coarse])
    # And the caller can see what was not computed, rather than getting a
    # silently shorter array or a zero-filled tail.
    assert np.all(np.isnan(filter_samples[n_coarse:]))
    assert np.count_nonzero(np.isfinite(filter_samples)) == n_coarse
    assert np.all(np.isfinite(exact_samples))


def test_filter_and_exact_disagree_by_a_real_amount(pipeline):
    """FILTER is an approximation, and the tests must not pretend otherwise."""
    config, grid, prepared, assembled = pipeline
    exact = one_absorber_log_evidence(prepared, assembled, grid, config, mode="exact")
    filtered = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="filter"
    )
    assert exact != filtered
    # Small enough to be an approximation rather than a different model, on this
    # spectrum. The corpus-wide statement lives in tools/compare_filter.py.
    assert abs(exact - filtered) < 1.0


def test_the_log_norm_stays_the_full_sample_count_under_filter(pipeline):
    """Subtle, and it is the reference's convention.

    Under FILTER only a prefix is averaged, but the ``log(N)`` round trip keeps
    using the *full* grid size. Getting this wrong shifts the evidence by
    ``log(N / n_coarse)`` -- 0.69 nat at the 10k operating point -- which is a
    detection-moving error, not a rounding one.
    """
    config, grid, prepared, assembled = pipeline
    n_coarse = coarse_scan_size(config)

    filtered, samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="filter", return_samples=True
    )

    evaluated = samples[:n_coarse]
    peak = np.nanmax(evaluated)
    log_norm = np.log(config.num_samples)
    expected = peak + np.log(np.nanmean(np.exp(evaluated - peak))) + log_norm
    assert filtered == pytest.approx(expected, rel=0, abs=0)

    wrong = peak + np.log(np.nanmean(np.exp(evaluated - peak))) + np.log(n_coarse)
    assert abs(filtered - wrong) == pytest.approx(np.log(config.num_samples / n_coarse))


def test_filter_is_deterministic(pipeline):
    config, grid, prepared, assembled = pipeline
    first = one_absorber_log_evidence(prepared, assembled, grid, config, mode="filter")
    second = one_absorber_log_evidence(prepared, assembled, grid, config, mode="filter")
    assert first == second


def test_the_mode_follows_the_configuration_not_a_hard_coded_default(pipeline):
    """``mode=None`` must follow the configuration, not a hard-coded default."""
    config, grid, prepared, assembled = pipeline
    assert config.evidence_mode == "exact"
    default = one_absorber_log_evidence(prepared, assembled, grid, config)
    explicit = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="exact"
    )
    assert default == explicit

    # And the opt-in direction: a config that asked for FILTER gets FILTER from
    # the same mode=None call. (Asserting that the exact config equals the exact
    # mode would now be tautological -- the field is already False.)
    filter_config = config.replace(filter_low_likelihood=True)
    assert filter_config.evidence_mode == "filter"
    assert one_absorber_log_evidence(
        prepared, assembled, grid, filter_config
    ) == one_absorber_log_evidence(prepared, assembled, grid, config, mode="filter")
    # The two modes are genuinely different numbers, so the check above has teeth.
    assert default != one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="filter"
    )


# --- raising num_samples does not improve a FILTER result --------------------
#
# The floor in ``max(num_samples // 20, 5000)`` means FILTER evaluates the same
# 5000-sample prefix at 10,000 samples and at 100,000. The exact estimator keeps
# converging over that range, so the *gap* grows with the operating point --
# which is why the measured classification flips went from 1/15 to 3/15 (see
# docs/filter.md). This test pins the mechanism behind that claim.


@pytest.mark.slow
def test_filter_ignores_a_larger_sample_budget():
    model = load_model()
    spectrum = make_spectrum()

    results = {}
    for num_samples, grid_name in (
        (10_000, "pw14_172_225_10000"),
        (100_000, "pw14_172_225_100000"),
    ):
        config = Config.desi_y3().replace(
            num_samples=num_samples, sample_grid=grid_name
        )
        prepared = prepare_spectrum(spectrum, model, config)
        assembled = assemble_model(prepared, model, config)
        grid = load_sample_grid(config.sample_grid)
        results[num_samples] = (
            one_absorber_log_evidence(prepared, assembled, grid, config, mode="filter"),
            coarse_scan_size(config),
        )

    (small, small_n), (large, large_n) = results[10_000], results[100_000]

    # The same prefix, whatever the budget.
    assert small_n == large_n == 5000

    # And therefore the same answer, to round-off. The measured worst case across
    # the fifteen-case corpus was 2.3e-13 nat; allow an order of magnitude.
    assert float(small) == pytest.approx(float(large), abs=1e-12)
