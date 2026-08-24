"""QMC sample-grid tests.

Two things must hold and be kept apart: the generator reproduces the reference
*algorithm* exactly, and the packaged grid's identity with the *deployed
production arrays* is unverified. Conflating those is the failure mode PI ruling
N11 guards against.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import samples as sm

DEFAULT = sm.DEFAULT_SAMPLE_GRID


@pytest.fixture(scope="session")
def grid():
    return sm.load_sample_grid()


# --------------------------------------------------------------------------
# Asset integrity
# --------------------------------------------------------------------------


def test_default_grid_is_bundled():
    assert DEFAULT in sm.available_sample_grids()


def test_grid_matches_the_production_operating_point(grid):
    assert grid.num_samples == 50_000
    assert grid.declared_support == (17.2, 22.5)
    lo, hi = grid.log_nhi_range
    assert 17.2 <= lo and hi <= 22.5


def test_nhi_samples_are_stored_and_consistent_with_log_nhi(grid):
    """Consistency at tolerance, not bitwise, and deliberately so.

    ``nhi_samples`` is carried in the asset rather than recomputed on load,
    because float64 ``**`` goes through the platform libm ``pow``, which is not
    correctly rounded: the same asset yielded different last bits on macOS/arm64
    and Linux/x86-64, which broke a checksum test in CI. Asserting bitwise
    equality here would reintroduce exactly that platform dependence.
    """
    assert np.allclose(grid.nhi_samples, 10.0**grid.log_nhi_samples, rtol=1e-15)


def test_stored_nhi_samples_are_preserved_verbatim():
    """Supplied values must be kept, not silently recomputed.

    The perturbation is a single ulp — inside the consistency tolerance, so it is
    accepted, and large enough that recomputation would be detectable. An earlier
    version of this test used a 1.5x sentinel, which proved verbatim preservation
    only by accepting a grid whose linear and log column densities disagreed by
    50 per cent.
    """
    log_nhi = np.linspace(20.0, 21.0, 8)
    supplied = np.nextafter(10.0**log_nhi, np.inf)
    assert not np.array_equal(supplied, 10.0**log_nhi)  # genuinely different bits

    built = sm.AbsorberSampleGrid(
        "t", np.linspace(0, 0.9, 8), log_nhi, nhi_samples=supplied
    )
    assert np.array_equal(built.nhi_samples, supplied)


def test_nhi_samples_shape_is_validated():
    with pytest.raises(ValueError, match="nhi_samples"):
        sm.AbsorberSampleGrid(
            "t",
            np.linspace(0, 0.9, 8),
            np.linspace(20.0, 21.0, 8),
            nhi_samples=np.zeros(3),
        )


def test_nhi_samples_inconsistent_with_log_nhi_are_rejected():
    """Shape agreement is not enough: the two arrays must mean the same thing."""
    log_nhi = np.linspace(20.0, 21.0, 8)
    with pytest.raises(ValueError, match="not consistent with"):
        sm.AbsorberSampleGrid(
            "t", np.linspace(0, 0.9, 8), log_nhi, nhi_samples=(10.0**log_nhi) * 1.5
        )


@pytest.mark.parametrize(
    ("nhi", "match"),
    [
        (np.array([1e20, np.nan, 1e20, 1e20]), "non-finite"),
        (np.array([1e20, np.inf, 1e20, 1e20]), "non-finite"),
        (np.array([1e20, 0.0, 1e20, 1e20]), "strictly positive"),
        (np.array([1e20, -1e20, 1e20, 1e20]), "strictly positive"),
    ],
)
def test_nhi_samples_must_be_finite_and_positive(nhi, match):
    with pytest.raises(ValueError, match=match):
        sm.AbsorberSampleGrid(
            "t", np.linspace(0, 0.9, 4), np.full(4, 20.0), nhi_samples=nhi
        )


def test_the_bundled_grids_satisfy_the_consistency_contract():
    """The tolerance must actually admit the shipped assets."""
    for name in sm.available_sample_grids():
        grid = sm.load_sample_grid(name)
        assert np.allclose(
            grid.nhi_samples,
            10.0**grid.log_nhi_samples,
            rtol=sm.NHI_CONSISTENCY_RTOL,
            atol=0.0,
        )


def test_offsets_span_the_unit_interval(grid):
    assert grid.offset_samples.min() >= 0.0
    assert grid.offset_samples.max() < 1.0
    # A QMC sequence should cover the interval far more evenly than random draws.
    counts, _ = np.histogram(grid.offset_samples, bins=100, range=(0.0, 1.0))
    assert counts.min() > 0
    assert counts.std() / counts.mean() < 0.05


def test_arrays_are_immutable(grid):
    for name in ("offset_samples", "log_nhi_samples", "nhi_samples"):
        array = getattr(grid, name)
        with pytest.raises(ValueError):
            array[0] = 0.0
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)


def test_malformed_grids_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        sm.AbsorberSampleGrid("x", np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="empty"):
        sm.AbsorberSampleGrid("x", np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="non-finite"):
        sm.AbsorberSampleGrid("x", np.array([0.5, np.nan]), np.zeros(2))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        sm.AbsorberSampleGrid("x", np.array([0.5, 1.0]), np.zeros(2))


def test_unknown_grid_lists_what_is_available():
    with pytest.raises(ValueError, match="unknown sample grid"):
        sm.load_sample_grid("pw14_made_up")


# --------------------------------------------------------------------------
# Provenance and identity honesty (PI ruling N11)
# --------------------------------------------------------------------------


def test_packaged_grid_declares_itself_unverified(grid):
    """The grid reproduces the algorithm, not the deployed arrays. It must say so."""
    assert grid.is_verified is False
    assert grid.identity_status == "regenerated, production-array identity unverified"


def test_identity_record_names_the_target_and_explains_the_gap():
    identity = sm.sample_grid_provenance()["identity"]
    assert identity["production_target"] == "pw_samples_a3_172_225_50000.mat"
    assert "numpy.random.seed(42)" in identity["explanation"]
    assert "NOT established" in identity["explanation"]


def test_provenance_records_everything_needed_to_regenerate():
    """N11 point 3 enumerates these; a missing one is a reproducibility hole."""
    prov = sm.sample_grid_provenance()
    prior, qmc, env = prov["prior"], prov["qmc"], prov["environment"]

    assert prior["support_log_nhi"] == (17.2, 22.5)
    assert prior["mixture_weight_pw14"] == 0.97
    assert prior["mixture_weight_uniform"] == pytest.approx(0.03)
    assert len(prior["spline_nodes_log_nhi"]) == len(prior["spline_values_log_f"]) == 8
    assert "f(N) * N * ln(10)" in prior["transformation_to_log_nhi"]
    assert prior["cdf_grid_points"] == 50_000

    assert qmc["engine"] == "scipy.stats.qmc.Halton"
    assert qmc["dimensions"] == 2
    assert qmc["dimension_order"] == ("log_nhi", "z_offset")
    assert qmc["scramble"] is True
    assert qmc["seed"] == 42
    assert qmc["num_samples"] == 50_000

    assert env["scipy"] and env["numpy"]
    for key in ("offset_samples", "log_nhi_samples", "nhi_samples"):
        assert len(prov["arrays"][key]["sha256_float64"]) == 64


def test_packaged_arrays_match_their_recorded_checksums(grid):
    import hashlib

    for name in ("offset_samples", "log_nhi_samples", "nhi_samples"):
        digest = hashlib.sha256(
            np.ascontiguousarray(getattr(grid, name), dtype=np.float64).tobytes()
        ).hexdigest()
        assert digest == sm.sample_grid_provenance()["arrays"][name]["sha256_float64"]


# --------------------------------------------------------------------------
# Redshift mapping
# --------------------------------------------------------------------------


def test_sample_redshifts_stretch_offsets_onto_the_window(grid):
    z = grid.sample_redshifts(2.0, 3.0)
    assert z.min() >= 2.0
    assert z.max() < 3.0
    assert np.allclose(z, 2.0 + grid.offset_samples)


def test_sample_redshifts_matches_the_reference_formula(grid):
    z_min, z_max = 1.9613, 2.5900
    assert np.array_equal(
        grid.sample_redshifts(z_min, z_max),
        z_min + (z_max - z_min) * grid.offset_samples,
    )


# --------------------------------------------------------------------------
# Generator equivalence with the reference
# --------------------------------------------------------------------------


@pytest.mark.needs_reference
@pytest.mark.parametrize(
    ("num_samples", "log_nhi"),
    [(5_000, (17.2, 22.5)), (10_000, (20.3, 23.0))],
)
def test_generator_is_bit_identical_to_the_reference(
    reference_repo, num_samples, log_nhi
):
    """The algorithm claim, at bitwise tolerance.

    This does NOT establish identity with the deployed production arrays; see
    ``test_packaged_grid_declares_itself_unverified``.
    """
    from gpy_dla_detection.generate_samples import generate_pw14_samples
    from tools.build_sample_grid import generate

    expected = generate_pw14_samples(
        num_samples=num_samples,
        min_log_nhi=log_nhi[0],
        max_log_nhi=log_nhi[1],
        alpha=0.97,
        seed=42,
    )
    actual = generate(num_samples, log_nhi, alpha=0.97, seed=42)
    for key in ("offset_samples", "log_nhi_samples", "nhi_samples"):
        assert np.array_equal(actual[key], expected[key]), key


@pytest.mark.needs_reference
def test_packaged_grid_reproduces_the_reference_generator(grid, reference_repo):
    """The bundled 50k asset itself, regenerated from the reference and compared."""
    from gpy_dla_detection.generate_samples import generate_pw14_samples

    expected = generate_pw14_samples(
        num_samples=50_000, min_log_nhi=17.2, max_log_nhi=22.5, alpha=0.97, seed=42
    )
    assert np.array_equal(grid.offset_samples, expected["offset_samples"])
    assert np.array_equal(grid.log_nhi_samples, expected["log_nhi_samples"])


@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "unknown",
        "unverified",
        "regenerated, production-array identity unverified",
        "Regenerated",  # capitalisation
        "verified",  # not on the allow-list
        "verified: something else entirely",
        "VERIFIED: byte-identical to the deployed production arrays",  # case
    ],
)
def test_verification_fails_closed(status):
    """Anything not on the exact allow-list is unverified (PI ruling N25).

    The earlier implementation asked whether the status did *not* start with
    "regenerated", so a typo or any new status string read as verified.
    """
    provenance = {} if status is None else {"identity": {"status": status}}
    grid = sm.AbsorberSampleGrid(
        "t", np.linspace(0, 0.9, 8), np.linspace(20.0, 21.0, 8), provenance=provenance
    )
    assert grid.is_verified is False


@pytest.mark.parametrize("status", sorted(sm.VERIFIED_IDENTITY_STATUSES))
def test_allow_listed_statuses_are_verified(status):
    grid = sm.AbsorberSampleGrid(
        "t",
        np.linspace(0, 0.9, 8),
        np.linspace(20.0, 21.0, 8),
        provenance={"identity": {"status": status}},
    )
    assert grid.is_verified is True


def test_missing_identity_block_is_unverified():
    grid = sm.AbsorberSampleGrid(
        "t", np.linspace(0, 0.9, 8), np.linspace(20.0, 21.0, 8)
    )
    assert grid.is_verified is False
    assert grid.identity_status == "unknown"


# --------------------------------------------------------------------------
# Named operating points (PI ruling N23)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grid_name", "expected_n"),
    [
        ("pw14_172_225_10000", 10_000),
        ("pw14_172_225_50000", 50_000),
        ("pw14_172_225_100000", 100_000),
    ],
)
def test_each_named_grid_is_bundled_with_its_sample_count(grid_name, expected_n):
    grid = sm.load_sample_grid(grid_name)
    assert grid.num_samples == expected_n
    assert grid.declared_support == (17.2, 22.5)


def test_every_bundled_grid_is_identity_unverified():
    """None has been compared against deployed arrays; 'refined' changes nothing."""
    for name in sm.available_sample_grids():
        assert sm.load_sample_grid(name).is_verified is False


@pytest.mark.parametrize(
    ("smaller", "larger"),
    [
        ("pw14_172_225_10000", "pw14_172_225_50000"),
        ("pw14_172_225_50000", "pw14_172_225_100000"),
    ],
)
@pytest.mark.parametrize("array", ["offset_samples", "log_nhi_samples", "nhi_samples"])
def test_smaller_grid_is_an_exact_prefix_of_the_larger(smaller, larger, array):
    """PI ruling N27/N32: prove the prefix relation directly, array by array.

    This underwrites the N28 custom-prefix rule — a prefix of a bundled grid is
    the same low-discrepancy sequence, not a resampling of it — so it has to be
    checked rather than assumed.

    An earlier revision claimed this test existed when it did not: a scripted
    string replacement silently failed to apply and the weaker head-versus-tail
    test remained in its place.
    """
    small = sm.load_sample_grid(smaller)
    big = sm.load_sample_grid(larger)
    n = small.num_samples
    assert big.num_samples > n
    assert np.array_equal(getattr(big, array)[:n], getattr(small, array))


def test_larger_grid_genuinely_extends_the_sequence():
    """Sharing a prefix must not mean the tail is a repeat of it."""
    small = sm.load_sample_grid("pw14_172_225_10000")
    large = sm.load_sample_grid("pw14_172_225_50000")
    n = small.num_samples
    assert not np.array_equal(large.offset_samples[:n], large.offset_samples[-n:])


def test_provenance_must_be_passed_by_keyword():
    """Guards the transposition that made a provenance mapping look like an array."""
    with pytest.raises(TypeError):
        sm.AbsorberSampleGrid(
            "t", np.linspace(0, 0.9, 8), np.linspace(20.0, 21.0, 8), {"identity": {}}
        )


def test_recorded_checksums_cover_only_stored_arrays():
    """Regression guard for the cross-platform failure this asset design fixes.

    Every array whose checksum is recorded must be one the asset actually stores.
    Checksumming a load-time-derived array made the test platform-dependent and
    broke Linux CI while passing on macOS.
    """
    from importlib import resources

    recorded = set(sm.sample_grid_provenance()["arrays"])
    handle = resources.files("gp_dla_finder.data.samples") / f"{DEFAULT}.npz"
    with resources.as_file(handle) as path, np.load(path) as data:
        stored = set(data.files)
    assert recorded <= stored, f"checksummed but not stored: {recorded - stored}"
