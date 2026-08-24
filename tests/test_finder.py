"""The Finder -> Result -> catalogue vertical slice (PI ruling N59).

This exercises **real package code** from a spectrum through a typed result into
a readable FITS product. It is deliberately bounded — null versus one absorber,
no multi-absorber path, no point estimator — and the tests assert the boundaries
as firmly as the behaviour, so the slice cannot be mistaken for the finished
product.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gp_dla_finder.config import Config
from gp_dla_finder.finder import (
    AbsorberCandidate,
    Finder,
    Result,
    results_to_catalogue,
    screening_score,
)
from gp_dla_finder.gp.spectrum import Spectrum
from synthetic import CORPUS, build, make_spectrum

fits = pytest.importorskip("astropy.io.fits", reason="catalogue I/O needs astropy")
from astropy.table import Table  # noqa: E402

from gp_dla_finder.io.fits import (  # noqa: E402
    read_catalogue_metadata,
    write_catalogue,
    write_legacy_catalogue,
)

BY_NAME = {case.name: case for case in CORPUS}


@pytest.fixture(scope="module")
def finder():
    # The fast preset: this is a correctness test, not a production run.
    # enable_tau_eb=False states the omission: this slice does not implement
    # per-spectrum empirical-Bayes mean-flux fitting, and Finder refuses a
    # configuration that asks for it rather than skipping it silently. The
    # override relabels the preset to desi_y3_fast+modified, which is the point.
    # max_absorbers=1: this module tests the null-versus-one path and the
    # catalogue, which cannot represent a selected M2 result. The multi-absorber
    # ladder has its own fixture in test_multi.py.
    return Finder(
        Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=1),
        warn_about_threads=False,
    )


# --------------------------------------------------------------------------
# One spectrum in, one typed result out
# --------------------------------------------------------------------------


def test_a_clean_spectrum_completes_with_a_low_posterior(finder):
    result = finder.run(build(BY_NAME["absorber-free-desi-grid"]), targetid=1)
    assert result.status == "completed"
    assert result.reason == ""
    assert np.isfinite(result.log_evidence_null)
    assert np.isfinite(result.log_evidence_absorber)
    assert 0.0 <= result.p_absorber < 0.2
    assert result.p_absorber + result.p_null == pytest.approx(1.0)
    assert not result.detected(0.98)


def test_a_strong_absorber_is_detected(finder):
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=2)
    assert result.status == "completed"
    assert result.p_absorber > 0.99
    assert result.detected(0.98)
    assert result.log_bayes_factor > 0
    assert len(result.absorber_candidates) == 1


def test_the_absorber_reports_a_grid_point_not_a_map_estimate(finder):
    """The names say `grid`, and the uncertainties stay NaN.

    No point estimator has been chosen or validated, so a field called
    `z_abs` would be a claim this package cannot support.
    """
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=3)
    absorber = result.absorber_candidates[0]
    assert isinstance(absorber, AbsorberCandidate)
    assert hasattr(absorber, "grid_z_abs") and not hasattr(absorber, "z_abs")
    assert 2.0 < absorber.grid_z_abs < result.z_qso
    assert 17.0 < absorber.grid_log_nhi < 23.0
    assert np.isnan(absorber.z_abs_err)
    assert np.isnan(absorber.log_nhi_err)


# --------------------------------------------------------------------------
# Failure states are results, not exceptions
# --------------------------------------------------------------------------


def test_a_fully_masked_spectrum_is_insufficient_data(finder):
    wave = np.arange(3600.0, 5600.0, 0.8)
    spectrum = Spectrum(
        wavelength=wave,
        flux=np.ones_like(wave),
        ivar=np.zeros_like(wave),  # everything masked
        z_qso=2.6,
    )
    result = finder.run(spectrum, targetid=4)
    assert result.status in {"insufficient_data", "quality_rejected"}
    assert result.reason
    # And it carries no inference results, so it cannot be read as a detection.
    assert np.isnan(result.p_absorber)
    assert np.isnan(result.log_evidence_null)
    assert result.absorber_candidates == ()
    assert not result.detected(0.5)


def test_a_quality_rejection_is_reported_as_a_decision(finder):
    """Not a failure, and not a non-detection."""
    wave = np.arange(3600.0, 5600.0, 0.8)
    rest = wave / (1 + 2.6)
    mask = (rest > 900.0) & (rest < 1230.0)
    # Mask 95% of the policy window, leaving the normalisation band intact.
    indices = np.flatnonzero(mask)
    blocked = np.zeros_like(wave, dtype=bool)
    blocked[indices[: int(0.95 * indices.size)]] = True
    spectrum = Spectrum(
        wavelength=wave,
        flux=np.ones_like(wave),
        ivar=np.full_like(wave, 25.0),
        z_qso=2.6,
        mask=blocked,
    )
    result = finder.run(spectrum, targetid=5)
    assert result.status == "quality_rejected"
    assert result.reason == "quality_policy_rejected"
    assert result.quality_fraction < 0.2
    assert np.isnan(result.p_absorber)


def test_a_run_never_raises_for_an_ordinary_bad_spectrum(finder):
    """A batch layer must be able to record and continue."""
    wave = np.arange(3600.0, 4000.0, 0.8)  # no normalisation coverage at z=2.6
    spectrum = Spectrum(
        wavelength=wave,
        flux=np.ones_like(wave),
        ivar=np.ones_like(wave),
        z_qso=2.6,
    )
    result = finder.run(spectrum, targetid=6)  # must not raise
    assert result.status != "completed"
    assert result.reason


# --------------------------------------------------------------------------
# Provenance and immutability
# --------------------------------------------------------------------------


def test_the_result_carries_the_provenance_a_reader_needs(finder):
    result = finder.run(make_spectrum(), targetid=7)
    provenance = dict(result.provenance)
    for key in (
        "gpdlf_version",
        "preset",
        "evidence_mode",
        "num_samples",
        "n_evaluated",
        "sample_grid",
        "model",
        "prior",
        "lsf_kernel",
        "quality_policy",
        "backend_backend",
        "compatibility_profile",
    ):
        assert key in provenance, f"provenance is missing {key!r}"
    assert provenance["evidence_mode"] == result.evidence_mode


def test_a_result_cannot_be_mutated_after_construction():
    result = Result(targetid=1, z_qso=2.6, status="completed", evidence_mode="exact")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "failed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.provenance["preset"] = "tampered"  # type: ignore[index]
    assert result.absorber_candidates == ()


def test_the_evaluated_count_reflects_the_mode(finder):
    """FILTER evaluates a prefix, and the result says how many."""
    result = finder.run(make_spectrum(), targetid=8)
    if result.evidence_mode == "filter":
        assert result.n_evaluated == 5000
        assert result.n_evaluated < finder.config.num_samples
    else:
        assert result.n_evaluated == finder.config.num_samples


# --------------------------------------------------------------------------
# End to end: results -> catalogue -> FITS -> read back
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def end_to_end(finder, tmp_path_factory):
    """One of each outcome, through the writers, read back with astropy."""
    directory = tmp_path_factory.mktemp("catalogue")
    wave = np.arange(3600.0, 5600.0, 0.8)

    results = [
        finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=101),
        finder.run(build(BY_NAME["absorber-free-desi-grid"]), targetid=102),
        finder.run(
            Spectrum(
                wavelength=wave,
                flux=np.ones_like(wave),
                ivar=np.zeros_like(wave),
                z_qso=2.6,
            ),
            targetid=103,
        ),
    ]
    catalogue = results_to_catalogue(results, detection_threshold=0.98)
    extended = write_catalogue(directory / "extended.fits", catalogue)
    legacy = write_legacy_catalogue(directory / "legacy.fits", catalogue)
    return results, catalogue, extended, legacy


def test_every_attempted_spectrum_reaches_the_catalogue(end_to_end):
    results, _, extended, _ = end_to_end
    table = Table.read(extended, hdu="SPECTRA")
    assert len(table) == len(results)
    assert set(table["TARGETID"]) == {101, 102, 103}


def test_only_detections_produce_absorber_rows(end_to_end):
    """The clean and unusable spectra appear in SPECTRA and nowhere else."""
    _, _, extended, _ = end_to_end
    absorbers = Table.read(extended, hdu="DLACAT")
    assert set(absorbers["TARGETID"]) == {101}
    assert list(absorbers["DLAID"]) == ["101000"]


def test_the_statuses_survive_to_the_file(end_to_end):
    _, _, extended, _ = end_to_end
    table = Table.read(extended, hdu="SPECTRA")
    by_target = {int(row["TARGETID"]): row for row in table}
    assert by_target[101]["GPDLF_STATUS"] == "completed"
    assert by_target[101]["GPDLF_N_ABSORBERS"] == 1
    assert by_target[102]["GPDLF_STATUS"] == "completed"
    assert by_target[102]["GPDLF_N_ABSORBERS"] == 0
    assert by_target[103]["GPDLF_STATUS"] != "completed"
    assert by_target[103]["GPDLF_REASON"]


def test_the_run_provenance_reaches_the_file(end_to_end):
    _, _, extended, _ = end_to_end
    run = dict(read_catalogue_metadata(extended))
    # The effective identity says it is modified, because it is: the fixture
    # opts out of the mean-flux fit. The base preset and digest say what it was
    # modified FROM and what exactly ran.
    assert run["GPDLF_PRESET"] == "desi_y3_fast+modified"
    assert run["GPDLF_BASE_PRESET"] == "desi_y3_fast"
    assert len(run["GPDLF_CONFIG_DIGEST"]) == 16
    assert run["GPDLF_MODEL"]
    assert run["GPDLF_EVIDENCE_MODE"] in {"exact", "filter"}
    assert run["GPDLF_COMPAT_PROFILE"] == "reference-d5b306e6"


def test_the_legacy_product_is_readable_and_loses_the_null_spectra(end_to_end):
    _, _, _, legacy = end_to_end
    table = Table.read(legacy, hdu="DLACAT")
    assert set(table["TARGETID"]) == {101}
    # The documented limitation, exercised end to end rather than in the abstract.
    assert len(table) == 1


def test_evidence_values_carry_their_mode_to_the_file(end_to_end):
    _, _, extended, _ = end_to_end
    absorbers = Table.read(extended, hdu="DLACAT")
    for row in absorbers:
        assert row["GPDLF_EVIDENCE_MODE"] in {"exact", "filter"}
    with fits.open(extended) as hdul:
        assert hdul["DLACAT"].header["GPDLFMOD"] in {"exact", "filter"}


def test_mixing_modes_in_one_catalogue_warns_and_is_labelled(finder):
    """A run-level mode cannot describe a mixed file; the row labels rule."""
    # Mixing evidence modes across spectra is the one permitted mixture, but
    # under N64 a config that opted into FILTER is relabelled -- so a genuine
    # mixed-mode run also differs in `preset`, and results_to_catalogue would
    # reject it on provenance before ever reaching the mode check. Constructing
    # the pair by hand isolates the mode difference, which is what this test is
    # about; the provenance rule has its own tests.
    filter_finder = Finder(
        Config.desi_y3_fast(
            filter_low_likelihood=True, enable_tau_eb=False, max_absorbers=1
        ),
        model=finder.model,
        prior=finder.prior,
        grid=finder.grid,
        warn_about_threads=False,
    )
    screened = filter_finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=201)
    full = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=202)
    shared = dict(full.provenance)
    mixed = [
        dataclasses.replace(
            screened,
            provenance={**shared, "evidence_mode": "filter"},
        ),
        full,
    ]
    with pytest.warns(UserWarning, match="mixes evidence modes"):
        catalogue = results_to_catalogue(mixed, detection_threshold=0.98)
    assert catalogue.run["GPDLF_EVIDENCE_MODE"] == "mixed"


# --------------------------------------------------------------------------
# The boundaries of the slice, asserted so they cannot be forgotten
# --------------------------------------------------------------------------


def test_a_one_absorber_configuration_reports_one_candidate(finder):
    """The null-versus-one path, on a spectrum that contains two absorbers.

    This used to assert a silent truncation: the configuration allowed four
    absorbers while the code found at most one. That is gone -- a configuration
    claiming more than the package evaluates is now refused outright, so the
    honest statement is narrower: ask for one absorber and you get at most one,
    whatever the spectrum contains.
    """
    assert finder.config.max_absorbers == 1
    result = finder.run(build(BY_NAME["two-separated-dlas"]), targetid=301)
    assert result.status == "completed"
    assert result.ladder is None
    assert len(result.absorber_candidates) <= 1
    if result.absorber_candidates:
        assert result.absorber_candidates[0].model == 1


def test_candidates_are_not_detections(finder):
    """A clean spectrum still gets a candidate. That is the whole point of C4.

    ``len(result.absorber_candidates) == 1`` must not be readable as "one DLA was
    detected": the best-fitting grid point exists in a spectrum with nothing in
    it, and the posterior is what decides.
    """
    result = finder.run(build(BY_NAME["absorber-free-desi-grid"]), targetid=401)
    assert result.status == "completed"
    assert len(result.absorber_candidates) == 1  # a candidate exists ...
    assert result.p_absorber < 0.5  # ... and it is not a detection
    assert not result.detected(0.98)

    # And the catalogue -- the policy layer -- keeps it out.
    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.absorbers == ()
    assert catalogue.spectra[0].n_absorbers == 0


def test_the_detection_threshold_has_no_default():
    """C6: the docstring said 'required'; the signature said 0.98."""
    import inspect

    assert inspect.signature(Result.detected).parameters["threshold"].default is (
        inspect.Parameter.empty
    )
    assert inspect.signature(results_to_catalogue).parameters[
        "detection_threshold"
    ].default is (inspect.Parameter.empty)

    with pytest.raises(TypeError):
        Result(targetid=1, z_qso=2.6, status="completed").detected()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        results_to_catalogue([])  # type: ignore[call-arg]


def test_the_threshold_is_recorded_so_a_consumer_can_recover_it(finder):
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=402)
    catalogue = results_to_catalogue([result], detection_threshold=0.90)
    assert catalogue.run["GPDLF_DETECTION_THRESHOLD"] == 0.90


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_an_impossible_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="probability in"):
        results_to_catalogue([], detection_threshold=bad)


def test_screening_score_is_populated_in_filter_mode_and_nan_otherwise(finder):
    """C3: the schema promised this field; the adapter left it NaN."""
    from gp_dla_finder.finder import screening_score

    filter_finder = Finder(
        Config.desi_y3_fast(
            filter_low_likelihood=True, enable_tau_eb=False, max_absorbers=1
        ),
        model=finder.model,
        prior=finder.prior,
        grid=finder.grid,
        warn_about_threads=False,
    )
    spectrum = build(BY_NAME["classical-dla-mid-z"])

    screened = filter_finder.run(spectrum, targetid=403)
    full_grid = finder.run(spectrum, targetid=404)

    assert screened.evidence_mode == "filter"
    assert full_grid.evidence_mode == "exact"

    # Defined: the FILTER-prefix log Bayes factor. A ranking statistic.
    assert screening_score(screened) == pytest.approx(screened.log_bayes_factor)
    assert np.isfinite(screening_score(screened))

    # NaN in full-grid mode means "not screened", never "scored zero".
    assert np.isnan(screening_score(full_grid))

    # A failed result has no evidences to combine.
    assert np.isnan(screening_score(Result(targetid=0, z_qso=2.6, status="failed")))


def test_the_screening_score_survives_the_round_trip_in_both_modes(finder, tmp_path):
    """Both modes, written and read back -- C3 asks for exactly this."""
    filter_finder = Finder(
        Config.desi_y3_fast(
            filter_low_likelihood=True, enable_tau_eb=False, max_absorbers=1
        ),
        model=finder.model,
        prior=finder.prior,
        grid=finder.grid,
        warn_about_threads=False,
    )
    spectrum = build(BY_NAME["classical-dla-mid-z"])

    for name, runner, expect_finite in (
        ("filter", filter_finder, True),
        ("exact", finder, False),
    ):
        result = runner.run(spectrum, targetid=405)
        catalogue = results_to_catalogue([result], detection_threshold=0.5)
        path = tmp_path / f"{name}.fits"
        write_catalogue(path, catalogue)

        # Read the raw column, not a masked Table view: "not screened" must be
        # a real IEEE NaN in the file, not a TNULL sentinel or a mask that a
        # non-astropy reader would interpret differently.
        with fits.open(path) as hdul:
            hdu = hdul["DLACAT"]
            column = [c for c in hdu.columns if c.name == "GPDLF_SCREENING_SCORE"][0]
            assert column.format == "D"
            assert column.null is None, "screening score must not use a TNULL"
            raw = hdu.data["GPDLF_SCREENING_SCORE"]
            assert len(raw) == 1, f"{name}: expected one absorber row"
            value = float(raw[0])

        if expect_finite:
            assert np.isfinite(value)
            assert value == pytest.approx(result.log_bayes_factor, rel=1e-9)
        else:
            assert np.isnan(value)


# --- C5: run-defining provenance must agree -------------------------------
#
# results_to_catalogue() used to copy the run record from the FIRST result. A
# catalogue could therefore name a model, prior, grid or backend that produced
# only some of its rows. Every field below is tested, not a representative one:
# the whole failure mode is that an untested field is the one that silently
# disagrees.


def _completed(targetid, **provenance):
    base = {
        "gpdlf_version": "0.1.0",
        "config_digest": "abcdef0123456789",
        "preset": "desi_y3",
        "base_preset": "desi_y3",
        "evidence_mode": "exact",
        "num_samples": 10_000,
        "sample_grid": "pw14_172_225_10000",
        "model": "m",
        "prior": "p",
        "lsf_kernel": "desi",
        "quality_policy": "q",
        "seed": 0,
        "backend_backend": "numpy",
        "backend_faddeeva_source": "scipy",
        "compatibility_profile": "reference",
    }
    base.update(provenance)
    return Result(
        targetid=targetid,
        z_qso=2.6,
        status="completed",
        evidence_mode=base["evidence_mode"],
        p_absorber=0.1,
        provenance=base,
    )


@pytest.mark.parametrize(
    "field_name, other",
    [
        ("gpdlf_version", "0.2.0"),
        ("config_digest", "0000000000000000"),
        ("preset", "desi_y3_refined"),
        ("base_preset", "eboss_dr16q"),
        ("num_samples", 100_000),
        # Two runs that searched to different depths are different runs: P_DLA
        # sums over the absorber models searched, so combining them would put
        # two normalisations in one column.
        ("max_absorbers", 2),
        ("sample_grid", "pw14_172_225_100000"),
        ("model", "another-model"),
        ("prior", "another-prior"),
        ("lsf_kernel", "boss"),
        ("quality_policy", "another-policy"),
        ("seed", 7),
        ("backend_backend", "libcerf"),
        ("backend_faddeeva_source", "libcerf"),
        ("compatibility_profile", "fast"),
    ],
)
def test_mixed_run_defining_provenance_is_rejected(field_name, other):
    results = [_completed(1), _completed(2, **{field_name: other})]
    with pytest.raises(ValueError, match="run-defining provenance"):
        results_to_catalogue(results, detection_threshold=0.98)


def test_every_run_defining_field_is_covered_by_a_test():
    """The parametrisation above must not drift from the tuple it guards."""
    from gp_dla_finder.finder import RUN_DEFINING_PROVENANCE

    # Read the parametrize argvalues directly off the marker, so adding a field
    # to RUN_DEFINING_PROVENANCE without adding a case fails here.
    marker = next(
        m
        for m in test_mixed_run_defining_provenance_is_rejected.pytestmark
        if m.name == "parametrize"
    )
    covered = {values[0] for values in marker.args[1]}
    assert covered == set(RUN_DEFINING_PROVENANCE)


def test_agreeing_provenance_is_accepted():
    catalogue = results_to_catalogue(
        [_completed(1), _completed(2)], detection_threshold=0.98
    )
    assert catalogue.run["GPDLF_PRESET"] == "desi_y3"
    assert catalogue.run["GPDLF_MODEL"] == "m"


def test_evidence_mode_is_the_one_permitted_mixture():
    """It has a schema-supported 'mixed' value; nothing else does."""
    from gp_dla_finder.finder import MIXED_RUN_FIELDS, RUN_DEFINING_PROVENANCE

    assert "evidence_mode" not in RUN_DEFINING_PROVENANCE
    assert MIXED_RUN_FIELDS["evidence_mode"] == "mixed"

    results = [_completed(1), _completed(2, evidence_mode="filter")]
    with pytest.warns(UserWarning, match="mixes evidence modes"):
        catalogue = results_to_catalogue(results, detection_threshold=0.98)
    assert catalogue.run["GPDLF_EVIDENCE_MODE"] == "mixed"


# --- correction 4: immutability must be true THROUGH the containers ---------
#
# MappingProxyType(dict(...)) protects only the outer mapping. The docstring
# promised container-deep immutability while a nested dict, list or array inside
# provenance stayed writable -- so a result could be edited after the fact and
# still report itself as the run that produced it.


def test_nested_provenance_mappings_are_frozen():
    result = Result(
        targetid=1,
        z_qso=2.6,
        status="completed",
        provenance={"backend": {"name": "numpy", "flags": {"o": "3"}}},
    )
    with pytest.raises((TypeError, AttributeError)):
        result.provenance["backend"]["name"] = "tampered"
    with pytest.raises((TypeError, AttributeError)):
        result.provenance["backend"]["flags"]["o"] = "0"
    assert result.provenance["backend"]["name"] == "numpy"


def test_nested_provenance_sequences_are_frozen():
    result = Result(
        targetid=1,
        z_qso=2.6,
        status="completed",
        provenance={"grids": ["a", "b"], "nested": [{"k": 1}]},
    )
    assert isinstance(result.provenance["grids"], tuple)
    with pytest.raises(AttributeError):
        result.provenance["grids"].append("c")
    # And one level deeper, inside the sequence.
    with pytest.raises((TypeError, AttributeError)):
        result.provenance["nested"][0]["k"] = 2


def test_mutating_the_caller_s_provenance_afterwards_does_not_leak():
    """The result must not alias structures the caller still holds."""
    source = {"backend": {"name": "numpy"}, "grids": ["a"]}
    result = Result(targetid=1, z_qso=2.6, status="completed", provenance=source)
    source["backend"]["name"] = "tampered"
    source["grids"].append("b")
    source["new"] = "added"

    assert result.provenance["backend"]["name"] == "numpy"
    assert result.provenance["grids"] == ("a",)
    assert "new" not in result.provenance


def test_a_provenance_array_cannot_be_written_through():
    """Arrays are the case a mapping proxy cannot help with at all."""
    values = np.array([1.0, 2.0, 3.0])
    result = Result(
        targetid=1, z_qso=2.6, status="completed", provenance={"probe": values}
    )
    stored = result.provenance["probe"]
    assert isinstance(stored, np.ndarray)
    # Not the caller's object ...
    assert not np.shares_memory(stored, values), (
        "provenance aliases the caller's array; mutating it would rewrite the "
        "result's own record"
    )
    # ... and not writable through either.
    with pytest.raises(ValueError):
        stored[0] = 99.0
    values[0] = 99.0
    assert result.provenance["probe"][0] == 1.0


# --- N63: the screening score is STORED, and survives refinement ------------
#
# It used to be derived from the result's current evidence fields at
# catalogue-write time. That breaks in exactly the workflow the PI approved as
# the future direction: a refinement stage rewrites those fields and the mode,
# so the derivation would return NaN for a spectrum that had certainly been
# screened. A screened non-detection also has no absorber row to carry it.


@pytest.fixture
def filter_finder(finder):
    return Finder(
        Config.desi_y3_fast(
            filter_low_likelihood=True, enable_tau_eb=False, max_absorbers=1
        ),
        model=finder.model,
        prior=finder.prior,
        grid=finder.grid,
        warn_about_threads=False,
    )


def test_filter_only_records_the_screening_stage(filter_finder):
    result = filter_finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=501)
    assert result.evidence_mode == "filter"
    assert result.was_screened
    assert result.screening_score == pytest.approx(result.log_bayes_factor)
    assert result.screening_n_evaluated == result.n_evaluated > 0
    assert np.isfinite(result.screening_log_evidence_null)
    assert np.isfinite(result.screening_log_evidence_absorber)


def test_full_grid_only_records_no_screening_stage(finder):
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=502)
    assert result.evidence_mode == "exact"
    assert not result.was_screened
    assert np.isnan(result.screening_score)
    assert result.screening_n_evaluated == 0


def test_the_screening_score_survives_a_refinement_stage(filter_finder):
    """The representation the two-stage workflow needs, tested before it exists."""
    screened = filter_finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=503)
    original = screened.screening_score
    assert np.isfinite(original)

    refined = screened.refined(
        log_evidence_null=-100.0,
        log_evidence_absorber=-40.0,
        p_absorber=0.995,
        p_null=0.005,
        logp_absorber=float(np.log(0.995)),
        logp_null=float(np.log(0.005)),
        n_evaluated=10_000,
    )

    # The final evidence and mode moved ...
    assert refined.evidence_mode == "exact"
    assert refined.log_bayes_factor == pytest.approx(60.0)
    assert refined.n_evaluated == 10_000
    # ... and the screening record did not.
    assert refined.screening_score == original
    assert refined.was_screened
    assert refined.screening_n_evaluated == screened.screening_n_evaluated
    assert screening_score(refined) == original

    # The old derivation would have produced the REFINED Bayes factor here,
    # silently relabelling a full-grid number as a screening statistic.
    assert refined.screening_score != pytest.approx(refined.log_bayes_factor)


def test_a_screened_non_detection_still_records_its_score(filter_finder):
    """No absorber row, so the spectrum row is the only place it can live."""
    result = filter_finder.run(build(BY_NAME["absorber-free-desi-grid"]), targetid=504)
    assert result.was_screened
    assert not result.detected(0.98)

    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.absorbers == ()
    row = catalogue.spectra[0]
    assert row.screening_score == pytest.approx(result.screening_score)
    assert row.screening_n_evaluated == result.screening_n_evaluated


def test_both_products_carry_the_screening_columns(filter_finder, tmp_path):
    result = filter_finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=505)
    catalogue = results_to_catalogue([result], detection_threshold=0.5)
    path = tmp_path / "screened.fits"
    write_catalogue(path, catalogue)

    spectra = Table.read(path, hdu="SPECTRA")
    assert float(spectra["GPDLF_SCREENING_SCORE"][0]) == pytest.approx(
        result.screening_score
    )
    assert int(spectra["GPDLF_SCREENING_N_EVALUATED"][0]) == (
        result.screening_n_evaluated
    )
    absorbers = Table.read(path, hdu="DLACAT")
    assert float(absorbers["GPDLF_SCREENING_SCORE"][0]) == pytest.approx(
        result.screening_score
    )


def test_a_full_grid_spectrum_row_says_not_screened(finder, tmp_path):
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=506)
    catalogue = results_to_catalogue([result], detection_threshold=0.5)
    path = tmp_path / "fullgrid.fits"
    write_catalogue(path, catalogue)
    with fits.open(path) as hdul:
        raw = hdul["SPECTRA"].data["GPDLF_SCREENING_SCORE"]
        assert np.isnan(float(raw[0]))
        assert int(hdul["SPECTRA"].data["GPDLF_SCREENING_N_EVALUATED"][0]) == 0


# --- the mean-flux fit is implemented, so the production preset runs ---------
#
# Finder used to refuse enable_tau_eb=True because nothing implemented the fit.
# It is implemented now, so the deployed operating point is runnable and those
# refusal tests are gone -- replaced by tests that the fit actually happens.


def test_the_production_preset_runs_without_disabling_the_fit():
    """The mean-flux fit no longer has to be switched off.

    max_absorbers=2 is still needed: the deployed preset says 4 and this package
    evaluates M0/M1/M2 only, so it refuses rather than truncating silently.
    """
    finder = Finder(Config.desi_y3_fast(max_absorbers=1), warn_about_threads=False)
    assert finder.config.enable_tau_eb is True

    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=701)
    assert result.status == "completed"


def test_the_fitted_tau_reaches_provenance(finder):
    """Per spectrum, so it belongs on the result, not the run record."""
    fitted = Finder(Config.desi_y3_fast(max_absorbers=1), warn_about_threads=False)
    result = fitted.run(build(BY_NAME["classical-dla-mid-z"]), targetid=702)

    assert "mean_flux_tau_0" in result.provenance
    assert result.provenance["mean_flux_objective"] == "null"
    assert result.provenance["mean_flux_seed_tau_0"] == Config.desi_y3_fast().prev_tau_0
    assert result.provenance["mean_flux_tau_0"] == pytest.approx(
        result.provenance["mean_flux_seed_tau_0"]
        * result.provenance["mean_flux_factor"]
    )
    # And the source it was ported from, so a result can be traced back.
    assert result.provenance["mean_flux_source_commit"] == "9aa20dc"

    # The no-fit configuration records nothing rather than a placeholder.
    assert (
        "mean_flux_tau_0"
        not in finder.run(
            build(BY_NAME["classical-dla-mid-z"]), targetid=703
        ).provenance
    )


def test_the_fit_changes_the_answer(finder):
    """If it made no difference there would be no reason to run it."""
    fitted = Finder(Config.desi_y3_fast(max_absorbers=1), warn_about_threads=False)
    spectrum = build(BY_NAME["classical-dla-mid-z"])

    with_fit = fitted.run(spectrum, targetid=704)
    without = finder.run(spectrum, targetid=705)

    assert with_fit.log_evidence_null != without.log_evidence_null


# --- correction 2: the FULL configuration decides compatibility -------------


def test_the_config_digest_is_run_defining():
    from gp_dla_finder.finder import RUN_DEFINING_PROVENANCE

    assert "config_digest" in RUN_DEFINING_PROVENANCE
    assert "base_preset" in RUN_DEFINING_PROVENANCE


def test_a_field_outside_the_twelve_still_changes_the_digest():
    """The point of the digest: fields nobody listed still count.

    The old tuple named twelve fields. A run differing only in, say, the
    wavelength window or the column-density support compared equal on all
    twelve and would have been combined into one catalogue.
    """
    base = Config.desi_y3(enable_tau_eb=False)
    for field_name, value in (
        ("max_lambda", 1240.0),
        ("log_nhi_range", (18.0, 22.5)),
        ("num_lines", 4),
        ("prev_tau_0", 0.0025),
        ("filter_n_initial_floor", 4000),
        ("max_z_cut_kms", 2000.0),
    ):
        other = base.replace(**{field_name: value})
        assert other.digest != base.digest, (
            f"changing {field_name} did not change the config digest"
        )


def test_a_missing_provenance_field_is_a_disagreement_not_a_skip():
    """A result from an older code path that never recorded a field.

    Skipping it would let exactly the unrecorded run slip into a catalogue that
    claims to describe it.
    """
    complete = _completed(1)
    partial_provenance = dict(complete.provenance)
    del partial_provenance["config_digest"]
    partial = Result(
        targetid=2,
        z_qso=2.6,
        status="completed",
        evidence_mode="exact",
        p_absorber=0.1,
        provenance=partial_provenance,
    )
    with pytest.raises(ValueError, match="run-defining provenance"):
        results_to_catalogue([complete, partial], detection_threshold=0.98)


def test_two_results_from_the_same_config_agree_on_the_digest(finder):
    first = finder.run(build(BY_NAME["absorber-free-desi-grid"]), targetid=601)
    second = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=602)
    assert first.provenance["config_digest"] == second.provenance["config_digest"]
    catalogue = results_to_catalogue([first, second], detection_threshold=0.98)
    assert catalogue.run["GPDLF_CONFIG_DIGEST"] == first.provenance["config_digest"]


# --- RNG provenance must distinguish "seed 0" from "no seed" -----------------


def test_a_deterministic_run_records_its_seed_and_mode(finder):
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=801)
    catalogue = results_to_catalogue([result], detection_threshold=0.98)

    assert catalogue.run["GPDLF_SEED"] == 0
    assert catalogue.run["GPDLF_RNG_MODE"] == "deterministic"
    assert "RandomState" in catalogue.run["GPDLF_RNG_ALGORITHM"]


def test_a_stochastic_run_is_not_recorded_as_seed_zero(finder):
    """`seed or 0` mapped seed=None onto 0, which is a different run."""
    stochastic = Finder(
        Config.desi_y3_fast(seed=None, enable_tau_eb=False, max_absorbers=1),
        warn_about_threads=False,
    )
    result = stochastic.run(build(BY_NAME["classical-dla-mid-z"]), targetid=802)
    catalogue = results_to_catalogue([result], detection_threshold=0.98)

    assert catalogue.run["GPDLF_SEED"] == -1
    assert catalogue.run["GPDLF_RNG_MODE"] == "stochastic"
    # The two are distinguishable in the file, which is the whole point.
    deterministic = results_to_catalogue(
        [finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=803)],
        detection_threshold=0.98,
    )
    assert catalogue.run["GPDLF_SEED"] != deterministic.run["GPDLF_SEED"]
    assert catalogue.run["GPDLF_RNG_MODE"] != deterministic.run["GPDLF_RNG_MODE"]
