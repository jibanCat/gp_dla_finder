"""The per-spectrum empirical-Bayes mean-flux fit.

Ported from the reference's ``fit_tau_eb`` (commit 9aa20dc) into this package's
own types. These tests cover the pieces PI ruling N68 named: the selected
factor, the objective values, mask behaviour, boundary selection, ties, invalid
inputs, and the fail-closed behaviour N69 requires for the absorber-aware
objective.

Fixtures are generated from named seeds. No private mock spectrum is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import load_model
from gp_dla_finder.config import Config
from gp_dla_finder.errors import NumericalError
from gp_dla_finder.gp.evidence import assemble_model, null_log_evidence
from gp_dla_finder.gp.spectrum import prepare_spectrum
from gp_dla_finder.mean_flux import (
    ALGORITHMIC_MAPPING,
    REFERENCE_SOURCE,
    SUPPORTED_OBJECTIVES,
    MeanFluxFit,
    ObjectiveNotSupported,
    fit_tau_0,
)
from synthetic import CORPUS, build

BY_NAME = {case.name: case for case in CORPUS}


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture
def prepared(model):
    config = Config.desi_y3_fast()
    return prepare_spectrum(build(BY_NAME["classical-dla-mid-z"]), model, config)


# --------------------------------------------------------------------------
# What it selects
# --------------------------------------------------------------------------


def test_it_selects_the_grid_point_with_the_highest_evidence(prepared, model):
    config = Config.desi_y3_fast()
    fit = fit_tau_0(prepared, model, config)

    assert fit.factor in config.tau_eb_factors
    assert fit.tau_0 == pytest.approx(fit.seed_tau_0 * fit.factor)
    # The winner really is the maximum of the recorded values.
    best = int(np.argmax(fit.log_evidence))
    assert fit.factors[best] == fit.factor


def test_the_recorded_evidences_are_the_ones_the_model_gives(prepared, model):
    """Recompute one grid point independently; it must match."""
    config = Config.desi_y3_fast()
    fit = fit_tau_0(prepared, model, config)

    index = 2
    assembled = assemble_model(
        prepared,
        model,
        config,
        tau_0=fit.seed_tau_0 * fit.factors[index],
        beta=config.prev_beta,
    )
    assert fit.log_evidence[index] == pytest.approx(
        float(null_log_evidence(prepared, assembled)), rel=0, abs=0
    )


def test_it_is_deterministic(prepared, model):
    config = Config.desi_y3_fast()
    first = fit_tau_0(prepared, model, config)
    second = fit_tau_0(prepared, model, config)
    assert first == second


def test_the_scan_covers_every_configured_factor(prepared, model):
    config = Config.desi_y3_fast()
    fit = fit_tau_0(prepared, model, config)
    assert fit.factors == tuple(float(f) for f in config.tau_eb_factors)
    assert len(fit.log_evidence) == len(fit.factors)
    assert all(np.isfinite(fit.log_evidence))


# --------------------------------------------------------------------------
# Boundaries, ties, and honesty about them
# --------------------------------------------------------------------------


def test_a_boundary_selection_is_reported_as_one(prepared, model):
    """A maximum at the edge is a bound, not an optimum.

    The reference extended its grid to 6x after exactly this.
    """
    config = Config.desi_y3_fast().replace(tau_eb_factors=(1.0, 2.0))
    fit = fit_tau_0(prepared, model, config)
    assert fit.at_grid_edge is True

    interior = MeanFluxFit(
        tau_0=0.005,
        factor=2.0,
        seed_tau_0=0.0025,
        factors=(1.0, 2.0, 3.0),
        log_evidence=(-3.0, -1.0, -2.0),
    )
    assert interior.at_grid_edge is False


def test_a_tie_resolves_to_the_first_factor_not_to_chance():
    """np.argmax takes the first maximum; ties must not be platform-dependent."""
    fit = MeanFluxFit(
        tau_0=0.0025,
        factor=1.0,
        seed_tau_0=0.0025,
        factors=(1.0, 2.0),
        log_evidence=(-1.0, -1.0),
    )
    assert fit.margin == 0.0
    assert int(np.argmax(fit.log_evidence)) == 0


def test_the_margin_says_how_decisive_the_choice_was(prepared, model):
    fit = fit_tau_0(prepared, model, Config.desi_y3_fast())
    assert fit.margin >= 0.0
    assert np.isfinite(fit.margin)


# --------------------------------------------------------------------------
# Invalid input
# --------------------------------------------------------------------------


def test_the_absorber_objective_fails_closed(prepared, model):
    """N69: it must never quietly fall back to the null objective."""
    with pytest.raises(ObjectiveNotSupported) as excinfo:
        fit_tau_0(prepared, model, Config.desi_y3_fast(), objective="absorber")

    message = str(excinfo.value)
    assert "absorber" in message
    assert "null" in message  # says what IS supported
    assert "'dla'" in message  # and names the reference's term for it


def test_an_unknown_objective_is_refused(prepared, model):
    with pytest.raises(ObjectiveNotSupported):
        fit_tau_0(prepared, model, Config.desi_y3_fast(), objective="nonsense")


def test_the_reference_name_is_not_accepted_as_input(prepared, model):
    """The package's vocabulary is 'absorber'; 'dla' is the reference's.

    Translating silently would blur two string interfaces that are not the same.
    """
    with pytest.raises(ObjectiveNotSupported):
        fit_tau_0(prepared, model, Config.desi_y3_fast(), objective="dla")


@pytest.mark.parametrize("factors", [(), (0.0, 1.0), (-1.0, 1.0)])
def test_an_impossible_factor_grid_is_rejected(prepared, model, factors):
    config = Config.desi_y3_fast().replace(tau_eb_factors=factors)
    with pytest.raises(ValueError):
        fit_tau_0(prepared, model, config)


def test_a_scan_with_no_finite_evidence_raises(prepared, model, monkeypatch):
    import gp_dla_finder.mean_flux as module

    monkeypatch.setattr(module, "null_log_evidence", lambda *a, **k: float("nan"))
    with pytest.raises(NumericalError, match="no finite log evidence"):
        fit_tau_0(prepared, model, Config.desi_y3_fast())


# --------------------------------------------------------------------------
# The HCD mask, which this package deliberately does not implement
# --------------------------------------------------------------------------


def test_no_hcd_masking_is_applied():
    """The reference's mask was retracted; production runs without it.

    On 90 random targets and 5000 random mock spectra the mask over-corrects:
    median column-density bias +0.135 -> -0.131 dex with it, +0.135 -> +0.026
    dex without. This package does not implement it at all, and the
    configuration flag that would have requested it stays off.
    """
    assert Config.desi_y3().tau_eb_apply_hcd_mask is False
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src"
        / "gp_dla_finder"
        / "mean_flux.py"
    ).read_text()
    # Documented, not silently absent.
    assert "retracted" in source.lower()
    assert "0.026" in source and "-0.131" in source


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_the_port_records_what_it_was_ported_from():
    assert REFERENCE_SOURCE["commit"] == "9aa20dc"
    assert REFERENCE_SOURCE["function"] == "fit_tau_eb"
    assert "GPModel" in ALGORITHMIC_MAPPING["NullGPMAT + learned_file"]
    # The objective mapping is recorded, including the one that is refused.
    assert "NOT IMPLEMENTED" in ALGORITHMIC_MAPPING["objective='dla'"]


def test_only_the_null_objective_is_advertised():
    assert SUPPORTED_OBJECTIVES == ("null",)


def test_the_fit_provenance_is_complete(prepared, model):
    fit = fit_tau_0(prepared, model, Config.desi_y3_fast())
    record = fit.provenance()
    for key in (
        "mean_flux_tau_0",
        "mean_flux_factor",
        "mean_flux_seed_tau_0",
        "mean_flux_objective",
        "mean_flux_at_grid_edge",
        "mean_flux_source_commit",
    ):
        assert key in record


# --------------------------------------------------------------------------
# The retracted mask must fail, not be ignored
# --------------------------------------------------------------------------


def test_requesting_the_hcd_mask_fails_closed(prepared, model):
    """Silently ignoring it would return an UNMASKED fit labelled as masked."""
    from gp_dla_finder.mean_flux import HCDMaskNotSupported

    config = Config.desi_y3_fast(tau_eb_apply_hcd_mask=True, max_absorbers=1)
    with pytest.raises(HCDMaskNotSupported) as excinfo:
        fit_tau_0(prepared, model, config)

    message = str(excinfo.value)
    assert "tau_eb_apply_hcd_mask" in message
    # It says WHY, with the measurement, not just "unsupported".
    assert "-0.131" in message and "+0.026" in message


def test_the_finder_path_also_fails_closed():
    """The public path, not only the function."""
    from gp_dla_finder.finder import Finder
    from gp_dla_finder.mean_flux import HCDMaskNotSupported

    finder = Finder(
        Config.desi_y3_fast(tau_eb_apply_hcd_mask=True, max_absorbers=1),
        warn_about_threads=False,
    )
    with pytest.raises(HCDMaskNotSupported):
        finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=1)


def test_the_mask_flag_is_still_accepted_as_configuration():
    """Kept for round-tripping a reference configuration; only using it fails."""
    config = Config.desi_y3_fast(tau_eb_apply_hcd_mask=True, max_absorbers=1)
    assert config.tau_eb_apply_hcd_mask is True
    assert config.is_modified  # and it is labelled as a departure


# --------------------------------------------------------------------------
# The scan survives Finder.run()
# --------------------------------------------------------------------------


def test_the_whole_scan_reaches_the_result():
    """The winner alone cannot say whether the grid was decisive."""
    import dataclasses

    from gp_dla_finder.finder import Finder

    finder = Finder(Config.desi_y3_fast(max_absorbers=1), warn_about_threads=False)
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=2)

    fit = result.mean_flux
    assert fit is not None
    assert fit.factors == tuple(float(f) for f in Config.desi_y3_fast().tau_eb_factors)
    assert len(fit.log_evidence) == len(fit.factors)
    assert fit.tau_0 == pytest.approx(fit.seed_tau_0 * fit.factor)
    assert isinstance(fit.at_grid_edge, bool)
    assert np.isfinite(fit.margin)

    # Immutable, like everything else on a Result.
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.factor = 99.0


def test_a_result_without_the_fit_carries_no_scan():
    from gp_dla_finder.finder import Finder

    finder = Finder(
        Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=1),
        warn_about_threads=False,
    )
    result = finder.run(build(BY_NAME["classical-dla-mid-z"]), targetid=3)
    assert result.mean_flux is None


def test_the_scan_is_auditable_after_the_run():
    """Recompute the winner from the retained vector."""
    from gp_dla_finder.finder import Finder

    finder = Finder(Config.desi_y3_fast(max_absorbers=1), warn_about_threads=False)
    fit = finder.run(build(BY_NAME["low-snr-dla"]), targetid=4).mean_flux

    best = int(np.argmax(fit.log_evidence))
    assert fit.factors[best] == fit.factor
    ordered = sorted(fit.log_evidence, reverse=True)
    assert fit.margin == pytest.approx(ordered[0] - ordered[1])
