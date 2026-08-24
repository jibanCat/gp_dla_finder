"""The multi-absorber pieces that do not depend on the resampler's RNG.

Traced from the reference's ``DLAGPMAT.log_model_evidences``. The resampler
itself is an injected callable because the reference draws from the unseeded
global NumPy generator, which makes every k >= 2 evidence irreproducible; how to
handle that is a PI checkpoint. Everything here is settled and testable now.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder.multi import (
    MAX_SEED,
    ModelLadder,
    combine_log_evidences,
    reject_close_pairs,
    seeded_resampler,
)

# --- minimum separation is a rejection, not a penalty ------------------------


def test_pairs_closer_than_the_minimum_are_rejected():
    redshifts = np.array([[2.00, 2.00, 2.00], [2.001, 2.50, 2.05]])
    rejected = reject_close_pairs(redshifts, min_separation=0.01)
    assert list(rejected) == [True, False, False]


def test_the_rule_looks_at_every_pair_not_just_neighbours_in_input_order():
    """Sorted first, so an unordered sample is still caught."""
    redshifts = np.array([[2.50], [2.00], [2.005]])
    assert reject_close_pairs(redshifts, min_separation=0.01)[0]


def test_one_absorber_can_never_be_rejected():
    redshifts = np.array([[2.0, 2.5, 3.0]])
    assert not np.any(reject_close_pairs(redshifts, min_separation=1.0))


def test_the_shape_contract_is_enforced():
    with pytest.raises(ValueError, match="k, n_samples"):
        reject_close_pairs(np.array([2.0, 2.5]), min_separation=0.01)


# --- the evidence arithmetic -------------------------------------------------


def test_the_normalisation_matches_the_reference_zero_based_convention():
    """Term by term against the reference's own loop variable.

    The reference iterates ``num_dlas`` from 0, and
    ``log_likelihoods_dla[num_dlas]`` is the (num_dlas + 1)-absorber model:

        log Z_k = max + log mean(exp(...)) + log N - log N * num_dlas
                = max + log mean(exp(...)) + log N - log N * (k - 1)

    An earlier version of this helper subtracted ``log N * k``, charging M2 one
    extra log N -- 9.21 nat at 10,000 samples, which changed which model was
    preferred. The test below is written from the reference formula, not from
    whatever the helper currently returns.
    """
    values = np.array([-10.0, -11.0, -10.5, -12.0])
    n = 4
    peak = values.max()
    base = peak + np.log(np.mean(np.exp(values - peak)))

    for k in (1, 2, 3):
        num_dlas = k - 1  # the reference's zero-based loop variable
        expected = base + np.log(n) - np.log(n) * num_dlas
        assert combine_log_evidences(values, n, k) == pytest.approx(
            expected, rel=0, abs=0
        )

    # The one-absorber model carries NO Occam penalty: num_dlas = 0.
    assert combine_log_evidences(values, n, 1) == pytest.approx(
        base + np.log(n), rel=0, abs=0
    )
    # And exactly one log N separates consecutive rungs.
    assert combine_log_evidences(values, n, 1) - combine_log_evidences(
        values, n, 2
    ) == pytest.approx(np.log(n))


def test_each_extra_absorber_costs_exactly_one_log_n():
    values = np.array([-10.0, -11.0, -10.5])
    n = 100
    one = combine_log_evidences(values, n, 1)
    two = combine_log_evidences(values, n, 2)
    assert one - two == pytest.approx(np.log(n))


def test_rejected_samples_leave_the_average_entirely():
    """NaN, not zero weight -- a dropped sample is not a sample that fitted badly."""
    kept = np.array([-10.0, -10.0])
    with_rejected = np.array([-10.0, -10.0, np.nan])

    # Same mean over the surviving samples, so the same evidence.
    assert combine_log_evidences(kept, 2, 2) == pytest.approx(
        combine_log_evidences(with_rejected, 2, 2)
    )


def test_an_entirely_rejected_model_reports_nan_rather_than_raising():
    """The reference's signal to stop the ladder."""
    assert np.isnan(combine_log_evidences(np.array([np.nan, np.nan]), 2, 2))


# --- the ladder --------------------------------------------------------------


def test_the_ladder_counts_only_the_rungs_that_were_evaluated():
    assert ModelLadder((-100.0, -95.0, -97.0)).evaluated == 2
    assert ModelLadder((-100.0, -95.0, float("nan"))).evaluated == 1


def test_posteriors_normalise_across_the_models():
    posteriors = ModelLadder((-100.0, -95.0, -97.0)).posteriors((0.0, -1.0, -3.0))
    assert sum(posteriors) == pytest.approx(1.0)
    assert posteriors[1] > posteriors[0] > 0.0


def test_a_stopped_rung_contributes_nothing_rather_than_evidence_against():
    """It was never measured; treating that as a zero likelihood would inflate
    the models that were."""
    stopped = ModelLadder((-100.0, -95.0, float("nan")))
    posteriors = stopped.posteriors((0.0, -1.0, -3.0))
    assert posteriors[2] == 0.0
    assert sum(posteriors) == pytest.approx(1.0)

    # And the surviving models keep their relative odds.
    two_model = ModelLadder((-100.0, -95.0)).posteriors((0.0, -1.0))
    assert posteriors[0] == pytest.approx(two_model[0])
    assert posteriors[1] == pytest.approx(two_model[1])


def test_a_ladder_needs_at_least_the_null_and_one_absorber_models():
    with pytest.raises(ValueError, match="at least"):
        ModelLadder((-100.0,))


def test_mismatched_priors_are_refused():
    with pytest.raises(ValueError, match="priors for"):
        ModelLadder((-100.0, -95.0)).posteriors((0.0,))


def test_a_ladder_with_no_finite_model_raises():
    with pytest.raises(ValueError, match="finite"):
        ModelLadder((float("nan"), float("nan"))).posteriors((0.0, 0.0))


# --- the injected resampler --------------------------------------------------


def test_the_seeded_resampler_is_reproducible():
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    first = seeded_resampler(20260820)(weights, 100)
    second = seeded_resampler(20260820)(weights, 100)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_draws():
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    assert not np.array_equal(
        seeded_resampler(1)(weights, 100), seeded_resampler(2)(weights, 100)
    )


def test_the_resampler_follows_the_weights():
    """Inverse-CDF sampling: a dominant weight should dominate the draws."""
    weights = np.array([0.97, 0.01, 0.01, 0.01])
    draws = seeded_resampler(3)(weights, 2000)
    assert np.mean(draws == 0) > 0.9
    assert draws.min() >= 0 and draws.max() < weights.size


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros(4), "positive finite sum"),
        (np.array([np.nan, 1.0]), "finite"),
        (np.array([np.inf, 1.0]), "finite"),
        (np.array([-0.5, 1.0]), "non-negative"),
    ],
)
def test_degenerate_weights_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        seeded_resampler(0)(bad, 10)


# --- the D4 contract ---------------------------------------------------------


def test_the_stream_is_local_and_never_touches_the_global_rng():
    """D4: no process-global RNG. Asserted by watching it.

    The legacy ``np.random`` calls below are deliberate and are what ruff's
    NPY002 would normally warn about: they ARE the global API, used here to
    observe that the resampler leaves it alone.
    """
    np.random.seed(4242)  # noqa: NPY002
    before = np.random.rand(3)  # noqa: NPY002

    np.random.seed(4242)  # noqa: NPY002
    seeded_resampler(11)(np.array([0.25, 0.75]), 1000)
    after = np.random.rand(3)  # noqa: NPY002

    assert np.array_equal(before, after), "the resampler advanced the global RNG"


def test_a_supplied_seed_reproduces_the_reference_stream_bitwise():
    """RandomState(s).rand(n) is the same float stream as np.random.seed(s).

    What this establishes, precisely: the RNG STREAM and the resampling INDICES
    agree under a controlled seed. It does not establish end-to-end M2 evidence
    parity against a live reference call -- that comparison is still owed.

    And the "no global state" claim needs care. The PACKAGE side is local: it
    never touches ``np.random``. The TEST side deliberately drives the legacy
    global stream, because that is the reference's idiom and reproducing it is
    the point of the comparison.
    """
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    ours = seeded_resampler(7)(weights, 200)

    np.random.seed(7)  # noqa: NPY002
    reference_uniforms = np.random.rand(200)  # noqa: NPY002
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    assert np.array_equal(ours, np.searchsorted(cumulative, reference_uniforms))


def test_an_unseeded_stream_is_the_explicit_stochastic_opt_in():
    weights = np.array([0.25, 0.75])
    first = seeded_resampler(None)(weights, 500)
    second = seeded_resampler(None)(weights, 500)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("bad", [-1, MAX_SEED + 1])
def test_an_out_of_range_seed_is_refused(bad):
    with pytest.raises(ValueError, match="seed must lie"):
        seeded_resampler(bad)


@pytest.mark.parametrize("bad", [1.5, "0", True])
def test_a_non_integer_seed_is_refused(bad):
    with pytest.raises(TypeError, match="integer or None"):
        seeded_resampler(bad)


def test_the_cdf_boundary_cannot_produce_an_out_of_range_index():
    """A CDF summing a hair below 1.0 would let a draw index past the end."""
    # Weights whose normalised cumulative sum lands below 1.0 in float64.
    weights = np.full(1000, 1.0 / 3.0)
    draws = seeded_resampler(3)(weights, 5000)
    assert draws.max() < weights.size
    assert draws.min() >= 0


# --- an incomplete ladder must say so ----------------------------------------


def test_a_complete_ladder_says_it_is_complete():
    ladder = ModelLadder((-100.0, -95.0, -97.0))
    assert ladder.complete is True
    assert ladder.stopped_at is None


def test_an_incomplete_ladder_reports_where_it_stopped():
    ladder = ModelLadder((-100.0, -95.0, float("nan")))
    assert ladder.complete is False
    assert ladder.stopped_at == 2


def test_the_posteriors_of_an_incomplete_ladder_are_conditional():
    """A zero here means "never measured", not "measured and impossible".

    Reporting it as a complete M0/M1/M2 posterior would present an unevaluated
    model as a scientifically measured zero probability.
    """
    ladder = ModelLadder((-100.0, -95.0, float("nan")))
    posteriors = ladder.posteriors((0.0, -1.0, -3.0))

    assert posteriors[2] == 0.0
    assert sum(posteriors) == pytest.approx(1.0)
    # And the caller can tell which kind of zero it is.
    assert ladder.complete is False


# --------------------------------------------------------------------------
# End to end through Finder: M0 / M1 / M2 on the generated corpus
# --------------------------------------------------------------------------


def _ladder_for(finder, case_name):
    from synthetic import CORPUS, build

    case = {c.name: c for c in CORPUS}[case_name]
    result = finder.run(build(case), targetid=1)
    assert result.status == "completed"
    assert result.ladder is not None
    return result.ladder


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_name, expected_model",
    [
        # Two well-separated DLAs.
        ("two-separated-dlas", 2),
        # Nothing there: M0 must win, and does, comfortably.
        ("absorber-free-desi-grid", 0),
    ],
)
def test_the_ladder_selects_the_right_model(ladder_finder, case_name, expected_model):
    """Selection by the JOINT probability, not by evidence alone.

    An evidence-only argmax assumes a uniform model prior, which is exactly
    what the absorber-existence prior says is false.
    """
    ladder = _ladder_for(ladder_finder, case_name)
    assert ladder.complete
    assert ladder.selected_model == expected_model, (
        f"{case_name}: selected {ladder.model_labels[ladder.selected_model]}, "
        f"expected M{expected_model} (evidences "
        f"{tuple(round(v, 1) for v in ladder.log_evidences)}, posteriors "
        f"{tuple(round(v, 3) for v in ladder.model_posteriors)})"
    )


@pytest.mark.slow
def test_a_one_absorber_control_currently_selects_M2(ladder_finder):
    """KNOWN VALIDATION WARNING -- recorded, not excused.

    A spectrum with ONE injected DLA selects M2. This test asserts the current
    behaviour so the number cannot drift unnoticed; it is not an endorsement.

    What is actually happening, measured: the best M2 pair contains the true
    absorber (z = 2.5495, log N_HI = 20.52 against an injection at 2.55, 20.5)
    plus a SPURIOUS one at z = 2.331, log N_HI = 20.04, and that spurious
    absorber buys +3.45 nat. The model prior charges M2 about 2.6 nat more than
    M1, which is not enough to cover it.

    The gain is identical across eight seeds -- min = max = +3.45 -- because the
    one-absorber posterior is so sharply peaked that every resampled partner
    lands on effectively the same grid point. So this is not Monte Carlo noise;
    it is the estimator preferring a two-absorber explanation of a
    one-absorber spectrum, deterministically.

    Whether that is the reference's behaviour faithfully ported or a defect in
    this port is not settled. It is the first thing the N13 multi-seed,
    multi-spectrum gate has to answer, and no k >= 2 result should be quoted
    before it does.
    """
    ladder = _ladder_for(ladder_finder, "classical-dla-mid-z")
    assert ladder.complete
    assert ladder.selected_model == 2, (
        "the one-absorber control no longer selects M2 -- if this changed "
        "deliberately, update the finding in the docstring; if not, "
        "investigate before relaxing anything"
    )
    gain = ladder.log_evidences[2] - ladder.log_evidences[1]
    assert gain == pytest.approx(3.45, abs=0.1), (
        f"the spurious-absorber gain moved to {gain:+.2f} nat"
    )


@pytest.mark.slow
def test_a_blended_pair_selects_one_absorber_on_the_posterior(ladder_finder):
    """Evidence and posterior disagree here, and the prior decides.

    Withdrawn from the previous increment: I described the blended pair's M1
    preference as a physical resolution limit. It was partly an arithmetic
    error -- M2 was being charged an extra log N. With that corrected the
    EVIDENCE now favours M2, and it is the model prior that still selects M1.

    That is a weaker and more honest statement: the data marginally prefer two
    absorbers, and the prior against multiplicity outweighs it. Whether that is
    the right call for a genuinely blended pair is an N13 question.
    """
    ladder = _ladder_for(ladder_finder, "two-blended-dlas")
    evidence_argmax = int(np.nanargmax(ladder.log_evidences))
    assert evidence_argmax == 2, "the corrected evidence should favour M2 here"
    assert ladder.selected_model == 1, (
        "the model prior should still select M1 for the blended pair"
    )


@pytest.mark.slow
def test_a_strong_plus_weak_pair_still_favours_two(ladder_finder):
    ladder = _ladder_for(ladder_finder, "strong-plus-weak")
    assert int(np.nanargmax(ladder.log_evidences)) == 2


@pytest.mark.slow
def test_the_same_seed_gives_the_same_ladder(ladder_finder):
    """D4: deterministic by default, including the resampled rung."""
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder
    from synthetic import CORPUS, build

    case = {c.name: c for c in CORPUS}["two-separated-dlas"]
    spectrum = build(case)

    first = ladder_finder.run(spectrum, targetid=1).ladder
    other = Finder(
        Config.desi_y3_fast(
            enable_tau_eb=False, max_absorbers=2, experimental_multi_absorber=True
        ),
        model=ladder_finder.model,
        prior=ladder_finder.prior,
        grid=ladder_finder.grid,
        warn_about_threads=False,
    )
    second = other.run(spectrum, targetid=1).ladder

    assert first.log_evidences == second.log_evidences


@pytest.mark.slow
def test_a_different_seed_moves_the_resampled_rung_only(ladder_finder):
    """M0 and M1 never resample, so a seed change must not touch them."""
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder
    from synthetic import CORPUS, build

    case = {c.name: c for c in CORPUS}["two-separated-dlas"]
    spectrum = build(case)

    base = ladder_finder.run(spectrum, targetid=1).ladder
    reseeded = (
        Finder(
            Config.desi_y3_fast(
                enable_tau_eb=False,
                max_absorbers=2,
                experimental_multi_absorber=True,
                seed=12345,
            ),
            model=ladder_finder.model,
            prior=ladder_finder.prior,
            grid=ladder_finder.grid,
            warn_about_threads=False,
        )
        .run(spectrum, targetid=1)
        .ladder
    )

    assert base.log_evidences[0] == reseeded.log_evidences[0]
    assert base.log_evidences[1] == reseeded.log_evidences[1]
    assert base.log_evidences[2] != reseeded.log_evidences[2]


# --- configured multiplicity must be truthful (correction 4) ------------------


def test_a_preset_claiming_more_absorbers_than_we_compute_is_refused():
    """The deployed preset says four; this path evaluates M0/M1/M2.

    Running it and stopping at two would put max_absorbers=4 in provenance for
    a calculation that never went past two.
    """
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder

    assert Config.desi_y3().max_absorbers == 4
    with pytest.raises(NotImplementedError, match="max_absorbers=4"):
        Finder(
            Config.desi_y3(max_absorbers=4, experimental_multi_absorber=True),
            warn_about_threads=False,
        )


@pytest.mark.parametrize("supported", [1, 2])
def test_the_supported_multiplicities_are_accepted(supported):
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder

    Finder(
        Config.desi_y3_fast(
            enable_tau_eb=False,
            max_absorbers=supported,
            experimental_multi_absorber=supported >= 2,
        ),
        warn_about_threads=False,
    )


# --- FILTER plus M2 is not a validated combination (correction 5) ------------


def test_filter_with_two_absorbers_is_refused():
    """The hybrid is neither full-grid M2 nor the reference's FILTER path.

    In FILTER mode the one-absorber proposal covers only the evaluated prefix
    while the two-absorber scan walks the whole first-absorber grid.
    """
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder

    with pytest.raises(NotImplementedError, match="FILTER is not supported"):
        Finder(
            Config.desi_y3_fast(
                enable_tau_eb=False,
                max_absorbers=2,
                filter_low_likelihood=True,
                experimental_multi_absorber=True,
            ),
            warn_about_threads=False,
        )


def test_filter_remains_available_for_the_one_absorber_path():
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder

    Finder(
        Config.desi_y3_fast(
            enable_tau_eb=False, max_absorbers=1, filter_low_likelihood=True
        ),
        warn_about_threads=False,
    )


# --- a coherent Result: one posterior, the selected model, both candidates ---


def test_the_default_finder_constructs_and_runs():
    """Finder() used to raise: it defaulted to a preset this path refuses."""
    from gp_dla_finder.finder import Finder
    from synthetic import CORPUS, build

    finder = Finder(warn_about_threads=False)
    assert finder.config.max_absorbers == 1
    result = finder.run(build({c.name: c for c in CORPUS}["classical-dla-mid-z"]))
    assert result.status == "completed"


@pytest.mark.slow
def test_one_absorber_probability_not_two(ladder_finder):
    """The ladder and the Result must not report different probabilities."""
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=1
    )
    assert result.p_absorber == pytest.approx(result.ladder.p_absorber)
    assert result.p_null == pytest.approx(result.ladder.model_posteriors[0])
    # The old two-model number is kept, named, and not used for anything.
    assert np.isfinite(result.legacy_two_model_p_absorber)


@pytest.mark.slow
def test_a_selected_M2_returns_both_candidates(ladder_finder):
    """Returning one grid point for a two-absorber model would drop an absorber."""
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=1
    )
    assert result.ladder.selected_model == 2
    assert len(result.absorber_candidates) == 2
    assert all(c.model == 2 for c in result.absorber_candidates)

    # Both injected absorbers are recovered, loosely -- these are grid points,
    # not estimates, so the tolerance is about locating them rather than
    # measuring them.
    found = sorted(c.grid_z_abs for c in result.absorber_candidates)
    assert found[0] == pytest.approx(2.20, abs=0.05)
    assert found[1] == pytest.approx(2.70, abs=0.05)


@pytest.mark.slow
def test_a_one_absorber_run_reports_one_candidate_from_M1(ladder_finder):
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder
    from synthetic import CORPUS, build

    finder = Finder(
        Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=1),
        model=ladder_finder.model,
        prior=ladder_finder.prior,
        grid=ladder_finder.grid,
        warn_about_threads=False,
    )
    result = finder.run(
        build({c.name: c for c in CORPUS}["classical-dla-mid-z"]), targetid=1
    )
    assert result.ladder is None
    assert len(result.absorber_candidates) == 1
    assert result.absorber_candidates[0].model == 1


@pytest.mark.slow
def test_an_m2_evaluated_m1_selected_result_is_coherent(ladder_finder):
    """The blended pair evaluates M2 and selects M1."""
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-blended-dlas"]), targetid=1
    )
    ladder = result.ladder
    assert ladder is not None
    assert ladder.selected_model == 1

    # p_absorber is P(M1) + P(M2), not the two-model value.
    assert result.p_absorber == pytest.approx(
        ladder.model_posteriors[1] + ladder.model_posteriors[2]
    )
    assert result.p_null == pytest.approx(ladder.model_posteriors[0])
    assert result.p_absorber + result.p_null == pytest.approx(1.0)

    # The legacy number is present, named, and NOT what p_absorber reports
    # (they would only coincide if P(M2) were exactly zero).
    assert np.isfinite(result.legacy_two_model_p_absorber)

    # One candidate, from M1, because that is the selected model.
    assert len(result.absorber_candidates) == 1
    assert result.absorber_candidates[0].model == 1


@pytest.mark.slow
def test_a_one_absorber_run_still_writes_normally(ladder_finder):
    """The supported path must not be caught by the M2 refusal."""
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder, results_to_catalogue
    from synthetic import CORPUS, build

    finder = Finder(
        Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=1),
        model=ladder_finder.model,
        prior=ladder_finder.prior,
        grid=ladder_finder.grid,
        warn_about_threads=False,
    )
    result = finder.run(
        build({c.name: c for c in CORPUS}["classical-dla-mid-z"]), targetid=1
    )
    assert result.ladder is None
    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert len(catalogue.spectra) == 1


# --- the experimental opt-in (PI ruling N80) ---------------------------------


def test_two_absorbers_needs_the_experimental_opt_in():
    """max_absorbers=2 alone is not enough. Two deliberate choices."""
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import ExperimentalFeatureNotEnabled, Finder

    with pytest.raises(ExperimentalFeatureNotEnabled) as excinfo:
        # Deliberately WITHOUT experimental_multi_absorber: that is the point.
        Finder(
            Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=2),
            warn_about_threads=False,
        )
    message = str(excinfo.value)
    assert "experimental_multi_absorber=True" in message
    # The message states the limitation, not just the flag name.
    assert "80%" in message and "close" in message


def test_the_opt_in_is_recorded_in_provenance(ladder_finder):
    from gp_dla_finder.finder import results_to_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=1
    )
    assert result.provenance["experimental"] == "multi_absorber"
    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.run["GPDLF_EXPERIMENTAL"] == "multi_absorber"


def test_a_default_run_records_no_experimental_features():
    from gp_dla_finder.finder import Finder, results_to_catalogue
    from synthetic import CORPUS, build

    result = Finder(warn_about_threads=False).run(
        build({c.name: c for c in CORPUS}["classical-dla-mid-z"]), targetid=1
    )
    assert result.provenance["experimental"] == ""
    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.run["GPDLF_EXPERIMENTAL"] == ""


# --- FITS is the flat DESI catalogue; the ladder lives in the JSON ------------
#
# PI ruling, increment 26. The FITS product is one row per absorber with the
# historical columns and nothing structural added: no MODELS HDU, no nesting,
# no variable-length arrays. Everything the flat table cannot express moves to
# the structured JSON output.


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_name, expected_selected, expected_rows",
    [("two-separated-dlas", 2, 2), ("two-blended-dlas", 1, 1)],
)
def test_m2_writes_flat_rows_with_the_traced_legacy_semantics(
    ladder_finder, tmp_path, case_name, expected_selected, expected_rows
):
    """The row semantics traced from the reference writer, asserted per field.

    Both fixtures the ruling names: selected-M2, and evaluated-M2/selected-M1.
    The mapping is in ``2026-08-21_legacy_multi_dla_writer_trace.md``:

        P_DLA     spectrum   sum over every absorber model searched
        P_NULL    spectrum   1 - P_DLA
        LOGP_NULL spectrum   joint of the null model
        LOGP_DLA  per row    joint of the (n+1)-absorber model
        MODEL_P   per row    posterior of the (n+1)-absorber model
    """
    pytest.importorskip("astropy")
    from astropy.table import Table

    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.fits import write_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}[case_name]), targetid=42
    )
    ladder = result.ladder
    assert ladder.selected_model == expected_selected

    path = tmp_path / f"{case_name}.fits"
    write_catalogue(path, results_to_catalogue([result], detection_threshold=0.98))

    rows = Table.read(path, hdu="DLACAT")
    assert len(rows) == expected_rows

    for n, row in enumerate(rows):
        # Spectrum-level: identical on every row of this spectrum.
        assert row["TARGETID"] == 42
        assert row["P_DLA"] == pytest.approx(ladder.p_absorber)
        assert row["P_NULL"] == pytest.approx(1.0 - ladder.p_absorber)
        assert row["LOGP_NULL"] == pytest.approx(ladder.log_joint[0])
        # Per absorber index: this row's model, not the spectrum's summary.
        assert row["LOGP_DLA"] == pytest.approx(ladder.log_joint[n + 1])
        assert row["MODEL_P"] == pytest.approx(ladder.model_posteriors[n + 1])
        assert row["DLAID"].strip() == f"4200{n}"

    # The distinction the old writer lost: on a two-row spectrum MODEL_P must
    # differ between the rows, because they are different models.
    if expected_rows > 1:
        model_p = [float(row["MODEL_P"]) for row in rows]
        assert model_p[0] != pytest.approx(model_p[1])
        assert sum(model_p) == pytest.approx(float(rows[0]["P_DLA"]))


@pytest.mark.slow
def test_model_p_is_one_model_not_the_sum(ladder_finder, tmp_path):
    """The specific error the ruling names, guarded directly.

    ``MODEL_P`` used to be written as ``P(M1) + P(M2)`` -- which is ``P_DLA``
    -- on every row. The regression signature is the two columns agreeing
    everywhere, so that is what this refuses.

    Note it cannot be "MODEL_P < P_DLA on every row": on this fixture P(M2)
    rounds to 1 and so does the sum, and the top rung legitimately equals
    P_DLA. It is the FIRST row that distinguishes the two definitions.
    """
    pytest.importorskip("astropy")
    from astropy.table import Table

    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.fits import write_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=42
    )
    ladder = result.ladder
    path = tmp_path / "not-a-sum.fits"
    write_catalogue(path, results_to_catalogue([result], detection_threshold=0.98))

    rows = Table.read(path, hdu="DLACAT")
    assert len(rows) == 2

    first = float(rows[0]["MODEL_P"])
    assert first == pytest.approx(ladder.model_posteriors[1])
    # P(M1) on a spectrum that selected M2 is small, and P_DLA is not.
    assert first < float(rows[0]["P_DLA"])
    assert not all(
        float(row["MODEL_P"]) == pytest.approx(float(row["P_DLA"])) for row in rows
    )


@pytest.mark.slow
def test_both_m2_candidates_reach_the_file(ladder_finder, tmp_path):
    pytest.importorskip("astropy")
    from astropy.table import Table

    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.fits import write_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=42
    )
    assert len(result.absorber_candidates) == 2

    path = tmp_path / "pair.fits"
    write_catalogue(path, results_to_catalogue([result], detection_threshold=0.98))

    absorbers = Table.read(path, hdu="DLACAT")
    assert len(absorbers) == 2
    # Distinct rows, distinct identifiers, and the uncertainties stay NaN
    # because no validated estimator supplied them.
    assert len(set(absorbers["DLAID"])) == 2
    assert np.all(np.isnan(np.asarray(absorbers["Z_DLA_ERR"], dtype=float)))


@pytest.mark.slow
@pytest.mark.parametrize("case_name", ["two-separated-dlas", "two-blended-dlas"])
def test_no_fits_product_carries_a_models_hdu(ladder_finder, tmp_path, case_name):
    """The format boundary, asserted on both products and both fixtures."""
    pytest.importorskip("astropy")
    from astropy.io import fits as afits

    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.fits import write_catalogue, write_legacy_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}[case_name]), targetid=42
    )
    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.models, "the in-memory catalogue still carries the ladder"

    for writer, name in (
        (write_catalogue, "extended"),
        (write_legacy_catalogue, "legacy"),
    ):
        path = tmp_path / f"{case_name}-{name}.fits"
        writer(path, catalogue)
        with afits.open(path) as hdul:
            names = {hdu.name for hdu in hdul}
        assert "MODELS" not in names, f"{name} product grew a MODELS HDU"
        # And nothing variable-length crept in either.
        with afits.open(path) as hdul:
            for hdu in hdul:
                columns = getattr(hdu, "columns", None)
                for code in getattr(columns, "formats", ()) or ():
                    assert "P(" not in str(code) and "Q(" not in str(code), (
                        f"{name} product has a variable-length column: {code}"
                    )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", ["two-separated-dlas", "two-blended-dlas"])
def test_the_ladder_survives_the_structured_json_round_trip(
    ladder_finder, tmp_path, case_name
):
    """Everything FITS no longer carries, recovered from the JSON."""
    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.structured import (
        read_structured_results,
        selected_models,
        write_structured_results,
    )
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}[case_name]), targetid=42
    )
    ladder = result.ladder

    path = tmp_path / f"{case_name}.json"
    write_structured_results(
        path, results_to_catalogue([result], detection_threshold=0.98)
    )
    payload = read_structured_results(path)

    (spectrum,) = payload["spectra"]
    assert spectrum["targetid"] == 42
    models = spectrum["models"]
    assert len(models) == len(ladder.log_evidences)

    for index, model in enumerate(models):
        assert model["model_index"] == index
        assert model["model_label"] == f"M{index}"
        assert model["log_evidence"] == pytest.approx(ladder.log_evidences[index])
        assert model["log_prior"] == pytest.approx(ladder.log_priors[index])
        assert model["posterior"] == pytest.approx(ladder.model_posteriors[index])
        assert model["evaluated"] is True
        assert model["selected"] is (index == ladder.selected_model)

    selected = [m for m in models if m["selected"]]
    assert len(selected) == 1
    assert selected_models(payload) == {42: f"M{ladder.selected_model}"}

    # Candidate membership. Every candidate belongs to the SELECTED model --
    # both members of an M2 pair are M2 -- and absorber_index is what
    # distinguishes them. An earlier version derived the model from the index
    # and labelled the pair "M1, M2", which is a different claim and a wrong
    # one (PI ruling, increment 27, correction 1).
    selected_label = f"M{ladder.selected_model}"
    for n, absorber in enumerate(spectrum["absorbers"]):
        assert absorber["absorber_index"] == n
        assert absorber["model_index"] == ladder.selected_model
        assert absorber["model_label"] == selected_label

    # And it came from the candidate, not from the loop counter.
    assert [c.model for c in result.absorber_candidates] == [
        ladder.selected_model
    ] * len(result.absorber_candidates)


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_name, expected_model, expected_count",
    [
        ("two-separated-dlas", 2, 2),
        ("two-blended-dlas", 1, 1),
        ("absorber-free-desi-grid", 0, 0),
    ],
)
def test_a_real_result_round_trips_to_json_with_the_right_membership(
    ladder_finder, tmp_path, case_name, expected_model, expected_count
):
    """Result -> Catalogue -> JSON -> back, from an actual inference run.

    The hand-built fixtures in test_structured.py check the document shape.
    This checks that a real Result reaches it intact, which is the path a user
    takes and the one the membership bug lived on.
    """
    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.structured import (
        read_structured_results,
        write_structured_results,
    )
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}[case_name]), targetid=77
    )
    assert result.ladder.selected_model == expected_model

    path = tmp_path / f"{case_name}-real.json"
    write_structured_results(
        path, results_to_catalogue([result], detection_threshold=0.98)
    )
    (spectrum,) = read_structured_results(path)["spectra"]

    assert len(spectrum["absorbers"]) == expected_count
    for absorber in spectrum["absorbers"]:
        assert absorber["model_index"] == expected_model
    # Contiguous membership indices within the selected model.
    assert [a["absorber_index"] for a in spectrum["absorbers"]] == list(
        range(expected_count)
    )
    # The ladder came through with it.
    assert len(spectrum["models"]) == len(result.ladder.log_evidences)


def test_a_run_without_a_ladder_writes_no_models(tmp_path):
    pytest.importorskip("astropy")
    from gp_dla_finder.finder import Finder, results_to_catalogue
    from gp_dla_finder.io.fits import write_catalogue
    from gp_dla_finder.io.structured import structured_payload
    from synthetic import CORPUS, build

    result = Finder(warn_about_threads=False).run(
        build({c.name: c for c in CORPUS}["classical-dla-mid-z"]), targetid=1
    )
    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.models == ()

    path = tmp_path / "m1.fits"
    write_catalogue(path, catalogue)

    payload = structured_payload(catalogue)
    assert payload["spectra"][0]["models"] == []


# --- the search depth has to be recoverable from the file --------------------


@pytest.mark.slow
def test_the_file_records_how_deep_the_search_went(ladder_finder, tmp_path):
    """``P_DLA`` cannot be read without it.

    The reference sums ``P_DLA`` over every absorber model it searched, so the
    same column means P(M1) on a one-absorber run and P(M1)+P(M2) on a
    two-absorber one. Nothing else in the file distinguishes them.
    """
    pytest.importorskip("astropy")
    from astropy.io import fits as afits

    from gp_dla_finder.finder import Finder, results_to_catalogue
    from gp_dla_finder.io.fits import (
        read_catalogue_metadata,
        write_catalogue,
        write_legacy_catalogue,
    )
    from synthetic import CORPUS, build

    cases = {c.name: c for c in CORPUS}
    deep = ladder_finder.run(build(cases["two-separated-dlas"]), targetid=42)
    shallow = Finder(warn_about_threads=False).run(
        build(cases["classical-dla-mid-z"]), targetid=1
    )

    for result, expected in ((deep, 2), (shallow, 1)):
        catalogue = results_to_catalogue([result], detection_threshold=0.98)
        assert catalogue.run["GPDLF_MAX_DLAS"] == expected

        extended = tmp_path / f"extended-{expected}.fits"
        write_catalogue(extended, catalogue)
        assert read_catalogue_metadata(extended)["GPDLF_MAX_DLAS"] == expected

        # And on the absorber HDU of BOTH products, so a strict-legacy reader
        # that never opens RUNINFO can still interpret P_DLA.
        for writer, name in (
            (write_catalogue, "ext"),
            (write_legacy_catalogue, "leg"),
        ):
            path = tmp_path / f"{name}-{expected}.fits"
            writer(path, catalogue)
            with afits.open(path) as hdul:
                assert hdul["DLACAT"].header["GPDLF_MAX_DLAS"] == expected


@pytest.mark.slow
def test_max_dlas_is_the_search_limit_not_the_detection_count(ladder_finder, tmp_path):
    """A spectrum that searched two and selected one still records two."""
    pytest.importorskip("astropy")
    from astropy.table import Table

    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.fits import write_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-blended-dlas"]), targetid=42
    )
    assert result.ladder.selected_model == 1

    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    assert catalogue.run["GPDLF_MAX_DLAS"] == 2

    path = tmp_path / "searched-two-found-one.fits"
    write_catalogue(path, catalogue)
    assert len(Table.read(path, hdu="DLACAT")) == 1


# --- the strict legacy view now carries M2, because it is flat ---------------


@pytest.mark.slow
def test_the_legacy_view_writes_a_pair_without_being_asked(ladder_finder, tmp_path):
    """No opt-in, and no refusal: two absorbers are two ordinary rows.

    The old refusal rested on a misreading of ``P_DLA`` as a two-model
    quantity. The trace shows the reference defines it as the sum over every
    absorber model searched, so the flat projection is not lossy in the way the
    refusal claimed -- what was missing was the record of the search depth.
    """
    pytest.importorskip("astropy")
    from astropy.io import fits as afits
    from astropy.table import Table

    from gp_dla_finder.catalogue import LEGACY_ABSORBER_COLUMNS
    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.io.fits import write_legacy_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=42
    )
    catalogue = results_to_catalogue([result], detection_threshold=0.98)

    path = tmp_path / "legacy-pair.fits"
    write_legacy_catalogue(path, catalogue)

    rows = Table.read(path, hdu="DLACAT")
    assert len(rows) == 2
    # Exactly the historical columns, in the historical order.
    assert tuple(rows.colnames) == LEGACY_ABSORBER_COLUMNS
    # One parent, contiguous identifiers.
    assert set(rows["TARGETID"]) == {42}
    assert [r["DLAID"].strip() for r in rows] == ["42000", "42001"]

    with afits.open(path) as hdul:
        assert {hdu.name for hdu in hdul} == {"PRIMARY", "DLACAT"}
