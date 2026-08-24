"""The flat FITS catalogue, checked against the reference writer's semantics.

PI ruling, increment 26: *"Add a controlled M2 catalogue comparison against the
read-only legacy writer, or an exact table-level fixture derived from that trace
when a direct call is impractical."*

A direct call is impractical, and it is worth being precise about why.
``dlasearch.py`` reads DESI coadd files through ``desispec.io.read_spectra``,
needs a survey catalogue with BAL columns, and writes its table only after
processing a whole file of spectra. None of that is available here, and none of
it is the part under test.

So this file splits the comparison:

* the **definitions** of ``P_DLA``, ``P_NULL``, ``model_posteriors`` and the
  DLA-model index set are taken by **calling the reference's own
  ``BayesModelSelect``** with the package's ladder numbers. Live reference code,
  not a transcription;
* the **row loop** — which quantity lands in which column of which row — is
  transcribed verbatim from ``dlasearch.py:602-628`` into
  :func:`_reference_rows` below, because that loop cannot be reached without the
  DESI stack. The transcription is short, and the source is quoted beside it.

Both halves were traced from the legacy DESI catalogue writer.

**No astropy here, deliberately.** These tests compare the rows the package
builds against the reference's definitions; they stop at the ``Catalogue``. The
separate question -- whether those rows survive the FITS round trip unchanged --
is ``test_multi.py``, which needs astropy but no reference.

The split is not cosmetic. The reference checkout and astropy live in different
CI jobs: ``canonical-parity`` has the reference and no astropy,
``catalogue-io`` has astropy and no reference. A test needing both skips in
both, which is what happened when this module first imported astropy -- it ran
on no machine but mine. Split this way, each half runs where its dependency is,
and together they chain reference -> rows -> file.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.needs_reference


@pytest.fixture(scope="module")
def reference_selector(reference_repo):
    """The reference's model selector, driven directly.

    ``all_max_dlas=[0, 2]`` with ``dla_model_ind=1`` is the
    ``single_absorber_model=True`` layout — ``[Null, 1-absorber, 2-absorber]``
    — which is this package's ladder. There is no sub-DLA rung, so
    ``num_subdla`` is 0.
    """
    from gpy_dla_detection.bayesian_model_selection import BayesModelSelect

    return BayesModelSelect(all_max_dlas=[0, 2], dla_model_ind=1)


def _reference_rows(selector, log_evidences, log_priors, n_selected, targetid):
    """The reference's per-spectrum rows, from its own definitions.

    The probability definitions come from ``selector`` — reference code. The
    loop is ``dlasearch.py:602-628``:

        for n in range(ndla):
            dlaid = str(tid) + "00" + str(n)
            pdlalist.append(p_dla)
            pnulllist.append(p_no_dla)
            logpdlalist.append(log_posteriors_dla[n])
            logpnulllist.append(log_posteriors_no_dla)
            modelplist.append(model_posteriors[1 + num_subdla + n])
    """
    selector.log_likelihoods = np.asarray(log_evidences, dtype=float)
    selector.log_priors = np.asarray(log_priors, dtype=float)
    selector.log_posteriors = selector.log_likelihoods + selector.log_priors

    model_posteriors = selector.model_posteriors
    p_dla = selector.p_dla
    p_no_dla = selector.p_no_dla

    # run_bayes_select.py:171-172
    log_posteriors_no_dla = selector.log_posteriors[0]
    log_posteriors_dla = selector.log_posteriors[-2:]

    num_subdla = 0
    return [
        {
            "DLAID": f"{targetid}00{n}",
            "P_DLA": float(p_dla),
            "P_NULL": float(p_no_dla),
            "LOGP_DLA": float(log_posteriors_dla[n]),
            "LOGP_NULL": float(log_posteriors_no_dla),
            "MODEL_P": float(model_posteriors[1 + num_subdla + n]),
        }
        for n in range(n_selected)
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_name, expected_selected",
    [("two-separated-dlas", 2), ("two-blended-dlas", 1)],
)
def test_the_flat_rows_match_the_reference_definitions(
    ladder_finder, reference_selector, case_name, expected_selected
):
    """Every legacy probability column, on every row, against reference code.

    Both fixtures the ruling names: one that selects M2, and one that evaluates
    M2 but selects M1.

    Compared at the ``Catalogue``, which is where the row semantics are decided
    -- ``results_to_catalogue`` assigns them, and the FITS writer only copies
    them out. ``test_multi.py`` checks the copy.
    """
    from gp_dla_finder.finder import results_to_catalogue
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}[case_name]), targetid=42
    )
    ladder = result.ladder
    assert ladder.selected_model == expected_selected

    expected = _reference_rows(
        reference_selector,
        ladder.log_evidences,
        ladder.log_priors,
        expected_selected,
        42,
    )

    catalogue = results_to_catalogue([result], detection_threshold=0.98)
    rows = catalogue.absorbers

    assert len(rows) == len(expected)
    for built, reference in zip(rows, expected, strict=True):
        assert built.dlaid == reference["DLAID"]
        for attribute, column in (
            ("p_dla", "P_DLA"),
            ("p_null", "P_NULL"),
            ("logp_dla", "LOGP_DLA"),
            ("logp_null", "LOGP_NULL"),
            ("model_p", "MODEL_P"),
        ):
            assert getattr(built, attribute) == pytest.approx(
                reference[column], rel=1e-12, abs=1e-12
            ), f"{column} differs on row {reference['DLAID']}"


@pytest.mark.slow
def test_p_dla_is_the_sum_over_absorber_models_not_the_top_rung(
    ladder_finder, reference_selector
):
    """The definition the increment-25 refusal got backwards.

    ``P_DLA`` was read as a two-model quantity that a three-model number would
    corrupt. The reference computes it as ``nansum`` over every absorber model,
    so the sum IS the definition.
    """
    from synthetic import CORPUS, build

    result = ladder_finder.run(
        build({c.name: c for c in CORPUS}["two-separated-dlas"]), targetid=42
    )
    ladder = result.ladder

    reference_selector.log_likelihoods = np.asarray(ladder.log_evidences, dtype=float)
    reference_selector.log_priors = np.asarray(ladder.log_priors, dtype=float)
    reference_selector.log_posteriors = (
        reference_selector.log_likelihoods + reference_selector.log_priors
    )

    # The index set the reference sums over is exactly the absorber models.
    assert list(reference_selector.dla_model_posterior_ind) == [False, True, True]
    assert reference_selector.p_dla == pytest.approx(ladder.p_absorber)
    assert reference_selector.p_no_dla == pytest.approx(1.0 - ladder.p_absorber)

    # And the package's summed value is what reaches the result.
    assert result.p_absorber == pytest.approx(reference_selector.p_dla)


@pytest.mark.slow
def test_the_built_rows_are_flat_and_well_formed(ladder_finder):
    """Ruling item 3: parent TARGETID, unique contiguous DLAID, consistent count."""
    from gp_dla_finder.finder import results_to_catalogue
    from synthetic import CORPUS, build

    cases = {c.name: c for c in CORPUS}
    results = [
        ladder_finder.run(build(cases["two-separated-dlas"]), targetid=42),
        ladder_finder.run(build(cases["two-blended-dlas"]), targetid=43),
    ]
    catalogue = results_to_catalogue(results, detection_threshold=0.98)
    absorbers = catalogue.absorbers

    # Every absorber row has a parent spectrum row.
    parents = {row.targetid for row in catalogue.spectra}
    assert {row.targetid for row in absorbers} <= parents

    # DLAID is unique catalogue-wide and contiguous from 0 within a spectrum.
    identifiers = [row.dlaid for row in absorbers]
    assert len(set(identifiers)) == len(identifiers)
    for targetid in sorted({row.targetid for row in absorbers}):
        mine = [row.dlaid for row in absorbers if row.targetid == targetid]
        assert mine == [f"{targetid}00{n}" for n in range(len(mine))]

    # The row count for a spectrum equals the selected absorber-count model.
    by_target = {r.targetid: r for r in results}
    for targetid in sorted({row.targetid for row in absorbers}):
        n_rows = sum(1 for row in absorbers if row.targetid == targetid)
        assert n_rows == by_target[targetid].ladder.selected_model
