"""The structured JSON result — the ladder's home now that FITS is flat.

Built from hand-made rows rather than a real inference run, so the document
shape is tested without paying for a Finder. The end-to-end round trip on real
ladders is in ``test_multi.py``.
"""

from __future__ import annotations

import json
import math

import pytest

from gp_dla_finder.catalogue import AbsorberRow, Catalogue, ModelRow, SpectrumRow
from gp_dla_finder.io.structured import (
    STRUCTURED_FORMAT_VERSION,
    read_structured_results,
    selected_models,
    structured_payload,
    write_structured_results,
)


def _spectrum(targetid: int, n_absorbers: int) -> SpectrumRow:
    return SpectrumRow(
        targetid=targetid,
        z_qso=2.6,
        status="completed",
        n_absorbers=n_absorbers,
        p_absorber=0.99,
        log_evidence_null=-210.0,
        log_evidence_absorber=-200.0,
        evidence_mode="exact",
        n_evaluated=10_000,
    )


def _absorber(
    targetid: int, index: int, *, model_p: float, logp: float, model_index: int = 1
):
    return AbsorberRow(
        model_index=model_index,
        targetid=targetid,
        dlaid=f"{targetid}00{index}",
        z_qso=2.6,
        z_dla=2.3 + 0.1 * index,
        nhi=20.5,
        p_dla=0.99,
        p_null=0.01,
        logp_dla=logp,
        logp_null=-212.0,
        model_p=model_p,
        evidence_mode="exact",
        log_evidence_absorber=-200.0,
        log_evidence_null=-210.0,
    )


@pytest.fixture
def ladder_catalogue() -> Catalogue:
    """One spectrum: M2 selected, M0/M1/M2 all evaluated, two absorbers."""
    return Catalogue(
        absorbers=[
            _absorber(42, 0, model_p=0.94, logp=-202.0, model_index=2),
            _absorber(42, 1, model_p=0.94, logp=-202.0, model_index=2),
        ],
        spectra=[_spectrum(42, 2)],
        models=[
            ModelRow(42, 0, -210.0, math.log(0.9), 0.01, True, False),
            ModelRow(42, 1, -206.0, math.log(0.09), 0.05, True, False),
            ModelRow(42, 2, -202.0, math.log(0.01), 0.94, True, True),
        ],
        run={"GPDLF_MAX_DLAS": 2, "GPDLF_EXPERIMENTAL": "multi_absorber"},
    )


def test_the_document_is_keyed_on_spectra(ladder_catalogue):
    payload = structured_payload(ladder_catalogue)
    assert payload["format"] == "gp_dla_finder.structured"
    assert payload["format_version"] == STRUCTURED_FORMAT_VERSION
    (spectrum,) = payload["spectra"]
    assert spectrum["targetid"] == 42
    assert len(spectrum["models"]) == 3
    assert len(spectrum["absorbers"]) == 2


def test_absorbers_state_which_model_they_belong_to(ladder_catalogue):
    """The membership the flat FITS rows leave implicit in the row count.

    Both members of a selected M2 pair belong to **M2**. Only their
    ``absorber_index`` distinguishes the first from the second. Deriving the
    model from the index -- which an earlier version did -- labels the pair
    "M1, M2", which is a different and wrong statement.
    """
    payload = structured_payload(ladder_catalogue)
    (spectrum,) = payload["spectra"]
    assert [a["model_index"] for a in spectrum["absorbers"]] == [2, 2]
    assert [a["model_label"] for a in spectrum["absorbers"]] == ["M2", "M2"]
    assert [a["absorber_index"] for a in spectrum["absorbers"]] == [0, 1]


def test_a_single_absorber_belongs_to_m1(tmp_path):
    catalogue = Catalogue(
        absorbers=[_absorber(7, 0, model_p=0.99, logp=-200.0)],
        spectra=[_spectrum(7, 1)],
        models=[
            ModelRow(7, 0, -210.0, math.log(0.9), 0.01, True, False),
            ModelRow(7, 1, -200.0, math.log(0.09), 0.99, True, True),
        ],
        run={"GPDLF_MAX_DLAS": 1},
    )
    (spectrum,) = structured_payload(catalogue)["spectra"]
    assert [a["model_index"] for a in spectrum["absorbers"]] == [1]


def test_an_absorber_row_cannot_belong_to_the_null_model():
    with pytest.raises(ValueError, match="model_index must be at least 1"):
        _absorber(1, 0, model_p=0.5, logp=-1.0, model_index=0)


def test_the_run_record_travels_with_the_ladder(ladder_catalogue):
    payload = structured_payload(ladder_catalogue)
    assert payload["run"]["GPDLF_MAX_DLAS"] == 2
    assert payload["run"]["GPDLF_EXPERIMENTAL"] == "multi_absorber"


def _same(left, right) -> bool:
    """Structural equality that treats NaN as equal to itself.

    Plain ``==`` cannot be used here: NaN is not equal to itself, and NaN is a
    value this document deliberately carries.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[k], right[k]) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(_same, left, right))
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def test_round_trip_preserves_every_field(ladder_catalogue, tmp_path):
    path = write_structured_results(tmp_path / "r.json", ladder_catalogue)
    assert _same(read_structured_results(path), structured_payload(ladder_catalogue))


def test_the_round_trip_check_would_notice_a_changed_value(ladder_catalogue):
    """_same must not be so permissive that it passes anything."""
    payload = structured_payload(ladder_catalogue)
    other = structured_payload(ladder_catalogue)
    other["spectra"][0]["models"][0]["posterior"] = 0.5
    assert not _same(payload, other)
    assert _same(payload, structured_payload(ladder_catalogue))


def test_an_unevaluated_rung_is_null_and_says_so(tmp_path):
    """Strict JSON (N84): ``null`` for the number, ``evaluated`` for the meaning.

    A posterior of 0 means "evaluated and very unlikely". A rung that was never
    reached has no evidence at all. Both are representable, because ``null``
    is not asked to carry the difference on its own.
    """
    catalogue = Catalogue(
        absorbers=[_absorber(7, 0, model_p=0.99, logp=-200.0)],
        spectra=[_spectrum(7, 1)],
        models=[
            ModelRow(7, 0, -210.0, math.log(0.9), 0.01, True, False),
            ModelRow(7, 1, -200.0, math.log(0.09), 0.99, True, True),
            ModelRow(7, 2, float("nan"), math.log(0.01), 0.0, False, False),
        ],
        run={"GPDLF_MAX_DLAS": 2},
    )
    path = write_structured_results(tmp_path / "stopped.json", catalogue)
    payload = read_structured_results(path)
    stopped = payload["spectra"][0]["models"][2]

    assert stopped["evaluated"] is False
    assert stopped["log_evidence"] is None

    # The evaluated-but-unlikely rung is a real number, not a null.
    evaluated = payload["spectra"][0]["models"][0]
    assert evaluated["evaluated"] is True
    assert evaluated["posterior"] == pytest.approx(0.01)

    # And the pair the ruling names, side by side.
    zero = ModelRow(8, 0, -210.0, math.log(0.9), 0.0, True, False)
    never = ModelRow(8, 2, float("nan"), math.log(0.01), float("nan"), False, False)
    rows = structured_payload(
        Catalogue(
            absorbers=[],
            spectra=[_spectrum(8, 0)],
            models=[zero, never],
            run={},
        )
    )["spectra"][0]["models"]
    assert (rows[0]["evaluated"], rows[0]["posterior"]) == (True, 0.0)
    assert (rows[1]["evaluated"], rows[1]["posterior"]) == (False, None)


def test_a_run_with_no_ladder_writes_an_empty_model_list(tmp_path):
    catalogue = Catalogue(
        absorbers=[_absorber(1, 0, model_p=0.99, logp=-200.0)],
        spectra=[_spectrum(1, 1)],
        models=[],
        run={"GPDLF_MAX_DLAS": 1},
    )
    payload = structured_payload(catalogue)
    assert payload["spectra"][0]["models"] == []
    assert selected_models(payload) == {}


def test_selected_models_omits_a_spectrum_without_a_ladder(ladder_catalogue):
    assert selected_models(structured_payload(ladder_catalogue)) == {42: "M2"}


def test_a_model_row_without_a_spectrum_is_refused():
    """Silently dropping it would lose the ladder the document exists to carry."""
    catalogue = Catalogue(
        absorbers=[],
        spectra=[],
        models=[ModelRow(99, 0, -210.0, math.log(0.9), 1.0, True, True)],
        run={},
    )
    with pytest.raises(ValueError, match="no spectrum row"):
        structured_payload(catalogue)


def test_an_unknown_major_format_version_is_refused(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"format": "gp_dla_finder.structured", "format_version": "9.0"})
    )
    with pytest.raises(ValueError, match="major version"):
        read_structured_results(path)


def test_a_foreign_document_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "something.else", "format_version": "1.0"}))
    with pytest.raises(ValueError, match="not a gp_dla_finder structured result"):
        read_structured_results(path)


def test_the_structured_writer_needs_no_optional_dependency():
    """It is standard library only, which is half the reason it is JSON.

    A user without astropy cannot write FITS, and must still be able to keep
    the full result.
    """
    import gp_dla_finder.io.structured as module

    source = module.__file__
    with open(source) as handle:
        text = handle.read()
    for forbidden in ("import astropy", "import h5py", "import numpy"):
        assert forbidden not in text


# --- strict RFC 8259 (PI ruling N84) ------------------------------------------


def _strict_loads(text: str):
    """Parse as a standards-compliant reader would.

    ``json.loads`` accepts ``NaN``, ``Infinity`` and ``-Infinity`` by default --
    they are a Python extension, not JSON. ``parse_constant`` is the hook that
    fires on exactly those three literals, so raising from it makes this parser
    behave like one written against the standard.
    """

    def reject(literal):
        raise AssertionError(f"non-standard JSON literal in the output: {literal!r}")

    return json.loads(text, parse_constant=reject)


def test_the_file_contains_no_non_standard_numeric_literals(ladder_catalogue, tmp_path):
    """The whole point of N84: any JSON parser reads this, not only Python's."""
    catalogue = Catalogue(
        absorbers=list(ladder_catalogue.absorbers),
        spectra=list(ladder_catalogue.spectra),
        models=[
            *ladder_catalogue.models,
            # A rung that was never evaluated, and an absorber whose
            # uncertainties are NaN. Both are the realistic sources of a
            # non-finite value.
            ModelRow(42, 3, float("nan"), math.log(0.001), float("nan"), False, False),
        ],
        run=dict(ladder_catalogue.run),
    )
    path = write_structured_results(tmp_path / "strict.json", catalogue)
    text = path.read_text()

    # Textually, first: the three literals must not appear at all.
    for literal in ("NaN", "Infinity", "-Infinity"):
        assert literal not in text, f"{literal!r} is in the file"

    # And behaviourally, through a parser that refuses them.
    _strict_loads(text)


def test_writing_a_non_finite_number_is_refused_rather_than_emitted():
    """If a field is ever added without going through _json_number.

    Constructed by bypassing the payload builder, because the builder is what
    this guards against a future change to.
    """
    payload = {"format": "gp_dla_finder.structured", "value": float("inf")}
    with pytest.raises(ValueError, match="Out of range|not valid JSON|allow_nan"):
        json.dumps(payload, allow_nan=False)


def test_reading_a_python_dialect_file_is_refused(tmp_path):
    """A file someone else wrote with Python's defaults."""
    path = tmp_path / "dialect.json"
    path.write_text(
        '{"format": "gp_dla_finder.structured", "format_version": "1.0", '
        '"run": {}, "spectra": [{"targetid": 1, "p_absorber": NaN}]}'
    )
    with pytest.raises(ValueError, match="non-standard JSON literal"):
        read_structured_results(path)


def test_infinities_are_nulled_too(tmp_path):
    """Not only NaN. An infinite log evidence is equally unwritable."""
    catalogue = Catalogue(
        absorbers=[],
        spectra=[_spectrum(9, 0)],
        models=[
            ModelRow(9, 0, float("-inf"), math.log(0.9), 0.0, True, False),
        ],
        run={},
    )
    path = write_structured_results(tmp_path / "inf.json", catalogue)
    assert "Infinity" not in path.read_text()
    model = _strict_loads(path.read_text())["spectra"][0]["models"][0]
    assert model["log_evidence"] is None
