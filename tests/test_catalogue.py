"""Catalogue schemas and FITS products (PI rulings N44, N49).

The tests that matter here are about what a *downstream reader* sees, so they go
through astropy rather than inspecting our own objects, and they cover the cases
the schema exists to handle:

* multiple absorbers on one spectrum, and the parent relationship between them;
* a completed spectrum with **no** absorber — which a DLA-only table cannot
  express, and which is the whole reason the per-spectrum table exists;
* every non-completed status;
* schema-version handling, including refusal of an unknown major version;
* the no-samples guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder.catalogue import (
    ABSORBER_SCHEMA,
    CATALOGUE_SCHEMA_VERSION,
    LEGACY_ABSORBER_COLUMNS,
    RUN_SCHEMA,
    SPECTRUM_SCHEMA,
    STATUSES,
    AbsorberRow,
    Catalogue,
    SpectrumRow,
    reference_dlaid,
    schema_for,
)

fits = pytest.importorskip("astropy.io.fits", reason="catalogue I/O needs astropy")
from astropy.table import Table  # noqa: E402

from gp_dla_finder.io.fits import (  # noqa: E402
    read_catalogue_metadata,
    write_catalogue,
    write_legacy_catalogue,
)

#: The historical column list, transcribed independently of the module under test
#: so a change to the schema cannot silently redefine what "legacy" means.
REFERENCE_COLUMNS = (
    "TARGETID",
    "RA",
    "DEC",
    "Z_QSO",
    "SNR_FOREST",
    "SNR_REDSIDE",
    "DLAID",
    "Z_DLA",
    "Z_DLA_ERR",
    "NHI",
    "NHI_ERR",
    "DLAFLAG",
    "P_DLA",
    "P_NULL",
    "LOGP_DLA",
    "LOGP_NULL",
    "MODEL_P",
)


@pytest.fixture
def catalogue() -> Catalogue:
    """One of every case the schema has to represent."""
    spectra = [
        # two absorbers
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            n_absorbers=2,
            p_absorber=0.995,
            log_evidence_null=-284.62,
            log_evidence_absorber=-280.11,
            evidence_mode="exact",
            quality_fraction=0.91,
            n_usable_pixels=1105,
            n_evaluated=50_000,
        ),
        # completed, nothing found: the case a DLA-only table cannot express
        SpectrumRow(
            targetid=2,
            z_qso=2.4,
            status="completed",
            n_absorbers=0,
            p_absorber=0.012,
            log_evidence_null=-300.0,
            log_evidence_absorber=-305.4,
            evidence_mode="exact",
            quality_fraction=0.88,
            n_usable_pixels=980,
            n_evaluated=50_000,
        ),
        SpectrumRow(
            targetid=3,
            z_qso=2.9,
            status="quality_rejected",
            reason="quality_policy_rejected",
            quality_fraction=0.05,
        ),
        SpectrumRow(
            targetid=4,
            z_qso=2.2,
            status="insufficient_data",
            reason="no_normalisation_coverage",
        ),
        SpectrumRow(targetid=5, z_qso=3.1, status="failed", reason="numerical_error"),
    ]
    absorbers = [
        AbsorberRow(
            targetid=1,
            dlaid=reference_dlaid(1, 0),
            z_qso=2.6,
            z_dla=2.312,
            nhi=20.63,
            p_dla=0.995,
            p_null=0.005,
            # Independent of p_dla: this is log P(1 absorber), not log(p_dla).
            logp_dla=-0.014,
            logp_null=-5.30,
            model_p=0.61,
            log_evidence_absorber=-280.11,
            log_evidence_null=-284.62,
            evidence_mode="exact",
        ),
        AbsorberRow(
            targetid=1,
            dlaid=reference_dlaid(1, 1),
            z_qso=2.6,
            z_dla=2.551,
            nhi=20.41,
            p_dla=0.995,
            p_null=0.005,
            logp_dla=-0.92,
            logp_null=-5.30,
            model_p=0.38,
            log_evidence_absorber=-281.02,
            log_evidence_null=-284.62,
            evidence_mode="exact",
        ),
    ]
    return Catalogue(
        absorbers=absorbers,
        spectra=spectra,
        run={
            "GPDLF_SCHEMA_VERSION": CATALOGUE_SCHEMA_VERSION,
            "GPDLF_PRESET": "desi_y3",
            "GPDLF_EVIDENCE_MODE": "exact",
            "GPDLF_NUM_SAMPLES": 50_000,
            "GPDLF_N_EVALUATED_CONFIGURED": 50_000,
            "GPDLF_SAMPLE_GRID": "pw14_172_225_50000",
            "GPDLF_VOIGT_BACKEND": "numpy",
            "GPDLF_COMPAT_PROFILE": "reference-d5b306e6",
            "GPDLF_QUALITY_POLICY": "desi-y3-reference",
        },
    )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def test_the_legacy_columns_match_the_reference_catalogue_exactly():
    """Name and order both, because some readers index by position."""
    assert LEGACY_ABSORBER_COLUMNS == REFERENCE_COLUMNS


def test_new_quantities_do_not_reuse_legacy_names():
    """A silently redefined legacy field is how downstream analysis goes wrong."""
    legacy = set(LEGACY_ABSORBER_COLUMNS)
    for column in ABSORBER_SCHEMA:
        if not column.legacy:
            assert column.name not in legacy
            assert column.name.startswith("GPDLF_"), (
                f"{column.name} is a new field and should be namespaced"
            )


def test_the_uncertainty_columns_are_marked_provisional():
    """No estimator exists, and the schema says so rather than implying one."""
    provisional = {c.name for c in ABSORBER_SCHEMA if c.provisional}
    assert provisional == {"Z_DLA_ERR", "NHI_ERR"}


def test_schema_lookup_refuses_an_unknown_table():
    assert schema_for("absorbers") is ABSORBER_SCHEMA
    assert schema_for("spectra") is SPECTRUM_SCHEMA
    assert schema_for("run") is RUN_SCHEMA
    with pytest.raises(KeyError, match="unknown catalogue table"):
        schema_for("dlas")


# --------------------------------------------------------------------------
# Row-level invariants
# --------------------------------------------------------------------------


def test_a_non_completed_spectrum_cannot_claim_absorbers():
    with pytest.raises(ValueError, match="cannot have"):
        SpectrumRow(
            targetid=9,
            z_qso=2.5,
            status="failed",
            reason="numerical_error",
            n_absorbers=1,
        )


def test_a_non_completed_spectrum_needs_a_reason_code():
    with pytest.raises(ValueError, match="requires a reason"):
        SpectrumRow(targetid=9, z_qso=2.5, status="quality_rejected")


def test_an_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="unknown status"):
        SpectrumRow(targetid=9, z_qso=2.5, status="maybe")


def test_absorbers_must_have_a_parent_spectrum_row():
    orphan = AbsorberRow(
        targetid=42,
        dlaid=reference_dlaid(42, 0),
        z_qso=2.5,
        z_dla=2.2,
        nhi=20.5,
        p_dla=0.9,
        p_null=0.1,
        logp_dla=-0.1,
        logp_null=-2.3,
        model_p=0.9,
        log_evidence_absorber=-1.0,
        log_evidence_null=-2.0,
        evidence_mode="exact",
    )
    with pytest.raises(ValueError, match="no spectrum row"):
        Catalogue(absorbers=[orphan], spectra=[])


def test_the_declared_absorber_count_must_match_the_rows():
    spectrum = SpectrumRow(
        targetid=1,
        z_qso=2.6,
        status="completed",
        n_absorbers=3,
        evidence_mode="exact",
    )
    absorber = AbsorberRow(
        targetid=1,
        dlaid=reference_dlaid(1, 0),
        z_qso=2.6,
        z_dla=2.2,
        nhi=20.5,
        p_dla=0.9,
        p_null=0.1,
        logp_dla=-0.1,
        logp_null=-2.3,
        model_p=0.9,
        log_evidence_absorber=-1.0,
        log_evidence_null=-2.0,
        evidence_mode="exact",
    )
    with pytest.raises(ValueError, match="declares 3 absorbers but 1"):
        Catalogue(absorbers=[absorber], spectra=[spectrum])


# --------------------------------------------------------------------------
# The strict legacy product, read back as a downstream reader would
# --------------------------------------------------------------------------


def test_legacy_export_has_exactly_the_historical_columns(tmp_path, catalogue):
    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    assert tuple(table.colnames) == REFERENCE_COLUMNS


def test_legacy_export_writes_nan_rather_than_inventing_uncertainties(
    tmp_path, catalogue
):
    """A plausible-looking uncertainty would be worse than a missing one."""
    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    with fits.open(path) as hdul:
        data = hdul["DLACAT"].data
        assert np.all(np.isnan(data["Z_DLA_ERR"]))
        assert np.all(np.isnan(data["NHI_ERR"]))
        comments = " ".join(str(c) for c in hdul["DLACAT"].header["COMMENT"])
        assert "validated estimator" in comments.lower()


def test_the_legacy_probability_fields_are_carried_independently(tmp_path, catalogue):
    """LOGP_DLA is NOT log(P_DLA), and deriving it corrupted the column.

    Traced in the reference's per-absorber loop: ``P_DLA``/``P_NULL`` are
    spectrum-level scalars repeated on every row, ``LOGP_DLA`` is
    ``log_posteriors_dla[n]`` — a per-absorber-index quantity — and ``MODEL_P`` is
    ``model_posteriors[1 + num_subdla + n]``. An earlier revision computed
    ``LOGP_DLA = log(P_DLA)`` and a test enforced it, which made the test
    ratify the bug.
    """
    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    with fits.open(path) as hdul:
        data = hdul["DLACAT"].data

    # Spectrum-level fields repeat unchanged across the two absorber rows.
    assert len(set(data["P_DLA"])) == 1
    assert len(set(data["P_NULL"])) == 1
    assert len(set(data["LOGP_NULL"])) == 1

    # Per-absorber-index fields differ between the rows.
    assert data["LOGP_DLA"][0] != data["LOGP_DLA"][1]
    assert data["MODEL_P"][0] != data["MODEL_P"][1]

    # And the values are carried through verbatim, not recomputed.
    assert np.allclose(data["LOGP_DLA"], [-0.014, -0.92])
    assert np.allclose(data["MODEL_P"], [0.61, 0.38])
    # The identity the old test asserted is false, and stays false.
    assert not np.allclose(data["LOGP_DLA"], np.log(data["P_DLA"]))


def test_dlaid_uses_the_public_reference_convention(tmp_path, catalogue):
    """``str(TARGETID) + "00" + str(n)``, transcribed from dlasearch.py.

    Downstream code parses these, so "compatible" has to mean the same bytes,
    not a similar idea. A hyphenated form of our own invention would not be.
    """
    assert reference_dlaid(1, 0) == "1000"
    assert reference_dlaid(39627, 0) == "39627000"
    assert reference_dlaid(39627, 2) == "39627002"

    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    assert set(table["DLAID"]) == {"1000", "1001"}
    for row in table:
        assert str(row["DLAID"]).startswith(f"{int(row['TARGETID'])}00")


def test_uncertainty_columns_are_present_and_preserved(tmp_path):
    """Present in both products; NaN unless an estimator supplies a value."""
    spectrum = SpectrumRow(
        targetid=7,
        z_qso=2.5,
        status="completed",
        n_absorbers=1,
        evidence_mode="exact",
    )
    supplied = AbsorberRow(
        targetid=7,
        dlaid=reference_dlaid(7, 0),
        z_qso=2.5,
        z_dla=2.2,
        nhi=20.5,
        p_dla=0.9,
        p_null=0.1,
        logp_dla=-0.1,
        logp_null=-2.3,
        model_p=0.9,
        log_evidence_absorber=-1.0,
        log_evidence_null=-2.0,
        evidence_mode="exact",
        z_dla_err=0.004,
        nhi_err=0.12,
    )
    cat = Catalogue(absorbers=[supplied], spectra=[spectrum])

    for writer, name in (
        (write_legacy_catalogue, "l.fits"),
        (write_catalogue, "e.fits"),
    ):
        path = writer(tmp_path / name, cat)
        with fits.open(path) as hdul:
            data = hdul["DLACAT"].data
            assert "Z_DLA_ERR" in data.names
            assert "NHI_ERR" in data.names
            # A supplied value is preserved, not overwritten with NaN.
            assert np.isclose(data["Z_DLA_ERR"][0], 0.004)
            assert np.isclose(data["NHI_ERR"][0], 0.12)


def test_legacy_export_cannot_represent_a_null_spectrum(tmp_path, catalogue):
    """Documented limitation, asserted rather than assumed.

    Five spectra go in; only the one with absorbers appears. This is exactly the
    blind spot the per-spectrum table exists to fill, and the test records it so
    nobody mistakes the legacy file for a complete account of a run.
    """
    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    assert len(catalogue.spectra) == 5
    assert set(table["TARGETID"]) == {1}


# --------------------------------------------------------------------------
# The extended product
# --------------------------------------------------------------------------


def test_extended_product_has_all_three_tables(tmp_path, catalogue):
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    with fits.open(path) as hdul:
        names = [hdu.name for hdu in hdul]
    assert "DLACAT" in names
    assert "SPECTRA" in names
    assert "RUNINFO" in names


def test_the_absorber_table_still_starts_with_the_legacy_columns(tmp_path, catalogue):
    """A reader that knows only the historical schema keeps working."""
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    assert tuple(table.colnames[: len(REFERENCE_COLUMNS)]) == REFERENCE_COLUMNS


def test_every_attempted_spectrum_appears_with_its_status(tmp_path, catalogue):
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    table = Table.read(path, hdu="SPECTRA")

    assert len(table) == 5
    by_target = {int(row["TARGETID"]): row for row in table}
    assert by_target[1]["GPDLF_STATUS"] == "completed"
    assert by_target[1]["GPDLF_N_ABSORBERS"] == 2
    # The case the legacy table loses entirely.
    assert by_target[2]["GPDLF_STATUS"] == "completed"
    assert by_target[2]["GPDLF_N_ABSORBERS"] == 0
    assert by_target[3]["GPDLF_STATUS"] == "quality_rejected"
    assert by_target[3]["GPDLF_REASON"] == "quality_policy_rejected"
    assert by_target[4]["GPDLF_STATUS"] == "insufficient_data"
    assert by_target[5]["GPDLF_STATUS"] == "failed"


def test_all_four_statuses_survive_a_round_trip(tmp_path, catalogue):
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    table = Table.read(path, hdu="SPECTRA")
    assert set(table["GPDLF_STATUS"]) == set(STATUSES)


def test_absorber_rows_join_to_their_parent(tmp_path, catalogue):
    """The identifier relationship a downstream analysis actually uses."""
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    absorbers = Table.read(path, hdu="DLACAT")
    spectra = Table.read(path, hdu="SPECTRA")

    parents = set(spectra["TARGETID"])
    assert set(absorbers["TARGETID"]) <= parents
    # DLAID is unique, and carries its parent.
    assert len(set(absorbers["DLAID"])) == len(absorbers)
    for row in absorbers:
        assert str(row["DLAID"]).startswith(f"{int(row['TARGETID'])}00")


def test_run_provenance_round_trips(tmp_path, catalogue):
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    run = dict(read_catalogue_metadata(path))
    assert run["GPDLF_SCHEMA_VERSION"] == CATALOGUE_SCHEMA_VERSION
    assert run["GPDLF_PRESET"] == "desi_y3"
    assert run["GPDLF_EVIDENCE_MODE"] == "exact"
    assert run["GPDLF_NUM_SAMPLES"] == 50_000
    assert run["GPDLF_COMPAT_PROFILE"] == "reference-d5b306e6"


def test_an_unknown_major_schema_version_is_refused(tmp_path, catalogue):
    """Silently accepting one is how a redefined column gets misread."""
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    with fits.open(path, mode="update") as hdul:
        for hdu in hdul:
            if "GPDLFVER" in hdu.header:
                hdu.header["GPDLFVER"] = "9.0"
    with pytest.raises(ValueError, match="major version 9"):
        read_catalogue_metadata(path)


def test_a_file_without_a_schema_version_is_refused(tmp_path):
    plain = tmp_path / "plain.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(plain)
    with pytest.raises(ValueError, match="carries no GPDLFVER"):
        read_catalogue_metadata(plain)


# --------------------------------------------------------------------------
# The no-samples guarantee
# --------------------------------------------------------------------------


def test_no_catalogue_column_holds_a_sample_array(tmp_path, catalogue):
    """Enforced by the schema allowlist, verified on the written file.

    Deliberately not implemented by rejecting arrays whose first dimension looks
    like a sample count: that heuristic fails in both directions. Every column
    that reaches a file comes from a schema, and the schema has no array columns.
    """
    for writer, name in (
        (write_catalogue, "ext.fits"),
        (write_legacy_catalogue, "legacy.fits"),
    ):
        path = writer(tmp_path / name, catalogue)
        with fits.open(path) as hdul:
            for hdu in hdul:
                if not isinstance(hdu, fits.BinTableHDU):
                    continue
                for column in hdu.columns:
                    # A FITS TFORM of "1D"/"D"/"64A" is scalar-per-row; a repeat
                    # count above one would be an array column.
                    repeat = "".join(ch for ch in column.format if ch.isdigit())
                    if column.format.endswith("A"):
                        continue  # strings carry their length in the repeat
                    assert repeat in ("", "1"), (
                        f"{hdu.name}.{column.name} has format {column.format!r}, "
                        "which is an array column; catalogues are summary-only"
                    )


def test_the_schemas_declare_no_array_columns():
    for schema in (ABSORBER_SCHEMA, SPECTRUM_SCHEMA, RUN_SCHEMA):
        for column in schema:
            assert column.dtype in {"int32", "int64", "float64", "str"}, (
                f"{column.name} declares {column.dtype!r}; catalogue columns are "
                "scalar per row"
            )


# --------------------------------------------------------------------------
# FILTER labelling (PI ruling, increment-12 clarification)
# --------------------------------------------------------------------------


def _row(targetid=1, index=0, mode="exact", **overrides):
    fields = dict(
        targetid=targetid,
        dlaid=reference_dlaid(targetid, index),
        z_qso=2.6,
        z_dla=2.3,
        nhi=20.5,
        p_dla=0.9,
        p_null=0.1,
        logp_dla=-0.1,
        logp_null=-2.3,
        model_p=0.9,
        log_evidence_absorber=-280.0,
        log_evidence_null=-284.0,
        evidence_mode=mode,
    )
    fields.update(overrides)
    return AbsorberRow(**fields)


def test_an_evidence_value_cannot_be_built_without_a_mode_label():
    """FILTER output may occupy an evidence field, but never unlabelled."""
    for bad in ("", "approximate", "EXACT", None):
        with pytest.raises(ValueError, match="evidence_mode must be one of"):
            _row(mode=bad)


def test_a_completed_spectrum_must_declare_a_mode():
    with pytest.raises(ValueError, match="must declare an evidence mode"):
        SpectrumRow(targetid=1, z_qso=2.6, status="completed", n_absorbers=0)


def test_the_mode_label_is_per_row_so_mixed_files_stay_readable(tmp_path):
    """A run-level label would be insufficient: one file can hold both modes."""
    spectra = [
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            n_absorbers=1,
            evidence_mode="exact",
        ),
        SpectrumRow(
            targetid=2,
            z_qso=2.5,
            status="completed",
            n_absorbers=1,
            evidence_mode="filter",
        ),
    ]
    absorbers = [
        _row(targetid=1, mode="exact"),
        _row(targetid=2, mode="filter", screening_score=0.87),
    ]
    path = write_catalogue(
        tmp_path / "mixed.fits", Catalogue(absorbers=absorbers, spectra=spectra)
    )

    table = Table.read(path, hdu="DLACAT")
    by_target = {int(r["TARGETID"]): r for r in table}
    assert by_target[1]["GPDLF_EVIDENCE_MODE"] == "exact"
    assert by_target[2]["GPDLF_EVIDENCE_MODE"] == "filter"

    with fits.open(path) as hdul:
        header = hdul["DLACAT"].header
        assert set(str(header["GPDLFMOD"]).split(",")) == {"exact", "filter"}
        comments = " ".join(str(c) for c in header["COMMENT"]).lower()
        assert "screening" in comments


def test_an_all_exact_file_says_so(tmp_path, catalogue):
    path = write_catalogue(tmp_path / "exact.fits", catalogue)
    with fits.open(path) as hdul:
        header = hdul["DLACAT"].header
        assert str(header["GPDLFMOD"]) == "exact"
        comments = " ".join(str(c) for c in header["COMMENT"]).lower()
        assert "exact-mode" in comments


def test_the_log_bayes_factor_is_functional_not_a_placeholder(tmp_path, catalogue):
    """It was advertised and always NaN, which is worse than absent."""
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    expected = [
        row.log_evidence_absorber - row.log_evidence_null for row in catalogue.absorbers
    ]
    assert np.allclose(table["GPDLF_LOG_BAYES_FACTOR"], expected)
    assert not np.any(np.isnan(table["GPDLF_LOG_BAYES_FACTOR"]))


# --------------------------------------------------------------------------
# Construction invariants
# --------------------------------------------------------------------------


def test_duplicate_target_ids_are_rejected():
    rows = [
        SpectrumRow(targetid=1, z_qso=2.6, status="completed", evidence_mode="exact"),
        SpectrumRow(targetid=1, z_qso=2.7, status="completed", evidence_mode="exact"),
    ]
    with pytest.raises(ValueError, match="duplicate spectrum TARGETID"):
        Catalogue(spectra=rows)


def test_duplicate_dlaids_are_rejected():
    spectrum = SpectrumRow(
        targetid=1, z_qso=2.6, status="completed", n_absorbers=2, evidence_mode="exact"
    )
    duplicated = [_row(targetid=1, index=0), _row(targetid=1, index=0)]
    with pytest.raises(ValueError, match="duplicate DLAID"):
        Catalogue(absorbers=duplicated, spectra=[spectrum])


def test_a_negative_absorber_count_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            n_absorbers=-1,
            evidence_mode="exact",
        )


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_probabilities_outside_the_unit_interval_are_rejected(value):
    with pytest.raises(ValueError, match="probability in"):
        _row(p_dla=value)
    with pytest.raises(ValueError, match="probability in"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            evidence_mode="exact",
            p_absorber=value,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_an_impossible_quality_fraction_is_rejected(value):
    with pytest.raises(ValueError, match="quality_fraction must be"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            evidence_mode="exact",
            quality_fraction=value,
        )


def test_a_catalogue_cannot_be_invalidated_after_construction():
    """A frozen dataclass holding the caller's list is not frozen."""
    spectra = [
        SpectrumRow(targetid=1, z_qso=2.6, status="completed", evidence_mode="exact")
    ]
    run = {"GPDLF_PRESET": "desi_y3"}
    catalogue = Catalogue(spectra=spectra, run=run)

    # Mutating what was passed in must not reach inside.
    spectra.append(
        SpectrumRow(targetid=1, z_qso=9.9, status="completed", evidence_mode="exact")
    )
    run["GPDLF_PRESET"] = "tampered"

    assert len(catalogue.spectra) == 1
    assert catalogue.run["GPDLF_PRESET"] == "desi_y3"
    with pytest.raises(TypeError):
        catalogue.run["GPDLF_PRESET"] = "also tampered"


def test_the_evaluated_count_is_recorded_per_spectrum(tmp_path):
    """FILTER can stop after different counts for different spectra."""
    spectra = [
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            evidence_mode="filter",
            n_evaluated=5000,
        ),
        SpectrumRow(
            targetid=2,
            z_qso=2.5,
            status="completed",
            evidence_mode="exact",
            n_evaluated=50_000,
        ),
    ]
    path = write_catalogue(tmp_path / "counts.fits", Catalogue(spectra=spectra))
    table = Table.read(path, hdu="SPECTRA")
    counts = {int(r["TARGETID"]): int(r["GPDLF_N_EVALUATED"]) for r in table}
    assert counts == {1: 5000, 2: 50_000}


# --------------------------------------------------------------------------
# The FITS surface itself, not just the column names
# --------------------------------------------------------------------------


def test_legacy_column_dtypes_and_identifier_format(tmp_path, catalogue):
    """TFORM, not just names: a downstream reader parses the binary layout."""
    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    with fits.open(path) as hdul:
        columns = {c.name: c.format for c in hdul["DLACAT"].columns}

    assert columns["TARGETID"] == "K"  # 64-bit integer, as the reference uses
    # int64, NOT int32. The reference declares DLAFLAG with numpy dtype="int",
    # which is 64-bit on the production platform. An earlier revision of this
    # test asserted "J" from a locally transcribed guess and was wrong.
    assert columns["DLAFLAG"] == "K"
    for name in ("RA", "DEC", "Z_QSO", "Z_DLA", "NHI", "P_DLA", "LOGP_DLA"):
        assert columns[name] == "D", f"{name} is {columns[name]}, expected float64"
    assert columns["DLAID"].endswith("A"), "DLAID must be a character column"


def test_units_are_declared_where_they_exist(tmp_path, catalogue):
    path = write_catalogue(tmp_path / "ext.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    assert str(table["RA"].unit) == "deg"
    assert str(table["DEC"].unit) == "deg"


def test_row_granularity_is_one_row_per_absorber(tmp_path, catalogue):
    path = write_legacy_catalogue(tmp_path / "legacy.fits", catalogue)
    table = Table.read(path, hdu="DLACAT")
    assert len(table) == len(catalogue.absorbers)
    assert len(table) == sum(row.n_absorbers for row in catalogue.spectra)


def test_every_declared_unit_is_a_fits_standard_unit():
    """A non-standard unit string makes every downstream reader warn.

    'log10(cm-2)' and 'nat' both did, and both are now carried in the column
    description instead. Interoperability is the point of the legacy surface, and
    a file that warns on open is not interoperable.
    """
    from astropy.units import Unit

    for schema in (ABSORBER_SCHEMA, SPECTRUM_SCHEMA, RUN_SCHEMA):
        for column in schema:
            if column.unit is None:
                continue
            Unit(column.unit, format="fits")  # raises on a non-standard string


def test_writing_a_catalogue_emits_no_warnings(tmp_path, catalogue):
    """The whole write path, with warnings promoted to errors."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_catalogue(tmp_path / "clean.fits", catalogue)
        write_legacy_catalogue(tmp_path / "clean-legacy.fits", catalogue)


# --------------------------------------------------------------------------
# The reference-generated datatype oracle (PI ruling, increment-13 correction 6)
# --------------------------------------------------------------------------

#: The reference's own `dtype=` tuple for the catalogue table, transcribed from
#: `dlasearch.py`. Note `"int"`, not `"int32"`/`"int64"`: numpy resolves it to the
#: platform default, which is 64-bit where the catalogues were produced.
REFERENCE_DTYPES = (
    "int",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "str",
    "float64",
    "float64",
    "float64",
    "float64",
    "int",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
)


def _locally_constructed_expectation(tmp_path, dlaid: str):
    """A FITS file built from the reference's declared conventions, transcribed here.

    Named for what it is (PI ruling, increment-14 correction 1). The ``names`` and
    ``dtype`` tuples are copied into this file, so this is an *independently
    constructed expectation*, not an oracle generated by the reference's own code.
    It is still worth having -- it caught DLAFLAG being int64 where a hand
    assertion said int32 -- but the transcription could itself drift.

    ``test_reference_parity.py`` carries the stronger check: it extracts the
    literal schema from the pinned reference source, so a drift in this
    transcription fails there.
    """
    values = [
        [39627000],
        [12.3],
        [4.5],
        [2.6],
        [3.0],
        [5.0],
        [dlaid],
        [2.3],
        [np.nan],
        [20.5],
        [np.nan],
        [0],
        [0.9],
        [0.1],
        [-0.1],
        [-2.3],
        [0.9],
    ]
    table = Table(values, names=list(REFERENCE_COLUMNS), dtype=REFERENCE_DTYPES)
    path = tmp_path / "oracle.fits"
    table.write(path, format="fits")
    return path


def test_strict_export_matches_the_locally_constructed_expectation(tmp_path):
    """Column order, numpy dtype and FITS TFORM against a transcribed schema."""
    targetid, index = 39627000, 0
    dlaid = reference_dlaid(targetid, index)

    spectrum = SpectrumRow(
        targetid=targetid,
        z_qso=2.6,
        status="completed",
        n_absorbers=1,
        evidence_mode="exact",
    )
    absorber = AbsorberRow(
        targetid=targetid,
        dlaid=dlaid,
        z_qso=2.6,
        z_dla=2.3,
        nhi=20.5,
        p_dla=0.9,
        p_null=0.1,
        logp_dla=-0.1,
        logp_null=-2.3,
        model_p=0.9,
        log_evidence_absorber=-280.0,
        log_evidence_null=-284.0,
        evidence_mode="exact",
        ra=12.3,
        dec=4.5,
        snr_forest=3.0,
        snr_redside=5.0,
    )
    ours = write_legacy_catalogue(
        tmp_path / "ours.fits", Catalogue(absorbers=[absorber], spectra=[spectrum])
    )
    oracle = _locally_constructed_expectation(tmp_path, dlaid)

    with fits.open(ours) as a, fits.open(oracle) as b:
        mine, theirs = a[1].columns, b[1].columns

        assert [c.name for c in mine] == [c.name for c in theirs], "column order"
        for column_a, column_b in zip(mine, theirs, strict=True):
            assert column_a.format == column_b.format, (
                f"{column_a.name}: TFORM {column_a.format!r} != reference "
                f"{column_b.format!r}"
            )

        # Identifier bytes, not just a similar-looking string.
        # 39627000 + "00" + "0"; both files must agree byte for byte.
        assert a[1].data["DLAID"][0] == b[1].data["DLAID"][0] == dlaid
        assert dlaid == "39627000000"
        # Null representation for the uncertainty columns.
        assert np.isnan(a[1].data["Z_DLA_ERR"][0])
        assert np.isnan(b[1].data["Z_DLA_ERR"][0])

    # And the numpy dtypes a downstream reader sees.
    mine_table = Table.read(ours, hdu="DLACAT")
    theirs_table = Table.read(oracle)
    for name in REFERENCE_COLUMNS:
        assert mine_table[name].dtype == theirs_table[name].dtype, (
            f"{name}: {mine_table[name].dtype} != reference {theirs_table[name].dtype}"
        )


def test_row_granularity_matches_the_reference_for_multiple_absorbers(tmp_path):
    """One row per absorber, parent fields repeated -- the reference's convention."""
    spectrum = SpectrumRow(
        targetid=7,
        z_qso=2.6,
        status="completed",
        n_absorbers=3,
        evidence_mode="exact",
    )
    absorbers = [_row(targetid=7, index=i) for i in range(3)]
    path = write_legacy_catalogue(
        tmp_path / "multi.fits", Catalogue(absorbers=absorbers, spectra=[spectrum])
    )
    table = Table.read(path, hdu="DLACAT")

    assert len(table) == 3
    assert list(table["DLAID"]) == ["7000", "7001", "7002"]
    assert len(set(table["TARGETID"])) == 1
    assert len(set(table["Z_QSO"])) == 1


# --------------------------------------------------------------------------
# Cross-row consistency (PI ruling, increment-14 correction 4)
# --------------------------------------------------------------------------


def test_a_dlaid_that_does_not_follow_its_parent_is_rejected():
    """Checked at construction, not only on written output.

    A row built by hand could otherwise reach a writer with an identifier no
    downstream join would resolve.
    """
    with pytest.raises(ValueError, match="reference convention"):
        AbsorberRow(
            targetid=1,
            dlaid="2000",  # belongs to TARGETID 2
            z_qso=2.6,
            z_dla=2.3,
            nhi=20.5,
            p_dla=0.9,
            p_null=0.1,
            logp_dla=-0.1,
            logp_null=-2.3,
            model_p=0.9,
            log_evidence_absorber=-1.0,
            log_evidence_null=-2.0,
            evidence_mode="exact",
        )
    with pytest.raises(ValueError, match="reference convention"):
        AbsorberRow(
            targetid=1,
            dlaid="1-0",  # the hyphenated form this project used to invent
            z_qso=2.6,
            z_dla=2.3,
            nhi=20.5,
            p_dla=0.9,
            p_null=0.1,
            logp_dla=-0.1,
            logp_null=-2.3,
            model_p=0.9,
            log_evidence_absorber=-1.0,
            log_evidence_null=-2.0,
            evidence_mode="exact",
        )


@pytest.mark.parametrize("field", ["p_dla", "p_null", "logp_null", "z_qso"])
def test_spectrum_level_fields_must_agree_across_a_spectrums_rows(field):
    """If they disagree, a reader taking "the" value gets an arbitrary one."""
    spectrum = SpectrumRow(
        targetid=1, z_qso=2.6, status="completed", n_absorbers=2, evidence_mode="exact"
    )
    rows = [_row(targetid=1, index=0), _row(targetid=1, index=1, **{field: 0.42})]
    with pytest.raises(ValueError, match="spectrum-level quantity"):
        Catalogue(absorbers=rows, spectra=[spectrum])


def test_absorber_rows_of_one_spectrum_cannot_mix_evidence_modes():
    spectrum = SpectrumRow(
        targetid=1, z_qso=2.6, status="completed", n_absorbers=2, evidence_mode="exact"
    )
    rows = [
        _row(targetid=1, index=0, mode="exact"),
        _row(targetid=1, index=1, mode="filter"),
    ]
    with pytest.raises(ValueError, match="mix evidence modes"):
        Catalogue(absorbers=rows, spectra=[spectrum])


def test_absorber_and_parent_modes_must_agree():
    spectrum = SpectrumRow(
        targetid=1, z_qso=2.6, status="completed", n_absorbers=1, evidence_mode="filter"
    )
    with pytest.raises(ValueError, match="the spectrum row declares"):
        Catalogue(absorbers=[_row(targetid=1, mode="exact")], spectra=[spectrum])


def test_absorber_indices_must_be_contiguous_from_zero():
    """A downstream join on DLAID depends on the reference's 0..n-1 indexing."""
    spectrum = SpectrumRow(
        targetid=1, z_qso=2.6, status="completed", n_absorbers=2, evidence_mode="exact"
    )
    rows = [_row(targetid=1, index=0), _row(targetid=1, index=2)]
    with pytest.raises(ValueError, match="not 0..1"):
        Catalogue(absorbers=rows, spectra=[spectrum])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("p_absorber", 0.9),
        ("log_evidence_null", -284.0),
        ("log_evidence_absorber", -280.0),
    ],
)
def test_a_non_completed_spectrum_cannot_carry_inference_results(field, value):
    """A rejected row carrying an evidence would be read as a real measurement."""
    with pytest.raises(ValueError, match="cannot carry inference results"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="quality_rejected",
            reason="quality_policy_rejected",
            **{field: value},
        )


def test_a_non_completed_spectrum_cannot_declare_a_mode_or_evaluated_samples():
    with pytest.raises(ValueError, match="cannot declare an evidence mode"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="failed",
            reason="numerical_error",
            evidence_mode="exact",
        )
    with pytest.raises(ValueError, match="cannot have evaluated"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="failed",
            reason="numerical_error",
            n_evaluated=5000,
        )


def test_a_completed_spectrum_cannot_carry_a_failure_reason():
    with pytest.raises(ValueError, match="must not carry a failure reason"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            evidence_mode="exact",
            reason="numerical_error",
        )


@pytest.mark.parametrize("field", ["n_usable_pixels", "n_evaluated"])
def test_negative_counts_are_rejected(field):
    with pytest.raises(ValueError, match="cannot be negative"):
        SpectrumRow(
            targetid=1,
            z_qso=2.6,
            status="completed",
            evidence_mode="exact",
            **{field: -1},
        )


def test_caller_metadata_cannot_override_the_schema_version():
    """The version is writer-controlled: a caller claiming another is a lie."""
    spectrum = SpectrumRow(
        targetid=1, z_qso=2.6, status="completed", evidence_mode="exact"
    )
    with pytest.raises(ValueError, match="writer-controlled"):
        Catalogue(spectra=[spectrum], run={"GPDLF_SCHEMA_VERSION": "99.0"})

    # The correct version is accepted, so a round-tripped record can be reused.
    Catalogue(
        spectra=[spectrum], run={"GPDLF_SCHEMA_VERSION": CATALOGUE_SCHEMA_VERSION}
    )


def test_schema_versions_must_agree_across_hdus(tmp_path, catalogue):
    """A file with two versions cannot be interpreted; refuse rather than guess."""
    path = write_catalogue(tmp_path / "mixed-version.fits", catalogue)
    with fits.open(path, mode="update") as hdul:
        hdul["SPECTRA"].header["GPDLFVER"] = "2.0"
    with pytest.raises(ValueError, match="inconsistent schema versions"):
        read_catalogue_metadata(path)


# --- schema versioning, asserted rather than assumed -------------------------
#
# The schema documents its own rule: adding a column bumps the MINOR part, and a
# reader that checks the major part keeps working. Increment 17 added four
# columns while the version stayed at 1.0, which contradicted that rule. These
# tests make the rule enforceable instead of aspirational.


def test_the_schema_version_is_the_one_the_columns_describe():
    assert CATALOGUE_SCHEMA_VERSION == "1.2"
    # The columns 1.1 exists for.
    spectrum_columns = {c.name for c in SPECTRUM_SCHEMA}
    assert "GPDLF_SCREENING_SCORE" in spectrum_columns
    assert "GPDLF_SCREENING_N_EVALUATED" in spectrum_columns
    run_columns = {c.name for c in RUN_SCHEMA}
    assert "GPDLF_BASE_PRESET" in run_columns
    assert "GPDLF_CONFIG_DIGEST" in run_columns
    # The column 1.2 exists for.
    assert "GPDLF_MAX_DLAS" in run_columns


def test_the_minor_bump_added_columns_and_removed_no_meaning():
    """1.2 must still be a minor bump: every 1.1 column still present.

    The MODELS HDU left FITS in 1.2, but it was never an ABSORBER/SPECTRUM/RUN
    column, so no reader keyed on those tables loses a field. If a future edit
    drops one of these, the version rule says that is a MAJOR bump and this
    test is where it gets caught.
    """
    absorber = {c.name for c in ABSORBER_SCHEMA}
    run = {c.name for c in RUN_SCHEMA}
    spectrum = {c.name for c in SPECTRUM_SCHEMA}
    for name in (
        "P_DLA",
        "P_NULL",
        "LOGP_DLA",
        "LOGP_NULL",
        "MODEL_P",
        "DLAID",
        "TARGETID",
    ):
        assert name in absorber
    for name in ("GPDLF_BASE_PRESET", "GPDLF_CONFIG_DIGEST", "GPDLF_EXPERIMENTAL"):
        assert name in run
    for name in ("GPDLF_SCREENING_SCORE", "GPDLF_SCREENING_N_EVALUATED"):
        assert name in spectrum


def test_a_reader_checking_the_major_version_accepts_this_file(catalogue, tmp_path):
    """The property the minor-bump rule is supposed to buy.

    A consumer written against 1.0 checks the major part and must still read a
    1.1 file. Asserted by actually reading one, not by reasoning about it.
    """
    path = tmp_path / "v11.fits"
    write_catalogue(path, catalogue)

    metadata = read_catalogue_metadata(path)
    assert metadata["GPDLF_SCHEMA_VERSION"] == "1.2"
    assert metadata["GPDLF_SCHEMA_VERSION"].split(".")[0] == "1"


def test_a_future_major_version_is_refused(catalogue, tmp_path):
    """The other half: a 2.x file must not be read as though it were 1.x."""
    path = tmp_path / "v2.fits"
    write_catalogue(path, catalogue)

    with fits.open(path, mode="update") as hdul:
        for hdu in hdul:
            if "GPDLFVER" in hdu.header:
                hdu.header["GPDLFVER"] = "2.0"

    with pytest.raises(ValueError, match="major version"):
        read_catalogue_metadata(path)


def test_every_hdu_carries_the_same_bumped_version(catalogue, tmp_path):
    path = tmp_path / "all-hdus.fits"
    write_catalogue(path, catalogue)

    with fits.open(path) as hdul:
        versions = {hdu.header["GPDLFVER"] for hdu in hdul if "GPDLFVER" in hdu.header}
    assert versions == {"1.2"}, f"HDUs disagree on schema version: {versions}"
