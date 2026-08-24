"""FITS writers for the compact legacy and extended catalog products.

Use the legacy product when old downstream code needs the historical table. Use
the extended product when you also need to know what happened to every spectrum.

:func:`write_legacy_catalogue`
    exactly the historical DESI columns, in the historical order, one row per
    absorber, in a single binary table. This product supports downstream software
    that expects the historical schema. It cannot represent a null or rejected
    spectrum, which is why the extended product also exists.

:func:`write_catalogue`
    the same absorber table plus a per-spectrum status table and a run/provenance
    extension.

The no-samples guarantee
------------------------
Catalog output contains **no QMC or posterior sample arrays**. An explicit
schema allowlist enforces this restriction.

astropy is an optional dependency. It is imported inside the functions so the
inference core keeps needing nothing but NumPy and SciPy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..catalogue import (
    ABSORBER_SCHEMA,
    CATALOGUE_SCHEMA_VERSION,
    LEGACY_ABSORBER_COLUMNS,
    RUN_SCHEMA,
    SPECTRUM_SCHEMA,
    Catalogue,
    Column,
)

__all__ = [
    "read_catalogue_metadata",
    "write_catalogue",
    "write_legacy_catalogue",
]

#: HDU names in the extended product. `EXTNAME` is how a reader finds a table
#: without depending on extension order.
ABSORBER_HDU = "DLACAT"
SPECTRUM_HDU = "SPECTRA"
RUN_HDU = "RUNINFO"

_NUMPY_DTYPE = {
    "int64": np.int64,
    "int32": np.int32,
    "float64": np.float64,
    "str": np.str_,
    # FITS logical columns ('L'). Used by the model ladder to distinguish
    # "evaluated and unlikely" from "never evaluated", which a float cannot.
    "bool": np.bool_,
}


def _require_astropy():
    try:
        from astropy.io import fits
        from astropy.table import Table
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "writing FITS catalogues needs astropy; install "
            "gp_dla_finder[desi] or gp_dla_finder[sdss]"
        ) from exc
    return fits, Table


def _empty_column(column: Column, n_rows: int):
    if column.dtype == "str":
        return np.full(n_rows, "", dtype="U1")
    if column.dtype.startswith("int"):
        return np.zeros(n_rows, dtype=_NUMPY_DTYPE[column.dtype])
    return np.full(n_rows, np.nan, dtype=np.float64)


def _table_from_rows(
    schema: Sequence[Column],
    rows: Sequence[Any],
    mapping: Mapping[str, str],
    *,
    include: Sequence[str] | None = None,
):
    """Build a Table from dataclass rows, one schema column at a time.

    ``mapping`` names the attribute backing each column. A column with no mapping
    is filled with the schema's empty value rather than silently dropped, because
    a reader expecting the column should find it present and empty rather than
    missing.
    """
    _, Table = _require_astropy()

    wanted = [c for c in schema if include is None or c.name in include]
    table = Table()
    for column in wanted:
        attribute = mapping.get(column.name)
        if attribute is None:
            table[column.name] = _empty_column(column, len(rows))
        elif column.dtype == "str":
            # Width follows the content, as the reference's Table(dtype="str")
            # does. A fixed U64 would emit TFORM '64A' where the reference emits
            # '11A' for the same data, which is a datatype divergence even though
            # the values match.
            values = [str(getattr(row, attribute)) for row in rows]
            table[column.name] = (
                np.array(values, dtype=str) if values else np.array([], dtype="U1")
            )
        else:
            table[column.name] = np.array(
                [getattr(row, attribute) for row in rows],
                dtype=_NUMPY_DTYPE[column.dtype],
            )
        if column.unit:
            table[column.name].unit = column.unit
        table[column.name].description = column.description[:70]
    return table


_ABSORBER_MAPPING = {
    "TARGETID": "targetid",
    "RA": "ra",
    "DEC": "dec",
    "Z_QSO": "z_qso",
    "SNR_FOREST": "snr_forest",
    "SNR_REDSIDE": "snr_redside",
    "DLAID": "dlaid",
    "Z_DLA": "z_dla",
    "NHI": "nhi",
    "DLAFLAG": "dlaflag",
    "P_DLA": "p_dla",
    "P_NULL": "p_null",
    # Carried independently, NOT derived. In the reference LOGP_DLA is
    # log_posteriors_dla[n], a per-absorber-index quantity, not log(P_DLA).
    "LOGP_DLA": "logp_dla",
    "LOGP_NULL": "logp_null",
    "MODEL_P": "model_p",
    "Z_DLA_ERR": "z_dla_err",
    "NHI_ERR": "nhi_err",
    "GPDLF_LOG_EVIDENCE_ABSORBER": "log_evidence_absorber",
    "GPDLF_LOG_EVIDENCE_NULL": "log_evidence_null",
    "GPDLF_EVIDENCE_MODE": "evidence_mode",
    "GPDLF_SCREENING_SCORE": "screening_score",
    "GPDLF_LOG_BAYES_FACTOR": "log_bayes_factor",
}

_SPECTRUM_MAPPING = {
    "TARGETID": "targetid",
    "RA": "ra",
    "DEC": "dec",
    "Z_QSO": "z_qso",
    "GPDLF_STATUS": "status",
    "GPDLF_REASON": "reason",
    "GPDLF_N_ABSORBERS": "n_absorbers",
    "GPDLF_P_ABSORBER": "p_absorber",
    "GPDLF_LOG_EVIDENCE_NULL": "log_evidence_null",
    "GPDLF_LOG_EVIDENCE_ABSORBER": "log_evidence_absorber",
    "GPDLF_EVIDENCE_MODE": "evidence_mode",
    "GPDLF_QUALITY_FRACTION": "quality_fraction",
    "GPDLF_N_USABLE_PIXELS": "n_usable_pixels",
    "GPDLF_N_EVALUATED": "n_evaluated",
    "GPDLF_SCREENING_SCORE": "screening_score",
    "GPDLF_SCREENING_N_EVALUATED": "screening_n_evaluated",
}


def _absorber_table(rows, *, include=None):
    """Every column comes straight from the row.

    Nothing is derived. An earlier revision computed ``LOGP_DLA = log(P_DLA)``,
    which is wrong: in the reference ``LOGP_DLA`` is ``log_posteriors_dla[n]``, a
    per-absorber-index quantity, while ``P_DLA`` is spectrum-level. They coincide
    only in special cases and diverge as soon as more than one absorber is
    modelled, so deriving one from the other silently corrupted the column.
    """
    return _table_from_rows(ABSORBER_SCHEMA, rows, _ABSORBER_MAPPING, include=include)


def _run_header(fits, run: Mapping[str, object]):
    header = fits.Header()
    header["GPDLFVER"] = (CATALOGUE_SCHEMA_VERSION, "gp_dla_finder catalogue schema")
    known = {column.name for column in RUN_SCHEMA}
    for key, value in run.items():
        if key not in known:
            continue
        # FITS keywords are 8 characters; the full name goes in a record card.
        header[f"HIERARCH {key}"] = value if value is not None else ""
    return header


def _require_mode_labels(catalogue: Catalogue) -> None:
    """No evidence value leaves this package without a co-located mode label.

    Run-level labelling would not be enough: one file can hold both modes, and a
    reader looking at a row must be able to distinguish a full-grid value from
    a FILTER screening value without consulting anything else.
    """
    from ..catalogue import EVIDENCE_MODES

    for row in catalogue.absorbers:
        if row.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"absorber {row.dlaid} carries evidence values with mode "
                f"{row.evidence_mode!r}; refusing to write an unlabelled evidence"
            )
    for row in catalogue.spectra:
        if row.status == "completed" and row.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"spectrum {row.targetid} completed but declares mode "
                f"{row.evidence_mode!r}; refusing to write an unlabelled evidence"
            )


def _mode_comments(hdu, catalogue: Catalogue) -> None:
    """State in the header what the evidence columns of this file actually hold."""
    modes = {row.evidence_mode for row in catalogue.absorbers}
    modes |= {r.evidence_mode for r in catalogue.spectra if r.status == "completed"}
    if "filter" in modes:
        hdu.header["COMMENT"] = (
            "Contains FILTER rows: where GPDLF_EVIDENCE_MODE='filter' the "
            "evidence columns hold APPROXIMATE SCREENING values, not exact "
            "evidences, and must not be used for model comparison."
        )
    if modes == {"exact"}:
        hdu.header["COMMENT"] = "All rows are exact-mode evidences."
    hdu.header["GPDLFMOD"] = (
        ",".join(sorted(modes)) or "none",
        "evidence modes present in this file",
    )


def _search_depth_header(hdu, catalogue: Catalogue) -> None:
    """Record how many absorber models were searched per spectrum.

    Without it a reader cannot interpret ``P_DLA``. The reference defines it as
    the posterior summed over every absorber model in the ladder, so the same
    column means P(M1) on a one-absorber run and P(M1)+P(M2) on a two-absorber
    one, and nothing else in the file distinguishes them.

    This is the search LIMIT. The number of absorbers found in a spectrum is the
    number of rows carrying its ``TARGETID``.
    """
    max_dlas = catalogue.run.get("GPDLF_MAX_DLAS")
    if max_dlas is None:
        return
    hdu.header["HIERARCH GPDLF_MAX_DLAS"] = (
        int(max_dlas),
        "max absorbers SEARCHED per spectrum, not detected",
    )
    if int(max_dlas) >= 2:
        hdu.header["COMMENT"] = (
            "Multi-absorber run: a spectrum may contribute several rows sharing "
            f"a TARGETID. P_DLA is summed over all {int(max_dlas)} absorber "
            "models; MODEL_P and LOGP_DLA are per-row, for that row's model only."
        )


def write_legacy_catalogue(path: str | Path, catalogue: Catalogue) -> Path:
    """Write the strict legacy view: historical columns only, one HDU.

    The DESI-compatible catalogue product. Flat, one row per absorber: a
    spectrum with two selected absorbers contributes two ordinary rows sharing a
    ``TARGETID``, with contiguous ``DLAID`` values. There is no nesting, no
    variable-length column and no model-ladder table.

    ``Z_DLA_ERR`` and ``NHI_ERR`` are written as **NaN**, documented as such in
    the header. No estimator for them has been chosen or validated, and emitting
    a plausible-looking number would be worse than emitting nothing.

    This product **cannot** represent a spectrum with no absorber. Use
    :func:`write_catalogue` when that matters — which is whenever the question is
    about a population rather than a list. Neither product carries the model
    ladder: that is
    :func:`gp_dla_finder.io.structured.write_structured_results`.
    """
    fits, _ = _require_astropy()
    _require_mode_labels(catalogue)
    path = Path(path)

    table = _absorber_table(catalogue.absorbers, include=LEGACY_ABSORBER_COLUMNS)
    hdu = fits.BinTableHDU(table, name=ABSORBER_HDU)
    hdu.header["GPDLFVER"] = (CATALOGUE_SCHEMA_VERSION, "gp_dla_finder schema")
    hdu.header["GPDLFVW"] = ("strict-legacy", "historical columns only")
    _search_depth_header(hdu, catalogue)
    hdu.header["COMMENT"] = (
        "Z_DLA_ERR/NHI_ERR are NaN unless a validated estimator supplied them."
    )
    _mode_comments(hdu, catalogue)
    hdu.header["COMMENT"] = (
        "This view cannot represent spectra with no absorber. See the extended "
        "product for per-spectrum status."
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return path


def write_catalogue(path: str | Path, catalogue: Catalogue) -> Path:
    """Write the extended product: absorbers, per-spectrum status, and run info.

    The absorber HDU keeps the legacy columns and order, so a reader that only
    knows the historical schema still works on it, and adds the new,
    unambiguously named evidence fields after them. Three flat tables, and no
    model-ladder HDU: the ladder is
    :func:`gp_dla_finder.io.structured.write_structured_results`.
    """
    fits, Table = _require_astropy()
    _require_mode_labels(catalogue)
    path = Path(path)

    absorbers = _absorber_table(catalogue.absorbers)
    spectra = _table_from_rows(SPECTRUM_SCHEMA, catalogue.spectra, _SPECTRUM_MAPPING)

    absorber_hdu = fits.BinTableHDU(absorbers, name=ABSORBER_HDU)
    absorber_hdu.header["GPDLFVER"] = (CATALOGUE_SCHEMA_VERSION, "schema version")
    absorber_hdu.header["GPDLFVW"] = ("extended", "legacy columns plus extensions")
    _search_depth_header(absorber_hdu, catalogue)
    absorber_hdu.header["COMMENT"] = (
        "Z_DLA_ERR/NHI_ERR are NaN unless a validated estimator supplied them."
    )
    _mode_comments(absorber_hdu, catalogue)

    spectrum_hdu = fits.BinTableHDU(spectra, name=SPECTRUM_HDU)
    spectrum_hdu.header["GPDLFVER"] = (CATALOGUE_SCHEMA_VERSION, "schema version")
    spectrum_hdu.header["COMMENT"] = (
        "One row per ATTEMPTED spectrum, including those with no absorber."
    )

    run_hdu = fits.ImageHDU(name=RUN_HDU, header=_run_header(fits, catalogue.run))

    hdus = [fits.PrimaryHDU(), absorber_hdu, spectrum_hdu, run_hdu]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def read_catalogue_metadata(path: str | Path) -> Mapping[str, object]:
    """Read schema version and run provenance back out of a written catalogue.

    Raises
    ------
    ValueError
        If the file's schema major version is not one this package understands.
        A reader that silently accepts an unknown major version is how a column
        whose meaning changed gets read as though it had not.
    """
    fits, _ = _require_astropy()
    with fits.open(path) as hdul:
        # Collect every version present, not just the first: a file whose HDUs
        # disagree cannot be interpreted, and taking the first would silently
        # pick one.
        versions = {
            str(hdu.header["GPDLFVER"]) for hdu in hdul if "GPDLFVER" in hdu.header
        }
        if len(versions) > 1:
            raise ValueError(
                f"{path} has inconsistent schema versions across its HDUs: "
                f"{sorted(versions)}"
            )
        version = next(iter(versions), None)
        if version is None:
            raise ValueError(
                f"{path} carries no GPDLFVER: not a gp_dla_finder catalogue"
            )

        major = version.split(".")[0]
        expected = CATALOGUE_SCHEMA_VERSION.split(".")[0]
        if major != expected:
            raise ValueError(
                f"catalogue schema version {version} has major version {major}; "
                f"this package understands {expected}.x only"
            )

        run: dict[str, object] = {"GPDLF_SCHEMA_VERSION": version}
        if RUN_HDU in [h.name for h in hdul]:
            header = hdul[RUN_HDU].header
            for column in RUN_SCHEMA:
                if column.name in header:
                    run[column.name] = header[column.name]
        return run
