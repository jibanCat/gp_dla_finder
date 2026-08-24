"""The structured result: everything the flat FITS catalogue does not carry.

The FITS product follows the DESI catalogue: flat, fixed columns, and one row per
absorber. We keep that layout because downstream analyses already read it.

The M0/M1/M2 ladder does not fit a flat table. Its length varies with the run,
and its posteriors are model-level rather than absorber-level. So it travels
here instead, in JSON:

* every model's log evidence, log prior and posterior;
* whether each rung was **evaluated at all**, and which was selected;
* which absorber candidate belongs to which model -- taken from the candidate
  itself, not inferred from its position in the list;
* the run provenance, unchanged from the FITS run record.

JSON works here because the structured result is small -- one object per
spectrum, not per pixel -- and it needs no dependency beyond the standard
library. If you later retain posterior samples or full inference state, use a
separate HDF5 product instead.

Strict JSON, and how a missing value stays distinguishable
----------------------------------------------------------
The output is **strict RFC 8259**: every non-finite number is written as
``null``, and serialisation runs with ``allow_nan=False`` so a future non-finite
value cannot silently reintroduce a bare ``NaN`` or ``Infinity`` literal. Any
JSON parser reads these files, not only Python's.

``null`` on its own would lose something, so it does not carry the meaning
alone. Each model states whether it ran:

.. code-block:: text

    evaluated = false, posterior = null   # the rung was never evaluated
    evaluated = true,  posterior = 0.0    # evaluated, and its posterior is zero

That is the distinction the ladder exists to record, and ``evaluated`` is what
records it. A ``null`` says only "no number here".
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..catalogue import CATALOGUE_SCHEMA_VERSION, Catalogue

__all__ = [
    "STRUCTURED_FORMAT_VERSION",
    "read_structured_results",
    "selected_models",
    "structured_payload",
    "write_structured_results",
]

#: Bumped when the shape of the JSON document changes. Independent of the FITS
#: schema version: the two products can move separately, and pinning them
#: together would force a FITS bump for a JSON-only change.
STRUCTURED_FORMAT_VERSION = "1.0"


def _json_number(value: float | None) -> float | None:
    """``None`` for any non-finite value, the number otherwise.

    Applied to **every** numeric field on the way out, so the document is strict
    RFC 8259 rather than Python's ``NaN``/``Infinity`` dialect. Infinities are
    included: an infinite log evidence is as unrepresentable in JSON as a NaN,
    and letting one through would be the same bug discovered later.

    What a ``null`` means is carried by its neighbours -- ``evaluated`` on a
    model, ``status`` on a spectrum -- never by the null itself.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def structured_payload(catalogue: Catalogue) -> dict:
    """The JSON document, as a dict, without writing it.

    Separate from the writer so it can be embedded in another document, sent
    over a wire, or asserted against in a test without a temporary file.
    """
    by_spectrum: dict[int, dict] = {}

    for row in catalogue.spectra:
        by_spectrum[row.targetid] = {
            "targetid": row.targetid,
            "z_qso": row.z_qso,
            "status": row.status,
            "reason": row.reason,
            "evidence_mode": row.evidence_mode,
            "n_absorbers": row.n_absorbers,
            "p_absorber": _json_number(row.p_absorber),
            "log_evidence_null": _json_number(row.log_evidence_null),
            "log_evidence_absorber": _json_number(row.log_evidence_absorber),
            "n_evaluated": row.n_evaluated,
            "quality_fraction": _json_number(row.quality_fraction),
            "n_usable_pixels": row.n_usable_pixels,
            "screening_score": _json_number(row.screening_score),
            "screening_n_evaluated": row.screening_n_evaluated,
            "models": [],
            "absorbers": [],
        }

    for model in catalogue.models:
        spectrum = by_spectrum.get(model.targetid)
        if spectrum is None:
            raise ValueError(
                f"model row for TARGETID {model.targetid} has no spectrum row; "
                "the structured document is keyed on spectra and would lose it"
            )
        spectrum["models"].append(
            {
                "model_index": model.model_index,
                "model_label": model.model_label,
                # A rung that was never evaluated has no evidence, so it is
                # null. `evaluated` below is what says which of "not measured"
                # and "measured as zero" this is -- the null does not.
                "log_evidence": _json_number(model.log_evidence),
                "log_prior": _json_number(model.log_prior),
                "posterior": _json_number(model.posterior),
                "evaluated": model.evaluated,
                "selected": model.selected,
            }
        )

    for absorber in catalogue.absorbers:
        spectrum = by_spectrum.get(absorber.targetid)
        if spectrum is None:
            raise ValueError(
                f"absorber {absorber.dlaid} has no spectrum row; the structured "
                "document is keyed on spectra and would lose it"
            )
        # Which model this absorber belongs to. The reference's flat catalogue
        # leaves this implicit in the row count; here it is stated, because a
        # consumer reading one absorber should not have to count its siblings.
        spectrum["absorbers"].append(
            {
                "dlaid": absorber.dlaid,
                # Two different questions, and they have different answers.
                # `absorber_index` is which member of the selected model this
                # is; `model_index` is which model that is. For a selected M2
                # pair both members carry model_index 2, with absorber_index 0
                # and 1.
                "absorber_index": len(spectrum["absorbers"]),
                "model_index": absorber.model_index,
                "model_label": f"M{absorber.model_index}",
                "z_dla": _json_number(absorber.z_dla),
                "nhi": _json_number(absorber.nhi),
                # NaN until a validated estimator supplies one, so null here.
                "z_dla_err": _json_number(absorber.z_dla_err),
                "nhi_err": _json_number(absorber.nhi_err),
                "p_dla": _json_number(absorber.p_dla),
                "p_null": _json_number(absorber.p_null),
                "logp_dla": _json_number(absorber.logp_dla),
                "logp_null": _json_number(absorber.logp_null),
                "model_p": _json_number(absorber.model_p),
                "evidence_mode": absorber.evidence_mode,
                "log_evidence_absorber": _json_number(absorber.log_evidence_absorber),
                "log_evidence_null": _json_number(absorber.log_evidence_null),
            }
        )

    return {
        "format": "gp_dla_finder.structured",
        "format_version": STRUCTURED_FORMAT_VERSION,
        "catalogue_schema_version": CATALOGUE_SCHEMA_VERSION,
        "run": dict(catalogue.run),
        "spectra": list(by_spectrum.values()),
    }


def write_structured_results(path: str | Path, catalogue: Catalogue) -> Path:
    """Write the complete structured result, ladder included.

    The companion to the FITS catalogue rather than a replacement for it: write
    both when the ladder matters, and point analyses that only need a DLA list
    at the FITS file.
    """
    path = Path(path)
    payload = structured_payload(catalogue)
    # allow_nan=False is the guard, not the formatting. Every numeric field
    # already goes through _json_number, so this should never fire -- and if a
    # field is ever added without it, this raises at write time instead of
    # producing a file that only Python can read.
    try:
        text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False)
    except ValueError as error:
        raise ValueError(
            f"refusing to write {path}: the document contains a non-finite "
            f"number, which is not valid JSON ({error}). Every numeric field "
            "must pass through _json_number()."
        ) from error
    path.write_text(text + "\n")
    return path


def read_structured_results(path: str | Path) -> Mapping[str, object]:
    """Read a structured result back.

    Raises
    ------
    ValueError
        If the document is not a structured result, or its major format version
        is one this package does not understand. Reading an unknown major
        version silently is how a field whose meaning changed gets misread.
    """

    # parse_constant fires on NaN/Infinity/-Infinity, which strict JSON does not
    # have. A file carrying them was not written by this package, or was edited
    # by something that speaks Python's dialect.
    def _reject(literal: str) -> object:
        raise ValueError(
            f"{path} contains the non-standard JSON literal {literal!r}; "
            "this package writes strict RFC 8259, using null with an explicit "
            "'evaluated' field instead"
        )

    payload = json.loads(Path(path).read_text(), parse_constant=_reject)
    if not isinstance(payload, dict) or payload.get("format") != (
        "gp_dla_finder.structured"
    ):
        raise ValueError(f"{path} is not a gp_dla_finder structured result")

    version = str(payload.get("format_version", ""))
    major = version.split(".")[0]
    expected = STRUCTURED_FORMAT_VERSION.split(".")[0]
    if major != expected:
        raise ValueError(
            f"structured result format version {version} has major version "
            f"{major}; this package understands {expected}.x only"
        )
    return payload


def selected_models(payload: Mapping[str, object]) -> dict[int, str]:
    """``{targetid: 'M0' | 'M1' | ...}`` for every spectrum that has a ladder.

    A spectrum with no ladder is absent rather than defaulted: a run that never
    evaluated the models has no selection to report, and reporting ``M0`` for it
    would claim a measurement that was not made.
    """
    out: dict[int, str] = {}
    spectra: Sequence[Mapping[str, object]] = payload.get("spectra", ())  # type: ignore[assignment]
    for spectrum in spectra:
        models: Sequence[Mapping[str, object]] = spectrum.get("models", ())  # type: ignore[assignment]
        for model in models:
            if model.get("selected"):
                out[int(spectrum["targetid"])] = str(model["model_label"])
                break
    return out
