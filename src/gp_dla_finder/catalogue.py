"""Catalog schemas for both DESI compatibility and complete run accounting.

The deployed DESI DLA catalog is a flat, one-row-per-DLA FITS table, and
downstream analyses already read it. We preserve its column names, data types,
row granularity, and identifier conventions as the compatibility surface.

A DLA-only table has one important blind spot: **a spectrum with no selected DLA
produces no row.** That is enough for a DLA list, but it cannot distinguish a
null result from a quality rejection, an inference failure, or a spectrum that
was never processed. The three schemas are:

:data:`ABSORBER_SCHEMA`
    one row per absorber. The legacy compatibility surface.
:data:`SPECTRUM_SCHEMA`
    one row per *attempted* spectrum, carrying its status and reason code. Null
    detections and rejections live here.
:data:`RUN_SCHEMA`
    run-level configuration and provenance, recorded once rather than repeated on
    every absorber row.

Two FITS products are built from them: a **strict legacy** export containing
only the absorber table with exactly the historical columns, for readers that
require that schema, and an **extended** product carrying all three. Both are
flat: one row per absorber, one row per spectrum, no nesting and no
variable-length arrays.

FITS is the compact catalogue, not the inference record
-------------------------------------------------------
The FITS product is the compact summary used by the DESI workflow. The full
M0/M1/M2 ladder -- priors, evidences, posteriors, the selected model, and which
rungs were evaluated -- travels in the structured JSON output,
:func:`gp_dla_finder.io.structured.write_structured_results`. The in-memory
:class:`~gp_dla_finder.finder.Result` keeps the same information either way.

A spectrum for which two absorbers were selected contributes **two ordinary
rows** sharing a ``TARGETID``, with ``DLAID`` values ``<targetid>000`` and
``<targetid>001``. That is the whole multi-absorber representation in FITS.

Legacy semantics
----------------
The reference implementation defines the probability columns independently
(``dlasearch.py``, in the per-absorber loop):

* ``P_DLA`` and ``P_NULL`` are **spectrum-level** scalars, repeated unchanged on
  every absorber row of that spectrum;
* ``LOGP_DLA`` is ``log_posteriors_dla[n]`` — a **per-absorber-index** quantity,
  the log posterior of the *n*-absorber model, not the log of ``P_DLA``;
* ``LOGP_NULL`` is a spectrum-level scalar;
* ``MODEL_P`` is ``model_posteriors[1 + num_subdla + n]``, the posterior of the
  specific absorber-count model.

These quantities coincide only in special cases and diverge when more than one
absorber is modeled. We therefore preserve all five independently.

``Z_DLA_ERR`` and ``NHI_ERR``
-----------------------------
Present in **both** products, because the reference produces them and downstream
readers expect the columns. The row model accepts and preserves them when a
validated estimator supplies a value. Until this package has such an estimator
they are written as documented **NaN**. The optional ``emcee`` validation samples
do not by themselves define a production uncertainty.

Other invariants
----------------
* **No sample arrays, ever.** Catalogue output is summary-only, enforced by an
  explicit field allowlist rather than by guessing from array shapes.
* A FILTER-derived value may occupy an evidence field, but **only when it is
  labelled**: every row carries ``GPDLF_EVIDENCE_MODE``, and constructing a row
  without a valid label raises. Run-level labelling alone would be insufficient,
  because one file can hold both modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

__all__ = [
    "ABSORBER_SCHEMA",
    "CATALOGUE_SCHEMA_VERSION",
    "LEGACY_ABSORBER_COLUMNS",
    "RUN_SCHEMA",
    "MODEL_SCHEMA",
    "ModelRow",
    "SPECTRUM_SCHEMA",
    "Column",
    "EVIDENCE_MODES",
    "STATUSES",
    "reference_dlaid",
    "SpectrumStatus",
    "schema_for",
]

#: Bumped when the *meaning* of a column changes or a column is removed. Adding a
#: column bumps the minor part; a reader that checks the major part keeps working.
#:
#: 1.1 added ``GPDLF_SCREENING_SCORE`` and ``GPDLF_SCREENING_N_EVALUATED`` to the
#: spectrum table, and ``GPDLF_BASE_PRESET`` and ``GPDLF_CONFIG_DIGEST`` to the
#: run record. Additions only, so a 1.0 reader that checks the major part reads a
#: 1.1 file without changes -- which is the property the rule exists to give, and
#: is asserted by the compatibility tests rather than assumed.
#:
#: 1.2 added ``GPDLF_MAX_DLAS`` to the run record and removed the ``MODELS`` HDU
#: from the FITS product: FITS is the flat DESI
#: catalogue, and the model ladder moved to the structured JSON output. No
#: column changed meaning, and every 1.1 column is still present, so a reader
#: keyed on the major version is unaffected -- but a reader that looked for
#: ``MODELS`` in FITS must now read the JSON.
CATALOGUE_SCHEMA_VERSION = "1.2"

#: Every state an attempted spectrum can end in. A catalogue that cannot
#: distinguish them cannot support population statistics: "no absorber found" and
#: "we never looked" are different facts.
SpectrumStatus = Literal[
    "completed",
    "insufficient_data",
    "quality_rejected",
    "failed",
]

STATUSES: tuple[str, ...] = (
    "completed",
    "insufficient_data",
    "quality_rejected",
    "failed",
)


def _is_nan(value: float) -> bool:
    return value != value


@dataclass(frozen=True)
class Column:
    """One catalogue column, with the intent behind it recorded."""

    name: str
    #: FITS/numpy datatype code, as astropy's ``Table`` understands it.
    dtype: str
    unit: str | None
    description: str
    #: True when the column exists to satisfy an existing downstream reader and
    #: its meaning is fixed by history rather than by this package.
    legacy: bool = False
    #: True when the column may be absent from the extended product because the
    #: quantity has no validated estimator yet.
    provisional: bool = False


# --------------------------------------------------------------------------
# 1. The absorber table -- the compatibility surface
# --------------------------------------------------------------------------

#: Transcribed from ``dlasearch.py`` in the reference implementation. The order
#: is the reference's order, because some readers index by position.
ABSORBER_SCHEMA: tuple[Column, ...] = (
    Column("TARGETID", "int64", None, "Parent spectrum identifier.", legacy=True),
    Column("RA", "float64", "deg", "Right ascension of the quasar.", legacy=True),
    Column("DEC", "float64", "deg", "Declination of the quasar.", legacy=True),
    Column(
        "Z_QSO",
        "float64",
        None,
        "Quasar emission redshift. Renamed from Z on 2024-10-25; catalogues "
        "written before that date use the old name.",
        legacy=True,
    ),
    Column(
        "SNR_FOREST", "float64", None, "Signal-to-noise in the forest.", legacy=True
    ),
    Column(
        "SNR_REDSIDE", "float64", None, "Signal-to-noise redward of Lya.", legacy=True
    ),
    Column(
        "DLAID",
        "str",
        None,
        "Absorber identifier, unique within the catalogue.",
        legacy=True,
    ),
    Column("Z_DLA", "float64", None, "Absorber redshift.", legacy=True),
    Column(
        "Z_DLA_ERR",
        "float64",
        None,
        "Uncertainty on Z_DLA. NaN unless a validated estimator supplied it; "
        "this package does not yet have one.",
        legacy=True,
        provisional=True,
    ),
    Column(
        "NHI",
        "float64",
        None,
        "Column density, log10(N_HI / cm^-2). Unit left blank because "
        "log10(cm-2) is not a FITS-standard unit string and makes every "
        "downstream reader warn.",
        legacy=True,
    ),
    Column(
        "NHI_ERR",
        "float64",
        None,
        "Uncertainty on NHI, in dex. NaN unless a validated estimator supplied it.",
        legacy=True,
        provisional=True,
    ),
    Column(
        "DLAFLAG",
        "int64",
        None,
        "Fit-quality flag. int64, not int32: the reference declares it with "
        "numpy dtype='int', which is 64-bit on the production platform, so a "
        "32-bit column would not be datatype-compatible.",
        legacy=True,
    ),
    Column(
        "P_DLA",
        "float64",
        None,
        "Spectrum-level posterior probability of the absorber model, repeated on "
        "each absorber row. Legacy meaning; NOT an evidence.",
        legacy=True,
    ),
    Column(
        "P_NULL",
        "float64",
        None,
        "Spectrum-level posterior probability of the no-absorber model. Legacy.",
        legacy=True,
    ),
    Column(
        "LOGP_DLA",
        "float64",
        None,
        "log posterior of the n-absorber model (reference log_posteriors_dla[n]). "
        "PER-ABSORBER-INDEX and NOT log(P_DLA); carried independently.",
        legacy=True,
    ),
    Column(
        "LOGP_NULL",
        "float64",
        None,
        "Spectrum-level log posterior of the no-absorber model. Legacy.",
        legacy=True,
    ),
    Column(
        "MODEL_P",
        "float64",
        None,
        "Posterior of this absorber-count model (reference "
        "model_posteriors[1 + num_subdla + n]). A MODEL-level quantity carried on "
        "the absorber row it belongs to.",
        legacy=True,
    ),
    # --- extensions, all new names -----------------------------------------
    Column(
        "GPDLF_LOG_EVIDENCE_ABSORBER",
        "float64",
        None,
        "log p(D | one absorber). A NEW field: the legacy LOGP_* columns are "
        "posterior probabilities, not evidences, and must not be reused for this.",
    ),
    Column(
        "GPDLF_LOG_EVIDENCE_NULL",
        "float64",
        None,
        "log p(D | no absorber), in nats. New. Unit left blank: nat is not a "
        "FITS-standard unit string.",
    ),
    Column(
        "GPDLF_LOG_BAYES_FACTOR",
        "float64",
        None,
        "log evidence ratio, absorber over null. Computed from the two evidence "
        "fields on this row, and carrying the same mode label as they do.",
    ),
    Column(
        "GPDLF_EVIDENCE_MODE",
        "str",
        None,
        "'exact' or 'filter', PER ROW. When 'filter' the evidence columns on this "
        "row hold APPROXIMATE SCREENING values, not exact evidences, and must not "
        "be used for model comparison. A row cannot be built without this label.",
    ),
    Column(
        "GPDLF_SCREENING_SCORE",
        "float64",
        None,
        "FILTER screening statistic: the log Bayes factor, absorber over null, "
        "from the FILTER prefix estimate. A RANKING statistic -- deliberately "
        "not named like a probability or an evidence, and not to be used as "
        "either. Populated only in filter mode; NaN means NOT SCREENED, never "
        "'screened and scored zero'.",
    ),
)

#: The strict legacy view: exactly the historical columns, in the historical
#: order, and nothing else.
LEGACY_ABSORBER_COLUMNS: tuple[str, ...] = tuple(
    column.name for column in ABSORBER_SCHEMA if column.legacy
)


# --------------------------------------------------------------------------
# 2. The per-spectrum table -- what the absorber table cannot express
# --------------------------------------------------------------------------

SPECTRUM_SCHEMA: tuple[Column, ...] = (
    Column(
        "TARGETID", "int64", None, "Spectrum identifier; joins to the absorber table."
    ),
    Column("RA", "float64", "deg", "Right ascension of the quasar."),
    Column("DEC", "float64", "deg", "Declination of the quasar."),
    Column("Z_QSO", "float64", None, "Quasar emission redshift."),
    Column(
        "GPDLF_STATUS",
        "str",
        None,
        "One of: completed, insufficient_data, quality_rejected, failed. This is "
        "the column that makes a null result distinguishable from an absent one.",
    ),
    Column(
        "GPDLF_REASON",
        "str",
        None,
        "Stable reason code when status is not 'completed'; empty otherwise. "
        "Never a free-text message, so a batch layer can aggregate causes.",
    ),
    Column(
        "GPDLF_N_ABSORBERS",
        "int32",
        None,
        "Absorber rows this spectrum contributed. Zero for a completed spectrum "
        "with no detection -- which is a result, not a missing row.",
    ),
    Column(
        "GPDLF_P_ABSORBER",
        "float64",
        None,
        "Model-posterior probability of at least one absorber. NaN when the "
        "status is not 'completed'.",
    ),
    Column(
        "GPDLF_LOG_EVIDENCE_NULL",
        "float64",
        None,
        "log p(D | no absorber), in nats.",
    ),
    Column(
        "GPDLF_LOG_EVIDENCE_ABSORBER",
        "float64",
        None,
        "log p(D | one absorber), in nats.",
    ),
    Column("GPDLF_EVIDENCE_MODE", "str", None, "'exact' or 'filter'."),
    Column(
        "GPDLF_QUALITY_FRACTION",
        "float64",
        None,
        "Measured usable fraction under the run's quality policy, whether or not "
        "it passed, so a rejection can be audited rather than merely trusted.",
    ),
    Column("GPDLF_N_USABLE_PIXELS", "int32", None, "Unmasked pixels in the window."),
    Column(
        "GPDLF_N_EVALUATED",
        "int64",
        None,
        "QMC samples actually evaluated for THIS spectrum. Per spectrum because "
        "FILTER can stop after different counts for different spectra.",
    ),
    Column(
        "GPDLF_SCREENING_SCORE",
        "float64",
        None,
        "FILTER screening statistic: the log Bayes factor, absorber over null, "
        "from the FILTER prefix. A RANKING statistic -- not a probability and "
        "not an evidence. Carried on the spectrum row as well as absorber rows, "
        "so a screened spectrum that did not pass the detection threshold still "
        "records what screening said. NaN means NOT SCREENED.",
    ),
    Column(
        "GPDLF_SCREENING_N_EVALUATED",
        "int64",
        None,
        "Samples the screening stage evaluated; 0 when no screening stage ran. "
        "Distinct from GPDLF_N_EVALUATED, which describes whichever stage "
        "produced the evidence columns.",
    ),
)


# --------------------------------------------------------------------------
# 2b. The model-ladder table -- one row per model, per spectrum
# --------------------------------------------------------------------------
#
# NOT a FITS table. The FITS catalogue is the flat DESI product and nothing
# else; the ladder travels in the structured JSON output
# (:mod:`gp_dla_finder.io.structured`), which has no schema pressure to stay
# rectangular and can grow a rung without a format change.
#
# It stays a declared schema rather than an ad-hoc dict because the field
# meanings need documenting wherever they are written, and because the JSON
# writer validates against it.


MODEL_SCHEMA: tuple[Column, ...] = (
    Column("TARGETID", "int64", None, "Parent spectrum."),
    Column(
        "GPDLF_MODEL_INDEX",
        "int32",
        None,
        "Number of absorbers this model assumes: 0 for the null model, k for "
        "the k-absorber model. Rows are contiguous from 0 for each spectrum.",
    ),
    Column("GPDLF_MODEL_LABEL", "str", None, "'M0', 'M1', ... for readability."),
    Column(
        "GPDLF_MODEL_LOG_EVIDENCE",
        "float64",
        None,
        "log p(D | M_k). NaN when the ladder stopped before evaluating this "
        "model -- which is NOT the same as an evidence of zero.",
    ),
    Column(
        "GPDLF_MODEL_LOG_PRIOR",
        "float64",
        None,
        "log P(M_k) under the absorber-existence prior. The top model absorbs "
        "the tail, so its prior is P(>= k), matching the reference.",
    ),
    Column(
        "GPDLF_MODEL_POSTERIOR",
        "float64",
        None,
        "P(M_k | D), normalised over the models that completed. Zero here can "
        "mean 'evaluated and very unlikely' OR 'never evaluated' -- read it "
        "with GPDLF_MODEL_EVALUATED.",
    ),
    Column(
        "GPDLF_MODEL_EVALUATED",
        "bool",
        None,
        "Whether this model produced a finite evidence.",
    ),
    Column(
        "GPDLF_MODEL_SELECTED",
        "bool",
        None,
        "Whether this is the model with the highest joint probability.",
    ),
)


# --------------------------------------------------------------------------
# 3. The run table -- recorded once, not per row
# --------------------------------------------------------------------------

RUN_SCHEMA: tuple[Column, ...] = (
    Column("GPDLF_SCHEMA_VERSION", "str", None, "Catalogue schema version."),
    Column("GPDLF_VERSION", "str", None, "Package version."),
    Column(
        "GPDLF_PRESET",
        "str",
        None,
        "EFFECTIVE configuration name. Ends in '+modified' when a "
        "scientifically consequential setting was overridden, so a run that is "
        "not the canonical operating point cannot claim its name.",
    ),
    Column(
        "GPDLF_BASE_PRESET",
        "str",
        None,
        "The named preset this configuration started from, kept even after "
        "overrides, so a reader can tell what it was a variant OF.",
    ),
    Column(
        "GPDLF_CONFIG_DIGEST",
        "str",
        None,
        "Stable short hash of every scientifically consequential setting. Two "
        "runs with the same digest computed the same thing whatever they are "
        "called; it is what the writer compares before combining results.",
    ),
    Column("GPDLF_EVIDENCE_MODE", "str", None, "'exact' or 'filter'."),
    Column("GPDLF_NUM_SAMPLES", "int64", None, "QMC grid size."),
    Column(
        "GPDLF_N_EVALUATED_CONFIGURED",
        "int64",
        None,
        "The CONFIGURED evaluation limit for the run. The count actually reached "
        "is per spectrum, in the SPECTRA table; this is not a claim that every "
        "spectrum evaluated this many.",
    ),
    Column("GPDLF_SAMPLE_GRID", "str", None, "Named QMC grid."),
    Column("GPDLF_MODEL", "str", None, "Trained model name."),
    Column("GPDLF_PRIOR", "str", None, "Absorber-existence prior name."),
    Column("GPDLF_COMPAT_PROFILE", "str", None, "Arithmetic compatibility profile."),
    Column("GPDLF_VOIGT_BACKEND", "str", None, "Voigt backend name."),
    Column("GPDLF_FADDEEVA_SOURCE", "str", None, "Faddeeva implementation."),
    Column("GPDLF_LSF_KERNEL", "str", None, "Named line-spread function."),
    Column("GPDLF_QUALITY_POLICY", "str", None, "Quality policy, or empty for none."),
    Column(
        "GPDLF_MAX_DLAS",
        "int64",
        None,
        "Maximum absorbers SEARCHED per spectrum: 1 for an ordinary M0/M1 run, "
        "2 for the experimental M0/M1/M2 ladder. Not the number detected in any "
        "spectrum -- that is the number of DLACAT rows sharing a TARGETID. "
        "Read P_DLA with this: P_DLA is P(at least one absorber) summed over "
        "every absorber model that was searched, so its normalisation depends "
        "on how deep the ladder went, and nothing else in the file records that.",
    ),
    Column(
        "GPDLF_EXPERIMENTAL",
        "str",
        None,
        "Comma-separated experimental features that were enabled, or empty. "
        "'multi_absorber' means the M0/M1/M2 ladder ran: reference-compatible "
        "on the tested surface, but not a validated close-pair method.",
    ),
    Column(
        "GPDLF_SEED",
        "int64",
        None,
        "RNG seed for the resampler, or -1 when the run was explicitly "
        "stochastic. Read WITH GPDLF_RNG_MODE: an unseeded run must never be "
        "recorded as the deterministic seed 0.",
    ),
    Column(
        "GPDLF_RNG_MODE",
        "str",
        None,
        "'deterministic' when a seed was supplied, 'stochastic' when the run "
        "was deliberately unseeded. Distinguishes 'seed 0' from 'no seed'.",
    ),
    Column(
        "GPDLF_RNG_ALGORITHM",
        "str",
        None,
        "The random stream used for multi-absorber resampling, e.g. "
        "'numpy.random.RandomState (MT19937)'. Recorded because a different "
        "stream gives different k>=2 evidences from the same seed.",
    ),
)


_SCHEMAS: Mapping[str, tuple[Column, ...]] = MappingProxyType(
    {
        "absorbers": ABSORBER_SCHEMA,
        "models": MODEL_SCHEMA,
        "spectra": SPECTRUM_SCHEMA,
        "run": RUN_SCHEMA,
    }
)


def schema_for(table: str) -> tuple[Column, ...]:
    """Return a named schema.

    Raises
    ------
    KeyError
        If ``table`` is unknown.
    """
    try:
        return _SCHEMAS[table]
    except KeyError:
        known = ", ".join(sorted(_SCHEMAS))
        raise KeyError(f"unknown catalogue table {table!r}; known: {known}") from None


#: Evidence modes a row may declare. A row must declare one: an unlabelled
#: evidence value would make the file scientifically ambiguous.
EVIDENCE_MODES: tuple[str, ...] = ("exact", "filter")


def reference_dlaid(targetid: int, index: int) -> str:
    """The public reference's absorber identifier: ``str(tid) + "00" + str(n)``.

    Transcribed from ``dlasearch.py``. Not a hyphenated form of our own
    invention: downstream code parses these, and "compatible" has to mean the
    same bytes, not a similar idea.
    """
    if index < 0:
        raise ValueError(f"absorber index must be non-negative, got {index}")
    return f"{targetid}00{index}"


@dataclass(frozen=True)
class AbsorberRow:
    """One absorber. Summary quantities only, by construction.

    The five legacy probability fields are carried **independently**. None is
    derived from another, because in the reference they are different quantities:
    ``p_dla``/``p_null`` are spectrum-level, ``logp_dla`` is per-absorber-index,
    and ``model_p`` is the posterior of that absorber-count model.
    """

    targetid: int
    dlaid: str
    z_qso: float
    z_dla: float
    nhi: float
    #: Spectrum-level posterior of the absorber model, repeated on each row.
    p_dla: float
    #: Spectrum-level posterior of the null model.
    p_null: float
    #: log posterior of the n-absorber model. NOT log(p_dla).
    logp_dla: float
    #: Spectrum-level log posterior of the null model.
    logp_null: float
    #: Posterior of this absorber-count model.
    model_p: float
    log_evidence_absorber: float
    log_evidence_null: float
    #: "exact" or "filter". Required: see EVIDENCE_MODES.
    evidence_mode: str
    #: Uncertainties, when a validated estimator supplies them. NaN otherwise.
    z_dla_err: float = float("nan")
    nhi_err: float = float("nan")
    ra: float = float("nan")
    dec: float = float("nan")
    snr_forest: float = float("nan")
    snr_redside: float = float("nan")
    dlaflag: int = 0
    screening_score: float = float("nan")
    #: Which absorber-count model this candidate belongs to: 1 for the
    #: one-absorber model, 2 for a member of the best two-absorber pair.
    #:
    #: **Not a FITS column, deliberately.** The DESI product is flat and stays
    #: flat; membership is implicit there in how many rows share a ``TARGETID``.
    #: This field exists so the structured JSON can state it, and so it does not
    #: have to be re-derived from the row index -- which gives the wrong answer,
    #: because both members of a selected M2 pair belong to M2 while their row
    #: indices are 0 and 1.
    model_index: int = 1

    def __post_init__(self) -> None:
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"evidence_mode must be one of {EVIDENCE_MODES}, got "
                f"{self.evidence_mode!r}. An evidence value without a mode label "
                "cannot be written: a reader could not tell an exact evidence "
                "from a FILTER screening value."
            )
        if self.model_index < 1:
            raise ValueError(
                f"model_index must be at least 1 -- an absorber row belongs to "
                f"an absorber model, not to the null model. Got "
                f"{self.model_index}"
            )
        for name in ("p_dla", "p_null", "model_p"):
            value = getattr(self, name)
            if not _is_nan(value) and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {value}")
        # The identifier must be derivable from its parent by the reference's own
        # rule. Checked here rather than only on written output, so a row built by
        # hand cannot reach a writer with an identifier a downstream join would
        # not resolve.
        prefix = f"{self.targetid}00"
        if not self.dlaid.startswith(prefix) or not self.dlaid[len(prefix) :].isdigit():
            raise ValueError(
                f"dlaid {self.dlaid!r} does not follow the reference convention "
                f"str(TARGETID) + '00' + str(index) for TARGETID {self.targetid}; "
                "use reference_dlaid()"
            )

    @property
    def log_bayes_factor(self) -> float:
        """log evidence ratio, absorber over null, in this row's mode."""
        return self.log_evidence_absorber - self.log_evidence_null


@dataclass(frozen=True)
class SpectrumRow:
    """One attempted spectrum, whatever happened to it."""

    targetid: int
    z_qso: float
    status: str
    reason: str = ""
    n_absorbers: int = 0
    p_absorber: float = float("nan")
    log_evidence_null: float = float("nan")
    log_evidence_absorber: float = float("nan")
    evidence_mode: str = ""
    quality_fraction: float = float("nan")
    n_usable_pixels: int = 0
    ra: float = float("nan")
    dec: float = float("nan")
    #: Samples actually evaluated for THIS spectrum. Recorded per spectrum because
    #: FILTER can stop after different counts for different spectra, and a
    #: run-level number would falsely claim one count for all of them.
    n_evaluated: int = 0
    #: FILTER-prefix log Bayes factor. Carried on the SPECTRUM row, not only on
    #: absorber rows, so a screened spectrum that did not pass the detection
    #: threshold still records what screening said about it -- it has no absorber
    #: row to carry it. NaN means no screening stage ran.
    screening_score: float = float("nan")
    #: Samples the screening stage evaluated; 0 when it did not run.
    screening_n_evaluated: int = 0

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"unknown status {self.status!r}; expected one of {STATUSES}"
            )
        if self.n_absorbers < 0:
            raise ValueError(f"n_absorbers cannot be negative, got {self.n_absorbers}")
        if self.status != "completed" and self.n_absorbers:
            raise ValueError(
                f"status {self.status!r} cannot have {self.n_absorbers} absorber "
                "rows: a spectrum that was not searched cannot have found anything"
            )
        if self.status != "completed" and not self.reason:
            raise ValueError(f"status {self.status!r} requires a reason code")
        if self.status == "completed" and self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(
                f"a completed spectrum must declare an evidence mode from "
                f"{EVIDENCE_MODES}, got {self.evidence_mode!r}"
            )
        if not _is_nan(self.p_absorber) and not 0.0 <= self.p_absorber <= 1.0:
            raise ValueError(
                f"p_absorber must be a probability in [0, 1], got {self.p_absorber}"
            )
        if not _is_nan(self.quality_fraction) and not (
            0.0 <= self.quality_fraction <= 1.0
        ):
            raise ValueError(
                f"quality_fraction must be in [0, 1], got {self.quality_fraction}"
            )
        if self.n_usable_pixels < 0:
            raise ValueError(
                f"n_usable_pixels cannot be negative, got {self.n_usable_pixels}"
            )
        if self.n_evaluated < 0:
            raise ValueError(f"n_evaluated cannot be negative, got {self.n_evaluated}")
        if self.status == "completed":
            if self.reason:
                raise ValueError(
                    f"a completed spectrum must not carry a failure reason, got "
                    f"{self.reason!r}"
                )
        else:
            # A spectrum that was never searched cannot have inference results.
            # Left unchecked, a rejected row carrying an evidence would be read
            # as a real measurement.
            carried = {
                "p_absorber": self.p_absorber,
                "log_evidence_null": self.log_evidence_null,
                "log_evidence_absorber": self.log_evidence_absorber,
            }
            present = [name for name, v in carried.items() if not _is_nan(v)]
            if present:
                raise ValueError(
                    f"status {self.status!r} cannot carry inference results, but "
                    f"{present} are set"
                )
            if self.evidence_mode:
                raise ValueError(
                    f"status {self.status!r} cannot declare an evidence mode, got "
                    f"{self.evidence_mode!r}"
                )
            if self.n_evaluated:
                raise ValueError(
                    f"status {self.status!r} cannot have evaluated "
                    f"{self.n_evaluated} samples"
                )


@dataclass(frozen=True)
class ModelRow:
    """One model of one spectrum's ladder."""

    targetid: int
    model_index: int
    log_evidence: float
    log_prior: float
    posterior: float
    evaluated: bool
    selected: bool

    @property
    def model_label(self) -> str:
        return f"M{self.model_index}"

    def __post_init__(self) -> None:
        if self.model_index < 0:
            raise ValueError(f"model_index cannot be negative: {self.model_index}")
        if not _is_nan(self.posterior) and not 0.0 <= self.posterior <= 1.0:
            raise ValueError(f"model posterior must be in [0, 1], got {self.posterior}")
        # An unevaluated model cannot be the selected one, and cannot carry a
        # finite evidence -- both would misrepresent what was computed.
        if not self.evaluated:
            if self.selected:
                raise ValueError(
                    f"{self.model_label} was not evaluated and cannot be selected"
                )
            if not _is_nan(self.log_evidence):
                raise ValueError(
                    f"{self.model_label} was not evaluated but carries a finite "
                    f"log evidence {self.log_evidence}"
                )


@dataclass(frozen=True)
class Catalogue:
    """Absorber rows, spectrum rows, the model ladder, and the run record."""

    absorbers: Sequence[AbsorberRow] = field(default_factory=tuple)
    spectra: Sequence[SpectrumRow] = field(default_factory=tuple)
    #: One row per model per spectrum, when a ladder was evaluated.
    models: Sequence[ModelRow] = field(default_factory=tuple)
    run: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A frozen dataclass holding a caller's list is not frozen: the caller can
        # append to it afterwards and invalidate every check below. Copy into
        # immutable containers before validating anything.
        object.__setattr__(self, "absorbers", tuple(self.absorbers))
        object.__setattr__(self, "spectra", tuple(self.spectra))
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "run", MappingProxyType(dict(self.run)))

        targetids = [row.targetid for row in self.spectra]
        if len(set(targetids)) != len(targetids):
            duplicates = sorted({t for t in targetids if targetids.count(t) > 1})
            raise ValueError(
                f"duplicate spectrum TARGETID(s): {duplicates[:5]}. One row per "
                "attempted spectrum, or a count of spectra is meaningless."
            )
        dlaids = [row.dlaid for row in self.absorbers]
        if len(set(dlaids)) != len(dlaids):
            duplicates = sorted({d for d in dlaids if dlaids.count(d) > 1})
            raise ValueError(f"duplicate DLAID(s): {duplicates[:5]}")

        known = {row.targetid for row in self.spectra}
        orphans = {row.targetid for row in self.absorbers} - known
        if orphans:
            raise ValueError(
                f"absorber rows reference {len(orphans)} TARGETID(s) with no "
                f"spectrum row: {sorted(orphans)[:5]}. Every absorber must have a "
                "parent, or a reader cannot tell a detection from a stray row."
            )

        counted: dict[int, int] = {}
        by_target: dict[int, list[AbsorberRow]] = {}
        for row in self.absorbers:
            counted[row.targetid] = counted.get(row.targetid, 0) + 1
            by_target.setdefault(row.targetid, []).append(row)
        spectra_by_target = {row.targetid: row for row in self.spectra}

        for row in self.spectra:
            actual = counted.get(row.targetid, 0)
            if actual != row.n_absorbers:
                raise ValueError(
                    f"TARGETID {row.targetid} declares {row.n_absorbers} absorbers "
                    f"but {actual} rows are present"
                )

        for targetid, rows in by_target.items():
            # Spectrum-level quantities are repeated on every absorber row of a
            # spectrum. If they disagree, at least one row is wrong and a reader
            # taking "the" value for that spectrum would get an arbitrary one.
            for name in ("p_dla", "p_null", "logp_null", "z_qso"):
                values = {getattr(r, name) for r in rows}
                if len(values) > 1:
                    raise ValueError(
                        f"TARGETID {targetid}: {name} is a spectrum-level quantity "
                        f"but differs across its absorber rows: {sorted(values)}"
                    )
            modes = {r.evidence_mode for r in rows}
            if len(modes) > 1:
                raise ValueError(
                    f"TARGETID {targetid}: absorber rows mix evidence modes "
                    f"{sorted(modes)}; a spectrum is inferred in one mode"
                )
            parent = spectra_by_target.get(targetid)
            if parent is not None and parent.evidence_mode:
                if modes and parent.evidence_mode not in modes:
                    raise ValueError(
                        f"TARGETID {targetid}: absorber rows declare "
                        f"{sorted(modes)} but the spectrum row declares "
                        f"{parent.evidence_mode!r}"
                    )
                # The indices must be the reference's 0..n-1, with no gaps: a
                # downstream join on DLAID depends on it.
                expected = {reference_dlaid(targetid, i) for i in range(len(rows))}
                if {r.dlaid for r in rows} != expected:
                    raise ValueError(
                        f"TARGETID {targetid}: absorber indices are not "
                        f"0..{len(rows) - 1}; "
                        f"got {sorted(r.dlaid for r in rows)}"
                    )

        # Caller metadata must not contradict the writer-controlled schema.
        declared = self.run.get("GPDLF_SCHEMA_VERSION")
        if declared is not None and str(declared) != CATALOGUE_SCHEMA_VERSION:
            raise ValueError(
                f"run metadata declares schema version {declared!r}, but this "
                f"package writes {CATALOGUE_SCHEMA_VERSION!r}. The version is "
                "writer-controlled and must not be overridden."
            )
