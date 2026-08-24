# Catalog output

There are two FITS outputs and one JSON output. They serve different purposes,
and for an M2 run you will often want both FITS and JSON.

**FITS is the flat DESI catalog.** One row per absorber, fixed columns, no
nesting and no variable-length arrays. A spectrum with two selected absorbers
contributes two ordinary rows sharing a `TARGETID`. That is the whole
multi-absorber representation in FITS, and it is deliberate — downstream
analyses already read this schema.

**JSON carries the full inference record.** This includes the M0/M1/M2 ladder,
its priors, evidences and posteriors, the selected model, the evaluated rungs,
and the membership of each absorber. Those relationships do not fit cleanly in
a flat table, so they stay in the structured output.

## Strict legacy

`gp_dla_finder.io.fits.write_legacy_catalogue` writes the historical DESI DLA
catalog columns in their original order. The `DLACAT` binary table contains one
row per absorber:

```
TARGETID  RA  DEC  Z_QSO  SNR_FOREST  SNR_REDSIDE
DLAID  Z_DLA  Z_DLA_ERR  NHI  NHI_ERR  DLAFLAG
P_DLA  P_NULL  LOGP_DLA  LOGP_NULL  MODEL_P
```

`TARGETID` identifies the parent spectrum and `DLAID` identifies the absorber,
as `f"{targetid}00{n}"` with `n` counting from 0 within the spectrum. The
meanings of `P_DLA`, `P_NULL`, `LOGP_DLA`, `LOGP_NULL` and `MODEL_P` are
unchanged from the deployed catalog, and they do not all have the same scope:

| column | scope | meaning |
|---|---|---|
| `P_DLA` | spectrum | P(at least one absorber), summed over **every absorber model searched** |
| `P_NULL` | spectrum | `1 - P_DLA` |
| `LOGP_NULL` | spectrum | unnormalized joint of the null model, `log p(D\|M) + log P(M)` |
| `LOGP_DLA` | **per row** | the same joint, for *this row's* absorber-count model |
| `MODEL_P` | **per row** | posterior of *this row's* absorber-count model |

The distinction matters as soon as more than one absorber is modeled. On a
two-absorber spectrum, row 0 carries `MODEL_P = P(M1)` and row 1 carries
`MODEL_P = P(M2)`, while both carry the same `P_DLA = P(M1) + P(M2)`.

Because `P_DLA` is summed over the models that were searched, its normalization
depends on how deep the search went — and the file records that in the
`GPDLF_MAX_DLAS` header keyword, on the `DLACAT` HDU of both products:

```
GPDLF_MAX_DLAS = 1    # an ordinary null-versus-one run
GPDLF_MAX_DLAS = 2    # the experimental M0/M1/M2 ladder
```

This is the **search limit**, not a detection count. The number of absorbers
found in a spectrum is the number of rows carrying its `TARGETID`.

:::{warning}
`Z_DLA_ERR` and `NHI_ERR` are written as **NaN** and identified as such in the
header. In the current `Finder` path they are not computed: `NHI_ERR` is not the
QMC grid spacing, the scatter of nearby grid points, or the numerical integration
error. It is reserved for a validated uncertainty on
$\log_{10}(N_{\rm HI}/{\rm cm}^{-2})$, in dex. `Z_DLA_ERR` is similarly reserved
for a validated absorber-redshift uncertainty. A future fixed-model HMC
calculation could supply conditional uncertainties, while RJMCMC or a comparable
model-selection sampler would be needed when absorber multiplicity is also
uncertain. Until then, the best grid locations are useful for inspection but the
error fields remain explicitly unavailable.
:::

:::{danger}
**This product cannot represent a spectrum with no absorber.** A null result, a
quality rejection, an inference failure, and an unprocessed spectrum are all
absent from the table. The strict product lists absorbers but cannot define a
processed parent sample for population analysis. If you need that, use the
extended product.
:::

## Extended

`write_catalogue` writes three HDUs:

| HDU | contents |
|---|---|
| `DLACAT` | the legacy columns, in order, **plus** new namespaced fields |
| `SPECTRA` | one row per *attempted* spectrum, with status and reason code |
| `RUNINFO` | run-level configuration and provenance, recorded once |

Old readers can keep using the historical columns, while newer code can use
namespaced fields such as `GPDLF_LOG_EVIDENCE_ABSORBER`,
`GPDLF_LOG_BAYES_FACTOR`, `GPDLF_EVIDENCE_MODE`, and
`GPDLF_SCREENING_SCORE`.

### Status

`SPECTRA.GPDLF_STATUS` is one of:

`completed`
: inference ran. `GPDLF_N_ABSORBERS` may be zero.

`insufficient_data`
: the spectrum is valid but cannot support inference because it is fully masked,
  lacks normalization coverage, or contains too few usable pixels.

`quality_rejected`
: a named quality policy rejected the spectrum. The measured usable fraction is
  recorded with the decision.

`failed`
: the numerics failed on otherwise valid input.

### FILTER mode

When `GPDLF_EVIDENCE_MODE` is `filter`, the evidence columns contain FILTER
estimates, not full-grid evidences. FILTER is opt-in, so you only get these
values when you ask for the screening path.

`GPDLF_SCREENING_SCORE` is the FILTER-prefix log Bayes factor and is used as a
ranking statistic. In full-grid mode it is `NaN`, which means that no screening
stage was run. See {doc}`filter`.

### The detection threshold

`GPDLF_DETECTION_THRESHOLD` records the posterior threshold used to select
absorber rows. `results_to_catalogue()` requires this value explicitly.

### One run, one provenance

One catalog has one run record, so all Results in it need to agree on the
run-defining configuration. If they do not, the writer raises instead of guessing
which metadata to keep.

Evidence mode is the only current exception. A mixed file is labeled `mixed` at
run level, while the per-row `GPDLF_EVIDENCE_MODE` value identifies the mode used
for each spectrum.

## The structured JSON result

```python
from gp_dla_finder.io.structured import (
    read_structured_results,
    write_structured_results,
)

write_structured_results("run.json", catalogue)
payload = read_structured_results("run.json")

for spectrum in payload["spectra"]:
    for model in spectrum["models"]:
        print(model["model_label"], model["posterior"], model["evaluated"])
```

One object per spectrum, carrying its models and its absorbers. It needs no
optional dependency — standard library only — so a user without astropy can
still keep the complete result.

Read `posterior` together with `evaluated`. A posterior of 0 means "evaluated
and very unlikely"; a rung that was never reached carries `NaN` and
`evaluated: false`. Those are different statements, and collapsing them to one
number is exactly what the flat table cannot avoid doing.

Write both files when the model ladder matters. If your downstream analysis only
needs a flat DLA list, use the FITS file.

## Compact by design

Catalog output contains no QMC or posterior sample arrays. It is the compact
summary used by the DESI workflow, and an explicit schema allowlist keeps large
arrays from entering it accidentally. If you need retained samples or the full
inference state, use a separate HDF5 product.

## Reading one back

```python
from gp_dla_finder.io.fits import read_catalogue_metadata

run = read_catalogue_metadata("catalogue.fits")
run["GPDLF_SCHEMA_VERSION"]
```

The reader rejects an unsupported schema **major** version.
