# Changelog

## 0.1.0rc2 - 2026-08-24

This release candidate packages the inference core of the GP-based DLA finder
so it can be used without the full DESI production pipeline: give it `wave`,
`flux`, `ivar`, `mask` and `z_qso`, and get absorber posteriors back.

### What it does

**The supported workflow** is a null-versus-one-absorber comparison on one
spectrum at a time, returning a typed `Result`. It includes the Voigt forward
model, named DESI and BOSS line-spread functions, the quasi-Monte-Carlo evidence
integral, an absorber-existence prior, per-spectrum empirical-Bayes mean-flux
fitting, and a full provenance record on every result.

**An experimental two-absorber calculation** computes an M0/M1/M2 model ladder.
For v0.1, it requires both `max_absorbers=2` and
`experimental_multi_absorber=True`; the validation limits are summarized
below.

**FILTER screening** evaluates a truncated prefix of the sample grid instead of
the whole thing. It is much faster and produces a screening score rather than a
full-grid evidence; it is off unless you ask for it.

### Output formats

* a **flat DESI-compatible FITS catalogue**, one row per absorber, with the
  historical column names, order and dtypes. A spectrum with two selected
  absorbers contributes two ordinary rows sharing a `TARGETID`;
* an **extended FITS product** adding a per-attempted-spectrum table and a
  run-level provenance record, so a null result and a rejected spectrum are
  distinguishable from one that was never processed;
* a **structured JSON result** carrying the full model ladder — evidences,
  priors, posteriors, which rungs were evaluated, which model was selected, and
  which absorber belongs to which model. Standard library only, so it works
  without astropy.

### Bundled assets

Two trained GP models, an absorber-existence prior, and three QMC sample grids.
Provenance, licences and redistribution terms for each are in
[`NOTICE.md`](NOTICE.md), which ships inside the distribution.

### Installation

`pip install gp_dla_finder` gets a pure-Python wheel and needs no compiler. The
Voigt backend is NumPy, which is the official one.

An optional compiled backend using [libcerf](https://jugit.fz-juelich.de/mlz/libcerf)
reproduces the Faddeeva implementation behind the deployed catalogues. It is not
in the wheel; request a source build with
`pip install --no-binary gp_dla_finder gp_dla_finder`. If libcerf is missing or
the numerical agreement check rejects the extension, installation still succeeds
on the NumPy backend.

### Limitations

These are the things to read before using a result for science.

* **Parameter estimates are preliminary.** Reported redshift and column density
  are the best evaluated grid points, useful for locating and inspecting a
  candidate but not precision measurements. Uncertainty fields are `NaN`,
  because no estimator for them has been validated here.
* **The two-absorber calculation is experimental.** With controlled seeds, its
  evidences reproduce the reference implementation bitwise, which makes it
  useful for fidelity work. In a small 60-spectrum benchmark it recovered the
  right multiplicity about 80% of the time, with clear weaknesses for close
  pairs (separation ~0.02 in redshift) and low signal-to-noise. This is a
  bounded test rather than a survey calibration, so it should not be treated
  as a validated close-pair method.
* **More than two absorbers is refused**, rather than silently truncated.
* **Production fidelity is not the same as scientific validation.** Under
  controlled seeds the null, one-absorber and two-absorber evidences, the FILTER
  path and the mean-flux fit all match the reference implementation on generated
  spectra. None of that reproduces a published catalogue or establishes
  population performance on survey data.
* **The reasonably trusted column-density regime is**
  $\log_{10} N_{\rm HI} > 20$. The prior extends lower because real sightlines
  contain sub-DLAs and LLSs, and a model that cannot represent them will try to
  explain them with something else. We have not independently validated that
  lower range as a science-catalogue target.
* **The packaged sample grids are regenerated**, and their byte identity with
  the deployed production arrays has not been confirmed.
* **The spectrum generator in `tools/` is a demonstration and injection tool**,
  not a validated physical mock generator. It is not a substitute for
  `quickquasars`, `fake_spectra`, or a survey mock, and should not be used as
  validation evidence for catalogue performance.
* **No survey I/O adapters or parallel execution.** One spectrum at a time; the
  survey-scale workflow is yours to provide.

Full detail is in [`docs/caveats.md`](docs/caveats.md).
