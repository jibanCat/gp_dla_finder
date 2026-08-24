# Scientific caveats

The `0.1.0rc3` release candidate is already useful for candidate finding and
method development, but several parts of the scientific validation are still
open. Before treating a Result as a science measurement, keep the following
limits in mind.

## Column-density range

The prior spans $\log_{10} N_{\rm HI} \in [17.2, 22.5]$. Values below the DLA
regime help regularize the DLA inference. Real sightlines contain LLSs and
sub-DLAs, so ignoring them would itself introduce model bias. We have **not
independently validated** these systems as science-catalog targets. For now, we
treat $\log_{10} N_{\rm HI} > 20$ as the reasonably trusted regime and describe
this package as a DLA finder.

## In-sample training

The default model was trained on the mock used for calibration. Performance on
that mock is therefore in-sample.

## No implicit operating point

You need to choose a named preset; a bare `Config()` raises. This way, the
numerical operating point is always visible.

```python
Config.desi_y3()  # the deployed operating point
Config.desi_y3_fast()  # 10,000 samples, for exploration
Config(preset="custom")  # explicitly non-standard
```

## Parameter estimates are still preliminary

The current Result coordinates are the best evaluated grid points. They are
usable for finding and inspecting a candidate, but probably not reliable enough
for science production yet. The package does not currently provide a validated
MAP estimate, posterior mean, or credible interval, so the uncertainty fields
remain `NaN`.

For a fixed absorber count, HMC or another continuous posterior sampler could
provide better estimates and credible intervals once it has been validated.
For a joint question in which the number of absorbers is also unknown, the
longer-term direction is RJMCMC or a model-selection sampler such as nested
sampling. HMC alone cannot move between $\mathcal M_1$ and $\mathcal M_2$.

## Two absorbers: operational model, experimental estimator

The two-absorber model can run in a production workflow and can write the flat
DESI-compatible FITS catalogue. The present M2 evidence, however, uses the
reference pipeline's sequential resampling heuristic. It has reference-fidelity
tests and a bounded injection/recovery study, but it is not yet a generally
validated posterior method. Close pairs and low-signal pairs are known weak
cases. This is why the feature is public and usable, but still requires an
explicit experimental flag.

## Line-spread functions differ

The reference implementation uses a BOSS $R=2000$ kernel in its Python module and
a DESI $R\simeq3000$ kernel in its compiled extension. These are different
forward models. The profiles differ by up to approximately $4\times10^{-2}$ at
$\log_{10} N_{\rm HI}=19$, with the largest difference below the validated DLA
regime. This package provides both kernels by name and requires an explicit
choice.

## What we have checked so far

For generated spectra, the null and one-absorber log evidences reproduce the
pinned reference implementation bitwise under both named kernels. The
comparison includes every per-sample likelihood, and FILTER reproduces the
reference `FILTER=1` one-absorber result. The per-spectrum mean-flux scan also
agrees bitwise with a live reference run on the three generated spectra we
tested.

These are fidelity checks, not independent science validation. They do not
reproduce a published catalogue, and we have not confirmed that the packaged
grids are identical to the deployed production arrays. Multiple-absorber
inference, population statistics, and performance on a representative set of
real survey spectra still need broader validation.
