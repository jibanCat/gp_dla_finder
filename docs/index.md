# gp_dla_finder

`gp_dla_finder` uses Gaussian processes and Bayesian model comparison to find
damped Lyman-α absorbers (DLAs) in quasar spectra. It is the numerical core of
the DESI GP-DLA finder, packaged so you can use it without the full production
pipeline.

:::{warning}
**Release candidate `0.1.0rc2`.** At the moment, the supported workflow compares
the null and one-absorber models. The two-absorber model is also workable in a
production workflow when you opt in. Its current statistical estimator remains
experimental: it follows the reference implementation on the cases we have
checked, but robust posterior inference, especially for close pairs, will need
a more advanced sampler.

Reported redshifts and column densities are the best evaluated grid points.
They are useful for locating and inspecting a candidate, but probably not
reliable enough to quote as precision measurements yet. The package is already
useful for candidate finding and method development; broader validation is
still needed before unrestricted science production. See {doc}`caveats`.
:::

For one spectrum, the input is
$\mathcal{D}=\{(\lambda_i, f_i, {\rm ivar}_i, m_i)\}_{i=1}^{n}$ together with
the quasar redshift $z_{\rm QSO}$. Here $\lambda_i$ is the observed-frame
wavelength, $f_i$ the flux, ${\rm ivar}_i$ the inverse variance, and $m_i$ the
bad-pixel mask. The result contains the model evidences
$\log p(\mathcal{D}\mid\mathcal{M}_0)$ and
$\log p(\mathcal{D}\mid\mathcal{M}_1)$, the model probabilities
$p(\mathcal{M}_k\mid\mathcal{D})$, and the best evaluated absorber grid point
$\hat{\boldsymbol\theta}_{\rm grid}=(\hat z_{\rm abs},
\widehat{\log_{10}N_{\rm HI}})$.

## Reading the outputs

The output contains four related quantities. They are easy to mix up, but they
answer different questions:

**full-grid model evidence**
: $\log p(D \mid \mathcal{M}_k)$, the marginal likelihood of a model with $k$
  absorbers, computed by quasi-Monte-Carlo integration over the **whole
  configured** absorber parameter grid. This is the v0.1 default. The API calls
  this mode `exact`, although it remains a finite numerical integral rather
  than an analytic result with zero error.

**model-posterior probability**
: $p(\mathcal{M}_k \mid D)$, obtained by combining evidences with the
  absorber-existence prior. This is the quantity used for a detection threshold.

**conditional parameter posterior**
: $p(z_{\rm abs}, N_{\rm HI} \mid D, \mathcal{M}_1)$ — where the absorber is,
  *given* that there is one. The current Result reports the best evaluated grid
  point. It is a usable preliminary location, but not a validated science
  measurement.

**FILTER screening score**
: the FILTER-prefix log Bayes factor. FILTER is fast, approximate, and
  **opt-in**. In a small set of 15 constructed examples, it changed the
  classification in three cases relative to the adopted 100,000-sample
  full-grid reference. This is a useful warning near the detection threshold,
  not a survey error rate. See {doc}`filter`.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
preview
preview_local
performance
```

```{toctree}
:maxdepth: 2
:caption: Science

tutorial
customisation
filter
caveats
```

```{toctree}
:maxdepth: 2
:caption: Reference

catalogue
api
provenance
```
