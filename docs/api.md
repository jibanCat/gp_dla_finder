# API reference

:::{warning}
The public API is **not frozen** and may change before v0.1. The supported
Finder/Result path is M0 versus M1. An opt-in M0/M1/M2 ladder is workable in a
production workflow, but its statistical estimator remains experimental;
searches above two absorbers are not implemented.
:::

## Finder and results

```{eval-rst}
.. automodule:: gp_dla_finder.finder
   :members: Finder, Result, AbsorberCandidate, screening_score,
             results_to_catalogue
```

## Configuration

```{eval-rst}
.. automodule:: gp_dla_finder.config
   :members: Config
```

## Forward model

```{eval-rst}
.. automodule:: gp_dla_finder.voigt
   :members: voigt_absorption, lsf_kernel, gaussian_lsf_kernel,
             available_backends, backend_provenance, backend_rejections
```

## Spectra and evidence

```{eval-rst}
.. automodule:: gp_dla_finder.gp.spectrum
   :members: Spectrum, PreparedSpectrum, prepare_spectrum, InsufficientData

.. automodule:: gp_dla_finder.gp.evidence
   :members: assemble_model, null_log_evidence, one_absorber_log_evidence,
             coarse_scan_size, absorber_search_window

.. automodule:: gp_dla_finder.gp.likelihood
   :members: log_mvnpdf_low_rank, effective_optical_depth
```

## Priors, grids and models

```{eval-rst}
.. automodule:: gp_dla_finder.prior
   :members: AbsorberPrior, load_prior, available_priors

.. automodule:: gp_dla_finder.samples
   :members: AbsorberSampleGrid, load_sample_grid

.. automodule:: gp_dla_finder.model
   :members: GPModel, load_model, model_provenance
```

## Catalogue output

```{eval-rst}
.. automodule:: gp_dla_finder.catalogue
   :members: Column, AbsorberRow, SpectrumRow, ModelRow, Catalogue, schema_for

.. automodule:: gp_dla_finder.io.fits
   :members: write_catalogue, write_legacy_catalogue, read_catalogue_metadata

.. automodule:: gp_dla_finder.io.structured
   :members: write_structured_results, read_structured_results,
             structured_payload, selected_models
```

## Policies and diagnostics

```{eval-rst}
.. automodule:: gp_dla_finder.quality
   :members: QualityPolicy, QualityAssessment, quality_policy

.. automodule:: gp_dla_finder.compat
   :members: CompatibilityProfile, compatibility_profile

.. automodule:: gp_dla_finder.performance
   :members: BLASPerformanceWarning, blas_thread_report, warn_once_about_blas_threads

.. automodule:: gp_dla_finder.errors
   :members:
```
