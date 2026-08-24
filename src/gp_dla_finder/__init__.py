"""Gaussian-process Bayesian detection of damped Lyman-alpha absorbers.

Give the finder one quasar spectrum -- observed-frame wavelengths, flux, inverse
variance, an optional bad-pixel mask, and the quasar redshift -- and it returns
posterior probabilities for the evaluated absorber models. The supported path
compares no absorber with one absorber; an opt-in two-absorber ladder is
experimental. Reported ``(z_abs, log10 N_HI)`` values are the best evaluated
grid points, not validated MAP estimates.

The package is survey-independent: reading DESI or SDSS files is the caller's job
(or an optional adapter's), and the numerical core needs only NumPy and SciPy.

Scientific caveats you should read before quoting a number
----------------------------------------------------------
* The column-density prior spans ``log10 N_HI`` in [17.2, 22.5]. The
  low-column-density end is there to regularise the inference; results below
  ``log10 N_HI ~ 20`` are **not independently validated**. This is a DLA finder.
* The default trained model was trained on a mock that is also the calibration
  mock, so performance figures quoted on that mock are in-sample.

References
----------
Garnett et al. (2017), arXiv:1605.04460;
Ho, Bird & Garnett (2020), arXiv:2003.11036;
Ho, Bird & Garnett (2021), arXiv:2103.10964.
"""

from __future__ import annotations

from importlib import metadata

from .config import Config
from .model import GPModel, available_models, load_model, model_provenance
from .prior import AbsorberPrior, available_priors, load_prior
from .samples import AbsorberSampleGrid, available_sample_grids, load_sample_grid
from .voigt import voigt_absorption

try:
    __version__ = metadata.version("gp_dla_finder")
except metadata.PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "AbsorberPrior",
    "AbsorberSampleGrid",
    "Config",
    "GPModel",
    "__version__",
    "available_models",
    "available_priors",
    "available_sample_grids",
    "load_model",
    "load_prior",
    "load_sample_grid",
    "model_provenance",
    "voigt_absorption",
]
