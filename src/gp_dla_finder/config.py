"""Inference configuration and the named operating points we support.

A :class:`Config` contains the knobs that actually affect inference. It is smaller
than the reference pipeline's ``Parameters``, which also mixes in training,
file-loading, and catalog-filtering settings. Keeping those out makes it easier
to see what changed the numerical result.

Model-coupled quantities are not part of the configuration. The GP rank,
rest-frame grid, and flux normalization band belong to the trained model.
:meth:`Config.validate_against` checks that the model and configuration agree.

Presets
-------
In practice, start from a preset instead of assembling a config by hand. The
preset records the operating point you started from.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, ClassVar, Literal

from .compat import (
    DEFAULT_COMPATIBILITY,
    CompatibilityProfile,
    compatibility_profile,
)
from .quality import QualityPolicy, quality_policy
from .voigt import (
    PRODUCTION_KERNEL,
    backend_provenance,
    get_backend,
    kernel_half_width,
    lsf_kernel,
)

if TYPE_CHECKING:  # pragma: no cover
    from .model import GPModel

__all__ = [
    "Config",
    "LYA_WAVELENGTH",
    "LYB_WAVELENGTH",
    "LYMAN_LIMIT",
    "SPEED_OF_LIGHT",
]

#: Lyman-alpha transition wavelength, angstroms.
LYA_WAVELENGTH: float = 1215.6701
#: Lyman-beta transition wavelength, angstroms.
LYB_WAVELENGTH: float = 1025.7223
#: Lyman limit, angstroms.
LYMAN_LIMIT: float = 911.7633
#: Speed of light, m/s.
SPEED_OF_LIGHT: float = 299792458.0


def _normalize(value):
    """A JSON-stable form of a config value, so digests are comparable."""
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, float) and value.is_integer():
        # 10.0 and 10 must not produce different digests.
        return int(value)
    return value


#: Sentinel for "no preset named". Not ``None``, so ``preset=None`` is still an
#: error rather than a way to opt out of naming the configuration.
_UNSET: str = "\x00unset"

EarlyStopMode = Literal["baseline", "no_null_stop", "pre_occam"]

#: Maps this package's early-stop names onto the reference implementation's.
#:
#: The reference names are ``"baseline"``, ``"A"`` and ``"D"``, which say nothing
#: about what they do. ``baseline`` is the deployed behaviour and has a documented
#: false-negative mechanism (it can stop before a real multi-absorber system is
#: found); the alternatives exist in the reference but were never deployed.
_EARLY_STOP_TO_REFERENCE = {
    "baseline": "baseline",
    "no_null_stop": "A",
    "pre_occam": "D",
}


@dataclass(frozen=True)
class Config:
    """Everything the inference path reads, apart from the trained model.

    Distances in velocity are given in km/s and converted internally, matching
    the reference implementation's ``kms_to_z``.
    """

    # --- rest-frame search window -------------------------------------------
    #: Rest-frame window the GP is evaluated on, angstroms.
    min_lambda: float = 911.75
    max_lambda: float = 1250.0

    # --- Lyman-series forward model -----------------------------------------
    #: Lyman-series members used for the mean-flux suppression / forest noise.
    num_forest_lines: int = 31
    #: Lyman-series members used for the absorber's own Voigt profile.
    num_lines: int = 3

    # --- mean-flux prior ------------------------------------------------------
    #: Effective-optical-depth normalisation, tau_eff = tau_0 (1+z)^beta.
    #: Default is Turner et al. (2024), arXiv:2405.06743.
    prev_tau_0: float = 0.00246
    prev_beta: float = 3.62

    # --- absorber redshift window --------------------------------------------
    #: z_abs search window is inset from the quasar redshift and the Lyman limit
    #: by these velocities.
    max_z_cut_kms: float = 3000.0
    min_z_cut_kms: float = 3000.0
    #: Minimum velocity separation between two absorbers in a multi-absorber model.
    min_z_separation_kms: float = 3000.0
    #: The absorber-existence prior counts catalogued quasars with
    #: ``z < z_qso + this``, expressed as a velocity.
    prior_z_qso_increase_kms: float = 30000.0

    # --- absorber model -------------------------------------------------------
    #: Maximum number of absorbers modelled per spectrum.
    max_absorbers: int = 4
    #: Number of quasi-Monte-Carlo samples in the evidence integral.
    num_samples: int = 50_000
    #: Name of the QMC sample grid this operating point uses. Kept explicit so a
    #: preset never inherits a grid by accident.
    sample_grid: str = "pw14_172_225_50000"
    #: Column-density prior range, log10(N_HI / cm^-2).
    #:
    #: The deployed range reaches well below the DLA threshold. The low end
    #: regularises the inference and is required to reproduce the production
    #: catalogs, but performance there is not independently validated. Treat
    #: log10 N_HI > 20 as the trusted regime.
    log_nhi_range: tuple[float, float] = (17.2, 22.5)
    #: Mixture weight of the Prochaska et al. (2014) component in the N_HI prior.
    log_nhi_prior_alpha: float = 0.97

    # --- evidence integration -------------------------------------------------
    #: Evaluate the one-absorber evidence on a truncated prefix of the sample
    #: grid instead of the whole configured grid.
    #:
    #: **Off by default for v0.1.** The full-grid path is the
    #: conservative one and is what a production preset selects; FILTER is a
    #: screening approximation that a caller must ask for by name. It changed
    #: classification in 3 of 15 constructed cases against the adopted 100k
    #: full-grid reference, so it is not interchangeable with the full-grid
    #: evidence. See ``docs/filter.md``.
    filter_low_likelihood: bool = False
    #: Floor on the coarse-scan budget; the scan uses max(num_samples // 20, this).
    filter_n_initial_floor: int = 5000
    #: When the coarse scan finds no viable region: False stops early with the
    #: coarse 1-absorber evidence (deployed), True falls through to the full
    #: sample set.
    filter_empty_mask_fallthrough: bool = False
    #: Multi-absorber early-stop policy. ``"baseline"`` is deployed.
    early_stop_mode: EarlyStopMode = "baseline"

    # --- instrument -----------------------------------------------------------
    #: Voigt backend name. ``"numpy"`` is the official backend and is always
    #: available; ``"libcerf"`` exists only where the optional compiled extension
    #: was built, and reproduces the Faddeeva implementation behind the deployed
    #: catalogues. Naming a backend that was not built raises at construction
    #: time rather than silently falling back to a different forward model.
    voigt_backend: str = "numpy"
    #: Named line-spread function; see :mod:`gp_dla_finder.voigt`.
    lsf_kernel: str = PRODUCTION_KERNEL
    #: Apply instrumental broadening to absorber profiles.
    broadening: bool = True
    #: Pixel spacing, dex, used to pad the spectrum for the LSF convolution.
    pixel_spacing_dex: float = 1e-4

    # --- mean-flux empirical Bayes -------------------------------------------
    #: Fit tau_0 per spectrum before inference instead of using ``prev_tau_0``.
    enable_tau_eb: bool = True
    #: Candidate multipliers of ``prev_tau_0`` scanned by the fit.
    tau_eb_factors: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    #: Objective for the fit: the null-model evidence (cheap) or the absorber model.
    tau_eb_objective: Literal["null", "absorber"] = "null"
    #: Mask strongly negative residuals during the fit. Off in production: at
    #: population scale it over-corrects.
    tau_eb_apply_hcd_mask: bool = False
    tau_eb_mask_threshold_sigma: float = 1.5

    # --- reproducibility -------------------------------------------------------
    #: Seed for the multi-absorber resampler. An integer makes a run reproducible;
    #: ``None`` opts into nondeterminism explicitly.
    seed: int | None = 0

    # --- data-quality selection ------------------------------------------------
    #: Named catalog-selection policy, or ``None`` for no policy. The production
    #: presets name the deployed DESI requirement; a custom
    #: configuration must choose one deliberately, because a quality cut decides
    #: which spectra reach a catalogue and must never be inferred from an
    #: instrument label. ``None`` leaves selection entirely to the caller and is
    #: not a validated catalogue rule -- see :mod:`gp_dla_finder.quality`.
    quality_policy: str | None = None

    # --- bit compatibility -----------------------------------------------------
    #: Named compatibility profile; see :mod:`gp_dla_finder.compat`. The default
    #: reproduces the pinned reference bitwise, including two floating-point
    #: no-ops it performs. Selecting ``"clean"`` drops them and changes results,
    #: which is why the profile name travels in result provenance.
    compatibility: str = DEFAULT_COMPATIBILITY

    # --- provenance ------------------------------------------------------------
    #: Name recording what this configuration reproduces.
    #:
    #: There is no default. Constructing a ``Config`` without naming a preset
    #: raises, so no workflow can silently inherit the DESI production operating
    #: point. Use :meth:`desi_y3`, or pass ``preset="custom"``
    #: to state deliberately that this configuration reproduces nothing standard.
    preset: str = _UNSET
    #: Opt in to the EXPERIMENTAL two-absorber path.
    #:
    #: ``max_absorbers=2`` alone is not enough: this must also be True. Two
    #: deliberate choices, because the M2 estimator matches the legacy
    #: implementation on the tested surface but is not a validated close-pair
    #: method -- it is weak on close pairs and low signal, and its bounded
    #: benchmark is not a survey calibration. A user should not reach it by
    #: nudging a number.
    experimental_multi_absorber: bool = False
    #: The named preset this configuration STARTED from, kept even after
    #: scientifically consequential overrides.
    #:
    #: ``preset`` is the *effective* identity and becomes ``"<base>+modified"``
    #: as soon as anything consequential is overridden; ``base_preset`` answers
    #: the other question a reader needs -- which operating point was the
    #: starting point. ``Config.desi_y3(filter_low_likelihood=True)`` is not the
    #: canonical ``desi_y3`` and must not claim that name, but it did not come
    #: from nowhere either.
    base_preset: str = ""

    def __post_init__(self) -> None:
        if self.preset is _UNSET:
            raise ValueError(
                "Config requires an explicit preset. Use a named preset, e.g.\n"
                "    Config.desi_y3()                       # deployed DESI Y3 "
                "operating point\n"
                "    Config.desi_y3().replace(num_samples=10_000)\n"
                "or pass preset='custom' to declare a non-standard configuration."
            )
        if not isinstance(self.preset, str) or not self.preset.strip():
            raise ValueError(f"preset must be a non-empty name, got {self.preset!r}")
        if not self.base_preset:
            # A config constructed directly is its own starting point.
            object.__setattr__(self, "base_preset", self.preset.split("+", 1)[0])
        if self.min_lambda >= self.max_lambda:
            raise ValueError(
                f"min_lambda ({self.min_lambda}) must be < "
                f"max_lambda ({self.max_lambda})"
            )
        if not 1 <= self.num_lines <= 31:
            raise ValueError(f"num_lines must be in [1, 31], got {self.num_lines}")
        if not 1 <= self.num_forest_lines <= 31:
            raise ValueError(
                f"num_forest_lines must be in [1, 31], got {self.num_forest_lines}"
            )
        if self.max_absorbers < 1:
            raise ValueError(f"max_absorbers must be >= 1, got {self.max_absorbers}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        lo, hi = self.log_nhi_range
        if not lo < hi:
            raise ValueError(
                f"log_nhi_range must be increasing, got {self.log_nhi_range}"
            )
        if self.early_stop_mode not in _EARLY_STOP_TO_REFERENCE:
            raise ValueError(
                f"early_stop_mode must be one of "
                f"{sorted(_EARLY_STOP_TO_REFERENCE)}, got {self.early_stop_mode!r}"
            )
        if self.tau_eb_objective not in ("null", "absorber"):
            raise ValueError(f"unknown tau_eb_objective {self.tau_eb_objective!r}")
        lsf_kernel(self.lsf_kernel)  # raises on an unknown kernel
        get_backend(self.voigt_backend)  # raises on an unknown/unbuilt backend
        compatibility_profile(self.compatibility)  # raises on an unknown profile
        if self.quality_policy is not None:
            quality_policy(self.quality_policy)  # raises on an unknown policy

    # --- derived quantities ---------------------------------------------------

    @property
    def compatibility_profile(self) -> CompatibilityProfile:
        """The arithmetic-fidelity profile this configuration runs under."""
        return compatibility_profile(self.compatibility)

    @property
    def selected_quality_policy(self) -> QualityPolicy | None:
        """The named quality policy, or ``None`` when selection is the caller's."""
        if self.quality_policy is None:
            return None
        return quality_policy(self.quality_policy)

    @property
    def backend_provenance(self) -> Mapping[str, object]:
        """What the selected Voigt backend is, for a result's provenance block.

        For a compiled backend this includes the libcerf version, the sha256 of
        the shared library it linked, the compiler and the redacted optimisation
        flags -- because a compiled backend's numbers depend on all of them.
        """
        return backend_provenance(self.voigt_backend)

    @property
    def evidence_mode(self) -> str:
        """How the one-absorber evidence integral is evaluated.

        ``"exact"`` evaluates the **whole configured sample grid** and is the
        v0.1 default for every production preset.

        The name ``"exact"`` is an API compatibility label, not a mathematical
        claim. It means "the package's full configured QMC-grid estimator" -- not
        an analytic integral, not zero numerical error, and not a formal
        convergence proof. The grid itself is finite and its own residual error
        is unquantified (see ``docs/filter.md``).

        ``"filter"`` evaluates only a prefix of that grid and is a **screening
        approximation**. It must be requested explicitly; no production preset
        selects it. The chosen mode is a scientific choice, not an optimisation
        detail, so it travels in result provenance and labels every value it
        produces.
        """
        return "filter" if self.filter_low_likelihood else "exact"

    @staticmethod
    def kms_to_z(kms: float) -> float:
        """Velocity in km/s as a redshift difference."""
        return (kms * 1000.0) / SPEED_OF_LIGHT

    @property
    def max_z_cut(self) -> float:
        return self.kms_to_z(self.max_z_cut_kms)

    @property
    def min_z_cut(self) -> float:
        return self.kms_to_z(self.min_z_cut_kms)

    @property
    def min_z_separation(self) -> float:
        return self.kms_to_z(self.min_z_separation_kms)

    @property
    def prior_z_qso_increase(self) -> float:
        return self.kms_to_z(self.prior_z_qso_increase_kms)

    @property
    def convolution_half_width(self) -> int:
        """Pixels of padding each side needed for the LSF convolution."""
        return kernel_half_width(self.lsf_kernel)

    @property
    def n_models(self) -> int:
        """Number of models compared: null plus 1..max_absorbers."""
        return 1 + self.max_absorbers

    @property
    def model_labels(self) -> tuple[str, ...]:
        """Labels for the model-posterior vector, in order."""
        return ("null",) + tuple(
            f"{k}_absorber" if k == 1 else f"{k}_absorbers"
            for k in range(1, self.max_absorbers + 1)
        )

    @property
    def _reference_early_stop_mode(self) -> str:
        """The reference implementation's name for this early-stop policy."""
        return _EARLY_STOP_TO_REFERENCE[self.early_stop_mode]

    #: Fields whose value cannot change what the inference computes, so
    #: overriding them does not make a configuration "modified".
    INERT_FIELDS: ClassVar[frozenset[str]] = frozenset({"preset", "base_preset"})

    def replace(self, **changes) -> Config:
        """A modified copy, relabelled if anything consequential changed.

        The effective ``preset`` becomes ``"<base>+modified"``; ``base_preset``
        is preserved, so provenance answers both "which named operating point
        was the starting point" and "what exactly ran".

        Overriding only inert bookkeeping does not relabel: a name is a claim
        about the numerics, and re-stating a value that cannot move a number is
        not a change to them.
        """
        consequential = {
            key: value
            for key, value in changes.items()
            if key not in self.INERT_FIELDS and getattr(self, key, None) != value
        }
        if consequential and "preset" not in changes:
            changes["preset"] = f"{self.base_preset}+modified"
        changes.setdefault("base_preset", self.base_preset)
        return replace(self, **changes)

    @property
    def is_modified(self) -> bool:
        """Whether this configuration diverges from its named base preset."""
        return self.preset != self.base_preset

    def normalized(self) -> dict[str, object]:
        """Every scientifically consequential field, in a stable order.

        The complete answer to "what exactly produced this result", as opposed
        to the preset name, which is only a label.
        """
        return {
            field.name: _normalize(getattr(self, field.name))
            for field in sorted(fields(self), key=lambda f: f.name)
            if field.name not in self.INERT_FIELDS
        }

    @property
    def digest(self) -> str:
        """Stable short hash of :meth:`normalized`.

        Two configurations with the same digest compute the same thing, whatever
        they are called. Comparing digests is how results are checked for
        compatibility before being combined into one catalogue.
        """
        payload = json.dumps(self.normalized(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # --- validation ------------------------------------------------------------

    def validate_against(self, model: GPModel) -> None:
        """Check a trained model can serve this configuration.

        Raises
        ------
        ValueError
            If the model's rest-frame grid does not span the search window, or
            if the model does not record a normalisation band. Both would
            otherwise fail deep inside the likelihood, or worse, silently
            produce a wrong prediction.
        """
        if not model.covers(self.min_lambda, self.max_lambda):
            lo, hi = model.rest_wavelength_range
            raise ValueError(
                f"model {model.name!r} covers rest-frame [{lo:.2f}, {hi:.2f}] A "
                f"but the configured search window is "
                f"[{self.min_lambda:.2f}, {self.max_lambda:.2f}] A"
            )
        lo_norm = model.normalization_min_lambda
        hi_norm = model.normalization_max_lambda
        if lo_norm is None or hi_norm is None:
            raise ValueError(
                f"model {model.name!r} does not record both ends of its "
                "normalisation band; flux normalisation would not match "
                "training. Supply a model that records one."
            )
        if math.isnan(lo_norm) or math.isnan(hi_norm):
            raise ValueError(
                f"model {model.name!r} was trained without flux normalisation "
                "(its normalisation band is NaN). Inference would normalise "
                "spectra on a band the model never saw."
            )
        if not lo_norm < hi_norm:
            raise ValueError(
                f"model {model.name!r} has a non-increasing normalisation band "
                f"[{lo_norm}, {hi_norm}]"
            )
        # NOTE: the normalisation band is deliberately NOT required to lie
        # inside the GP interpolation grid. Normalisation is applied to the input
        # spectrum before the GP is evaluated on the search window, and the
        # legacy eBOSS model is a real counter-example: its band (1425-1475 A)
        # sits redward of a grid that ends near 1421 A. Coverage of the band is
        # a property of the *spectrum*, and is checked at inference time.

    # --- presets ---------------------------------------------------------------

    @classmethod
    def _from_preset(cls, name: str, defaults: dict, overrides: dict) -> Config:
        """Build a preset, relabelling it if an override changes the numerics.

        The classmethod path and :meth:`replace` follow ONE convention (PI
        ``Config.desi_y3(filter_low_likelihood=True)`` is not the
        canonical ``desi_y3`` and does not keep that name, exactly as
        ``Config.desi_y3().replace(filter_low_likelihood=True)`` does not.
        ``base_preset`` records where it started in both cases.
        """
        consequential = {
            key: value
            for key, value in overrides.items()
            if key not in cls.INERT_FIELDS and defaults.get(key, _UNSET) != value
        }
        effective = f"{name}+modified" if consequential else name
        return cls(
            **{
                **defaults,
                "preset": overrides.get("preset", effective),
                "base_preset": overrides.get("base_preset", name),
                **{k: v for k, v in overrides.items() if k not in cls.INERT_FIELDS},
            }
        )

    @classmethod
    def desi_y3(cls, **overrides) -> Config:
        """The deployed DESI Y3 production operating point.

        This is the configuration behind the production catalogs: a two-way
        model comparison (null versus k absorbers, no separate sub-DLA channel),
        a column-density prior spanning [17.2, 22.5], and the full configured
        evidence grid.
        """
        defaults = {
            "quality_policy": "desi-y3-reference",
            # Stated, not inherited: a production preset must never pick up
            # a screening approximation from a field default.
            "filter_low_likelihood": False,
        }
        return cls._from_preset("desi_y3", defaults, overrides)

    @classmethod
    def desi_y3_refined(cls, **overrides) -> Config:
        """As :meth:`desi_y3` but with the 100,000-sample QMC grid.

        A denser numerical integration grid, selected explicitly. At least one
        archived mock-catalog production used 100k rather than 50k, so
        reproducing that run means choosing this preset deliberately and recording
        it.

        "Refined" describes the integration grid only. It does **not** establish
        identity with any deployed production array, and does not by itself make a
        result more accurate scientifically.
        """
        defaults = {
            "quality_policy": "desi-y3-reference",
            # Stated, not inherited: a production preset must never pick up
            # a screening approximation from a field default.
            "filter_low_likelihood": False,
            "num_samples": 100_000,
            "sample_grid": "pw14_172_225_100000",
        }
        return cls._from_preset("desi_y3_refined", defaults, overrides)

    @classmethod
    def desi_y3_fast(cls, **overrides) -> Config:
        """As :meth:`desi_y3` but with a smaller sample budget.

        For exploration and tutorials. The evidence integral is noisier, so
        multi-absorber results in particular will differ from production.
        """
        defaults = {
            "quality_policy": "desi-y3-reference",
            # Stated, not inherited: a production preset must never pick up
            # a screening approximation from a field default.
            "filter_low_likelihood": False,
            "num_samples": 10_000,
            "sample_grid": "pw14_172_225_10000",
        }
        return cls._from_preset("desi_y3_fast", defaults, overrides)
