"""The high-level entry point: give it a spectrum and get a typed Result back.

The finder runs a
:class:`~gp_dla_finder.gp.spectrum.Spectrum` through the quality policy, evidence
calculation, and absorber prior, then returns a typed result that the catalog
writer can consume.

It covers:

* the **null versus one-absorber** comparison, in exact or FILTER mode;
* all four statuses — ``completed``, ``insufficient_data``, ``quality_rejected``
  and ``failed`` — as *results*, not exceptions;
* the legacy probability fields, computed as the reference defines them rather
  than derived from one another;
* full provenance: preset, model, prior, grid, backend, compatibility profile,
  quality policy, evidence mode and the per-spectrum evaluated-sample count.

With ``max_absorbers=2`` and ``experimental_multi_absorber=True``, it also
computes the two-absorber evidence and returns an M0/M1/M2 ladder with aligned
priors, posteriors and a selected model. The model is workable in a production
workflow, while its current statistical estimator remains experimental. It
reproduces the reference evidences bitwise under a controlled seed, but close
pairs and low-signal pairs are known weaknesses. An M2 result writes to a
catalogue as two flat rows sharing a
``TARGETID``; the ladder itself travels in
:mod:`gp_dla_finder.io.structured`. See :mod:`gp_dla_finder.multi`.

It does **not** yet cover:

* **more than two absorbers.** ``max_absorbers > 2`` is refused rather than
  truncated, so a configuration cannot claim a multiplicity that was never
  evaluated;
* **FILTER together with two absorbers**, refused for the same reason: the
  hybrid is neither full-grid M2 evidence nor the reference's multi-absorber
  FILTER path;
* **validated point estimates.** The current redshift and column density come
  from the best evaluated grid point. They are usable for inspection, but not
  yet reliable enough to quote as science measurements; uncertainty fields stay
  NaN;
* multi-absorber early stops beyond the NaN rung.

Per-spectrum empirical-Bayes mean-flux fitting **is** implemented -- see
:mod:`gp_dla_finder.mean_flux`.

Unavailable quantities remain absent from the result as NaN or an empty list.
They are not replaced with values that could be read as measurements.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from . import __version__
from ._immutable import deep_freeze
from .catalogue import (
    AbsorberRow,
    Catalogue,
    ModelRow,
    SpectrumRow,
    reference_dlaid,
)
from .config import Config
from .errors import NumericalError, SpectrumError
from .gp.evidence import (
    absorber_search_window,
    assemble_model,
    coarse_scan_size,
    null_log_evidence,
    one_absorber_log_evidence,
)
from .gp.spectrum import InsufficientData, PreparedSpectrum, Spectrum, prepare_spectrum
from .mean_flux import MeanFluxFit, fit_tau_0
from .model import GPModel, load_model
from .multi import (
    RNG_ALGORITHM,
    ModelLadder,
    seeded_resampler,
    two_absorber_log_evidence,
)
from .performance import warn_once_about_blas_threads
from .prior import AbsorberPrior, load_prior
from .samples import AbsorberSampleGrid, load_sample_grid
from .voigt import backend_provenance

__all__ = [
    "AbsorberCandidate",
    "ExperimentalFeatureNotEnabled",
    "MultiAbsorberNotRepresentable",
    "SampleGridMismatch",
    "Finder",
    "Result",
    "results_to_catalogue",
    "screening_score",
]


@dataclass(frozen=True)
class AbsorberCandidate:
    """One absorber candidate: an evaluated grid point, not a detection.

    A completed result carries a candidate whenever the evidence path evaluated
    the absorber model, including spectra whose absorber posterior is far below
    a useful threshold. It is where that model fits best, which is still a
    well-defined place in a spectrum with no absorber.

    In other words, ``len(result.absorber_candidates) == 1`` means "the
    search ran", not "a DLA was found". Whether a candidate is a detection is a
    policy question answered by
    :meth:`Result.detected` at an explicit threshold, and applied at the
    catalogue boundary -- never silently inside the scientific result.

    ``grid_z_abs`` and ``grid_log_nhi`` are the **best evaluated grid point**.
    They provide a usable preliminary location, but they are not a validated MAP
    estimate. The field names keep ``grid`` explicit so their status is clear.
    """

    grid_z_abs: float
    grid_log_nhi: float
    #: Which model this candidate belongs to: 1 for the one-absorber model, 2
    #: for a member of the best two-absorber pair. Without it, two candidates in
    #: a list say nothing about which fit produced them.
    model: int = 1
    #: Uncertainties, if a validated estimator ever supplies them. NaN for now.
    z_abs_err: float = float("nan")
    log_nhi_err: float = float("nan")


@dataclass(frozen=True)
class Result:
    """The inference record for one spectrum, including an incomplete run.

    Immutable, and immutable *through* its containers. The dataclass is frozen,
    the candidate list becomes a tuple, and provenance is recursively frozen:
    nested mappings become read-only proxies, nested sequences become tuples, and
    arrays are copied and marked non-writable. Mutating whatever the caller
    passed in cannot reach the result afterwards.

    This keeps the provenance tied to the run that produced it, even if the
    caller later changes the objects originally passed to the package.
    """

    targetid: int
    z_qso: float
    status: str
    reason: str = ""
    #: Log evidences for M0, M1 and (when ``max_absorbers >= 2``) M2.
    #:
    #: ``None`` when only the null-versus-one comparison ran. Check
    #: :attr:`ModelLadder.complete` before reading the posteriors: a stopped
    #: rung is a model that was never measured, not one measured as impossible.
    ladder: ModelLadder | None = None
    #: The per-spectrum mean-flux scan, when one ran. ``None`` otherwise.
    #:
    #: Carried whole rather than summarised: the winner alone cannot tell a
    #: reader whether the grid was decisive or whether the maximum sat against an
    #: edge, and both change how much the fitted value should be trusted.
    mean_flux: MeanFluxFit | None = None
    #: Evaluated absorber candidates -- NOT detections. See
    #: :class:`AbsorberCandidate`; use :meth:`detected` for the policy question.
    absorber_candidates: Sequence[AbsorberCandidate] = field(default_factory=tuple)

    #: Model evidences. NaN unless the status is ``completed``.
    log_evidence_null: float = float("nan")
    log_evidence_absorber: float = float("nan")
    #: Two-model posterior of at least one absorber, under the named prior.
    #: Posterior probability of AT LEAST ONE absorber.
    #:
    #: On an M2 run this is the multi-model value, summed over the completed
    #: absorber models. On a null-versus-one run it is the two-model value.
    #: One number, one meaning -- previously the ladder and this field could
    #: disagree, and ``detected()`` silently used the two-model one.
    p_absorber: float = float("nan")
    p_null: float = float("nan")
    #: The two-model M0/M1 posterior, kept for legacy comparability.
    #:
    #: Equal to :attr:`p_absorber` when only M0 and M1 were evaluated. On an M2
    #: run it is the value the OLD two-model calculation would have given, and
    #: it is never used for selection or detection.
    legacy_two_model_p_absorber: float = float("nan")
    #: log posterior of the one-absorber model. Carried independently of
    #: ``p_absorber``: the reference treats them as different quantities.
    logp_absorber: float = float("nan")
    logp_null: float = float("nan")

    #: ``"exact"`` or ``"filter"``. Empty when the status is not ``completed``.
    #: This describes the evidence fields *above*, which a later refinement stage
    #: may replace.
    evidence_mode: str = ""
    n_evaluated: int = 0

    # --- the screening stage, recorded independently of the final evidence ---
    #
    # Stored, not derived. Deriving it from the evidence fields
    # at catalogue-write time meant a future full-grid refinement -- which
    # rewrites those fields and the mode -- would erase the record of what
    # screening actually said, and a screened non-detection with no absorber row
    # had nowhere to carry it at all.
    #:
    #: FILTER-prefix log Bayes factor, absorber over null. A ranking statistic:
    #: not a probability, not an evidence. NaN means **no screening stage ran**,
    #: never "screened and scored zero".
    screening_score: float = float("nan")
    #: Samples the screening stage evaluated. 0 when it did not run.
    screening_n_evaluated: int = 0
    #: The two evidences the screening stage produced, kept so the screening
    #: decision can be re-derived after a refinement stage replaces the finals.
    screening_log_evidence_null: float = float("nan")
    screening_log_evidence_absorber: float = float("nan")
    n_usable_pixels: int = 0
    quality_fraction: float = float("nan")

    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "absorber_candidates", tuple(self.absorber_candidates))
        # deep_freeze, not MappingProxyType: the proxy protects only the OUTER
        # mapping, so a nested dict or list inside provenance stayed writable
        # while the docstring promised immutability through the containers.
        object.__setattr__(self, "provenance", deep_freeze(self.provenance))

    @property
    def log_bayes_factor(self) -> float:
        """log evidence ratio, absorber over null, in this result's mode."""
        return self.log_evidence_absorber - self.log_evidence_null

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def was_screened(self) -> bool:
        """Whether a FILTER screening stage ran for this spectrum."""
        return not np.isnan(self.screening_score)

    def refined(
        self,
        *,
        log_evidence_null: float,
        log_evidence_absorber: float,
        p_absorber: float,
        p_null: float,
        logp_absorber: float,
        logp_null: float,
        n_evaluated: int,
        provenance: Mapping[str, object] | None = None,
    ) -> Result:
        """A copy with full-grid evidences replacing the screened ones.

        The forward-compatibility path for the two-stage workflow: screen with
        FILTER, then re-run candidates near a decision boundary on the full grid.
        The screening fields are carried through **unchanged**, so the product
        records both what screening said and what refinement concluded.

        Nothing calls this yet -- the refinement stage is not implemented. It
        exists so the representation is testable now rather than discovered to be
        lossy later.
        """
        return replace(
            self,
            log_evidence_null=log_evidence_null,
            log_evidence_absorber=log_evidence_absorber,
            p_absorber=p_absorber,
            p_null=p_null,
            # Carried forward, not recomputed: it records what the ORIGINAL
            # stage said, which is the whole reason it exists.
            legacy_two_model_p_absorber=self.legacy_two_model_p_absorber,
            logp_absorber=logp_absorber,
            logp_null=logp_null,
            evidence_mode="exact",
            n_evaluated=n_evaluated,
            provenance=dict(provenance)
            if provenance is not None
            else dict(self.provenance),
        )

    def detected(self, threshold: float) -> bool:
        """Whether this counts as a detection at ``threshold``.

        ``threshold`` is required and has no default. A detection depends on the
        analysis threshold, so callers must state that policy explicitly.
        """
        return self.completed and bool(self.p_absorber >= threshold)


class ExperimentalFeatureNotEnabled(RuntimeError):
    """An experimental path was requested without opting in to it."""


class SampleGridMismatch(ValueError):
    """The configuration describes a different N_HI prior than its grid uses.

    Raised at ``Finder`` construction. See
    :meth:`Finder._check_grid_matches_config` for why this is an error rather
    than a warning.
    """


class MultiAbsorberNotRepresentable(NotImplementedError):
    """A result reports more absorbers than its ladder has models.

    Not reachable in normal use -- the selected model index *is* the absorber
    count, so the two agree by construction. It exists because the alternative
    on a malformed result is a NaN in ``LOGP_DLA`` or ``MODEL_P``, and a wrong
    number in a legacy column is worse than a refusal.

    Ordinary multi-absorber results are **not** refused: they write as several
    flat rows sharing a ``TARGETID``, which is what the reference does.
    """


class Finder:
    """Run the current inference path for one spectrum at a time.

    Construct once and reuse: the model, prior and sample grid are loaded here,
    not per spectrum, and loading them per call would dominate the runtime.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: GPModel | None = None,
        prior: AbsorberPrior | None = None,
        grid: AbsorberSampleGrid | None = None,
        warn_about_threads: bool = True,
    ) -> None:
        # The v0.1 default is the null-versus-one path, which is the part that
        # is validated. Config.desi_y3() declares max_absorbers=4, which this
        # package refuses, so defaulting to it made Finder() raise as soon as
        # the assets loaded. The DESI operating point stays available and
        # truthful; it just is not the default.
        self.config = config if config is not None else Config.desi_y3(max_absorbers=1)
        self.model = model if model is not None else load_model()
        self.prior = prior if prior is not None else load_prior()
        self.grid = (
            grid if grid is not None else load_sample_grid(self.config.sample_grid)
        )
        self.config.validate_against(self.model)
        self._check_multi_absorber_support(self.config)
        self._check_grid_matches_config(self.config, self.grid)
        self._warn_about_threads = warn_about_threads

    @staticmethod
    def _check_grid_matches_config(config: Config, grid: AbsorberSampleGrid) -> None:
        """Refuse a configuration that describes a different grid than it uses.

        ``log_nhi_range`` and ``log_nhi_prior_alpha`` are **declarative**: the
        column-density prior is drawn once, into the sample grid, and nothing
        re-reads those fields at inference time. So a configuration naming a
        support the grid does not have changes the config digest, the recorded
        preset, and the scientific description of the run -- and changes nothing
        about the calculation.

        That is worse than an error, because the record is wrong in a way a
        reader cannot detect: the catalogue would say the search covered
        ``log10 N_HI > 20.3`` while the samples went down to 17.2. So it fails
        here rather than warning.

        A grid that carries no provenance is refused outright rather than
        checked on sample count alone: a count match is not evidence that the
        prior is what the configuration claims.
        """
        # A grid that cannot be checked cannot be used. Two ways to fail: the
        # provenance may be absent, or it may be present and not describe this
        # file -- a sidecar copied from another grid, or arrays edited after it
        # was written. Both leave the recorded configuration unverifiable
        # against what actually ran, which is the whole point of the check.
        if not grid.usable_for_inference:
            reasons = "\n".join(f"  {reason}" for reason in grid.unusable_because)
            raise SampleGridMismatch(
                f"the sample grid {grid.name!r} cannot be used for inference. "
                f"Its provenance is either incomplete or does not describe the "
                f"file and arrays that were loaded, so nothing about it can be "
                f"trusted against the configuration.\n\n"
                f"{reasons}\n\n"
                "tools/build_sample_grid.py writes all of this into a "
                "<name>.json sidecar beside the .npz, so a grid built the "
                "supported way needs nothing extra -- keep the two files "
                "together:\n\n"
                "    python tools/build_sample_grid.py --name my_grid \\\n"
                "        --num-samples 50000 --log-nhi 20.3 22.5 --out ./grids\n"
                "    grid = load_sample_grid(path='grids/my_grid.npz')\n"
                "    Finder(config, grid=grid)\n\n"
                "Loading such a grid is still fine for inspecting the arrays. "
                "It is inference that is refused, because a run using it would "
                "record a configuration that nothing can confirm it used."
            )

        disagreements = []

        # config.sample_grid enters the config digest and the GPDLF_SAMPLE_GRID
        # record, so an explicit grid whose name differs makes both describe a
        # grid the run did not use.
        if grid.name != config.sample_grid:
            disagreements.append(
                f"  sample_grid: the configuration names {config.sample_grid!r}, "
                f"the grid is {grid.name!r}"
            )

        if grid.num_samples != config.num_samples:
            disagreements.append(
                f"  num_samples: the configuration says {config.num_samples}, "
                f"the grid holds {grid.num_samples}"
            )

        support = grid.declared_support
        if support is not None and tuple(support) != tuple(config.log_nhi_range):
            disagreements.append(
                f"  log_nhi_range: the configuration says "
                f"{tuple(config.log_nhi_range)}, the grid was generated over "
                f"{tuple(support)}"
            )

        alpha = grid.declared_prior_alpha
        if alpha is not None and alpha != config.log_nhi_prior_alpha:
            disagreements.append(
                f"  log_nhi_prior_alpha: the configuration says "
                f"{config.log_nhi_prior_alpha}, the grid was generated with "
                f"{alpha}"
            )

        if not disagreements:
            return

        raise SampleGridMismatch(
            f"the configuration describes a different column-density prior than "
            f"the sample grid {grid.name!r} it would use:\n\n"
            + "\n".join(disagreements)
            + "\n\nThe N_HI prior is baked into the grid when the grid is "
            "built; log_nhi_range and log_nhi_prior_alpha describe it and are "
            "not read at inference time. Running anyway would record a prior "
            "the calculation never used.\n\n"
            "Either name a packaged grid that matches --\n"
            "    Config.desi_y3(num_samples=10_000, "
            "sample_grid='pw14_172_225_10000')\n"
            "-- or build one and pass it in:\n"
            "    python tools/build_sample_grid.py --name my_grid \\\n"
            "        --num-samples 50000 --log-nhi 20.3 22.5 --out ./grids\n"
            "    grid = load_sample_grid(path='grids/my_grid.npz')\n"
            "    Finder(config, grid=grid)\n\n"
            "Set sample_grid to the custom grid's name as well, so the config "
            "digest and the catalogue's GPDLF_SAMPLE_GRID name the grid the run "
            "actually used."
        )

    @staticmethod
    def _check_multi_absorber_support(config: Config) -> None:
        """Refuse configurations whose multiplicity this path cannot deliver.

        Two separate refusals, both about not claiming more than is computed.

        ``max_absorbers > 2``: only M0, M1 and M2 exist here. Running a preset
        that says four and quietly stopping at two would put "max_absorbers=4"
        in provenance for a calculation that never went past two.

        ``filter_low_likelihood`` with ``max_absorbers >= 2``: in FILTER mode the
        one-absorber proposal covers only the evaluated prefix while the
        two-absorber loop still walks the whole first-absorber grid. That hybrid
        is neither full-grid M2 evidence nor the reference's multi-absorber
        FILTER path, and labelling it as either would be wrong.
        """
        if config.max_absorbers > 2:
            raise NotImplementedError(
                f"max_absorbers={config.max_absorbers} is not implemented: this "
                "package evaluates M0, M1 and M2 only. Use max_absorbers=2 for "
                "the bounded multi-absorber path, or max_absorbers=1 for the "
                "null-versus-one comparison. A preset claiming more would "
                "record a multiplicity the calculation never reached."
            )
        if config.max_absorbers >= 2 and config.filter_low_likelihood:
            raise NotImplementedError(
                "FILTER is not supported with max_absorbers >= 2. The "
                "one-absorber proposal would cover only the evaluated prefix "
                "while the two-absorber scan walks the full grid, which is "
                "neither full-grid M2 evidence nor the reference's "
                "multi-absorber FILTER path. Use the full grid "
                "(filter_low_likelihood=False) or max_absorbers=1."
            )
        if config.max_absorbers >= 2 and not config.experimental_multi_absorber:
            raise ExperimentalFeatureNotEnabled(
                "max_absorbers >= 2 needs experimental_multi_absorber=True as "
                "well.\n\n"
                "The two-absorber path reproduces the legacy implementation's "
                "evidences bitwise under a controlled seed, so it is useful for "
                "reference-fidelity work. It is NOT a validated close-pair "
                "method: on a bounded 60-spectrum benchmark it recovers the "
                "right multiplicity about 80% of the time, it is weak on close "
                "pairs and low signal, and that benchmark is not a survey "
                "calibration.\n\n"
                "    Config.desi_y3_fast(max_absorbers=2, "
                "experimental_multi_absorber=True)"
            )

    # -- provenance ---------------------------------------------------------
    def _provenance(
        self,
        evidence_mode: str,
        n_evaluated: int,
        *,
        mean_flux: MeanFluxFit | None = None,
    ) -> dict[str, object]:
        backend = dict(backend_provenance(self.config.voigt_backend))
        return {
            "gpdlf_version": __version__,
            "preset": self.config.preset,
            "base_preset": self.config.base_preset,
            # The COMPLETE configuration identity. The twelve named fields below
            # are a readable summary; this is what actually decides whether two
            # results were computed the same way. A field added to Config
            # joins it automatically.
            "config_digest": self.config.digest,
            "evidence_mode": evidence_mode,
            "num_samples": self.config.num_samples,
            "n_evaluated": n_evaluated,
            # The SEARCH LIMIT, not a detection count. It is what tells a reader
            # of the catalogue how many absorber models P_DLA was summed over.
            "max_absorbers": self.config.max_absorbers,
            # The grid OBJECT's name, not the configuration's string. The two
            # are equal -- _check_grid_matches_config refuses a run where they
            # are not -- but taking it from the grid means provenance names what
            # was actually read rather than what the preset happened to say.
            "sample_grid": self.grid.name,
            "model": self.model.name,
            "prior": self.prior.name,
            "lsf_kernel": self.config.lsf_kernel,
            "quality_policy": self.config.quality_policy or "",
            "seed": self.config.seed,
            "rng_algorithm": RNG_ALGORITHM,
            "experimental": (
                "multi_absorber" if self.config.experimental_multi_absorber else ""
            ),
            **{f"backend_{k}": v for k, v in backend.items()},
            **dict(self.config.compatibility_profile.provenance()),
            # The mean-flux fit is per spectrum, so its record belongs on the
            # result rather than in the run-level configuration.
            **(mean_flux.provenance() if mean_flux is not None else {}),
        }

    def _failed(self, targetid, z_qso, status, reason, **extra) -> Result:
        return Result(
            targetid=targetid,
            z_qso=z_qso,
            status=status,
            reason=reason,
            provenance=self._provenance("", 0),
            **extra,
        )

    # -- the run ------------------------------------------------------------
    def run(self, spectrum: Spectrum, *, targetid: int = 0) -> Result:
        """Score one spectrum. Never raises for an ordinary bad input.

        A processing failure is a *result* with a status and a reason code, not
        an exception, because a batch layer needs to record it and continue —
        and because a failure silently reported as "no absorber" would corrupt
        any population statistic built from the output.
        """
        if self._warn_about_threads:
            # Once, at the run boundary. Never at import, never per likelihood.
            warn_once_about_blas_threads()
            self._warn_about_threads = False

        policy = self.config.selected_quality_policy
        quality_fraction = float("nan")
        if policy is not None:
            assessment = policy.assess(spectrum)
            quality_fraction = assessment.usable_fraction
            if not assessment.passed:
                return self._failed(
                    targetid,
                    spectrum.z_qso,
                    "quality_rejected",
                    assessment.reason or "quality_policy_rejected",
                    quality_fraction=quality_fraction,
                )

        try:
            prepared = prepare_spectrum(spectrum, self.model, self.config)
        except InsufficientData as exc:
            return self._failed(
                targetid,
                spectrum.z_qso,
                "insufficient_data",
                exc.reason,
                quality_fraction=quality_fraction,
            )

        try:
            return self._score(prepared, targetid, quality_fraction)
        except (NumericalError, SpectrumError, np.linalg.LinAlgError) as exc:
            return self._failed(
                targetid,
                spectrum.z_qso,
                "failed",
                type(exc).__name__,
                quality_fraction=quality_fraction,
                n_usable_pixels=prepared.n_pixels,
            )

    def _score(
        self, prepared: PreparedSpectrum, targetid: int, quality_fraction: float
    ) -> Result:
        config, grid = self.config, self.grid

        # Per-spectrum empirical-Bayes mean flux, when the configuration asks
        # for it. The fitted tau_0 replaces the prior for THIS spectrum only.
        mean_flux: MeanFluxFit | None = None
        if config.enable_tau_eb:
            mean_flux = fit_tau_0(
                prepared, self.model, config, objective=config.tau_eb_objective
            )
            assembled = assemble_model(
                prepared, self.model, config, tau_0=mean_flux.tau_0
            )
        else:
            assembled = assemble_model(prepared, self.model, config)

        mode = config.evidence_mode

        log_z_null = null_log_evidence(prepared, assembled)
        log_z_one, samples = one_absorber_log_evidence(
            prepared, assembled, grid, config, mode=mode, return_samples=True
        )

        # The two-absorber rung, when the configuration asks for it. A fresh
        # local stream per spectrum: one generator reused across
        # spectra would make a result depend on how many preceded it.
        ladder: ModelLadder | None = None
        two_absorber_samples = None
        partners = None
        log_z_two = float("nan")
        if config.max_absorbers >= 2:
            log_z_two, two_absorber_samples, partners = two_absorber_log_evidence(
                prepared,
                assembled,
                grid,
                config,
                one_absorber_samples=samples,
                resampler=seeded_resampler(config.seed),
            )

        n_evaluated = (
            coarse_scan_size(config) if mode == "filter" else config.num_samples
        )

        # Two-model posterior under the named prior. Computed here rather than
        # inside the evidence path, because it depends on the prior and the
        # evidence does not.
        log_prior_one = float(
            self.prior.log_priors(prepared.z_qso, 1, config.prior_z_qso_increase)[0]
        )
        log_prior_null = self.prior.log_prior_no_absorber(
            prepared.z_qso, config.prior_z_qso_increase
        )
        if config.max_absorbers >= 2:
            # Aligned model priors: P(no absorber), P(exactly 1), P(>= 2). The
            # top model absorbs the tail, matching the reference convention.
            absorber_priors = self.prior.log_priors(
                prepared.z_qso, 2, config.prior_z_qso_increase
            )
            ladder = ModelLadder(
                (log_z_null, log_z_one, log_z_two),
                (
                    log_prior_null,
                    float(absorber_priors[0]),
                    float(absorber_priors[1]),
                ),
            )

        joint = np.array([log_prior_null + log_z_null, log_prior_one + log_z_one])
        shifted = joint - joint.max()
        weights = np.exp(shifted)
        total = weights.sum()
        p_null = float(weights[0] / total)
        p_absorber = float(weights[1] / total)
        with np.errstate(divide="ignore"):
            logp_absorber = float(np.log(p_absorber))
            logp_null = float(np.log(p_null))

        # The best evaluated grid point. NOT a MAP estimate, and NOT a detection:
        # this is populated whenever anything was evaluated, however low the
        # posterior. See AbsorberCandidate.
        candidates: list[AbsorberCandidate] = []
        finite = np.isfinite(samples)
        if np.any(finite):
            z_min, z_max = absorber_search_window(prepared, config)
            z_samples = grid.sample_redshifts(z_min, z_max)
            best = int(np.nanargmax(np.where(finite, samples, -np.inf)))
            candidates.append(
                AbsorberCandidate(
                    grid_z_abs=float(z_samples[best]),
                    grid_log_nhi=float(np.log10(grid.nhi_samples[best])),
                )
            )

        # Recorded now, while this IS the screening stage. A later refinement
        # replaces the evidence fields; these survive it.
        screened = mode == "filter"

        # When M2 is the selected model, report BOTH members of the best
        # evaluated pair. Returning the single best M1 grid point for a result
        # whose selected model has two absorbers would drop one of them.
        if (
            ladder is not None
            and ladder.selected_model >= 2
            and two_absorber_samples is not None
            and partners is not None
            and np.any(np.isfinite(two_absorber_samples))
        ):
            z_min, z_max = absorber_search_window(prepared, config)
            z_samples = grid.sample_redshifts(z_min, z_max)
            best_pair = int(
                np.nanargmax(
                    np.where(
                        np.isfinite(two_absorber_samples),
                        two_absorber_samples,
                        -np.inf,
                    )
                )
            )
            partner = int(partners[best_pair])
            candidates = [
                AbsorberCandidate(
                    grid_z_abs=float(z_samples[index]),
                    grid_log_nhi=float(np.log10(grid.nhi_samples[index])),
                    model=2,
                )
                for index in (best_pair, partner)
            ]

        return Result(
            targetid=targetid,
            z_qso=prepared.z_qso,
            status="completed",
            mean_flux=mean_flux,
            ladder=ladder,
            absorber_candidates=candidates,
            screening_score=float(log_z_one - log_z_null) if screened else float("nan"),
            screening_n_evaluated=n_evaluated if screened else 0,
            screening_log_evidence_null=log_z_null if screened else float("nan"),
            screening_log_evidence_absorber=(log_z_one if screened else float("nan")),
            log_evidence_null=log_z_null,
            log_evidence_absorber=log_z_one,
            p_absorber=(ladder.p_absorber if ladder is not None else p_absorber),
            p_null=(ladder.model_posteriors[0] if ladder is not None else p_null),
            legacy_two_model_p_absorber=p_absorber,
            logp_absorber=logp_absorber,
            logp_null=logp_null,
            evidence_mode=mode,
            n_evaluated=n_evaluated,
            n_usable_pixels=prepared.n_pixels,
            quality_fraction=quality_fraction,
            provenance=self._provenance(mode, n_evaluated, mean_flux=mean_flux),
        )


def screening_score(result: Result) -> float:
    """The stored FILTER screening statistic, or NaN if no screening ran.

    Reads :attr:`Result.screening_score`. It used to *derive* the value from the
    result's current evidence fields, which broke in two ways the two-stage
    workflow would have hit immediately: a refinement stage
    rewrites those fields and the mode, so the derivation returned NaN for a
    spectrum that certainly had been screened; and a screened non-detection has
    no absorber row to carry it.

    **Definition.** The log Bayes factor -- one-absorber over null -- computed
    from the FILTER prefix estimate. It is a *ranking* statistic: larger means
    the one-absorber model fits better on the samples that were evaluated. It is
    deliberately **not a probability and not an evidence**, which is why the
    column is named ``GPDLF_SCREENING_SCORE`` rather than anything resembling
    ``P_`` or ``LOGZ_``.

    **Why it is not redundant with the row's log Bayes factor.** The two are
    currently equal because FILTER evidence columns contain the prefix values.
    A future two-stage workflow can refine candidates near a decision boundary
    on the full grid while retaining this value from the screening stage. The
    catalog can then record both stages.

    **NaN in full-grid mode**, because there was no screening stage. A NaN here
    means "not screened", never "screened and scored zero".

    Populated only for completed results; a failed one has no evidences to
    combine.
    """
    return float(result.screening_score)


#: Provenance fields that define *the run*, not the spectrum. Every result in
#: one catalogue must agree on all of them: the run-level RUNINFO record has one
#: slot each, so combining results that disagree would produce an invalid run
#: description.
#:
#: ``evidence_mode`` is deliberately absent -- it is the one field with a
#: schema-supported mixed representation, see :data:`MIXED_RUN_FIELDS`.
RUN_DEFINING_PROVENANCE: tuple[str, ...] = (
    "gpdlf_version",
    "config_digest",
    "preset",
    "base_preset",
    "num_samples",
    "max_absorbers",
    "sample_grid",
    "model",
    "prior",
    "lsf_kernel",
    "quality_policy",
    "seed",
    "backend_backend",
    "backend_faddeeva_source",
    "compatibility_profile",
)

#: Fields a run may legitimately mix, and the label the run record carries when
#: it does. Per-row labels stay authoritative.
MIXED_RUN_FIELDS: Mapping[str, str] = MappingProxyType({"evidence_mode": "mixed"})


def _check_run_provenance(results: Sequence[Result]) -> Mapping[str, object]:
    """The one run-level provenance record, or an error naming the disagreement.

    Silently taking the first result's metadata would let a catalogue claim a
    model, prior, grid or backend that produced only some of its rows.
    """
    #: A field that is MISSING from one result and present in another is a
    #: disagreement, not something to skip: the whole failure mode is a result
    #: produced by an older or different code path that never recorded it.
    _ABSENT = "<absent>"

    seen: dict[str, set] = {}
    for result in results:
        if not result.provenance:
            continue
        for field_name in RUN_DEFINING_PROVENANCE:
            seen.setdefault(field_name, set()).add(
                result.provenance.get(field_name, _ABSENT)
            )

    disagreements = {k: sorted(map(repr, v)) for k, v in seen.items() if len(v) > 1}
    if disagreements:
        detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in disagreements.items())
        raise ValueError(
            "results in one catalogue disagree on run-defining provenance, so a "
            "single run record cannot describe them: "
            f"{detail}. Write them as separate catalogues, or re-run them under "
            "one configuration. (Only evidence_mode has a mixed representation.)"
        )

    for result in results:
        if result.provenance:
            return result.provenance
    return {}


def results_to_catalogue(
    results: Sequence[Result], *, detection_threshold: float
) -> Catalogue:
    """Turn results into the catalogue model the FITS writers consume.

    ``detection_threshold`` is required and has no default. It decides which
    completed results contribute an absorber row, so it determines what the
    catalog contains. This is a policy
    decision that belongs to the caller, not to a library default. It is recorded
    in the run record as ``GPDLF_DETECTION_THRESHOLD`` so a consumer can recover
    the selection that produced the file.

    All results must agree on :data:`RUN_DEFINING_PROVENANCE`; see
    :func:`_check_run_provenance`.
    """
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError(
            "detection_threshold must be a probability in [0, 1], got "
            f"{detection_threshold}"
        )

    # An M2 result writes as two ordinary rows sharing a TARGETID -- the flat
    # form the reference itself uses. The earlier
    # refusal rested on a misreading: P_DLA looked like a two-model quantity
    # being fed a three-model number, but the reference defines it as the sum
    # over every absorber model that was searched. What was actually missing was
    # any record of the search depth, which GPDLF_MAX_DLAS now carries.
    provenance = _check_run_provenance(results)

    absorbers: list[AbsorberRow] = []
    spectra: list[SpectrumRow] = []
    models: list[ModelRow] = []

    for result in results:
        detected = result.detected(detection_threshold) and bool(
            result.absorber_candidates
        )
        rows = result.absorber_candidates if detected else ()

        spectra.append(
            SpectrumRow(
                targetid=result.targetid,
                z_qso=result.z_qso,
                status=result.status,
                reason=result.reason,
                n_absorbers=len(rows),
                p_absorber=result.p_absorber,
                log_evidence_null=result.log_evidence_null,
                log_evidence_absorber=result.log_evidence_absorber,
                evidence_mode=result.evidence_mode,
                quality_fraction=result.quality_fraction,
                n_usable_pixels=result.n_usable_pixels,
                n_evaluated=result.n_evaluated,
                screening_score=result.screening_score,
                screening_n_evaluated=result.screening_n_evaluated,
            )
        )
        # The model ladder, recorded faithfully rather than flattened away.
        # One row per model: its evidence, prior, posterior, whether it was
        # evaluated at all, and whether it won. This does NOT reach FITS -- it
        # is the structured JSON output's payload.
        if result.ladder is not None:
            ladder_rows = result.ladder
            model_posteriors = ladder_rows.model_posteriors
            selected = ladder_rows.selected_model
            for model_index in range(len(ladder_rows.model_labels)):
                evidence = ladder_rows.log_evidences[model_index]
                evaluated = bool(np.isfinite(evidence))
                models.append(
                    ModelRow(
                        targetid=result.targetid,
                        model_index=model_index,
                        log_evidence=evidence,
                        log_prior=ladder_rows.log_priors[model_index],
                        posterior=model_posteriors[model_index],
                        evaluated=evaluated,
                        selected=evaluated and model_index == selected,
                    )
                )

        # The five legacy probability fields, each with the scope used by the
        # reference writer (traced from dlasearch.py:602-628 and
        # run_bayes_select.py:171-172):
        #
        #   P_DLA     spectrum   sum over every absorber model -- P(>=1)
        #   P_NULL    spectrum   1 - P_DLA, which is how the reference forms it
        #   LOGP_NULL spectrum   the null model's unnormalised joint
        #   LOGP_DLA  per row    joint of the (n+1)-absorber model
        #   MODEL_P   per row    posterior of the (n+1)-absorber model
        #
        # The last two were previously written as spectrum-level values, so on a
        # two-absorber spectrum both rows carried the one-absorber numbers and
        # MODEL_P held P(M1)+P(M2) -- a sum, in a field whose meaning is one
        # model's posterior.
        p_dla = result.p_absorber
        if result.ladder is None:
            p_null = result.p_null
            logp_null = result.logp_null
            # A null-versus-one run has exactly one absorber model, so the
            # spectrum-level values ARE the row values.
            per_row = [(result.logp_absorber, result.p_absorber)] * len(rows)
        else:
            joint = result.ladder.log_joint
            rung_posteriors = result.ladder.model_posteriors
            # 1 - p_dla, which is how the reference forms P_NULL. It differs
            # from model_posteriors[0] whenever the ladder has a rung that is
            # neither the null model nor an absorber model.
            p_null = 1.0 - p_dla
            logp_null = joint[0]
            if len(rows) >= len(joint):
                # The selection IS a rung index, so this cannot happen; but a
                # silent NaN here would be a wrong number in a legacy column
                # rather than a missing one.
                raise MultiAbsorberNotRepresentable(
                    f"spectrum {result.targetid} reports {len(rows)} absorbers "
                    f"but its ladder has {len(joint)} models, so LOGP_DLA and "
                    "MODEL_P have no value for the last row"
                )
            per_row = [(joint[n + 1], rung_posteriors[n + 1]) for n in range(len(rows))]

        for index, candidate in enumerate(rows):
            logp_dla, model_p = per_row[index]
            absorbers.append(
                AbsorberRow(
                    targetid=result.targetid,
                    dlaid=reference_dlaid(result.targetid, index),
                    z_qso=result.z_qso,
                    z_dla=candidate.grid_z_abs,
                    nhi=candidate.grid_log_nhi,
                    z_dla_err=candidate.z_abs_err,
                    nhi_err=candidate.log_nhi_err,
                    p_dla=p_dla,
                    p_null=p_null,
                    logp_dla=logp_dla,
                    logp_null=logp_null,
                    model_p=model_p,
                    log_evidence_absorber=result.log_evidence_absorber,
                    log_evidence_null=result.log_evidence_null,
                    evidence_mode=result.evidence_mode,
                    screening_score=screening_score(result),
                    # The candidate's OWN membership, carried rather than
                    # re-derived. Both members of a selected M2 pair belong to
                    # M2; only their row index distinguishes them.
                    model_index=candidate.model,
                )
            )

    run: dict[str, object] = {}
    if provenance:
        run = {
            "GPDLF_VERSION": provenance.get("gpdlf_version", ""),
            # Effective identity, starting point, and exact configuration --
            # the three things a reader needs to answer "what produced this".
            "GPDLF_PRESET": provenance.get("preset", ""),
            "GPDLF_BASE_PRESET": provenance.get("base_preset", ""),
            "GPDLF_CONFIG_DIGEST": provenance.get("config_digest", ""),
            "GPDLF_EVIDENCE_MODE": provenance.get("evidence_mode", ""),
            "GPDLF_NUM_SAMPLES": provenance.get("num_samples", 0),
            "GPDLF_N_EVALUATED_CONFIGURED": provenance.get("n_evaluated", 0),
            "GPDLF_MAX_DLAS": int(provenance.get("max_absorbers", 1)),
            "GPDLF_SAMPLE_GRID": provenance.get("sample_grid", ""),
            "GPDLF_MODEL": provenance.get("model", ""),
            "GPDLF_PRIOR": provenance.get("prior", ""),
            "GPDLF_LSF_KERNEL": provenance.get("lsf_kernel", ""),
            "GPDLF_QUALITY_POLICY": provenance.get("quality_policy", ""),
            "GPDLF_VOIGT_BACKEND": provenance.get("backend_backend", ""),
            "GPDLF_FADDEEVA_SOURCE": provenance.get("backend_faddeeva_source", ""),
            "GPDLF_COMPAT_PROFILE": provenance.get("compatibility_profile", ""),
            # -1, not 0. `seed or 0` mapped an explicitly stochastic run
            # (seed=None) onto the deterministic seed 0, which is a different
            # run and would have been unrecoverable from the file.
            "GPDLF_SEED": (
                -1 if provenance.get("seed") is None else int(provenance["seed"])
            ),
            "GPDLF_RNG_MODE": (
                "stochastic" if provenance.get("seed") is None else "deterministic"
            ),
            "GPDLF_RNG_ALGORITHM": provenance.get("rng_algorithm", ""),
            "GPDLF_EXPERIMENTAL": provenance.get("experimental", ""),
            # The selection that produced this file, so it can be recovered.
            "GPDLF_DETECTION_THRESHOLD": float(detection_threshold),
        }

    # A run whose spectra were inferred in different modes cannot carry one
    # run-level mode; the per-row labels remain authoritative either way. This is
    # the single permitted mixture -- see MIXED_RUN_FIELDS.
    modes = {r.evidence_mode for r in results if r.completed}
    if len(modes) > 1 and run:
        run["GPDLF_EVIDENCE_MODE"] = MIXED_RUN_FIELDS["evidence_mode"]
        warnings.warn(
            "this catalogue mixes evidence modes; the run-level "
            "GPDLF_EVIDENCE_MODE is 'mixed' and the per-row "
            "GPDLF_EVIDENCE_MODE column is authoritative",
            UserWarning,
            stacklevel=2,
        )

    return Catalogue(absorbers=absorbers, spectra=spectra, models=models, run=run)
