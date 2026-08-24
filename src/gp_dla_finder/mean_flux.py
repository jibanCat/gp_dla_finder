"""Per-spectrum empirical-Bayes fit of the mean-flux optical depth.

The production pipeline does not use the prior ``tau_0`` for every spectrum. It
fits one per spectrum: build the null GP at each of a small grid of candidate
``tau_0`` values, evaluate a log-likelihood, and keep the best. The chosen value
then replaces ``prev_tau_0`` for that spectrum's inference.

This is a port of the algorithm, re-expressed against this package's
:class:`~gp_dla_finder.gp.spectrum.PreparedSpectrum` and
:class:`~gp_dla_finder.model.GPModel`, not a transliteration of the reference's
class hierarchy or its ``learned_file`` path handling. The source is recorded in
:data:`REFERENCE_SOURCE` and the mapping in :data:`ALGORITHMIC_MAPPING`.

What production actually runs
-----------------------------

The settings below come from the executed Matterhorn/LOA submission scripts.
They differ from both the library defaults and the example in the reference
docstring.

**This is only the mean-flux subconfiguration.** It does not make the package
reproduce the complete Matterhorn operating point. Matterhorn also used FILTER,
a 50,000-sample setting, and a different multi-absorber workflow. Those choices
remain separate from the v0.1 full-grid workflow and its 100,000-sample
validation reference.

============================  ==========================================
``enable_tau_eb``             ``1`` -- on
objective                     ``"null"``
``tau_factors``               ``(0.5, 1, 1.5, 2, 3, 4, 5, 6)``
seed ``tau_0`` / ``beta``     ``0.00246`` / ``3.62`` (Turner+2024)
HCD mask                      **off**
============================  ==========================================

The HCD mask needs a separate caveat. The reference recipe originally masked
suspected high-column pixels while choosing ``tau_0``, and reported that this
closed ~80% of the column-density bias. That result did not survive a broader
test. It came from a picker that selected high-signal, single-absorber
sightlines; on 90
random targets and on 5000 random mock spectra the mask *over-corrects* --
median bias moved from +0.135 dex to **-0.131** dex with the mask, against
**+0.026** dex without it. Production has run with the mask off ever since, and
this package does not implement it at all.

The mean-flux fit itself still improves the measured median bias from +0.135 to
+0.026 dex without the mask. That is why the fit is retained while the mask is
not.

Scope
-----

Only the ``"null"`` objective is implemented, which is what production runs. The
absorber-aware objective is refused explicitly -- see :class:`ObjectiveNotSupported`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .config import Config
from .errors import NumericalError
from .gp.evidence import assemble_model, null_log_evidence
from .gp.spectrum import PreparedSpectrum
from .model import GPModel

__all__ = [
    "ALGORITHMIC_MAPPING",
    "HCDMaskNotSupported",
    "MeanFluxFit",
    "ObjectiveNotSupported",
    "REFERENCE_SOURCE",
    "SUPPORTED_OBJECTIVES",
    "fit_tau_0",
]

#: The reference implementation this was ported from, pinned so parity claims
#: name a specific commit rather than "the reference".
REFERENCE_SOURCE = MappingProxyType(
    {
        "repository": "desi_gpy_dla_detection",
        "module": "gpy_dla_detection/tau_eb.py",
        "function": "fit_tau_eb",
        "commit": "9aa20dc",
        "branch": "desi_y3",
    }
)

#: How the reference's objects map onto this package's. Recorded because a port
#: that cannot say what it mapped is not reviewable.
ALGORITHMIC_MAPPING = MappingProxyType(
    {
        "NullGPMAT + learned_file": "GPModel + assemble_model()",
        "set_data(...) then log_model_evidence()": (
            "null_log_evidence(prepared, assembled)"
        ),
        "pixel_mask": "PreparedSpectrum (masking already applied)",
        "prev_tau_0_seed * tau_factors[j]": "candidate tau_0, identical arithmetic",
        "objective='null'": "objective='null'",
        "objective='dla'": "objective='absorber' -- NOT IMPLEMENTED, refused",
    }
)

#: Objectives this package implements. The reference also offers an
#: absorber-aware objective it calls ``"dla"``; this package's public vocabulary
#: is ``"absorber"``, and the two string interfaces are **not** interchangeable.
#: Translation happens only at the parity boundary, never silently.
SUPPORTED_OBJECTIVES: tuple[str, ...] = ("null",)

#: The reference's name for the absorber-aware objective, kept so a parity test
#: or a reader coming from the reference can find it. Not accepted as input.
_REFERENCE_OBJECTIVE_NAMES = MappingProxyType({"absorber": "dla", "null": "null"})


class ObjectiveNotSupported(NotImplementedError):
    """A requested mean-flux objective exists in the reference but not here."""


class HCDMaskNotSupported(NotImplementedError):
    """The retracted HCD-masked variant was requested.

    ``Config`` keeps ``tau_eb_apply_hcd_mask`` so a configuration recorded
    against the reference still round-trips, but this package does not implement
    the mask -- it was withdrawn after over-correcting at scale. Ignoring the
    request would hand back an UNMASKED fit labelled as masked, which is worse
    than refusing.
    """


@dataclass(frozen=True)
class MeanFluxFit:
    """The chosen optical depth, and enough to audit the choice."""

    #: The fitted value, ``seed_tau_0 * factor``. Use this in place of
    #: ``config.prev_tau_0`` for this spectrum.
    tau_0: float
    #: Which grid point won.
    factor: float
    #: The seed the grid multiplies.
    seed_tau_0: float
    #: The grid that was scanned, in order.
    factors: tuple[float, ...]
    #: Log evidence at each grid point, aligned with ``factors``. Retained so a
    #: reader can see whether the maximum was decisive or a coin-flip.
    log_evidence: tuple[float, ...]
    objective: str = "null"

    @property
    def at_grid_edge(self) -> bool:
        """Whether the best value sits at an end of the grid.

        True means the optimum may lie outside the scanned range, so the fit is
        a bound rather than a maximum. The reference extended its grid to 6x
        after exactly this showed up.
        """
        return self.factor in (self.factors[0], self.factors[-1])

    @property
    def margin(self) -> float:
        """Log-evidence gap between the winner and the runner-up.

        Small means the grid barely distinguished two candidates.
        """
        ordered = sorted(self.log_evidence, reverse=True)
        if len(ordered) < 2:
            return float("inf")
        return float(ordered[0] - ordered[1])

    def provenance(self) -> dict[str, object]:
        """The compact record: the choice and how confident it was.

        Deliberately not the whole scan. The full per-factor vector lives on the
        object and on ``Result.mean_flux``, which is in memory only -- there is
        no full-output writer in this package yet, so nothing persists it to
        disk. The catalogue keeps the selected value and these diagnostics,
        which is what a compact schema should carry; persisting the whole scan
        is open work.
        """
        return {
            "mean_flux_tau_0": self.tau_0,
            "mean_flux_factor": self.factor,
            "mean_flux_seed_tau_0": self.seed_tau_0,
            "mean_flux_objective": self.objective,
            "mean_flux_at_grid_edge": self.at_grid_edge,
            "mean_flux_source_commit": REFERENCE_SOURCE["commit"],
        }


def fit_tau_0(
    prepared: PreparedSpectrum,
    model: GPModel,
    config: Config,
    *,
    objective: str = "null",
) -> MeanFluxFit:
    """Choose ``tau_0`` for one spectrum by scanning the configured grid.

    Builds the null model at ``config.prev_tau_0 * factor`` for each factor in
    ``config.tau_eb_factors``, takes the log evidence at each, and returns the
    argmax. ``beta`` is held at ``config.prev_beta`` throughout -- the reference
    scans one parameter, not two.

    The scan uses the spectrum exactly as prepared. The reference has an option
    to mask suspected high-column pixels while choosing, which is **not**
    implemented here and is off in production: at scale it over-corrects (see
    the module docstring).

    Raises
    ------
    ObjectiveNotSupported
        For any objective other than ``"null"``. It never falls back.
    ValueError
        For an empty or non-positive factor grid.
    """
    if getattr(config, "tau_eb_apply_hcd_mask", False):
        raise HCDMaskNotSupported(
            "tau_eb_apply_hcd_mask=True requests the HCD-masked variant, which "
            "this package does not implement. It was retracted after measurement:"
            " on 90 random targets and 5000 random mock spectra the mask "
            "over-corrects, moving the median column-density bias from +0.135 "
            "dex to -0.131 dex, where the unmasked fit moves it to +0.026 dex. "
            "Production has run with the mask off since. Set "
            "tau_eb_apply_hcd_mask=False."
        )

    if objective not in SUPPORTED_OBJECTIVES:
        reference_name = _REFERENCE_OBJECTIVE_NAMES.get(objective)
        detail = (
            f" The reference calls this objective {reference_name!r}."
            if reference_name
            else ""
        )
        raise ObjectiveNotSupported(
            f"mean-flux objective {objective!r} is not implemented; this package "
            f"supports {SUPPORTED_OBJECTIVES}.{detail} The absorber-aware "
            "objective costs a full grid scan per candidate tau_0 and has not "
            "been validated here, so it is refused rather than silently "
            "replaced by the null objective."
        )

    factors = tuple(float(f) for f in config.tau_eb_factors)
    if not factors:
        raise ValueError("tau_eb_factors is empty; there is nothing to scan")
    if any(f <= 0.0 for f in factors):
        raise ValueError(f"tau factors must be positive, got {factors}")

    seed = float(config.prev_tau_0)
    beta = float(config.prev_beta)

    log_evidence: list[float] = []
    for factor in factors:
        assembled = assemble_model(
            prepared, model, config, tau_0=seed * factor, beta=beta
        )
        log_evidence.append(float(null_log_evidence(prepared, assembled)))

    values = np.asarray(log_evidence, dtype=float)
    if not np.any(np.isfinite(values)):
        raise NumericalError(
            "the mean-flux scan produced no finite log evidence on any grid "
            f"point (factors={factors})"
        )

    # argmax over finite values only, and FIRST on a tie -- np.argmax already
    # returns the first maximum, which matches the reference's np.argmax and
    # makes ties deterministic rather than platform-dependent.
    best = int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))

    return MeanFluxFit(
        tau_0=seed * factors[best],
        factor=factors[best],
        seed_tau_0=seed,
        factors=factors,
        log_evidence=tuple(log_evidence),
        objective=objective,
    )
