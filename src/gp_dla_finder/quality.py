"""Named data-quality policies, kept separate from basic spectrum validation.

There are two different questions here: can we run on this spectrum, and should
it enter a selected catalog? This module keeps those questions separate.

*Structural* validation asks whether the input is a usable spectrum: correct
shapes, increasing wavelengths, non-negative inverse variance, and whether
enough of it survives masking to compute anything. That lives in
:mod:`gp_dla_finder.gp.spectrum` and is not negotiable.

*Quality* selection asks whether a well-formed spectrum is suitable for a
catalog. This is a survey decision rather than an inference decision. The
deployed DESI pipeline requires at least 20% of pixels in rest-frame 900–1230 Å
to have non-zero inverse variance. This module provides that named policy.

We enforce three properties:

* the low-level inference API never applies a survey cut on its own. A
  configuration selects a named policy, or none at all;
* a quality rejection is not an inference failure. It has a separate status and
  reason code;
* whether a policy ran, which one, its threshold and window, the measured usable
  fraction and the verdict all go into result provenance. A catalog built with
  a cut and one built without must not be indistinguishable after the fact.

Disabling the policy transfers quality selection to the caller. Structural
validation alone is **not** a scientifically validated catalog-selection rule and
must not be presented as one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "DESI_Y3_REFERENCE",
    "QUALITY_POLICIES",
    "QualityAssessment",
    "QualityPolicy",
    "quality_policy",
]


@dataclass(frozen=True)
class QualityAssessment:
    """What a policy measured on one spectrum."""

    policy: str
    policy_version: str
    passed: bool
    usable_fraction: float
    n_usable: int
    n_in_window: int
    threshold: float
    rest_lambda_min: float
    rest_lambda_max: float

    #: Stable code for a rejection, or ``None`` when the spectrum passed. Distinct
    #: from :class:`~gp_dla_finder.gp.spectrum.InsufficientData` reasons, which
    #: describe an inability to compute rather than a decision not to.
    @property
    def reason(self) -> str | None:
        return None if self.passed else "quality_policy_rejected"

    def provenance(self) -> Mapping[str, object]:
        """Flat, JSON-friendly record for a result's provenance block."""
        return MappingProxyType(
            {
                "quality_policy": self.policy,
                "quality_policy_version": self.policy_version,
                "quality_passed": self.passed,
                "quality_usable_fraction": self.usable_fraction,
                "quality_n_usable": self.n_usable,
                "quality_n_in_window": self.n_in_window,
                "quality_threshold": self.threshold,
                "quality_rest_lambda_min": self.rest_lambda_min,
                "quality_rest_lambda_max": self.rest_lambda_max,
            }
        )


@dataclass(frozen=True)
class QualityPolicy:
    """A named, versioned catalogue-selection rule."""

    name: str
    #: Bumped whenever the *meaning* changes. A policy must never decide
    #: differently under a fixed name and version.
    version: str
    summary: str
    #: Minimum fraction of pixels in the window that must be usable.
    min_usable_fraction: float
    #: Rest-frame window the fraction is measured over, angstroms.
    rest_lambda_min: float
    rest_lambda_max: float

    def assess(self, spectrum) -> QualityAssessment:
        """Measure ``spectrum`` against this policy.

        Measures rather than decides: the caller reads
        :attr:`QualityAssessment.passed`. A policy never raises on a well-formed
        spectrum, because "this does not belong in the catalogue" is a result,
        not an error.
        """
        rest = np.asarray(spectrum.wavelength, dtype=np.float64) / (
            1.0 + spectrum.z_qso
        )
        in_window = (rest > self.rest_lambda_min) & (rest < self.rest_lambda_max)
        n_in_window = int(np.count_nonzero(in_window))

        if n_in_window == 0:
            # No coverage at all is a rejection, not a division by zero.
            return QualityAssessment(
                policy=self.name,
                policy_version=self.version,
                passed=False,
                usable_fraction=0.0,
                n_usable=0,
                n_in_window=0,
                threshold=self.min_usable_fraction,
                rest_lambda_min=self.rest_lambda_min,
                rest_lambda_max=self.rest_lambda_max,
            )

        mask = np.asarray(spectrum.mask, dtype=bool)
        n_usable = int(np.count_nonzero(in_window & ~mask))
        fraction = n_usable / n_in_window

        return QualityAssessment(
            policy=self.name,
            policy_version=self.version,
            passed=bool(fraction >= self.min_usable_fraction),
            usable_fraction=float(fraction),
            n_usable=n_usable,
            n_in_window=n_in_window,
            threshold=self.min_usable_fraction,
            rest_lambda_min=self.rest_lambda_min,
            rest_lambda_max=self.rest_lambda_max,
        )


#: The deployed DESI requirement, transcribed from the reference's ``dlasearch``:
#: at least 20 % of the pixels strictly inside rest-frame 900-1230 A must have
#: non-zero inverse variance.
#:
#: Two things about it are worth knowing before relying on it. The window is
#: **not** the GP search window -- the reference itself carries a TODO saying so
#: -- and the comparison is strict inequality on both edges, matching
#: ``np.ma.masked_inside``. Both are reproduced rather than tidied, because the
#: point of this policy is to select the same spectra the published catalogue
#: selected.
DESI_Y3_REFERENCE = QualityPolicy(
    name="desi-y3-reference",
    version="1",
    summary=(
        "At least 20% of pixels in rest-frame 900-1230 A usable. Reproduces the "
        "selection applied when the deployed DESI catalogues were produced."
    ),
    min_usable_fraction=0.2,
    rest_lambda_min=900.0,
    rest_lambda_max=1230.0,
)

_POLICIES: dict[str, QualityPolicy] = {DESI_Y3_REFERENCE.name: DESI_Y3_REFERENCE}

#: Read-only registry. No default and no nearest match: a silently substituted
#: selection rule changes which spectra are in a catalogue.
QUALITY_POLICIES: Mapping[str, QualityPolicy] = MappingProxyType(_POLICIES)


def quality_policy(name: str) -> QualityPolicy:
    """Return a named quality policy.

    Raises
    ------
    KeyError
        If ``name`` is unknown.
    """
    try:
        return _POLICIES[name]
    except KeyError:
        known = ", ".join(sorted(_POLICIES))
        raise KeyError(
            f"unknown quality policy {name!r}; known policies: {known}"
        ) from None
