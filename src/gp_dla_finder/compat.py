"""Named compatibility profiles for reproducing the reference arithmetic.

Reproducing the reference implementation *bitwise* required reproducing two
pieces of arithmetic that are mathematically no-ops:

``rest_frame_round_trip``
    The reference derives observed wavelengths as ``wave / (1 + z) * (1 + z)``
    rather than reusing the input array. In floating point that round trip is not
    the identity: it moves wavelengths by ~1e-13 A, which propagates into the
    effective optical depth and then into every evidence.

``log_norm_round_trip``
    Every per-sample log-likelihood carries ``- log(N)`` and the quasi-Monte-Carlo
    estimator adds ``+ log(N)`` back. The pair cancels analytically. It does not
    cancel in floating point, because the subtraction happens before a
    log-mean-exp and the addition after it.

Neither operation changes the analytic model. We retain them because this port
must reproduce a pinned reference bitwise; comparisons at the $10^{-12}$ level
identified both effects.

We isolate these operations in a named, versioned compatibility profile and
record the profile in result provenance. Two consequences follow:

* the package reproduces the arithmetic of a specific commit. If the
  reference is ever corrected, :data:`REFERENCE_D5B306E6` must not be edited --
  add a new profile, so old results stay reproducible;
* a clean-arithmetic mode exists (:data:`CLEAN`) for comparison. Selecting it
  changes numerical results, and the provenance records that choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "CLEAN",
    "COMPATIBILITY_PROFILES",
    "DEFAULT_COMPATIBILITY",
    "CompatibilityProfile",
    "compatibility_profile",
]


@dataclass(frozen=True)
class CompatibilityProfile:
    """A named set of reference-fidelity arithmetic choices."""

    #: Stable identifier, recorded in result provenance.
    name: str
    #: Bumped whenever the *meaning* of the profile changes. A profile's numbers
    #: must never change under a fixed name and version.
    version: str
    summary: str
    #: Reproduce ``observed = rest * (1 + z)`` instead of reusing the input grid.
    rest_frame_round_trip: bool
    #: Carry ``- log(N)`` per sample and add ``+ log(N)`` back in the estimator.
    log_norm_round_trip: bool
    #: The reference this profile was measured against, if any.
    reference_repo: str | None = None
    reference_commit: str | None = None

    def provenance(self) -> Mapping[str, object]:
        """Flat, JSON-friendly record for a result's provenance block."""
        return MappingProxyType(
            {
                "compatibility_profile": self.name,
                "compatibility_version": self.version,
                "rest_frame_round_trip": self.rest_frame_round_trip,
                "log_norm_round_trip": self.log_norm_round_trip,
                "compatibility_reference_repo": self.reference_repo,
                "compatibility_reference_commit": self.reference_commit,
            }
        )


#: Bit-compatible with the pinned public reference. This is the default, because
#: reproducing published catalogues is the package's primary obligation.
REFERENCE_D5B306E6 = CompatibilityProfile(
    name="reference-d5b306e6",
    version="1",
    summary=(
        "Bitwise agreement with desi_gpy_dla_detection at commit d5b306e6, "
        "including two floating-point no-ops that the reference performs."
    ),
    rest_frame_round_trip=True,
    log_norm_round_trip=True,
    reference_repo="jibanCat/desi_gpy_dla_detection",
    reference_commit="d5b306e6e2c8d89cdb38a6201b690557f2798f28",
)

#: The same model without the reference's redundant arithmetic. Provided so the
#: cost of bug-compatibility can be *measured*. Results under this profile will
#: not match published catalogues bitwise and are not a parity claim.
CLEAN = CompatibilityProfile(
    name="clean",
    version="1",
    summary=(
        "Drops the reference's two floating-point no-ops. Mathematically the "
        "same model; not bitwise comparable with published catalogues."
    ),
    rest_frame_round_trip=False,
    log_norm_round_trip=False,
)

_PROFILES: dict[str, CompatibilityProfile] = {
    REFERENCE_D5B306E6.name: REFERENCE_D5B306E6,
    CLEAN.name: CLEAN,
}

#: Read-only registry. There is no default lookup and no nearest match.
COMPATIBILITY_PROFILES: Mapping[str, CompatibilityProfile] = MappingProxyType(_PROFILES)

#: What a named production preset selects unless it says otherwise.
DEFAULT_COMPATIBILITY: str = REFERENCE_D5B306E6.name


def compatibility_profile(name: str) -> CompatibilityProfile:
    """Return a named compatibility profile.

    Raises
    ------
    KeyError
        If ``name`` is unknown. Substituting a profile would silently change the
        arithmetic, so there is no fallback.
    """
    try:
        return _PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(_PROFILES))
        raise KeyError(
            f"unknown compatibility profile {name!r}; known profiles: {known}"
        ) from None
