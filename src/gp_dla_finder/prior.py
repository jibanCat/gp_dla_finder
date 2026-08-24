"""Absorber-existence prior: P(k absorbers | z_qso).

The prior is empirical. It comes from counting, among catalogued sightlines below
a quasar's redshift, what fraction host a damped absorber:

.. math::

    P(\\ge 1 \\text{ absorber} \\mid z_\\mathrm{QSO}) = M / N

with ``M`` the number of catalogued absorbers and ``N`` the number of quasars with
``z < z_QSO + delta``. Multi-absorber priors follow by assuming independence,
``P(>= k) = (M/N)^k``, and differencing.

Representation
--------------
The reference pipeline recomputes this from ~115 MB of SDSS catalogues on every
run, but touches them through a single counting call. Since that call is a
monotone step function of one scalar, it is stored here as a sorted redshift array
plus a cumulative absorber count -- an *exact* representation, about 0.1 MB, with
no interpolation and no tolerance. ``tools/build_prior_table.py`` builds it and
proves the equivalence at every breakpoint and on a dense grid.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._immutable import deep_freeze, frozen_array

__all__ = ["AbsorberPrior", "DEFAULT_PRIOR", "available_priors", "load_prior"]

#: Prior built from the SDSS DR9Q concordance catalogue, as used by every
#: published version of this method and by the deployed production runs.
DEFAULT_PRIOR = "dr9q_concordance"

_DATA_PACKAGE = "gp_dla_finder.data.priors"


@dataclass(frozen=True)
class AbsorberPrior:
    """An empirical absorber-existence prior as an exact step table.

    Attributes
    ----------
    name
        Asset name, or ``"<external>"``.
    z_qsos
        Sorted quasar redshifts of the selected catalogue sample, shape ``(N,)``.
    cumulative_absorbers
        ``cumulative_absorbers[i]`` is the number of sightlines among
        ``z_qsos[:i+1]`` that host an absorber.
    provenance
        Read-only record of sources, checksums, selection, and the equivalence
        proof.
    """

    name: str
    z_qsos: np.ndarray
    cumulative_absorbers: np.ndarray
    provenance: Any = field(default_factory=lambda: MappingProxyType({}), kw_only=True)

    def __post_init__(self) -> None:
        # Counts are widened to int64 on load; the on-disk asset stores int32,
        # which has ample headroom at catalog scale.
        z = frozen_array(self.z_qsos, dtype=np.float64)
        counts = frozen_array(self.cumulative_absorbers, dtype=np.int64)

        if z.ndim != 1 or counts.shape != z.shape:
            raise ValueError(
                f"z_qsos {z.shape} and cumulative_absorbers {counts.shape} "
                "must be 1-D and the same length"
            )
        if z.size == 0:
            raise ValueError("prior table is empty")
        if not np.all(np.isfinite(z)):
            raise ValueError("z_qsos must all be finite")
        if not np.all(np.diff(z) >= 0):
            raise ValueError("z_qsos must be sorted ascending")
        if counts[0] < 0:
            raise ValueError("cumulative_absorbers must be non-negative")
        if not np.all(np.diff(counts) >= 0):
            raise ValueError("cumulative_absorbers must be non-decreasing")
        # A prefix of i+1 sightlines cannot contain more than i+1 absorbers.
        if not np.all(counts <= np.arange(1, counts.size + 1)):
            raise ValueError(
                "cumulative_absorbers exceeds the number of sightlines represented"
            )

        object.__setattr__(self, "z_qsos", z)
        object.__setattr__(self, "cumulative_absorbers", counts)
        object.__setattr__(self, "provenance", deep_freeze(self.provenance))

    @property
    def n_sightlines(self) -> int:
        return int(self.z_qsos.size)

    @property
    def z_qso_range(self) -> tuple[float, float]:
        return float(self.z_qsos[0]), float(self.z_qsos[-1])

    def supports(self, z_qso: float) -> bool:
        """Whether ``z_qso`` lies within the catalogue's redshift support."""
        lo, hi = self.z_qso_range
        return lo <= float(z_qso) <= hi

    def counts(self, z_qso: float, z_increase: float) -> tuple[int, int]:
        """Number of catalogued absorbers and quasars below ``z_qso + z_increase``.

        This is the low-level reference-parity operation. Below the catalogue
        floor it reproduces the reference's historical clamp, which the
        reference source describes as temporary, and emits a warning. Above the
        top of the catalog, the
        counts saturate, which is equally an extrapolation.

        The public finder rejects unsupported redshifts by default; use
        :meth:`supports` to test before calling.
        """
        floor = float(self.z_qsos[0])
        if float(z_qso) < floor:
            warnings.warn(
                f"z_qso={z_qso} is below this prior's catalogue floor "
                f"({floor:.4f}); the reference implementation's historical clamp "
                "is being applied, so the prior returned is that of a quasar at "
                "the floor. This is an extrapolation, not a measurement.",
                UserWarning,
                stacklevel=2,
            )
        elif float(z_qso) > float(self.z_qsos[-1]):
            warnings.warn(
                f"z_qso={z_qso} is above this prior's highest catalogued "
                f"sightline ({self.z_qsos[-1]:.4f}); the counts saturate, so the "
                "prior is an extrapolation, not a measurement.",
                UserWarning,
                stacklevel=2,
            )
        z = max(float(z_qso), floor)
        idx = int(np.searchsorted(self.z_qsos, z + z_increase, side="left"))
        if idx == 0:
            return 0, 0
        return int(self.cumulative_absorbers[idx - 1]), idx

    def absorber_fraction(self, z_qso: float, z_increase: float) -> float:
        """``M / N``: P(at least one absorber | z_qso)."""
        n_absorbers, n_quasars = self.counts(z_qso, z_increase)
        if n_quasars == 0:
            raise ValueError(
                f"no catalogued sightlines below z={z_qso} + {z_increase}; "
                f"this prior covers z_qso >= {self.z_qsos[0]:.4f}"
            )
        return n_absorbers / n_quasars

    def log_prior_no_absorber(self, z_qso: float, z_increase: float) -> float:
        """``log P(no absorber | z_qso)``."""
        return float(np.log(1.0 - self.absorber_fraction(z_qso, z_increase)))

    def log_priors(
        self, z_qso: float, max_absorbers: int, z_increase: float
    ) -> np.ndarray:
        """``log P(exactly k absorbers | z_qso)`` for ``k = 1 .. max_absorbers``.

        ``P(>= k) = (M/N)^k`` under independence, differenced to give exactly-k.
        The last entry is left as ``P(>= max_absorbers)``, matching the reference:
        the top model absorbs the tail.
        """
        if max_absorbers < 1:
            raise ValueError(f"max_absorbers must be >= 1, got {max_absorbers}")
        fraction = self.absorber_fraction(z_qso, z_increase)
        p = fraction ** np.arange(1, max_absorbers + 1)
        for i in range(max_absorbers - 1):
            p[i] = p[i] - p[i + 1]
        with np.errstate(divide="ignore"):
            return np.log(p)

    def __repr__(self) -> str:  # pragma: no cover - display only
        lo, hi = self.z_qso_range
        return (
            f"AbsorberPrior(name={self.name!r}, n_sightlines={self.n_sightlines}, "
            f"z_qso=[{lo:.3f}, {hi:.3f}])"
        )


def available_priors() -> tuple[str, ...]:
    """Names of prior tables bundled with this installation."""
    root = resources.files(_DATA_PACKAGE)
    return tuple(
        sorted(
            p.name[: -len(".npz")] for p in root.iterdir() if p.name.endswith(".npz")
        )
    )


def prior_provenance(name: str = DEFAULT_PRIOR):
    """Provenance record for a bundled prior: sources, checksums, equivalence proof."""
    handle = resources.files(_DATA_PACKAGE) / f"{name}.json"
    if not handle.is_file():
        raise ValueError(_unknown_prior_message(name))
    return deep_freeze(json.loads(handle.read_text()))


def _unknown_prior_message(name: str) -> str:
    return (
        f"unknown prior {name!r}; bundled priors: {', '.join(available_priors())}. "
        "To load one from disk, pass path=... instead."
    )


def load_prior(
    name: str = DEFAULT_PRIOR, *, path: str | Path | None = None
) -> AbsorberPrior:
    """Load an absorber-existence prior table."""
    if path is not None:
        with np.load(Path(path)) as data:
            return AbsorberPrior(
                name="<external>",
                z_qsos=data["z_qsos"],
                cumulative_absorbers=data["cumulative_absorbers"],
            )

    handle = resources.files(_DATA_PACKAGE) / f"{name}.npz"
    if not handle.is_file():
        raise ValueError(_unknown_prior_message(name))
    with resources.as_file(handle) as npz_path, np.load(npz_path) as data:
        return AbsorberPrior(
            name=name,
            z_qsos=data["z_qsos"],
            cumulative_absorbers=data["cumulative_absorbers"],
            provenance=prior_provenance(name),
        )
