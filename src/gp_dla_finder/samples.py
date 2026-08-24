"""Quasi-Monte-Carlo absorber sample grids.

The evidence for a ``k``-absorber model is an integral over absorber parameters,
approximated by a QMC average:

.. math::

    p(D \\mid k) \\approx \\frac{1}{N} \\sum_i p(D \\mid z_i, N_{\\mathrm{HI},i})

A :class:`AbsorberSampleGrid` holds those samples. Column densities are drawn from
the Prochaska et al. (2014) CDDF prior mixed with a uniform component; absorber
redshifts are stored as offsets in [0, 1) and stretched onto each spectrum's own
search window at inference time, so one grid serves every spectrum.

Provenance status
-----------------
Grids carry an explicit identity status. A grid marked ``"regenerated,
production-array identity unverified"`` reproduces the reference *generator*
bitwise. It has not been shown byte-identical to the deployed arrays, whose QMC
state was not recorded reproducibly. You can use such a grid for development
and tutorials, but it does not support a production-equivalence claim.
:attr:`AbsorberSampleGrid.is_verified` exposes that distinction to callers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._immutable import deep_freeze, frozen_array

__all__ = [
    "DEFAULT_SAMPLE_GRID",
    "canonical_array_digest",
    "canonical_file_digest",
    "verify_grid_integrity",
    "REQUIRED_GRID_METADATA",
    "VERIFIED_IDENTITY_STATUSES",
    "AbsorberSampleGrid",
    "available_sample_grids",
    "load_sample_grid",
]

#: Grid matching the deployed production operating point: 50,000 samples over
#: log10 N_HI in [17.2, 22.5]. See the identity caveat above.
DEFAULT_SAMPLE_GRID = "pw14_172_225_50000"

_DATA_PACKAGE = "gp_dla_finder.data.samples"


def canonical_array_digest(array: np.ndarray) -> str:
    """The one definition of an array digest in this project.

    ``float64``, C-contiguous, raw bytes. ``tools/build_sample_grid.py`` imports
    this rather than defining its own, so a grid's recorded hash and the hash
    checked at load time cannot drift into two different conventions.
    """
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).tobytes()
    ).hexdigest()


def canonical_file_digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks. Same definition as the builder's."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Arrays a grid file must contain, and which the sidecar must account for.
_REQUIRED_ARRAYS: tuple[str, ...] = (
    "offset_samples",
    "log_nhi_samples",
    "nhi_samples",
)


def _json_kind(value: object) -> str:
    """What a JSON reader would call this value.

    Reported instead of the Python type name because the reader freezes the
    parsed document -- a JSON array arrives here as a ``tuple`` -- and telling
    someone editing a ``.json`` file that it contains a "tuple" describes this
    package's internals rather than their file.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, (str, bytes)):
        return "a string"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, Sequence):
        return "a list"
    return f"a {type(value).__name__}"


def _as_mapping(value: object) -> Mapping | None:
    """``value`` if it is a mapping, else ``None``.

    Sidecars are JSON written by a tool, but nothing stops one being hand-edited
    into a shape the reader did not expect. Every structural access goes through
    this, so a wrong type becomes a named integrity problem rather than an
    ``AttributeError`` escaping from a load.
    """
    return value if isinstance(value, Mapping) else None


def _as_shape(value: object) -> tuple[int, ...] | None:
    """A declared array shape as a tuple of ints, or ``None`` if it is not one."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def verify_grid_integrity(
    path: Path, provenance: object, arrays: Mapping[str, np.ndarray]
) -> tuple[str, ...]:
    """Check that the sidecar actually describes this file and these arrays.

    Returns the problems found; empty means verified.

    A sidecar carrying fields with the right *names* proves nothing on its own:
    it could have been copied from a different grid, or the arrays could have
    been edited after it was written. So the recorded digests are recomputed and
    compared, along with the declared shapes and the sample count.

    **Every structural assumption is checked rather than assumed.** A sidecar
    that is valid JSON but the wrong shape -- a list at the top level, a string
    where a mapping belongs, a sample count that is not a number -- is a failed
    integrity record, and must read as one. A raw ``TypeError`` escaping from a
    load is not an integrity check; it is a crash that happens to prevent one.
    """
    record = _as_mapping(provenance)
    if record is None:
        return (
            f"the sidecar's top level is {_json_kind(provenance)}, not an "
            "object; it cannot describe a grid",
        )

    problems: list[str] = []

    recorded_file = record.get("sha256")
    if not isinstance(recorded_file, str) or not recorded_file.strip():
        problems.append("sha256: the sidecar records no usable file digest")
    else:
        actual_file = canonical_file_digest(path)
        if actual_file != recorded_file.strip():
            problems.append(
                f"sha256: the sidecar describes a file with digest "
                f"{recorded_file[:16]}..., but {path.name} hashes to "
                f"{actual_file[:16]}...  (a sidecar from a different grid, or "
                "an edited file)"
            )

    recorded_arrays = _as_mapping(record.get("arrays"))
    if recorded_arrays is None:
        problems.append(
            "arrays: the sidecar has no per-array record, or it is not an object"
        )
        recorded_arrays = {}

    for name in _REQUIRED_ARRAYS:
        actual = arrays.get(name)
        if actual is None:
            problems.append(f"arrays.{name}: absent from the file")
            continue

        entry = _as_mapping(recorded_arrays.get(name))
        if entry is None:
            problems.append(
                f"arrays.{name}: the sidecar has no record for it, or the "
                "record is not an object"
            )
            continue

        recorded_digest = entry.get("sha256_float64")
        if not isinstance(recorded_digest, str) or not recorded_digest.strip():
            problems.append(f"arrays.{name}.sha256_float64: missing or not a string")
        elif canonical_array_digest(actual) != recorded_digest.strip():
            problems.append(
                f"arrays.{name}: content does not match the recorded digest"
            )

        # Required, not optional. A sidecar that omits the shape has not
        # described the array, and accepting the omission would let the weakest
        # sidecar past the same gate as a complete one.
        if "shape" not in entry:
            problems.append(f"arrays.{name}.shape: missing")
        else:
            declared_shape = _as_shape(entry["shape"])
            if declared_shape is None:
                problems.append(
                    f"arrays.{name}.shape: {entry['shape']!r} is not a list of integers"
                )
            elif declared_shape != tuple(np.shape(actual)):
                problems.append(
                    f"arrays.{name}.shape: the sidecar says {declared_shape}, "
                    f"the file holds {tuple(np.shape(actual))}"
                )

    qmc = _as_mapping(record.get("qmc"))
    if qmc is None:
        problems.append("qmc: missing from the sidecar, or not an object")
    elif "num_samples" not in qmc:
        problems.append("qmc.num_samples: missing")
    else:
        declared = qmc["num_samples"]
        # bool is an int in Python, and `True` is not a sample count.
        if isinstance(declared, bool) or not isinstance(declared, int):
            problems.append(f"qmc.num_samples: {declared!r} is not an integer")
        else:
            reference_array = arrays.get("log_nhi_samples")
            if reference_array is None:
                problems.append(
                    "qmc.num_samples: cannot be checked, log_nhi_samples is "
                    "absent from the file"
                )
            elif declared != int(np.shape(reference_array)[0]):
                problems.append(
                    f"qmc.num_samples: the sidecar says {declared}, the arrays "
                    f"hold {int(np.shape(reference_array)[0])}"
                )

    return tuple(problems)


#: Provenance a grid must carry before it may be used for inference.
#:
#: ``tools/build_sample_grid.py`` writes all of these into the ``<name>.json``
#: sidecar, so a grid built the supported way is ready as it stands. The list
#: exists for the case where one is not: it names what is missing rather than
#: leaving the caller to guess.
REQUIRED_GRID_METADATA: tuple[str, ...] = (
    "name",
    "sha256",
    "prior.family",
    "prior.support_log_nhi",
    "prior.mixture_weight_pw14",
    "qmc.num_samples",
    "qmc.seed",
)

#: Relative tolerance for ``nhi_samples`` versus ``10 ** log_nhi_samples``.
#:
#: A few float64 ulps. Wide enough to absorb the cross-architecture disagreement
#: in ``libm``'s ``pow`` -- measured at ~1e-16 relative between arm64 and x86-64 --
#: and far too tight to admit a genuinely inconsistent array.
NHI_CONSISTENCY_RTOL: float = 1e-12

#: Identity statuses that count as verified against the deployed production
#: arrays. An exact allow-list, so verification fails closed: anything not
#: listed here -- including a new or misspelled status -- is unverified.
VERIFIED_IDENTITY_STATUSES: frozenset[str] = frozenset(
    {
        "verified: byte-identical to the deployed production arrays",
        "verified: array hashes match the deployed production grid",
    }
)


@dataclass(frozen=True)
class AbsorberSampleGrid:
    """QMC samples of ``(redshift offset, log10 N_HI)``.

    Attributes
    ----------
    name
        Asset name, or ``"<external>"``.
    offset_samples
        Absorber-redshift offsets in [0, 1), shape ``(N,)``. Mapped onto a
        spectrum's search window by :meth:`sample_redshifts`.
    log_nhi_samples
        log10(N_HI / cm^-2) samples, shape ``(N,)``.
    provenance
        Read-only record of the prior, the QMC construction, the generating
        environment, per-array checksums, and the identity status.
    """

    name: str
    offset_samples: np.ndarray
    log_nhi_samples: np.ndarray
    # Keyword-only: these two were positionally adjacent and easy to transpose,
    # which silently passed a provenance mapping as an array.
    nhi_samples: np.ndarray | None = field(default=None, kw_only=True)
    #: Integrity problems found when this grid was loaded, or ``None`` if the
    #: check never ran.
    #:
    #: ``None`` and ``()`` are deliberately different. ``()`` means the sidecar
    #: was checked against the file and the arrays and matched; ``None`` means
    #: nothing was verified, which is what a hand-constructed grid gets. Only
    #: the first is inference-ready -- see :attr:`usable_for_inference`.
    integrity_problems: tuple[str, ...] | None = field(default=None, kw_only=True)
    provenance: Any = field(default_factory=lambda: MappingProxyType({}), kw_only=True)

    def __post_init__(self) -> None:
        offsets = frozen_array(self.offset_samples)
        log_nhi = frozen_array(self.log_nhi_samples)

        if offsets.ndim != 1 or log_nhi.shape != offsets.shape:
            raise ValueError(
                f"offset_samples {offsets.shape} and log_nhi_samples "
                f"{log_nhi.shape} must be 1-D and the same length"
            )
        if offsets.size == 0:
            raise ValueError("sample grid is empty")
        if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(log_nhi)):
            raise ValueError("sample grid contains non-finite values")
        if offsets.min() < 0.0 or offsets.max() >= 1.0:
            raise ValueError(
                f"offset_samples must lie in [0, 1), got "
                f"[{offsets.min()}, {offsets.max()}]"
            )

        # nhi_samples is carried, not derived. Computing 10**log_nhi at load time
        # would make the array platform-dependent: float64 `**` goes through the
        # platform libm `pow`, which is not correctly rounded, so the same asset
        # yields different last bits on macOS/arm64 and Linux/x86-64. That was a
        # real CI failure, not a hypothetical. When absent (an older asset or a
        # hand-built grid) it is derived, and the platform caveat then applies.
        nhi = (
            frozen_array(self.nhi_samples)
            if self.nhi_samples is not None
            else frozen_array(10.0**log_nhi)
        )
        if nhi.ndim != 1 or nhi.shape != log_nhi.shape:
            raise ValueError(
                f"nhi_samples {nhi.shape} must be 1-D and match log_nhi_samples "
                f"{log_nhi.shape}"
            )
        if not np.all(np.isfinite(nhi)):
            raise ValueError("nhi_samples contains non-finite values")
        if not np.all(nhi > 0.0):
            raise ValueError("nhi_samples must be strictly positive")

        # Semantic check: the linear and logarithmic column densities must agree.
        # Shape agreement alone would let a grid carry mutually inconsistent
        # arrays straight into the evidence integral.
        #
        # The tolerance is deliberate, not slack. Storing nhi_samples is what
        # makes the asset bit-reproducible across architectures, precisely
        # because 10**x through the platform libm is not; so a bitwise check
        # here would reintroduce the platform dependence it exists to avoid. The
        # bound admits last-bit disagreement and nothing more.
        if not np.allclose(nhi, 10.0**log_nhi, rtol=NHI_CONSISTENCY_RTOL, atol=0.0):
            worst = np.max(np.abs(nhi - 10.0**log_nhi) / (10.0**log_nhi))
            raise ValueError(
                f"nhi_samples is not consistent with 10**log_nhi_samples: worst "
                f"relative difference {worst:.3e} exceeds {NHI_CONSISTENCY_RTOL:.0e}"
            )

        object.__setattr__(self, "offset_samples", offsets)
        object.__setattr__(self, "log_nhi_samples", log_nhi)
        object.__setattr__(self, "nhi_samples", nhi)
        object.__setattr__(self, "provenance", deep_freeze(self.provenance))

    @property
    def num_samples(self) -> int:
        return int(self.offset_samples.size)

    @property
    def log_nhi_range(self) -> tuple[float, float]:
        """Support actually spanned by the samples."""
        return float(self.log_nhi_samples.min()), float(self.log_nhi_samples.max())

    @property
    def declared_support(self) -> tuple[float, float] | None:
        """The prior support the grid was generated over, from provenance.

        ``None`` when the grid carries no provenance, or when what it carries is
        not the expected shape. A malformed record is not a support range, and
        reporting one would be worse than reporting nothing.
        """
        prior = _as_mapping(self.provenance.get("prior")) if self.provenance else None
        if prior is None:
            return None
        support = prior.get("support_log_nhi")
        if isinstance(support, (str, bytes)) or not isinstance(support, Sequence):
            return None
        try:
            values = tuple(float(item) for item in support)
        except (TypeError, ValueError):
            return None
        return values if len(values) == 2 else None  # type: ignore[return-value]

    @property
    def inference_metadata(self) -> tuple[str, ...]:
        """Which required provenance fields this grid is **missing**.

        Empty means the grid can be checked against a configuration, and so can
        be used for inference. Anything else names what a caller has to supply
        before it can.

        The fields are the ones a consistency check needs, not everything the
        builder records: a stable name, the sample count, the declared support,
        the prior family and mixture weight, how it was generated, and a hash
        to pin the arrays.
        """
        missing: list[str] = []
        provenance = _as_mapping(self.provenance)
        if not provenance:
            return REQUIRED_GRID_METADATA

        name = provenance.get("name")
        if not isinstance(name, str) or not name.strip():
            missing.append("name")
        digest = provenance.get("sha256")
        if not isinstance(digest, str) or not digest.strip():
            missing.append("sha256")

        prior = _as_mapping(provenance.get("prior"))
        if prior is None:
            missing.extend(
                ["prior.family", "prior.support_log_nhi", "prior.mixture_weight_pw14"]
            )
        else:
            if "support_log_nhi" not in prior:
                missing.append("prior.support_log_nhi")
            if "mixture_weight_pw14" not in prior:
                missing.append("prior.mixture_weight_pw14")
            family = prior.get("family")
            if not isinstance(family, str) or not family.strip():
                missing.append("prior.family")

        qmc = _as_mapping(provenance.get("qmc"))
        if qmc is None:
            missing.extend(["qmc.num_samples", "qmc.seed"])
        else:
            if "num_samples" not in qmc:
                missing.append("qmc.num_samples")
            if "seed" not in qmc:
                missing.append("qmc.seed")

        return tuple(missing)

    @property
    def unusable_because(self) -> tuple[str, ...]:
        """Everything standing between this grid and inference, in one list.

        Missing metadata and failed integrity checks are different faults with
        the same consequence, and a caller fixing one wants to see the other in
        the same message rather than after another attempt.
        """
        reasons = [f"missing metadata: {field}" for field in self.inference_metadata]
        if self.integrity_problems is None:
            reasons.append(
                "integrity was never checked: this grid was not loaded from a "
                "file with its provenance sidecar"
            )
        else:
            reasons.extend(
                f"integrity: {problem}" for problem in self.integrity_problems
            )
        return tuple(reasons)

    @property
    def usable_for_inference(self) -> bool:
        """Whether this grid may be used to evaluate a spectrum.

        Two conditions, and both are required:

        * every field in :data:`REQUIRED_GRID_METADATA` is present; and
        * the recorded digests were **checked against this file and these
          arrays** and matched.

        Metadata presence alone is not enough. A sidecar can be copied from a
        different grid, or the arrays edited after it was written, and the
        field names would look correct either way. A grid loaded from a bare
        ``.npz``, or one built by hand, fails the second condition because
        nothing was ever verified.

        Such a grid still loads and can be inspected -- the arrays are all
        there. It is inference that :class:`~gp_dla_finder.finder.Finder`
        refuses.
        """
        return not self.inference_metadata and self.integrity_problems == ()

    @property
    def declared_prior_alpha(self) -> float | None:
        """The PW14 mixture weight the grid was generated with, from provenance.

        ``None`` when the grid carries no provenance, which is the case for an
        ``.npz`` loaded without its sidecar. A caller that needs the check must
        keep the sidecar; see :func:`load_sample_grid`.
        """
        prior = _as_mapping(self.provenance.get("prior")) if self.provenance else None
        if prior is None or "mixture_weight_pw14" not in prior:
            return None
        try:
            return float(prior["mixture_weight_pw14"])
        except (TypeError, ValueError):
            return None

    @property
    def is_verified(self) -> bool:
        """Whether this grid's identity with a deployed production grid is proven.

        **Fails closed.** Only a status on :data:`VERIFIED_IDENTITY_STATUSES`
        counts as verified; missing, unknown, misspelled, regenerated or
        ``"unverified"`` all return ``False``. An earlier implementation asked
        whether the status *failed* to start with ``"regenerated"``, which would
        have treated a typo — or any new status string — as verified.
        """
        identity = self.provenance.get("identity") if self.provenance else None
        if not identity:
            return False
        return identity.get("status") in VERIFIED_IDENTITY_STATUSES

    @property
    def identity_status(self) -> str:
        identity = self.provenance.get("identity") if self.provenance else None
        return str(identity.get("status", "unknown")) if identity else "unknown"

    def sample_redshifts(self, z_min: float, z_max: float) -> np.ndarray:
        """Stretch the stored offsets onto a spectrum's absorber search window.

        Mirrors the reference: ``z_i = z_min + (z_max - z_min) * offset_i``.
        """
        return z_min + (z_max - z_min) * self.offset_samples

    def __repr__(self) -> str:  # pragma: no cover - display only
        lo, hi = self.log_nhi_range
        return (
            f"AbsorberSampleGrid(name={self.name!r}, n={self.num_samples}, "
            f"log_nhi=[{lo:.2f}, {hi:.2f}], verified={self.is_verified})"
        )


def available_sample_grids() -> tuple[str, ...]:
    """Names of sample grids bundled with this installation."""
    root = resources.files(_DATA_PACKAGE)
    return tuple(
        sorted(
            p.name[: -len(".npz")] for p in root.iterdir() if p.name.endswith(".npz")
        )
    )


def sample_grid_provenance(name: str = DEFAULT_SAMPLE_GRID):
    """Provenance for a bundled grid: prior, QMC construction, hashes, identity."""
    handle = resources.files(_DATA_PACKAGE) / f"{name}.json"
    if not handle.is_file():
        raise ValueError(_unknown_grid_message(name))
    return deep_freeze(json.loads(handle.read_text()))


def _unknown_grid_message(name: str) -> str:
    return (
        f"unknown sample grid {name!r}; bundled grids: "
        f"{', '.join(available_sample_grids())}. "
        "To load one from disk, pass path=... instead."
    )


def load_sample_grid(
    name: str = DEFAULT_SAMPLE_GRID, *, path: str | Path | None = None
) -> AbsorberSampleGrid:
    """Load a QMC absorber sample grid.

    With ``name``, one of the packaged grids. With ``path``, an ``.npz`` built by
    ``tools/build_sample_grid.py`` anywhere on disk — you do **not** need to put
    a custom grid inside the installed package.

    An external grid keeps its provenance. The builder writes ``<name>.json``
    beside the ``.npz``; if that sidecar is present it is loaded too, and the
    grid takes its recorded name.

    **Without the sidecar the arrays still load, but the grid cannot be used for
    inference.** It has no declared support, prior mixture or stable name, so
    nothing can check it against a configuration, and a run using it would
    record a configuration describing a different grid. Inspect such a grid
    freely; :class:`~gp_dla_finder.finder.Finder` refuses it.
    Check :attr:`AbsorberSampleGrid.usable_for_inference` if you need to know
    before constructing one.
    """
    if path is not None:
        path = Path(path)
        with np.load(path) as data:
            arrays = {
                "offset_samples": data["offset_samples"],
                "log_nhi_samples": data["log_nhi_samples"],
                "nhi_samples": data["nhi_samples"] if "nhi_samples" in data else None,
            }
        sidecar = path.with_suffix(".json")
        provenance = {}
        grid_name = f"<external:{path.stem}>"
        integrity: tuple[str, ...] | None = None
        if sidecar.is_file():
            try:
                provenance = deep_freeze(json.loads(sidecar.read_text()))
            except json.JSONDecodeError as error:
                # A malformed sidecar is not "no sidecar": it claims to
                # describe this grid and cannot be read. Say so, and leave the
                # grid unusable rather than silently falling back.
                provenance = {}
                integrity = (f"{sidecar.name} is not valid JSON: {error}",)
            else:
                # The name is read through the same type guard as everything
                # else. A sidecar whose top level is a list or a string parses
                # fine and has no .get, and reaching for one here would crash
                # the load before the integrity check could report the problem.
                record = _as_mapping(provenance)
                declared_name = record.get("name") if record is not None else None
                if isinstance(declared_name, str) and declared_name.strip():
                    grid_name = declared_name.strip()
                integrity = verify_grid_integrity(path, provenance, arrays)
        return AbsorberSampleGrid(
            name=grid_name,
            provenance=provenance,
            integrity_problems=integrity,
            **arrays,
        )

    handle = resources.files(_DATA_PACKAGE) / f"{name}.npz"
    if not handle.is_file():
        raise ValueError(_unknown_grid_message(name))
    provenance = sample_grid_provenance(name)
    with resources.as_file(handle) as npz_path, np.load(npz_path) as data:
        arrays = {
            "offset_samples": data["offset_samples"],
            "log_nhi_samples": data["log_nhi_samples"],
            "nhi_samples": data["nhi_samples"] if "nhi_samples" in data else None,
        }
        # The packaged grids are verified on the same path as an external one.
        # A bundled asset that had been corrupted in transit or rebuilt without
        # its sidecar would otherwise be the one case nothing checked.
        integrity = verify_grid_integrity(npz_path, provenance, arrays)
    return AbsorberSampleGrid(
        name=name,
        provenance=provenance,
        integrity_problems=integrity,
        **arrays,
    )
