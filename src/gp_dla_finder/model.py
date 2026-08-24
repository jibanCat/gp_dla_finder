"""Trained Gaussian-process quasar-emission models.

A :class:`GPModel` is the learned prior over unabsorbed quasar spectra: a mean
flux ``mu`` on a rest-frame wavelength grid, a rank-``k`` factor ``M`` giving the
low-rank covariance ``K = M M^T``, a per-pixel amplitude ``log_omega`` for the
Lyman-forest absorption noise, and three learned scalars ``(c_0, tau_0, beta)``
parameterising that noise.

Model-coupled quantities, including the rank, rest-frame grid, and flux
normalization band, are properties of the model. They are not global defaults;
the package checks them against the configuration before inference.

Assets
------
Packaged models live in ``gp_dla_finder/data/models`` as an ``.npz`` of inference
arrays plus a ``.json`` of provenance. Arrays may be stored as float32 where that
is bitwise lossless (see ``tools/convert_model.py``); they are always loaded as
float64 so the arithmetic matches the reference implementation exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._immutable import deep_freeze, frozen_array

__all__ = [
    "GPModel",
    "DEFAULT_MODEL",
    "available_models",
    "load_model",
    "model_provenance",
]

#: The model that reproduces the deployed DESI Y3 production catalogues.
#:
#: The name records its training provenance rather than a status word: it is the
#: deployed model, which is not necessarily the newest retrain.
#: Trained on the 2LPT-0 (loa-124) mock, which is also the calibration mock, so
#: finder performance quoted on 2LPT-0 is in-sample.
#:
#: The mock spectra this model was trained on are **not** redistributable and are
#: not included in this package. The trained parameters are a
#: separate artifact from the training data.
DEFAULT_MODEL = "phase2_2lpt_loa124_nohcd_nobal_wide_m"

_DATA_PACKAGE = "gp_dla_finder.data.models"


@dataclass(frozen=True)
class GPModel:
    """A trained GP prior over quasar emission.

    Attributes
    ----------
    name
        Asset name, or ``"<external>"`` for a model loaded from an arbitrary path.
    rest_wavelengths
        Rest-frame grid the model is defined on, angstroms, shape ``(W,)``.
    mu
        Mean flux on that grid, shape ``(W,)``.
    M
        Low-rank covariance factor, shape ``(W, k)``; ``K = M M^T``.
    log_omega
        Log amplitude of the absorption-noise term, shape ``(W,)``.
    log_c_0, log_tau_0, log_beta
        Learned scalars of the Lyman-forest noise model.
    normalization_min_lambda, normalization_max_lambda
        Rest-frame band the training spectra were normalised over. ``None`` for
        older models that do not record it; ``nan`` if trained unnormalised.
    provenance
        Read-only provenance mapping; empty for externally loaded models.
    """

    name: str
    rest_wavelengths: np.ndarray
    mu: np.ndarray
    M: np.ndarray
    log_omega: np.ndarray
    log_c_0: float
    log_tau_0: float
    log_beta: float
    normalization_min_lambda: float | None = None
    normalization_max_lambda: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Freezing the dataclass stops attribute reassignment but not in-place
        # edits to the arrays, and a caller who kept a reference to what they
        # passed in could otherwise mutate the model's likelihood inputs. Take
        # private, genuinely immutable copies (see _immutable.frozen_array: a
        # read-only flag alone can be turned back on by the caller).
        for attr in ("rest_wavelengths", "mu", "M", "log_omega"):
            object.__setattr__(self, attr, frozen_array(getattr(self, attr)))
        object.__setattr__(self, "provenance", deep_freeze(self.provenance))

        w = self.rest_wavelengths.shape[0]
        for attr in ("mu", "log_omega"):
            got = getattr(self, attr).shape
            if got != (w,):
                raise ValueError(f"{attr} has shape {got}, expected {(w,)}")
        if self.M.ndim != 2 or self.M.shape[0] != w:
            raise ValueError(f"M has shape {self.M.shape}, expected ({w}, k)")
        if not np.all(np.diff(self.rest_wavelengths) > 0):
            raise ValueError("rest_wavelengths must be strictly increasing")

    @property
    def rank(self) -> int:
        """Rank ``k`` of the low-rank covariance factor."""
        return self.M.shape[1]

    @property
    def rest_wavelength_range(self) -> tuple[float, float]:
        return float(self.rest_wavelengths[0]), float(self.rest_wavelengths[-1])

    @property
    def learned_tau_0(self) -> float:
        return float(np.exp(self.log_tau_0))

    @property
    def learned_beta(self) -> float:
        return float(np.exp(self.log_beta))

    @property
    def learned_c_0(self) -> float:
        return float(np.exp(self.log_c_0))

    def covers(self, min_lambda: float, max_lambda: float) -> bool:
        """Whether the model's grid spans a rest-frame window, inclusive."""
        lo, hi = self.rest_wavelength_range
        return lo <= min_lambda and max_lambda <= hi

    def __repr__(self) -> str:  # pragma: no cover - display only
        lo, hi = self.rest_wavelength_range
        return (
            f"GPModel(name={self.name!r}, rank={self.rank}, "
            f"grid=[{lo:.2f}, {hi:.2f}] A x {self.rest_wavelengths.size}, "
            f"norm=[{self.normalization_min_lambda}, {self.normalization_max_lambda}])"
        )


def available_models() -> tuple[str, ...]:
    """Names of models bundled with this installation."""
    root = resources.files(_DATA_PACKAGE)
    return tuple(
        sorted(
            p.name[: -len(".npz")] for p in root.iterdir() if p.name.endswith(".npz")
        )
    )


def model_provenance(name: str = DEFAULT_MODEL) -> Mapping[str, Any]:
    """Provenance record for a bundled model: source checksum, training run, grid."""
    handle = resources.files(_DATA_PACKAGE) / f"{name}.json"
    if not handle.is_file():
        raise ValueError(_unknown_model_message(name))
    return deep_freeze(json.loads(handle.read_text()))


def _unknown_model_message(name: str) -> str:
    return (
        f"unknown model {name!r}; bundled models: {', '.join(available_models())}. "
        "To load a model from disk, pass path=... instead."
    )


def _from_arrays(name: str, data: Mapping[str, Any], provenance: Mapping) -> GPModel:
    def arr(key: str) -> np.ndarray:
        return np.asarray(data[key], dtype=np.float64)

    def scalar(key: str) -> float | None:
        return float(data[key]) if key in data else None

    return GPModel(
        name=name,
        rest_wavelengths=arr("rest_wavelengths"),
        mu=arr("mu"),
        M=arr("M"),
        log_omega=arr("log_omega"),
        log_c_0=float(data["log_c_0"]),
        log_tau_0=float(data["log_tau_0"]),
        log_beta=float(data["log_beta"]),
        normalization_min_lambda=scalar("normalization_min_lambda"),
        normalization_max_lambda=scalar("normalization_max_lambda"),
        provenance=MappingProxyType(dict(provenance)),
    )


def load_model(name: str = DEFAULT_MODEL, *, path: str | Path | None = None) -> GPModel:
    """Load a trained GP model.

    Parameters
    ----------
    name
        Bundled asset name. Ignored when ``path`` is given.
    path
        Load from disk instead: either a packaged ``.npz`` or an HDF5 ``.h5`` /
        MATLAB v7.3 ``.mat`` file from the reference pipeline. HDF5 input needs
        the optional ``legacy`` extra (``pip install 'gp_dla_finder[legacy]'``).

    Returns
    -------
    GPModel

    Examples
    --------
    >>> model = load_model()                      # doctest: +SKIP
    >>> model.rank                                # doctest: +SKIP
    30
    """
    if path is not None:
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                return _from_arrays("<external>", dict(data), {})
        return _load_hdf5(path)

    handle = resources.files(_DATA_PACKAGE) / f"{name}.npz"
    if not handle.is_file():
        raise ValueError(_unknown_model_message(name))
    with resources.as_file(handle) as npz_path, np.load(npz_path) as data:
        return _from_arrays(name, dict(data), model_provenance(name))


def _load_hdf5(path: Path) -> GPModel:
    """Read a reference-pipeline trained model directly.

    Mirrors the reference loader, including its DESI-vs-MATLAB layout sniff: DESI
    exports store ``log_tau_0`` as a scalar, the older MATLAB export as a column.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "reading .h5/.mat models needs h5py: pip install 'gp_dla_finder[legacy]'"
        ) from exc

    with h5py.File(path, "r") as handle:
        is_desi = handle["log_tau_0"].ndim == 0
        if is_desi:
            data: dict[str, Any] = {
                "rest_wavelengths": handle["rest_wavelengths"][:],
                "mu": handle["mu"][:],
                "M": handle["M"][:],
                "log_omega": handle["log_omega"][:],
                "log_c_0": handle["log_c_0"][()],
                "log_tau_0": handle["log_tau_0"][()],
                "log_beta": handle["log_beta"][()],
            }
        else:
            data = {
                "rest_wavelengths": handle["rest_wavelengths"][:, 0],
                "mu": handle["mu"][:, 0],
                "M": handle["M"][()].T,
                "log_omega": handle["log_omega"][:, 0],
                "log_c_0": handle["log_c_0"][0, 0],
                "log_tau_0": handle["log_tau_0"][0, 0],
                "log_beta": handle["log_beta"][0, 0],
            }
        for key in ("normalization_min_lambda", "normalization_max_lambda"):
            if key in handle:
                data[key] = handle[key][()]

    return _from_arrays(path.stem, data, {"source": {"filename": path.name}})
