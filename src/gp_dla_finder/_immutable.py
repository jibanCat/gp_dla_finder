"""Genuinely immutable arrays and mappings.

Shared, module-level tables that define the forward model -- LSF kernels, atomic
data, trained-model parameters, prior tables -- must not be editable by a caller.
Editing one in place changes every subsequent computation in the process, silently
and without any record in the result.

``setflags(write=False)`` is *not* sufficient. An array that owns its storage can
have the flag turned back on::

    kernel = lsf_kernel("desi-r3000-7tap")
    kernel.setflags(write=True)     # succeeds when the array owns its data
    kernel[3] = 0.4327              # forward model silently changed

:func:`frozen_array` closes that path by backing the array with an immutable
``bytes`` object. The array does not own its storage, so ``setflags(write=True)``
raises, and the base object is a ``bytes`` that cannot be unlocked either.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = ["deep_freeze", "frozen_array"]


def frozen_array(values, dtype=np.float64) -> np.ndarray:
    """Return an immutable copy of ``values``.

    The result is a normal NumPy array for every read purpose -- arithmetic,
    slicing, broadcasting -- but no mutation route reaches its storage: item and
    slice assignment raise, ``setflags(write=True)`` raises because the array does
    not own its data, and its ``.base`` is an immutable ``bytes`` object.

    Callers who need to modify the values should take an explicit ``.copy()``,
    which is writable as usual.
    """
    source = np.ascontiguousarray(values, dtype=dtype)
    flat = np.frombuffer(source.tobytes(), dtype=dtype)
    return flat if source.ndim == 1 else flat.reshape(source.shape)


def deep_freeze(value: Any) -> Any:
    """Recursively make a nested provenance structure immutable.

    A top-level :class:`~types.MappingProxyType` still exposes mutable nested
    dicts and lists, so provenance could be edited after loading and then
    reported as though it came from the asset.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        # Copied AND made read-only. A mapping proxy cannot protect an array at
        # all: without the copy the frozen structure aliases whatever the caller
        # still holds, so mutating theirs rewrites the record.
        frozen = np.array(value, copy=True)
        frozen.flags.writeable = False
        return frozen
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    return value
