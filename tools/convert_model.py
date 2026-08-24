#!/usr/bin/env python3
"""Convert a trained GP quasar-emission model into a packaged asset.

The trained models produced by the reference pipeline are HDF5 files that carry
both the inference parameters and the full training history. Inference reads only
a small subset. This tool extracts that subset and writes:

* ``<name>.npz``  -- the inference arrays, compressed;
* ``<name>.json`` -- provenance, checksums, and the conversion audit.

Precision rule
-----------------------------
Arrays may be stored as float32 **only** where an exhaustive round-trip check
proves every stored value was already a float32 value upcast on write. That is
the case for models trained by the PyTorch trainer: the file is float64 on disk
but the mantissas are float32. The check is per-array and per-element; any array
that fails is stored at full float64 precision instead. This is lossless
repacking, not a precision reduction, and the loader upcasts back to float64 so
inference arithmetic is unchanged.

Usage
-----
    python tools/convert_model.py SOURCE.h5 --name <asset-name> [--out DIR]

Run with ``--check`` to re-verify an existing asset against its source.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

# Datasets the inference path reads. Anything else in the source file is training
# provenance and is deliberately not carried into the package.
ARRAY_KEYS = ("rest_wavelengths", "mu", "M", "log_omega")
SCALAR_KEYS = ("log_c_0", "log_tau_0", "log_beta")
OPTIONAL_SCALAR_KEYS = ("normalization_min_lambda", "normalization_max_lambda")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "src" / "gp_dla_finder" / "data" / "models"

# Training attributes safe to copy verbatim: scalar hyperparameters that describe
# the run, not where it ran. Everything else is redacted (see _sanitise_attrs).
#
# This is an allowlist rather than a denylist on purpose. Trained files carry
# whatever the trainer happened to stamp on them, which has included absolute
# cluster paths containing usernames; a denylist only catches the leaks already
# thought of.
SAFE_ATTR_KEYS = frozenset({"lr", "n_iters", "n_spectra", "vectorized"})

_PATH_MARKERS = ("/", "\\", "~")


def looks_like_path(value: object) -> bool:
    """Whether a provenance value could carry a filesystem location."""
    return isinstance(value, str) and any(m in value for m in _PATH_MARKERS)


def _sanitise_attrs(attrs) -> tuple[dict, dict]:
    """Split source attributes into a safe record and a redaction audit.

    Returns ``(safe, redacted)``. Values on the allowlist are copied; every other
    value is dropped. For a path-like value, we keep only the final path
    component's parent directory name, which is the training-run name already
    public in the asset name. We do not publish a checksum of the original
    path, since that would let someone with a good guess confirm the withheld
    location.
    """
    safe: dict[str, object] = {}
    redacted: dict[str, object] = {}

    for key, raw in attrs.items():
        value = raw.item() if hasattr(raw, "item") else raw
        if key in SAFE_ATTR_KEYS and not looks_like_path(value):
            safe[key] = value
            continue

        entry: dict[str, object] = {"reason": "not on the provenance allowlist"}
        if looks_like_path(value):
            parts = [p for p in str(value).replace("\\", "/").split("/") if p]
            entry = {
                "reason": "path-like value withheld",
                # e.g. the training-run directory name, which is not a location
                # and is already public in this asset's own name.
                "training_run": parts[-2] if len(parts) >= 2 else None,
            }
        redacted[key] = entry

    return safe, redacted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Checksum of an array's exact bytes, in canonical float64/C order."""
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def float32_is_lossless(array: np.ndarray) -> bool:
    """True when every element survives a float64 -> float32 -> float64 round trip."""
    return np.array_equal(array, array.astype(np.float32).astype(np.float64))


def read_source(path: Path) -> tuple[dict, dict]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - tooling-only path
        raise SystemExit(
            "h5py is required to convert models: pip install 'gp_dla_finder[legacy]'"
        ) from exc

    values: dict[str, np.ndarray | float] = {}
    source_info: dict[str, object] = {}
    with h5py.File(path, "r") as handle:
        present = sorted(handle.keys())
        source_info["datasets_present"] = present
        source_info["datasets_read"] = []

        # The reference loader distinguishes DESI-format files (scalar log_tau_0)
        # from the older MATLAB export (column vectors). Mirror that exactly.
        is_desi = handle["log_tau_0"].ndim == 0
        source_info["source_format"] = "desi-h5" if is_desi else "matlab-v7.3"

        for key in ARRAY_KEYS:
            raw = handle[key]
            if is_desi:
                values[key] = np.asarray(raw[:], dtype=np.float64)
            else:
                values[key] = (
                    np.asarray(raw[()], dtype=np.float64).T
                    if key == "M"
                    else np.asarray(raw[:, 0], dtype=np.float64)
                )
            source_info["datasets_read"].append(key)

        for key in SCALAR_KEYS:
            values[key] = (
                float(handle[key][()]) if is_desi else float(handle[key][0, 0])
            )
            source_info["datasets_read"].append(key)

        for key in OPTIONAL_SCALAR_KEYS:
            if key in handle:
                values[key] = float(handle[key][()])
                source_info["datasets_read"].append(key)

        safe_attrs, redacted_attrs = _sanitise_attrs(handle.attrs)
        source_info["attrs"] = safe_attrs
        source_info["attrs_redacted"] = redacted_attrs

    source_info["datasets_unused"] = [
        k
        for k in source_info["datasets_present"]
        if k not in source_info["datasets_read"]
    ]
    return values, source_info


def convert(
    source: Path,
    name: str,
    out_dir: Path,
    notes: str | None,
    normalization_band: tuple[float, float] | None = None,
    normalization_source: str | None = None,
) -> Path:
    values, source_info = read_source(source)

    # Some historical artifacts do not embed a normalisation band. It may be
    # supplied, but only with an explicit statement of where the value came from:
    # the provenance must never imply it was read out of the file.
    stamped_band = None
    if normalization_band is not None:
        embedded = "normalization_min_lambda" in values
        if embedded:
            raise SystemExit(
                f"refusing to stamp a normalisation band onto {source.name}: the "
                "file already embeds one; that value is authoritative"
            )
        if not normalization_source:
            raise SystemExit(
                "--normalization-source is required with --normalization-band, so "
                "the provenance records where the value came from"
            )
        lo, hi = normalization_band
        if not lo < hi:
            raise SystemExit(f"normalisation band must be increasing, got {lo}, {hi}")
        values["normalization_min_lambda"] = float(lo)
        values["normalization_max_lambda"] = float(hi)
        stamped_band = {
            "embedded_in_source": False,
            "stamped_at_conversion": True,
            "value": [float(lo), float(hi)],
            "attributed_to": normalization_source,
            "note": (
                "The source artifact records no normalisation metadata. This band "
                "was supplied at conversion time from the stated authority and was "
                "NOT extracted from the file."
            ),
        }

    arrays: dict[str, np.ndarray] = {}
    audit: dict[str, dict] = {}
    for key in ARRAY_KEYS:
        original = values[key]
        lossless = float32_is_lossless(original)
        stored = original.astype(np.float32) if lossless else original
        arrays[key] = stored

        # Prove it here, not just in a test: round-trip and compare bitwise.
        restored = np.asarray(stored, dtype=np.float64)
        if not np.array_equal(restored, original):
            raise SystemExit(
                f"refusing to write {name}: array {key!r} does not round-trip bitwise"
            )
        audit[key] = {
            "shape": list(original.shape),
            "stored_dtype": str(stored.dtype),
            "float32_lossless": bool(lossless),
            "sha256_float64": sha256_array(original),
        }

    scalars = {k: float(values[k]) for k in SCALAR_KEYS if k in values}
    scalars.update({k: float(values[k]) for k in OPTIONAL_SCALAR_KEYS if k in values})
    for key, value in scalars.items():
        arrays[key] = np.float64(value)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    np.savez_compressed(npz_path, **arrays)

    rest = values["rest_wavelengths"]
    provenance = {
        "name": name,
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            # Filename + checksum identify the artifact without recording a
            # machine-specific or cluster path.
            "filename": source.name,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "format": source_info["source_format"],
            "training_attrs": source_info["attrs"],
            "training_attrs_redacted": source_info["attrs_redacted"],
            "datasets_unused": source_info["datasets_unused"],
            "normalization_embedded": "normalization_min_lambda"
            in source_info["datasets_read"],
        },
        "model": {
            "rank": int(values["M"].shape[1]),
            "rest_wavelength_min": float(rest[0]),
            "rest_wavelength_max": float(rest[-1]),
            "rest_wavelength_count": int(rest.size),
            "normalization_min_lambda": scalars.get("normalization_min_lambda"),
            "normalization_max_lambda": scalars.get("normalization_max_lambda"),
            "learned_tau_0": float(np.exp(scalars["log_tau_0"])),
            "learned_beta": float(np.exp(scalars["log_beta"])),
            "learned_c_0": float(np.exp(scalars["log_c_0"])),
            "normalization_provenance": stamped_band
            or {
                "embedded_in_source": True,
                "stamped_at_conversion": False,
                "note": "Read from the trained artifact's own metadata.",
            },
        },
        "conversion": {
            "rule": (
                "float32 storage only where a per-element float64->float32->float64 "
                "round trip is exact; verified at write time and re-verified by "
                "tests/test_model_assets.py"
            ),
            "arrays": audit,
            "npz_sha256": None,
        },
        "notes": notes or "",
    }
    provenance["conversion"]["npz_sha256"] = sha256_file(npz_path)

    (out_dir / f"{name}.json").write_text(json.dumps(provenance, indent=2) + "\n")

    src_mb = source.stat().st_size / 1e6
    out_mb = npz_path.stat().st_size / 1e6
    print(f"wrote {npz_path}  ({src_mb:.2f} MB -> {out_mb:.2f} MB)")
    for key, info in audit.items():
        flag = "float32 (lossless)" if info["float32_lossless"] else "float64 (kept)"
        print(f"  {key:18s} {str(info['shape']):14s} {flag}")
    print(
        f"  unused in source: {', '.join(source_info['datasets_unused']) or '(none)'}"
    )
    return npz_path


def check(source: Path, name: str, out_dir: Path) -> None:
    """Re-verify a packaged asset reproduces its source bitwise."""
    values, _ = read_source(source)
    with np.load(out_dir / f"{name}.npz") as packed:
        for key in ARRAY_KEYS:
            restored = np.asarray(packed[key], dtype=np.float64)
            if not np.array_equal(restored, values[key]):
                raise SystemExit(f"MISMATCH in {key!r}")
        for key in SCALAR_KEYS:
            if float(packed[key]) != values[key]:
                raise SystemExit(f"MISMATCH in scalar {key!r}")
    print(f"{name}: reproduces {source.name} bitwise")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trained model .h5 / .mat")
    parser.add_argument("--name", required=True, help="asset name (provenance-bearing)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--normalization-band",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Rest-frame flux-normalisation band, angstroms. Only for artifacts "
            "that embed none; requires --normalization-source."
        ),
    )
    parser.add_argument(
        "--normalization-source",
        default=None,
        help="Authority for a stamped --normalization-band (recorded verbatim).",
    )
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    args = parser.parse_args()

    if args.check:
        check(args.source, args.name, args.out)
    else:
        convert(
            args.source,
            args.name,
            args.out,
            args.notes,
            normalization_band=(
                tuple(args.normalization_band) if args.normalization_band else None
            ),
            normalization_source=args.normalization_source,
        )


if __name__ == "__main__":
    main()
