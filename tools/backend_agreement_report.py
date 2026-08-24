"""Where exactly do two Voigt backends disagree, and by how much.

The compiled backend was being refused registration
on Linux by a *relative* decrement test that divides by a decrement which goes
to zero in the far wings -- so a 1e-14 absolute difference could read as a
1e-8 relative one. The ruling requires that the replacement gate's floor be
chosen from measurements of the worst point, not picked to sit just above the
number that happened to fail.

This prints, for the worst point of each metric: the platform and library
identities, the redshift, the column density, the wavelength, the two profile
values, the expected decrement, and the absolute and relative decrement
differences.

Provenance is sanitised: no executable,
home, repository, virtualenv or shared-library paths, only portable identities.

    python tools/backend_agreement_report.py
    python tools/backend_agreement_report.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _identities() -> dict:
    """Portable identities only -- versions and names, never paths."""
    import numpy

    record = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_build": " ".join(platform.python_build()),
        "numpy": numpy.__version__,
    }
    try:
        import scipy

        record["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        record["scipy"] = "absent"

    from gp_dla_finder.voigt import available_backends, backend_provenance

    record["available_backends"] = list(available_backends())
    if "libcerf" in available_backends():
        provenance = dict(backend_provenance("libcerf"))
        # Keep identity and hashes; drop anything that could name a filesystem.
        record["libcerf"] = {
            key: value
            for key, value in provenance.items()
            if not isinstance(value, str) or ("/" not in value and "\\" not in value)
        }
    else:
        from gp_dla_finder.voigt import backend_rejections

        record["libcerf_rejected"] = dict(backend_rejections())
    return record


def compare(backend_name: str = "libcerf") -> dict:
    """Every probe point, with the worst of each metric located exactly."""
    from gp_dla_finder.voigt import (
        _BACKENDS,
        PRODUCTION_KERNEL,
        get_backend,
        kernel_half_width,
    )

    reference = _BACKENDS["numpy"]
    try:
        backend = get_backend(backend_name)
    except Exception:
        # Rejected backends are not in the registry; go to the extension direct.
        import gp_dla_finder._voigt_ext as ext  # noqa: PLC0415

        from gp_dla_finder.voigt import LibcerfVoigtBackend

        backend = LibcerfVoigtBackend(ext)

    probe = np.linspace(3600.0, 6000.0, 512)
    padded = np.linspace(3600.0, 6000.0, 512 + 2 * kernel_half_width(PRODUCTION_KERNEL))

    worst = {
        "absolute_profile": {"value": -1.0},
        "absolute_decrement": {"value": -1.0},
        "relative_decrement": {"value": -1.0},
        "broadened_absolute": {"value": -1.0},
    }
    # How the relative metric behaves as a function of how deep the decrement is.
    by_decade: dict[str, dict] = {}

    for z_dla in (2.0, 2.6, 3.4):
        for log_nhi in (17.2, 19.0, 20.3, 22.0, 23.0):
            nhi = 10.0**log_nhi
            expected = reference.absorption(
                probe, nhi, z_dla, 3, PRODUCTION_KERNEL, False
            )
            actual = backend.absorption(probe, nhi, z_dla, 3, PRODUCTION_KERNEL, False)

            absolute = np.abs(actual - expected)
            index = int(np.argmax(absolute))
            if absolute[index] > worst["absolute_profile"]["value"]:
                worst["absolute_profile"] = {
                    "value": float(absolute[index]),
                    "z_dla": z_dla,
                    "log_nhi": log_nhi,
                    "wavelength": float(probe[index]),
                    "expected": float(expected[index]),
                    "actual": float(actual[index]),
                }

            weak = expected > 0.5
            if np.any(weak):
                expected_decrement = 1.0 - expected[weak]
                actual_decrement = 1.0 - actual[weak]
                difference = np.abs(actual_decrement - expected_decrement)
                waves = probe[weak]

                j = int(np.argmax(difference))
                if difference[j] > worst["absolute_decrement"]["value"]:
                    worst["absolute_decrement"] = {
                        "value": float(difference[j]),
                        "z_dla": z_dla,
                        "log_nhi": log_nhi,
                        "wavelength": float(waves[j]),
                        "expected_decrement": float(expected_decrement[j]),
                        "relative_here": float(difference[j] / expected_decrement[j])
                        if expected_decrement[j] > 0
                        else float("inf"),
                    }

                positive = expected_decrement > 0
                if np.any(positive):
                    relative = difference[positive] / expected_decrement[positive]
                    k = int(np.argmax(relative))
                    if relative[k] > worst["relative_decrement"]["value"]:
                        worst["relative_decrement"] = {
                            "value": float(relative[k]),
                            "z_dla": z_dla,
                            "log_nhi": log_nhi,
                            "wavelength": float(waves[positive][k]),
                            "expected": float(expected[weak][positive][k]),
                            "actual": float(actual[weak][positive][k]),
                            "expected_decrement": float(
                                expected_decrement[positive][k]
                            ),
                            "absolute_decrement_difference": float(
                                difference[positive][k]
                            ),
                        }

                    # Bucket by how deep the decrement is: this is the whole
                    # question -- whether the large relative errors live only
                    # where the decrement is vanishing.
                    for decade in range(-16, 1):
                        lo, hi = 10.0**decade, 10.0 ** (decade + 1)
                        band = (expected_decrement[positive] >= lo) & (
                            expected_decrement[positive] < hi
                        )
                        if not np.any(band):
                            continue
                        key = f"1e{decade}"
                        entry = by_decade.setdefault(
                            key, {"n": 0, "worst_relative": 0.0, "worst_absolute": 0.0}
                        )
                        entry["n"] += int(np.sum(band))
                        entry["worst_relative"] = max(
                            entry["worst_relative"], float(np.max(relative[band]))
                        )
                        entry["worst_absolute"] = max(
                            entry["worst_absolute"],
                            float(np.max(difference[positive][band])),
                        )

            expected_broadened = reference.absorption(
                padded, nhi, z_dla, 3, PRODUCTION_KERNEL, True
            )
            actual_broadened = backend.absorption(
                padded, nhi, z_dla, 3, PRODUCTION_KERNEL, True
            )
            broadened = float(np.max(np.abs(actual_broadened - expected_broadened)))
            if broadened > worst["broadened_absolute"]["value"]:
                worst["broadened_absolute"] = {
                    "value": broadened,
                    "z_dla": z_dla,
                    "log_nhi": log_nhi,
                }

    return {"worst": worst, "by_decrement_decade": dict(sorted(by_decade.items()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="libcerf")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    record = {"identities": _identities(), **compare(args.backend)}

    print("=== identities ===")
    for key, value in record["identities"].items():
        print(f"  {key}: {value}")

    print("\n=== worst point of each metric ===")
    for metric, detail in record["worst"].items():
        print(f"  {metric}:")
        for key, value in detail.items():
            formatted = f"{value:.6e}" if isinstance(value, float) else value
            print(f"      {key}: {formatted}")

    print("\n=== relative error vs how deep the decrement is ===")
    print(f"  {'decrement':>12s} {'n':>7s} {'worst rel':>13s} {'worst abs':>13s}")
    for decade, entry in record["by_decrement_decade"].items():
        print(
            f"  {decade:>12s} {entry['n']:>7d} "
            f"{entry['worst_relative']:13.3e} {entry['worst_absolute']:13.3e}"
        )

    if args.json:
        args.json.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"\nrecord written to {args.json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
