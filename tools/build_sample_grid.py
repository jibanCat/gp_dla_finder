#!/usr/bin/env python3
"""Build a quasi-Monte-Carlo absorber sample grid.

The evidence integral is a QMC average over absorber parameters
``(z_abs, log10 N_HI)``. This tool generates that grid and records everything
needed to reproduce or audit it.

Scientific content, ported verbatim from the reference
``gpy_dla_detection/generate_samples.py``:

* the column-density prior is a mixture,
  ``p(logN) = alpha * p_PW14(logN) + (1 - alpha) * Uniform(min, max)``;
* ``p_PW14`` is the Prochaska et al. (2014) CDDF spline transformed to a density
  in log N_HI, proportional to ``f(N) * N * ln(10)``;
* ``f(N)`` is a monotone cubic Hermite (PCHIP) interpolation of the Table 2
  spline-model nodes;
* samples come from a two-dimensional scrambled Halton sequence: coordinate 0 is
  mapped through the inverse CDF to give log N_HI, coordinate 1 is the uniform
  absorber-redshift offset in [0, 1).

Provenance honesty
----------------------------------
A grid built here is **regenerated**. The reference notebook that produced the
historical grids constructed ``Halton(d=2, scramble=True)`` *without* a seed and
then called ``numpy.random.seed(42)``, which does not seed a SciPy QMC engine.
So a newly seeded grid is deterministic and reproducible going forward, but its
byte identity with a deployed grid is **unverified** until the deployed arrays or
their hashes are compared directly. The generated metadata says so.

Usage
-----
    python tools/build_sample_grid.py --name pw14_172_225_50000 \\
        --num-samples 50000 --log-nhi 17.2 22.5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats.qmc import Halton

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "src" / "gp_dla_finder" / "data" / "samples"

# Prochaska et al. (2014), Table 2 "Spline Model" (their Figure 7).
# Transcribed verbatim from the reference implementation. Do not adjust.
LOGNHI_NODES = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
LOGF_NODES = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])

#: CDF grid resolution used to invert the prior. Part of the algorithm identity.
CDF_GRID_POINTS = 50_000
#: Halton coordinate 0 -> log N_HI, coordinate 1 -> redshift offset.
HALTON_DIMENSION_ORDER = ("log_nhi", "z_offset")


# The digests are IMPORTED, not redefined here. The package recomputes them when
# it loads a grid, and two independent definitions of "the hash of this array"
# would eventually disagree over dtype, byte order or contiguity -- at which
# point every correctly built grid would start failing its own integrity check.
sys.path.insert(0, str(REPO_ROOT / "src"))
from gp_dla_finder.samples import (  # noqa: E402
    canonical_array_digest as sha256_array,
)
from gp_dla_finder.samples import (  # noqa: E402
    canonical_file_digest as sha256_file,
)


def f_pw14(log_nhi: np.ndarray) -> np.ndarray:
    """PW14 CDDF ``f(N_HI, X)``, clipped to the spline's node range."""
    spline = PchipInterpolator(LOGNHI_NODES, LOGF_NODES)
    clipped = np.clip(np.asarray(log_nhi), LOGNHI_NODES[0], LOGNHI_NODES[-1])
    return 10.0 ** spline(clipped)


def build_prior(min_log_nhi: float, max_log_nhi: float, alpha: float):
    """Normalised log N_HI prior and its inverse CDF."""

    def unnormalised(log_nhi):
        log_nhi = np.asarray(log_nhi)
        return f_pw14(log_nhi) * (10.0**log_nhi) * np.log(10.0)

    norm, _ = quad(unnormalised, min_log_nhi, max_log_nhi)

    def pdf(log_nhi):
        log_nhi = np.asarray(log_nhi)
        pw = unnormalised(log_nhi) / norm
        width = max_log_nhi - min_log_nhi
        uniform = ((log_nhi >= min_log_nhi) & (log_nhi <= max_log_nhi)) / width
        return alpha * pw + (1.0 - alpha) * uniform

    x = np.linspace(min_log_nhi, max_log_nhi, CDF_GRID_POINTS)
    cdf = np.cumsum(pdf(x))
    cdf /= cdf[-1]
    inverse_cdf = interp1d(
        cdf, x, bounds_error=False, assume_sorted=True, fill_value=(x[0], x[-1])
    )
    return pdf, inverse_cdf


def generate(num_samples: int, log_nhi_range, alpha: float, seed: int) -> dict:
    """Generate the grid. Deterministic given ``(num_samples, range, alpha, seed)``."""
    min_log_nhi, max_log_nhi = log_nhi_range
    halton = Halton(d=2, scramble=True, seed=seed).random(num_samples)
    _, inverse_cdf = build_prior(min_log_nhi, max_log_nhi, alpha)

    log_nhi_samples = inverse_cdf(halton[:, 0])
    offset_samples = halton[:, 1]
    return {
        "offset_samples": offset_samples,
        "log_nhi_samples": log_nhi_samples,
        "nhi_samples": 10.0**log_nhi_samples,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - tooling convenience
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument(
        "--log-nhi", type=float, nargs=2, metavar=("MIN", "MAX"), required=True
    )
    parser.add_argument("--alpha", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--production-target",
        default=None,
        help="Filename of the deployed grid this is intended to reproduce.",
    )
    args = parser.parse_args()

    samples = generate(args.num_samples, tuple(args.log_nhi), args.alpha, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / f"{args.name}.npz"
    # All three arrays are stored, including nhi_samples, even though it equals
    # 10**log_nhi_samples. Deriving it on load makes the asset platform-dependent:
    # float64 `**` uses the platform libm `pow`, which is not correctly rounded,
    # so macOS/arm64 and Linux/x86-64 disagree in the last bits. Storing keeps the
    # asset bit-reproducible everywhere, and matches the reference generator,
    # whose save_samples_to_mat also writes nhi_samples.
    np.savez_compressed(
        npz_path,
        offset_samples=samples["offset_samples"],
        log_nhi_samples=samples["log_nhi_samples"],
        nhi_samples=samples["nhi_samples"],
    )

    provenance = {
        "name": args.name,
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "identity": {
            # Say plainly what this is and is not.
            "status": "regenerated, production-array identity unverified",
            "production_target": args.production_target,
            "explanation": (
                "Generated by this builder with an explicit seed. The historical "
                "grids were produced by a notebook that created a scrambled Halton "
                "engine without a seed and then called numpy.random.seed(42), "
                "which does not seed a SciPy QMC engine. Byte identity with the "
                "deployed grid is therefore NOT established, and this grid must "
                "not pass a production-equivalence, population, or release gate "
                "until the deployed arrays or their hashes are compared directly."
            ),
        },
        "prior": {
            "family": "Prochaska et al. (2014) CDDF spline, mixed with uniform",
            "reference": "arXiv:1402.0548",
            "support_log_nhi": list(args.log_nhi),
            "mixture_weight_pw14": args.alpha,
            "mixture_weight_uniform": 1.0 - args.alpha,
            "spline_nodes_log_nhi": LOGNHI_NODES.tolist(),
            "spline_values_log_f": LOGF_NODES.tolist(),
            "spline_kind": "PchipInterpolator on (log N_HI, log10 f), clipped to nodes",
            "transformation_to_log_nhi": "p(logN) proportional to f(N) * N * ln(10)",
            "cdf_grid_points": CDF_GRID_POINTS,
            "inverse_cdf": "linear interp1d on the cumulative sum, clipped at edges",
        },
        "qmc": {
            "engine": "scipy.stats.qmc.Halton",
            "dimensions": 2,
            "dimension_order": list(HALTON_DIMENSION_ORDER),
            "scramble": True,
            "seed": args.seed,
            "num_samples": args.num_samples,
        },
        "environment": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "builder_commit": _git_commit(),
            "builder": "tools/build_sample_grid.py",
        },
        "arrays": {
            "offset_samples": {
                "shape": [args.num_samples],
                "sha256_float64": sha256_array(samples["offset_samples"]),
            },
            "log_nhi_samples": {
                "shape": [args.num_samples],
                "sha256_float64": sha256_array(samples["log_nhi_samples"]),
            },
            "nhi_samples": {
                "shape": [args.num_samples],
                "sha256_float64": sha256_array(samples["nhi_samples"]),
                "note": (
                    "stored, not derived on load: 10**x via the platform libm is "
                    "not bit-reproducible across architectures"
                ),
            },
        },
    }
    provenance["sha256"] = sha256_file(npz_path)
    (args.out / f"{args.name}.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {npz_path}  ({npz_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  status: {provenance['identity']['status']}")
    for key, info in provenance["arrays"].items():
        print(f"  {key:18s} sha256 {info['sha256_float64'][:16]}…")


if __name__ == "__main__":
    main()
