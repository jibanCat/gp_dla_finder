"""Exact versus FILTER: what the deployed approximation costs.

FILTER must be implemented and compared against the working
exact integral before v0.1 behaviour is frozen, over a corpus spanning redshift,
signal to noise, usable-pixel fraction, masking pattern and pixel grid, and
including absorber-free, weak, strong and high-column systems.

What FILTER actually is, at one absorber
----------------------------------------
Verified against the reference (``tests/test_reference_parity.py::
test_filter_mode_reproduces_the_reference_filter_evidence``, bitwise): the
reference's FILTER=1 one-absorber evidence is the *same* quasi-Monte-Carlo
estimator restricted to the first ``max(N // 20, 5000)`` samples. Its adaptive
region-A refinement never reaches the one-absorber evidence, because "FILTER fix
#5" discards the refined samples at ``k = 0``.

Two consequences worth stating plainly:

* the speed-up is not 20x at every operating point. At ``N = 10000`` the floor of
  5000 makes it 2x; only at ``N = 100000`` is it 20x;
* the approximation is a *prefix* of a quasi-Monte-Carlo sequence, not a random
  subsample. A Halton prefix is itself low-discrepancy, which is why the error is
  far smaller than a random 5000-of-100000 subsample would give -- but it is also
  why the error is deterministic rather than averaging away.

Reported per case
-----------------
evidence difference, log-Bayes-factor difference, absorber-posterior difference
under a named prior, whether a detection classification flips at each of several
thresholds, the shift in the best sample-grid point, and wall time.

Usage::

    python tools/compare_filter.py --samples 10000 --json filter_10k.json
    python tools/compare_filter.py --samples 100000 --cases classical-dla-mid-z
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from gp_dla_finder import load_model, load_prior, load_sample_grid  # noqa: E402
from gp_dla_finder.config import Config  # noqa: E402
from gp_dla_finder.gp.evidence import (  # noqa: E402
    absorber_search_window,
    assemble_model,
    coarse_scan_size,
    null_log_evidence,
    one_absorber_log_evidence,
)
from gp_dla_finder.gp.spectrum import InsufficientData, prepare_spectrum  # noqa: E402
from synthetic import CORPUS, absorbers_of, build  # noqa: E402

_GRID_FOR_SAMPLES = {
    10_000: "pw14_172_225_10000",
    50_000: "pw14_172_225_50000",
    100_000: "pw14_172_225_100000",
}

#: Detection thresholds a catalogue might cut on. A flip at any of these is a
#: scientific difference, not a rounding difference.
THRESHOLDS = (0.5, 0.9, 0.98)


def absorber_posterior(
    log_evidence_null: float,
    log_evidence_absorber: float,
    prior,
    config: Config,
    z_qso: float,
) -> float:
    """Two-model posterior probability of one absorber, under a named prior."""
    log_prior_absorber = float(
        prior.log_priors(z_qso, 1, config.prior_z_qso_increase)[0]
    )
    log_prior_null = prior.log_prior_no_absorber(z_qso, config.prior_z_qso_increase)
    log_joint = np.array(
        [
            log_prior_null + log_evidence_null,
            log_prior_absorber + log_evidence_absorber,
        ]
    )
    log_joint -= log_joint.max()
    weights = np.exp(log_joint)
    return float(weights[1] / weights.sum())


def best_grid_point(
    samples: np.ndarray, z_samples: np.ndarray, nhi_samples: np.ndarray
):
    """Highest-likelihood grid point among the samples that were evaluated."""
    finite = np.isfinite(samples)
    if not np.any(finite):
        return None, None
    index = int(np.nanargmax(np.where(finite, samples, -np.inf)))
    return float(z_samples[index]), float(np.log10(nhi_samples[index]))


def compare_case(case, *, samples: int, prior, model) -> dict:
    config = Config.desi_y3().replace(
        num_samples=samples, sample_grid=_GRID_FOR_SAMPLES[samples]
    )
    grid = load_sample_grid(config.sample_grid)
    spectrum = build(case)

    try:
        prepared = prepare_spectrum(spectrum, model, config)
    except InsufficientData as exc:
        return {"case": case.name, "status": "insufficient_data", "reason": exc.reason}

    assembled = assemble_model(prepared, model, config)
    null = null_log_evidence(prepared, assembled)

    z_min, z_max = absorber_search_window(prepared, config)
    z_samples = grid.sample_redshifts(z_min, z_max)

    results = {}
    for mode in ("exact", "filter"):
        start = time.perf_counter()
        evidence, per_sample = one_absorber_log_evidence(
            prepared, assembled, grid, config, mode=mode, return_samples=True
        )
        elapsed = time.perf_counter() - start
        grid_z, grid_log_nhi = best_grid_point(per_sample, z_samples, grid.nhi_samples)
        results[mode] = {
            "log_evidence": evidence,
            "log_bayes_factor": evidence - null,
            "posterior": absorber_posterior(
                null, evidence, prior, config, prepared.z_qso
            ),
            "grid_z_abs": grid_z,
            "grid_log_nhi": grid_log_nhi,
            "seconds": elapsed,
            "n_evaluated": int(np.count_nonzero(np.isfinite(per_sample))),
        }

    exact, filtered = results["exact"], results["filter"]
    flips = {
        f"p>{threshold}": bool(
            (exact["posterior"] >= threshold) != (filtered["posterior"] >= threshold)
        )
        for threshold in THRESHOLDS
    }

    return {
        "case": case.name,
        "status": "completed",
        "truth_absorbers": absorbers_of(case),
        "z_qso": case.z_qso,
        "pixel_scale": case.pixel_scale,
        "ivar": case.ivar,
        "masked_fraction": case.masked_fraction,
        "n_usable_pixels": prepared.n_pixels,
        "null_log_evidence": null,
        "exact": exact,
        "filter": filtered,
        "delta_log_evidence": filtered["log_evidence"] - exact["log_evidence"],
        "delta_log_bayes_factor": (
            filtered["log_bayes_factor"] - exact["log_bayes_factor"]
        ),
        "delta_posterior": filtered["posterior"] - exact["posterior"],
        "delta_grid_z_abs": (
            None
            if exact["grid_z_abs"] is None or filtered["grid_z_abs"] is None
            else filtered["grid_z_abs"] - exact["grid_z_abs"]
        ),
        "delta_grid_log_nhi": (
            None
            if exact["grid_log_nhi"] is None or filtered["grid_log_nhi"] is None
            else filtered["grid_log_nhi"] - exact["grid_log_nhi"]
        ),
        "classification_flips": flips,
        "speedup": exact["seconds"] / filtered["seconds"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples", type=int, default=10_000, choices=sorted(_GRID_FOR_SAMPLES)
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="case names (default: the whole corpus)",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    cases = CORPUS
    if args.cases:
        wanted = set(args.cases)
        cases = tuple(case for case in CORPUS if case.name in wanted)
        missing = wanted - {case.name for case in cases}
        if missing:
            parser.error(f"unknown cases: {sorted(missing)}")

    model = load_model()
    prior = load_prior()
    config = Config.desi_y3().replace(num_samples=args.samples)

    print(f"operating point   {args.samples} QMC samples")
    print(
        f"coarse scan       {coarse_scan_size(config)} samples "
        f"({args.samples / coarse_scan_size(config):.1f}x fewer)"
    )
    print(f"prior             {prior}")
    print(f"corpus            {len(cases)} generated cases\n")

    header = (
        f"{'case':26s} {'dlogZ':>9s} {'dlogBF':>9s} {'p_exact':>8s} "
        f"{'p_filter':>9s} {'dp':>9s} {'dz_grid':>8s} {'dlogN':>7s} {'flip':>5s} "
        f"{'speedup':>8s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for case in cases:
        record = compare_case(case, samples=args.samples, prior=prior, model=model)
        records.append(record)
        if record["status"] != "completed":
            print(f"{record['case']:26s} {record['status']} ({record['reason']})")
            continue
        flipped = any(record["classification_flips"].values())
        dz = record["delta_grid_z_abs"]
        dn = record["delta_grid_log_nhi"]
        print(
            f"{record['case']:26s} "
            f"{record['delta_log_evidence']:+9.4f} "
            f"{record['delta_log_bayes_factor']:+9.4f} "
            f"{record['exact']['posterior']:8.5f} "
            f"{record['filter']['posterior']:9.5f} "
            f"{record['delta_posterior']:+9.2e} "
            f"{(0.0 if dz is None else dz):+8.4f} "
            f"{(0.0 if dn is None else dn):+7.3f} "
            f"{'YES' if flipped else '-':>5s} "
            f"{record['speedup']:7.2f}x"
        )

    completed = [r for r in records if r["status"] == "completed"]
    if completed:
        worst_evidence = max(abs(r["delta_log_evidence"]) for r in completed)
        worst_posterior = max(abs(r["delta_posterior"]) for r in completed)
        flips = sum(any(r["classification_flips"].values()) for r in completed)
        print(
            f"\nworst |d log Z| {worst_evidence:.4f} nat   "
            f"worst |d p_absorber| {worst_posterior:.3e}   "
            f"classification flips {flips}/{len(completed)}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {"num_samples": args.samples, "results": records},
                indent=2,
                default=float,
            )
        )
        print(f"\nrecord written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
