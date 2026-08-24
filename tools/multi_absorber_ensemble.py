"""Injection/recovery ensemble for the bounded M0/M1/M2 path.

The eight-seed result in an earlier increment varied only the RESAMPLER seed on
one fixed spectrum, and showed the SIR draw had collapsed there. It said nothing
about behaviour across spectra. This varies the noise realisation as well as the
injected population, which is what "how often does it get the multiplicity
right" actually needs.

Everything is generated from named seeds in this repository. No private mock
spectrum or reconstructable derivative is involved.

    python tools/multi_absorber_ensemble.py --realisations 12 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@dataclass(frozen=True)
class Injection:
    """One named point in the ensemble."""

    name: str
    absorbers: tuple[tuple[float, float], ...]
    ivar: float
    truth: int  # how many absorbers were injected


#: Spans what the ruling asked to vary: redshift, column density, separation,
#: signal to noise, and strong/weak pairing.
INJECTIONS: tuple[Injection, ...] = (
    Injection("none", (), 25.0, 0),
    Injection("none-low-snr", (), 1.0, 0),
    Injection("one-mid", ((2.55, 20.5),), 25.0, 1),
    Injection("one-weak", ((2.40, 20.0),), 25.0, 1),
    Injection("one-low-snr", ((2.55, 20.5),), 1.0, 1),
    Injection("two-wide", ((2.20, 20.5), (2.70, 20.6)), 25.0, 2),
    Injection("two-medium", ((2.30, 20.5), (2.55, 20.4)), 25.0, 2),
    Injection("two-close", ((2.45, 20.5), (2.47, 20.4)), 25.0, 2),
    Injection("strong-plus-weak", ((2.30, 21.0), (2.65, 19.2)), 25.0, 2),
    Injection("two-low-snr", ((2.20, 20.5), (2.70, 20.6)), 1.0, 2),
)


def _spectrum(injection: Injection, seed: int):
    """One realisation: same injection, different noise."""
    from synthetic import Case, build

    return build(
        Case(
            name=f"{injection.name}-{seed}",
            z_qso=2.9,
            wave_start=3600.0,
            wave_stop=6000.0,
            pixel_scale=0.8,
            ivar=injection.ivar,
            masked_fraction=0.02,
            absorber=injection.absorbers or None,
            seed=seed,
        )
    )


def run(realisations: int, base_seed: int) -> dict:
    from gp_dla_finder import Config
    from gp_dla_finder.finder import Finder

    config = Config.desi_y3_fast(
        enable_tau_eb=False, max_absorbers=2, experimental_multi_absorber=True
    )
    finder = Finder(config, warn_about_threads=False)

    records = []
    for injection in INJECTIONS:
        for index in range(realisations):
            seed = base_seed + 1000 * index
            result = finder.run(_spectrum(injection, seed), targetid=index)
            if result.status != "completed" or result.ladder is None:
                records.append(
                    {
                        "injection": injection.name,
                        "seed": seed,
                        "status": result.status,
                    }
                )
                continue
            ladder = result.ladder
            records.append(
                {
                    "injection": injection.name,
                    "truth": injection.truth,
                    "seed": seed,
                    "selected": ladder.selected_model,
                    "log_evidences": list(ladder.log_evidences),
                    "posteriors": list(ladder.model_posteriors),
                    "p_absorber": ladder.p_absorber,
                    "candidates": [
                        [c.grid_z_abs, c.grid_log_nhi]
                        for c in result.absorber_candidates
                    ],
                }
            )
    return {"realisations": realisations, "base_seed": base_seed, "records": records}


def summarise(record_set: dict) -> None:
    records = [r for r in record_set["records"] if "selected" in r]
    print(
        f"\n{'injection':18s} {'truth':>5s} {'n':>3s} "
        f"{'M0':>5s} {'M1':>5s} {'M2':>5s}  {'exact':>6s}"
    )
    for injection in INJECTIONS:
        rows = [r for r in records if r["injection"] == injection.name]
        if not rows:
            continue
        counts = [sum(1 for r in rows if r["selected"] == k) for k in (0, 1, 2)]
        exact = counts[injection.truth] / len(rows)
        print(
            f"{injection.name:18s} {injection.truth:>5d} {len(rows):>3d} "
            f"{counts[0]:>5d} {counts[1]:>5d} {counts[2]:>5d}  {exact:>6.0%}"
        )

    overall = sum(1 for r in records if r["selected"] == r["truth"]) / len(records)
    print(f"\nexact multiplicity recovered: {overall:.0%} of {len(records)} spectra")

    # The two failure modes worth naming separately.
    over = sum(1 for r in records if r["selected"] > r["truth"])
    under = sum(1 for r in records if r["selected"] < r["truth"])
    print(f"  over-counted  {over:>3d}   under-counted {under:>3d}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realisations", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=90001)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    record_set = run(args.realisations, args.base_seed)
    summarise(record_set)
    if args.json:
        args.json.write_text(json.dumps(record_set, indent=1) + "\n")
        print(f"\nrecord written to {args.json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
