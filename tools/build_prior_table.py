#!/usr/bin/env python3
"""Build the compact absorber-existence prior table.

Why a table
-----------
The reference pipeline computes the absorber-existence prior from ~115 MB of SDSS
catalogues, but the inference path touches them through exactly one call:

    num_absorbers, num_quasars = prior.less_ind(z_qso)

which counts sightlines with ``z_qso_i < z + delta`` and how many of them host a
DLA. That is a monotone step function of one scalar, so it can be stored exactly
as a sorted redshift array plus a cumulative count -- no interpolation, no
tolerance, no catalogue.

Sources (public)
----------------
* ``catalog.mat`` from https://github.com/jibanCat/gp_dr12_trained (MIT).
  Supplies ``thing_ids``, ``z_qsos``, ``in_dr9``, ``filter_flags``.
* ``BOSSLyaDR9_cat.txt`` from the public SDSS DR9 BOSS Lyman-alpha forest catalogue,
  http://data.sdss3.org/sas/dr9/boss/lya/cat/. The concordance line-of-sight and
  DLA lists are derived from it with the column selection documented in the
  reference pipeline's ``download_catalogs.sh``:
      dla_catalog:  rows with column 15 > 0  ->  (col 4, col 15, col 16)
      los_catalog:  all rows                 ->  (col 4)

Selection, reproduced from ``model_priors.PriorCatalog``
--------------------------------------------------------
1. ``prior_ind = in_dr9 & los_ind & (filter_flags == 0)``
2. DLAs whose Lyman-alpha lands blueward of the quasar's Lyman limit are dropped.

Usage
-----
    python tools/build_prior_table.py --catalog catalog.mat \\
        --lya-cat BOSSLyaDR9_cat.txt --name dr9q_concordance
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

LYA_WAVELENGTH = 1215.6701
LYMAN_LIMIT = 911.7633

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "src" / "gp_dla_finder" / "data" / "priors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lya_catalogue(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the concordance line-of-sight and DLA lists from the SDSS DR9 file.

    Mirrors the two gawk selections in the reference ``download_catalogs.sh``:
    1-based columns 4 (THING_ID), 15 (z_DLA) and 16 (log N_HI), skipping the
    header, keeping DLA rows only where column 15 is positive.
    """
    thing_ids_los: list[int] = []
    dla_rows: list[tuple[int, float, float]] = []

    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 16:
                continue
            thing_id = int(fields[3])
            thing_ids_los.append(thing_id)
            z_dla = float(fields[14])
            if z_dla > 0:
                dla_rows.append((thing_id, z_dla, float(fields[15])))

    los = np.array(thing_ids_los, dtype=np.int64)
    if dla_rows:
        ids, z, n = zip(*dla_rows, strict=True)
        return los, np.array(ids, dtype=np.int64), np.array(z), np.array(n)
    return los, np.empty(0, np.int64), np.empty(0), np.empty(0)


def build_selection(catalog_path: Path, lya_path: Path):
    """Reproduce ``PriorCatalog``'s selected (z_qso, is_dla) sample."""
    import h5py

    with h5py.File(catalog_path, "r") as catalog:
        in_dr9 = catalog["in_dr9"][0, :].astype(np.bool_)
        z_qsos = catalog["z_qsos"][0, :]
        filter_flags = catalog["filter_flags"][0, :]
        thing_ids = catalog["thing_ids"][0, :].astype(np.int64)

    ids_los, ids_dla, z_dlas_raw, _ = read_lya_catalogue(lya_path)

    los_ind = np.isin(thing_ids, ids_los)
    dla_ind = np.isin(thing_ids, ids_dla)

    z_dlas = np.full(dla_ind.shape, np.nan)
    z_dlas[dla_ind] = z_dlas_raw[np.isin(ids_dla, thing_ids)]

    prior_ind = in_dr9 & los_ind & (filter_flags == 0)
    z_qsos = z_qsos[prior_ind]
    dla_ind = dla_ind[prior_ind]
    z_dlas = z_dlas[prior_ind]

    # Drop DLAs whose Lyman-alpha falls blueward of the quasar's Lyman limit.
    selected = np.where(dla_ind)[0]
    below_limit = LYA_WAVELENGTH * (1 + z_dlas[dla_ind]) < LYMAN_LIMIT * (
        1 + z_qsos[dla_ind]
    )
    dla_ind[selected[below_limit]] = False

    return z_qsos, dla_ind


def less_ind_reference(z_qsos, dla_ind, z, delta):
    """Brute-force ``less_ind``: the definition, evaluated directly."""
    z = max(z, float(z_qsos.min()))
    mask = z_qsos < (z + delta)
    return int(dla_ind[mask].sum()), int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--lya-cat", type=Path, required=True)
    parser.add_argument("--name", default="dr9q_concordance")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    z_qsos, dla_ind = build_selection(args.catalog, args.lya_cat)
    order = np.argsort(z_qsos, kind="stable")
    z_sorted = np.ascontiguousarray(z_qsos[order])
    cum_dla = np.cumsum(dla_ind[order]).astype(np.int32)

    print(f"selected sightlines: {z_sorted.size}   hosting a DLA: {int(cum_dla[-1])}")
    print(f"z_qso range: [{z_sorted[0]:.4f}, {z_sorted[-1]:.4f}]")

    # ---- equivalence proof: every breakpoint plus a dense grid ----------------
    delta = (30000.0 * 1000) / 299792458.0
    probes = np.unique(
        np.concatenate(
            [
                z_sorted,  # every breakpoint
                np.nextafter(z_sorted, -np.inf),  # just below each
                np.nextafter(z_sorted, np.inf),  # just above each
                z_sorted - delta,  # breakpoints of the shifted comparison
                np.linspace(0.0, 8.0, 200_001),  # dense sweep
            ]
        )
    )
    lo = np.maximum(probes, z_sorted[0])
    idx = np.searchsorted(z_sorted, lo + delta, side="left")
    table_quasars = idx
    table_dlas = np.where(idx > 0, cum_dla[np.maximum(idx - 1, 0)], 0)

    ref_dlas = np.empty_like(table_dlas)
    ref_quasars = np.empty_like(table_quasars)
    for i, z in enumerate(probes):
        ref_dlas[i], ref_quasars[i] = less_ind_reference(
            z_sorted, dla_ind[order], z, delta
        )

    if not (
        np.array_equal(table_dlas, ref_dlas)
        and np.array_equal(table_quasars, ref_quasars)
    ):
        raise SystemExit("table does NOT reproduce less_ind exactly")
    print(f"equivalence: exact at all {probes.size} probe redshifts")

    # ---- write ---------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / f"{args.name}.npz"
    np.savez_compressed(npz_path, z_qsos=z_sorted, cumulative_absorbers=cum_dla)

    provenance = {
        "name": args.name,
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "Absorber-existence prior as an exact step table: sorted quasar "
            "redshifts and the cumulative count of those hosting a DLA."
        ),
        "sources": [
            {
                "role": "quasar catalogue",
                "filename": args.catalog.name,
                "sha256": sha256_file(args.catalog),
                "origin": "https://github.com/jibanCat/gp_dr12_trained",
                "licence": "MIT",
                "attribution": "Ming-Feng Ho, gp_dr12_trained",
            },
            {
                "role": "DLA / line-of-sight catalogue",
                "filename": args.lya_cat.name,
                "sha256": sha256_file(args.lya_cat),
                "origin": (
                    "http://data.sdss3.org/sas/dr9/boss/lya/cat/BOSSLyaDR9_cat.txt"
                ),
                "licence": "public SDSS-III DR9 data",
                "attribution": "SDSS-III BOSS DR9 Lyman-alpha forest catalogue",
            },
        ],
        "selection": (
            "in_dr9 & los_ind & (filter_flags == 0), then DLAs with "
            "lya*(1+z_dla) < lyman_limit*(1+z_qso) removed"
        ),
        "n_sightlines": int(z_sorted.size),
        "n_with_absorber": int(cum_dla[-1]),
        "z_qso_min": float(z_sorted[0]),
        "z_qso_max": float(z_sorted[-1]),
        "equivalence_proof": {
            "method": (
                "compared against a brute-force evaluation of the definition at "
                "every breakpoint, the floats either side of each breakpoint, the "
                "breakpoints of the shifted comparison, and a dense sweep"
            ),
            "n_probe_redshifts": int(probes.size),
            "result": "exact",
        },
        "builder": "tools/build_prior_table.py",
    }
    provenance["sha256"] = sha256_file(npz_path)
    (args.out / f"{args.name}.json").write_text(json.dumps(provenance, indent=2) + "\n")

    src_mb = (args.catalog.stat().st_size + args.lya_cat.stat().st_size) / 1e6
    print(
        f"wrote {npz_path}  "
        f"({src_mb:.1f} MB of sources -> {npz_path.stat().st_size / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
