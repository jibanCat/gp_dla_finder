"""Compare two libcerf builds against each other, directly.

An earlier measurement reported that a
source-built and a pre-built libcerf each differed from the NumPy backend by
exactly ``9.439671266875393e-14``, and concluded they were interchangeable. That
inference is not sound: two profiles can share a maximum deviation from a third
without being equal to each other. This compares them **directly**, element by
element, on the profile and on the end-to-end evidence.

It also serves the version-detection fix: each build reports the version and
provenance resolved from *its own* prefix, not from whatever libcerf happens to
sit on the default pkg-config path.

Usage::

    # build the source variant first
    tools/build_libcerf.sh
    python tools/compare_libcerf_builds.py \\
        --prefix-a /opt/homebrew \\
        --prefix-b build/libcerf-2.4-prefix

Each prefix is built into a separate temporary copy of the package tree, so the
two extensions never overwrite one another and no stale artefact can be linked by
accident.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Redshift and column density spanning the sample grid, saturated regime included.
GRID = [(z, n) for z in (2.0, 2.6, 3.4) for n in (17.2, 19.0, 20.3, 22.0, 23.0)]

_PROBE_SCRIPT = r"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from gp_dla_finder import voigt as V

if "libcerf" not in V.available_backends():
    print(json.dumps({"error": "libcerf backend not built",
                      "rejections": dict(V.backend_rejections())}))
    raise SystemExit(1)

probe = np.linspace(3600.0, 6000.0, 512)
grid = json.loads(sys.argv[1])
profiles = {}
for z_dla, log_nhi in grid:
    key = f"{z_dla}_{log_nhi}"
    raw = V.voigt_absorption(probe, nhi=10.0 ** log_nhi, z_dla=z_dla,
                             num_lines=3, broadening=False, backend="libcerf")
    profiles[key] = raw.tobytes().hex()

record = {
    "provenance": {k: v for k, v in V.backend_provenance("libcerf").items()},
    "profiles": profiles,
}

# End-to-end evidence, the quantity a user actually reads.
sys.path.insert(0, "tests")
from synthetic import make_spectrum
from gp_dla_finder import load_model, load_sample_grid
from gp_dla_finder.config import Config
from gp_dla_finder.gp.evidence import (assemble_model, null_log_evidence,
                                       one_absorber_log_evidence)
from gp_dla_finder.gp.spectrum import prepare_spectrum

model = load_model()
config = Config.desi_y3_fast().replace(voigt_backend="libcerf")
sample_grid = load_sample_grid(config.sample_grid)
prepared = prepare_spectrum(make_spectrum(), model, config)
assembled = assemble_model(prepared, model, config)
record["null_log_evidence"] = repr(null_log_evidence(prepared, assembled))
record["one_absorber_log_evidence"] = repr(
    one_absorber_log_evidence(prepared, assembled, sample_grid, config, mode="filter")
)
print(json.dumps(record))
"""


def build_and_probe(prefix: Path, label: str) -> dict:
    """Build the extension against ``prefix`` in an isolated tree, then probe it."""
    include = prefix / "include"
    library = prefix / "lib"
    if not (include / "cerf.h").is_file():
        raise SystemExit(f"{label}: no cerf.h under {include}")

    workdir = Path(tempfile.mkdtemp(prefix=f"gpdlf-{label}-"))
    tree = workdir / "pkg"
    shutil.copytree(
        ROOT,
        tree,
        ignore=shutil.ignore_patterns(
            ".git", "build", "*.egg-info", "__pycache__", "*.so", "*.dylib"
        ),
    )
    for stale in ("src/gp_dla_finder/_build_info.py", "src/gp_dla_finder/_voigt_ext.c"):
        (tree / stale).unlink(missing_ok=True)

    environment = {
        "GP_DLA_FINDER_LIBCERF_INCLUDE": str(include),
        "GP_DLA_FINDER_LIBCERF_LIB": str(library),
    }
    import os

    build = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=tree,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise SystemExit(f"{label}: build failed\n{build.stderr[-2000:]}")

    probe = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, json.dumps(GRID)],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"{label}: probe failed\n{probe.stdout}\n{probe.stderr[-2000:]}"
        )

    result = json.loads(probe.stdout.strip().splitlines()[-1])
    result["_tree"] = str(tree)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix-a", type=Path, required=True)
    parser.add_argument("--prefix-b", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--keep", action="store_true", help="keep the temporary build trees"
    )
    args = parser.parse_args(argv)

    a = build_and_probe(args.prefix_a.resolve(), "a")
    b = build_and_probe(args.prefix_b.resolve(), "b")

    print("build A")
    for key in (
        "libcerf_version",
        "libcerf_version_source",
        "libcerf_sha256",
        "libcerf_provenance",
        "libcerf_build_flags",
    ):
        print(f"  {key:24s} {a['provenance'].get(key)}")
    print("build B")
    for key in (
        "libcerf_version",
        "libcerf_version_source",
        "libcerf_sha256",
        "libcerf_provenance",
        "libcerf_build_flags",
    ):
        print(f"  {key:24s} {b['provenance'].get(key)}")

    same_binary = a["provenance"]["libcerf_sha256"] == b["provenance"]["libcerf_sha256"]
    print(f"\nsame library bytes: {same_binary}")

    import numpy as np

    identical, differing, worst = 0, 0, 0.0
    for key, hex_a in a["profiles"].items():
        arr_a = np.frombuffer(bytes.fromhex(hex_a), dtype=np.float64)
        arr_b = np.frombuffer(bytes.fromhex(b["profiles"][key]), dtype=np.float64)
        if np.array_equal(arr_a, arr_b):
            identical += 1
        else:
            differing += 1
            worst = max(worst, float(np.max(np.abs(arr_a - arr_b))))

    print(
        f"profiles bitwise identical: {identical}/{identical + differing}"
        + (f"   worst |A-B| = {worst:.3e}" if differing else "")
    )
    print(f"null log evidence     A {a['null_log_evidence']}")
    print(f"                      B {b['null_log_evidence']}")
    print(f"one-absorber evidence A {a['one_absorber_log_evidence']}")
    print(f"                      B {b['one_absorber_log_evidence']}")
    print(
        "evidence bitwise identical: "
        f"{a['one_absorber_log_evidence'] == b['one_absorber_log_evidence']}"
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "a": {k: v for k, v in a.items() if k != "profiles"},
                    "b": {k: v for k, v in b.items() if k != "profiles"},
                    "same_library_bytes": same_binary,
                    "profiles_identical": identical,
                    "profiles_differing": differing,
                    "worst_absolute_profile_difference": worst,
                },
                indent=2,
            )
        )
        print(f"\nrecord written to {args.json}")

    if not args.keep:
        for result in (a, b):
            shutil.rmtree(Path(result["_tree"]).parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
