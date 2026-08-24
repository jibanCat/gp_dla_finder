"""Reproducible benchmark for the evidence path.

Retained rather than ad hoc, because a performance number without its
environment is not a baseline. Every
run records: the CPU, the thread settings that were actually in force, the number
of warm-up and measured repetitions, median and spread rather than a single
timing, the spectrum and model dimensions, the Voigt backend, the package commit,
and the exact command that produced the record.

Measurement boundaries are named and reported separately, because "3 seconds"
means nothing without knowing what was inside it:

``prepare``
    validation, normalisation, masking, padding.
``assemble``
    interpolating the trained model and applying the forest terms.
``null``
    one low-rank log density.
``one_absorber``
    the full quasi-Monte-Carlo integral over the sample grid: N Voigt profiles
    and N low-rank log densities. This dominates, and is what a throughput
    estimate should be based on.
``voigt_component`` / ``likelihood_component``
    the two halves of a *single* absorber sample, timed separately in a
    controlled loop. These are diagnostic: they answer "where does the time go",
    and their sum is deliberately not forced to equal ``one_absorber`` per
    sample, since timing them apart changes cache behaviour.

Examples
--------
Default (10k samples, NumPy backend)::

    python tools/benchmark.py

Compare backends at the deployed operating point, JSON to a file::

    python tools/benchmark.py --samples 50000 --backend numpy libcerf \\
        --repeats 5 --json bench.json

Thread settings are read, not set: run under ``OMP_NUM_THREADS=1`` (etc.) if you
want a single-core number, and the record will say so.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_dla_finder import load_model, load_sample_grid  # noqa: E402
from gp_dla_finder import voigt as voigt_module  # noqa: E402
from gp_dla_finder.config import Config  # noqa: E402
from gp_dla_finder.gp import evidence as evidence_module  # noqa: E402
from gp_dla_finder.gp.evidence import (  # noqa: E402
    absorber_search_window,
    assemble_model,
    coarse_scan_size,
    null_log_evidence,
    one_absorber_log_evidence,
)
from gp_dla_finder.gp.likelihood import log_mvnpdf_low_rank  # noqa: E402
from gp_dla_finder.gp.spectrum import Spectrum, prepare_spectrum  # noqa: E402

#: Thread-count variables that actually change BLAS/OpenMP behaviour. Recorded
#: as observed; the harness never sets them, so a record always describes the
#: environment the operator chose.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: Sample counts with packaged grids.
_GRID_FOR_SAMPLES = {
    10_000: "pw14_172_225_10000",
    50_000: "pw14_172_225_50000",
    100_000: "pw14_172_225_100000",
}


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover - environment probing only
        pass
    return platform.processor() or "unknown"


def _git_commit(root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:  # pragma: no cover
        return "unknown"


def _blas_info() -> str:
    try:
        config = np.show_config(mode="dicts")
        build = config.get("Build Dependencies", {})
        blas = build.get("blas", {})
        return f"{blas.get('name', '?')} {blas.get('version', '?')}"
    except Exception:  # pragma: no cover
        return "unknown"


def make_spectrum(*, z_qso: float, pixel_scale: float, seed: int) -> Spectrum:
    """The benchmark spectrum. Deterministic, and described in the record."""
    wave = np.arange(3600.0, 5600.0, pixel_scale)
    rng = np.random.default_rng(seed)
    flux = 1.0 + 0.3 * np.sin(wave / 180.0) + rng.normal(0, 0.2, wave.size)
    mask = np.zeros_like(wave, dtype=bool)
    mask[500:520] = True
    return Spectrum(
        wavelength=wave,
        flux=flux,
        ivar=np.full_like(wave, 25.0),
        z_qso=z_qso,
        mask=mask,
    )


def _timed(function, *, warmup: int, repeats: int) -> dict[str, float]:
    """Run ``function``, reporting median and spread rather than one number."""
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "repeats": repeats,
        "warmup": warmup,
    }


def benchmark_backend(
    backend: str,
    *,
    samples: int,
    z_qso: float,
    pixel_scale: float,
    seed: int,
    warmup: int,
    repeats: int,
    component_repeats: int,
    mode: str,
) -> dict:
    model = load_model()
    # The backend is an ordinary configuration field, selected the way a caller
    # would select it. It used to be forced by monkeypatching the evidence
    # module's import, which meant the benchmark measured a path no user could
    # reach.
    config = Config.desi_y3().replace(
        num_samples=samples,
        sample_grid=_GRID_FOR_SAMPLES[samples],
        voigt_backend=backend,
    )
    grid = load_sample_grid(config.sample_grid)
    spectrum = make_spectrum(z_qso=z_qso, pixel_scale=pixel_scale, seed=seed)

    prepared = prepare_spectrum(spectrum, model, config)
    assembled = assemble_model(prepared, model, config)

    stages = {
        "prepare": _timed(
            lambda: prepare_spectrum(spectrum, model, config),
            warmup=warmup,
            repeats=repeats,
        ),
        "assemble": _timed(
            lambda: assemble_model(prepared, model, config),
            warmup=warmup,
            repeats=repeats,
        ),
        "null": _timed(
            lambda: null_log_evidence(prepared, assembled),
            warmup=warmup,
            repeats=repeats,
        ),
        "one_absorber": _timed(
            lambda: one_absorber_log_evidence(
                prepared, assembled, grid, config, mode=mode
            ),
            warmup=min(warmup, 1),
            repeats=repeats,
        ),
    }

    # --- component split, one absorber sample at a time -------------------
    z_min, z_max = absorber_search_window(prepared, config)
    z_samples = grid.sample_redshifts(z_min, z_max)
    nhi_samples = grid.nhi_samples
    base_variance = prepared.noise_variance
    profile = evidence_module._absorber_profile(
        prepared, config, z_samples[0], nhi_samples[0]
    )

    def voigt_component():
        for i in range(component_repeats):
            evidence_module._absorber_profile(
                prepared, config, z_samples[i], nhi_samples[i]
            )

    def likelihood_component():
        for _ in range(component_repeats):
            log_mvnpdf_low_rank(
                prepared.flux,
                assembled.mean * profile,
                assembled.factor * profile[:, None],
                assembled.absorption_variance * profile**2 + base_variance,
            )

    components = {
        "voigt_component": _timed(voigt_component, warmup=1, repeats=repeats),
        "likelihood_component": _timed(likelihood_component, warmup=1, repeats=repeats),
    }
    for record in components.values():
        record["per_call_ms"] = record["median_s"] / component_repeats * 1e3

    value = one_absorber_log_evidence(prepared, assembled, grid, config, mode=mode)

    # Per *evaluated* sample. Under FILTER only the coarse-scan prefix is
    # computed, so dividing by the grid size would report a per-sample cost that
    # no sample ever had, and would make the component shares exceed 100%.
    n_evaluated = coarse_scan_size(config) if mode == "filter" else samples
    per_sample_ms = stages["one_absorber"]["median_s"] / n_evaluated * 1e3
    voigt_share = (
        components["voigt_component"]["per_call_ms"] / per_sample_ms
        if per_sample_ms
        else float("nan")
    )

    return {
        "backend": backend,
        "n_evaluated_samples": n_evaluated,
        "backend_provenance": dict(voigt_module.backend_provenance(backend)),
        "num_samples": samples,
        "evidence_mode": mode,
        "stages": stages,
        "components": components,
        "per_sample_ms": per_sample_ms,
        "voigt_share_of_per_sample": voigt_share,
        "one_absorber_log_evidence": value,
        "dimensions": {
            "n_input_pixels": int(spectrum.wavelength.size),
            "n_usable_pixels": prepared.n_pixels,
            "n_padded_pixels": int(prepared.padded_wavelength.size),
            "model_rank": assembled.rank,
            "pixel_scale_angstrom": pixel_scale,
            "z_qso": z_qso,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        choices=sorted(_GRID_FOR_SAMPLES),
        help="QMC operating point (default: 10000)",
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        default=None,
        help="Voigt backends to benchmark (default: every available one)",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--component-repeats",
        type=int,
        default=200,
        help="absorber samples timed per component measurement",
    )
    parser.add_argument("--z-qso", type=float, default=2.6)
    parser.add_argument("--pixel-scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--mode",
        default="exact",
        choices=("exact", "filter"),
        help=(
            "evidence integral: 'exact' evaluates every sample and is the "
            "throughput number to quote; 'filter' evaluates the coarse-scan "
            "prefix, as deployed catalogue production does (default: exact)"
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    backends = args.backend or list(voigt_module.available_backends())
    for backend in backends:
        voigt_module.get_backend(backend)  # fail early on a typo

    root = Path(__file__).resolve().parents[1]
    import scipy

    record = {
        "environment": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "blas": _blas_info(),
            "thread_env": {
                name: os.environ.get(name, "(unset)") for name in _THREAD_VARS
            },
        },
        "package_commit": _git_commit(root),
        "command": " ".join([sys.executable.split("/")[-1], *sys.argv]),
        "available_backends": list(voigt_module.available_backends()),
        "evidence_mode": args.mode,
        "results": [],
    }

    for backend in backends:
        record["results"].append(
            benchmark_backend(
                backend,
                samples=args.samples,
                z_qso=args.z_qso,
                pixel_scale=args.pixel_scale,
                seed=args.seed,
                warmup=args.warmup,
                repeats=args.repeats,
                component_repeats=args.component_repeats,
                mode=args.mode,
            )
        )

    environment = record["environment"]
    print(
        f"cpu              {environment['cpu']} ({environment['logical_cpus']} logical)"
    )
    print(f"platform         {environment['platform']}")
    print(
        f"python/numpy/scipy {environment['python']} / {environment['numpy']} / "
        f"{environment['scipy']}"
    )
    print(f"blas             {environment['blas']}")
    print(
        "threads          "
        + ", ".join(f"{k}={v}" for k, v in environment["thread_env"].items())
    )
    print(f"commit           {record['package_commit']}")
    print(f"command          {record['command']}")
    first = record["results"][0]["dimensions"]
    print(
        f"spectrum         {first['n_input_pixels']} input -> "
        f"{first['n_usable_pixels']} usable ({first['n_padded_pixels']} padded), "
        f"rank {first['model_rank']}, {first['pixel_scale_angstrom']} A/pixel, "
        f"z_qso {first['z_qso']}"
    )
    print(f"operating point  {args.samples} QMC samples, mode={args.mode}")
    print(
        f"statistic        median of {args.repeats} repeats "
        f"after {args.warmup} warm-up\n"
    )

    header = (
        f"{'backend':10s} {'one_absorber':>13s} {'stdev':>8s} {'per sample':>11s} "
        f"{'voigt':>9s} {'likelihood':>11s} {'voigt share':>12s}"
    )
    print(header)
    print("-" * len(header))
    for result in record["results"]:
        print(
            f"{result['backend']:10s} "
            f"{result['stages']['one_absorber']['median_s']:11.3f} s "
            f"{result['stages']['one_absorber']['stdev_s']:7.3f}s "
            f"{result['per_sample_ms']:9.4f} ms "
            f"{result['components']['voigt_component']['per_call_ms']:7.4f}ms "
            f"{result['components']['likelihood_component']['per_call_ms']:9.4f}ms "
            f"{100 * result['voigt_share_of_per_sample']:11.1f}%"
        )

    print("\nstage medians (s):")
    for result in record["results"]:
        stages = ", ".join(
            f"{name}={values['median_s']:.4g}"
            for name, values in result["stages"].items()
        )
        print(f"  {result['backend']:10s} {stages}")

    if args.json:
        args.json.write_text(json.dumps(record, indent=2, default=float))
        print(f"\nrecord written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
