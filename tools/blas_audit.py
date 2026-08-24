"""Is the GP likelihood actually BLAS-bound, and is the BLAS set up correctly?

Before proposing any change to the
likelihood algorithm, audit whether BLAS is configured and used correctly, and
repeat the bottleneck measurement under controlled conditions. Profile allocation
and matrix shapes as well as nominal FLOP rate.

What this reports
-----------------
``environment``
    The linked BLAS and LAPACK, the thread pools ``threadpoolctl`` can actually
    see, and the thread-count environment variables as observed. These are not
    the same thing, and the difference matters: on macOS the default BLAS is
    Apple Accelerate, which exposes no controllable pool, so setting
    ``VECLIB_MAXIMUM_THREADS`` may have no effect at all.

``threading``
    Measured, not assumed. CPU time divided by wall time during a tight loop is
    the honest answer to "how many threads is this really using": 1.0 means
    single-threaded whatever the environment claims.

``shapes``
    Every matrix operation in one likelihood evaluation, with its shape and FLOP
    count. This is where the answer usually is -- an operation can be at peak
    throughput and still be the bottleneck because it is called 100,000 times.

``allocation``
    Bytes allocated per likelihood call, via ``tracemalloc``. Temporaries are the
    usual reason a small-matrix routine falls short of its FLOP ceiling.

``rate``
    Achieved GFLOP/s against a measured single-core ceiling for the same dtype,
    so "is this slow?" gets a comparison rather than an adjective.

Usage::

    python tools/blas_audit.py
    python tools/blas_audit.py --json audit.json --repeats 4000
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
import tracemalloc
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from gp_dla_finder import load_model  # noqa: E402
from gp_dla_finder.config import Config  # noqa: E402
from gp_dla_finder.gp.evidence import assemble_model  # noqa: E402
from gp_dla_finder.gp.likelihood import log_mvnpdf_low_rank  # noqa: E402
from gp_dla_finder.gp.spectrum import prepare_spectrum  # noqa: E402
from synthetic import make_spectrum  # noqa: E402

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
    except Exception:  # pragma: no cover
        pass
    return platform.processor() or "unknown"


def _blas_identity() -> dict:
    record = {}
    try:
        config = np.show_config(mode="dicts") or {}
        for name, info in (config.get("Build Dependencies") or {}).items():
            if name in ("blas", "lapack"):
                record[name] = {
                    key: info.get(key)
                    for key in ("name", "version", "detection method")
                }
    except Exception:  # pragma: no cover
        record["error"] = "np.show_config(mode='dicts') unavailable"

    try:
        from threadpoolctl import threadpool_info

        record["threadpool_info"] = _sanitise_pool_records(threadpool_info())
    except ImportError:
        record["threadpool_info"] = "threadpoolctl not installed"
    return record


def _sanitise_pool_records(pools):
    """Keep a BLAS pool's identity, drop the absolute path that names the host.

    ``threadpoolctl`` reports ``filepath`` as an absolute library path, which
    on a venv install spells out the home directory and the repository. The
    library *file name* is the part that identifies the backend; the
    directories are provenance leakage. Everything
    reproducibility needs — internal API, version, threading layer, architecture,
    thread count — is untouched.
    """
    if not isinstance(pools, list):
        return pools
    cleaned = []
    for pool in pools:
        if not isinstance(pool, dict):
            cleaned.append(pool)
            continue
        entry = {k: v for k, v in pool.items() if k != "filepath"}
        path = pool.get("filepath")
        if isinstance(path, str) and path:
            entry["library_file"] = PurePosixPath(path).name
        cleaned.append(entry)
    return cleaned


def _cpu_over_wall(function, repeats: int) -> float:
    """Threads actually used, measured. 1.0 means single-threaded."""
    function()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for _ in range(repeats):
        function()
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return cpu / wall if wall > 0 else float("nan")


def _timed(function, repeats: int) -> float:
    """Median seconds per call over a few blocks, to resist a noisy machine."""
    function()
    blocks = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        blocks.append((time.perf_counter() - start) / repeats)
    return statistics.median(blocks)


def _array_layout(name: str, a: np.ndarray) -> dict:
    """Record what a benchmark operand actually is, not just its shape."""
    return {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "strides": list(a.strides),
        "c_contiguous": bool(a.flags["C_CONTIGUOUS"]),
        "f_contiguous": bool(a.flags["F_CONTIGUOUS"]),
        "owns_data": bool(a.flags["OWNDATA"]),
    }


def _generic_shape_microbenchmark(n: int, k: int, repeats: int) -> dict:
    """Bare matmuls on fresh C-contiguous random arrays of the right shape.

    Named for what it is. It is a *shape*
    microbenchmark and nothing more: the operands are freshly allocated, densely
    packed and cache-warm, which the likelihood's are not. It must not be
    presented as a floor the likelihood could reach.
    """
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, k))
    D_inv_M = rng.standard_normal((n, k))
    small = rng.standard_normal((k, k))
    wide = rng.standard_normal((k, n))

    seconds_nk2 = _timed(lambda: M.T @ D_inv_M, repeats)
    seconds_k2n = _timed(lambda: small @ wide, repeats)
    return {
        "matmul_n_k2_seconds": seconds_nk2,
        "matmul_n_k2_gflops": 2.0 * n * k * k / seconds_nk2 / 1e9,
        "matmul_k2_n_seconds": seconds_k2n,
        "matmul_k2_n_gflops": 2.0 * k * k * n / seconds_k2n / 1e9,
        "three_matmul_seconds": seconds_nk2 + 2 * seconds_k2n,
        "caveat": (
            "fresh C-contiguous random operands; not a floor for the likelihood"
        ),
    }


def _real_operand_microbenchmark(y, mu, M, d, repeats: int) -> dict:
    """The same three products on the likelihood's *actual* operands.

    Same arrays, same dtypes, same strides, same contiguity and ordering the
    likelihood produces -- including the LAPACK outputs and the transposed views
    it feeds to BLAS. The layouts are recorded so the measurement can be checked
    rather than trusted.
    """
    from scipy.linalg import lapack

    n, k = M.shape
    d_inv = 1 / d[:, None]
    D_inv_M = d_inv * M
    B = np.matmul(M.T, D_inv_M)
    B.ravel()[0 :: (k + 1)] += 1
    L = np.linalg.cholesky(B)
    L_inv = lapack.dtrtri(np.asfortranarray(L), lower=1)[0]
    U_inv = lapack.dtrtri(np.asfortranarray(L.T), lower=0)[0]
    tmp = np.matmul(L_inv, D_inv_M.T)

    seconds_b = _timed(lambda: np.matmul(M.T, D_inv_M), repeats)
    seconds_tmp = _timed(lambda: np.matmul(L_inv, D_inv_M.T), repeats)
    seconds_c = _timed(lambda: np.matmul(U_inv, tmp), repeats)

    return {
        "B_seconds": seconds_b,
        "B_gflops": 2.0 * n * k * k / seconds_b / 1e9,
        "tmp_seconds": seconds_tmp,
        "tmp_gflops": 2.0 * k * k * n / seconds_tmp / 1e9,
        "C_seconds": seconds_c,
        "C_gflops": 2.0 * k * k * n / seconds_c / 1e9,
        "three_matmul_seconds": seconds_b + seconds_tmp + seconds_c,
        "operand_layouts": [
            _array_layout("M", M),
            _array_layout("M.T", M.T),
            _array_layout("D_inv_M", D_inv_M),
            _array_layout("D_inv_M.T", D_inv_M.T),
            _array_layout("L_inv", L_inv),
            _array_layout("U_inv", U_inv),
            _array_layout("tmp", tmp),
        ],
    }


def _preallocation_comparison(y, mu, M, d, repeats: int) -> dict:
    """The rejected optimisation, retained so the rejection is checkable.

    Increment 9 reported that preallocating every buffer was slower and not
    bitwise. The ruling requires that claim be reproducible from repository
    tooling rather than quoted from a scratch script, so the experiment lives
    here.
    """
    from scipy.linalg import lapack

    from gp_dla_finder.gp.likelihood import _LOG_2PI

    n, k = M.shape
    buffers = {
        "resid": np.empty((n, 1)),
        "d_inv": np.empty((n, 1)),
        "D_inv_y": np.empty((n, 1)),
        "D_inv_M": np.empty((n, k)),
        "B": np.empty((k, k)),
        "tmp": np.empty((k, n)),
        "C": np.empty((k, n)),
        "Cy": np.empty((k, 1)),
        "DMCy": np.empty((n, 1)),
    }

    def buffered():
        b = buffers
        np.subtract(y[:, None], mu[:, None], out=b["resid"])
        np.divide(1.0, d[:, None], out=b["d_inv"])
        np.multiply(b["d_inv"], b["resid"], out=b["D_inv_y"])
        np.multiply(b["d_inv"], M, out=b["D_inv_M"])
        np.matmul(M.T, b["D_inv_M"], out=b["B"])
        b["B"].ravel()[0 :: (k + 1)] = b["B"].ravel()[0 :: (k + 1)] + 1
        L = np.linalg.cholesky(b["B"])
        np.matmul(
            lapack.dtrtri(np.asfortranarray(L), lower=1)[0],
            b["D_inv_M"].T,
            out=b["tmp"],
        )
        np.matmul(
            lapack.dtrtri(np.asfortranarray(L.T), lower=0)[0], b["tmp"], out=b["C"]
        )
        np.matmul(b["C"], b["resid"], out=b["Cy"])
        np.matmul(b["D_inv_M"], b["Cy"], out=b["DMCy"])
        K_inv_y = b["D_inv_y"] - b["DMCy"]
        log_det_K = np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(L)))
        return -0.5 * (
            np.matmul(b["resid"].T, K_inv_y).sum() + log_det_K + n * _LOG_2PI
        )

    reference = log_mvnpdf_low_rank(y, mu, M, d)
    candidate = buffered()
    current = _timed(lambda: log_mvnpdf_low_rank(y, mu, M, d), repeats)
    preallocated = _timed(buffered, repeats)

    return {
        "bitwise_identical": bool(reference == candidate),
        "absolute_difference": abs(float(reference) - float(candidate)),
        "current_seconds": current,
        "preallocated_seconds": preallocated,
        "speedup": current / preallocated,
        "verdict": (
            "rejected: not bitwise and not faster"
            if not (reference == candidate) or preallocated >= current
            else "candidate"
        ),
    }


def _operation_breakdown(y, mu, M, d, repeats: int) -> list[dict]:
    """Time each operation the likelihood performs, individually.

    Where the time actually goes, as opposed to where the FLOPs are. Retained
    because the thread-scaling conclusion rests on it.
    """
    from scipy.linalg import lapack

    n, k = M.shape
    resid = y[:, None] - mu[:, None]
    d_inv = 1 / d[:, None]
    D_inv_M = d_inv * M
    B = np.matmul(M.T, D_inv_M)
    B.ravel()[0 :: (k + 1)] += 1
    L = np.linalg.cholesky(B)
    L_inv = lapack.dtrtri(np.asfortranarray(L), lower=1)[0]
    U_inv = lapack.dtrtri(np.asfortranarray(L.T), lower=0)[0]
    tmp = np.matmul(L_inv, D_inv_M.T)
    C = np.matmul(U_inv, tmp)

    steps = [
        ("resid = y[:,None] - mu[:,None]", lambda: y[:, None] - mu[:, None], n),
        ("d_inv = 1 / d[:,None]", lambda: 1 / d[:, None], n),
        ("D_inv_y = d_inv * resid", lambda: d_inv * resid, n),
        ("D_inv_M = d_inv * M", lambda: d_inv * M, n * k),
        ("B = M.T @ D_inv_M", lambda: np.matmul(M.T, D_inv_M), 2 * n * k * k),
        ("cholesky(B)", lambda: np.linalg.cholesky(B), k**3 // 3),
        (
            "dtrtri x2",
            lambda: (
                lapack.dtrtri(np.asfortranarray(L), lower=1),
                lapack.dtrtri(np.asfortranarray(L.T), lower=0),
            ),
            2 * k**3 // 3,
        ),
        ("tmp = L_inv @ D_inv_M.T", lambda: np.matmul(L_inv, D_inv_M.T), 2 * k * k * n),
        ("C = U_inv @ tmp", lambda: np.matmul(U_inv, tmp), 2 * k * k * n),
        (
            "D_inv_M @ (C @ resid)",
            lambda: np.matmul(D_inv_M, np.matmul(C, resid)),
            4 * n * k,
        ),
        ("sum(log(d))", lambda: np.sum(np.log(d)), 2 * n),
    ]

    breakdown = []
    for name, function, flops in steps:
        seconds = _timed(function, repeats)
        breakdown.append(
            {
                "operation": name,
                "microseconds": seconds * 1e6,
                "flops": flops,
                "gflops": flops / seconds / 1e9 if seconds else float("nan"),
            }
        )
    return breakdown


def audit(repeats: int) -> dict:
    model = load_model()
    config = Config.desi_y3_fast()
    prepared = prepare_spectrum(make_spectrum(), model, config)
    assembled = assemble_model(prepared, model, config)

    y = prepared.flux
    mu = assembled.mean
    M = assembled.factor
    d = assembled.absorption_variance + prepared.noise_variance
    n, k = M.shape

    def call():
        log_mvnpdf_low_rank(y, mu, M, d)

    per_call = _timed(call, repeats)

    # --- the operations, named, with their shapes and FLOP counts -------------
    operations = [
        ("D_inv_M = M / d[:, None]", f"({n}, {k})", 1 * n * k),
        ("B = M.T @ D_inv_M", f"({k}, {n}) @ ({n}, {k})", 2 * n * k * k),
        ("cholesky(B)", f"({k}, {k})", k**3 // 3),
        ("dtrtri(L)", f"({k}, {k})", k**3 // 3),
        ("tmp = L_inv @ D_inv_M.T", f"({k}, {k}) @ ({k}, {n})", 2 * k * k * n),
        ("dtrtri(L.T)", f"({k}, {k})", k**3 // 3),
        ("C = U_inv @ tmp", f"({k}, {k}) @ ({k}, {n})", 2 * k * k * n),
        ("C @ y", f"({k}, {n}) @ ({n}, 1)", 2 * k * n),
        ("D_inv_M @ (C @ y)", f"({n}, {k}) @ ({k}, 1)", 2 * n * k),
        ("sum(log(d))", f"({n},)", 2 * n),
    ]
    total_flops = sum(flops for _, _, flops in operations)

    # --- allocation ----------------------------------------------------------
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(100):
        call()
    after = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    peak_bytes = after - before

    # --- threading, measured -------------------------------------------------
    big = np.random.default_rng(0).standard_normal((2000, 2000))
    threading = {
        "large_matmul_2000_cpu_over_wall": _cpu_over_wall(lambda: big @ big, 3),
        "likelihood_cpu_over_wall": _cpu_over_wall(call, min(repeats, 2000)),
    }

    achieved = total_flops / per_call / 1e9
    generic = _generic_shape_microbenchmark(n, k, repeats)
    real_operands = _real_operand_microbenchmark(y, mu, M, d, repeats)
    breakdown = _operation_breakdown(y, mu, M, d, repeats)
    preallocation = _preallocation_comparison(y, mu, M, d, min(repeats, 3000))

    return {
        "dimensions": {"n_pixels": n, "rank": k},
        # Reported on its own. Everything below is a microbenchmark and is NOT
        # subtracted from it: separately
        # timed operations run under different cache, allocation and scheduling
        # conditions, and a sum of independently chosen medians is not the median
        # of their composition. An independent rerun produced a sum of parts
        # LARGER than the whole call, so the subtraction can even change sign.
        "whole_call": {
            "per_call_ms": per_call * 1e3,
            "achieved_gflops_over_whole_call": achieved,
            "total_flops_per_call": total_flops,
        },
        "operations": [
            {"operation": name, "shape": shape, "flops": flops}
            for name, shape, flops in operations
        ],
        # Descriptive profiling only. Not a forecast of a compiled kernel's gain.
        "operation_breakdown": breakdown,
        "generic_shape_microbenchmark": generic,
        "real_operand_microbenchmark": real_operands,
        "peak_traced_bytes_per_100_calls": peak_bytes,
        "threading": threading,
        "preallocation_experiment": preallocation,
    }


def thread_sweep(counts: list[int], repeats: int) -> dict:
    """Time the likelihood at several BLAS thread counts, in one process.

    The advisory threshold must come from a measured onset, not
    from interpolating between two points.

    Uses ``threadpoolctl`` to resize the pool between measurements. That is a
    real limitation and is recorded with the result: resizing an existing pool is
    not identical to starting a process with the variable set, which is the
    configuration a user actually has. The sweep shows the *shape* of the
    penalty; the per-count absolutes should be confirmed with separate processes
    before a threshold is frozen.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return {"available": False, "reason": "threadpoolctl is not installed"}

    model = load_model()
    config = Config.desi_y3_fast()
    prepared = prepare_spectrum(make_spectrum(), model, config)
    assembled = assemble_model(prepared, model, config)
    y, mu, M = prepared.flux, assembled.mean, assembled.factor
    d = assembled.absorption_variance + prepared.noise_variance

    def call():
        log_mvnpdf_low_rank(y, mu, M, d)

    measurements = []
    # Ascending then descending, so a warming or throttling machine cannot be
    # mistaken for a monotonic thread effect.
    order = list(counts) + list(reversed(counts))
    for threads in order:
        with threadpool_limits(limits=threads, user_api="blas"):
            seconds = _timed(call, repeats)
            ratio = _cpu_over_wall(call, min(repeats, 2000))
        measurements.append(
            {"threads": threads, "per_call_ms": seconds * 1e3, "cpu_over_wall": ratio}
        )

    # Fold the two passes together by taking the median per thread count.
    folded = {}
    for entry in measurements:
        folded.setdefault(entry["threads"], []).append(entry)
    summary = []
    for threads in counts:
        entries = folded.get(threads, [])
        if not entries:
            continue
        per_call = statistics.median(e["per_call_ms"] for e in entries)
        summary.append(
            {
                "threads": threads,
                "per_call_ms": per_call,
                "cpu_over_wall": statistics.median(e["cpu_over_wall"] for e in entries),
            }
        )

    baseline = next((e["per_call_ms"] for e in summary if e["threads"] == 1), None)
    for entry in summary:
        entry["overhead_vs_one_thread"] = (
            entry["per_call_ms"] / baseline - 1.0 if baseline else float("nan")
        )

    return {
        "available": True,
        "method": "threadpoolctl.threadpool_limits within one process",
        "caveat": (
            "resizing a live pool is not identical to starting a process with "
            "the thread variable set; treat the shape as informative and confirm "
            "absolutes with separate processes"
        ),
        "order": order,
        "measurements": measurements,
        "summary": summary,
    }


#: The child measures one thread count and prints one JSON line. Kept as a
#: string so the controller can launch it with `python -c`, which guarantees the
#: thread variables are set before numpy and scipy are imported -- the whole
#: point of the separate-process mode.
_CHILD = r"""
import json, os, statistics, sys, time
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import numpy as np
from synthetic import make_spectrum
from gp_dla_finder import load_model
from gp_dla_finder.config import Config
from gp_dla_finder.gp.evidence import assemble_model
from gp_dla_finder.gp.likelihood import log_mvnpdf_low_rank
from gp_dla_finder.gp.spectrum import prepare_spectrum

repeats = int(sys.argv[3])
model, config = load_model(), Config.desi_y3_fast()
prepared = prepare_spectrum(make_spectrum(), model, config)
assembled = assemble_model(prepared, model, config)
y, mu, M = prepared.flux, assembled.mean, assembled.factor
d = assembled.absorption_variance + prepared.noise_variance

def call():
    log_mvnpdf_low_rank(y, mu, M, d)

call()
blocks = []
for _ in range(5):
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    blocks.append((time.perf_counter() - start) / repeats)

call()
w0, c0 = time.perf_counter(), time.process_time()
for _ in range(min(repeats, 2000)):
    call()
wall, cpu = time.perf_counter() - w0, time.process_time() - c0

pools = "unavailable"
try:
    from threadpoolctl import threadpool_info
    pools = [p for p in threadpool_info() if p.get("user_api") == "blas"]
except ImportError:
    pass

print("@@RESULT@@" + json.dumps({
    "blocks_s": blocks,
    "per_call_ms": statistics.median(blocks) * 1e3,
    "block_min_ms": min(blocks) * 1e3,
    "block_max_ms": max(blocks) * 1e3,
    "block_stdev_ms": statistics.stdev(blocks) * 1e3,
    "cpu_over_wall": cpu / wall if wall else float("nan"),
    "repeats": repeats,
    "warmup": 1,
    "blocks": len(blocks),
    "pools": pools,
    "thread_env": {k: os.environ.get(k, "(unset)") for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
}))
"""


def _portable_command(threads: int, repeats: int) -> str:
    """The child command in a form that identifies the run without naming a host."""
    return (
        f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
        f"MKL_NUM_THREADS={threads} VECLIB_MAXIMUM_THREADS={threads} "
        f"NUMEXPR_NUM_THREADS={threads} "
        f"python -c <blas_audit child> <repo>/src <repo>/tests {repeats}"
    )


def separate_process_sweep(
    counts: list[int], repeats: int, timeout: float = 900.0
) -> dict:
    """Measure each thread count in a FRESH process.

    The live-pool sweep resizes an existing pool, which is not what a user has:
    a user sets ``OPENBLAS_NUM_THREADS`` before the process starts, and the pool
    is sized once at import. This launches a child per count with the variables
    already in the environment, so the measurement matches the configuration
    people actually run.

    Order-balanced: ascending then descending, so a warming or throttling machine
    cannot be mistaken for a thread effect. Every child's exact command and
    environment is retained.
    """

    #: A retained record must be shareable. An absolute sys.executable and raw
    #: child stderr both leak home and repository paths, and this project has
    #: already shipped one private path by accident, so this redacts rather
    #: than trusting the caller.
    def _redact(text: str) -> str:
        import re

        text = text.replace(str(Path.home()), "~")
        text = text.replace(str(ROOT), "<repo>")
        return re.sub(r"(/[^\s:'\"]+){2,}", "<path>", text)

    seen = set()
    for value in counts:
        if value < 1:
            raise ValueError(f"thread counts must be >= 1, got {value}")
        if value in seen:
            raise ValueError(f"duplicate thread count {value}")
        seen.add(value)

    measurements = []
    failures = 0
    order = list(counts) + list(reversed(counts))
    for threads in order:
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
        }
        command = [
            sys.executable,
            "-c",
            _CHILD,
            str(ROOT / "src"),
            str(ROOT / "tests"),
            str(repeats),
        ]
        try:
            completed = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            failures += 1
            measurements.append(
                {
                    "threads": threads,
                    "command": _portable_command(threads, repeats),
                    "returncode": None,
                    "error": f"timed out after {timeout}s",
                }
            )
            continue

        entry = {
            "threads": threads,
            # Portable: no absolute interpreter path, no repository path.
            "command": _portable_command(threads, repeats),
            "returncode": completed.returncode,
        }
        if completed.returncode != 0:
            failures += 1
            entry["error"] = _redact(completed.stderr[-800:])
        else:
            payload = [
                line
                for line in completed.stdout.splitlines()
                if line.startswith("@@RESULT@@")
            ]
            if payload:
                entry.update(json.loads(payload[-1][len("@@RESULT@@") :]))
                # The child reports raw threadpoolctl records; strip the paths
                # before they reach a retained artifact.
                if "pools" in entry:
                    entry["pools"] = _sanitise_pool_records(entry["pools"])
            else:
                failures += 1
                entry["error"] = "child produced no result line"
        measurements.append(entry)

    folded: dict[int, list] = {}
    for entry in measurements:
        if "per_call_ms" in entry:
            folded.setdefault(entry["threads"], []).append(entry)

    summary = []
    for threads in counts:
        entries = folded.get(threads, [])
        if not entries:
            summary.append({"threads": threads, "error": "no successful run"})
            continue
        summary.append(
            {
                "threads": threads,
                "per_call_ms": statistics.median(e["per_call_ms"] for e in entries),
                "cpu_over_wall": statistics.median(e["cpu_over_wall"] for e in entries),
                "passes": len(entries),
            }
        )

    baseline = next(
        (e["per_call_ms"] for e in summary if e["threads"] == 1 and "per_call_ms" in e),
        None,
    )
    for entry in summary:
        if "per_call_ms" in entry:
            entry["overhead_vs_one_thread"] = (
                entry["per_call_ms"] / baseline - 1.0 if baseline else float("nan")
            )

    complete = failures == 0 and all("per_call_ms" in e for e in summary)
    return {
        # False when ANY child failed: a partially completed sweep reported as
        # available would be read as a full result.
        "available": complete,
        "status": "complete"
        if complete
        else f"incomplete: {failures} child failure(s)",
        "failures": failures,
        "method": "one fresh process per thread count, variables set before import",
        "python_version": platform.python_version(),
        "python_build": " ".join(platform.python_build()),
        "order": order,
        "measurements": measurements,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument(
        "--thread-sweep",
        nargs="*",
        type=int,
        default=None,
        help="thread counts to sweep, e.g. --thread-sweep 1 2 4 8 10",
    )
    parser.add_argument(
        "--child-timeout",
        type=float,
        default=900.0,
        help="seconds before a sweep child is killed and recorded as failed",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=("separate-process", "live-pool", "both"),
        default="separate-process",
        help=(
            "separate-process launches a fresh interpreter per thread count "
            "with the variables set before import, which is what a user "
            "actually has. live-pool resizes an existing pool and is retained "
            "only as a labelled diagnostic."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    import scipy

    record = {
        "environment": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_env": {
                name: os.environ.get(name, "(unset)") for name in _THREAD_VARS
            },
        },
        "blas": _blas_identity(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "audit": audit(args.repeats),
    }

    if args.thread_sweep is not None:
        counts = args.thread_sweep or [1, 2, 4, 8]
        if args.sweep_mode in ("separate-process", "both"):
            record["thread_sweep_separate_process"] = separate_process_sweep(
                counts, args.repeats, timeout=args.child_timeout
            )
        if args.sweep_mode in ("live-pool", "both"):
            # Kept separate and separately named: its values must never be mixed
            # silently with the separate-process ones.
            record["thread_sweep_live_pool"] = thread_sweep(counts, args.repeats)

    environment = record["environment"]
    print(
        f"cpu            {environment['cpu']} ({environment['logical_cpus']} logical)"
    )
    print(f"numpy/scipy    {environment['numpy']} / {environment['scipy']}")
    for name in ("blas", "lapack"):
        if name in record["blas"]:
            print(f"{name:14s} {record['blas'][name]}")
    pools = record["blas"].get("threadpool_info")
    print(f"thread pools   {pools if pools else '(none visible to threadpoolctl)'}")
    print(
        "thread env     "
        + ", ".join(f"{k}={v}" for k, v in environment["thread_env"].items())
    )

    result = record["audit"]
    print(
        f"\nlikelihood     n={result['dimensions']['n_pixels']} "
        f"k={result['dimensions']['rank']}   "
        f"{result['whole_call']['per_call_ms']:.4f} ms/call"
    )
    print(
        f"total flops    "
        f"{result['whole_call']['total_flops_per_call'] / 1e6:.2f} Mflop/call"
    )
    whole = result["whole_call"]
    generic = result["generic_shape_microbenchmark"]
    real = result["real_operand_microbenchmark"]
    achieved = whole["achieved_gflops_over_whole_call"]
    print(f"achieved       {achieved:.1f} GFLOP/s over the whole call")
    print(
        f"real operands  the three matmuls on the likelihood's own arrays take "
        f"{real['three_matmul_seconds'] * 1e6:.1f} us "
        f"({real['B_gflops']:.0f} / {real['tmp_gflops']:.0f} / "
        f"{real['C_gflops']:.0f} GFLOP/s)"
    )
    print(
        f"generic shapes fresh random arrays of the same shape take "
        f"{generic['three_matmul_seconds'] * 1e6:.1f} us "
        f"(microbenchmark only, not a floor)"
    )
    print(
        f"allocation     {result['peak_traced_bytes_per_100_calls'] / 1024:.0f} KiB "
        "peak over 100 calls"
    )
    print(
        f"threads used   large matmul "
        f"{result['threading']['large_matmul_2000_cpu_over_wall']:.2f}x cpu/wall, "
        f"likelihood {result['threading']['likelihood_cpu_over_wall']:.2f}x"
    )

    prealloc = result["preallocation_experiment"]
    print(
        f"prealloc expt  {prealloc['speedup']:.2f}x, "
        f"bitwise={prealloc['bitwise_identical']} -> {prealloc['verdict']}"
    )

    print("\noperation breakdown (measured):")
    print(f"  {'operation':32s} {'us':>8s} {'Mflop':>8s} {'GFLOP/s':>9s}")
    measured_total = 0.0
    for operation in result["operation_breakdown"]:
        measured_total += operation["microseconds"]
        print(
            f"  {operation['operation']:32s} {operation['microseconds']:8.2f} "
            f"{operation['flops'] / 1e6:8.3f} {operation['gflops']:9.1f}"
        )
    print(f"  {'--- sum of parts':32s} {measured_total:8.2f}")
    print(f"  {'--- whole call':32s} {whole['per_call_ms'] * 1e3:8.2f}")
    print(
        "  (descriptive only: the two are measured under different cache and\n"
        "   allocation conditions, so their difference is NOT Python dispatch\n"
        "   and must not be used to forecast a compiled kernel's gain)"
    )

    def _print_sweep(sweep, title):
        if not sweep:
            return
        if not sweep.get("available"):
            print(f"\n{title} unavailable: {sweep.get('reason')}")
            return
        print(f"\n{title} ({sweep['method']}):")
        print(
            f"  {'threads':>8s} {'ms/call':>9s} {'cpu/wall':>9s} {'vs 1 thread':>12s}"
        )
        for entry in sweep["summary"]:
            if "per_call_ms" not in entry:
                print(f"  {entry['threads']:8d}    {entry.get('error', 'failed')}")
                continue
            print(
                f"  {entry['threads']:8d} {entry['per_call_ms']:9.4f} "
                f"{entry['cpu_over_wall']:9.2f} "
                f"{100 * entry['overhead_vs_one_thread']:+11.1f}%"
            )
        if "caveat" in sweep:
            print(f"  caveat: {sweep['caveat']}")

    _print_sweep(
        record.get("thread_sweep_separate_process"), "thread sweep, separate processes"
    )
    _print_sweep(
        record.get("thread_sweep_live_pool"),
        "thread sweep, LIVE POOL (diagnostic only)",
    )

    if args.json:
        args.json.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nrecord written to {args.json}")

    separate = record.get("thread_sweep_separate_process")
    if separate and not separate.get("available"):
        print(f"\nSWEEP INCOMPLETE: {separate.get('status')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
