"""BLAS thread diagnostics for a workload where more threads can be slower.

The GP likelihood multiplies matrices of shape ``(1105, 30)`` — about 2 Mflop per
product. Whether extra BLAS threads help or hurt at that size is **not** obvious
and is **not** monotonic, so it was measured rather than assumed. On the audited
host, freshly started 2-, 4- and 8-thread processes all beat one thread.

Two sweeps on the same 10-core machine (OpenBLAS 0.3.23, order-balanced), and
they do not agree:

* 1 thread: live pool 0.2438 ms, fresh process 0.2341 ms
* 2 threads: live pool 0.2196 (-9.9 %), fresh process 0.1882 (-19.6 %)
* 4 threads: live pool 0.1693 (-30.6 %), fresh process 0.1547 (-33.9 %)
* 8 threads: live pool 0.4272 (+75 %), fresh process 0.2040 (**-12.9 %**)
* 10 threads: live pool 1.7508 (+618 %), fresh process 1.6164 (+591 %)

The separate-process measurements best represent normal use because thread
limits are set before a process starts. Resizing a live pool makes eight threads
appear 75% slower, while a process started with eight threads remains 13% faster
than one thread. We therefore keep the live-pool sweep as a separate diagnostic.

On this host, a few threads improve performance, eight threads still help, and
performance degrades when the pool uses all ten cores. This result describes one
machine and should not be treated as a universal optimum.

What this does and does not show
--------------------------------
It shows single-process scaling as a function of pool size. It does **not** show
the multi-worker regime (``worker processes x BLAS threads > cores``), which has
not been measured here. One BLAS thread per worker remains sound guidance for
catalogue production, but that is reasoning about how independent processes
contend, not a measurement this project has made.

A shared 2-core CI runner measured 20-23 % overhead at 2 threads on one occasion
and 1.6 % on another, with different host CPUs — and on a 2-core machine, 2
threads *is* the whole machine. Small-pool behaviour is sensitive to the host.

Runtime integration
-------------------
:class:`~gp_dla_finder.finder.Finder` calls
:func:`warn_once_about_blas_threads` at the inference boundary. The diagnostic
runs once per process, not at import time or inside the likelihood loop.

What this module does, and deliberately does not
------------------------------------------------
It detects and warns once. It does not set environment variables, impose a
process-wide limit, or change the caller's thread policy.

Detection needs :mod:`threadpoolctl`, an **optional** dependency
(``pip install 'gp_dla_finder[performance]'``). Without it the package works
normally and simply cannot offer the diagnostic: no warning, no behavior
change, no noise.

The caveat that belongs with the number
---------------------------------------
Neither measurement settles what a production Linux machine will do. The Linux
number above comes from a **2-core** shared CI runner, which cannot spawn enough
threads to produce the failure mode at all — it bounds the small-pool case and
says nothing about a 64-core node. That is why the warning says "measured
performance risk" rather than "your run is four times slower".
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "BLASPerformanceWarning",
    "blas_thread_report",
    "usable_cpu_count",
    "reset_thread_advisory",
    "warn_once_about_blas_threads",
]


class BLASPerformanceWarning(UserWarning):
    """A BLAS thread configuration likely to be slow for this workload.

    Its own class so it can be silenced without also silencing scientific or
    numerical warnings::

        warnings.filterwarnings("ignore", category=BLASPerformanceWarning)

    It is a **performance** advisory. It never indicates an inference error, and
    it is not proof that a particular workload is slower — only that the
    configuration carries a measured risk for small-matrix work.
    """


def usable_cpu_count() -> tuple[int, str]:
    """CPUs this **process** may use, and where the number came from.

    ``os.cpu_count()`` reports the machine, not the process. Under a cgroup, a
    CPU affinity mask, scheduler, or batch system, a process can be confined to
    a fraction of the host. Comparing a BLAS pool against the full machine would
    then give the wrong diagnostic.
    """
    getter = getattr(os, "process_cpu_count", None)
    if getter is not None:
        value = getter()
        if value:
            return int(value), "os.process_cpu_count"
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return len(affinity(0)), "os.sched_getaffinity"
        except OSError:  # pragma: no cover - platform dependent
            pass
    return int(os.cpu_count() or 1), "os.cpu_count"


def _pool_covers_available_cpus(max_threads: int) -> bool:
    """True when one BLAS pool is sized to all the CPU this process can use.

    Deliberately **not** an absolute thread count. Two were tried and both were
    wrong: 2 was a false alarm at 1.6 %, and 4 was chosen from a live-pool sweep
    that said 8 threads were slow when a properly started 8-thread process is
    13 % faster. What tracked the collapse was the pool covering the available
    CPUs — 10 threads on 10 was 6x slower, 8 on 10 was fine.

    Provisional, and narrow: one macOS machine, one BLAS. It is a warning-only
    heuristic, not a performance law, and it does not generalise to a 4-core
    laptop or a 64-core node on this evidence alone.
    """
    return max_threads >= usable_cpu_count()[0]


_already_warned = False


def reset_thread_advisory() -> None:
    """Re-arm the one-time advisory. For tests, and for long-lived processes."""
    global _already_warned
    _already_warned = False


def blas_thread_report() -> Mapping[str, object]:
    """What BLAS pools this process has, as far as they can be detected.

    Returns a mapping with ``available`` false when :mod:`threadpoolctl` is not
    installed, rather than guessing at the active thread pools.
    NumPy and SciPy frequently ship *separate* BLAS libraries, so ``pools`` may
    hold more than one entry with different thread counts, and the advisory keys
    off the largest.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return MappingProxyType(
            {
                "available": False,
                "reason": (
                    "threadpoolctl is not installed; install "
                    "gp_dla_finder[performance] for BLAS thread diagnostics"
                ),
                "pools": (),
            }
        )

    pools = tuple(
        {
            "internal_api": entry.get("internal_api"),
            "prefix": entry.get("prefix"),
            "version": entry.get("version"),
            "num_threads": entry.get("num_threads"),
            "threading_layer": entry.get("threading_layer"),
            "architecture": entry.get("architecture"),
        }
        for entry in threadpool_info()
        if entry.get("user_api") == "blas"
    )
    max_threads = max((p["num_threads"] or 1 for p in pools), default=1)
    return MappingProxyType(
        {
            "available": True,
            "pools": pools,
            "max_threads": max_threads,
            "usable_cpus": usable_cpu_count()[0],
            "usable_cpu_source": usable_cpu_count()[1],
            # Named for what it measures: one pool covering the available CPUs.
            # NOT multi-worker oversubscription, which needs a worker count this
            # module does not have.
            "full_pool_performance_risk": _pool_covers_available_cpus(max_threads),
        }
    )


def warn_once_about_blas_threads(stacklevel: int = 3) -> bool:
    """Warn at most once if a threaded BLAS looks badly configured for this work.

    Call this from the high-level inference boundary on first use — **not** at
    import, and never per likelihood evaluation. Returns whether a warning was
    issued, which is what makes it testable.

    Silent, and returns ``False``, when :mod:`threadpoolctl` is absent, when the
    pools look fine, or when it has already fired.
    """
    global _already_warned
    if _already_warned:
        return False

    report = blas_thread_report()
    if not report["available"] or not report.get("full_pool_performance_risk"):
        # Mark as done either way: re-probing on every spectrum is pointless, and
        # a process's BLAS pools do not change under it.
        _already_warned = True
        return False

    pools = ", ".join(
        f"{pool['prefix'] or pool['internal_api']} {pool['version'] or ''}"
        f" ({pool['num_threads']} threads)".replace("  ", " ")
        for pool in report["pools"]
    )
    cores, source = usable_cpu_count()
    warnings.warn(
        f"This BLAS is configured with up to {report['max_threads']} threads "
        f"({pools}); this process can use {cores} CPUs ({source}), so one pool "
        "covers all of them. On this project's separate-process thread sweep, a "
        "few BLAS threads HELP the GP likelihood (4 threads were 34% faster than "
        "1, and 8 were still 13% faster), but sizing the pool to every core was "
        "6x SLOWER. This is a performance risk, not an inference error, and not "
        "proof that your run is slower.\n"
        "\n"
        "Note this is single-process scaling. The multi-worker regime "
        "(workers x threads > cores) was NOT measured here.\n"
        "\n"
        "Which setting is right depends on how you parallelise:\n"
        "  * one spectrum at a time in one process: a few threads help; leave "
        "some cores free.\n"
        "  * many spectra in parallel, one worker per core: give each worker ONE "
        "BLAS thread. Independent processes each spawning a full pool "
        "oversubscribe the machine -- reasoning about the usual "
        "catalogue-production model, not a measurement this project has made.\n"
        "\n"
        "Thread counts must be set BEFORE numpy and scipy are imported:\n"
        "\n"
        "    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\n"
        "        python your_script.py\n"
        "\n"
        "This package will not change your thread configuration for you. To "
        "silence this advisory once you have chosen a policy:\n"
        "\n"
        "    import warnings\n"
        "    from gp_dla_finder.performance import BLASPerformanceWarning\n"
        "    warnings.filterwarnings('ignore', category=BLASPerformanceWarning)\n"
        "\n"
        "These figures are from one 10-core machine and one BLAS; the rule is "
        "PROVISIONAL until confirmed on a realistic Linux baseline.",
        BLASPerformanceWarning,
        stacklevel=stacklevel,
    )
    _already_warned = True
    return True
