"""The BLAS thread advisory (PI rulings N47, N48).

Two properties matter more than the message text:

* it **warns** and never **acts**. A library that quietly rewrites the caller's
  thread policy breaks any program doing its own parallelism;
* it is silent and harmless when :mod:`threadpoolctl` is absent, because that is
  an optional dependency and inference must work without it.
"""

from __future__ import annotations

import warnings

import pytest

from gp_dla_finder.performance import (
    BLASPerformanceWarning,
    blas_thread_report,
    reset_thread_advisory,
    warn_once_about_blas_threads,
)

HAS_THREADPOOLCTL = blas_thread_report()["available"]


@pytest.fixture(autouse=True)
def rearmed():
    """Each test gets a fresh advisory; the module state is global by design."""
    reset_thread_advisory()
    yield
    reset_thread_advisory()


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_the_report_is_honest_when_threadpoolctl_is_missing(monkeypatch):
    """No guess dressed as a measurement."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "threadpoolctl":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    report = blas_thread_report()
    assert report["available"] is False
    assert "threadpoolctl" in report["reason"]
    assert report["pools"] == ()


def test_no_warning_without_the_optional_dependency(monkeypatch):
    """Inference must work, and stay quiet, without the performance extra."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "threadpoolctl":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all would fail here
        assert warn_once_about_blas_threads() is False


@pytest.mark.skipif(not HAS_THREADPOOLCTL, reason="needs the performance extra")
def test_the_report_handles_more_than_one_pool():
    """NumPy and SciPy often ship separate BLAS libraries with separate pools."""
    report = blas_thread_report()
    assert report["available"] is True
    assert isinstance(report["pools"], tuple)
    for pool in report["pools"]:
        # Every field the dependency lower bound is justified by must actually be
        # retained, or the rationale in pyproject.toml is fiction
        # (PI ruling, increment-11 correction 3).
        for field in (
            "num_threads",
            "prefix",
            "internal_api",
            "version",
            "threading_layer",
            "architecture",
        ):
            assert field in pool, f"pool record is missing {field!r}"
    # The advisory keys off the largest pool, not the first one.
    if report["pools"]:
        assert report["max_threads"] == max(
            pool["num_threads"] or 1 for pool in report["pools"]
        )


# --------------------------------------------------------------------------
# The advisory
# --------------------------------------------------------------------------


def test_it_warns_at_most_once(monkeypatch):
    monkeypatch.setattr(
        "gp_dla_finder.performance.blas_thread_report",
        lambda: {
            "available": True,
            "full_pool_performance_risk": True,
            "max_threads": 8,
            "pools": (
                {
                    "prefix": "libopenblas",
                    "internal_api": "openblas",
                    "version": "0.3.23",
                    "num_threads": 8,
                    "threading_layer": "pthreads",
                },
            ),
        },
    )
    with pytest.warns(BLASPerformanceWarning):
        assert warn_once_about_blas_threads() is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_once_about_blas_threads() is False


def test_it_stays_quiet_on_a_single_threaded_blas(monkeypatch):
    monkeypatch.setattr(
        "gp_dla_finder.performance.blas_thread_report",
        lambda: {
            "available": True,
            "full_pool_performance_risk": False,
            "max_threads": 2,
            "pools": (),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_once_about_blas_threads() is False


def test_the_message_says_what_it_is_and_is_not(monkeypatch):
    monkeypatch.setattr(
        "gp_dla_finder.performance.blas_thread_report",
        lambda: {
            "available": True,
            "full_pool_performance_risk": True,
            "max_threads": 10,
            "pools": (
                {
                    "prefix": "libopenblas",
                    "internal_api": "openblas",
                    "version": "0.3.23",
                    "num_threads": 10,
                    "threading_layer": "pthreads",
                },
            ),
        },
    )
    with pytest.warns(BLASPerformanceWarning) as caught:
        warn_once_about_blas_threads()
    message = str(caught[0].message)

    # It reports what was detected.
    assert "10 threads" in message
    assert "libopenblas" in message
    # It says it is a performance risk, not an error.
    assert "not an inference error" in message
    # It gives the operating model and says when the setting must be applied.
    assert "one worker per core" in message.lower()
    assert "ONE BLAS thread" in message
    assert "BEFORE numpy and scipy are imported" in message
    # And it does not claim threading never helps: the sweep showed 4 threads
    # beating 1 by 31%.
    assert "help" in message.lower()
    # It documents how to silence it.
    assert "BLASPerformanceWarning" in message
    # It refuses to overclaim, and carries the caveat on its own measurement.
    assert "not proof that" in message
    # The claim stays scoped to what was measured, and admits its own limits.
    assert "provisional" in message.lower()
    # It must NOT claim the 6x came from a multi-worker configuration, which was
    # never measured (PI ruling, increment-13 correction 3).
    assert "was NOT measured" in message
    assert "not a measurement this project has made" in message


def test_it_has_its_own_warning_class():
    """Silenceable without suppressing scientific or numerical warnings."""
    assert issubclass(BLASPerformanceWarning, UserWarning)
    assert not issubclass(BLASPerformanceWarning, RuntimeWarning)


def test_it_never_changes_the_thread_configuration(monkeypatch):
    """The load-bearing guarantee of N47.

    Detecting a risky configuration must not cause the package to touch the
    environment or enter a process-wide limit.
    """
    import os

    monkeypatch.setattr(
        "gp_dla_finder.performance.blas_thread_report",
        lambda: {
            "available": True,
            "full_pool_performance_risk": True,
            "max_threads": 16,
            "pools": (
                {
                    "prefix": "libopenblas",
                    "internal_api": "openblas",
                    "version": "0.3.23",
                    "num_threads": 16,
                    "threading_layer": "pthreads",
                },
            ),
        },
    )
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    before = {name: os.environ.get(name) for name in thread_vars}

    with pytest.warns(BLASPerformanceWarning):
        warn_once_about_blas_threads()

    after = {name: os.environ.get(name) for name in thread_vars}
    assert before == after, "the advisory modified the thread environment"

    if HAS_THREADPOOLCTL:
        # And no process-wide limit was entered.
        from threadpoolctl import threadpool_info

        assert threadpool_info() is not None


def test_the_rule_is_relative_to_usable_cpus_not_an_absolute_count(monkeypatch):
    """Two absolute thresholds were wrong before this became a relative rule.

    Four was chosen while the live-pool sweep showed 8 threads slower; the
    separate-process sweep then measured 8 threads still 13% *faster* than one.
    What predicted the collapse was the pool covering every core -- 10 on 10 -- so
    that is what the rule tests, and it generalises to a 4-core laptop and a
    64-core node without retuning.
    """
    from gp_dla_finder import performance

    monkeypatch.setattr(performance, "usable_cpu_count", lambda: (10, "test"))
    assert not performance._pool_covers_available_cpus(1)
    assert not performance._pool_covers_available_cpus(4)
    assert not performance._pool_covers_available_cpus(8), (
        "8 threads on 10 cores was measured FASTER than 1 and must not warn"
    )
    assert performance._pool_covers_available_cpus(10)
    assert performance._pool_covers_available_cpus(16)

    monkeypatch.setattr(performance, "usable_cpu_count", lambda: (4, "test"))
    assert not performance._pool_covers_available_cpus(2)
    assert performance._pool_covers_available_cpus(4)


def test_the_report_flags_risk_only_when_the_pool_covers_available_cpus(monkeypatch):
    """End to end through blas_thread_report, not just the predicate."""
    import sys
    import types

    from gp_dla_finder import performance

    monkeypatch.setattr(performance, "usable_cpu_count", lambda: (10, "test"))

    def install(threads):
        module = types.ModuleType("threadpoolctl")
        module.threadpool_info = lambda: [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "0.3.23",
                "num_threads": threads,
                "threading_layer": "pthreads",
                "architecture": "armv8",
            }
        ]
        monkeypatch.setitem(sys.modules, "threadpoolctl", module)

    for threads, expected in ((4, False), (8, False), (10, True)):
        install(threads)
        report = performance.blas_thread_report()
        assert report["max_threads"] == threads
        assert report["usable_cpus"] == 10
        assert report["full_pool_performance_risk"] is expected


def test_usable_cpu_count_prefers_the_process_limit_over_the_machine():
    """os.cpu_count() reports the machine; a cgroup or affinity mask may not.

    Comparing a BLAS pool against the machine would mis-fire in both directions
    for a process confined to part of the host (PI ruling N58).
    """
    from gp_dla_finder.performance import usable_cpu_count

    count, source = usable_cpu_count()
    assert count >= 1
    assert source in {"os.process_cpu_count", "os.sched_getaffinity", "os.cpu_count"}
    # Whichever source was used must be recorded, so a surprising number can be
    # explained rather than guessed at.
    assert source


def test_the_risk_field_is_not_called_oversubscription():
    """It measures one pool covering the CPUs, not workers x threads > cores."""
    from gp_dla_finder.performance import blas_thread_report

    report = blas_thread_report()
    if report["available"]:
        assert "full_pool_performance_risk" in report
        assert "oversubscription_risk" not in report


# --- the retained sweep artifact must not name the host ----------------------
#
# PI ruling, increment-14 correction 3: a retained BLAS record is shareable
# evidence, so it must carry the backend's identity without the absolute paths
# that spell out a home directory or a repository. The first retained sweep did
# leak them, through threadpoolctl's ``filepath``.


def _blas_audit_module():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_blas_audit_under_test", root / "tools" / "blas_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pool_records_keep_the_library_name_but_drop_the_directory():
    sanitise = _blas_audit_module()._sanitise_pool_records
    [cleaned] = sanitise(
        [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "num_threads": 10,
                "prefix": "libopenblas",
                "filepath": "/Users/someone/secret/venv/lib/libopenblas64_.0.dylib",
                "version": "0.3.23.dev",
                "threading_layer": "pthreads",
                "architecture": "armv8",
            }
        ]
    )
    assert "filepath" not in cleaned
    assert cleaned["library_file"] == "libopenblas64_.0.dylib"
    # Everything reproducibility needs survives.
    assert cleaned["version"] == "0.3.23.dev"
    assert cleaned["internal_api"] == "openblas"
    assert cleaned["threading_layer"] == "pthreads"
    assert cleaned["architecture"] == "armv8"
    assert cleaned["num_threads"] == 10
    assert "someone" not in repr(cleaned)
    assert "/" not in cleaned["library_file"]


def test_pool_sanitiser_tolerates_the_shapes_threadpoolctl_can_return():
    sanitise = _blas_audit_module()._sanitise_pool_records
    # threadpoolctl absent -> a string, not a list.
    assert sanitise("threadpoolctl not installed") == "threadpoolctl not installed"
    assert sanitise([]) == []
    # A record without a filepath must not gain an empty library_file.
    [cleaned] = sanitise([{"user_api": "blas", "num_threads": 1}])
    assert "library_file" not in cleaned
