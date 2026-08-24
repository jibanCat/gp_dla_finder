"""The canonical-parity skip allowlist, and the four ways it must fail.

PI ruling N87. The job used to pass whenever *something* passed, so a test that
silently stopped running was invisible -- and one was, for a whole increment.
Counting skips would not have helped: the count was right and the wrong tests
were skipping.

So the contract names which test may skip and why, and these tests pin the four
deviations the ruling requires it to catch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "parity" / "expected_skips.txt"
CHECKER = ROOT / "tools" / "check_parity_skips.py"

#: The skip set of the last green canonical-parity run, verbatim. Kept as the
#: fixture so these tests exercise the real format rather than an idealised one.
GREEN_REPORT = """176 passed, 6 skipped, 505 deselected, 2 warnings in 164.51s
SKIPPED [2] tests/test_catalogue.py:34: catalogue I/O needs astropy
SKIPPED [1] tests/test_examples.py:23: the examples write FITS
SKIPPED [1] tests/test_finder.py:28: catalogue I/O needs astropy
SKIPPED [1] tests/test_prior.py:210: set GP_DLA_FINDER_PRIOR_SOURCES to a \
directory of source catalogues
SKIPPED [1] tests/test_reference_parity.py:528: reading a note from a pinned \
commit needs git; the canonical-parity container does not ship it
"""


def _check(report: str) -> list[str]:
    sys.path.insert(0, str(ROOT / "tools"))
    from check_parity_skips import check

    return check(report, CONTRACT.read_text())


def test_the_contract_file_exists_and_parses():
    sys.path.insert(0, str(ROOT / "tools"))
    from check_parity_skips import parse_contract

    entries = parse_contract(CONTRACT.read_text())
    assert len(entries) == 5, "the reviewed set is five entries covering six skips"
    assert sum(e["count"] for e in entries) == 6


def test_the_last_green_run_matches_the_contract():
    assert _check(GREEN_REPORT) == []


# --- the four deviations the ruling requires ---------------------------------


def test_a_new_skip_fails():
    """The case that hid the legacy-writer parity tests for an increment."""
    report = GREEN_REPORT + (
        "SKIPPED [1] tests/test_legacy_writer_parity.py:99: "
        "could not import 'astropy'\n"
    )
    (problem,) = _check(report)
    assert "unexpected skip" in problem
    assert "test_legacy_writer_parity.py" in problem


def test_an_allowlisted_test_skipping_for_a_different_reason_fails():
    """Same file, same count, different cause -- which is a different fact."""
    report = GREEN_REPORT.replace(
        "tests/test_prior.py:210: set GP_DLA_FINDER_PRIOR_SOURCES to a "
        "directory of source catalogues",
        "tests/test_prior.py:210: the prior asset failed to load",
    )
    problems = _check(report)
    assert any("no longer present" in p for p in problems), problems
    assert any("test_prior.py" in p for p in problems)


def test_a_renamed_or_deleted_allowlisted_test_fails():
    """A silent disappearance is the failure mode a count cannot see."""
    report = "\n".join(
        line for line in GREEN_REPORT.splitlines() if "test_examples.py" not in line
    )
    (problem,) = _check(report)
    assert "no longer present" in problem
    assert "test_examples.py" in problem


def test_a_changed_count_for_an_allowlisted_test_fails():
    """Two astropy skips in test_catalogue.py, not three."""
    report = GREEN_REPORT.replace(
        "SKIPPED [2] tests/test_catalogue.py:34",
        "SKIPPED [3] tests/test_catalogue.py:34",
    )
    (problem,) = _check(report)
    assert "skip count changed" in problem
    assert "expected 2, saw 3" in problem


def test_a_clean_run_with_no_skips_also_fails():
    """Not a pass by default.

    If every allowlisted test suddenly ran, that is good news -- but it means
    the contract is stale, and the list should be shortened deliberately rather
    than left describing a world that no longer exists.
    """
    problems = _check("176 passed in 164.51s\n")
    assert len(problems) == 5
    assert all("no longer present" in p for p in problems)


# --- the script itself, as CI invokes it -------------------------------------


def test_the_checker_exits_non_zero_on_an_unexpected_skip(tmp_path):
    report = tmp_path / "report.txt"
    report.write_text(
        GREEN_REPORT + "SKIPPED [1] tests/test_new_thing.py:1: some new reason\n"
    )
    completed = subprocess.run(
        [sys.executable, str(CHECKER), str(report), str(CONTRACT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 1
    assert "::error::" in completed.stdout
    assert "test_new_thing.py" in completed.stdout


def test_the_checker_prints_the_skips_so_they_stay_visible(tmp_path):
    """The exception must remain readable in the log, not merely tolerated."""
    report = tmp_path / "report.txt"
    report.write_text(GREEN_REPORT)
    completed = subprocess.run(
        [sys.executable, str(CHECKER), str(report), str(CONTRACT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    for name in ("test_catalogue.py", "test_prior.py", "test_reference_parity.py"):
        assert name in completed.stdout


def test_the_workflow_invokes_the_checker():
    """The contract is worthless if CI does not run it."""
    workflow = (ROOT / ".github" / "workflows" / "parity.yml").read_text()
    assert "tools/check_parity_skips.py" in workflow
    assert "parity/expected_skips.txt" in workflow
