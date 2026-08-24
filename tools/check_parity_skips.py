#!/usr/bin/env python3
"""Check a canonical-parity report against the reviewed skip allowlist.

The job used to pass whenever *something* passed, which meant a
test that silently stopped running was invisible -- and one was: the
legacy-writer parity tests needed both the reference checkout and astropy, which
live in different jobs, so they skipped in both for an entire increment.

Counting skips would not have caught that either, because the count was right;
the wrong tests were skipping. So the contract names *which* test may skip and
*why*, and this script fails on any deviation:

* a test skips that is not on the list;
* a listed test skips for a different reason;
* a listed entry matches nothing, which is what a rename or deletion looks like;
* the count for a listed entry changes.

Usage::

    python tools/check_parity_skips.py parity-report.txt parity/expected_skips.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: ``SKIPPED [2] tests/test_catalogue.py:34: catalogue I/O needs astropy``
#:
#: The line number is captured but deliberately NOT part of the contract: it
#: moves whenever anything above it is edited, and pinning it would turn every
#: unrelated edit into a CI failure.
_SKIP_LINE = re.compile(
    r"^SKIPPED\s+\[(?P<count>\d+)\]\s+(?P<file>[^:]+):(?P<line>\d+):\s*(?P<reason>.*)$"
)


def parse_report(text: str) -> list[dict]:
    """Every SKIPPED entry in a pytest ``-rs`` report."""
    found = []
    for raw in text.splitlines():
        match = _SKIP_LINE.match(raw.strip())
        if match:
            found.append(
                {
                    "count": int(match.group("count")),
                    "file": match.group("file").strip(),
                    "reason": match.group("reason").strip(),
                }
            )
    return found


def parse_contract(text: str) -> list[dict]:
    """The allowlist: ``count | file | reason substring`` per line."""
    entries = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise ValueError(
                f"{number}: expected 'count | file | reason substring', got {raw!r}"
            )
        count, path, reason = parts
        if not count.isdigit():
            raise ValueError(f"{number}: count must be a number, got {count!r}")
        entries.append({"count": int(count), "file": path, "reason": reason})
    return entries


def check(report: str, contract: str) -> list[str]:
    """Problems, most specific first. Empty means the run matched the contract."""
    observed = parse_report(report)
    expected = parse_contract(contract)
    problems: list[str] = []

    unmatched = list(observed)
    for entry in expected:
        hits = [
            item
            for item in unmatched
            if item["file"] == entry["file"] and entry["reason"] in item["reason"]
        ]
        if not hits:
            # A rename, a deletion, or a reason that changed wording. All three
            # mean the reviewed contract no longer describes the run.
            same_file = [i for i in observed if i["file"] == entry["file"]]
            detail = (
                f" (that file skipped for: {same_file[0]['reason']!r})"
                if same_file
                else " (that file did not skip at all)"
            )
            problems.append(
                f"expected skip no longer present: {entry['file']} "
                f"~ {entry['reason']!r}{detail}"
            )
            continue
        total = sum(item["count"] for item in hits)
        if total != entry["count"]:
            problems.append(
                f"skip count changed for {entry['file']} ~ {entry['reason']!r}: "
                f"expected {entry['count']}, saw {total}"
            )
        for item in hits:
            unmatched.remove(item)

    for item in unmatched:
        problems.append(
            f"unexpected skip: {item['file']} [{item['count']}] {item['reason']!r}"
        )

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="a pytest -rs report")
    parser.add_argument("contract", type=Path, help="the allowlist file")
    args = parser.parse_args(argv)

    report = args.report.read_text()
    contract = args.contract.read_text()

    print("--- allowlisted skips, as they occurred ---")
    for item in parse_report(report):
        print(f"  [{item['count']}] {item['file']}: {item['reason']}")
    if not parse_report(report):
        print("  (nothing skipped)")

    problems = check(report, contract)
    if problems:
        print()
        for problem in problems:
            print(f"::error::{problem}")
        print(
            "\nThe canonical-parity skip set is a reviewed contract "
            f"({args.contract}). If a change here is intended, update that file "
            "in the same commit so the exception stays visible."
        )
        return 1

    print(f"\nmatches the reviewed contract ({args.contract})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
