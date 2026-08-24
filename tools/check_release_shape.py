#!/usr/bin/env python3
"""Check that a built distribution set has the shape a release should have.

Run against the ``dist/`` directory the release workflow just produced, before
anything is uploaded. It is a script rather than an inline shell block so that
the check can be run locally, exactly as CI runs it, and so a heredoc inside a
YAML block scalar is not load-bearing.

The rules:

* exactly one wheel, tagged ``py3-none-any``. A platform wheel means the
  optional extension was compiled into it, which is the wrong release shape:
  ``pip install`` should need no compiler and should work on any platform;
* no compiled objects inside the wheel, for the same reason;
* exactly one sdist, which keeps the extension source so a user who wants the
  libcerf backend can build it;
* both carry ``NOTICE.md`` and ``LICENSE``, because the asset provenance and
  redistribution terms have to travel with the distribution rather than only
  with the repository.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

COMPILED_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")


def check(dist: Path) -> list[str]:
    problems: list[str] = []

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))

    if len(wheels) != 1:
        problems.append(f"expected exactly one wheel, found {len(wheels)}")
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist, found {len(sdists)}")

    for wheel in wheels:
        if not wheel.name.endswith("-py3-none-any.whl"):
            problems.append(
                f"{wheel.name} is not a universal wheel; the release ships "
                "py3-none-any so that installing needs no compiler"
            )
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        compiled = [n for n in names if n.endswith(COMPILED_SUFFIXES)]
        if compiled:
            problems.append(f"{wheel.name} contains compiled objects: {compiled}")
        for required in ("NOTICE.md", "LICENSE"):
            if not any(n.endswith(required) for n in names):
                problems.append(f"{wheel.name} does not carry {required}")

    for sdist in sdists:
        with tarfile.open(sdist) as archive:
            names = archive.getnames()
        if not any(n.endswith("_voigt_ext.pyx") for n in names):
            problems.append(
                f"{sdist.name} has no extension source; the documented source "
                "build for the libcerf backend would not work"
            )
        if any(n.endswith(COMPILED_SUFFIXES) for n in names):
            problems.append(f"{sdist.name} contains compiled objects")
        for required in ("NOTICE.md", "LICENSE"):
            if not any(n.endswith(required) for n in names):
                problems.append(f"{sdist.name} does not carry {required}")

    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dist = Path(argv[0]) if argv else Path("dist")
    if not dist.is_dir():
        print(f"::error::{dist} is not a directory")
        return 1

    problems = check(dist)
    for name in sorted(p.name for p in dist.iterdir() if p.is_file()):
        print(f"  {name}")
    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        return 1
    print("release shape is correct")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
