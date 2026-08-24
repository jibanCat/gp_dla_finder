#!/usr/bin/env python3
"""Refuse to release a version nobody asked for.

A typed confirmation says *that* someone meant to publish. It says nothing about
*what*. This checks the second question, before anything is built and again
after, because a name on PyPI cannot be reused and a wrong upload cannot be
recalled.

Three things must agree:

* the version the dispatcher typed;
* the version in ``pyproject.toml``; and
* the version in the built filenames.

And the version must be a real release. A ``.dev`` version is a working number
that changes whenever someone feels like it, and publishing one would put a
moving target on an index where every filename is permanent.

Usage::

    python tools/check_release_version.py --expected 0.1.0rc1
    python tools/check_release_version.py --expected 0.1.0rc1 --dist dist/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse(version: str):
    """``packaging.version.Version``, or ``None`` if it is not a version.

    An authoritative parser rather than a hand-written pattern. A regular
    expression that "looks like PEP 440" gets normalisation wrong in ways that
    matter here -- ``0.01.0`` and ``0.1.0`` are the same version to any index,
    and a grammar that accepts both as distinct canonical spellings would let
    two different strings name one file.

    ``packaging`` is a direct requirement of ``build``, which the release job
    installs before anything else runs, so this adds nothing to the bootstrap.
    """
    from packaging.version import InvalidVersion, Version

    try:
        return Version(version)
    except InvalidVersion:
        return None


#: What this gate accepts, stated plainly because "canonical PEP 440" alone
#: would be vague:
#:
#: * the string must already be the NORMALISED spelling of the version it
#:   denotes -- ``str(Version(x)) == x``. So ``0.1.0rc1`` is accepted and
#:   ``0.1.0-rc1``, ``0.1.0.rc1``, ``v0.1.0`` and ``0.01.0`` are not, even
#:   though a parser understands all of them. They name the same release
#:   through different spellings, and a release should have one name;
#: * no development segment and no local segment;
#: * no surrounding or embedded whitespace.
#:
#: Prereleases, post-releases and final releases are all accepted, so this same
#: path can validate ``0.1.0`` later and not only ``0.1.0rc1``.
RELEASE_VERSION_POLICY = (
    "already-normalised PEP 440, no .dev segment, no local segment, no whitespace"
)


def _read_toml(text: str) -> dict:
    """Parse TOML on any interpreter this package supports.

    ``tomllib`` is standard library from 3.11; this package supports 3.10, and
    importing it at module scope broke the 3.10 job while every local check ran
    on 3.12. ``tomli`` is the same parser under its pre-stdlib name and is a
    development dependency below 3.11.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
        import tomli as tomllib
    return tomllib.loads(text)


def project_version(root: Path) -> str:
    data = _read_toml((root / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def check(expected: str, declared: str, dist: Path | None = None) -> list[str]:
    """Problems, in the order a person would want to fix them."""
    problems: list[str] = []

    expected = expected or ""
    if not expected:
        return [
            "no expected version was given. The release path requires the "
            "version to be stated explicitly, so that a correct confirmation "
            "phrase cannot publish the wrong thing."
        ]

    # Refused, not stripped. Silently cleaning the input would mean the value
    # that was validated and the value the dispatcher typed are different
    # strings, and the second is the one a later step might use. An actionable
    # refusal is clearer than a quiet repair -- and it also removes any way for
    # a newline to reach a $GITHUB_OUTPUT line.
    if expected != expected.strip() or any(c.isspace() for c in expected):
        return [
            f"{expected!r} contains whitespace. State the exact version with "
            "no leading, trailing or embedded spaces."
        ]

    parsed = _parse(expected)
    if parsed is None:
        # Genuinely unparseable. The examples here must be things packaging
        # itself rejects -- 'v0.1.0' belongs in the other message, because it
        # parses fine and is refused for its spelling.
        problems.append(
            f"{expected!r} cannot be parsed as a PEP 440 version at all. "
            "Expected something like '0.1.0', '0.1.0rc1' or '1.2.0b3'."
        )
        return problems

    if str(parsed) != expected:
        # Parses, but is not the spelling this policy requires. 'v0.1.0',
        # '0.1.0-rc1' and '0.01.0' all land here: each is a real PEP 440
        # version that names the same release through a different string.
        problems.append(
            f"{expected!r} is a valid PEP 440 version, but not the normalised "
            f"spelling of it: it means {str(parsed)!r}. Use that instead, so "
            "the release has one name rather than several that resolve to it."
        )
    if parsed.is_devrelease:
        problems.append(
            f"{expected!r} is a development version. A prerelease such as "
            "'0.1.0rc1' or a final release such as '0.1.0' is fine; a .dev "
            "number moves whenever someone feels like it, and an index "
            "filename is permanent."
        )
    if parsed.local:
        problems.append(
            f"{expected!r} carries a local version segment. A local version "
            "describes one machine's build and cannot be published to an index."
        )

    if expected != declared:
        problems.append(
            f"the expected version {expected!r} does not match pyproject.toml, "
            f"which declares {declared!r}. Set the version in the reviewed "
            "release commit rather than overriding it here."
        )

    if dist is not None:
        problems.extend(_check_filenames(expected, dist))

    return problems


def _check_filenames(expected: str, dist: Path) -> list[str]:
    """The built files must carry the version too.

    Checked separately from the metadata because a stale ``dist/`` is exactly
    how the right metadata and the wrong bytes end up in the same upload.
    """
    problems: list[str] = []
    if not dist.is_dir():
        return [f"{dist} is not a directory, so no built files could be checked"]

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels:
        problems.append(f"no wheel in {dist}")
    if not sdists:
        problems.append(f"no sdist in {dist}")

    # Wheel: name-version-pytag-abitag-platform.whl
    for wheel in wheels:
        parts = wheel.name.split("-")
        built = parts[1] if len(parts) > 1 else "?"
        if built != expected:
            problems.append(f"{wheel.name} is version {built!r}, expected {expected!r}")
    # sdist: name-version.tar.gz
    for sdist in sdists:
        built = sdist.name[: -len(".tar.gz")].rsplit("-", 1)[-1]
        if built != expected:
            problems.append(f"{sdist.name} is version {built!r}, expected {expected!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", default="", help="the version being released")
    parser.add_argument(
        "--dist", type=Path, default=None, help="also check built filenames here"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help=(
            "on success, append version=<validated> to $GITHUB_OUTPUT. The "
            "workflow uses this rather than echoing its own input, so the value "
            "that reaches later jobs is provably the one that was validated."
        ),
    )
    args = parser.parse_args(argv)

    declared = project_version(args.root)
    problems = check(args.expected, declared, args.dist)

    print(f"  expected : {args.expected or '(none given)'}")
    print(f"  pyproject: {declared}")
    if args.dist is not None and args.dist.is_dir():
        for path in sorted(args.dist.iterdir()):
            if path.suffix in {".whl", ".gz"}:
                print(f"  built    : {path.name}")

    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        return 1

    if args.github_output:
        # What is exported is the string that was validated, unchanged. That is
        # sound because this policy REFUSES any input it would otherwise have
        # to alter -- whitespace and non-normalised spellings both fail above,
        # so nothing reaches here that differs from what was checked.
        #
        # It is worth being exact about the limit: `check()` returns problems,
        # not a value. If the policy ever normalised instead of refusing, this
        # would export the raw argument rather than the normalised one, and
        # would have to be changed at the same time.
        validated = args.expected
        if any(c.isspace() for c in validated):  # pragma: no cover - unreachable
            print("::error::refusing to emit a version containing whitespace")
            return 1
        destination = os.environ.get("GITHUB_OUTPUT")
        if not destination:
            print("::error::--github-output given but GITHUB_OUTPUT is not set")
            return 1
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write(f"version={validated}\n")
        print(f"exported version={validated}")

    print(f"release version {args.expected} confirmed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
