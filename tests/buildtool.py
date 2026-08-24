"""Running ``python -m build`` without the repository shadowing it.

``pip install -e .`` leaves a ``build/`` directory in the repository root.
Python treats it as a namespace package, so from the repository root:

* ``importlib.util.find_spec("build")`` succeeds -- and therefore so does
  ``pytest.importorskip("build")`` -- even when the build *tool* is not
  installed at all;
* ``python -m build`` then fails with "No module named build.__main__;
  'build' is a package and cannot be directly executed".

That combination is how the ``minimum-deps`` CI job, which installs only the
package and pytest, ran the build tests instead of skipping them, and failed on
a missing dependency it was never meant to have.

Both halves are fixed here: the availability check runs the tool rather than
importing a name, and every build runs from a neutral working directory with an
absolute source path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def require_build_tool() -> None:
    """Skip unless ``python -m build`` actually runs.

    Deliberately not ``importorskip``: the question is whether the tool can be
    executed, and the repository's own ``build/`` directory makes the import
    answer that question wrongly.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "build", "--version"],
            capture_output=True,
            text=True,
            cwd=Path.home(),
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:  # pragma: no cover
        pytest.skip(f"the build tool could not be run: {error}")
    if completed.returncode != 0:
        pytest.skip(
            "the 'build' tool is not installed "
            f"({completed.stderr.strip().splitlines()[-1:] or ['no output']})"
        )


def run_build(
    outdir: Path, *args: str, env: dict[str, str] | None = None
) -> list[Path]:
    """Build into ``outdir`` from a working directory that cannot shadow anything.

    ``cwd=outdir`` rather than the repository root, with the source directory
    passed absolutely, so ``build/`` in the checkout is never on the import path
    of the subprocess.
    """
    require_build_tool()
    outdir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "build", *args, "--outdir", str(outdir), str(ROOT)],
        capture_output=True,
        text=True,
        cwd=outdir,
        env={**os.environ, **(env or {})},
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return sorted(p for p in outdir.iterdir() if p.is_file())
