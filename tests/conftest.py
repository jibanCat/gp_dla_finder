"""Shared test fixtures.

Some tests compare this package against the *reference implementation* it was
ported from (``desi_gpy_dla_detection``). That repository is not a dependency and
is not redistributable, so those tests are marked ``needs_reference`` and skip
cleanly when it is absent. Point ``GP_DLA_FINDER_REFERENCE`` at a checkout to run
them.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

_REFERENCE_ENV = "GP_DLA_FINDER_REFERENCE"


def _reference_path() -> Path | None:
    raw = os.environ.get(_REFERENCE_ENV)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if (path / "gpy_dla_detection").is_dir() else None


def _install_reference() -> Path:
    """Put the reference implementation on ``sys.path``, or skip the test.

    The reference imports ``desiutil.log``, a DESI-stack dependency this package
    deliberately does not have. A minimal stand-in is installed so the pure
    numerics modules import.
    """
    path = _reference_path()
    if path is None:
        pytest.skip(f"set {_REFERENCE_ENV} to a desi_gpy_dla_detection checkout")

    if "desiutil.log" not in sys.modules:
        import logging
        import types

        pkg = types.ModuleType("desiutil")
        pkg.__path__ = []  # mark as a package
        log_mod = types.ModuleType("desiutil.log")
        log_mod.log = logging.getLogger("desiutil-test-stub")
        log_mod.log.addHandler(logging.NullHandler())
        sys.modules.setdefault("desiutil", pkg)
        sys.modules["desiutil.log"] = log_mod

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


@pytest.fixture(scope="session")
def reference_repo() -> Path:
    """Path to the reference checkout, with it importable."""
    return _install_reference()


@pytest.fixture(scope="session")
def reference_voigt():
    """The reference implementation's ``voigt`` module, or skip."""
    _install_reference()
    from gpy_dla_detection import voigt as reference

    return reference


# --------------------------------------------------------------------------
# Optional-dependency discovery (PI ruling, increment-15 correction 8)
# --------------------------------------------------------------------------
#
# The CI guard against silent skips used to name test files in a shell list.
# L-20: nothing maintained that list, and test_finder.py -- which carries the
# end-to-end Finder -> FITS round trip -- was added without being added to it,
# so the round trip could have gone green while never running.
#
# So the marker is applied by DISCOVERY, not by hand. Any test module that
# mentions the optional dependency gets it automatically, and the `catalogue-io`
# job runs `-m needs_astropy` and fails on any skip. A future astropy-backed
# module is guarded the moment it is written.

_OPTIONAL_DEPENDENCY_MARKERS = {"astropy": "needs_astropy"}

_source_cache: dict[Path, str] = {}


def _module_source(path: Path) -> str:
    if path not in _source_cache:
        try:
            _source_cache[path] = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable test file
            _source_cache[path] = ""
    return _source_cache[path]


def module_uses_optional_dependency(source: str, dependency: str) -> bool:
    """Whether ``source`` actually imports ``dependency``.

    Parsed, not pattern-matched. A regex over the text matched the dependency
    name inside *string literals* -- including this rule's own unit test, which
    quotes example imports -- and that marked tests/test_asset_hygiene.py, whose
    unrelated environment skips then failed the CI guard.

    Recognises ``import astropy``, ``from astropy.x import y``, and
    ``pytest.importorskip("astropy")``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a test file that cannot compile
        return False

    def names_it(name: str | None) -> bool:
        return bool(name) and (name == dependency or name.startswith(f"{dependency}."))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(names_it(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and names_it(node.module):
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            attribute = getattr(func, "attr", None) or getattr(func, "id", None)
            if attribute == "importorskip" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and names_it(first.value):
                    return True
    return False


def pytest_collection_modifyitems(config, items):
    """Mark every test whose module imports a declared optional dependency."""
    for item in items:
        path = Path(str(getattr(item, "fspath", "")))
        if not path.name.endswith(".py"):
            continue
        source = _module_source(path)
        for dependency, marker in _OPTIONAL_DEPENDENCY_MARKERS.items():
            if module_uses_optional_dependency(source, dependency):
                item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture(scope="session")
def ladder_finder():
    """A Finder configured for the bounded, experimental M0/M1/M2 ladder.

    Session-scoped because loading the model and sample grid is the expensive
    part and nothing here mutates the Finder. Shared by the multi-absorber
    tests and the legacy-writer parity tests, so it lives here rather than in
    one of them.
    """
    from gp_dla_finder.config import Config
    from gp_dla_finder.finder import Finder

    return Finder(
        Config.desi_y3_fast(
            enable_tau_eb=False, max_absorbers=2, experimental_multi_absorber=True
        ),
        warn_about_threads=False,
    )
