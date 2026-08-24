"""Optional compiled Voigt backend.

Everything about this package's metadata lives in ``pyproject.toml``. This file
exists for one reason: to *attempt* to build the libcerf-backed Voigt extension
during a normal ``pip install``, and to carry on without it when that is not
possible.

The fallback has to be genuinely safe, because the default install must not
require a compiler. Two mechanisms provide that:

* ``Extension(optional=True)`` tells setuptools that a failure to build this
  extension is not a failure to install the package;
* the extension is only offered to setuptools at all when a libcerf header and
  library are actually found, so the common case is a clean skip rather than a
  compiler error in the install log.

When the extension is absent the package runs on the NumPy backend, which is the
official backend and not a degraded mode. ``gp_dla_finder.voigt.available_backends()``
reports what a given installation actually has, and a backend that was requested
but not built raises rather than being silently substituted.

Overrides, for environments where libcerf is somewhere unusual:

    GP_DLA_FINDER_LIBCERF_INCLUDE=/path/to/include
    GP_DLA_FINDER_LIBCERF_LIB=/path/to/lib
    GP_DLA_FINDER_NO_COMPILE=1      # skip the extension entirely
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py

PACKAGE_DIR = Path(__file__).parent / "src" / "gp_dla_finder"
PYX = PACKAGE_DIR / "_voigt_ext.pyx"
GENERATED_C = PACKAGE_DIR / "_voigt_ext.c"

#: Prefixes searched for ``cerf.h`` and the libcerf shared library, in order.
#: Conda first: if the user is in a conda environment, that is almost always the
#: libcerf they mean.
_CANDIDATE_PREFIXES = (
    os.environ.get("CONDA_PREFIX"),
    sys.prefix,
    "/opt/homebrew",  # Apple silicon Homebrew
    "/usr/local",  # Intel Homebrew, hand-built installs
    "/usr",  # Debian/Ubuntu libcerf-dev, Fedora libcerf-devel
)

_LIB_SUFFIXES = (".so", ".dylib", ".a", ".dll.a")

#: Written next to the extension at build time. A compiled backend's numbers
#: depend on which libcerf it linked and how it was compiled, so a result that
#: names the backend but not the build is not reproducible.
BUILD_INFO = PACKAGE_DIR / "_build_info.py"


#: Manifest written by tools/build_libcerf.sh into a source-built prefix. It is
#: the authoritative record for a build this project made itself.
_BUILD_MANIFEST = "gp_dla_finder_libcerf_build.json"


def _libcerf_build_manifest(library_dir: str) -> dict:
    """Read the manifest tools/build_libcerf.sh leaves in its install prefix."""
    import json

    for candidate in (
        Path(library_dir) / _BUILD_MANIFEST,
        Path(library_dir).parent / _BUILD_MANIFEST,
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return {}
    return {}


def _libcerf_version(library_dir: str, explicit: bool) -> tuple[str, str]:
    """``(version, how it was determined)`` for the libcerf actually selected.

    Order matters, and the previous order was wrong: it ran a bare
    ``pkg-config --modversion libcerf`` first, which answers about whatever
    libcerf is on the default search path -- not the one being linked. When a
    prefix is chosen explicitly, that is a different library and the answer was
    simply the wrong number.

    Resolution now starts from the selected prefix and never falls back to the
    system default when the prefix was explicit.
    """
    import re
    import shutil
    import subprocess

    manifest = _libcerf_build_manifest(library_dir)
    if manifest.get("version"):
        return str(manifest["version"]), "build manifest"

    # pkg-config, but pointed at the selected prefix only.
    pkgconfig_dir = Path(library_dir) / "pkgconfig"
    if shutil.which("pkg-config") and pkgconfig_dir.is_dir():
        try:
            out = subprocess.check_output(
                ["pkg-config", "--modversion", "libcerf"],
                text=True,
                stderr=subprocess.DEVNULL,
                env={**os.environ, "PKG_CONFIG_LIBDIR": str(pkgconfig_dir)},
            )
            if out.strip():
                return out.strip(), "pkg-config in the selected prefix"
        except Exception:
            pass

    for candidate in sorted(Path(library_dir).glob("libcerf*")):
        match = re.search(r"libcerf[.-](\d+(?:\.\d+)*)", candidate.name)
        if match:
            return match.group(1), "inferred from the library filename"

    if not explicit and shutil.which("pkg-config"):
        # Only meaningful when the prefix was auto-detected as the default one.
        try:
            out = subprocess.check_output(
                ["pkg-config", "--modversion", "libcerf"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if out.strip():
                return out.strip(), "pkg-config on the default search path"
        except Exception:
            pass

    return "unknown", "not determined"


def _resolved_library(library_dir: str) -> tuple[str, str]:
    """``(real path, sha256)`` of the libcerf the extension will link against.

    Resolved through symlinks, because ``libcerf.dylib`` is usually a link and
    "which file did we actually link" is the question provenance has to answer.
    """
    import hashlib

    for suffix in (".so", ".dylib", ".a"):
        candidate = Path(library_dir) / f"libcerf{suffix}"
        if candidate.exists():
            real = candidate.resolve()
            try:
                digest = hashlib.sha256(real.read_bytes()).hexdigest()
            except OSError:
                digest = "unreadable"
            return str(real), digest
    return "not found", "unknown"


#: Compiler options whose argument is a path. Their *values* are machine-specific
#: and must not travel in provenance; their presence is not scientifically
#: interesting either, since the include search path does not change the numerics.
_PATH_VALUED_FLAGS = ("-I", "-isystem", "-L", "-B", "--sysroot", "-isysroot", "-F")


def _redact_flags(raw: str) -> tuple[str, int]:
    """Strip machine-specific paths out of a compiler flag string.

    Provenance is written into results, and results get shared. A private path in
    a packaged artefact has bitten this project once already, so flags are
    filtered rather than copied. What survives is what can change the numbers --
    optimisation level, architecture, fast-math and friends. What goes is include
    and library search paths, and anything else containing a path separator.

    Returns the redacted string and the number of tokens dropped, so the removal
    is visible rather than silent.
    """
    kept: list[str] = []
    dropped = 0
    tokens = raw.split()
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            dropped += 1
            continue
        if token in _PATH_VALUED_FLAGS:
            skip_next = True
            dropped += 1
            continue
        if (
            any(token.startswith(flag) for flag in _PATH_VALUED_FLAGS)
            and len(token) > 2
        ):
            dropped += 1
            continue
        if "/" in token or "\\" in token:
            dropped += 1
            continue
        kept.append(token)
    return " ".join(kept), dropped


def _redact_path(path: str) -> str:
    """Replace a home-directory prefix with ``~``, keeping the rest legible."""
    try:
        home = str(Path.home())
    except Exception:  # pragma: no cover
        return path
    return f"~{path[len(home) :]}" if home and path.startswith(home) else path


def _toolchain_family(compiler: str) -> str:
    """``clang`` / ``gcc`` / ``msvc`` rather than a path to somebody's compiler.

    The family is what can change the numbers; the path is machine-local and has
    no place in a record that gets shared.
    """
    name = Path(compiler.split()[0]).name.lower() if compiler.strip() else ""
    for family in ("clang", "gcc", "icc", "cl", "cc"):
        if family in name:
            return {"cl": "msvc", "cc": "cc"}.get(family, family)
    return "unknown"


def _write_build_info(include_dir: str, library_dir: str, explicit: bool) -> None:
    """Record what this build linked and how.

    Split in two, deliberately:

    ``SHAREABLE``
        goes into Result provenance and therefore into files people send each
        other. Version, content hash, backend identity, toolchain *family*. No
        paths, not even home-abbreviated ones.
    ``LOCAL``
        stays on the machine that built it, for diagnosing a build. Full paths
        live here and nowhere else.
    """
    import sysconfig

    real_path, digest = _resolved_library(library_dir)
    try:
        from Cython import __version__ as cython_version
    except ImportError:
        cython_version = None

    # The wrapper's flags and libcerf's own build flags are different things and
    # were previously conflated.
    wrapper_cflags, cflags_dropped = _redact_flags(
        sysconfig.get_config_var("CFLAGS") or ""
    )
    wrapper_opt, opt_dropped = _redact_flags(sysconfig.get_config_var("OPT") or "")
    compiler = os.environ.get("CC") or sysconfig.get_config_var("CC") or ""

    version, version_source = _libcerf_version(library_dir, explicit)
    manifest = _libcerf_build_manifest(library_dir)

    record = {
        "libcerf_version": version,
        "libcerf_version_source": version_source,
        # The identity that actually matters: the bytes, not where they sat.
        "libcerf_sha256": digest,
        "libcerf_provenance": manifest.get("provenance", "pre-built or unknown"),
        "libcerf_source_sha256": manifest.get("source_sha256"),
        "libcerf_build_flags": manifest.get("c_flags"),
        "libcerf_build_type": manifest.get("build_type"),
        "wrapper_toolchain": _toolchain_family(compiler),
        "wrapper_cflags": wrapper_cflags,
        "wrapper_opt_flags": wrapper_opt,
        "wrapper_flags_redacted_count": cflags_dropped + opt_dropped,
        "platform": sysconfig.get_platform(),
        "cython_version": cython_version,
    }
    local = {
        "libcerf_include_dir": include_dir,
        "libcerf_library_dir": library_dir,
        "libcerf_resolved_path": real_path,
        "compiler": compiler,
    }

    lines = [
        '"""Generated at build time. Do not edit, do not commit.',
        "",
        "Records which libcerf the compiled Voigt backend linked against and how it",
        "was compiled.",
        "",
        "SHAREABLE is what travels in Result provenance and therefore into files",
        "people send each other: version, content hash, toolchain family. No paths.",
        "LOCAL stays here, for diagnosing a build on the machine that made it.",
        '"""',
        "",
        "SHAREABLE = {",
    ]
    lines += [f"    {key!r}: {value!r}," for key, value in record.items()]
    lines += ["}", "", "LOCAL = {"]
    lines += [f"    {key!r}: {value!r}," for key, value in local.items()]
    lines += ["}", ""]
    BUILD_INFO.write_text("\n".join(lines))


def _library_dirs(root: Path) -> list[Path]:
    """Library directories to search under a prefix, in order.

    Debian and Ubuntu install shared libraries into a **multiarch** directory --
    ``/usr/lib/x86_64-linux-gnu`` -- not ``/usr/lib``. Searching only ``lib`` and
    ``lib64`` therefore missed libcerf on the most standard Linux setup there is:
    CI installed ``libcerf-dev`` successfully and the extension silently did not
    build. Found by actually running the matrix rather than reading the workflow.
    """
    import sysconfig

    directories = [root / "lib", root / "lib64"]
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        directories.insert(0, root / "lib" / multiarch)
    # Fall back to discovering the triplet, for interpreters that do not report it.
    directories.extend(sorted((root / "lib").glob("*-linux-gnu")))
    return [d for d in directories if d.is_dir()]


def _find_libcerf() -> tuple[str, str] | None:
    """Return ``(include_dir, library_dir)`` for libcerf, or ``None``."""
    override_include = os.environ.get("GP_DLA_FINDER_LIBCERF_INCLUDE")
    override_lib = os.environ.get("GP_DLA_FINDER_LIBCERF_LIB")
    if override_include and override_lib:
        return override_include, override_lib

    for prefix in _CANDIDATE_PREFIXES:
        if not prefix:
            continue
        root = Path(prefix)
        header = root / "include" / "cerf.h"
        if not header.is_file():
            continue
        for directory in _library_dirs(root):
            if any(
                (directory / f"libcerf{suffix}").exists() for suffix in _LIB_SUFFIXES
            ):
                return str(root / "include"), str(directory)
    return None


def _extensions() -> list[Extension]:
    if os.environ.get("GP_DLA_FINDER_NO_COMPILE"):
        return []

    explicit = bool(
        os.environ.get("GP_DLA_FINDER_LIBCERF_INCLUDE")
        and os.environ.get("GP_DLA_FINDER_LIBCERF_LIB")
    )
    found = _find_libcerf()
    if found is None:
        # Not an error. The NumPy backend is the official one; libcerf only adds
        # a second, catalogue-faithful Faddeeva implementation.
        return []
    include_dir, library_dir = found

    try:
        from Cython.Build import cythonize
    except ImportError:
        cythonize = None

    if cythonize is None and not GENERATED_C.is_file():
        # No Cython and no pre-generated C: nothing to build. Skipping keeps the
        # install clean rather than emitting a confusing compiler error.
        return []

    # RELATIVE to this file, not absolute. setuptools records the source path
    # verbatim in the sdist's SOURCES.txt, so an absolute path publishes the
    # build machine's home directory and username to anyone who downloads it.
    # PACKAGE_DIR is absolute because Path(__file__).parent is absolute whenever
    # setup.py is invoked by path, which `python -m build <root>` does.
    absolute = PYX if cythonize is not None else GENERATED_C
    source = os.path.relpath(absolute, Path(__file__).parent)

    try:
        import numpy as np

        numpy_include = np.get_include()
    except ImportError:
        return []

    _write_build_info(include_dir, library_dir, explicit=explicit)

    extension = Extension(
        "gp_dla_finder._voigt_ext",
        sources=[source],
        include_dirs=[numpy_include, include_dir],
        library_dirs=[library_dir],
        runtime_library_dirs=[library_dir] if sys.platform != "win32" else [],
        libraries=["cerf"],
        # The whole point: a build failure here must not fail the install.
        optional=True,
    )

    if cythonize is None:
        return [extension]

    built = cythonize([extension], language_level=3)
    # cythonize() rebuilds the Extension objects and drops `optional`, silently
    # resetting it to False. Left unfixed, any compiler failure -- a toolchain
    # without a usable SDK, a libcerf built for another architecture -- would
    # abort the whole install instead of falling back to the NumPy backend. Set
    # it again, and assert, so a future Cython release cannot quietly undo this.
    for module in built:
        module.optional = True
    assert all(module.optional for module in built)
    return built


class build_py(_build_py):
    """Copy the repository's NOTICE.md into the package.

    The notice is canonical at the repository root -- that is where a reader
    looks -- but it must also travel inside the wheel, because the asset
    attribution and redistribution terms have to reach anyone who installs the
    package without the source tree.
    """

    def run(self) -> None:
        notice = Path(__file__).parent / "NOTICE.md"
        if notice.is_file():
            self.copy_file(str(notice), str(PACKAGE_DIR / "NOTICE.md"))
        super().run()


setup(ext_modules=_extensions(), cmdclass={"build_py": build_py})
