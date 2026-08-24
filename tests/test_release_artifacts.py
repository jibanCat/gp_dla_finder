"""The v0.1 distribution shape: a universal wheel plus a source distribution.

PI ruling N85. ``pip install gp_dla_finder`` should be portable and need no
compiler: it gets the NumPy Voigt backend, which is the official one. A user who
wants the libcerf backend builds from the sdist and asks for it.

These tests build the artifacts, so they are slow and marked accordingly.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import zipfile

import pytest

from buildtool import ROOT, run_build

#: The documented release build. Recorded here as data, so the test and the
#: release instructions cannot drift: if this changes, the docs check below
#: fails until they agree.
RELEASE_WHEEL_ENV = {"GP_DLA_FINDER_NO_COMPILE": "1"}

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def universal_wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("wheel")
    (built,) = run_build(out, "--wheel", env=RELEASE_WHEEL_ENV)
    return built


@pytest.fixture(scope="module")
def sdist(tmp_path_factory):
    out = tmp_path_factory.mktemp("sdist")
    (built,) = run_build(out, "--sdist")
    return built


def test_the_release_wheel_is_tagged_universal(universal_wheel):
    """py3-none-any, so one file serves every platform and interpreter.

    A platform wheel here would mean most users silently fall back to building
    from source, or to no wheel at all on their platform.
    """
    assert universal_wheel.name.endswith("-py3-none-any.whl"), universal_wheel.name
    with zipfile.ZipFile(universal_wheel) as archive:
        (metadata,) = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
        text = archive.read(metadata).decode()
    assert "Tag: py3-none-any" in text
    assert "Root-Is-Purelib: true" in text


def test_the_release_wheel_carries_no_compiled_extension(universal_wheel):
    with zipfile.ZipFile(universal_wheel) as archive:
        compiled = [
            n
            for n in archive.namelist()
            if n.endswith((".so", ".pyd", ".dylib", ".dll"))
        ]
    assert not compiled, f"the universal wheel contains compiled objects: {compiled}"


def test_the_release_wheel_still_carries_the_data_assets(universal_wheel):
    """Portable must not mean stripped."""
    with zipfile.ZipFile(universal_wheel) as archive:
        names = archive.namelist()
    assert any(n.endswith("NOTICE.md") for n in names)
    assert [n for n in names if n.endswith(".npz")], "no sample grids in the wheel"
    assert [n for n in names if "/models/" in n], "no trained model in the wheel"


def test_the_sdist_keeps_the_optional_extension_buildable(sdist):
    """The source build is the documented route to the libcerf backend.

    If the sdist lost the extension source or the build script, that route would
    quietly stop existing while the documentation still described it.
    """
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    assert any(n.endswith("_voigt_ext.pyx") for n in names), "extension source missing"
    assert any(n.endswith("/setup.py") for n in names), "build script missing"
    assert any(n.endswith("/pyproject.toml") for n in names)


def test_the_documented_release_command_matches_what_is_tested():
    """The env var this test builds with must be the one the docs tell you to use."""
    instructions = (ROOT / "docs" / "install.md").read_text()
    for name in RELEASE_WHEEL_ENV:
        assert name in instructions, (
            f"{name} is how the release wheel is built but docs/install.md does "
            "not mention it"
        )


def test_the_default_numpy_backend_needs_no_compiler(universal_wheel, tmp_path):
    """Install the universal wheel into a clean environment and run a spectrum.

    The point of N85: the ordinary install path works without a compiler and
    uses the supported backend.
    """
    venv = tmp_path / "clean"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=600)
    python = venv / "bin" / "python"
    if not python.exists():  # pragma: no cover - Windows layout
        python = venv / "Scripts" / "python.exe"

    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", str(universal_wheel)],
        check=True,
        timeout=1800,
    )
    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import numpy as np;"
                "from gp_dla_finder import Config;"
                "from gp_dla_finder.finder import Finder;"
                "from gp_dla_finder.gp.spectrum import Spectrum;"
                "f = Finder(Config.desi_y3_fast(max_absorbers=1, enable_tau_eb=False),"
                "           warn_about_threads=False);"
                "w = np.arange(3600.0, 5600.0, 0.8);"
                "rng = np.random.default_rng(1);"
                "s = Spectrum(wavelength=w,"
                "             flux=1.0 + rng.normal(0, 0.2, w.size),"
                "             ivar=np.full_like(w, 25.0), z_qso=2.6,"
                "             mask=np.zeros_like(w, dtype=bool));"
                "r = f.run(s, targetid=1);"
                "print(r.status, r.provenance['backend_backend'])"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "completed" in completed.stdout
    # And it used the NumPy backend, not a compiled one it could not have had.
    assert "numpy" in completed.stdout


# --- nothing about the build machine ships ------------------------------------
#
# The packaged-asset scan in test_asset_hygiene.py reads files inside the
# INSTALLED package. It cannot see egg-info, generated C, or anything else that
# reaches a distribution by another route. This scans the artifacts themselves,
# every file, byte-wise -- which is what caught an absolute path to the build
# machine's home directory in the sdist's SOURCES.txt.

_PATH_LIKE = re.compile(
    rb"/(?:Users|home|nfs|pscratch|scratch|global|mnt|media|opt|var)/[A-Za-z0-9._-]+"
)

#: Path prefixes that legitimately appear in a distribution.
#:
#: The first two are deliberately private-LOOKING fixtures: they are the inputs
#: to the tests that check the sanitiser strips paths, and tests/ ships so an
#: installation can be verified where it is installed. The rest are the standard
#: library search locations setup.py probes for libcerf -- they describe the
#: filesystem layout of an operating system, not of a person.
_ALLOWED_PREFIXES = (
    b"/nfs/EXAMPLE",
    b"/Users/someone",
    b"/opt/homebrew",
    b"/opt/local",
    b"/usr/local",
    b"/var/empty",
)


def _offending_paths(members):
    offenders = []
    for name, data in members:
        for match in _PATH_LIKE.finditer(data):
            hit = match.group(0)
            if any(hit.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
                continue
            offenders.append(f"{name}: {hit.decode(errors='replace')}")
    return offenders


def test_the_wheel_carries_no_build_machine_paths(universal_wheel):
    with zipfile.ZipFile(universal_wheel) as archive:
        members = [(n, archive.read(n)) for n in archive.namelist()]
    offenders = _offending_paths(members)
    assert not offenders, "build-machine paths in the wheel:\n" + "\n".join(offenders)


def test_the_sdist_carries_no_build_machine_paths(sdist):
    """This is the one that failed.

    ``setup.py`` passed an absolute source path to ``Extension``, setuptools
    recorded it verbatim in ``SOURCES.txt``, and the sdist therefore published
    the build machine's username and directory layout. A stale ``egg-info`` in
    the working tree keeps such an entry alive across rebuilds, so if this fails
    after the fix, delete ``src/*.egg-info`` and build again.
    """
    with tarfile.open(sdist) as archive:
        members = []
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                members.append((member.name, handle.read()))
    offenders = _offending_paths(members)
    assert not offenders, "build-machine paths in the sdist:\n" + "\n".join(offenders)


def test_the_sdist_ships_no_generated_c(sdist):
    """It is a build artifact, and it carries the paths Cython saw.

    Cython embeds the include paths it used -- including the temporary build
    environment -- so the generated C makes the sdist both non-reproducible and
    a carrier of build-machine paths. Cython is a declared build requirement, so
    every PEP 517 build regenerates it from the .pyx.
    """
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    assert not [n for n in names if n.endswith("_voigt_ext.c")], (
        "the generated C file is in the sdist; MANIFEST.in should exclude it"
    )
    assert [n for n in names if n.endswith("_voigt_ext.pyx")], "the source is missing"
