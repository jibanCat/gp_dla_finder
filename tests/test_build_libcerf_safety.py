"""``tools/build_libcerf.sh`` must never delete a directory it does not own.

An earlier version accepted any prefix and ran ``rm -rf`` on it whenever it
already existed, so ``tools/build_libcerf.sh /usr/local`` would have asked to
delete ``/usr/local``. PI ruling (increment-10 correction 3) requires that this
be fixed and tested, and the tests must name the dangerous targets explicitly
rather than checking the happy path.

Every test here stops the script **before** it does any network or build work, by
running it with an environment that makes the safety check the only thing that
runs. A test that actually built libcerf would be slow and would need the
network; the safety logic is what needs guarding.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_libcerf.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the build script is POSIX shell"
)


def run(prefix: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run the script with a prefix, expecting it to stop at the safety check.

    ``curl`` is shadowed by a failing stub, so if a prefix is *accepted* the run
    dies at the download instead of proceeding to build anything. That keeps the
    test fast and offline while still distinguishing "refused" from "accepted".
    """
    stub = ROOT / "build" / "_test_stub_bin"
    stub.mkdir(parents=True, exist_ok=True)
    fake_curl = stub / "curl"
    fake_curl.write_text("#!/bin/sh\nexit 42\n")
    fake_curl.chmod(0o755)

    return subprocess.run(
        ["bash", str(SCRIPT), prefix],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PATH": f"{stub}:{os.environ.get('PATH', '')}"},
    )


def assert_refused(result: subprocess.CompletedProcess, prefix: str) -> None:
    assert result.returncode != 0, f"{prefix!r} was accepted"
    assert "refusing to use prefix" in result.stderr, (
        f"{prefix!r} failed, but not at the safety check: {result.stderr[-400:]}"
    )
    assert "clearing" not in result.stdout.lower(), (
        f"{prefix!r} reached the deletion step"
    )


# --------------------------------------------------------------------------
# The targets that must never be accepted
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        "/",
        "/usr",
        "/usr/local",
        "/etc",
        "/opt",
        "/tmp",
        "/Users",
        "/System",
        "/var",
    ],
)
def test_system_directories_are_refused(prefix):
    assert_refused(run(prefix), prefix)


def test_the_home_directory_is_refused():
    home = str(Path.home())
    assert_refused(run(home), home)


def test_the_repository_root_is_refused():
    assert_refused(run(str(ROOT)), str(ROOT))


def test_the_whole_build_area_is_refused():
    target = str(ROOT / "build")
    assert_refused(run(target), target)


def test_an_unexpanded_variable_or_glob_is_refused():
    """A destructive target must never be an unresolved expansion."""
    for prefix in ("$UNSET_PREFIX/lib", "/tmp/foo*/bar", "`whoami`"):
        assert_refused(run(prefix), prefix)


def test_an_empty_prefix_is_refused():
    result = subprocess.run(
        ["bash", str(SCRIPT), ""], capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    assert "refusing to use prefix" in result.stderr


def test_a_relative_path_is_refused():
    assert_refused(run("relative/prefix"), "relative/prefix")


def test_a_shallow_absolute_path_is_refused():
    """Two components is too broad to claim ownership of."""
    assert_refused(run("/gpdlf-prefix"), "/gpdlf-prefix")


# --------------------------------------------------------------------------
# What must still be allowed
# --------------------------------------------------------------------------


def test_an_unrelated_non_empty_directory_is_refused(tmp_path):
    """The case that matters: someone else's populated directory."""
    target = tmp_path / "someones" / "existing" / "install"
    target.mkdir(parents=True)
    (target / "important.txt").write_text("do not delete me")

    result = run(str(target))
    assert_refused(result, str(target))
    # And it is still there.
    assert (target / "important.txt").is_file()


def test_an_empty_caller_supplied_directory_is_accepted(tmp_path):
    target = tmp_path / "a" / "b" / "empty-prefix"
    target.mkdir(parents=True)
    result = run(str(target))
    # Accepted, therefore it proceeds to the download, where the stub fails.
    assert "refusing to use prefix" not in result.stderr
    assert "using existing empty prefix" in result.stdout


def test_a_forged_manifest_does_not_grant_deletion(tmp_path):
    """File existence is not ownership evidence.

    An earlier revision deleted any directory containing a file named
    `gp_dla_finder_libcerf_build.json` -- and the test that covered it wrote `{}`,
    so an empty file was sufficient. Anyone able to drop a file into a directory
    could have made this script delete it. Only the script's own default prefix
    is now removable, so a forged manifest changes nothing.
    """
    target = tmp_path / "a" / "b" / "not-really-ours"
    (target / "lib").mkdir(parents=True)
    (target / "lib" / "gp_dla_finder_libcerf_build.json").write_text("{}")
    (target / "important.txt").write_text("do not delete me")

    result = run(str(target))
    assert_refused(result, str(target))
    assert (target / "important.txt").is_file()


def test_a_malformed_manifest_does_not_grant_deletion(tmp_path):
    target = tmp_path / "a" / "b" / "malformed"
    (target / "lib").mkdir(parents=True)
    (target / "lib" / "gp_dla_finder_libcerf_build.json").write_text("not json {{{")
    (target / "keep.txt").write_text("keep")

    assert_refused(run(str(target)), str(target))
    assert (target / "keep.txt").is_file()


def test_traversal_through_a_nonexistent_parent_is_canonicalised(tmp_path):
    """The hole review found: `..` under a parent that does not exist.

    The earlier resolver only canonicalised when the parent existed, so a path
    shaped like `<missing>/../../..//usr/local` could reach a system directory
    while evading the exact string comparisons.
    """
    # Canonicalises to exactly /usr/local, through a component that does not
    # exist. The earlier resolver would have left the `..` in place and compared
    # the raw string against its system-path list, which would not have matched.
    sneaky = "/usr/local/no-such-directory/.."
    assert Path(sneaky).resolve(strict=False) == Path("/usr/local")
    assert_refused(run(sneaky), sneaky)

    # And one that reaches the root.
    to_root = "/no-such-top-level/.."
    assert Path(to_root).resolve(strict=False) == Path("/")
    assert_refused(run(to_root), to_root)


def test_traversal_reaching_the_home_directory_is_refused():
    home = Path.home()
    sneaky = str(home / "no-such-dir" / "..")
    assert_refused(run(sneaky), sneaky)


def test_a_symlink_target_is_refused(tmp_path):
    """A deletion target must not be indirect."""
    real = tmp_path / "real-contents"
    real.mkdir()
    (real / "important.txt").write_text("do not delete me")
    link = tmp_path / "a" / "b" / "link-to-real"
    link.parent.mkdir(parents=True)
    link.symlink_to(real, target_is_directory=True)

    assert_refused(run(str(link)), str(link))
    assert (real / "important.txt").is_file()


def test_extra_positional_arguments_are_refused(tmp_path):
    """A destructive target must be unambiguous."""
    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "one"), str(tmp_path / "two")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "positional arguments" in result.stderr


def test_the_default_prefix_is_the_only_directory_cleared(tmp_path):
    """The one directory the script owns by construction."""
    default = ROOT / "build" / "libcerf-2.4-prefix"
    default.mkdir(parents=True, exist_ok=True)
    (default / "stale.txt").write_text("left over from an earlier build")
    try:
        result = run(str(default))
        assert "refusing to use prefix" not in result.stderr
        assert "clearing the default prefix" in result.stdout
        assert not (default / "stale.txt").exists()
    finally:
        import shutil

        shutil.rmtree(default, ignore_errors=True)


def test_a_new_nonexistent_prefix_is_accepted(tmp_path):
    target = tmp_path / "a" / "b" / "brand-new"
    result = run(str(target))
    assert "refusing to use prefix" not in result.stderr
