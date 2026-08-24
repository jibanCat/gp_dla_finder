"""Packaged assets must not leak private paths, and must not be mutable.

These tests exist because both classes of defect were found by review rather than
by the suite: a trained file's attributes carried an absolute cluster path into a
committed provenance JSON, and the module-level forward-model tables were editable
in place by any caller.
"""

from __future__ import annotations

import re
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from gp_dla_finder import model as m
from gp_dla_finder import voigt as v

# --------------------------------------------------------------------------
# 1. No private filesystem paths in any packaged asset
# --------------------------------------------------------------------------

#: Absolute-path shapes that should never appear in a distributed file.
#: Deliberately broad: the leak that prompted this was a cluster path nobody had
#: thought to look for.
_PRIVATE_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home|nfs|pscratch|scratch|global|mnt|media|opt|var)/\S"),
    re.compile(r"[A-Za-z]:\\\\?\S"),  # Windows drive paths
    re.compile(r"~/\S"),  # home-relative paths
)

_TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".cff", ".csv"}


def _packaged_text_files():
    """Every text-ish file shipped inside the package's data directories."""
    root = resources.files("gp_dla_finder")
    stack, found = [root], []
    while stack:
        node = stack.pop()
        for child in node.iterdir():
            if child.is_dir():
                stack.append(child)
            elif child.name[child.name.rfind(".") :] in _TEXT_SUFFIXES:
                found.append(child)
    return found


#: Repository files that are copied into the distributions verbatim. Scanned at
#: their SOURCE, not only inside the built package: ``setup.py`` copies
#: ``NOTICE.md`` into ``src/gp_dla_finder/`` at build time, so a path added to
#: the root file is invisible to the packaged-asset scan until someone
#: reinstalls. That gap is not hypothetical -- it let a private path sit in a
#: committed NOTICE.md for one commit.
_SOURCE_DISTRIBUTED_FILES = ("NOTICE.md", "README.md", "CITATION.cff")


def test_no_source_file_that_ships_contains_a_private_path():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for name in _SOURCE_DISTRIBUTED_FILES:
        path = root / name
        if not path.is_file():
            continue
        for pattern in _PRIVATE_PATH_PATTERNS:
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                offenders.append(f"{name}: {match.group(0)[:80]}")
    assert not offenders, (
        "private paths in files that ship:\n"
        + "\n".join(offenders)
        + "\n\nDescribe the path instead of quoting it."
    )


def test_the_packaged_notice_matches_the_repository_one():
    """They are the same file, copied at build time.

    A stale copy is how the packaged scan passes while the source it was made
    from does not -- or the reverse.
    """
    root = Path(__file__).resolve().parents[1]
    packaged = root / "src" / "gp_dla_finder" / "NOTICE.md"
    if not packaged.is_file():
        pytest.skip("the packaged NOTICE.md is generated at build time")
    assert packaged.read_text() == (root / "NOTICE.md").read_text(), (
        "src/gp_dla_finder/NOTICE.md is stale; it is copied from the repository "
        "root at build time, so reinstall or re-copy it"
    )


def test_there_are_packaged_assets_to_scan():
    """Guards the scan below against silently passing on an empty file list."""
    names = [f.name for f in _packaged_text_files()]
    assert any(n.endswith(".json") for n in names), names


def test_no_packaged_asset_contains_a_private_path():
    offenders = []
    for handle in _packaged_text_files():
        text = handle.read_text(encoding="utf-8", errors="replace")
        for pattern in _PRIVATE_PATH_PATTERNS:
            for match in pattern.finditer(text):
                offenders.append(f"{handle.name}: {match.group(0)[:80]}")
    assert not offenders, "private paths in packaged assets:\n" + "\n".join(offenders)


def test_model_provenance_redacts_path_like_training_attributes():
    """The redaction keeps the run identity without confirming the location."""
    prov = m.model_provenance()
    redacted = prov["source"]["training_attrs_redacted"]
    assert "preload_source" in redacted, "expected the training-set path to be withheld"
    entry = redacted["preload_source"]
    assert "sha256_of_original" not in entry
    assert entry["training_run"] == "2lpt_loa124_nohcd_nobal_wide_v2_1778186324"
    for value in prov["source"]["training_attrs"].values():
        assert not isinstance(value, str) or "/" not in value


def test_converter_rejects_path_like_values_outside_the_allowlist():
    from tools.convert_model import SAFE_ATTR_KEYS, _sanitise_attrs, looks_like_path

    assert looks_like_path("/nfs/EXAMPLE/USER/run/trainset.h5")
    assert looks_like_path("~/data/x.h5")
    assert looks_like_path("C:\\Users\\EXAMPLE\\x.h5")
    assert not looks_like_path(1500)

    safe, redacted = _sanitise_attrs(
        {
            "lr": 0.005,
            "preload_source": "/nfs/EXAMPLE/USER/run/trainset.h5",
            "who": "me",
        }
    )
    assert safe == {"lr": 0.005}
    assert set(redacted) == {"preload_source", "who"}
    assert "lr" in SAFE_ATTR_KEYS


def test_allowlisted_key_holding_a_path_is_still_redacted():
    """An allowlist entry must not become a bypass if a trainer reuses the name."""
    from tools.convert_model import _sanitise_attrs

    safe, redacted = _sanitise_attrs({"n_spectra": "/nfs/EXAMPLE/USER/oops"})
    assert safe == {}
    assert "n_spectra" in redacted


# --------------------------------------------------------------------------
# 2. Forward-model tables are immutable
# --------------------------------------------------------------------------


def test_lsf_kernel_registry_cannot_be_extended_or_replaced():
    with pytest.raises(TypeError):
        v.LSF_KERNELS["evil"] = np.ones(7) / 7  # type: ignore[index]
    with pytest.raises(TypeError):
        v.LSF_KERNELS[v.PRODUCTION_KERNEL] = np.ones(7) / 7  # type: ignore[index]


def test_lsf_kernel_array_cannot_be_edited_in_place():
    kernel = v.lsf_kernel(v.PRODUCTION_KERNEL)
    with pytest.raises(ValueError, match="read-only"):
        kernel[3] = 0.4327  # the stale BOSS centre tap


def test_lsf_kernel_write_flag_cannot_be_re_enabled():
    """The setflags(write=True) bypass, closed.

    A read-only flag alone is reversible on an array that owns its storage. This
    is the exact escape route an earlier review demonstrated changing a later
    Voigt profile by ~1e-2.
    """
    kernel = v.lsf_kernel(v.PRODUCTION_KERNEL)
    assert not kernel.flags.owndata
    with pytest.raises(ValueError, match="WRITEABLE"):
        kernel.setflags(write=True)
    with pytest.raises((ValueError, AttributeError)):
        kernel.base.setflags(write=True)


def test_attempted_kernel_mutation_cannot_change_a_later_profile():
    """Before/after equality: the actual guarantee, not just that an error raised."""
    grid = np.arange(3600.0, 4400.0, 0.8)
    before = v.voigt_absorption(grid, nhi=10**20.8, z_dla=2.4)

    for attempt in (
        lambda: v.lsf_kernel(v.PRODUCTION_KERNEL).__setitem__(slice(None), 1 / 7),
        lambda: v.lsf_kernel(v.PRODUCTION_KERNEL).setflags(write=True),
        lambda: v.lsf_kernel(v.PRODUCTION_KERNEL).base.setflags(write=True),
        lambda: v.lsf_kernel(v.PRODUCTION_KERNEL).fill(1 / 7),
        lambda: v.LSF_KERNELS.__setitem__("desi-r3000-7tap", np.ones(7) / 7),
    ):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            attempt()

    after = v.voigt_absorption(grid, nhi=10**20.8, z_dla=2.4)
    assert np.array_equal(before, after)


@pytest.mark.parametrize(
    "table",
    [
        "TRANSITION_WAVELENGTHS",
        "OSCILLATOR_STRENGTHS",
        "LEADING_CONSTANTS",
        "GAMMAS_CGS",
    ],
)
def test_atomic_data_tables_are_read_only(table):
    array = getattr(v, table)
    with pytest.raises(ValueError, match="read-only"):
        array[0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)


def test_a_copied_kernel_is_still_writable():
    """Immutability must not stop legitimate downstream use."""
    kernel = v.lsf_kernel(v.PRODUCTION_KERNEL).copy()
    kernel[3] = 0.5  # must not raise


# --------------------------------------------------------------------------
# 3. Model-owned arrays and provenance are immutable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attr", ["rest_wavelengths", "mu", "M", "log_omega"])
def test_model_arrays_are_read_only(attr):
    model = m.load_model()
    with pytest.raises(ValueError, match="read-only"):
        getattr(model, attr)[0] = 0.0


@pytest.mark.parametrize("attr", ["rest_wavelengths", "mu", "M", "log_omega"])
def test_model_array_write_flag_cannot_be_re_enabled(attr):
    """Every model-owned array, against the setflags bypass, not just assignment."""
    model = m.load_model()
    array = getattr(model, attr)
    assert not array.flags.owndata
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)
    with pytest.raises((ValueError, AttributeError)):
        array.base.setflags(write=True)


def test_model_survives_every_mutation_route_and_still_computes_the_same():
    """Before/after equality on a value derived from the model, not just raises."""
    model = m.load_model()
    before = float(model.mu[:100].sum()), float(model.M[:100].sum())
    for array in (model.rest_wavelengths, model.mu, model.M, model.log_omega):
        for attempt in (
            lambda a=array: a.__setitem__(0, 0.0),
            lambda a=array: a.setflags(write=True),
            lambda a=array: a.base.setflags(write=True),
            lambda a=array: a.fill(0.0),
        ):
            with pytest.raises((ValueError, AttributeError)):
                attempt()
    assert (float(model.mu[:100].sum()), float(model.M[:100].sum())) == before


def test_model_does_not_alias_the_arrays_it_was_constructed_from():
    """A caller keeping a reference must not be able to reach into the model."""
    caller_owned = np.linspace(900.0, 1300.0, 50)
    model = m.GPModel(
        name="t",
        rest_wavelengths=caller_owned,
        mu=np.ones(50),
        M=np.zeros((50, 2)),
        log_omega=np.zeros(50),
        log_c_0=0.0,
        log_tau_0=0.0,
        log_beta=0.0,
    )
    caller_owned[0] = -999.0
    assert model.rest_wavelengths[0] == 900.0


def test_mutation_attempts_cannot_affect_a_freshly_loaded_model():
    first = m.load_model()
    for attr in ("rest_wavelengths", "mu", "M", "log_omega"):
        with pytest.raises(ValueError):
            getattr(first, attr).fill(0.0)

    second = m.load_model()
    for attr in ("rest_wavelengths", "mu", "M", "log_omega"):
        assert np.array_equal(getattr(first, attr), getattr(second, attr))
    assert second.mu.max() > 0.0


def test_model_provenance_is_immutable_all_the_way_down():
    prov = m.model_provenance()
    with pytest.raises(TypeError):
        prov["name"] = "spoofed"  # type: ignore[index]
    with pytest.raises(TypeError):
        prov["source"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        prov["conversion"]["arrays"]["mu"]["float32_lossless"] = False  # type: ignore[index]


def test_model_provenance_lists_are_frozen_too():
    prov = m.model_provenance()
    unused = prov["source"]["datasets_unused"]
    assert isinstance(unused, tuple)
    with pytest.raises(TypeError):
        unused[0] = "x"  # type: ignore[index]


def test_loaded_model_provenance_is_also_deeply_frozen():
    model = m.load_model()
    with pytest.raises(TypeError):
        model.provenance["source"]["sha256"] = "0" * 64  # type: ignore[index]


# --------------------------------------------------------------------------
# 4. Prior-table arrays are immutable too
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attr", ["z_qsos", "cumulative_absorbers"])
def test_prior_arrays_resist_every_mutation_route(attr):
    from gp_dla_finder import load_prior

    prior = load_prior()
    array = getattr(prior, attr)
    assert not array.flags.owndata
    for attempt in (
        lambda: array.__setitem__(0, 0),
        lambda: array.setflags(write=True),
        lambda: array.base.setflags(write=True),
        lambda: array.fill(0),
    ):
        with pytest.raises((ValueError, AttributeError)):
            attempt()


# --------------------------------------------------------------------------
# 5. Built distributions carry the notice, the assets and the builders
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_built_wheel_and_sdist_contain_notice_assets_and_tools(tmp_path):
    """Verify the artifacts, not the source tree.

    The reproducibility and attribution claims are only true if the notice and
    the asset builders actually ship.
    """
    import tarfile
    import zipfile

    from buildtool import run_build

    # run_build rather than a bare `python -m build`: `pip install -e .` leaves a
    # build/ directory in the repository root, which Python treats as a namespace
    # package. That makes importorskip("build") succeed where the build TOOL is
    # absent, and then `python -m build` fails on a missing __main__.
    run_build(tmp_path)

    wheel = next(tmp_path.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert "gp_dla_finder/NOTICE.md" in names
    assert any(n.endswith("LICENSE") for n in names)
    assert any(n.endswith(".npz") and "data/models/" in n for n in names)
    assert any(n.endswith(".npz") and "data/priors/" in n for n in names)

    sdist = next(tmp_path.glob("*.tar.gz"))
    inner = {n.split("/", 1)[1] for n in tarfile.open(sdist).getnames() if "/" in n}
    assert "NOTICE.md" in inner
    assert "CITATION.cff" in inner
    assert "tools/convert_model.py" in inner
    assert "tools/build_prior_table.py" in inner
    # The benchmark harness ships too: a performance claim that cannot be
    # re-measured by the reader is not a baseline (increment-7 correction 4).
    assert "tools/benchmark.py" in inner
    # The optional extension ships as source, so an sdist install can build it.
    assert "setup.py" in inner
    assert "src/gp_dla_finder/_voigt_ext.pyx" in inner
    # Compiled artifacts must never travel in a source distribution.
    assert not [n for n in inner if n.endswith((".so", ".dylib", ".pyd"))]


# --------------------------------------------------------------------------
# 5a. The README banner asset
# --------------------------------------------------------------------------


def test_the_logo_is_present_and_unmodified():
    """PI-supplied artwork, pinned by hash.

    Pinned because the NOTICE.md entry records this specific image's
    provenance, ownership and licence. A silent replacement would leave all
    three describing something else.
    """
    import hashlib

    root = Path(__file__).resolve().parents[1]
    logo = root / "logo.jpg"
    assert logo.is_file(), "logo.jpg is missing from the repository root"
    digest = hashlib.sha256(logo.read_bytes()).hexdigest()
    assert digest == (
        "10590b53395cfaf58aadd5c8295493c3f0453f5ca9918484bf2e689008b2fc28"
    ), "logo.jpg does not match the hash recorded in NOTICE.md"


def test_the_logo_licence_is_recorded_wherever_the_logo_is_offered():
    """CC BY 4.0, adopted by PI ruling (increment 26 §B).

    In both files, because someone who reads only the README should not have to
    open NOTICE.md to learn whether they may reuse the artwork -- and the two
    must not be able to drift apart silently.
    """
    root = Path(__file__).resolve().parents[1]
    for name in ("NOTICE.md", "README.md"):
        text = (root / name).read_text()
        assert "CC BY 4.0" in text, f"{name} does not state the logo licence"
        assert "creativecommons.org/licenses/by/4.0" in text, (
            f"{name} names the licence without linking it"
        )
        lowered = text.lower()
        assert "trademark" in lowered, (
            f"{name} does not say the licence grants no trademark rights"
        )
        assert "endorsement" in lowered, (
            f"{name} does not say reuse must not imply endorsement"
        )


def test_the_readme_embeds_the_logo_by_relative_path():
    """No hotlink to a temporary attachment or an external host.

    Plain Markdown image syntax, not <picture>. GitHub's mobile app does not
    rewrite relative paths inside raw HTML, so the <picture> banner rendered as
    a BROKEN LINK there however correct the path was. Markdown image syntax is
    rewritten by every GitHub renderer.

    One file therefore has to serve both themes, which is what the universal
    variant is for: measured at 4.5:1 contrast on white and 4.2:1 on GitHub's
    dark canvas rather than tuned for either.
    """
    import re as _re

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    header = readme.split("## Installation")[0]
    # Comments explain the rule; they must not be read as breaking it.
    visible = _re.sub(r"<!--.*?-->", "", header, flags=_re.S)

    assert _re.search(r"!\[[^\]]*\]\(docs/_static/logo-universal\.svg\)", visible), (
        "the banner must be a Markdown image on the universal SVG wrapper"
    )
    assert "GP DLA Finder" in visible, "the banner has no alt text"

    # The renderer traps, asserted as absences.
    assert "<picture>" not in visible, "GitHub's app renders <picture> as broken"
    assert "srcset=" not in visible, "srcset only works inside <picture>"

    # The LOGO comes from this repository, not a hotlinked attachment or an
    # external host. CI badges are legitimately external and are not the point.
    logo_references = _re.findall(r"!\[[^\]]*\]\(([^)]*logo[^)]*)\)", visible)
    assert logo_references, "no logo reference found"
    for target in logo_references:
        assert not target.startswith("http"), f"logo hotlinked: {target}"
        assert target.startswith("docs/_static/"), f"logo path is odd: {target}"


def test_the_derived_logo_variants_exist_and_are_transparent():
    """Keyed-out background, so the artwork is not a white box on a dark page."""
    root = Path(__file__).resolve().parents[1]
    for name in ("logo-light.png", "logo-dark.png", "logo-universal.png"):
        path = root / "docs" / "_static" / name
        assert path.is_file(), f"{name} is missing; run tools/make_logo_variants.py"
        # PNG signature, then verify an alpha channel without needing Pillow:
        # colour type 6 (truecolour with alpha) at byte 25 of the IHDR chunk.
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert data[25] == 6, f"{name} has no alpha channel"


def test_the_readme_promises_no_estimator_that_does_not_exist():
    """MAP positions and 1-sigma errors have no estimator yet.

    Documentation that promises them is a factual error, not a rounding of the
    truth, and it was one (PI ruling, increment-9 correction 8).
    """
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    assert "with MAP `(z_abs, log10 N_HI)` estimates and 1σ errors" not in readme
    # And the four distinct quantities stay distinguished. All four, not two: a
    # line-wrap once split "FILTER screening score" across two lines, and only
    # the half that was on the list caught it.
    for phrase in (
        "model evidence",
        "model-posterior probability",
        "conditional parameter posterior",
        "FILTER screening score",
    ):
        assert phrase in readme, f"README no longer names the {phrase!r}"


def test_the_logo_does_not_ship_in_the_distributions():
    """Presentation, not a package asset."""
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text()
    assert "logo.jpg" not in manifest
    pyproject = (root / "pyproject.toml").read_text()
    assert "logo.jpg" not in pyproject


# --------------------------------------------------------------------------
# 5b. Shareable provenance must carry no machine-local paths
# --------------------------------------------------------------------------


def test_no_backend_provenance_field_contains_a_path():
    """Provenance is copied into results, and results get sent to other people.

    A build directory is machine-local information with no scientific content.
    What identifies a library is its version and content hash; what can change
    the numbers is the toolchain family and optimisation flags. Paths are neither,
    and this project has already shipped one private path by accident
    (PI ruling, increment-9 correction 3).

    Every field of every backend's provenance is scanned, not a chosen few, so a
    field added later cannot quietly reintroduce a leak.
    """
    from gp_dla_finder import voigt as v

    home = str(Path.home())
    for name in v.available_backends():
        for key, value in v.backend_provenance(name).items():
            text = str(value)
            assert home not in text, f"{name}.{key} leaks the home directory"
            assert "~/" not in text, (
                f"{name}.{key} contains a home-abbreviated path; abbreviating is "
                "not redaction"
            )
            # An absolute path as a whole value, or embedded as a token.
            assert not re.match(r"^[A-Za-z]:[\\/]", text), f"{name}.{key} is a path"
            for token in text.split():
                assert not token.startswith("/"), (
                    f"{name}.{key} contains the absolute path {token!r}"
                )


def test_local_diagnostics_are_reachable_but_separate():
    """The paths still exist -- somewhere a person can look, not in a result."""
    from gp_dla_finder import voigt as v

    if "libcerf" not in v.available_backends():
        pytest.skip("the optional libcerf extension was not built here")

    local = dict(v.backend_local_diagnostics("libcerf"))
    shareable = dict(v.backend_provenance("libcerf"))
    assert local, "local diagnostics should carry the build paths"
    assert set(local) & set(shareable) == set(), (
        "a field appears in both the shareable and local records; the split is "
        "the whole point"
    )


def test_libcerf_version_is_resolved_from_the_selected_build():
    """Not from whatever libcerf happens to sit on the default search path.

    A bare ``pkg-config --modversion libcerf`` answers about the system default,
    which is a *different library* whenever an explicit prefix was selected. The
    recorded source of the answer is asserted, so a regression to the wrong
    lookup is visible (PI ruling, increment-9 correction 4).
    """
    from gp_dla_finder import voigt as v

    if "libcerf" not in v.available_backends():
        pytest.skip("the optional libcerf extension was not built here")

    record = dict(v.backend_provenance("libcerf"))
    assert record["libcerf_version"] != "unknown"
    assert record["libcerf_version_source"] in {
        "build manifest",
        "pkg-config in the selected prefix",
        "inferred from the library filename",
        "pkg-config on the default search path",
    }
    # libcerf's own build flags are recorded separately from the wrapper's.
    assert "wrapper_cflags" in record
    assert "libcerf_build_flags" in record


# --------------------------------------------------------------------------
# 6. The optional extension must never be able to fail an install
# --------------------------------------------------------------------------


def test_the_compiled_extension_is_marked_optional():
    """A compiler failure must degrade to the NumPy backend, not fail the install.

    This is a regression guard with teeth. ``Cython.Build.cythonize`` rebuilds the
    Extension objects it is given and **drops** ``optional``, silently resetting
    it to False -- which was caught here by a real build failure on a toolchain
    with no usable SDK: the install aborted instead of falling back. setup.py sets
    the flag again after cythonizing, and this test fails if that ever stops
    happening.
    """
    import importlib.util

    # setuptools is a BUILD-time requirement, and Python 3.12 stopped bundling it
    # in fresh environments. Importing setup.py therefore fails on 3.12/3.13
    # unless it happens to be installed -- which is not a defect in the package.
    pytest.importorskip("setuptools", reason="setuptools is a build-time dependency")

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_gpdlf_setup", root / "setup.py")
    module = importlib.util.module_from_spec(spec)
    # Loading setup.py executes setup(); with no command-line arguments
    # setuptools parses argv, so run it with a harmless one.
    argv = sys.argv
    sys.argv = ["setup.py", "--version"]
    try:
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
    finally:
        sys.argv = argv

    extensions = module._extensions()
    if not extensions:
        pytest.skip("libcerf not present, so no extension is offered to setuptools")
    assert all(extension.optional for extension in extensions), (
        "the compiled extension is not marked optional; a compiler failure would "
        "abort the install instead of falling back to the NumPy backend"
    )


def test_the_fallback_logo_is_legible_on_both_github_themes():
    """The mobile-app fix, checked numerically rather than by eye.

    The fallback is what renders wherever <picture> media queries are ignored, so
    it has to work on white and on GitHub's dark canvas at once. WCAG contrast of
    the artwork's darkest ink against both, measured on the committed file.
    """
    pytest.importorskip("PIL", reason="logo checks need Pillow")
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    image = Image.open(root / "docs" / "_static" / "logo-universal.png")
    rgb = np.asarray(image.convert("RGB")).astype(float)
    alpha = np.asarray(image.convert("RGBA"))[..., 3]

    def relative_luminance(colour):
        c = np.asarray(colour, dtype=float) / 255.0
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    def contrast(a, b):
        hi, lo = max(a, b), min(a, b)
        return (hi + 0.05) / (lo + 0.05)

    # The darkest opaque ink: the title text.
    opaque = alpha > 200
    darkness = rgb.mean(axis=2)
    ink = rgb[opaque & (darkness < np.percentile(darkness[opaque], 5))].mean(axis=0)
    ink_luminance = relative_luminance(ink)

    on_white = contrast(ink_luminance, relative_luminance((255, 255, 255)))
    on_dark = contrast(ink_luminance, relative_luminance((13, 17, 23)))

    assert on_white >= 3.0, f"unreadable on a light theme: {on_white:.2f}:1"
    assert on_dark >= 3.0, f"unreadable on a dark theme: {on_dark:.2f}:1"


# --- the optional-dependency guard discovers, it does not remember ----------
#
# PI ruling, increment-15 correction 8 (closes L-20). The previous guard named
# test files in a shell list; test_finder.py was written without being added to
# it. These tests fail if the discovery mechanism stops working, which is the
# only thing standing between a new astropy-backed module and a silent skip.


def test_a_new_astropy_module_would_be_discovered():
    """The discovery rule itself, as a unit.

    Deliberately not by writing a probe file into tests/ and shelling out: that
    raced with other pytest processes and broke collection in the job where
    astropy is intentionally absent.
    """
    from conftest import module_uses_optional_dependency as uses

    for source in (
        "import astropy\n",
        "from astropy.io import fits\n",
        "from astropy.table import Table  # noqa: E402\n",
        'fits = pytest.importorskip("astropy.io.fits")\n',
        "pytest.importorskip('astropy')\n",
        "import numpy as np\nfrom astropy import units\n",
    ):
        assert uses(source, "astropy"), f"missed: {source!r}"

    for source in (
        "import numpy as np\n",
        "# astropy is mentioned only in this comment\n",
        '"""A docstring naming astropy."""\n',
        "ASTROPY_NOTE = 'astropy'\n",
        # The one that actually bit: a module quoting example imports as DATA.
        # A regex over the text matched these, marked this very file, and its
        # unrelated environment skips then failed the CI guard.
        'EXAMPLES = ["import astropy", "from astropy.io import fits"]\n',
        "SNIPPET = 'pytest.importorskip(\"astropy\")'\n",
        'def f():\n    """See: from astropy import units."""\n',
        # A relative import that merely shares the name.
        "from .astropy import helper\n",
    ):
        assert not uses(source, "astropy"), f"false positive: {source!r}"

    # And a near-miss name must not match.
    assert not uses("import astropy_helpers\n", "astropy")
    assert not uses("from astropyx.io import fits\n", "astropy")


def test_the_known_astropy_modules_carry_the_marker():
    """The modules CI must never silently skip.

    Reads the marker off collected items in-process. Shelling out to
    `--collect-only` returned exit code 5 wherever astropy is absent, because
    the modules skip at import and nothing is collected -- which made this test
    fail for the wrong reason in two jobs.
    """
    from conftest import module_uses_optional_dependency as uses

    root = Path(__file__).resolve().parents[1]
    discovered = {
        path.name
        for path in (root / "tests").glob("test_*.py")
        if uses(path.read_text(), "astropy")
    }
    for expected in ("test_catalogue.py", "test_finder.py", "test_examples.py"):
        assert expected in discovered, (
            f"{expected} would not be marked needs_astropy; the CI guard would "
            f"not cover it. Discovered: {sorted(discovered)}"
        )


def test_no_module_needs_both_the_reference_and_astropy():
    """A test needing both dependencies runs in NO continuous-integration job.

    The two live in different jobs and neither has the other:

    * ``canonical-parity`` mounts the reference checkout, and its pinned image
      has no astropy;
    * ``catalogue-io`` installs astropy, and has no reference checkout.

    So a module marked ``needs_reference`` that also imports astropy skips in
    both -- and the failure is silent, because ``canonical-parity`` only checks
    that *something* passed. ``tests/test_legacy_writer_parity.py`` was written
    that way and ran on no machine but the author's, which is exactly the
    silent-skip class this file already guards for astropy alone (L-20).

    Splitting is the fix, not weakening a job: compare against the reference
    where the reference is, and check the FITS round trip where astropy is.
    """
    from conftest import module_uses_optional_dependency as uses

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((root / "tests").glob("test_*.py")):
        source = path.read_text()
        if not uses(source, "astropy"):
            continue
        # Cheap and deliberate: the marker is applied at module scope in every
        # reference-parity module we have, so a textual check is enough and
        # avoids importing modules whose dependencies may be absent.
        if "pytest.mark.needs_reference" in source:
            offenders.append(path.name)

    assert not offenders, (
        "these modules need BOTH the reference checkout and astropy, so they "
        f"skip in every CI job: {offenders}. Split them."
    )


#: Shorthand that means something only to someone with the project's private
#: decision ledger. A public reader has no way to resolve "N83" or
#: "increment 26", so shipped text must give the durable technical reason
#: instead.
_GOVERNANCE_SHORTHAND = (
    re.compile(r"\.claude(?:/|\\)"),
    re.compile(r"PI ruling"),
    re.compile(r"increment[- ]\d"),
    re.compile(r"\bN\d{1,3}\b(?!_)"),
)

#: Everything a user reads as the package: the installed code and its notices,
#: the published documentation, the packaging metadata, and the parity
#: environment that ships beside them.
#:
#: ``tests/`` is deliberately NOT here, and this is a **deferred cleanup rather
#: than a permanent exemption**. It ships in the sdist, so it is technically
#: readable, but it is internal source rather than documentation or a notice.
#: Roughly 75 references across 28 files remain, and rewriting them immediately
#: before the first release candidate would be a large diff with no effect on
#: runtime behaviour or on any scientific claim.
#:
#: Post-release-candidate work: replace them gradually with self-contained
#: technical explanations. When doing so, **keep the reasoning** -- the point is
#: to make the "why" readable without the private ledger, not to delete it along
#: with the label.
#:
#: New tests should state their rationale directly rather than adding a
#: reference here.
_PUBLIC_TEXT_ROOTS = (
    ("src/gp_dla_finder", (".py", ".pyx", ".md", ".json")),
    ("docs", (".md", ".py")),
    # parity/ and tools/ ship in the sdist, so a user who downloads the source
    # reads them too. Neither is user-facing documentation, but both are
    # distributed, and both were small enough to sweep.
    ("parity", (".md", ".txt", ".lock", "")),
    ("tools", (".py",)),
)
_PUBLIC_TEXT_FILES = (
    "README.md",
    "NOTICE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    # Ships in both artifacts and is rendered on the package index page.
    "pyproject.toml",
    "MANIFEST.in",
    "setup.py",
)


def _public_text_paths(root: Path):
    for name in _PUBLIC_TEXT_FILES:
        path = root / name
        if path.is_file():
            yield path
    for folder, suffixes in _PUBLIC_TEXT_ROOTS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in suffixes:
                continue
            # Build output and the generated C, neither of which is authored.
            if "_build" in path.parts or path.name.endswith("_voigt_ext.c"):
                continue
            yield path


def test_no_shipped_text_cites_the_private_decision_ledger():
    """Public prose must give the reason, not the reference.

    "Refused under N83" tells a user nothing they can act on. "The prior is
    baked into the sample grid, so a configuration that disagrees would record
    a prior the calculation never used" tells them why, and survives the ledger
    being unavailable to them.

    This is about the shorthand only. The scientific caveats it used to sit
    beside must stay exactly as strong.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in _public_text_paths(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _GOVERNANCE_SHORTHAND:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(root)}:{line}: {match.group(0)!r}")
    assert not offenders, (
        "private decision references in text that ships to users:\n"
        + "\n".join(offenders)
        + "\n\nReplace with the durable technical reason. Private notes under "
        ".claude/ may keep the ruling numbers."
    )


# --- documentation figures are generated, and must match their generator -----
#
# The figures are computed from real package code (Voigt profiles, the actual
# per-sample integrand, the real coarse-scan size). A committed figure that no
# longer matches what the code produces is a false statement about the package,
# which is exactly the failure mode the logo variants already have a guard for.


def test_every_documentation_figure_has_both_theme_variants():
    root = Path(__file__).resolve().parents[1]
    figures = root / "docs" / "_static" / "figures"
    assert figures.is_dir(), "docs/_static/figures/ is missing"

    names = {p.name for p in figures.glob("*.svg")}
    assert names, "no figures committed"
    stems = {n.rsplit("-", 1)[0] for n in names}
    for stem in stems:
        for theme in ("light", "dark"):
            assert f"{stem}-{theme}.svg" in names, f"{stem} has no {theme} variant"


def test_the_figure_variants_are_not_the_same_file():
    """A dark variant that is a copy of the light one is the logo bug again."""
    root = Path(__file__).resolve().parents[1]
    figures = root / "docs" / "_static" / "figures"
    stems = {p.name.rsplit("-", 1)[0] for p in figures.glob("*.svg")}
    for stem in stems:
        light = (figures / f"{stem}-light.svg").read_bytes()
        dark = (figures / f"{stem}-dark.svg").read_bytes()
        assert light != dark, f"{stem}: the dark variant is identical to light"


def test_the_figures_carry_no_opaque_background():
    """Transparent, so neither theme paints a white slab on a dark page."""
    root = Path(__file__).resolve().parents[1]
    for path in (root / "docs" / "_static" / "figures").glob("*.svg"):
        head = path.read_text(errors="replace")[:4000]
        assert 'style="fill: #ffffff"' not in head, f"{path.name} paints a canvas"


@pytest.mark.slow
def test_the_committed_figures_match_the_generator():
    """Regenerate and compare the plotted DATA. Requires the `docs` extra.

    Not a byte comparison of the SVGs -- those differ between platforms.
    """
    import subprocess
    import sys

    pytest.importorskip("matplotlib", reason="figure generation needs matplotlib")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools" / "make_docs_figures.py"), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        "committed figures differ from the generator; re-run "
        "`python tools/make_docs_figures.py`\n" + completed.stdout + completed.stderr
    )


# --- rendered size and badge contracts (GitHub app) -------------------------
#
# The README references the logo by plain Markdown, which carries NO width
# attribute, so GitHub renders it at its intrinsic size. At 640 px it dominated
# the page in the GitHub app. The size contract therefore lives in the asset,
# not in the markup, and has to be asserted there.


def test_the_raster_variants_keep_the_source_resolution():
    """Resolution is image information; it is not a layout mechanism.

    An earlier version downsampled the universal variant to 260 px so plain
    Markdown would render it small. That discarded image data to achieve a
    layout effect. Display size is now carried by the SVG wrapper instead.
    """
    pytest.importorskip("PIL", reason="needs Pillow")
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    with Image.open(root / "logo.jpg") as canonical:
        source_edge = max(canonical.size)

    for name in ("logo-light.png", "logo-dark.png", "logo-universal.png"):
        with Image.open(root / "docs" / "_static" / name) as image:
            assert max(image.size) == source_edge, (
                f"{name} is {image.size}; the derived variants keep the "
                f"canonical {source_edge}px resolution"
            )


def test_the_display_wrapper_states_a_small_rendered_size():
    """Layout lives in the SVG's width/height, separately from resolution."""
    import re as _re

    root = Path(__file__).resolve().parents[1]
    svg = (root / "docs" / "_static" / "logo-universal.svg").read_text()

    width = int(_re.search(r'width="(\d+)"', svg).group(1))
    height = int(_re.search(r'height="(\d+)"', svg).group(1))
    assert width == height
    assert 150 <= width <= 300, f"rendered width {width} is not banner-sized"

    # Self-contained: no second relative path for a renderer to resolve.
    assert "data:image/png;base64," in svg
    assert 'xlink:href="docs/' not in svg
    # Accessible.
    assert "<title>" in svg and 'role="img"' in svg


def test_the_wrapper_raster_is_sharp_at_the_stated_display_size():
    """At least 2x the display width, so it is not soft on a dense screen."""
    pytest.importorskip("PIL", reason="needs Pillow")
    import base64
    import io
    import re as _re

    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    svg = (root / "docs" / "_static" / "logo-universal.svg").read_text()
    width = int(_re.search(r'width="(\d+)"', svg).group(1))
    payload = _re.search(r"base64,([^\"]+)", svg).group(1)

    with Image.open(io.BytesIO(base64.b64decode(payload))) as embedded:
        raster_edge = max(embedded.size)

    assert raster_edge >= 2 * width, (
        f"embedded raster is {raster_edge}px for a {width}px display width"
    )
    # And not the whole 1254px image, which made a 1 MB banner.
    assert raster_edge <= 800, f"embedded raster {raster_edge}px is oversized"


def test_the_readme_points_at_the_wrapper_not_the_bare_raster():
    import re as _re

    root = Path(__file__).resolve().parents[1]
    readme = _re.sub(r"<!--.*?-->", "", (root / "README.md").read_text(), flags=_re.S)
    header = readme.split("## Installation")[0]

    assert _re.search(r"!\[[^\]]*\]\(docs/_static/logo-universal\.svg\)", header), (
        "the banner must reference the SVG wrapper, which carries the size"
    )
    assert "logo-universal.png)" not in header, (
        "referencing the bare PNG renders it at its full intrinsic size"
    )


def test_private_repository_workflow_badges_are_not_images():
    """A private repo's badge SVG 404s without credentials.

    github.com renders them because the browser session is authenticated; the
    GitHub app's image loader is not, so they appear as broken images. The
    workflows exist and the paths are right -- this is authentication. They are
    text links until the repository is public or a cross-renderer solution is
    verified in BOTH GitHub web and the app.
    """
    import re as _re

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    visible = _re.sub(r"<!--.*?-->", "", readme, flags=_re.S)

    badge_images = _re.findall(
        r"!\[[^\]]*\]\((https://github\.com/[^)]*badge\.svg[^)]*)\)", visible
    )
    assert not badge_images, (
        f"private-repository badge images will break in the GitHub app: {badge_images}"
    )

    # The information is still there, as links.
    for workflow in ("tests.yml", "parity.yml"):
        assert f"actions/workflows/{workflow}" in visible, (
            f"no link to the {workflow} workflow"
        )


# --- dependency bounds (PI ruling N79) ---------------------------------------


def test_runtime_dependencies_have_major_version_ceilings():
    """Broad ceilings only: stop an untested major version, do not pin minors."""
    import re as _re

    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()

    dependencies = _re.search(r"^dependencies = \[(.*?)\]", text, _re.S | _re.M)
    assert dependencies, "no runtime dependencies declared"
    block = dependencies.group(1)

    for package, ceiling in (("numpy", "<3"), ("scipy", "<2")):
        assert _re.search(rf'"{package}>=[\d.]+,{ceiling}"', block), (
            f"{package} has no {ceiling} ceiling: an untested major version "
            "could install silently"
        )
        # And no narrow pin, which would date the package.
        assert not _re.search(rf'"{package}==', block), f"{package} is pinned"


def test_the_numpy_build_requirement_carries_the_same_ceiling():
    """NumPy is also a build requirement; the bound must agree."""
    import re as _re

    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    build = _re.search(r"^requires = \[(.*?)\]", text, _re.S | _re.M)
    assert build
    assert _re.search(r'"numpy>=[\d.]+,<3"', build.group(1)), (
        "the build-time NumPy bound disagrees with the runtime one"
    )


# --- workflow files must be well formed --------------------------------------
#
# A duplicate mapping key is legal YAML (last wins) and rejected by GitHub, so
# a broken workflow shows up only as a failed run named after its filename. A
# strict load catches it before pushing.


def test_no_workflow_has_a_duplicate_key():
    yaml = pytest.importorskip("yaml")

    class Strict(yaml.SafeLoader):
        pass

    def no_duplicates(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"duplicate key {key!r}")
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    Strict.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates
    )

    root = Path(__file__).resolve().parents[1]
    workflows = sorted((root / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no workflows found"
    for path in workflows:
        try:
            yaml.load(path.read_text(), Loader=Strict)
        except Exception as error:  # pragma: no cover - the failure IS the point
            raise AssertionError(f"{path.name}: {error}") from error


def test_the_push_matrix_stays_small():
    """CI minutes are finite; macOS bills at 10x.

    An ordinary push should not fan out across every interpreter and both
    platforms. The full sweep is on workflow_dispatch and the weekly schedule.
    """
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github" / "workflows" / "tests.yml").read_text()

    # The matrices are event-dependent expressions, not literal lists.
    assert "fromJSON((github.event_name == 'schedule'" in text, (
        "the test matrix is no longer event-dependent; a push would fan out "
        "across every interpreter"
    )
    assert '"3.10","3.13"' in text, "the push matrix should be the endpoints"
    assert '"ubuntu-latest"]' in text, "macOS should not run on an ordinary push"
    # And a superseded push must stop.
    assert "cancel-in-progress: true" in text
