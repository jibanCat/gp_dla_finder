"""The documented examples are executed, not paraphrased.

PI ruling, increment-15 correction 1. The previous tutorial test inspected import
names and then ran a *separately written* body, so a page that passed an
undefined ``results`` variable to ``results_to_catalogue()`` tested green. The
fix is structural: ``docs/examples/*.py`` are the canonical source, the pages
include regions of them verbatim, and these tests run the files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
TUTORIAL = ROOT / "docs" / "tutorial.md"

pytest.importorskip("astropy", reason="the examples write FITS")


def _example_files():
    return sorted(EXAMPLES.glob("*.py"))


def test_there_is_at_least_one_example():
    assert _example_files(), "docs/examples/ is empty; the tutorial has no source"


#: What each example is expected to leave behind, relative to its working
#: directory. An example that produces nothing checkable is not evidence that it
#: did anything.
_EXAMPLE_ARTIFACTS = {
    "quickstart.py": ("absorbers.fits", "run.json"),
    "custom_grid.py": (),
    "two_absorbers.py": (),
}


@pytest.mark.parametrize("example", _example_files(), ids=lambda p: p.name)
def test_the_example_runs_as_written(example, tmp_path):
    """Execute the file exactly as a reader would, in a scratch directory.

    ``cwd=tmp_path`` because the examples write into the working directory, and
    a test must not drop files into the repository.
    """
    assert example.name in _EXAMPLE_ARTIFACTS, (
        f"{example.name} is not listed in _EXAMPLE_ARTIFACTS; say what it "
        "should produce so this test can check it ran"
    )
    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        f"{example.name} failed:\n{completed.stdout}\n{completed.stderr}"
    )
    for artifact in _EXAMPLE_ARTIFACTS[example.name]:
        assert (tmp_path / artifact).is_file(), (
            f"{example.name} did not produce {artifact}"
        )


def test_no_example_writes_into_the_repository(tmp_path):
    """An example must leave its output where it was run, not in the checkout.

    custom_grid.py ran its builder with ``cwd=<repository>``, so it wrote a grid
    into the repository root even when the test gave it a scratch directory --
    and one of those files was committed by accident.
    """
    before = {p.relative_to(ROOT) for p in ROOT.rglob("*") if ".git" not in p.parts}

    for example in _example_files():
        completed = subprocess.run(
            [sys.executable, str(example)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

    after = {p.relative_to(ROOT) for p in ROOT.rglob("*") if ".git" not in p.parts}
    created = sorted(str(p) for p in after - before)
    # __pycache__ and egg-info are the interpreter's and setuptools', not the
    # example's, and they are already ignored.
    created = [
        p
        for p in created
        if "__pycache__" not in p and ".egg-info" not in p and not p.endswith(".pyc")
    ]
    assert not created, f"an example wrote into the repository: {created}"


def test_the_custom_grid_example_demonstrates_the_guard(tmp_path):
    """It is supposed to SHOW the refusal, not merely survive it.

    An example that quietly stopped demonstrating the failure -- because the
    guard regressed, or because the try/except was removed -- would still exit
    0 and still print a result.
    """
    example = EXAMPLES / "custom_grid.py"
    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "refused, as it should be" in completed.stdout
    assert "log_nhi_range" in completed.stdout
    # And it finishes by running successfully with a matching configuration.
    assert "(17.2, 23.0)" in completed.stdout
    assert "completed" in completed.stdout
    assert "extended_nhi_2000" in completed.stdout


def test_the_two_absorber_tutorial_reaches_the_m2_result(tmp_path):
    """The public example should exercise M2, not merely enable its flag."""
    example = EXAMPLES / "two_absorbers.py"
    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "selected model: M2" in completed.stdout
    candidates = [
        line for line in completed.stdout.splitlines() if line.startswith("2 ")
    ]
    assert len(candidates) == 2


def test_the_tutorial_includes_the_example_rather_than_copying_it():
    """No hand-typed python blocks in the section that has a canonical source.

    A ``python`` fence there would be a second copy, free to drift. The one
    allowance is the short conceptual contrast in "A candidate is not a
    detection", which is deliberately not runnable code.
    """
    text = TUTORIAL.read_text()
    # Locate the section by CONTENT, not by heading text. Pinning the heading
    # made a prose pass that renamed it fail a test about code duplication,
    # which is not what this checks.
    marker = text.index("literalinclude")
    section = text[text.rindex("\n## ", 0, marker) :]

    includes = re.findall(r"```\{literalinclude\}\s+(\S+)", section)
    assert includes, "the tutorial no longer includes the canonical example"
    for target in includes:
        assert (TUTORIAL.parent / target).is_file(), f"missing include: {target}"

    fences = re.findall(r"^```python$", section, re.M)
    assert len(fences) <= 1, (
        f"{len(fences)} hand-written python blocks in the Finder section; "
        "these drift from docs/examples/. Use a literalinclude."
    )


def test_every_included_region_exists_in_its_source():
    """A renamed marker must fail here, not silently render an empty block."""
    text = TUTORIAL.read_text()
    blocks = re.findall(r"```\{literalinclude\}\s+(\S+)\n(.*?)```", text, re.S)
    assert blocks
    checked = 0
    for target, options in blocks:
        source = (TUTORIAL.parent / target).read_text()
        for marker in re.findall(r':(?:start-after|end-before):\s*"([^"]+)"', options):
            assert marker in source, f"{target} has no marker {marker!r}"
            checked += 1
    assert checked >= 6, f"expected several markers, checked {checked}"


# --- the README's runnable snippets must run ---------------------------------
#
# `Finder(Config.desi_y3())` sat in the README while the Finder refused
# max_absorbers=4, so the headline example raised as soon as anyone tried it.
# Documentation that cannot run is worse than none.

README = ROOT / "README.md"


def _readme_config_expressions():
    text = README.read_text()
    return sorted(set(re.findall(r"^\s*(Config\.[a-z_0-9]+\([^)]*\))", text, re.M)))


def test_every_readme_config_expression_constructs():
    from gp_dla_finder import Config  # noqa: F401  (used by eval)

    expressions = _readme_config_expressions()
    assert expressions, "no Config examples found in the README"
    for expression in expressions:
        try:
            eval(expression)  # noqa: S307 - our own documentation, not input
        except Exception as error:  # pragma: no cover - the failure IS the point
            raise AssertionError(
                f"README example {expression!r} does not construct: {error}"
            ) from error


def test_the_readme_finder_example_runs():
    """The headline snippet, executed."""
    from gp_dla_finder.finder import Finder
    from synthetic import make_spectrum

    text = README.read_text()
    assert "Finder().run(spectrum)" in text, (
        "the README's headline example changed; update this test with it"
    )
    result = Finder(warn_about_threads=False).run(make_spectrum())
    assert result.status == "completed"
    assert 0.0 <= result.p_absorber <= 1.0


def test_the_readme_does_not_show_a_refused_configuration():
    """max_absorbers above the implemented limit raises; do not advertise it."""
    import re as _re

    for expression in _readme_config_expressions():
        if "max_absorbers" in expression:
            value = int(_re.search(r"max_absorbers\s*=\s*(\d+)", expression).group(1))
            assert value <= 2, f"README advertises {expression}, which is refused"


# --- the customisation page's snippets must run too --------------------------
#
# It is a page made almost entirely of configuration examples. A snippet that
# does not construct is the whole failure mode.

CUSTOMISATION = ROOT / "docs" / "customisation.md"


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", path.read_text(), re.S | re.M)


def test_the_customisation_page_has_examples_to_check():
    assert len(_python_blocks(CUSTOMISATION)) >= 8


def test_every_customisation_config_expression_constructs():
    """Every ``Config.<preset>(...)`` on the page, executed.

    The ones that are supposed to raise are checked separately below, so this
    deliberately skips the two lines documented as refusals rather than
    pretending they succeed.
    """
    from gp_dla_finder import Config  # noqa: F401  (used by eval)

    text = CUSTOMISATION.read_text()
    expressions = sorted(
        set(re.findall(r"(Config\.[a-z_0-9]+\((?:[^()]|\([^()]*\))*\))", text))
    )
    assert len(expressions) >= 8, f"only found {len(expressions)} Config examples"

    refusals = ("tau_eb_objective", "tau_eb_apply_hcd_mask")
    checked = 0
    for expression in expressions:
        if any(name in expression for name in refusals):
            continue
        if "my_dla_only" in expression:
            # Names a grid the reader is told to build; it cannot exist here.
            continue
        try:
            eval(expression)  # noqa: S307 - our own documentation, not input
        except Exception as error:  # pragma: no cover - the failure IS the point
            raise AssertionError(
                f"customisation example {expression!r} does not construct: {error}"
            ) from error
        checked += 1
    assert checked >= 6


def test_the_page_names_only_assets_that_exist():
    """Every quoted preset, grid, prior, model and kernel name is real."""
    from gp_dla_finder import (
        available_models,
        available_priors,
        available_sample_grids,
    )
    from gp_dla_finder.quality import QUALITY_POLICIES
    from gp_dla_finder.voigt import LSF_KERNELS

    text = CUSTOMISATION.read_text()
    known = (
        set(available_models())
        | set(available_priors())
        | set(available_sample_grids())
        | set(LSF_KERNELS)
        | set(QUALITY_POLICIES)
    )
    # Quoted bare names that look like assets: lowercase with an underscore or
    # a hyphen, inside double quotes.
    quoted = set(re.findall(r'"([a-z][a-z0-9]*(?:[-_][a-z0-9]+)+)"', text))
    # Names the page invents on purpose, as things the reader would create.
    invented = {"my_dla_only_50000", "custom-boss-fast", "desi_y3_fast"}
    # Result-provenance keys, quoted the same way but not asset names.
    provenance_keys = {"base_preset", "config_digest"}
    invented |= provenance_keys
    for name in quoted - invented:
        assert name in known, f"the page names {name!r}, which does not exist"


def test_the_documented_refusals_actually_refuse():
    """The page tells readers these raise. If they stop raising, it is wrong."""
    from gp_dla_finder import Config
    from gp_dla_finder.finder import Finder
    from gp_dla_finder.mean_flux import HCDMaskNotSupported, ObjectiveNotSupported

    text = CUSTOMISATION.read_text()
    assert "ObjectiveNotSupported" in text
    assert "HCDMaskNotSupported" in text

    # On run(), not on construction -- which is what the page now says. The
    # first draft claimed the Finder refused, and it does not.
    sys.path.insert(0, str(ROOT / "tests"))
    from synthetic import CORPUS, build

    spectrum = build({c.name: c for c in CORPUS}["classical-dla-mid-z"])

    for overrides, error in (
        ({"tau_eb_objective": "absorber"}, ObjectiveNotSupported),
        ({"tau_eb_apply_hcd_mask": True}, HCDMaskNotSupported),
    ):
        finder = Finder(
            Config.desi_y3_fast(max_absorbers=1, **overrides),
            warn_about_threads=False,
        )
        with pytest.raises(error):
            finder.run(spectrum, targetid=1)


def test_the_page_is_right_that_overriding_renames_the_preset():
    """The mechanism the whole page rests on."""
    from gp_dla_finder import Config

    reference = Config.desi_y3_fast()
    modified = Config.desi_y3_fast(lsf_kernel="boss-r2000-7tap")

    assert reference.preset == "desi_y3_fast"
    assert modified.preset == "desi_y3_fast+modified"
    assert modified.base_preset == "desi_y3_fast"
    assert modified.digest != reference.digest


def test_the_page_is_right_that_log_nhi_range_does_not_resample():
    """The claim most likely to bite a reader, so it is pinned.

    ``log_nhi_range`` describes the grid; it does not build one. If the package
    ever starts consulting it at run time, this test fails and the page needs
    rewriting -- which is the point.
    """
    from gp_dla_finder import Config, load_sample_grid

    narrow = Config.desi_y3_fast(log_nhi_range=(20.3, 22.5))
    grid = load_sample_grid(narrow.sample_grid)

    # The samples still span the full packaged range.
    assert min(grid.log_nhi_samples) < 20.3
    # And the configuration is nonetheless a different run.
    assert narrow.digest != Config.desi_y3_fast().digest


def test_every_packaged_preset_agrees_with_its_own_grid():
    """Nothing enforces this, so it is at least measured.

    A preset whose ``log_nhi_range`` disagreed with the grid it names would
    describe a prior it does not sample -- the exact trap the page warns about,
    shipped as a default.
    """
    from gp_dla_finder import Config, load_sample_grid

    for name in ("desi_y3", "desi_y3_refined", "desi_y3_fast"):
        config = getattr(Config, name)()
        grid = load_sample_grid(config.sample_grid)
        assert grid.declared_support == config.log_nhi_range, (
            f"preset {name} declares log_nhi_range {config.log_nhi_range} but "
            f"its grid {config.sample_grid} covers {grid.declared_support}"
        )
        assert grid.num_samples == config.num_samples, (
            f"preset {name} asks for {config.num_samples} samples but its grid "
            f"holds {grid.num_samples}"
        )


def test_every_finder_the_page_builds_can_actually_be_built():
    """The README once shipped ``Finder(Config.desi_y3())``, which raised.

    Every preset declares ``max_absorbers=4`` and the Finder refuses more than
    two, so a page that builds a Finder from a bare preset is broken for its
    reader. This runs each one, and expects a refusal only where the page says
    to expect one.
    """
    from gp_dla_finder import Config  # noqa: F401  (used by eval)
    from gp_dla_finder.finder import Finder  # noqa: F401  (used by eval)

    text = CUSTOMISATION.read_text()
    calls = re.findall(r"(Finder\(Config\.[a-z_0-9]+\((?:[^()]|\([^()]*\))*\)\))", text)
    assert len(calls) >= 3, f"the page builds only {len(calls)} Finders"

    from gp_dla_finder.finder import SampleGridMismatch

    for call in calls:
        # Two refusals fire at construction, and the page shows both on purpose:
        # a bare preset (max_absorbers=4), and a configuration describing a
        # different N_HI prior than its grid. The mean-flux refusals fire on
        # run() and are checked above.
        expects_error = "max_absorbers" not in call or "log_nhi_range" in call
        expression = call.strip()
        try:
            eval(f"{expression[:-1]}, warn_about_threads=False)")  # noqa: S307
        except (NotImplementedError, SampleGridMismatch):
            assert expects_error, (
                f"the page shows {expression!r} as usable, but it raises"
            )
        else:
            assert not expects_error, (
                f"the page says {expression!r} raises, but it succeeded"
            )
