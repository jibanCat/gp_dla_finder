"""The documentation manifest must not contradict the files it points at.

PI ruling, increment-15 correction 2. The manifest is a canonical handoff
product -- GPT works from it -- and increment 15's copy claimed no tutorial
covered ``Finder`` and that the "not yet available" section was un-updated, when
that same increment had done both. Prose claims cannot all be machine-checked;
these are the ones that can.

The manifest lives under ``.claude/`` and is never committed, so these tests skip
when it is absent (a fresh clone, or CI).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".claude" / "notes" / "DOCUMENTATION_MANIFEST.md"

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="the manifest is a local, uncommitted note"
)


def _manifest() -> str:
    return MANIFEST.read_text()


def _referenced_paths() -> set[str]:
    """Repository paths named in backticks or table cells."""
    text = _manifest()
    found = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:md|py|yaml|yml|toml|cff))`", text))
    found |= set(re.findall(r"`(docs/_static/[A-Za-z0-9_{},.*-]+)`", text))
    return found


def test_every_path_the_manifest_names_exists():
    missing = []
    for path in sorted(_referenced_paths()):
        if any(ch in path for ch in "*{"):
            continue  # a glob or a brace set, checked separately below
        if not (ROOT / path).exists():
            missing.append(path)
    assert not missing, f"the manifest names paths that do not exist: {missing}"


def test_the_generated_globs_match_something():
    figures = ROOT / "docs" / "_static" / "figures"
    assert list(figures.glob("*.svg")), "manifest claims generated figures exist"
    for stem in ("logo-light", "logo-dark", "logo-universal"):
        assert (ROOT / "docs" / "_static" / f"{stem}.png").is_file()


def test_the_page_to_source_map_covers_every_page():
    """A page added without a manifest row is exactly the drift this prevents."""
    text = _manifest()
    mapped = set(re.findall(r"\|\s*`?(\w+)\.html`?\s*\|", text))
    pages = {p.stem for p in (ROOT / "docs").glob("*.md")}
    unmapped = pages - mapped
    assert not unmapped, f"pages missing from the page-to-source map: {unmapped}"


def test_the_manifest_does_not_repeat_the_stale_increment_15_claims():
    """The two specific false statements the PI caught, asserted gone."""
    text = _manifest().lower()
    for claim in (
        "no tutorial page covers `finder` yet",
        "it needs updating and i have not done it",
    ):
        assert claim not in text, f"stale increment-15 claim survives: {claim!r}"


def test_manifest_claims_about_the_tutorial_are_true():
    tutorial = (ROOT / "docs" / "tutorial.md").read_text()
    # It claims the tutorial includes the example rather than copying it.
    assert "literalinclude" in tutorial
    assert "examples/quickstart.py" in tutorial
    # It claims the tutorial covers Finder and the candidate/detection split.
    assert "Finder" in tutorial
    assert "candidate is not a detection" in tutorial.lower()
    # It claims None was corrected to NaN. Check the CLAIM, not the wording:
    # matching an exact sentence makes every prose edit a test failure, and it
    # did -- a rephrasing that said the same thing more clearly broke this.
    assert "`NaN`" in tutorial, "the tutorial no longer says the fields are NaN"
    assert not re.search(r"inference[^.]{0,40}fields are `None`", tutorial), (
        "the tutorial claims the fields are None"
    )


def test_manifest_claims_about_the_readme_are_true():
    readme = (ROOT / "README.md").read_text()
    # It claims the long BLAS tables moved out of the README.
    assert "0.2438 ms" not in readme, "the full BLAS table is back in the README"
    # It claims a documentation index table was added.
    assert "docs/preview.md" in readme
    # It claims the status block no longer denies end-to-end operation.
    assert "cannot yet run inference end to\n> end" not in readme


def test_manifest_claims_about_the_api_rename_are_true():
    finder = (ROOT / "src" / "gp_dla_finder" / "finder.py").read_text()
    assert "class AbsorberCandidate" in finder
    assert "absorber_candidates" in finder
    assert "def screening_score" in finder
    assert "RUN_DEFINING_PROVENANCE" in finder
    assert "def detected(self, threshold: float)" in finder


# --- figures must actually render, in BOTH renderers ------------------------
#
# The figures were added as MyST `{image}` directives, which Sphinx renders and
# GitHub shows as a literal code block. Since the docs site is not published,
# GitHub is where these pages are read -- so the figures were invisible exactly
# where anyone would look at them. They are now <picture> elements, which both
# renderers understand. These tests stop that regressing.


def _pages_with_figures():
    return [
        ROOT / "docs" / "tutorial.md",
        ROOT / "docs" / "customisation.md",
        ROOT / "docs" / "filter.md",
    ]


def test_the_figure_pages_actually_reference_figures():
    for page in _pages_with_figures():
        text = page.read_text()
        assert "_static/figures/" in text, f"{page.name} references no figure"


def test_figures_use_markup_every_github_renderer_rewrites():
    """Plain Markdown images, not raw HTML.

    Two renderer facts, both learned the hard way:
      * a MyST ``{image}`` directive shows as a literal code block on GitHub;
      * GitHub's mobile app does not rewrite relative paths inside raw HTML,
        so <picture>/<img> renders as a BROKEN LINK there however correct the
        path is -- which is what happened to the logo.
    Markdown image syntax is rewritten by every GitHub renderer and by Sphinx.
    """
    for page in _pages_with_figures():
        text = _without_html_comments(page.read_text())
        assert "```{image}" not in text, (
            f"{page.name} uses a MyST image directive; GitHub shows it as code"
        )
        assert "<picture>" not in text, (
            f"{page.name} uses <picture>; GitHub's app shows a broken link"
        )
        assert re.search(r"!\[[^\]]*\]\(_static/figures/", text), (
            f"{page.name} has no Markdown figure reference"
        )


def test_figures_reference_the_universal_variant():
    """One file has to work on both canvases, so it must be the measured one."""
    for page in _pages_with_figures():
        text = page.read_text()
        for target in re.findall(r"!\[[^\]]*\]\((_static/figures/[^)]+)\)", text):
            assert target.endswith("-universal.svg"), (
                f"{page.name} references {target}; a single-file reference must "
                "be the universal variant, which is legible on light and dark"
            )


def _without_html_comments(text: str) -> str:
    """Comments explain the rule; they must not be mistaken for breaking it."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def test_the_readme_banner_is_also_markdown():
    """Same renderer trap, and the one the PI actually reported."""
    readme = _without_html_comments((ROOT / "README.md").read_text())
    assert "<picture>" not in readme, (
        "the README banner uses <picture>; GitHub's app shows a broken link"
    )
    assert re.search(r"!\[[^\]]*\]\(docs/_static/logo-universal\.svg\)", readme), (
        "the README banner must be a Markdown image on the universal SVG wrapper"
    )


# --- malformed captions ------------------------------------------------------
#
# A regex substitution once duplicated a caption's final clause and left the
# emphasis unbalanced: "...forest lines are narrow.**width** — the damping wings
# span tens of angstroms where forest lines are narrow.*". These check the
# STRUCTURE rather than any exact sentence, so a prose edit does not fail them.


def _emphasis_runs(text: str) -> int:
    """Count single-asterisk emphasis markers, ignoring ** and list bullets."""
    return len(re.findall(r"(?<![*\w])\*(?!\*)|(?<!\*)\*(?![*\s])", text))


def test_no_documentation_page_has_unbalanced_emphasis():
    for page in sorted((ROOT / "docs").glob("*.md")):
        text = page.read_text()
        # Strip fenced code, inline code and math, where asterisks are literal.
        stripped = re.sub(r"```.*?```", "", text, flags=re.S)
        stripped = re.sub(r"`[^`]*`", "", stripped)
        stripped = re.sub(r"\$\$.*?\$\$", "", stripped, flags=re.S)
        stripped = re.sub(r"\$[^$\n]*\$", "", stripped)
        assert stripped.count("**") % 2 == 0, (
            f"{page.name} has an odd number of ** markers"
        )


def test_no_page_repeats_a_clause_verbatim():
    """The signature of a bad substitution: the same long clause twice."""
    for page in sorted((ROOT / "docs").glob("*.md")):
        text = re.sub(r"```.*?```", "", page.read_text(), flags=re.S)
        # Sentence-ish fragments long enough that a repeat is not coincidence.
        fragments = [
            fragment.strip()
            for fragment in re.split(r"[.;]\s", text)
            if len(fragment.strip()) > 60
        ]
        seen: dict[str, int] = {}
        for fragment in fragments:
            key = " ".join(fragment.split())
            seen[key] = seen.get(key, 0) + 1
        repeats = {k: n for k, n in seen.items() if n > 1}
        assert not repeats, (
            f"{page.name} repeats a clause verbatim, which is how a broken "
            f"substitution looks: {list(repeats)[:1]}"
        )


def test_captions_do_not_run_emphasis_into_a_word():
    """`narrow.**width**` -- emphasis opening immediately after a full stop."""
    for page in sorted((ROOT / "docs").glob("*.md")):
        text = re.sub(r"```.*?```", "", page.read_text(), flags=re.S)
        assert not re.search(r"[a-z]\.\*\*\w", text), (
            f"{page.name} opens emphasis directly after a sentence end, which "
            "is what a duplicated clause looks like"
        )
