"""Derive theme-aware logo variants from the PI-supplied artwork.

``logo.jpg`` is the original and is never modified. It has a near-white
background, which renders as a bright square on GitHub's dark theme, and its
title text is dark navy — so simply making the background transparent would leave
dark text on a dark page.

Two derived files solve that, selected by the reader's theme through a
``<picture>`` element in the README:

``docs/_static/logo-light.png``
    background keyed out, colours unchanged.
``docs/_static/logo-dark.png``
    background keyed out, and the dark strokes lightened so the title and
    subtitle stay legible on a dark page. Hues are preserved; only lightness
    moves.
``docs/_static/logo-universal.png``
    a mid-tone compromise, legible on **either** background. It exists because
    the GitHub mobile app ignores ``<picture>`` media queries and renders the
    ``<img src>`` fallback: with the light variant there, a dark-theme phone
    showed dark navy text on a dark page. This is the fallback, so a renderer
    that cannot choose still gets something readable. Measured contrast against
    white 4.5:1 and against GitHub's dark canvas 4.2:1.

Both are **derived assets**. The PI's original stays canonical, and NOTICE.md
records the relationship. Rerun after any change to ``logo.jpg``::

    python tools/make_logo_variants.py

Requires Pillow, which is a documentation/tooling dependency and not part of the
inference core.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logo.jpg"
OUTPUT_DIR = ROOT / "docs" / "_static"

#: Pixels at least this bright in every channel are treated as background. The
#: supplied artwork's background measures 252-254, and its lightest real content
#: is well below this, so the threshold is not delicate.
_BACKGROUND_THRESHOLD = 246

#: How far the dark-theme variant lifts the darkest strokes, as a fraction of the
#: way to white. Enough to read comfortably on a dark page without washing the
#: artwork out into grey.
_DARK_LIFT = 0.72

#: Raster resolution per variant. This is IMAGE INFORMATION, not layout.
#:
#: An earlier version downsampled the universal variant to 260 px to make plain
#: Markdown render it small. That is the wrong instrument: it throws away image
#: data to achieve a layout effect, and the artwork is then permanently low
#: resolution on any renderer that shows it larger. Display size is controlled
#: separately -- see LOGO_DISPLAY_WIDTH and the generated SVG wrapper.
DEFAULT_MAX_SIZE = 1254  # the canonical source resolution; nothing is discarded
MAX_SIZE: dict[str, int] = {}

#: How wide the README banner should APPEAR, in CSS pixels. Carried by the SVG
#: wrapper's width/height, so it is layout metadata rather than a resampling.
LOGO_DISPLAY_WIDTH = 220

#: Device-pixel ratio the wrapper's embedded raster is sized for.
#:
#: The canonical 1254 px variants stay in the repository at full resolution --
#: no image information is discarded anywhere. This governs only what the
#: SELF-CONTAINED SVG carries, and it is a transport decision: embedding the
#: full 1254 px raster as base64 produced a 1 MB banner that every clone and
#: every page view would pay for, to render at 220 px.
#:
#: 2.5 rather than a round guess: calibrated against fkeruzore/halox, which
#: ships a 1200x400 logo displayed at width=500 -- a ratio of 2.4 -- for 69 kB
#: transferred. That is one repository, not a rule: it is evidence that this
#: neighbourhood is reasonable, not that 2.4-2.5 is what everyone does.
#:
#: The reason a wrapper is needed here at all is narrower than a style
#: preference. halox and the other repositories surveyed reference their logo by
#: an ABSOLUTE raw URL, which resolves everywhere; both absolute forms return 404
#: for this repository because it is private. That is the compatibility problem
#: the self-contained wrapper solves, and it disappears if the repository ever
#: becomes public.
LOGO_WRAPPER_DPR = 2.5


def _target_size(name: str) -> int:
    return MAX_SIZE.get(name, DEFAULT_MAX_SIZE)


#: The universal variant's lift, chosen by measuring WCAG contrast of the
#: artwork's darkest *rendered* ink -- the 5th percentile of opaque pixels, which
#: includes anti-aliased stroke edges -- against both canvases:
#:
#:   lift 0.34 -> 7.04:1 on white, 2.69:1 on dark   (fails dark)
#:   lift 0.48 -> 4.46:1 on white, 4.25:1 on dark   (balanced)
#:   lift 0.60 -> 3.23:1 on white, 5.85:1 on dark   (favours dark)
#:
#: 0.48 is the crossover, and is the only value that clears 3:1 on both. An
#: earlier 0.34 was chosen from the mean ink rather than the darkest, and would
#: have left the mobile-app dark theme barely readable -- which is the bug this
#: variant exists to fix.
_UNIVERSAL_LIFT = 0.48


def _alpha_from_background(rgb: np.ndarray) -> np.ndarray:
    """Soft alpha: fully opaque content, fully transparent background.

    A hard threshold would leave a jagged halo on the anti-aliased curves, so the
    alpha ramps over the last few levels instead.
    """
    brightness = rgb.min(axis=2).astype(np.float64)
    alpha = (_BACKGROUND_THRESHOLD - brightness) / 12.0
    return np.clip(alpha, 0.0, 1.0)


def build_light(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    alpha = _alpha_from_background(rgb)
    out = np.dstack([rgb, (alpha * 255).round().astype(np.uint8)])
    return Image.fromarray(out, mode="RGBA")


def _lift(image: Image.Image, amount: float) -> Image.Image:
    """Move dark pixels towards white in proportion to how dark they are.

    Preserves hue. Inverting instead would turn a blue logo orange.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.float64)
    alpha = _alpha_from_background(np.asarray(image.convert("RGB")))
    darkness = 1.0 - rgb.mean(axis=2, keepdims=True) / 255.0
    lifted = np.clip(rgb + (255.0 - rgb) * (amount * darkness), 0, 255).astype(np.uint8)
    return Image.fromarray(
        np.dstack([lifted, (alpha * 255).round().astype(np.uint8)]), mode="RGBA"
    )


def build_universal(image: Image.Image) -> Image.Image:
    """Legible on either background; the fallback for renderers that cannot choose."""
    return _lift(image, _UNIVERSAL_LIFT)


def build_dark(image: Image.Image) -> Image.Image:
    """Lighten dark strokes, preserving hue.

    Each pixel is moved towards white in proportion to how dark it is, so the
    navy title lifts a lot and the already-light purple band barely moves. The
    alternative — inverting — would turn a blue logo orange.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.float64)
    alpha = _alpha_from_background(np.asarray(image.convert("RGB")))

    darkness = 1.0 - rgb.mean(axis=2, keepdims=True) / 255.0
    lifted = rgb + (255.0 - rgb) * (_DARK_LIFT * darkness)
    lifted = np.clip(lifted, 0, 255).astype(np.uint8)

    out = np.dstack([lifted, (alpha * 255).round().astype(np.uint8)])
    return Image.fromarray(out, mode="RGBA")


#: Every derived variant, and the function that produces it.
_VARIANTS = (
    ("logo-light.png", build_light),
    ("logo-dark.png", build_dark),
    ("logo-universal.png", build_universal),
)


def check(tolerance: int = 2) -> int:
    """Verify the committed variants still match what the generator produces.

    Compares **decoded pixels**, not file bytes. A byte comparison fails for
    reasons that have nothing to do with the artwork: JPEG decoding, LANCZOS
    resampling and PNG encoding all differ slightly between platforms and Pillow
    releases, and the first CI run on Linux duly failed against files generated
    on macOS. What actually needs guarding is that nobody hand-edited a variant
    or changed the generator without regenerating -- and that shows up as a large
    pixel difference, not a one-bit one.
    """
    source = Image.open(SOURCE)
    worst = 0
    for name, builder in _VARIANTS:
        target = OUTPUT_DIR / name
        if not target.is_file():
            print(f"MISSING {target}")
            return 1
        expected = builder(source)
        edge = _target_size(name)
        expected.thumbnail((edge, edge), Image.LANCZOS)
        committed = Image.open(target).convert("RGBA")
        if committed.size != expected.size:
            print(f"SIZE MISMATCH {name}: {committed.size} != {expected.size}")
            return 1
        difference = int(
            np.max(
                np.abs(
                    np.asarray(committed, dtype=np.int16)
                    - np.asarray(expected, dtype=np.int16)
                )
            )
        )
        worst = max(worst, difference)
        print(f"{name}: max pixel difference {difference} (tolerance {tolerance})")

    if worst > tolerance:
        print(
            f"FAIL: the committed variants differ from the generator by {worst}, "
            "which is more than encoder noise. Rerun tools/make_logo_variants.py "
            "and commit the result."
        )
        return 1
    print("committed logo variants reproduce within tolerance")
    return 0


def write_display_wrapper() -> Path:
    """An SVG carrying a high-density raster at a stated display size.

    Plain Markdown image syntax has no width attribute, so a PNG renders at its
    intrinsic size; raw HTML with a relative ``src`` has broken in GitHub's
    mobile app. An SVG is referenced by the SAME plain Markdown syntax that does
    work there, and carries its own ``width``/``height`` -- so the display size
    travels with the asset instead of being faked by discarding pixels.

    The PNG is embedded as a data URI, so the file is self-contained and no
    renderer has to resolve a second relative path.
    """
    import base64
    import io

    edge = LOGO_DISPLAY_WIDTH
    raster_edge = int(round(edge * LOGO_WRAPPER_DPR))
    with Image.open(OUTPUT_DIR / "logo-universal.png") as full:
        embedded = full.copy()
    embedded.thumbnail((raster_edge, raster_edge), Image.LANCZOS)
    buffer = io.BytesIO()
    embedded.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{edge}" height="{edge}" viewBox="0 0 {edge} {edge}" '
        f'role="img" aria-label="GP DLA Finder">'
        f"<title>GP DLA Finder</title>"
        f'<image x="0" y="0" width="{edge}" height="{edge}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'xlink:href="data:image/png;base64,{encoded}"/>'
        f"</svg>\n"
    )
    target = OUTPUT_DIR / "logo-universal.svg"
    target.write_text(svg, encoding="utf-8")
    return target


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"missing {SOURCE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = Image.open(SOURCE)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print(f"source  {SOURCE.name}  {source.size[0]}x{source.size[1]}  sha256 {digest}")

    for name, builder in _VARIANTS:
        target = OUTPUT_DIR / name
        image = builder(source)
        # Keep the committed raster variants at the source resolution. Display
        # size is controlled separately by the README wrapper.
        edge = _target_size(name)
        image.thumbnail((edge, edge), Image.LANCZOS)
        image.save(target, optimize=True)
        size_kb = target.stat().st_size / 1024
        dimensions = f"{image.size[0]}x{image.size[1]}"
        print(f"wrote   {target.relative_to(ROOT)}  {dimensions}  {size_kb:.0f} KiB")

    wrapper = write_display_wrapper()
    size_kb = wrapper.stat().st_size / 1024
    print(
        f"wrote   {wrapper.relative_to(ROOT)}  "
        f"{LOGO_DISPLAY_WIDTH}x{LOGO_DISPLAY_WIDTH} displayed, "
        f"{int(round(LOGO_DISPLAY_WIDTH * LOGO_WRAPPER_DPR))}px raster  "
        f"{size_kb:.0f} KiB"
    )
    return 0


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        raise SystemExit(check())
    raise SystemExit(main())
