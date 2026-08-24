"""Sphinx configuration for the public Read the Docs and local builds."""

from __future__ import annotations

project = "gp_dla_finder"
author = "Ming-Feng Ho"
copyright = "2026, Ming-Feng Ho"  # noqa: A001

try:
    from importlib.metadata import version as _version

    release = _version("gp_dla_finder")
except Exception:  # pragma: no cover - docs may build from a source tree
    release = "0.1.0rc3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

# Notebooks in the docs are executed at build time, so a published page can never
# show output that the current code would not produce.
nb_execution_mode = "auto"
nb_execution_timeout = 600

# html_image: figures are written as <picture> so the SAME markup renders in
# the hosted Sphinx site and in GitHub's source view. MyST {image} directives
# show up as literal code blocks on GitHub.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_image",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo-light.png"
html_favicon = None
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "alt_text": "GP DLA Finder",
    },
    "github_url": "https://github.com/jibanCat/gp_dla_finder",
    "show_prev_next": True,
    # Keep the initial documentation simple: no version switcher or banner yet.
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
