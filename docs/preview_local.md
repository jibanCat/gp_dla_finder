# Maintainer check for the documentation build

The hosted documentation is prepared but **not connected**. The
`.readthedocs.yaml` file is ready, but nothing in this repository publishes the
site by itself.

This page gives the maintainer checks that match the planned Read the Docs
build. If you only want to open the site locally, the shorter walkthrough is in
{doc}`preview`.

## One command

From the repository root, in an environment with the `docs` extra installed:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

Then open:

```
docs/_build/html/index.html
```

Most desktops open that with a double-click. From a terminal, `open` on macOS
and `xdg-open` on Linux both take the path directly.

If you would rather not install anything into your usual environment:

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install -e '.[docs]'
.venv-docs/bin/python -m sphinx -W -b html docs docs/_build/html
```

## Why `-W`

Warnings are errors. A renamed symbol, missing anchor, or missing figure should
fail while it is still easy to fix. Read the Docs uses the same setting
(`fail_on_warning: true`), so a clean local build exercises the same check.

## Checking the links

```bash
python -m sphinx -b linkcheck docs docs/_build/linkcheck
```

This resolves internal cross-references and external URLs. External links can
fail for reasons that have nothing to do with this repository — a site that is
briefly down, or one that redirects to a login page — so read the report rather
than the exit status.

## Regenerating the figures

The figures are committed as SVGs, so an ordinary docs build does not need
matplotlib. If you change a figure:

```bash
python tools/make_docs_figures.py            # rewrite the SVGs
python tools/make_docs_figures.py --check    # compare the PLOTTED DATA
```

`--check` regenerates in memory and compares the underlying numbers against the
committed record, so a figure whose data drifted fails even if the picture still
looks reasonable.

## What matches the hosted build, and what does not

| | local | Read the Docs |
|---|---|---|
| Sphinx configuration | `docs/conf.py` | `docs/conf.py` |
| warnings are errors | yes, with `-W` | yes, `fail_on_warning` |
| dependencies | the `docs` extra | the `docs` extra |
| Python | whatever you use | 3.12 |
| operating system | yours | Ubuntu 24.04 |

The interpreter and operating system are the only real differences, and neither
has changed a build so far. If you want to eliminate them too, build in a 3.12
environment.
