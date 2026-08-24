# Previewing these pages locally

The documentation is not online yet, but you can view the complete rendered site
locally. The commands below were tested on macOS/arm64 with the versions listed
at the end of the page.

## From scratch

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open **<http://127.0.0.1:8000/>**. Press `Ctrl-C` in the terminal to stop the
server.

On macOS you can open it in one step:

```bash
open http://127.0.0.1:8000/
```

:::{note}
`--bind 127.0.0.1` limits the preview to the local machine.
:::

## After editing a page

After you edit a page, rebuild and reload the browser. You can leave the server
running:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The equivalent Makefile command is:

```bash
make -C docs html
```

`make -C docs serve` builds the pages and starts the local server:

```bash
make -C docs serve
```

The docs environment must be active for both Makefile commands. If
`sphinx-build` is not found, run `source .venv-docs/bin/activate`. The interpreter
and port can be changed with
`make -C docs serve PYTHON=python3.12 SERVE_PORT=8001`.

## `-W` means warnings are errors

We use `-W` so warnings fail the build, just like they do in documentation CI.
This catches things such as missing anchors and renamed API objects.

## Matching the Read the Docs build

`.readthedocs.yaml` installs the same `docs` extra and uses `docs/conf.py` with
`fail_on_warning: true`. A successful local build therefore exercises the same
Sphinx configuration.

No Read the Docs project is connected yet. The configuration is ready for the
later deployment, but it does not publish anything by itself.

## Versions this was tested with

| | |
|---|---|
| Python | 3.10.6 |
| Sphinx | 8.1.3 |
| pydata-sphinx-theme | 0.19.0 |
| myst-nb | 1.4.0 |
| Platform | macOS 14.6, arm64 |
