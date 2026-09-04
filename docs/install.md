# Installation

Install the `0.1.0` release from PyPI:

```bash
python -m pip install 'gp_dla_finder==0.1.0'
```

The inference core requires NumPy and SciPy. It does not require a compiler.

For development, install from a checkout instead:

```bash
pip install -e '/path/to/gp_dla_finder[dev]'
```

## Extras

| extra | what it adds |
|---|---|
| `legacy` | `h5py`, for reading the original MATLAB v7.3 / HDF5 model and sample files |
| `desi` | `astropy`, `fitsio`; survey readers that are not required by the core |
| `sdss` | `astropy` |
| `performance` | `threadpoolctl`, for the BLAS thread diagnostic; see {doc}`performance` |
| `plot` | `matplotlib` |
| `dev` | everything above plus the test and lint toolchain |
| `docs` | Sphinx and the theme used to build these pages |

## What the released install gives you

The package provides a **universal wheel** (`py3-none-any`) and a source
distribution. The ordinary PyPI installation

```bash
pip install gp_dla_finder
```

installs the wheel, so there is no compiler or local build step. It uses the
official NumPy Voigt backend, which is the configuration we recommend for v0.1.

The wheel does not carry a compiled extension. If you want the optional
libcerf backend for fidelity work, ask for a source build:

```bash
pip install --no-binary gp_dla_finder gp_dla_finder
```

That runs the build described in the next section. It compiles the extension
when libcerf is available and falls back to the NumPy backend when it is not.
Either way, the installation should still succeed.

## The optional compiled backend

If [libcerf](https://jugit.fz-juelich.de/mlz/libcerf) is already available, the
installer tries to compile an additional Voigt backend. We keep this backend for
fidelity, not speed. Roman Garnett's original
[`gp_dla_detection`](https://github.com/rmgarnett/gp_dla_detection) MATLAB
package used libcerf for its Faddeeva-function calculation, so the optional
backend keeps that historical numerical path available. Its end-to-end runtime
differs from the NumPy backend by only a few percent in the measured tests. The
backend is registered only if it passes the numerical agreement checks for that
installation.

```bash
brew install libcerf          # macOS
apt install libcerf-dev       # Debian / Ubuntu
conda install -c conda-forge libcerf
```

If libcerf is missing, compilation fails, or the agreement check rejects the
extension, installation still succeeds with the official NumPy backend. You can
check which backends are actually available:

```python
from gp_dla_finder.voigt import available_backends, backend_provenance

available_backends()
dict(backend_provenance("numpy"))
```

If you select a backend that is not available, the package raises an error. It
will not quietly switch you to a different forward model.

```python
from gp_dla_finder.config import Config

Config.desi_y3().replace(voigt_backend="libcerf")  # raises if not built
```

## Source distributions

The source distribution contains `_voigt_ext.pyx`, and Cython is used only when
building the optional extension. It is the route to the libcerf backend, since
the released wheel does not carry one.

You can also turn the extension off explicitly, which is how the release wheel
itself is built:

```bash
GP_DLA_FINDER_NO_COMPILE=1 python -m build --wheel
```

That produces a `py3-none-any` wheel on any machine, whether or not libcerf is
installed there.

Platform wheels with a prebuilt extension are not planned for v0.1. Supporting
them across the operating-system and interpreter matrix would add substantial
CI work, while the NumPy backend remains the recommended configuration. We can
revisit binary extension wheels later if there is enough demand.
