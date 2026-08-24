# Provenance and reproducibility

A `Result` should still make sense after it leaves the machine that produced it,
so we record the numerical choices that affect the answer.

## Compatibility profiles

To match the reference bitwise, we need to keep two floating-point operations
that cancel mathematically: a rest-frame wavelength round trip and a $-\log N$ /
$+\log N$ pair around the log-mean-exp. We put them in named compatibility
profiles so they do not become hidden parts of the scientific model.

```python
from gp_dla_finder.compat import COMPATIBILITY_PROFILES

sorted(COMPATIBILITY_PROFILES)  # ['clean', 'reference-d5b306e6']
```

`reference-d5b306e6` is the default. `clean` removes both operations. On the
retained reference spectrum, the null evidence changes by
$4.2\times10^{-12}$ nat.

## Quality policies

There are two separate questions: can we run on this spectrum, and do you want it
in your selected catalog? Structural validation answers the first. A named
quality policy answers the second.

```python
from gp_dla_finder.quality import DESI_Y3_REFERENCE

DESI_Y3_REFERENCE.summary
```

A quality rejection has its own status and reason code. It is not reported as a
non-detection.

## Backend identity

The compiled and NumPy backends do not agree in exactly the same way on every
platform. What we measured is:

* Linux/x86-64, Debian libcerf 2.4, in the project's CI environment: bitwise
  identical on all 15 probed profiles and on the retained end-to-end evidence
  workload;
* macOS/arm64, Homebrew libcerf 2.4: differ by up to ~10⁻¹³ absolute.

We should not generalize these numbers to other inputs, library versions,
compilers, or architectures. That is why the provenance records the backend,
libcerf version, linked-library content hash, and toolchain family.

```python
from gp_dla_finder.voigt import backend_provenance

dict(backend_provenance("numpy"))
```

Shareable provenance contains **no filesystem paths**. Machine-local paths are
available only through `backend_local_diagnostics()`.

## Asset provenance

The source-code license does not automatically cover the bundled model
parameters, prior tables, or sample grids. `NOTICE.md` records
where they came from and how they may be redistributed.

The package contains no 2LPT mock spectra, truth catalogs, cutouts, private mock
identifiers, or reconstructable extracts. Test spectra are generated from named
seeds by code in the repository.
