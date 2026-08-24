# Canonical parity environment — `canonical_reference_v0.1`

Formal **bitwise** equivalence claims between this package and the reference
implementation are made only inside this environment. Everywhere
else, parity is asserted at documented tolerances.

## Honest limitation

**This is not the historical DESI production environment.** No artifact reachable
by this project records the production stack's Python, NumPy, SciPy or BLAS
versions — the archived `BASELINE.env` records filesystem paths and science knobs
only. This image is therefore the *closest verifiable* reference environment, not
a reconstruction, and it is named `canonical_reference_v0.1` rather than anything
implying historical identity. If evidence establishing the production environment
is recovered, this record is superseded and the name changes.

## What is pinned

| item | value |
|---|---|
| base image | `python:3.10.14-slim-bookworm` |
| digest | `sha256:2407c61b1a18067393fecd8a22cf6fceede893b6aaca817bf9fbfe65e33614a3` |
| architecture | `linux/amd64` |
| Python | 3.10.14 |
| NumPy / SciPy | see `requirements.lock` |
| BLAS threading | forced to 1 thread — multi-threaded reductions reorder floating-point summation, which defeats bitwise comparison |
| `PYTHONHASHSEED` | 0 |

## Parity command

```bash
docker build --platform linux/amd64 -f parity/Dockerfile -t gpdlf-parity:v0.1 .
docker run --rm --platform linux/amd64 \
  -v /path/to/desi_gpy_dla_detection:/ref:ro \
  -e GP_DLA_FINDER_REFERENCE=/ref \
  gpdlf-parity:v0.1 \
  python -m pytest -q -m "not slow"
```

Both implementations run inside the image; the reference is mounted read-only.

## Lock generation

The lock covers top-level **and** transitive dependencies with hashes. It was
generated cross-platform — no Linux machine needed — by downloading the
`manylinux2014_x86_64` / `py3-none-any` wheels and hashing them:

```bash
pip download --only-binary=:all: --platform manylinux2014_x86_64 \
    --python-version 310 --implementation cp --dest wheels numpy==2.2.6 scipy==1.15.3
pip download --only-binary=:all: --platform any \
    --python-version 310 --implementation py --dest wheels pytest==8.3.3
for w in wheels/*.whl; do pip hash "$w"; done
```

## Status

**Built and run in CI, not on the development machine** (Docker is unavailable
there). `.github/workflows/parity.yml` has two jobs:

* `container-build` — builds this image and runs the offline suite inside it on
  every push to `main`. This is what makes the environment real rather than
  documented.
* `canonical-parity` — additionally checks out the private reference
  implementation and runs the comparison tests inside the image. It **fails** if
  the reference was unavailable or if no comparison test actually ran, because a
  green job containing no comparison is not a parity result.

The reference is mounted read-only and is never published or bundled by this
repository.

Every parity number reported **before** the first successful `canonical-parity`
run was obtained on macOS/arm64 with Python 3.10.6, NumPy 2.2.6, SciPy 1.15.3, and
is labelled non-canonical. Those are component comparisons, not end-to-end
inference parity.

### A concrete reason this environment matters

While repairing CI, a checksum test failed on Linux and passed on macOS. The cause
was not a flake: `nhi_samples` was being derived at load time as
`10 ** log_nhi_samples`, and float64 `**` goes through the platform `libm` `pow`,
which is not correctly rounded. The same asset therefore produced different last
bits on the two architectures. The fix was to store the array rather than derive
it. Any quantity reaching a bitwise claim through `pow`, `exp` or similar is
exposed to the same effect, which is precisely why formal claims are pinned to one
environment.

## Record to retain per formal parity run

Image digest · architecture · library versions · lock file · reference commit ·
package commit · asset hashes · parity command · full report.
