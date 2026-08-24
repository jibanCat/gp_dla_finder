# Canonical reference environment for bitwise parity claims.
#
# NAME: canonical_reference_v0.1
#
# This is NOT the historical DESI production environment. That environment's
# exact Python/NumPy/SciPy versions are not recorded in any artifact available to
# this project -- the archived production BASELINE.env records paths and science
# knobs but no environment manifest. This image is the closest environment that
# can actually be verified, and is labelled accordingly.
#
# Both the reference implementation and this package run INSIDE this image when a
# bitwise parity claim is made. Other platforms get tolerance-based portability
# tests only.
#
# Base image pinned by digest so the environment is reproducible even if the tag
# is re-pointed upstream.
FROM python@sha256:2407c61b1a18067393fecd8a22cf6fceede893b6aaca817bf9fbfe65e33614a3
# ^ python:3.10.14-slim-bookworm, linux/amd64

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Pin BLAS threading: multi-threaded reductions can reorder floating-point
    # summation, which defeats bitwise comparison.
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /work

# Build context is the REPOSITORY ROOT, so the lock is at parity/requirements.lock.
# An earlier revision copied `requirements.lock`, which cannot resolve from that
# context -- the documented build command could never have worked.
COPY parity/requirements.lock /work/parity/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /work/parity/requirements.lock

# Adding setup.py for the optional extension made pip take its LEGACY
# `setup.py develop` path, which skips build isolation and therefore used the
# base image's setuptools 65 -- too old for the PEP 639 `license = "MIT"`
# expression, so the canonical image stopped building. Pin a modern setuptools
# into the image and force the PEP 517 path with no isolation, so the build uses
# exactly what is installed here and stays reproducible.
RUN python -m pip install --no-cache-dir "setuptools==80.9.0"

COPY . /work
RUN python -m pip install --no-cache-dir --no-build-isolation --use-pep517 \
    --no-deps -e .

CMD ["python", "-m", "pytest", "-q"]
