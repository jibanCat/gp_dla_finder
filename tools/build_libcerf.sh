#!/usr/bin/env bash
# Build a pinned libcerf release from source with CMake.
#
# PI ruling N37 required this to be investigated and measured, because the
# increment-8 report compiled the Cython wrapper against an already-installed
# Homebrew binary and could not therefore say whether a *properly built* libcerf
# would perform differently.
#
# The measured answer is: it does not. See the increment-9 report. A build with
# -O3 -march=native is indistinguishable from the distributed Homebrew binary,
# both in the Voigt component (0.0734 vs 0.0735 ms) and end to end, and the two
# builds agree with the NumPy backend to exactly the same value
# (9.439671266875393e-14 absolute), so they are numerically interchangeable.
#
# This script is retained so that result can be re-measured rather than trusted.
#
# IMPORTANT -- this is a developer/benchmark tool and is deliberately NOT wired
# into package installation. PI ruling N37 forbids an unpinned network download
# during install, and a source build at install time would be exactly that. The
# tarball URL and its sha256 are pinned here; the checksum is verified before
# anything is unpacked.
#
# Usage:
#   tools/build_libcerf.sh [PREFIX]           # default: ./build/libcerf-prefix
#
# Then point the extension build at it:
#   GP_DLA_FINDER_LIBCERF_INCLUDE=$PREFIX/include \
#   GP_DLA_FINDER_LIBCERF_LIB=$PREFIX/lib \
#     python setup.py build_ext --inplace
#
# Licence: libcerf is MIT (Copyright (c) 2012 Massachusetts Institute of
# Technology, (c) 2013 Forschungszentrum Juelich GmbH). It is NOT vendored into
# this repository and NOT redistributed by it -- this script fetches it onto the
# developer's own machine. See NOTICE.md.

set -euo pipefail

VERSION="2.4"
URL="https://jugit.fz-juelich.de/mlz/libcerf/-/archive/v${VERSION}/libcerf-v${VERSION}.tar.gz"
# Verified 2026-08-19. A mismatch means the archive changed under a fixed tag,
# which is a reason to stop, not a reason to continue.
SHA256="85c2f2c84a118f2f1714406d9251482e86ed4f15bac44f37888fa9d883fff04a"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_PREFIX="${REPO_ROOT}/build/libcerf-${VERSION}-prefix"

# ---------------------------------------------------------------------------
# Destructive-target safety.
#
# An early version of this script accepted any prefix and ran `rm -rf` on it, so
# `build_libcerf.sh /usr/local` would have requested deletion of /usr/local. The
# first fix refused a list of system paths, but review found three holes: a
# nonexistent parent left `..` components uncanonicalised so a traversal-shaped
# path could evade the exact comparisons; the mere PRESENCE of a manifest file
# was taken as proof of ownership; and extra positional arguments were ignored.
#
# The policy is now the narrowest one the PI approved (increment-11 correction 2):
#
#   * ONLY the exact, narrowly named default prefix under this repository's
#     build/ area is ever deleted automatically;
#   * a caller-supplied prefix must be nonexistent or empty. An existing
#     non-empty directory is REFUSED, never cleaned -- create a new prefix
#     instead;
#   * every path is fully canonicalised BEFORE any comparison or deletion, so a
#     traversal cannot reach a system directory through a nonexistent parent.
#
# The build manifest is still written, but it is provenance, not authority: no
# file on disk can grant this script permission to delete anything.
# ---------------------------------------------------------------------------
canonicalise() {
  # Resolve symlinks and `..` even when the path does not exist yet. Python is
  # already a hard requirement of this project, and `realpath -m` is not portable
  # to a stock macOS.
  python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "$1"
}

refuse() {
  echo "refusing to use prefix: $1" >&2
  echo "  the default prefix is the only directory this script will clear:" >&2
  echo "  $DEFAULT_PREFIX" >&2
  exit 1
}

if [ "$#" -gt 1 ]; then
  echo "usage: $(basename "$0") [PREFIX]" >&2
  echo "  refusing: $# positional arguments given; a destructive target must be" >&2
  echo "  unambiguous." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  RAW_PREFIX="$DEFAULT_PREFIX"
else
  # `${1:-default}` would substitute the default for an EXPLICITLY EMPTY argument
  # as well as a missing one, silently reinterpreting `build_libcerf.sh ""` as a
  # request for the default. Argument count distinguishes them.
  RAW_PREFIX="$1"
fi
[ -n "$RAW_PREFIX" ] || refuse "empty path"

case "$RAW_PREFIX" in
  *'$'* | *'*'* | *'?'* | *'`'*)
    refuse "unexpanded variable, glob or command substitution in '$RAW_PREFIX'"
    ;;
esac
# Checked BEFORE canonicalisation: a relative path would resolve against whatever
# directory the caller happened to be in, which is exactly the ambiguity a
# deletion target must not have.
case "$RAW_PREFIX" in
  /*) ;;
  *) refuse "'$RAW_PREFIX' is relative; give an absolute path" ;;
esac

PREFIX="$(canonicalise "$RAW_PREFIX")"
CANONICAL_DEFAULT="$(canonicalise "$DEFAULT_PREFIX")"
CANONICAL_REPO="$(canonicalise "$REPO_ROOT")"
CANONICAL_HOME="$(canonicalise "${HOME:-/nonexistent}")"

case "$PREFIX" in
  /*) ;;
  *) refuse "'$RAW_PREFIX' did not canonicalise to an absolute path" ;;
esac
case "$PREFIX" in
  *'/../'* | */..) refuse "'$PREFIX' still contains a traversal component" ;;
esac

# Broad targets, checked on the CANONICAL path so a traversal cannot slip past.
case "$PREFIX" in
  /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/opt|/private|/proc|/root|/sbin|/srv|/sys|/tmp|/usr|/usr/bin|/usr/lib|/usr/local|/usr/local/bin|/usr/local/lib|/var|/Applications|/Library|/System|/Users|/Volumes)
    refuse "'$PREFIX' is a system or root-level directory"
    ;;
esac
[ "$PREFIX" != "$CANONICAL_HOME" ] || refuse "'$PREFIX' is the home directory"
[ "$PREFIX" != "$CANONICAL_REPO" ] || refuse "'$PREFIX' is the repository root"
[ "$PREFIX" != "${CANONICAL_REPO}/build" ] || refuse "'$PREFIX' is the whole build/ area"
if [ "$(echo "$PREFIX" | awk -F/ '{print NF-1}')" -lt 3 ]; then
  refuse "'$PREFIX' is too close to the filesystem root to be tool-owned"
fi

if [ -e "$PREFIX" ] || [ -L "$PREFIX" ]; then
  if [ -L "$PREFIX" ]; then
    refuse "'$RAW_PREFIX' is a symlink; deletion targets must not be indirect"
  elif [ ! -d "$PREFIX" ]; then
    refuse "'$PREFIX' exists and is not a directory"
  elif [ "$PREFIX" = "$CANONICAL_DEFAULT" ]; then
    # The one directory this script owns by construction.
    echo "clearing the default prefix $PREFIX"
    rm -rf "$PREFIX"
  elif [ -z "$(ls -A "$PREFIX" 2>/dev/null)" ]; then
    echo "using existing empty prefix $PREFIX"
  else
    refuse "'$PREFIX' already exists and is not empty; this script only clears its own default prefix"
  fi
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "libcerf ${VERSION} -> ${PREFIX}"

curl -sSL --fail -o "$WORK/libcerf.tar.gz" "$URL"

if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$WORK/libcerf.tar.gz" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$WORK/libcerf.tar.gz" | cut -d' ' -f1)"
fi
if [ "$actual" != "$SHA256" ]; then
  echo "checksum mismatch for libcerf ${VERSION}" >&2
  echo "  expected $SHA256" >&2
  echo "  actual   $actual" >&2
  exit 1
fi
echo "checksum ok: $actual"

tar xzf "$WORK/libcerf.tar.gz" -C "$WORK"
SRC="$(find "$WORK" -maxdepth 1 -type d -name 'cerf-*' | head -1)"

cmake -S "$SRC" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_FLAGS="${LIBCERF_CFLAGS:--O3}" \
  -DCERF_CPP=OFF
cmake --build "$WORK/build" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
cmake --install "$WORK/build"

# Leave an authoritative record in the prefix. Without it the extension build has
# to guess the version from pkg-config or a filename, and when an explicit prefix
# is selected a bare pkg-config answers about a DIFFERENT library entirely
# (PI ruling, increment-9 correction 4). It also keeps libcerf's own build flags
# distinct from the Python wrapper's, which were previously conflated.
cat > "$PREFIX/lib/gp_dla_finder_libcerf_build.json" <<JSON
{
  "version": "${VERSION}",
  "provenance": "built from pinned upstream source with CMake",
  "source_url": "${URL}",
  "source_sha256": "${SHA256}",
  "build_type": "Release",
  "c_flags": "${LIBCERF_CFLAGS:--O3}",
  "cerf_cpp": "OFF"
}
JSON

echo
echo "installed:"
ls -l "$PREFIX/lib" | grep -E 'libcerf' || true
echo
echo "point the extension build at it with:"
echo "  GP_DLA_FINDER_LIBCERF_INCLUDE=$PREFIX/include \\"
echo "  GP_DLA_FINDER_LIBCERF_LIB=$PREFIX/lib \\"
echo "    python setup.py build_ext --inplace"
