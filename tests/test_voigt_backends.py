"""The optional compiled backend: availability, agreement, and honesty.

PI increment-8 priorities 1 and 2 asked for automatic compilation with a safe
NumPy fallback, an explicit active backend, and *measured* compiled-versus-NumPy
parity. These tests cover all four, and are written so they say something useful
whether or not the extension was built:

* the fallback tests always run;
* the agreement tests skip cleanly, and loudly, when the extension is absent.

The headline measurement lives in
:func:`test_libcerf_agrees_with_numpy_within_the_declared_tolerance`: libcerf and
SciPy are *not* the same function, and the size of the difference is the reason
the backend name has to travel in result provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from gp_dla_finder import voigt as V

COMPILED = "libcerf" in V.available_backends()
needs_compiled = pytest.mark.skipif(
    not COMPILED, reason="the optional libcerf extension was not built here"
)

#: Redshift and column density spanning the sample grid, including the saturated
#: regime where profiles underflow.
GRID = [(z, n) for z in (2.0, 2.6, 3.4) for n in (17.2, 19.0, 20.3, 22.0, 23.0)]

PROBE = np.linspace(3600.0, 6000.0, 512)


# --------------------------------------------------------------------------
# Availability and fallback
# --------------------------------------------------------------------------


def test_the_numpy_backend_is_always_available():
    """The default install requires no compiler (PI ruling D3)."""
    assert "numpy" in V.available_backends()
    assert V.get_backend("numpy").name == "numpy"


def test_an_unbuilt_backend_raises_rather_than_falling_back():
    """Silently substituting a backend would change the forward model."""
    with pytest.raises(ValueError, match="unknown or unavailable"):
        V.get_backend("no-such-backend")
    with pytest.raises(ValueError, match="unknown or unavailable"):
        V.voigt_absorption(PROBE, nhi=1e20, z_dla=2.5, backend="no-such-backend")


def test_backend_provenance_names_the_faddeeva_implementation():
    record = dict(V.backend_provenance("numpy"))
    assert record["faddeeva_source"] == "scipy.special.wofz"
    assert record["compiled"] is False


@needs_compiled
def test_compiled_backend_provenance_records_its_measured_agreement():
    record = dict(V.backend_provenance("libcerf"))
    assert record["compiled"] is True
    assert record["faddeeva_source"] == "libcerf"
    # Registration is conditional on these being measured, so they must be present.
    assert (
        0.0 < record["max_absolute_difference_from_numpy"] <= V.BACKEND_AGREEMENT_ATOL
    )
    assert 0.0 < record["max_absolute_difference_broadened"] <= V.BACKEND_AGREEMENT_ATOL
    # 0.0 is now a MEANINGFUL value: it says no probe point disagreed by more
    # than BACKEND_DECREMENT_ATOL, so the relative test never applied. The
    # absolute figure is recorded next to it precisely so that zero is readable.
    relative = record["max_relative_decrement_difference"]
    assert 0.0 <= relative <= V.BACKEND_DECREMENT_RTOL
    assert record["max_absolute_decrement_difference"] >= 0.0
    if relative == 0.0:
        assert record["max_absolute_decrement_difference"] <= V.BACKEND_DECREMENT_ATOL


@needs_compiled
def test_compiled_backend_provenance_identifies_what_it_linked():
    """A compiled backend's numbers depend on which libcerf it linked and how.

    Naming the backend without naming the build makes a result unreproducible
    (PI ruling, increment-8 correction 4).
    """
    record = dict(V.backend_provenance("libcerf"))
    for key in (
        "libcerf_version",
        "libcerf_version_source",
        "libcerf_sha256",
        "wrapper_toolchain",
        "wrapper_cflags",
        "libcerf_build_flags",
        "platform",
        "cython_version",
    ):
        assert key in record, f"backend provenance is missing {key!r}"

    # The identity is the bytes, and the path is deliberately NOT here.
    assert len(record["libcerf_sha256"]) == 64
    assert record["libcerf_version"] != "unknown"
    assert "libcerf_resolved_path" not in record
    assert record["wrapper_toolchain"] in {"clang", "gcc", "msvc", "cc", "unknown"}


@needs_compiled
def test_build_provenance_carries_no_machine_specific_paths():
    """Provenance travels with results, and results get shared.

    Compiler flag strings routinely contain the builder's home directory via
    ``-isystem``/``-I``. Those are redacted; the count of dropped tokens is
    recorded so the removal is visible rather than silent.
    """
    import re

    record = dict(V.backend_provenance("libcerf"))
    home = str(Path.home())
    for key, value in record.items():
        text = str(value)
        assert home not in text, f"{key} leaks the build machine's home directory"
    # No bare absolute paths survive in the flag strings.
    for key in ("wrapper_cflags", "wrapper_opt_flags"):
        assert not re.search(r"(^|\s)/", str(record.get(key, ""))), (
            f"{key} still contains an absolute path"
        )
    assert record["wrapper_flags_redacted_count"] >= 0


# --------------------------------------------------------------------------
# Agreement, measured rather than assumed
# --------------------------------------------------------------------------


@needs_compiled
@pytest.mark.parametrize(("z_dla", "log_nhi"), GRID)
def test_libcerf_agrees_with_numpy_within_the_declared_tolerance(z_dla, log_nhi):
    """Absolute agreement, because the profile multiplies the model mean.

    Relative agreement is deliberately *not* asserted: in the saturated core of a
    strong absorber ``exp(-tau)`` underflows towards zero, and a last-bit
    difference in an optical depth of order 1e4 shows up as a large relative
    error on a number that is already 1e-23. Absolute error is the quantity with
    a physical interpretation here.
    """
    kwargs = dict(nhi=10.0**log_nhi, z_dla=z_dla, num_lines=3, broadening=False)
    expected = V.voigt_absorption(PROBE, backend="numpy", **kwargs)
    actual = V.voigt_absorption(PROBE, backend="libcerf", **kwargs)

    assert np.max(np.abs(actual - expected)) <= V.BACKEND_AGREEMENT_ATOL
    assert np.all(actual >= 0.0) and np.all(actual <= 1.0)


@needs_compiled
def test_libcerf_and_scipy_agreement_is_platform_dependent():
    """Whether they agree bitwise depends on the platform -- which is the point.

    Scope, stated precisely (PI ruling, increment-11 correction 5). These are
    results for *tested environments and a finite probe set*, not general facts:

    * on the macOS/arm64 development machine with Homebrew libcerf 2.4, the two
      differ on most probed pixels, up to ~1e-13 absolute;
    * in the named Linux/x86-64 GitHub Actions environment with Debian libcerf
      2.4, they were measured bitwise identical on all 15 probes and on the
      retained end-to-end evidence workload.

    Neither generalises to other inputs, other libcerf or SciPy versions, or
    other compiler and architecture combinations.

    An earlier version of this test *required* them to differ, and duly failed in
    CI on Linux. That was asserting an accident of one platform. What is actually
    load-bearing -- and what this asserts -- is that the difference is always
    bounded, and the count is reported so the platform-dependence stays visible.

    It is precisely because agreement varies by platform and build that the
    backend identity has to travel in result provenance.
    """
    differing = 0
    worst = 0.0
    for z_dla, log_nhi in GRID:
        kwargs = dict(nhi=10.0**log_nhi, z_dla=z_dla, num_lines=3, broadening=False)
        expected = V.voigt_absorption(PROBE, backend="numpy", **kwargs)
        actual = V.voigt_absorption(PROBE, backend="libcerf", **kwargs)
        if not np.array_equal(expected, actual):
            differing += 1
            worst = max(worst, float(np.max(np.abs(actual - expected))))

    print(
        f"\nlibcerf vs scipy: {differing}/{len(GRID)} probes differ, "
        f"worst {worst:.3e} absolute"
    )
    assert worst <= V.BACKEND_AGREEMENT_ATOL


@needs_compiled
def test_the_broadening_convolution_is_shared_by_both_backends():
    """A backend switch must change the Voigt function and nothing else."""
    kernel = V.PRODUCTION_KERNEL
    half = V.kernel_half_width(kernel)
    padded = np.linspace(3600.0, 6000.0, 512)

    raw_numpy = V.voigt_absorption(
        padded, nhi=1e20, z_dla=2.5, kernel=kernel, broadening=False, backend="numpy"
    )
    broadened_numpy = V.voigt_absorption(
        padded, nhi=1e20, z_dla=2.5, kernel=kernel, broadening=True, backend="numpy"
    )
    broadened_libcerf = V.voigt_absorption(
        padded, nhi=1e20, z_dla=2.5, kernel=kernel, broadening=True, backend="libcerf"
    )

    assert broadened_numpy.size == raw_numpy.size - 2 * half
    assert broadened_libcerf.size == broadened_numpy.size
    # Same convolution applied to profiles that differ only at 1e-14.
    assert np.max(np.abs(broadened_libcerf - broadened_numpy)) <= (
        V.BACKEND_AGREEMENT_ATOL
    )


@needs_compiled
def test_the_compiled_backend_carries_no_atomic_data_of_its_own():
    """Single source of truth: the constants come from this module.

    Passing deliberately wrong constants must change the answer. If it did not,
    the extension would be carrying its own copy, which is exactly the drift this
    design avoids.
    """
    from gp_dla_finder import _voigt_ext

    normal = _voigt_ext.raw_absorption(
        np.ascontiguousarray(PROBE),
        1e20,
        2.5,
        V.TRANSITION_WAVELENGTHS[:3],
        V.LEADING_CONSTANTS[:3],
        V.GAMMAS_CGS[:3],
        V.SIGMA_CGS,
        V.C_CGS,
    )
    perturbed = _voigt_ext.raw_absorption(
        np.ascontiguousarray(PROBE),
        1e20,
        2.5,
        V.TRANSITION_WAVELENGTHS[:3],
        np.ascontiguousarray(np.asarray(V.LEADING_CONSTANTS[:3]) * 2.0),
        V.GAMMAS_CGS[:3],
        V.SIGMA_CGS,
        V.C_CGS,
    )
    assert not np.array_equal(normal, perturbed)


# --------------------------------------------------------------------------
# The gate itself must fail closed (PI ruling N40)
# --------------------------------------------------------------------------


class _FakeBackend:
    """A backend that misbehaves in exactly one named way."""

    name = "fake"

    def __init__(self, mode):
        self.mode = mode
        self.real = V.get_backend("numpy")

    def absorption(self, wavelengths, nhi, z_dla, num_lines, kernel, broadening):
        out = np.array(
            self.real.absorption(wavelengths, nhi, z_dla, num_lines, kernel, broadening)
        )
        if self.mode == "nan":
            out[len(out) // 2] = np.nan
        elif self.mode == "short":
            out = out[:-1]
        elif self.mode == "above_one":
            out[0] = 1.5
        elif self.mode == "negative":
            out[0] = -1e-9
        elif self.mode == "biased":
            out = out * (1.0 + 1e-6)
        elif self.mode == "decrement":
            # Absolutely tiny, relatively enormous: shifts a profile of ~0.999999
            # by 1e-9, which is a 0.1% error on the decrement that carries the
            # signal. The absolute gate alone would wave this through.
            out = np.where(out > 0.5, out - 1e-9, out)
        return out


@pytest.mark.parametrize(
    "mode", ["nan", "short", "above_one", "negative", "biased", "decrement"]
)
def test_the_agreement_gate_rejects_a_broken_backend(mode):
    """Each failure mode must be caught, and the NaN one directly.

    The NaN case is not hypothetical: the previous gate accumulated a running
    maximum with ``max(previous, np.nan)``, which in Python returns ``previous``.
    A backend emitting NaN could therefore register with its error recorded as
    zero. That is now an explicit finiteness check, before any maximum is taken.
    """
    with pytest.raises(V.BackendRejected):
        V._verify_against_numpy(_FakeBackend(mode))


def test_python_max_really_does_swallow_nan():
    """The mechanism behind the hole, pinned so the reasoning stays checkable."""
    assert max(0.0, float("nan")) == 0.0
    assert np.isnan(max(float("nan"), 0.0))


def test_the_gate_accepts_the_numpy_backend_against_itself():
    """A sanity floor: the checks must not be so strict they reject the truth."""
    measured = V._verify_against_numpy(V.get_backend("numpy"))
    assert measured["max_absolute_difference_from_numpy"] == 0.0
    assert measured["max_absolute_difference_broadened"] == 0.0
    assert measured["max_relative_decrement_difference"] == 0.0


def test_a_rejected_backend_is_reported_rather_than_silently_missing():
    """ "Not built" and "built and refused" are different states."""
    rejections = V.backend_rejections()
    assert isinstance(rejections, Mapping)
    if COMPILED:
        assert "libcerf" not in rejections


@needs_compiled
def test_end_to_end_evidence_agrees_across_backends():
    """The check deliberately left out of the import-time gate.

    Profile agreement is necessary but not sufficient: what matters is whether
    the quantity a user reads changes. Too slow for import, so it lives here.
    """
    from gp_dla_finder import load_model, load_sample_grid
    from gp_dla_finder.config import Config
    from gp_dla_finder.gp.evidence import assemble_model, one_absorber_log_evidence
    from gp_dla_finder.gp.spectrum import prepare_spectrum
    from synthetic import make_spectrum

    model = load_model()
    values = {}
    for backend in ("numpy", "libcerf"):
        config = Config.desi_y3_fast().replace(voigt_backend=backend)
        grid = load_sample_grid(config.sample_grid)
        prepared = prepare_spectrum(make_spectrum(), model, config)
        assembled = assemble_model(prepared, model, config)
        values[backend] = one_absorber_log_evidence(
            prepared, assembled, grid, config, mode="filter"
        )

    difference = abs(values["numpy"] - values["libcerf"])
    # Bounded, not necessarily non-zero. On Linux/x86-64 the two agree exactly;
    # on macOS/arm64 they differ by ~7e-13 nat. Requiring a non-zero difference
    # here asserted an accident of one platform and failed in CI on the other.
    assert difference < 1e-9, f"backends differ by {difference:g} nat"


@needs_compiled
def test_the_backend_is_selectable_through_the_ordinary_configuration():
    """No monkeypatching required, and an unbuilt backend raises at config time."""
    from gp_dla_finder.config import Config

    assert Config.desi_y3().voigt_backend == "numpy"
    assert Config.desi_y3().replace(voigt_backend="libcerf").voigt_backend == "libcerf"
    with pytest.raises(ValueError, match="unknown or unavailable"):
        Config.desi_y3().replace(voigt_backend="not-a-backend")


@needs_compiled
def test_mismatched_atomic_data_is_rejected():
    from gp_dla_finder import _voigt_ext

    with pytest.raises(ValueError, match="matching lengths"):
        _voigt_ext.raw_absorption(
            np.ascontiguousarray(PROBE),
            1e20,
            2.5,
            V.TRANSITION_WAVELENGTHS[:3],
            V.LEADING_CONSTANTS[:2],
            V.GAMMAS_CGS[:3],
            V.SIGMA_CGS,
            V.C_CGS,
        )


# --- the hybrid decrement gate (PI ruling N62) ------------------------------
#
# The old gate divided by a decrement that goes to zero in the far wings, so a
# 1e-16 round-off difference against a 1e-8 decrement read as a 1e-8 relative
# error. That refused the compiled backend on Linux while its absolute profile
# agreement was ~1e-14. The floor was chosen from measurement; these tests pin
# both sides of it.


class _WingNoiseBackend:
    """Round-off in the far wings only: the case that must NOT be rejected."""

    name = "wing-noise"

    def __init__(self):
        self.real = V.get_backend("numpy")

    def absorption(self, wavelengths, nhi, z_dla, num_lines, kernel, broadening):
        out = np.array(
            self.real.absorption(wavelengths, nhi, z_dla, num_lines, kernel, broadening)
        )
        # Perturb by one ulp-ish amount, far below the decrement floor, exactly
        # where the decrement is vanishing.
        shallow = out > 1.0 - 1e-6
        out[shallow] = out[shallow] - 1e-16
        return np.clip(out, 0.0, 1.0)


def test_round_off_in_the_wings_is_not_a_rejection():
    measured = V._verify_against_numpy(_WingNoiseBackend())
    assert measured["max_relative_decrement_difference"] <= V.BACKEND_DECREMENT_RTOL


class _SubtleDecrementBias:
    """Biased below the ABSOLUTE gate, but badly wrong on a deep decrement.

    This is the only reason the decrement test exists. ``_FakeBackend("decrement")``
    shifts by 1e-9 and is caught by the absolute profile gate long before the
    decrement one, so it does not demonstrate that the decrement gate earns its
    place. A 5e-12 shift does: it passes the 1e-11 absolute gate, and against a
    decrement of ~1e-6 it is a 5e-6 relative error on the quantity that carries
    the signal.
    """

    name = "subtle"

    def __init__(self):
        self.real = V.get_backend("numpy")

    def absorption(self, wavelengths, nhi, z_dla, num_lines, kernel, broadening):
        out = np.array(
            self.real.absorption(wavelengths, nhi, z_dla, num_lines, kernel, broadening)
        )
        deep = (out > 0.5) & (out < 1.0 - 1e-7)
        out[deep] = out[deep] - 5e-12
        return np.clip(out, 0.0, 1.0)


def test_a_meaningful_decrement_bias_is_still_rejected():
    """The floor must not sit high enough to let real bias through."""
    with pytest.raises(V.BackendRejected):
        V._verify_against_numpy(_FakeBackend("decrement"))


def test_the_decrement_gate_catches_what_the_absolute_gate_cannot():
    backend = _SubtleDecrementBias()

    # It really does pass the primary absolute gate ...
    reference = V.get_backend("numpy")
    probe = np.linspace(3600.0, 6000.0, 512)
    expected = reference.absorption(probe, 10**20.3, 2.6, 3, V.PRODUCTION_KERNEL, False)
    actual = backend.absorption(probe, 10**20.3, 2.6, 3, V.PRODUCTION_KERNEL, False)
    assert float(np.max(np.abs(actual - expected))) < V.BACKEND_AGREEMENT_ATOL

    # ... and is still rejected, by the decrement test.
    with pytest.raises(V.BackendRejected, match="decrement"):
        V._verify_against_numpy(backend)


def test_the_floor_sits_between_the_real_disagreement_and_a_real_bias():
    """The numbers the floor was chosen from, asserted as an invariant.

    Measured worst absolute decrement difference between the NumPy and libcerf
    backends: 4.8e-14 on macOS/arm64, 3.3e-16 on Linux/x86-64. The biased fake
    shifts by 1e-9. The floor has to separate those two populations, and a floor
    that stopped doing so would silently accept a biased backend.
    """
    assert 4.8e-14 < V.BACKEND_DECREMENT_ATOL < 1e-9
    # And a decade of headroom on each side, so a slightly noisier platform or a
    # slightly subtler bias does not land on the boundary.
    assert V.BACKEND_DECREMENT_ATOL > 10 * 4.8e-14
    assert V.BACKEND_DECREMENT_ATOL < 1e-9 / 10


def test_the_relative_figure_reads_with_its_absolute_companion():
    """A zero relative difference must be explained, not ambiguous."""
    measured = V._verify_against_numpy(V.get_backend("numpy"))
    assert measured["max_relative_decrement_difference"] == 0.0
    assert measured["max_absolute_decrement_difference"] == 0.0
