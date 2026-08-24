"""Voigt forward-model tests.

Two kinds:

* property tests, which need nothing external and always run;
* an equivalence test against the reference implementation, which is the actual
  guarantee that the port is scientifically inert.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import voigt as v

DESI_GRID = np.arange(3600.0, 5600.0, 0.8)


# --------------------------------------------------------------------------
# Kernel handling
# --------------------------------------------------------------------------


def test_production_kernel_is_normalised_and_symmetric():
    k = v.lsf_kernel(v.PRODUCTION_KERNEL)
    assert k.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.array_equal(k, k[::-1])
    assert len(k) % 2 == 1


def test_production_kernel_is_the_desi_one_not_the_boss_one():
    """Guards against reintroducing the stale BOSS R=2000 kernel (PI ruling D3).

    The BOSS kernel's centre tap is 0.4327; the DESI kernel's is 0.6483. Anything
    near the former means the wrong forward model is installed.
    """
    k = v.lsf_kernel(v.PRODUCTION_KERNEL)
    assert k[3] == pytest.approx(0.6482673794178272, abs=1e-15)
    sigma_pixels = np.sqrt((k * np.arange(-3, 4.0) ** 2).sum())
    assert sigma_pixels == pytest.approx(0.609, abs=0.005)


def test_unknown_kernel_raises_rather_than_substituting():
    with pytest.raises(KeyError, match="unknown LSF kernel"):
        v.lsf_kernel("boss-r2000")


def test_unknown_backend_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match="unknown or unavailable"):
        v.voigt_absorption(DESI_GRID, nhi=1e21, z_dla=2.5, backend="ctypes")


# --------------------------------------------------------------------------
# Profile properties
# --------------------------------------------------------------------------


def test_profile_is_a_transmission_and_shortens_by_the_kernel_width():
    p = v.voigt_absorption(DESI_GRID, nhi=10**21.0, z_dla=2.5)
    assert p.shape == (DESI_GRID.size - 2 * v.kernel_half_width(v.PRODUCTION_KERNEL),)
    assert p.min() >= 0.0
    assert p.max() <= 1.0 + 1e-12


def test_no_broadening_preserves_length():
    p = v.voigt_absorption(DESI_GRID, nhi=10**21.0, z_dla=2.5, broadening=False)
    assert p.shape == DESI_GRID.shape


@pytest.mark.parametrize("z_dla", [2.0, 3.0, 4.0])
def test_equivalent_width_increases_with_column_density(z_dla):
    ew = [
        np.sum(1.0 - v.voigt_absorption(DESI_GRID, nhi=10.0**n, z_dla=z_dla))
        for n in (19.0, 20.0, 21.0, 22.0)
    ]
    assert np.all(np.diff(ew) > 0)


def test_line_centre_lands_at_the_redshifted_lyman_alpha_wavelength():
    z_dla = 2.5
    p = v.voigt_absorption(DESI_GRID, nhi=10**20.5, z_dla=z_dla, broadening=False)
    lya_angstrom = v.TRANSITION_WAVELENGTHS[0] * 1e8
    assert DESI_GRID[p.argmin()] == pytest.approx(lya_angstrom * (1 + z_dla), abs=1.0)


def test_more_lyman_lines_add_absorption_and_never_remove_it():
    one = v.voigt_absorption(DESI_GRID, nhi=10**21.0, z_dla=3.5, num_lines=1)
    three = v.voigt_absorption(DESI_GRID, nhi=10**21.0, z_dla=3.5, num_lines=3)
    assert np.all(three <= one + 1e-15)
    assert np.sum(1.0 - three) > np.sum(1.0 - one)


@pytest.mark.parametrize("num_lines", [0, -1, v.MAX_LYMAN_LINES + 1])
def test_num_lines_out_of_range_raises(num_lines):
    with pytest.raises(ValueError, match="num_lines"):
        v.voigt_absorption(DESI_GRID, nhi=1e21, z_dla=2.5, num_lines=num_lines)


def test_backend_is_deterministic():
    a = v.voigt_absorption(DESI_GRID, nhi=10**20.7, z_dla=2.9)
    b = v.voigt_absorption(DESI_GRID, nhi=10**20.7, z_dla=2.9)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Equivalence with the reference implementation
# --------------------------------------------------------------------------


@pytest.mark.needs_reference
@pytest.mark.parametrize("z_dla", [2.0, 2.5, 3.2, 4.1])
@pytest.mark.parametrize("log_nhi", [17.2, 19.0, 20.3, 21.0, 22.5])
@pytest.mark.parametrize("num_lines", [1, 3, 6, 31])
def test_bit_identical_to_reference(reference_voigt, z_dla, log_nhi, num_lines):
    """Port equivalence, at the strictest tolerance there is: bitwise.

    The reference's pure-Python module differs from the compiled extension it
    stands in for *only* in its ``instrument_profile`` array. Substituting the
    production kernel isolates the maths, which must then agree exactly.
    """
    reference_voigt.instrument_profile = v.lsf_kernel(v.PRODUCTION_KERNEL)

    expected = reference_voigt.voigt_absorption(
        DESI_GRID, nhi=10.0**log_nhi, z_dla=z_dla, num_lines=num_lines
    )
    actual = v.voigt_absorption(
        DESI_GRID, nhi=10.0**log_nhi, z_dla=z_dla, num_lines=num_lines
    )
    assert np.array_equal(actual, expected)


@pytest.mark.needs_reference
def test_atomic_data_matches_reference(reference_voigt):
    assert np.array_equal(
        v.TRANSITION_WAVELENGTHS, reference_voigt.transition_wavelengths
    )
    assert np.array_equal(v.OSCILLATOR_STRENGTHS, reference_voigt.oscillator_strengths)
    assert np.array_equal(v.LEADING_CONSTANTS, reference_voigt.leading_constants)
    assert np.array_equal(v.GAMMAS_CGS, reference_voigt.gammas)
    assert v.SIGMA_CGS == reference_voigt.sigma
    assert v.C_CGS == reference_voigt.c


# --------------------------------------------------------------------------
# BOSS/eBOSS kernel (PI ruling N22)
# --------------------------------------------------------------------------


def test_boss_kernel_is_available_and_distinct_from_desi():
    boss = v.lsf_kernel(v.BOSS_KERNEL)
    desi = v.lsf_kernel(v.PRODUCTION_KERNEL)
    assert boss.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.array_equal(boss, boss[::-1])
    assert not np.array_equal(boss, desi)
    # The BOSS kernel is the broader of the two: R=2000 versus R=3000.
    offsets = np.arange(-3, 4.0)
    assert np.sqrt((boss * offsets**2).sum()) > np.sqrt((desi * offsets**2).sum())


def test_boss_kernel_centre_tap_is_the_historical_value():
    assert v.lsf_kernel(v.BOSS_KERNEL)[3] == pytest.approx(
        0.4327074389374541, abs=1e-15
    )


def test_boss_kernel_is_immutable():
    kernel = v.lsf_kernel(v.BOSS_KERNEL)
    with pytest.raises(ValueError, match="read-only"):
        kernel[3] = 0.5
    with pytest.raises(ValueError, match="WRITEABLE"):
        kernel.setflags(write=True)


@pytest.mark.needs_reference
@pytest.mark.parametrize("z_dla", [2.2, 3.4])
@pytest.mark.parametrize("log_nhi", [19.0, 20.3, 21.5])
def test_boss_kernel_is_bit_identical_to_the_unmodified_reference(
    reference_voigt, z_dla, log_nhi
):
    """Parity with the reference's *native* kernel, no substitution needed.

    The reference's pure-Python module ships the BOSS kernel, so this compares
    against it as written rather than against a patched copy — a stronger form of
    the equivalence argument than the DESI case can make.
    """
    import importlib

    reference = importlib.reload(reference_voigt)  # undo any earlier patching
    assert reference.instrument_profile[3] == pytest.approx(0.4327074389374541)

    expected = reference.voigt_absorption(
        DESI_GRID, nhi=10.0**log_nhi, z_dla=z_dla, num_lines=3
    )
    actual = v.voigt_absorption(
        DESI_GRID, nhi=10.0**log_nhi, z_dla=z_dla, num_lines=3, kernel=v.BOSS_KERNEL
    )
    assert np.array_equal(actual, expected)


# --------------------------------------------------------------------------
# Flexible resolving-power kernels (PI ruling N22)
# --------------------------------------------------------------------------


def test_gaussian_kernel_conventions():
    kernel = v.gaussian_lsf_kernel(
        resolving_power=3000.0, pixel_scale=0.8, wavelength=3800.0
    )
    assert len(kernel) % 2 == 1
    assert kernel.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.array_equal(kernel, kernel[::-1])
    assert kernel.argmax() == len(kernel) // 2


def test_gaussian_kernel_width_follows_the_fwhm_definition():
    """FWHM = lambda / R, so sigma_pixels = lambda / (R * 2.355 * pixel_scale)."""
    R, pixel_scale, wavelength = 3000.0, 0.8, 3800.0
    kernel = v.gaussian_lsf_kernel(R, pixel_scale, wavelength)
    half = len(kernel) // 2
    offsets = np.arange(-half, half + 1.0)
    measured = np.sqrt((kernel * offsets**2).sum())
    expected = wavelength / (R * 2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale)
    # Truncation at 4 sigma removes a little of the tail, so the measured second
    # moment is slightly below the analytic one.
    assert measured == pytest.approx(expected, rel=0.12)


def test_higher_resolving_power_gives_a_narrower_kernel():
    def width(R):
        k = v.gaussian_lsf_kernel(R, 0.2, 4000.0)
        half = len(k) // 2
        return np.sqrt((k * np.arange(-half, half + 1.0) ** 2).sum()) * 0.2

    assert width(2000.0) > width(3000.0) > width(5000.0)


def test_gaussian_kernel_is_immutable():
    kernel = v.gaussian_lsf_kernel(3000.0, 0.8, 3800.0)
    with pytest.raises(ValueError, match="WRITEABLE"):
        kernel.setflags(write=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolving_power": 0.0},
        {"resolving_power": -100.0},
        {"resolving_power": float("nan")},
        {"pixel_scale": 0.0},
        {"wavelength": float("inf")},
        {"truncate_sigma": 0.0},
    ],
)
def test_gaussian_kernel_rejects_invalid_inputs(kwargs):
    args = {"resolving_power": 3000.0, "pixel_scale": 0.8, "wavelength": 3800.0}
    args.update(kwargs)
    with pytest.raises(ValueError, match="finite and positive"):
        v.gaussian_lsf_kernel(**args)


def test_undersampled_lsf_is_refused_not_rounded_up():
    """A kernel narrower than a pixel is silently wrong, so it raises.

    Under-sampling comes from *high* resolving power on a coarse pixel grid: at
    R = 8000 and 3800 A on 0.8 A pixels the LSF sigma is ~0.25 pixels.
    """
    with pytest.raises(ValueError, match="under-samples"):
        v.gaussian_lsf_kernel(
            resolving_power=8000.0, pixel_scale=0.8, wavelength=3800.0
        )


def test_a_finer_pixel_grid_makes_the_same_resolving_power_representable():
    """The error message's suggested remedy actually works."""
    kernel = v.gaussian_lsf_kernel(
        resolving_power=8000.0, pixel_scale=0.2, wavelength=3800.0
    )
    assert kernel.sum() == pytest.approx(1.0, abs=1e-15)


def test_a_generated_kernel_can_drive_a_profile():
    grid = np.arange(3600.0, 4400.0, 0.8)
    kernel = v.gaussian_lsf_kernel(3000.0, 0.8, 3800.0)
    half = len(kernel) // 2
    raw = v.voigt_absorption(grid, nhi=10**20.8, z_dla=2.4, broadening=False)
    convolved = np.convolve(raw, kernel, "valid")
    assert convolved.shape == (grid.size - 2 * half,)
    assert convolved.min() >= 0.0
