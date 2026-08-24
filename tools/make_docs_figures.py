"""Generate the documentation figures from real package code.

Every figure here is *computed*, not drawn: the Voigt panels call
``voigt_absorption``, the spectrum panel imprints an absorber with the same
forward model the inference uses, and the landscape panel plots the actual
per-sample integrand the evidence path returns. An illustrative sketch that did
not match what the code does would be worse than no figure.

Two variants of each are written, ``-light`` and ``-dark``, and the pages select
between them with the theme's ``only-light`` / ``only-dark`` classes. The PI
reported the logo being unreadable in GitHub's dark theme; figures with baked-in
black axes would have the same problem.

    python tools/make_docs_figures.py           # write
    python tools/make_docs_figures.py --check   # verify committed files match
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "_static" / "figures"

# One palette per theme. Only these change between variants.
THEMES = {
    "light": {
        "fg": "#1b1f24",
        "muted": "#5b6570",
        "grid": "#d7dce1",
        "accent": ["#1f4e79", "#2e7d32", "#b8860b", "#a63603"],
        "band": "#1f4e79",
    },
    "dark": {
        "fg": "#e6edf3",
        "muted": "#9aa7b2",
        "grid": "#30363d",
        "accent": ["#6cb6ff", "#7ee787", "#e3b341", "#ff9b72"],
        "band": "#6cb6ff",
    },
    # A mid-tone rendering legible on BOTH canvases, for renderers that cannot
    # choose. GitHub's mobile app does not rewrite relative paths inside raw
    # HTML, so a <picture> block is a broken link there -- the pages use plain
    # Markdown image syntax, which every GitHub renderer rewrites, and that
    # syntax can only name one file. Same reasoning as the logo's universal
    # variant, which was measured rather than guessed.
    "universal": {
        "fg": "#7d8894",
        "muted": "#7d8894",
        "grid": "#8b939c",
        "accent": ["#4f86c6", "#4f9d69", "#b08a3e", "#c0693f"],
        "band": "#4f86c6",
    },
}


C_KMS = 299792.458


#: Say this wherever the generator's output appears. It is easy to mistake a
#: plausible-looking forest for a usable one.
FOREST_DISCLAIMER = (
    "This generator is for demonstrations, controlled injections and bounded "
    "validation tests. It is not a validated cosmological Lyman-alpha mock "
    "generator, it is not a substitute for quickquasars, fake_spectra or a "
    "survey mock, and it should not be used to produce a science mock "
    "catalogue or as evidence about catalogue performance."
)


def _fgpa_forest(
    *,
    observed,
    mean_tau,
    smoothing_sigma_kms,
    thermal_sigma_kms,
    oversampling,
    seed,
    field=None,
):
    """Transmission from FGPA on an oversampled velocity grid.

    Optical depth is built and thermally broadened on a grid ``oversampling``
    times finer than ``observed``, then rebinned by averaging TRANSMISSION --
    a detector integrates flux, not optical depth.

    Returns ``(transmission, diagnostics)``; the diagnostics record every
    quantity the construction depends on, so a reader can check the conversions
    rather than take them on trust.
    """
    kms_per_pixel = float(np.median(np.diff(observed)) / np.mean(observed) * C_KMS)
    fine_kms_per_pixel = kms_per_pixel / oversampling
    fine_size = observed.size * oversampling

    def _gaussian(sigma_kms):
        sigma_pixels = sigma_kms / fine_kms_per_pixel
        half = max(1, int(np.ceil(4 * sigma_pixels)))
        offsets = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
        return kernel / kernel.sum(), sigma_pixels, half

    smoothing_kernel, smoothing_pixels, half = _gaussian(smoothing_sigma_kms)
    thermal_kernel, thermal_pixels, thermal_half = _gaussian(thermal_sigma_kms)

    # Padding, and then the CENTRAL valid interval -- not the leading one.
    # `mode="same"` zero-pads both ends, so the first and last `half` pixels of
    # the result are contaminated. Taking [:fine_size] kept the contaminated
    # leading edge; taking the middle discards both.
    if field is None:
        rng = np.random.default_rng(seed)
        white = rng.normal(0.0, 1.0, fine_size + 2 * half)
        smoothed = np.convolve(white, smoothing_kernel, mode="same")
        smooth = smoothed[half : half + fine_size]
        smooth /= smooth.std()
    else:
        # A caller-supplied field, so a convergence test can hold the underlying
        # realisation fixed while changing only the oversampling. Drawing a new
        # field per factor measures the draw, not the numerics.
        smooth = np.asarray(field, dtype=float)
        if smooth.size != fine_size:
            raise ValueError(
                f"field has {smooth.size} points, expected {fine_size} for "
                f"oversampling={oversampling}"
            )

    # Lognormal density with <1 + delta> = 1, then FGPA: tau ~ (1 + delta)^1.6.
    sigma = 1.1
    shape = np.exp(sigma * smooth - 0.5 * sigma**2) ** 1.6

    # Thermal broadening on TAU, on the fine grid where it is resolved.
    #
    # Edge convention stated rather than inherited: pad by EDGE REPLICATION.
    # Zero-padding here would pull the first and last pixels of tau towards
    # zero optical depth -- i.e. invent transparency at the panel edges.
    padded = np.concatenate(
        [
            np.full(thermal_half, shape[0]),
            shape,
            np.full(thermal_half, shape[-1]),
        ]
    )
    shape = np.convolve(padded, thermal_kernel, mode="same")[
        thermal_half : thermal_half + fine_size
    ]

    fine_mean_tau = np.repeat(np.asarray(mean_tau), oversampling)
    absorbing = fine_mean_tau > 0
    fine = np.ones(fine_size)

    index = np.flatnonzero(absorbing)
    block = 64 * oversampling
    for lo in range(0, index.size, block):
        chunk = index[lo : lo + block]
        want = float(np.exp(-np.mean(fine_mean_tau[chunk])))
        a_lo, a_hi = 1e-8, 500.0
        for _ in range(60):
            a_mid = 0.5 * (a_lo + a_hi)
            if float(np.mean(np.exp(-a_mid * shape[chunk]))) > want:
                a_lo = a_mid
            else:
                a_hi = a_mid
        fine[chunk] = np.exp(-0.5 * (a_lo + a_hi) * shape[chunk])

    transmission = fine.reshape(observed.size, oversampling).mean(axis=1)

    return transmission, {
        "kms_per_pixel": kms_per_pixel,
        "fine_kms_per_pixel": fine_kms_per_pixel,
        "oversampling": float(oversampling),
        "smoothing_sigma_kms": smoothing_sigma_kms,
        "smoothing_sigma_fine_pixels": smoothing_pixels,
        "thermal_sigma_kms": thermal_sigma_kms,
        "thermal_sigma_fine_pixels": thermal_pixels,
        "thermal_kernel_half_width_fine_pixels": float(thermal_half),
        "rebinning": "mean transmission per display pixel",
        "seed": float(seed),
    }


def _absorption(wave, *, nhi, z_dla):
    """``voigt_absorption`` on a padded grid, returned on ``wave``.

    The convolution trims the kernel half-width from each end, so an unpadded
    call returns a shorter array than it was given.
    """
    from gp_dla_finder.voigt import (
        PRODUCTION_KERNEL,
        kernel_half_width,
        voigt_absorption,
    )

    half = kernel_half_width(PRODUCTION_KERNEL)
    step = float(wave[1] - wave[0])
    padded = np.concatenate(
        [
            wave[0] - step * np.arange(half, 0, -1),
            wave,
            wave[-1] + step * np.arange(1, half + 1),
        ]
    )
    profile = voigt_absorption(
        padded, nhi=nhi, z_dla=z_dla, num_lines=3, kernel=PRODUCTION_KERNEL
    )
    assert profile.shape == wave.shape, (profile.shape, wave.shape)
    return profile


def _style(fig, axes, theme):
    """Transparent background, themed ink. No opaque canvas in either variant."""
    fig.patch.set_alpha(0.0)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("none")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(theme["muted"])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=theme["muted"], labelcolor=theme["fg"])
        ax.xaxis.label.set_color(theme["fg"])
        ax.yaxis.label.set_color(theme["fg"])
        ax.title.set_color(theme["fg"])
        ax.grid(True, color=theme["grid"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_alpha(0.0)
            legend.get_frame().set_edgecolor(theme["muted"])
            for text in legend.get_texts():
                text.set_color(theme["fg"])


def figure_voigt(theme):
    """The damping wings, at four column densities."""
    import matplotlib.pyplot as plt

    z_abs = 2.5
    wave = np.linspace(4180.0, 4330.0, 3000)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    profiles = {}
    for colour, log_nhi in zip(theme["accent"], (19.0, 20.0, 20.3, 21.5), strict=True):
        profile = _absorption(wave, nhi=10.0**log_nhi, z_dla=z_abs)
        profiles[str(log_nhi)] = profile
        ax.plot(
            wave,
            profile,
            color=colour,
            linewidth=1.6,
            label=rf"$\log_{{10}} N_{{\rm HI}} = {log_nhi}$",
        )
    ax.axvline(1215.6701 * (1 + z_abs), color=theme["muted"], linewidth=0.8, ls=":")
    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel("transmission")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Voigt absorption at $z_{{\\rm abs}} = {z_abs}$ (DESI kernel)")
    ax.legend(frameon=True, fontsize=8)
    _style(fig, ax, theme)
    fig.tight_layout()
    return fig, {"wave": wave, "profiles": profiles}


def figure_spectrum(theme):
    """A realistic quasar spectrum: the GP mean, a forest, and an injected HCD.

    The continuum is the **trained GP model's own mean** -- the learned quasar
    shape, with the real emission lines in it -- not a sine wave. An earlier
    version of this figure used sin/cos, which looked nothing like a quasar and
    taught the reader the wrong thing about what the finder sees.

    On top of it: Lyman-alpha forest absorption blueward of Lya, and one damped
    absorber imprinted with the same forward model the inference uses.

    The forest's MEAN optical depth is the package's own
    ``effective_optical_depth`` with the mean-flux prior the inference applies,
    so the figure is self-consistent with what the code actually computes: mean
    transmission 0.89 at z = 1.9 and 0.80 at z = 2.5, against ~0.88 and ~0.74
    measured. Close enough to look like a quasar, and not silently substituted
    from the literature.
    """
    import matplotlib.pyplot as plt

    from gp_dla_finder import load_model

    model = load_model()
    z_qso, z_abs, log_nhi = 2.5, 2.15, 20.6

    rest = model.rest_wavelengths
    keep = (rest >= 1000.0) & (rest <= 1400.0)
    rest = rest[keep]
    continuum = np.asarray(model.mu)[keep]
    observed = rest * (1.0 + z_qso)

    LYA = 1215.6701

    # Lyman-alpha forest: FGPA on an OVERSAMPLED velocity grid, thermally
    # broadened there, then rebinned to the display grid.
    #
    # Why oversample. Thermal broadening at these temperatures is narrower than
    # one display pixel, and an earlier version skipped the convolution when
    # that happened. Sub-pixel broadening is not zero broadening -- it still
    # smooths structure below the pixel scale, and that structure then rebins
    # into what you see. Building tau on a finer grid and rebinning is the
    # correct treatment, and it costs almost nothing at this size.
    #
    # Physical inputs, in velocity first and converted explicitly:
    #
    #   SMOOTHING_SIGMA_KMS -- the Gaussian sigma applied to the underlying
    #       Gaussian field. This is the generator INPUT and is NOT the same
    #       number as the measured 1/e autocorrelation length of the resulting
    #       transmission: the FGPA exponent and the lognormal transform both
    #       change that. 50 km/s is an ILLUSTRATIVE choice in the range where
    #       forest structure lives. The 1D flux-power measurements of McDonald
    #       et al. 2006 (ApJS 163, 80) and Palanque-Delabrouille et al. 2013
    #       (A&A 559, A85) constrain k ~ 0.01-0.02 s/km, but turning a power
    #       spectrum turnover into a correlation length depends on reading 1/k
    #       or 2*pi/k, so this is not quoted as a measured value.
    #
    #   TEMPERATURE_K -- IGM temperature at mean density. Becker et al. 2011
    #       (MNRAS 410, 1096) put T0 near 1-2 x 1e4 K at these redshifts.
    #       b = sqrt(2 k T / m_H), and the Gaussian sigma in velocity is
    #       b / sqrt(2).
    #
    # Instrumental broadening is a SEPARATE operation and is not applied here.
    # Peculiar velocities and redshift-space distortion are out of scope: doing
    # them consistently needs a velocity field from this same density field.
    SMOOTHING_SIGMA_KMS = 50.0
    TEMPERATURE_K = 2.0e4
    OVERSAMPLING = 8

    DOPPLER_B_KMS = float(np.sqrt(2 * 1.380649e-23 * TEMPERATURE_K / 1.6735e-27) / 1e3)
    THERMAL_SIGMA_KMS = DOPPLER_B_KMS / np.sqrt(2.0)

    from gp_dla_finder.config import Config
    from gp_dla_finder.gp.evidence import effective_optical_depth

    config = Config.desi_y3(enable_tau_eb=False)
    mean_tau = np.sum(
        effective_optical_depth(
            observed,
            config.prev_beta,
            config.prev_tau_0,
            z_qso,
            config.num_forest_lines,
        ),
        axis=1,
    )

    forest, forest_diagnostics = _fgpa_forest(
        observed=observed,
        mean_tau=mean_tau,
        smoothing_sigma_kms=SMOOTHING_SIGMA_KMS,
        thermal_sigma_kms=THERMAL_SIGMA_KMS,
        oversampling=OVERSAMPLING,
        seed=20260820,
    )
    rng = np.random.default_rng(20260820)

    absorbed = _absorption(observed, nhi=10.0**log_nhi, z_dla=z_abs)

    # Two different things, and the figure has to keep them apart:
    #
    #   DATA  = continuum x stochastic forest x DLA + noise   (what you observe)
    #   MODEL = continuum x MEAN mean-flux x DLA              (what the GP fits)
    #
    # The GP null model carries the MEAN suppression, not individual forest
    # lines -- to it the lines are noise, absorbed by the omega^2 term. Drawing
    # the model with the stochastic forest in it would claim the finder knows
    # where every forest line is, which is exactly what it does not do.
    mean_flux = np.exp(-mean_tau)
    data = continuum * forest * absorbed
    model_null = continuum * mean_flux
    model_dla = model_null * absorbed
    noise = rng.normal(0.0, 0.10 * np.median(continuum), rest.size)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(
        observed,
        data + noise,
        color=theme["accent"][0],
        linewidth=0.7,
        alpha=0.8,
        label="data: continuum × forest × DLA + noise",
    )
    ax.plot(
        observed,
        continuum,
        color=theme["muted"],
        linewidth=1.1,
        ls="--",
        label="GP mean, unsuppressed (continuum)",
    )
    ax.plot(
        observed,
        model_null,
        color=theme["accent"][1],
        linewidth=1.3,
        label="GP null model (continuum × mean flux)",
    )
    ax.plot(
        observed,
        model_dla,
        color=theme["accent"][3],
        linewidth=1.6,
        label=(
            rf"GP null + DLA, $\log_{{10}} N_{{\rm HI}} = {log_nhi}$"
            rf" at $z = {z_abs}$"
        ),
    )

    # Name the emission lines the GP mean actually contains, so a reader can see
    # this is a quasar and not a curve.
    for label, line in (
        (r"Ly$\beta$", 1025.72),
        (r"Ly$\alpha$", LYA),
        ("N V", 1240.81),
        ("Si IV", 1399.8),
    ):
        x = line * (1.0 + z_qso)
        if observed[0] < x < observed[-1]:
            ax.axvline(x, color=theme["muted"], linewidth=0.6, ls=":", alpha=0.7)
            ax.annotate(
                label,
                xy=(x, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=theme["muted"],
                va="bottom",
            )

    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel("flux (model units)")
    ax.set_title(rf"What the finder sees ($z_{{\rm QSO}} = {z_qso}$)")
    ax.legend(frameon=True, fontsize=7.5, loc="upper left", framealpha=0.85)
    _style(fig, ax, theme)
    fig.tight_layout()
    return fig, {
        "observed": observed,
        "continuum": continuum,
        "forest": forest,
        "model_dla": model_dla,
        "doppler_b_kms": np.array([DOPPLER_B_KMS]),
        "temperature_k": np.array([TEMPERATURE_K]),
        **{
            k: np.array([v])
            for k, v in forest_diagnostics.items()
            if not isinstance(v, str)
        },
    }


def figure_landscape(theme):
    """The real per-sample integrand, and the grid point the finder reports."""
    import matplotlib.pyplot as plt

    from gp_dla_finder import Config, load_model, load_sample_grid
    from gp_dla_finder.gp.evidence import (
        absorber_search_window,
        assemble_model,
        one_absorber_log_evidence,
    )
    from gp_dla_finder.gp.spectrum import Spectrum, prepare_spectrum

    rng = np.random.default_rng(20260820)
    wave = np.arange(3600.0, 5600.0, 0.8)
    flux = (1.0 + 0.3 * np.sin(wave / 180.0)) * _absorption(
        wave, nhi=10**20.6, z_dla=2.35
    ) + rng.normal(0.0, 0.2, wave.size)
    spectrum = Spectrum(
        wavelength=wave,
        flux=flux,
        ivar=np.full_like(wave, 25.0),
        z_qso=2.6,
        mask=np.zeros_like(wave, dtype=bool),
    )

    config = Config.desi_y3_fast()
    model = load_model()
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(spectrum, model, config)
    assembled = assemble_model(prepared, model, config)
    _, samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="exact", return_samples=True
    )

    z_min, z_max = absorber_search_window(prepared, config)
    z_samples = grid.sample_redshifts(z_min, z_max)
    log_nhi = np.log10(grid.nhi_samples)
    finite = np.isfinite(samples)
    best = int(np.nanargmax(np.where(finite, samples, -np.inf)))

    # A scatter of all 10,000 samples is unreadable: the integrand is so
    # concentrated that every competitive point collapses into one colour, and
    # the picture is a uniform rectangle. The profile below carries the same
    # information legibly -- and needs no rasterised layer, so the file stays
    # small.
    #
    # For each bin, the LARGEST log integrand contribution in it: the height of
    # the ridge, not an average over samples that mostly sit far off the peak.
    def profile(coordinate, bins):
        index = np.digitize(coordinate[finite], bins) - 1
        out = np.full(len(bins) - 1, np.nan)
        for b in range(len(bins) - 1):
            in_bin = index == b
            if np.any(in_bin):
                out[b] = np.max(samples[finite][in_bin])
        return out

    z_bins = np.linspace(z_samples.min(), z_samples.max(), 90)
    n_bins = np.linspace(log_nhi.min(), log_nhi.max(), 70)
    z_profile = profile(z_samples, z_bins)
    n_profile = profile(log_nhi, n_bins)
    peak = float(np.nanmax(samples[finite]))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, bins, values, injected, found, label in (
        (axes[0], z_bins, z_profile, 2.35, z_samples[best], r"$z_{\rm abs}$"),
        (axes[1], n_bins, n_profile, 20.6, log_nhi[best], r"$\log_{10} N_{\rm HI}$"),
    ):
        centres = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(centres, values - peak, color=theme["accent"][0], linewidth=1.6)
        ax.axvline(
            injected, color=theme["fg"], linewidth=1.2, ls="--", label="injected"
        )
        ax.axvline(
            found,
            color=theme["accent"][3],
            linewidth=1.2,
            ls=":",
            label="best grid point",
        )
        ax.set_xlabel(label)
        ax.set_ylim(-60, 4)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("log integrand,\nrelative to peak")
    fig.suptitle(
        "Where the one-absorber model fits (10,000-sample grid)",
        color=theme["fg"],
        fontsize=11,
    )
    _style(fig, axes, theme)
    for ax in axes:
        for text in ax.get_legend().get_texts():
            text.set_color(theme["fg"])
    fig.tight_layout()
    return fig, {
        "z_profile": z_profile,
        "n_profile": n_profile,
        "peak": np.array([peak]),
        "best": np.array([z_samples[best], log_nhi[best]]),
    }


#: Dynamic range of the posterior-weight colour scale, in nats below the peak.
#: Chosen, not tuned: the integrand spans thousands of nats, so an unclipped
#: scale puts every competitive sample in one colour and the panel becomes a
#: uniform rectangle -- the first draft of this figure did exactly that.
#: exp(-25) is about 1e-11 of the peak weight, far below anything that
#: contributes to the evidence, so nothing informative is clipped.
POSTERIOR_FLOOR = -25.0

#: (log10 N_HI, ivar, label) for the two panels. A strong absorber in a clean
#: spectrum and a weak one in a noisy spectrum, because the difference between
#: them is the point of the figure.
POSTERIOR_CASES = (
    (20.6, 25.0, "strong absorber, high S/N"),
    (20.1, 4.0, "weak absorber, low S/N"),
)


def _one_absorber_samples(log_nhi_injected, ivar, *, z_injected=2.35):
    """Per-sample log integrand for one generated spectrum, on the real path.

    Returns the sample coordinates and the log integrand, which is what the
    conditional posterior is made of -- see :func:`figure_conditional_posterior`
    for why the two are the same thing up to a constant.
    """
    from gp_dla_finder import Config, load_model, load_sample_grid
    from gp_dla_finder.gp.evidence import (
        absorber_search_window,
        assemble_model,
        one_absorber_log_evidence,
    )
    from gp_dla_finder.gp.spectrum import Spectrum, prepare_spectrum

    rng = np.random.default_rng(20260820)
    wave = np.arange(3600.0, 5600.0, 0.8)
    flux = (1.0 + 0.3 * np.sin(wave / 180.0)) * _absorption(
        wave, nhi=10**log_nhi_injected, z_dla=z_injected
    ) + rng.normal(0.0, 1.0 / np.sqrt(ivar), wave.size)
    spectrum = Spectrum(
        wavelength=wave,
        flux=flux,
        ivar=np.full_like(wave, ivar),
        z_qso=2.6,
        mask=np.zeros_like(wave, dtype=bool),
    )

    config = Config.desi_y3_fast()
    model = load_model()
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(spectrum, model, config)
    assembled = assemble_model(prepared, model, config)
    _, samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="exact", return_samples=True
    )
    z_min, z_max = absorber_search_window(prepared, config)
    return grid.sample_redshifts(z_min, z_max), np.log10(grid.nhi_samples), samples


def _effective_sample_size(log_weights):
    """Kish ESS of the self-normalised weights.

    The number that says how many of the 10,000 samples the integral actually
    rests on. It belongs on the figure because it is the single most useful
    thing the picture has to say.
    """
    finite = log_weights[np.isfinite(log_weights)]
    weights = np.exp(finite - finite.max())
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights**2))


def figure_conditional_posterior(theme):
    """p(z_abs, log10 N_HI | D, M1), as the QMC samples that represent it.

    **What is plotted.** The QMC grid draws its samples from the prior
    pi(theta), so the per-sample log integrand is log L(theta_i) up to a
    constant, and the self-normalised weight

        w_i = exp(logL_i) / sum_j exp(logL_j)

    is the conditional posterior evaluated at that sample. The colour is
    log(w_i / w_max) in nats: zero at the best sample, negative everywhere
    else. That is a monotone relabelling of the posterior weight, and it avoids
    quoting a normalisation that depends on the grid size.

    **No smoothing, no density estimate, no contours.** These are the samples
    themselves. The faint grey layer is every sample -- the prior grid -- and
    the coloured layer is the samples within POSTERIOR_FLOOR nats of the peak.

    POSTERIOR_FLOOR is a **display threshold, not an inference cut.** Every
    sample contributes to the evidence exactly as it did before the figure
    existed; the floor only decides which ones get a colour.

    **Why two panels.** A strong absorber in a clean spectrum concentrates the
    posterior onto a handful of the 10,000 samples: the effective sample size
    is about 1. That is a real property of the inference, not a plotting
    artefact, and it is why the package reports a grid point rather than a
    posterior mean. The second panel is the same machinery on a weak absorber
    in a noisy spectrum, where the posterior is genuinely broad.
    """
    import matplotlib.pyplot as plt

    z_injected = 2.35
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    payload = {}
    points = None

    for ax, (nhi_injected, ivar, label) in zip(axes, POSTERIOR_CASES, strict=True):
        z_samples, log_nhi, samples = _one_absorber_samples(nhi_injected, ivar)
        finite = np.isfinite(samples)
        peak = float(np.max(samples[finite]))
        best = int(np.nanargmax(np.where(finite, samples, -np.inf)))
        weight = samples - peak
        ess = _effective_sample_size(samples)

        # Every sample, faintly: the grid the search actually covered. Without
        # it the coloured knot would look like the whole story.
        # theme["muted"] rather than theme["grid"]: the grid colour is tuned to
        # sit just off the background, which makes it a fine gridline and an
        # invisible data layer on the dark canvas.
        ax.scatter(
            z_samples,
            log_nhi,
            s=1.5,
            color=theme["muted"],
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        # The samples that carry the posterior, drawn faintest-first so the
        # peak is not buried under its own neighbours.
        shown = finite & (weight >= POSTERIOR_FLOOR)
        order = np.argsort(weight[shown])
        points = ax.scatter(
            z_samples[shown][order],
            log_nhi[shown][order],
            c=weight[shown][order],
            s=26,
            cmap="viridis",
            vmin=POSTERIOR_FLOOR,
            vmax=0.0,
            linewidths=0,
            zorder=3,
        )

        # Two different marks for two different things: where the absorber was
        # injected, and a grid point the search evaluated. They do not
        # coincide, and the figure should not suggest they do.
        ax.plot(
            z_injected,
            nhi_injected,
            marker="+",
            markersize=13,
            markeredgewidth=2.0,
            color=theme["accent"][3],
            linestyle="none",
            zorder=4,
            label="injected truth",
        )
        ax.plot(
            z_samples[best],
            log_nhi[best],
            marker="o",
            markersize=12,
            markerfacecolor="none",
            markeredgewidth=1.8,
            color=theme["fg"],
            linestyle="none",
            zorder=5,
            label="best evaluated grid point",
        )

        ax.set_title(label, color=theme["fg"], fontsize=9.5)
        ax.set_xlabel(r"$z_{\rm abs}$")
        ax.text(
            0.03,
            0.96,
            f"{int(shown.sum())} of {samples.size} samples shown\n"
            f"effective sample size {ess:.0f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color=theme["fg"],
        )
        payload[label.split(",")[0].replace(" ", "_")] = {
            "weight": weight,
            "peak": np.array([peak]),
            "ess": np.array([ess]),
            "best": np.array([z_samples[best], log_nhi[best]]),
            "n_shown": np.array([float(shown.sum())]),
        }

    axes[0].set_ylabel(r"$\log_{10}\,(N_{\rm HI}\,/\,{\rm cm^{-2}})$")

    bar = fig.colorbar(points, ax=axes, pad=0.015, fraction=0.045)
    bar.set_label(
        "log posterior weight, relative to peak (nats)", color=theme["fg"], fontsize=9
    )
    bar.ax.tick_params(colors=theme["muted"], labelcolor=theme["fg"], labelsize=8)
    bar.outline.set_edgecolor(theme["muted"])

    fig.suptitle(
        r"$p(z_{\rm abs},\ \log_{10} N_{\rm HI}\ |\ D,\ M_1)$"
        "   -- 10,000 QMC samples, no smoothing",
        color=theme["fg"],
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.13),
    )
    for text in legend.get_texts():
        text.set_color(theme["fg"])
    _style(fig, axes, theme)
    return fig, payload


def figure_column_density_regimes(theme):
    """The PW14 CDDF with the conventional absorber regimes marked."""
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "tools"))
    from build_sample_grid import f_pw14

    log_nhi = np.linspace(17.2, 23.0, 600)
    log_cddf = np.log10(f_pw14(log_nhi))

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for lo, hi, label, colour in (
        (17.2, 19.0, "LLS", theme["accent"][0]),
        (19.0, 20.3, "sub-DLA", theme["accent"][1]),
        (20.3, 23.0, "DLA", theme["accent"][3]),
    ):
        ax.axvspan(lo, hi, color=colour, alpha=0.10, label=label)

    measured = log_nhi <= 22.0
    ax.plot(
        log_nhi[measured],
        log_cddf[measured],
        color=theme["fg"],
        linewidth=1.8,
        label="PW14 spline",
    )
    ax.plot(
        log_nhi[~measured],
        log_cddf[~measured],
        color=theme["fg"],
        linewidth=1.4,
        linestyle=":",
        label="builder endpoint continuation",
    )
    for boundary in (19.0, 20.3, 22.5):
        ax.axvline(boundary, color=theme["muted"], linewidth=0.8, linestyle="--")
    ax.text(
        22.5,
        ax.get_ylim()[0] if ax.get_ylim()[0] else -27,
        " packaged upper limit",
        color=theme["muted"],
        fontsize=7.5,
        rotation=90,
        va="bottom",
        ha="right",
    )
    ax.set_xlabel(r"$\log_{10}\,(N_{\rm HI}\,/\,{\rm cm}^{-2})$")
    ax.set_ylabel(r"$\log_{10} f(N_{\rm HI},X)$")
    ax.set_title("Column-density distribution and absorber regimes")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    _style(fig, ax, theme)
    fig.tight_layout()
    return fig, {"log_nhi": log_nhi, "log_cddf": log_cddf}


def figure_absorber_existence_prior(theme):
    """The catalog-derived prior probability of at least one absorber."""
    import matplotlib.pyplot as plt

    from gp_dla_finder import Config, load_prior

    prior = load_prior()
    increase = Config.desi_y3_fast().prior_z_qso_increase
    lo, hi = prior.z_qso_range
    z_qsos = np.linspace(lo + 0.01, min(hi, 4.5), 220)
    p_one = np.array([prior.absorber_fraction(z, increase) for z in z_qsos])

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(z_qsos, p_one, color=theme["accent"][0], linewidth=1.9)
    ax.fill_between(z_qsos, 0.0, p_one, color=theme["accent"][0], alpha=0.12)
    ax.set_xlabel(r"quasar redshift $z_{\rm QSO}$")
    ax.set_ylabel(r"$p(\mathcal{M}_{k\geq 1}\mid z_{\rm QSO})$")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Absorber-existence prior: {prior.name}")
    _style(fig, ax, theme)
    fig.tight_layout()
    return fig, {"z_qsos": z_qsos, "p_one": p_one}


def figure_priors(theme):
    """The two priors a custom configuration is most likely to want to change.

    Left: the absorber-existence prior, P(at least one absorber | z_QSO), read
    off the packaged catalogue-derived prior. It is what turns a Bayes factor
    into a posterior probability, and it is a function of quasar redshift --
    a fact that is easy to forget when quoting one detection threshold for a
    whole survey.

    Right: the column-density prior, as it is actually represented -- the
    log10 N_HI coordinates of the packaged QMC grids. This is the point the
    customisation page makes concrete: the N_HI prior is not a Config number
    that gets consulted at run time, it is baked into the sample grid, and
    changing it means building a new grid.
    """
    import matplotlib.pyplot as plt

    from gp_dla_finder import (
        Config,
        available_sample_grids,
        load_prior,
        load_sample_grid,
    )

    prior = load_prior()
    config = Config.desi_y3_fast()
    increase = config.prior_z_qso_increase

    lo, hi = prior.z_qso_range
    z_qsos = np.linspace(lo + 0.01, min(hi, 4.5), 220)
    p_one = np.array([prior.absorber_fraction(z, increase) for z in z_qsos])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))

    axes[0].plot(z_qsos, p_one, color=theme["accent"][0], linewidth=1.8)
    axes[0].set_xlabel(r"$z_{\rm QSO}$")
    axes[0].set_ylabel(r"$P(\geq 1$ absorber$)$")
    axes[0].set_title(
        f"absorber-existence prior\n{prior.name}", color=theme["fg"], fontsize=9.5
    )

    # Every packaged grid, so the effect of the sample budget on how finely the
    # prior is represented is visible rather than asserted.
    grids = sorted(
        available_sample_grids(), key=lambda n: load_sample_grid(n).num_samples
    )
    for index, name in enumerate(grids):
        grid = load_sample_grid(name)
        axes[1].hist(
            grid.log_nhi_samples,
            bins=60,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=theme["accent"][index % len(theme["accent"])],
            label=f"{grid.num_samples:,} samples",
        )
    # A rebuilt grid, generated here exactly as tools/build_sample_grid.py does,
    # so the page's claim that a custom grid is a DIFFERENT prior is visible
    # rather than asserted. Seeded, so the figure is reproducible.
    sys.path.insert(0, str(ROOT / "tools"))
    from build_sample_grid import generate

    custom = generate(
        num_samples=20_000, log_nhi_range=(20.3, 22.5), alpha=0.97, seed=20260822
    )
    axes[1].hist(
        custom["log_nhi_samples"],
        bins=60,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=theme["accent"][3],
        label="custom grid: 20.3-22.5",
    )
    axes[1].axvline(
        20.3,
        color=theme["fg"],
        linewidth=1.1,
        ls="--",
        label="DLA threshold",
    )
    axes[1].set_xlabel(r"$\log_{10}\,(N_{\rm HI}\,/\,{\rm cm^{-2}})$")
    axes[1].set_ylabel("prior density")
    axes[1].set_yscale("log")
    axes[1].set_title(
        "column-density prior, as sampled\nbaked into the grid, not read from Config",
        color=theme["fg"],
        fontsize=9.5,
    )
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")

    _style(fig, axes, theme)
    fig.tight_layout()
    return fig, {
        "z_qsos": z_qsos,
        "p_one": p_one,
        "grid_sizes": np.array([float(load_sample_grid(n).num_samples) for n in grids]),
        "custom_log_nhi": np.asarray(custom["log_nhi_samples"], dtype=float),
    }


def figure_filter_prefix(theme):
    """What FILTER evaluates, and what it skips."""
    import matplotlib.pyplot as plt

    from gp_dla_finder import Config
    from gp_dla_finder.gp.evidence import coarse_scan_size

    budgets = np.array([10_000, 50_000, 100_000])
    grids = {
        10_000: "pw14_172_225_10000",
        50_000: "pw14_172_225_50000",
        100_000: "pw14_172_225_100000",
    }
    evaluated = np.array(
        [
            coarse_scan_size(
                Config.desi_y3().replace(num_samples=int(n), sample_grid=grids[int(n)])
            )
            for n in budgets
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(budgets))
    ax.bar(
        x - 0.19, budgets, width=0.38, color=theme["accent"][0], label="configured grid"
    )
    ax.bar(
        x + 0.19,
        evaluated,
        width=0.38,
        color=theme["accent"][2],
        label="evaluated by FILTER",
    )
    for i, (total, seen) in enumerate(zip(budgets, evaluated, strict=True)):
        ax.text(
            i + 0.19,
            seen * 1.15,
            f"{total // seen}×",
            ha="center",
            fontsize=9,
            color=theme["fg"],
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{n:,}" for n in budgets])
    ax.set_xlabel("configured sample budget")
    ax.set_ylabel("samples (log scale)")
    ax.set_title("FILTER evaluates a fixed 5,000-sample prefix")
    ax.legend(frameon=True, fontsize=8)
    _style(fig, ax, theme)
    fig.tight_layout()
    return fig, {
        "budgets": budgets.astype(float),
        "evaluated": evaluated.astype(float),
    }


FIGURES = {
    "voigt-profiles": figure_voigt,
    "spectrum-with-absorber": figure_spectrum,
    "column-density-regimes": figure_column_density_regimes,
    "evidence-landscape": figure_landscape,
    "conditional-posterior": figure_conditional_posterior,
    "absorber-existence-prior": figure_absorber_existence_prior,
    "priors": figure_priors,
    "filter-prefix": figure_filter_prefix,
}


DATA_FILE = OUT / "figure-data.json"

#: Tolerance for the data comparison. Not zero: the evidence path's numbers
#: differ in the last bits between platforms -- this project already measured
#: scipy/libcerf agreement as platform-dependent -- so a bit-exact check would
#: fail for a reason that says nothing about the figures.
RTOL, ATOL = 1e-6, 1e-9


def _render(name, builder, theme_name, *, write=True):
    """Build one figure. With ``write=False`` nothing on disk is touched.

    A check that regenerates the files it is checking cannot detect a missing
    one -- it recreates it first -- and mutating the working tree is not
    something a ``--check`` should do at all.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    # Deterministic ids, and no <dc:date> stamp.
    matplotlib.rcParams["svg.hashsalt"] = "gp_dla_finder"
    import matplotlib.pyplot as plt

    fig, data = builder(THEMES[theme_name])
    path = OUT / f"{name}-{theme_name}.svg"
    target = path if write else io.BytesIO()
    fig.savefig(
        target,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        dpi=110,
        metadata={"Date": None},
    )
    plt.close(fig)
    return path, data


#: How many evenly spaced points of each series to retain. Storing the full
#: arrays made the record 484 kB -- larger than the figures it guards. A
#: decimated summary plus the extremes still moves if the curve does.
DIGEST_POINTS = 41


def _digest(values):
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    index = np.linspace(0, array.size - 1, min(DIGEST_POINTS, array.size))
    return {
        "n": float(array.size),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "at": [float(array[int(round(i))]) for i in index],
    }


def _flatten(data, prefix=""):
    """``{name: digest}`` from a possibly nested builder payload."""
    flat = {}
    for key, value in data.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{label}."))
        else:
            flat[label] = _digest(value)
    return flat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare the PLOTTED DATA with the committed record",
    )
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    produced = {}
    for name, builder in FIGURES.items():
        for theme_name in THEMES:
            path, data = _render(name, builder, theme_name, write=not args.check)
            print(f"{'checked' if args.check else 'wrote':>8s}  {path.name}")
        # The data does not depend on the palette; record it once.
        produced[name] = _flatten(data)

    if not args.check:
        DATA_FILE.write_text(json.dumps(produced, indent=1, sort_keys=True) + "\n")
        print(f"{'wrote':>8s}  {DATA_FILE.name}")
        return 0

    # --- checking ---------------------------------------------------------
    #
    # The SVGs themselves are NOT byte-compared. matplotlib's SVG output differs
    # between platforms (font metrics, path rounding), and the first version of
    # this check failed on Linux against files generated on macOS while the
    # figures were perfectly correct. What is worth guarding is whether a figure
    # still shows what the code computes -- so the numbers are compared, with a
    # tolerance, and the rendering is only required to exist.
    problems = []
    for name in FIGURES:
        for theme_name in THEMES:
            target = OUT / f"{name}-{theme_name}.svg"
            if not target.is_file() or target.stat().st_size == 0:
                problems.append(f"{target.name}: missing or empty")

    if not DATA_FILE.is_file():
        problems.append(f"{DATA_FILE.name}: missing; run without --check")
    else:
        committed = json.loads(DATA_FILE.read_text())
        for name, series in produced.items():
            if name not in committed:
                problems.append(f"{name}: absent from {DATA_FILE.name}")
                continue
            for key, values in series.items():
                if key not in committed[name]:
                    problems.append(f"{name}.{key}: absent from the record")
                    continue
                was, now = committed[name][key], values
                for field in ("n", "min", "max", "mean", "at"):
                    old = np.asarray(was.get(field), dtype=float)
                    new = np.asarray(now[field], dtype=float)
                    if old.shape != new.shape:
                        problems.append(f"{name}.{key}.{field}: shape changed")
                    elif not np.allclose(
                        old, new, rtol=RTOL, atol=ATOL, equal_nan=True
                    ):
                        worst = float(np.nanmax(np.abs(old - new)))
                        problems.append(
                            f"{name}.{key}.{field}: differs, worst |delta| {worst:.3g}"
                        )

    if problems:
        print("\nSTALE FIGURES:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nRe-run: python tools/make_docs_figures.py",
            file=sys.stderr,
        )
        return 1
    print(f"{'checked':>8s}  {DATA_FILE.name} ({len(produced)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
