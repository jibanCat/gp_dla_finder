"""Assemble the GP model and calculate null and one-absorber evidences.

The workflow first interpolates the trained model onto the spectrum, applies
the Lyman-series mean-flux suppression and absorption noise, and then evaluates
the null evidence. The one-absorber evidence is a quasi-Monte-Carlo average over
the absorber sample grid.

The arithmetic follows ``NullGP.get_interp``, ``NullGP.log_model_evidence`` and
the reference ``DLAGP`` sample-likelihood path. A few details that look
incidental affect reference fidelity:

* ``mu``, ``log_omega`` and each of the ``k`` columns of ``M`` are interpolated
  **separately** with 1-D linear interpolation, matching the reference's
  per-eigenvector loop;
* the absorption-noise term is multiplied by the *squared* mean-flux suppression,
  because the whole model is re-levelled to ``mu * a_lya`` rather than ``mu``;
* every per-sample log-likelihood carries ``- log(N)``, and the evidence adds
  ``+ log(N)`` back. The pair cancels analytically, but the order changes the
  floating-point result and is therefore preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from ..config import Config
from ..errors import NumericalError
from ..model import GPModel
from ..samples import AbsorberSampleGrid
from ..voigt import voigt_absorption
from .likelihood import effective_optical_depth, log_mvnpdf_low_rank
from .spectrum import PreparedSpectrum

__all__ = [
    "coarse_scan_size",
    "AssembledModel",
    "assemble_model",
    "null_log_evidence",
    "one_absorber_log_evidence",
]


@dataclass(frozen=True)
class AssembledModel:
    """The trained GP evaluated on one spectrum's pixels."""

    #: Mean flux, mean-flux-suppressed: ``mu * a_lya``.
    mean: np.ndarray
    #: Low-rank covariance factor, suppressed: ``M * a_lya``.
    factor: np.ndarray
    #: Absorption-noise variance, ``omega^2 * s(z)^2 * a_lya^2``.
    absorption_variance: np.ndarray
    #: Mean-flux suppression itself, retained for diagnostics.
    mean_flux_suppression: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.factor.shape[1])


def assemble_model(
    prepared: PreparedSpectrum,
    model: GPModel,
    config: Config,
    *,
    tau_0: float | None = None,
    beta: float | None = None,
) -> AssembledModel:
    """Interpolate the model and apply the forest terms.

    Parameters
    ----------
    tau_0, beta
        Mean-flux prior. Defaults to the configuration's values; the per-spectrum
        empirical-Bayes fit will pass its own.
    """
    tau_0 = config.prev_tau_0 if tau_0 is None else tau_0
    beta = config.prev_beta if beta is None else beta

    rest = prepared.rest_wavelength
    observed = prepared.wavelength

    grid = model.rest_wavelengths
    if rest.min() < grid[0] or rest.max() > grid[-1]:
        raise NumericalError(
            f"spectrum pixels span rest-frame [{rest.min():.3f}, {rest.max():.3f}] A, "
            f"outside the model grid [{grid[0]:.3f}, {grid[-1]:.3f}] A"
        )

    # --- stage 4: interpolate the trained model ------------------------------
    this_mu = interp1d(grid, model.mu)(rest)
    this_log_omega = interp1d(grid, model.log_omega)(rest)
    # Per-column, matching the reference's loop over eigenvectors.
    this_M = np.empty((model.rank, rest.shape[0]))
    for i, column in enumerate(model.M.T):
        this_M[i, :] = interp1d(grid, column)(rest)
    this_M = this_M.T

    this_omega2 = np.exp(2 * this_log_omega)

    # --- stage 5a: mean-flux suppression -------------------------------------
    total_optical_depth = effective_optical_depth(
        observed, beta, tau_0, prepared.z_qso, config.num_forest_lines
    )
    lya_absorption = np.exp(-np.sum(total_optical_depth, axis=1))

    this_mu = this_mu * lya_absorption
    this_M = this_M * lya_absorption[:, None]

    # --- stage 5b: absorption-noise scaling ----------------------------------
    # A second optical depth, with the model's own LEARNED (tau_0, beta), not the
    # mean-flux prior's. These are different quantities and the reference keeps
    # them separate.
    lya_optical_depth = effective_optical_depth(
        observed,
        np.exp(model.log_beta),
        np.exp(model.log_tau_0),
        prepared.z_qso,
        config.num_forest_lines,
    )
    scaling_factor = (
        1 - np.exp(-np.sum(lya_optical_depth, axis=1)) + np.exp(model.log_c_0)
    )
    this_omega2 = this_omega2 * scaling_factor**2
    # Re-level the noise to mu * a_lya, matching the mean.
    this_omega2 = this_omega2 * lya_absorption**2

    return AssembledModel(
        mean=this_mu,
        factor=this_M,
        absorption_variance=this_omega2,
        mean_flux_suppression=lya_absorption,
        diagnostics={
            "tau_0": float(tau_0),
            "beta": float(beta),
            "learned_tau_0": model.learned_tau_0,
            "learned_beta": model.learned_beta,
            "learned_c_0": model.learned_c_0,
            "num_forest_lines": config.num_forest_lines,
            "mean_flux_suppression_range": (
                float(lya_absorption.min()),
                float(lya_absorption.max()),
            ),
        },
    )


def null_log_evidence(prepared: PreparedSpectrum, assembled: AssembledModel) -> float:
    """Stage 6: the null-model log evidence."""
    total_variance = assembled.absorption_variance + prepared.noise_variance
    if not np.all(np.isfinite(total_variance)) or np.any(total_variance <= 0):
        raise NumericalError(
            "null-model noise variance is not finite and positive on every pixel"
        )
    try:
        return float(
            log_mvnpdf_low_rank(
                prepared.flux, assembled.mean, assembled.factor, total_variance
            )
        )
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            f"null-model covariance is not positive definite: {exc}"
        ) from exc


def absorber_search_window(
    prepared: PreparedSpectrum, config: Config
) -> tuple[float, float]:
    """The ``z_abs`` interval searched for this spectrum.

    Verbatim from the reference's ``Parameters.min_z_dla`` / ``max_z_dla``: the
    window is the part of the modelled region where a Lyman-alpha absorber could
    fall, inset from the quasar and from the Lyman limit by the configured
    velocities.
    """
    from ..config import LYA_WAVELENGTH, LYMAN_LIMIT

    window = prepared.window_wavelength
    z_qso = prepared.z_qso

    z_max = float(
        np.min(
            [
                (np.max(window) / LYA_WAVELENGTH - 1) - config.max_z_cut,
                z_qso - config.max_z_cut,
            ]
        )
    )
    z_min = float(
        np.max(
            [
                np.min(window) / LYA_WAVELENGTH - 1,
                LYMAN_LIMIT * (1 + z_qso) / LYA_WAVELENGTH - 1 + config.min_z_cut,
            ]
        )
    )
    return z_min, z_max


def _absorber_profile(
    prepared: PreparedSpectrum,
    config: Config,
    z_abs: float,
    nhi: float,
) -> np.ndarray:
    """Voigt absorption on the kept pixels, via the padded grid."""
    wavelengths = (
        prepared.padded_wavelength if config.broadening else prepared.window_wavelength
    )
    absorption = voigt_absorption(
        wavelengths,
        nhi=nhi,
        z_dla=z_abs,
        num_lines=config.num_lines,
        kernel=config.lsf_kernel,
        broadening=config.broadening,
        backend=config.voigt_backend,
    )
    return absorption[prepared.mask_in_window]


def coarse_scan_size(config: Config) -> int:
    """How many samples the FILTER coarse scan evaluates.

    ``max(num_samples // 20, filter_n_initial_floor)``, verbatim from the
    reference's ``parallel_log_model_evidences``. The floor of 5000 reproduces
    the historical hard-coded value, so at the 10k operating point the "20x
    reduction" is actually a factor of two, and only at 100k does it become 20x.
    """
    return min(
        config.num_samples,
        max(config.num_samples // 20, config.filter_n_initial_floor),
    )


def one_absorber_log_evidence(
    prepared: PreparedSpectrum,
    assembled: AssembledModel,
    grid: AbsorberSampleGrid,
    config: Config,
    *,
    return_samples: bool = False,
    mode: str | None = None,
):
    """Stage 7: the one-absorber log evidence, by QMC average over the grid.

    .. math::

        p(D \\mid 1) \\approx \\frac{1}{N} \\sum_i
            p(D \\mid z_i, N_{\\mathrm{HI},i})

    computed in log space with the reference's exact bookkeeping: each sample
    likelihood carries ``- log N``, and the estimator adds ``+ log N`` back after
    the log-mean-exp.

    Parameters
    ----------
    mode
        None takes the mode from :attr:`Config.evidence_mode`. Production
        presets use the full configured grid. Passing a mode explicitly
        overrides the configuration, as in the reference-parity tests.
        "exact" evaluates all num_samples grid points.
        "filter" evaluates only the first :func:`coarse_scan_size` of them. For
        the one-absorber evidence that is the whole of what the reference's
        FILTER=1 path does: its adaptive region-A machinery selects a
        high-likelihood subset for refinement, but the reference's "FILTER fix
        #5" then discards those refined samples when forming the 1-absorber
        evidence and uses the coarse scan alone. The refinement only reaches the
        multi-absorber evidences, so at k = 1 FILTER is not an adaptive
        approximation at all: it is the same estimator on a fixed prefix of the
        QMC sequence. Note the normalisation: the log(N) in the round trip stays
        the full sample count even when only a prefix is evaluated, exactly as
        the reference does it.

    Returns
    -------
    float or tuple
        The log evidence, or a pair of the log evidence and the per-sample log
        likelihoods when ``return_samples`` is set. In filter mode the
        un-evaluated tail of that array is NaN, so a caller can see what was and
        was not computed. The per-sample array is not retained by default.
    """
    mode = config.evidence_mode if mode is None else mode
    if mode not in ("exact", "filter"):
        raise ValueError(f"mode must be 'exact' or 'filter', got {mode!r}")

    n_samples = grid.num_samples
    if config.num_samples != n_samples:
        raise ValueError(
            f"config expects {config.num_samples} samples but the grid has "
            f"{n_samples}; named presets require an exact match"
        )

    n_evaluated = n_samples if mode == "exact" else coarse_scan_size(config)

    z_min, z_max = absorber_search_window(prepared, config)
    z_samples = grid.sample_redshifts(z_min, z_max)
    nhi_samples = grid.nhi_samples

    total_variance_base = prepared.noise_variance
    # A named compatibility behaviour, not a modelling choice: the reference
    # subtracts log(N) from every sample and adds it back after the log-mean-exp.
    # The pair cancels analytically and not in floating point. See
    # :mod:`gp_dla_finder.compat`.
    log_norm = (
        np.log(n_samples) if config.compatibility_profile.log_norm_round_trip else 0.0
    )

    sample_log_likelihoods = np.full(n_samples, np.nan)
    for i in range(n_evaluated):
        absorption = _absorber_profile(prepared, config, z_samples[i], nhi_samples[i])
        sample_log_likelihoods[i] = (
            log_mvnpdf_low_rank(
                prepared.flux,
                assembled.mean * absorption,
                assembled.factor * absorption[:, None],
                assembled.absorption_variance * absorption**2 + total_variance_base,
            )
            - log_norm
        )

    evaluated = sample_log_likelihoods[:n_evaluated]
    if np.all(np.isnan(evaluated)):
        raise NumericalError("every absorber sample produced a NaN log-likelihood")

    # log-mean-exp over what was evaluated, then undo the per-sample -log(N).
    max_log_likelihood = np.nanmax(evaluated)
    probabilities = np.exp(evaluated - max_log_likelihood)
    log_evidence = float(
        max_log_likelihood + np.log(np.nanmean(probabilities)) + log_norm
    )

    if return_samples:
        return log_evidence, sample_log_likelihoods
    return log_evidence
