"""Experimental multi-absorber model evidence.

The reference's k-absorber integral is not a k-dimensional grid. The first
absorber uses the fixed QMC grid; absorbers 2..k are drawn from the previous
iteration's posterior by sequential importance resampling. Ported from
``DLAGPMAT.log_model_evidences`` (``desi_gpy_dla_detection``, ``desi_y3``).

The reference resamples with ``np.random.rand`` -- the unseeded
process-global generator -- so every k >= 2 evidence there is a different number
on every run. This package instead uses a local seeded stream, created fresh
inside each spectrum evaluation and never touching NumPy's global state.
``seed=0`` is deterministic by default.

The stream uses :class:`numpy.random.RandomState` rather than the newer
``Generator``. ``RandomState(s).rand(n)`` produces the
same floats as ``np.random.seed(s); np.random.rand(n)``, so a controlled-seed
test can compare this package against the reference **bitwise** without either
side depending on global state. That is a test of the algorithm under a supplied
seed. It is not a claim that a historical production run can be reconstructed --
that run's RNG state was never recorded and is gone.

The remaining steps follow the traced reference:

* the minimum-separation rule rejects the sample rather than applying a prior
  penalty -- samples whose sorted redshifts have any gap below
  ``min_z_separation`` become NaN and drop out of the average;
* the per-sample likelihood carries ``-log N``, and the evidence adds
  ``+log N - log N * (k - 1)`` -- the first undoing that shift, the second an
  Occam penalty per extra absorber;
* a NaN evidence stops the ladder, because the higher-order terms have
  underflowed rather than become small.

What this estimator does and does not establish
-----------------------------------------------

The intended quantity is

    Z_2 = INT L_2(t1, t2) pi(t1) pi(t2) dt1 dt2

but the second absorber is not drawn from the prior. It comes from the
one-absorber posterior,

    q(t2) ~ L_1(t2) pi(t2)

and a textbook importance-sampling derivation would then carry a pointwise
weight

    pi(t2) / q(t2) ~ 1 / L_1(t2)

The reference does not apply that factor, and neither does this port. What
it applies instead is the sequential resampling construction above plus the
inherited per-model ``-log N`` term.

This is therefore a **reference-compatible legacy evidence heuristic**, not a
proved unbiased estimator of the integral above. Unbiasedness and population
calibration have not been established. The narrower result we do have is
reference fidelity: under a controlled seed on a generated spectrum, the M1
and M2 evidences, resampled partner indices and best evaluated pair agree
bitwise with the reference. A separate comment in the reference about a mildly
biased, ESS-dependent ratio concerns the optional clustering pair-prior
correction, which production leaves off; it does not establish the properties
of the base M2 proposal density.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MAX_SEED",
    "RNG_ALGORITHM",
    "ModelLadder",
    "Resampler",
    "combine_log_evidences",
    "reject_close_pairs",
    "seeded_resampler",
    "two_absorber_log_evidence",
]

#: Draw ``size`` indices from normalised weights ``weights``. The reference's
#: implementation is ``np.searchsorted(np.cumsum(w), np.random.rand(size))``.
Resampler = Callable[[np.ndarray, int], np.ndarray]


#: Largest seed a legacy ``RandomState`` accepts.
MAX_SEED = 2**32 - 1

#: What produced the random stream. Recorded in provenance because the same seed
#: through a different generator gives different k >= 2 evidences.
RNG_ALGORITHM = "numpy.random.RandomState (MT19937)"


def seeded_resampler(seed: int | None) -> Resampler:
    """A local, reference-compatible resampling stream.

    ``seed=None`` is the explicit stochastic opt-in: the stream is drawn from
    OS entropy rather than the process-global state, so it is still not the
    reference's global RNG -- it just is not reproducible, which is the point of
    asking for it.

    Create one of these **inside each spectrum evaluation**. A single stream
    reused across spectra makes a result depend on how many spectra preceded it,
    which would make the result depend on scheduling.
    """
    if seed is not None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError(f"seed must be an integer or None, got {seed!r}")
        if not 0 <= int(seed) <= MAX_SEED:
            raise ValueError(f"seed must lie in [0, {MAX_SEED}], got {seed}")

    state = np.random.RandomState(seed)

    def resample(weights: np.ndarray, size: int) -> np.ndarray:
        values = np.asarray(weights, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("resampling weights must all be finite")
        if np.any(values < 0):
            raise ValueError("resampling weights must be non-negative")
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("resampling weights must have a positive finite sum")

        cumulative = np.cumsum(values / total)
        # Floating-point summation can leave the final entry a hair below 1.0,
        # and a draw above it would searchsorted past the end of the array. Pin
        # the top of the CDF rather than clipping the returned index, so the
        # last bin keeps exactly the probability it should have.
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, state.rand(size))

    return resample


def reject_close_pairs(redshifts: np.ndarray, min_separation: float) -> np.ndarray:
    """Which samples place two absorbers closer than ``min_separation``.

    ``redshifts`` is ``(k, n_samples)``. Returns a boolean mask over samples,
    True where the sample must be dropped.

    A rejection, not a penalty. The reference sets those samples' likelihoods to
    NaN so they leave the average entirely, which is a different model from
    down-weighting them -- and the difference matters for blended pairs, which
    are exactly the interesting case.
    """
    if redshifts.ndim != 2:
        raise ValueError(f"expected (k, n_samples), got shape {redshifts.shape}")
    if redshifts.shape[0] < 2:
        return np.zeros(redshifts.shape[1], dtype=bool)
    ordered = np.sort(redshifts, axis=0)
    return np.any(np.diff(ordered, axis=0) < min_separation, axis=0)


def combine_log_evidences(
    sample_log_likelihoods: np.ndarray, n_samples: int, n_absorbers: int
) -> float:
    """The k-absorber log evidence from its per-sample log likelihoods.

    ``sample_log_likelihoods`` already carries the reference's ``-log N``
    per-sample shift and has NaN where a sample was rejected.

    The arithmetic is the reference's, including both normalisation terms:

    * ``+log N`` undoes the per-sample shift, which would otherwise bias the
      Monte Carlo estimator by exactly that amount (the reference carries a
      2026-05-14 comment recording this as a bias fix);
    * ``-log N * (k - 1)`` is a further Occam penalty for each absorber beyond
      the first.

    Returns NaN when every sample was rejected or the values underflowed, which
    is the reference's signal to stop the ladder rather than an error.
    """
    values = np.asarray(sample_log_likelihoods, dtype=float)
    if not np.any(np.isfinite(values)):
        return float("nan")

    peak = float(np.nanmax(values))
    probabilities = np.exp(values - peak)
    mean = float(np.nanmean(probabilities))
    if not np.isfinite(mean) or mean <= 0:
        return float("nan")

    # The reference's loop variable is ZERO-BASED: `num_dlas` runs 0..max-1 and
    # `log_likelihoods_dla[num_dlas]` is the (num_dlas + 1)-absorber model. Its
    # penalty is therefore `-log N * num_dlas`, i.e. `-log N * (k - 1)` for the
    # k-absorber model -- one penalty per absorber BEYOND the first.
    #
    # This function takes the absorber COUNT, so it must subtract (k - 1). An
    # earlier version subtracted k, which charged M2 an extra log N: 9.21 nat at
    # the 10,000-sample operating point, easily enough to change which model is
    # preferred. It did.
    log_n = float(np.log(n_samples))
    return peak + float(np.log(mean)) + log_n - log_n * (n_absorbers - 1)


@dataclass(frozen=True)
class ModelLadder:
    """Evidences, priors and posteriors for M0..Mk, aligned by index.

    ``[0]`` is the null model and ``[j]`` is the model with ``j`` absorbers, in
    every tuple here. NaN in ``log_evidences`` marks a rung the ladder stopped
    at.

    Model selection is by the JOINT probability -- evidence times prior -- not
    by evidence alone. An evidence-only argmax silently assumes every model is
    equally likely a priori, which is exactly what the absorber-existence prior
    exists to say is false, and at these evidence separations it changes the
    answer.
    """

    log_evidences: tuple[float, ...]
    log_priors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.log_evidences) < 2:
            raise ValueError("a ladder needs at least the null and one-absorber models")
        object.__setattr__(self, "log_evidences", tuple(map(float, self.log_evidences)))
        if self.log_priors:
            if len(self.log_priors) != len(self.log_evidences):
                raise ValueError(
                    f"{len(self.log_priors)} priors for "
                    f"{len(self.log_evidences)} models"
                )
            object.__setattr__(self, "log_priors", tuple(map(float, self.log_priors)))

    @property
    def model_labels(self) -> tuple[str, ...]:
        """``("M0", "M1", ...)``, aligned with every other tuple here."""
        return tuple(f"M{i}" for i in range(len(self.log_evidences)))

    @property
    def log_joint(self) -> tuple[float, ...]:
        """``log Z_k + log P(M_k)``, the quantity selection is based on."""
        if not self.log_priors:
            raise ValueError(
                "this ladder carries no model priors, so no joint probability "
                "can be formed; selection by evidence alone assumes a uniform "
                "model prior, which the absorber-existence prior contradicts"
            )
        return tuple(
            evidence + prior
            for evidence, prior in zip(self.log_evidences, self.log_priors, strict=True)
        )

    @property
    def model_posteriors(self) -> tuple[float, ...]:
        """Normalised posteriors over the models that completed."""
        return self.posteriors(self.log_priors)

    @property
    def selected_model(self) -> int:
        """Index of the model with the highest JOINT probability."""
        joint = np.asarray(self.log_joint, dtype=float)
        if not np.any(np.isfinite(joint)):
            raise ValueError("no model produced a finite joint log probability")
        return int(np.nanargmax(np.where(np.isfinite(joint), joint, -np.inf)))

    @property
    def p_absorber(self) -> float:
        """Posterior probability of AT LEAST ONE absorber.

        Summed over every completed absorber model, which is what the phrase
        means once there is more than one of them. The old two-model number is
        not this quantity when M2 is active.
        """
        return float(sum(self.model_posteriors[1:]))

    @property
    def evaluated(self) -> int:
        """How many absorber models produced a finite evidence."""
        return sum(1 for value in self.log_evidences[1:] if np.isfinite(value))

    @property
    def complete(self) -> bool:
        """Whether every rung of the ladder produced a finite evidence.

        When this is False the posteriors below are **conditional on the models
        that completed**. They are not a posterior over M0..Mk, because one of
        those models was never measured.
        """
        return all(np.isfinite(value) for value in self.log_evidences)

    @property
    def stopped_at(self) -> int | None:
        """Index of the first unevaluated model, or ``None`` if complete."""
        for index, value in enumerate(self.log_evidences):
            if not np.isfinite(value):
                return index
        return None

    def posteriors(self, log_priors: tuple[float, ...]) -> tuple[float, ...]:
        """Model posteriors over the rungs that were evaluated.

        A rung the ladder stopped at contributes nothing rather than a zero
        likelihood -- it was never measured, and treating "not evaluated" as
        "evidence against" would inflate the models that were.

        **Read these together with :attr:`complete`.** A zero here can mean
        either "measured, and very unlikely" or "never measured at all", and
        those are different facts. When ``complete`` is False the numbers are a
        conditional posterior and must be reported as one.
        """
        if len(log_priors) != len(self.log_evidences):
            raise ValueError(
                f"{len(log_priors)} priors for {len(self.log_evidences)} models"
            )
        joint = np.asarray(self.log_evidences, dtype=float) + np.asarray(
            log_priors, dtype=float
        )
        finite = np.isfinite(joint)
        if not np.any(finite):
            raise ValueError("no model produced a finite joint log probability")

        shifted = np.where(finite, joint - np.max(joint[finite]), -np.inf)
        weights = np.where(finite, np.exp(shifted), 0.0)
        return tuple(float(value) for value in weights / weights.sum())


def two_absorber_log_evidence(
    prepared,
    assembled,
    grid,
    config,
    *,
    one_absorber_samples: np.ndarray,
    resampler: Resampler,
) -> tuple[float, np.ndarray]:
    """The two-absorber log evidence, by one SIR pass over the same grid.

    ``one_absorber_samples`` is the per-sample log-likelihood vector the
    one-absorber evidence already produced. Its exponentiated values are the
    proposal for where the *second* absorber goes -- which is the whole idea:
    the two-absorber integral reuses the one-absorber grid rather than forming a
    two-dimensional one.

    Returns ``(log_evidence, sample_log_likelihoods, partners)``. ``partners``
    is the resampled second-absorber index for each sample, retained so the best
    evaluated PAIR can be recovered -- discarding it left the caller able to
    report only one absorber from a two-absorber model.

    The evidence is NaN when every sample was rejected or the values
    underflowed, which is the caller's signal to stop the ladder rather than an
    error.
    """
    from .gp.evidence import _absorber_profile, absorber_search_window
    from .gp.likelihood import log_mvnpdf_low_rank

    n_samples = grid.num_samples
    z_min, z_max = absorber_search_window(prepared, config)
    z_samples = grid.sample_redshifts(z_min, z_max)
    nhi_samples = grid.nhi_samples

    finite = np.isfinite(one_absorber_samples)
    if not np.any(finite):
        return (
            float("nan"),
            np.full(n_samples, np.nan),
            np.zeros(n_samples, dtype=int),
        )

    weights = np.zeros(n_samples)
    peak = float(np.nanmax(one_absorber_samples))
    weights[finite] = np.exp(one_absorber_samples[finite] - peak)

    second = resampler(weights, n_samples)

    log_norm = (
        np.log(n_samples) if config.compatibility_profile.log_norm_round_trip else 0.0
    )

    sample_log_likelihoods = np.full(n_samples, np.nan)
    for i in range(n_samples):
        j = int(second[i])
        absorption = _absorber_profile(
            prepared, config, z_samples[i], nhi_samples[i]
        ) * _absorber_profile(prepared, config, z_samples[j], nhi_samples[j])
        sample_log_likelihoods[i] = (
            log_mvnpdf_low_rank(
                prepared.flux,
                assembled.mean * absorption,
                assembled.factor * absorption[:, None],
                assembled.absorption_variance * absorption**2 + prepared.noise_variance,
            )
            - log_norm
        )

    # Reject samples whose two absorbers sit closer than the configured
    # separation. A rejection, not a penalty -- see reject_close_pairs.
    pairs = np.vstack([z_samples, z_samples[second]])
    sample_log_likelihoods[reject_close_pairs(pairs, config.min_z_separation)] = np.nan

    return (
        combine_log_evidences(sample_log_likelihoods, n_samples, 2),
        sample_log_likelihoods,
        second.astype(int),
    )
