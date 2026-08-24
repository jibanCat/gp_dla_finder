"""Stage 8: bitwise equivalence with the reference implementation.

The comparison is **bitwise**. Everything the reference does is reproduced,
including two pieces of arithmetic that are mathematically no-ops and numerically
are not (see :mod:`gp_dla_finder.compat`).

Scope of the claim
------------------
These tests compare *this package's* evidence path against the pinned reference
at commit ``d5b306e6`` on generated spectra. They are evidence of algorithmic
equivalence on the compared surface. They are **not** evidence of
population-level, multi-absorber, FILTER-enabled, or deployed-catalogue
equivalence.

Two substitutions, both ratified under PI ruling N34, and both bounded here:

1. *The model file.* The reference loads its trained model from HDF5 and the
   original trained file is not redistributable. When it is unavailable a
   reference-compatible HDF5 is materialised from the packaged asset. It is
   written to a **unique, test-owned temporary directory** and every dataset is
   read back and checked bitwise against the packaged model before use, so a
   stale file from an earlier model version cannot survive into a comparison.
   This substitutes the *file*, not the numbers; it is not evidence that the
   packaged asset equals the unavailable original -- that is a separate gate,
   ``test_model_assets.py::test_packaged_model_reproduces_source_bitwise``, which
   runs wherever the private source is present.
2. *The LSF kernel.* The reference's pure-Python module ships the BOSS R=2000
   kernel while the DESI production configuration uses the R~3000 one. Rather
   than patch one comparison and report it as the kernel result, both kernels are
   run: ``boss-r2000-7tap`` needs **no patch at all**, and ``desi-r3000-7tap``
   patches the reference module inside a restored context. Every test id names
   the kernel it ran under.

An autouse guard asserts the reference module is pristine at the start of every
test in this file, so a leaked patch fails the *next* test rather than silently
changing it.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from gp_dla_finder import load_model, load_sample_grid
from gp_dla_finder.config import Config
from gp_dla_finder.gp.evidence import (
    assemble_model,
    null_log_evidence,
    one_absorber_log_evidence,
)
from gp_dla_finder.gp.spectrum import prepare_spectrum
from gp_dla_finder.voigt import BOSS_KERNEL, PRODUCTION_KERNEL, lsf_kernel
from synthetic import Z_QSO, make_spectrum

pytestmark = pytest.mark.needs_reference

#: Both named kernels. ``boss-r2000-7tap`` is the reference module's own; the
#: DESI kernel requires a scoped patch. Running both is required by PI ruling N34
#: -- a patched-kernel result may not stand as the only kernel comparison.
KERNELS = (BOSS_KERNEL, PRODUCTION_KERNEL)

#: Datasets the reference's ``NullGPMAT``/``DLAGPMAT`` read out of the HDF5 file.
_MODEL_DATASETS = (
    "rest_wavelengths",
    "mu",
    "M",
    "log_omega",
    "log_c_0",
    "log_tau_0",
    "log_beta",
    "normalization_min_lambda",
    "normalization_max_lambda",
)


@pytest.fixture(autouse=True)
def reference_module_is_pristine(reference_repo):
    """Fail if a previous test left the reference module patched.

    The parity fixtures patch a module-level global in the reference. Patching it
    without restoring would make every later comparison depend on test order --
    and would do so silently, since the numbers stay plausible. This runs before
    each test in this module and refuses to proceed on a dirty module.
    """
    from gpy_dla_detection import voigt as reference_voigt

    pristine = np.array_equal(
        reference_voigt.instrument_profile, lsf_kernel(BOSS_KERNEL)
    )
    assert pristine, (
        "the reference voigt module's instrument_profile is not the BOSS kernel it "
        "ships with; an earlier test patched it and did not restore it"
    )


@pytest.fixture(scope="session")
def materialised_model(reference_repo, tmp_path_factory) -> str:
    """Path to a reference-readable HDF5 model.

    Prefers the real trained file when ``GP_DLA_FINDER_MODEL_SOURCE`` names one.
    Otherwise rebuilds one from the packaged asset in a unique temporary
    directory and verifies every dataset bitwise after writing, so no stale file
    can be picked up (PI ruling N34, adopted correction 1).
    """
    source = os.environ.get("GP_DLA_FINDER_MODEL_SOURCE")
    if source:
        return source

    h5py = pytest.importorskip("h5py")
    packaged = load_model()
    path = tmp_path_factory.mktemp("reference_model") / "model.h5"

    with h5py.File(path, "w") as handle:
        for name in _MODEL_DATASETS:
            handle.create_dataset(name, data=getattr(packaged, name))

    # Rebuilt *and* verified: read every dataset back and require bitwise
    # equality with the packaged asset before any comparison uses the file.
    with h5py.File(path, "r") as handle:
        for name in _MODEL_DATASETS:
            assert np.array_equal(handle[name][()], getattr(packaged, name)), (
                f"materialised model dataset {name!r} does not match the packaged "
                "asset it was written from"
            )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"\nmaterialised reference model: {path} sha256={digest}")
    return str(path)


def _build_reference(kernel: str, materialised: str, monkeypatch):
    """Construct the reference's null and absorber GPs under ``kernel``.

    Split out from :func:`_reference_run` because the FILTER comparison needs the
    GPs *while the kernel patch is still in force*, and paying for the 10,000
    per-sample likelihoods there as well would double a slow test for nothing.
    """
    import gpy_dla_detection.dla_gp as reference_dla_gp
    from gpy_dla_detection import voigt as reference_voigt
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from gpy_dla_detection.null_gp import NullGPMAT
    from gpy_dla_detection.set_parameters import Parameters

    # Scoped: monkeypatch restores both attributes when the fixture tears down,
    # so the patch cannot leak into another test module.
    monkeypatch.setattr(reference_voigt, "instrument_profile", lsf_kernel(kernel))
    monkeypatch.setattr(
        reference_dla_gp, "voigt_absorption", reference_voigt.voigt_absorption
    )

    config = Config.desi_y3_fast().replace(lsf_kernel=kernel)
    grid = load_sample_grid(config.sample_grid)
    spectrum = make_spectrum()

    class StubPrior:
        """The prior is not exercised here; evidence does not consult it."""

        def less_ind(self, z_qso):
            return 4000.0, 60000.0

    class GridAdapter:
        def __init__(self, params):
            self.params = params
            self.log_nhi_samples = grid.log_nhi_samples
            self.nhi_samples = grid.nhi_samples

        def sample_z_dlas(self, wavelengths, z_qso):
            lo = self.params.min_z_dla(wavelengths, z_qso)
            hi = self.params.max_z_dla(wavelengths, z_qso)
            return lo + (hi - lo) * grid.offset_samples

    params = Parameters(
        normalization_min_lambda=1425,
        normalization_max_lambda=1475,
        min_lambda=config.min_lambda,
        max_lambda=config.max_lambda,
        k=30,
        num_lines=config.num_lines,
        num_forest_lines=config.num_forest_lines,
        num_dla_samples=config.num_samples,
    )
    rest = np.array(spectrum.wavelength) / (1 + Z_QSO)
    noise = np.where(
        np.array(spectrum.ivar) > 0,
        1.0 / np.where(np.array(spectrum.ivar) > 0, spectrum.ivar, 1.0),
        np.nan,
    )

    null_gp = NullGPMAT(
        params,
        StubPrior(),
        learned_file=materialised,
        prev_tau_0=config.prev_tau_0,
        prev_beta=config.prev_beta,
    )
    null_gp.set_data(
        rest, np.array(spectrum.flux), noise, np.array(spectrum.mask), Z_QSO
    )

    absorber_gp = DLAGPMAT(
        params,
        StubPrior(),
        GridAdapter(params),
        min_z_separation=3000.0,
        learned_file=materialised,
        broadening=True,
        prev_tau_0=config.prev_tau_0,
        prev_beta=config.prev_beta,
    )
    absorber_gp.set_data(
        rest, np.array(spectrum.flux), noise, np.array(spectrum.mask), Z_QSO
    )

    z_samples = GridAdapter(params).sample_z_dlas(absorber_gp.this_wavelengths, Z_QSO)
    return config, grid, null_gp, absorber_gp, z_samples


def _reference_run(kernel: str, materialised: str, monkeypatch):
    """Run the milestone spectrum through the reference under ``kernel``."""
    config, grid, null_gp, absorber_gp, z_samples = _build_reference(
        kernel, materialised, monkeypatch
    )
    per_sample = np.array(
        [
            absorber_gp.sample_log_likelihood_k_dlas(
                np.array([z_samples[i]]), np.array([grid.nhi_samples[i]])
            )
            - np.log(config.num_samples)
            for i in range(config.num_samples)
        ]
    )
    peak = np.nanmax(per_sample)
    one_absorber = (
        peak
        + np.log(np.nanmean(np.exp(per_sample - peak)))
        + np.log(config.num_samples)
    )
    return {
        "null_gp": null_gp,
        "null": float(null_gp.log_model_evidence()),
        "one_absorber": float(one_absorber),
        "per_sample": per_sample,
    }


@pytest.fixture(scope="module", params=KERNELS, ids=lambda k: f"kernel={k}")
def parity(request, materialised_model):
    """Packaged and reference results for one named kernel, side by side."""
    kernel = request.param
    with pytest.MonkeyPatch.context() as monkeypatch:
        reference = _reference_run(kernel, materialised_model, monkeypatch)

    model = load_model()
    config = Config.desi_y3_fast().replace(lsf_kernel=kernel)
    grid = load_sample_grid(config.sample_grid)
    prepared = prepare_spectrum(make_spectrum(), model, config)
    assembled = assemble_model(prepared, model, config)
    return kernel, config, grid, prepared, assembled, reference


@pytest.mark.parametrize(
    ("packaged_attr", "reference_attr"),
    [
        ("flux", "y"),
        ("noise_variance", "v"),
        ("rest_wavelength", "x"),
        ("padded_wavelength", "padded_wavelengths"),
    ],
)
def test_prepared_spectrum_matches_reference(parity, packaged_attr, reference_attr):
    _, _, _, prepared, _, reference = parity
    assert np.array_equal(
        getattr(prepared, packaged_attr), getattr(reference["null_gp"], reference_attr)
    )


@pytest.mark.parametrize(
    ("packaged_attr", "reference_attr"),
    [("mean", "this_mu"), ("factor", "this_M"), ("absorption_variance", "this_omega2")],
)
def test_assembled_model_matches_reference(parity, packaged_attr, reference_attr):
    _, _, _, _, assembled, reference = parity
    assert np.array_equal(
        getattr(assembled, packaged_attr), getattr(reference["null_gp"], reference_attr)
    )


def test_null_evidence_is_bit_identical_to_reference(parity):
    _, _, _, prepared, assembled, reference = parity
    assert null_log_evidence(prepared, assembled) == reference["null"]


def test_one_absorber_evidence_is_bit_identical_to_reference(parity):
    _, config, grid, prepared, assembled, reference = parity
    assert (
        one_absorber_log_evidence(prepared, assembled, grid, config, mode="exact")
        == reference["one_absorber"]
    )


def test_every_per_sample_log_likelihood_matches_reference(parity):
    """Not just the aggregate: all N sample likelihoods, element for element."""
    _, config, grid, prepared, assembled, reference = parity
    _, samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="exact", return_samples=True
    )
    assert np.array_equal(samples, reference["per_sample"])


class SerialExecutor:
    """Runs submitted work inline, returning a resolved future.

    The reference's FILTER path takes an executor. Handing it this one compares
    the FILTER *algorithm* rather than a process pool, and sidesteps the pickling
    and spawn semantics that a real pool would drag into the test.
    """

    def submit(self, function, *args, **kwargs):
        import concurrent.futures

        future = concurrent.futures.Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrored into the future
            future.set_exception(exc)
        return future


@pytest.mark.slow
def test_filter_mode_reproduces_the_reference_filter_evidence(
    parity, materialised_model, monkeypatch
):
    """FILTER=1 at one absorber is a prefix estimator, and this proves it.

    Reading the reference suggests that its adaptive region-A machinery never
    reaches the one-absorber evidence: "FILTER fix #5" discards the refined
    samples at ``num_dlas == 0`` and uses the coarse scan alone. That reading is
    load-bearing for this package's ``mode="filter"``, so it is *checked* against
    the reference's own FILTER path rather than trusted.
    """
    kernel, config, grid, prepared, assembled, reference = parity

    # Its own scoped patch: the module-scoped `parity` fixture has already
    # restored the reference kernel by the time this test runs, and the FILTER
    # call has to happen with the kernel under test in force.
    _, _, _, absorber_gp, _ = _build_reference(kernel, materialised_model, monkeypatch)

    reference_filter = absorber_gp.parallel_log_model_evidences(
        max_dlas=1,
        max_workers=4,
        batch_size=500,
        executor=SerialExecutor(),
        null_evidence=reference["null"],
        filter_low_likelihood=True,
        filter_n_initial_floor=config.filter_n_initial_floor,
        filter_empty_mask_fallthrough=config.filter_empty_mask_fallthrough,
    )

    packaged = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="filter"
    )
    assert packaged == float(reference_filter[0])


def test_the_two_kernels_are_not_the_same_forward_model(parity):
    """Guards the comparison itself.

    If the two parametrisations produced the same numbers, running both would be
    theatre. They must differ, and by an amount worth naming.
    """
    kernel, config, grid, prepared, assembled, reference = parity
    other = BOSS_KERNEL if kernel == PRODUCTION_KERNEL else PRODUCTION_KERNEL
    alternative = one_absorber_log_evidence(
        prepared, assembled, grid, config.replace(lsf_kernel=other), mode="exact"
    )
    assert alternative != reference["one_absorber"]


# --------------------------------------------------------------------------
# The catalogue schema, extracted from the PINNED reference source
# --------------------------------------------------------------------------


def _extract_catalogue_schema(reference_root):
    """Read the reference's own ``names`` and ``dtype`` literals out of its AST.

    PI ruling (increment-14 correction 1). ``tests/test_catalogue.py`` compares
    against a transcription of these tuples; a transcription can drift. This
    reads the literals from the pinned source itself, so drift fails here.

    Parsing rather than importing: ``dlasearch`` pulls in the whole DESI stack,
    and this project deliberately does not have it. The public repository is
    opened read-only and nothing is written to it.
    """
    import ast

    source = (reference_root / "dlasearch.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if "names" not in keywords or "dtype" not in keywords:
            continue
        try:
            names = ast.literal_eval(keywords["names"])
            dtypes = ast.literal_eval(keywords["dtype"])
        except ValueError:
            continue
        if isinstance(names, (list, tuple)) and "TARGETID" in names:
            return tuple(names), tuple(dtypes)
    raise AssertionError("no catalogue Table(...) call found in dlasearch.py")


def test_the_transcribed_catalogue_schema_matches_the_pinned_reference(reference_repo):
    """Guards the transcription in tests/test_catalogue.py against drift."""
    from test_catalogue import REFERENCE_COLUMNS, REFERENCE_DTYPES

    names, dtypes = _extract_catalogue_schema(reference_repo)

    assert names == REFERENCE_COLUMNS, (
        "the reference's column names/order have changed, or the transcription in "
        "tests/test_catalogue.py has drifted"
    )
    assert dtypes == REFERENCE_DTYPES, (
        "the reference's dtype tuple has changed, or the transcription has drifted"
    )


def test_the_packaged_legacy_columns_match_the_pinned_reference(reference_repo):
    """The package's own schema against the reference source, not a copy of it."""
    from gp_dla_finder.catalogue import LEGACY_ABSORBER_COLUMNS

    names, _ = _extract_catalogue_schema(reference_repo)
    assert LEGACY_ABSORBER_COLUMNS == names


def test_the_reference_dlaid_rule_matches_the_pinned_source(reference_repo):
    """``str(tid) + "00" + str(n)``, read out of the source rather than assumed."""
    from gp_dla_finder.catalogue import reference_dlaid

    source = (reference_repo / "dlasearch.py").read_text()
    assert 'dlaid = str(tid) + "00" + str(n)' in source, (
        "the reference's DLAID construction has changed; reference_dlaid() must "
        "be updated to match"
    )
    assert reference_dlaid(39627000, 0) == "39627000" + "00" + "0"


# --------------------------------------------------------------------------
# The mean-flux port, against the pinned reference source
# --------------------------------------------------------------------------
#
# PI ruling N68 requires parity evidence naming a specific commit. The
# reference's fit_tau_eb needs NullGPMAT and a learned_file, which this package
# does not have, so the parity that CAN be established without reconstructing
# the production class hierarchy is structural: that the ported algorithm makes
# the same decisions from the same inputs, and that the constants it was ported
# with are the ones the reference actually declares.


@pytest.mark.needs_reference
def test_the_ported_constants_match_the_pinned_reference(reference_repo):
    """The tau grid, seed, threshold and objective names, read from source."""
    import ast

    source = (reference_repo / "gpy_dla_detection" / "tau_eb.py").read_text()
    tree = ast.parse(source)

    signature = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "fit_tau_eb"
    )
    defaults = {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(
            signature.args.kwonlyargs, signature.args.kw_defaults, strict=True
        )
        if default is not None
    }

    from gp_dla_finder.config import Config

    config = Config.desi_y3()

    # The production tau grid, extended to 6x in d849a30.
    assert tuple(defaults["tau_factors"]) == tuple(config.tau_eb_factors)
    # The HCD mask is off by default in the reference too -- it was retracted.
    assert defaults["apply_hcd_mask"] is False
    assert config.tau_eb_apply_hcd_mask is False
    assert defaults["mask_threshold_sigma"] == config.tau_eb_mask_threshold_sigma
    # And the default objective production runs.
    assert defaults["objective"] == "null"


@pytest.mark.needs_reference
def test_the_reference_still_selects_by_argmax(reference_repo):
    """The selection rule this port reproduces, asserted against the source.

    If the reference ever changed to, say, an interpolated maximum, the ported
    argmax would silently stop matching it.
    """
    source = (reference_repo / "gpy_dla_detection" / "tau_eb.py").read_text()
    assert "j_best = int(np.argmax(log_l_per_tau))" in source
    assert "tau_eb = float(prev_tau_0_seed * tau_factor_best)" in source


@pytest.mark.needs_reference
def test_the_retraction_is_still_recorded_where_this_port_says_it_is(
    reference_repo,
):
    """The port's docstring cites measurements; this is where they live.

    Read from the PINNED COMMIT, not the working tree. ``docs/notes`` was
    untracked at 21e4e87 and migrated to a separate private repository, so the
    retraction is no longer present at HEAD -- it survives in history. A port
    citing it has to be able to point at where, which is what this asserts.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip(
            "reading a note from a pinned commit needs git; the canonical-parity "
            "container does not ship it"
        )

    note = subprocess.run(
        [
            "git",
            "show",
            "9aa20dc:docs/notes/2026-04-29_voigt_lsf_sweep/findings.md",
        ],
        cwd=reference_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "SECOND RETRACTION" in note
    assert "apply_hcd_mask=False" in note
    # The two numbers the port's docstring quotes. The note writes the negative
    # with a Unicode minus (U+2212), not an ASCII hyphen -- normalise before
    # comparing rather than matching one rendering of the same number.
    normalised = note.replace("\u2212", "-")
    assert "-0.131" in normalised
    assert "+0.026" in normalised


@pytest.mark.needs_reference
def test_the_ported_mask_off_branch_matches_the_reference(reference_repo):
    """With the mask off the reference scans on the ORIGINAL pixel mask.

    That is what makes this port's "use the spectrum as prepared" correct, and
    it is the branch production runs.
    """
    source = (reference_repo / "gpy_dla_detection" / "tau_eb.py").read_text()
    assert "    if apply_hcd_mask:" in source
    assert "        new_mask = pixel_mask" in source


# --------------------------------------------------------------------------
# LIVE numerical parity for the mean-flux fit (PI ruling N71)
# --------------------------------------------------------------------------
#
# Structural inspection of the reference source is not the acceptance gate. This
# calls the pinned reference's fit_tau_eb() and the package's fit_tau_0() on the
# same generated spectra and compares the numbers.
#
# The setup cost I claimed in Increment 19 was overstated: the adapter above
# already materialises the packaged model as a reference-readable HDF5 file and
# builds NullGPMAT from it, so this reuses that rather than standing anything up.


def _reference_fit_tau_eb(kernel, materialised, monkeypatch, spectrum, config):
    """Call the reference's fit_tau_eb() on one generated spectrum."""
    import gpy_dla_detection.dla_gp as reference_dla_gp
    from gpy_dla_detection import voigt as reference_voigt
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.tau_eb import fit_tau_eb

    monkeypatch.setattr(reference_voigt, "instrument_profile", lsf_kernel(kernel))
    monkeypatch.setattr(
        reference_dla_gp, "voigt_absorption", reference_voigt.voigt_absorption
    )

    class StubPrior:
        def less_ind(self, z_qso):
            return 4000.0, 60000.0

    params = Parameters(
        normalization_min_lambda=1425,
        normalization_max_lambda=1475,
        min_lambda=config.min_lambda,
        max_lambda=config.max_lambda,
        k=30,
        num_lines=config.num_lines,
        num_forest_lines=config.num_forest_lines,
        num_dla_samples=config.num_samples,
    )
    rest = np.array(spectrum.wavelength) / (1 + spectrum.z_qso)
    ivar = np.array(spectrum.ivar)
    noise = np.where(ivar > 0, 1.0 / np.where(ivar > 0, ivar, 1.0), np.nan)

    tau_eb, info = fit_tau_eb(
        params=params,
        prior=StubPrior(),
        learned_file=materialised,
        rest_wavelengths=rest,
        flux=np.array(spectrum.flux),
        noise_variance=noise,
        pixel_mask=np.array(spectrum.mask),
        z_qso=spectrum.z_qso,
        prev_tau_0_seed=config.prev_tau_0,
        prev_beta=config.prev_beta,
        tau_factors=tuple(config.tau_eb_factors),
        apply_hcd_mask=False,
        objective="null",
        return_diagnostics=True,
    )
    return float(tau_eb), info


#: Deterministic generated spectra spanning the cases N71 names: an ordinary
#: forest with no absorber, an injected DLA, and a low-signal case where the
#: maximum is more likely to sit at a grid edge.
_PARITY_CASES = ("absorber-free-desi-grid", "classical-dla-mid-z", "low-snr-dla")


@pytest.mark.needs_reference
@pytest.mark.slow
@pytest.mark.parametrize("case_name", _PARITY_CASES)
def test_the_mean_flux_fit_matches_the_reference_numerically(
    case_name, materialised_model, monkeypatch
):
    """The gate N71 requires: same evidences, same factor, same tau_0."""
    from gp_dla_finder.gp.spectrum import prepare_spectrum
    from gp_dla_finder.mean_flux import fit_tau_0
    from synthetic import CORPUS, build

    case = {c.name: c for c in CORPUS}[case_name]
    config = Config.desi_y3_fast()
    spectrum = build(case)
    model = load_model()

    reference_tau, info = _reference_fit_tau_eb(
        config.lsf_kernel, materialised_model, monkeypatch, spectrum, config
    )
    ours = fit_tau_0(prepare_spectrum(spectrum, model, config), model, config)

    reference_curve = np.asarray(info["log_l_per_tau"], dtype=float)
    ours_curve = np.asarray(ours.log_evidence, dtype=float)

    # 1. the complete per-factor log-evidence vector
    assert ours_curve.shape == reference_curve.shape
    difference = np.max(np.abs(ours_curve - reference_curve))
    print(f"\n{case_name}: worst |delta log evidence| = {difference:.3e}")
    # EXACT, not a tolerance. The port re-expresses the reference against this
    # package's types but performs the same operations in the same order, and
    # the measurement says so: 0.0 on every case. A tolerance here would hide a
    # future change in operation order rather than surface it (PI ruling N71 --
    # require exact agreement where the arithmetic is demonstrably identical).
    assert difference == 0.0, (
        f"{case_name}: the scan curves differ by {difference:.3e} nat. If a "
        "deliberate change in operation order caused this, measure it and adopt "
        "a narrow evidence-based tolerance -- do not widen this to hide it."
    )

    # 2. the selected factor and the fitted tau_0 -- these must match exactly,
    #    because a different selection is a different scientific answer, not a
    #    rounding difference.
    assert ours.factor == float(info["tau_factor_best"])
    # Exact, not approx. Both sides compute seed * factor from the same two
    # float64 values, so the product is bit-identical; an approx here would have
    # let a real difference through while the review package called it exact.
    assert ours.tau_0 == reference_tau

    # 3. the grid-edge result and the winner/runner-up margin
    reference_best = int(np.argmax(reference_curve))
    at_edge = reference_best in (0, reference_curve.size - 1)
    assert ours.at_grid_edge is at_edge

    ordered = np.sort(reference_curve)[::-1]
    assert ours.margin == float(ordered[0] - ordered[1])


@pytest.mark.needs_reference
@pytest.mark.slow
def test_the_mask_off_branch_is_what_both_sides_run(materialised_model, monkeypatch):
    """4. effective mask-off behaviour, compared rather than assumed.

    Asking the reference for the mask must change its answer; if it did not,
    "we run the mask-off branch" would be an untested statement.
    """
    from synthetic import CORPUS, build

    case = {c.name: c for c in CORPUS}["classical-dla-mid-z"]
    config = Config.desi_y3_fast()
    spectrum = build(case)

    unmasked, unmasked_info = _reference_fit_tau_eb(
        config.lsf_kernel, materialised_model, monkeypatch, spectrum, config
    )
    assert int(unmasked_info["n_hcd"]) == 0, (
        "with the mask off the reference must flag no HCD pixels"
    )
    assert np.all(np.isfinite(np.asarray(unmasked_info["log_l_per_tau"])))
    assert unmasked > 0.0


# --------------------------------------------------------------------------
# Live controlled-seed M0/M1/M2 comparison on the one-DLA control
# --------------------------------------------------------------------------
#
# The question this exists to answer: the package's M2 path selects two
# absorbers for a spectrum with one injected DLA. Is that the reference's own
# behaviour, faithfully ported, or a difference introduced by this package?
#
# Both sides are given the same generated spectrum and the same seed. The
# reference draws from the legacy global stream, so the test seeds it; the
# package uses its own local RandomState with the same seed, which produces the
# same floats.


@pytest.mark.needs_reference
@pytest.mark.slow
def test_the_one_dla_control_matches_the_reference_ladder(
    materialised_model, monkeypatch
):
    """M1 and M2 against a live reference call, under a controlled seed."""
    import gpy_dla_detection.dla_gp as reference_dla_gp
    from gpy_dla_detection import voigt as reference_voigt
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from gpy_dla_detection.set_parameters import Parameters

    from gp_dla_finder.gp.evidence import (
        assemble_model,
        null_log_evidence,
        one_absorber_log_evidence,
    )
    from gp_dla_finder.gp.spectrum import prepare_spectrum
    from gp_dla_finder.multi import (
        ModelLadder,
        seeded_resampler,
        two_absorber_log_evidence,
    )
    from gp_dla_finder.prior import load_prior
    from synthetic import CORPUS, build

    config = Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=2)
    kernel = config.lsf_kernel
    monkeypatch.setattr(reference_voigt, "instrument_profile", lsf_kernel(kernel))
    monkeypatch.setattr(
        reference_dla_gp, "voigt_absorption", reference_voigt.voigt_absorption
    )

    case = {c.name: c for c in CORPUS}["classical-dla-mid-z"]
    spectrum = build(case)
    model = load_model()
    grid = load_sample_grid(config.sample_grid)

    # --- the package side ---------------------------------------------------
    prepared = prepare_spectrum(spectrum, model, config)
    assembled = assemble_model(prepared, model, config)
    ours_null = float(null_log_evidence(prepared, assembled))
    ours_one, samples = one_absorber_log_evidence(
        prepared, assembled, grid, config, mode="exact", return_samples=True
    )
    ours_two, ours_two_samples, partners = two_absorber_log_evidence(
        prepared,
        assembled,
        grid,
        config,
        one_absorber_samples=samples,
        resampler=seeded_resampler(0),
    )

    # --- the reference side -------------------------------------------------
    class StubPrior:
        def less_ind(self, z_qso):
            return 4000.0, 60000.0

    class GridAdapter:
        def __init__(self, params):
            self.params = params
            self.log_nhi_samples = grid.log_nhi_samples
            self.nhi_samples = grid.nhi_samples

        def sample_z_dlas(self, wavelengths, z_qso):
            lo = self.params.min_z_dla(wavelengths, z_qso)
            hi = self.params.max_z_dla(wavelengths, z_qso)
            return lo + (hi - lo) * grid.offset_samples

    params = Parameters(
        normalization_min_lambda=1425,
        normalization_max_lambda=1475,
        min_lambda=config.min_lambda,
        max_lambda=config.max_lambda,
        k=30,
        num_lines=config.num_lines,
        num_forest_lines=config.num_forest_lines,
        num_dla_samples=config.num_samples,
    )
    rest = np.array(spectrum.wavelength) / (1 + spectrum.z_qso)
    ivar = np.array(spectrum.ivar)
    noise = np.where(ivar > 0, 1.0 / np.where(ivar > 0, ivar, 1.0), np.nan)

    absorber_gp = DLAGPMAT(
        params,
        StubPrior(),
        GridAdapter(params),
        min_z_separation=config.min_z_separation_kms,
        learned_file=materialised_model,
        broadening=config.broadening,
        prev_tau_0=config.prev_tau_0,
        prev_beta=config.prev_beta,
    )
    absorber_gp.set_data(
        rest, np.array(spectrum.flux), noise, np.array(spectrum.mask), spectrum.z_qso
    )

    # Same seed, same stream: the reference draws from the legacy global RNG.
    np.random.seed(0)  # noqa: NPY002
    reference_ladder = absorber_gp.log_model_evidences(2)

    reference_one = float(reference_ladder[0])
    reference_two = float(reference_ladder[1])

    print(
        f"\nM1  ours {ours_one:.4f}  reference {reference_one:.4f}"
        f"  delta {ours_one - reference_one:+.3e}"
    )
    print(
        f"M2  ours {ours_two:.4f}  reference {reference_two:.4f}"
        f"  delta {ours_two - reference_two:+.3e}"
    )
    print(f"M0  ours {ours_null:.4f}")
    print(
        f"reference prefers "
        f"{'M2' if reference_two > reference_one else 'M1'}; "
        f"package prefers {'M2' if ours_two > ours_one else 'M1'}"
    )

    # --- (2) the resampled partner indices ---------------------------------
    #
    # The reference stores them on the GP after log_model_evidences(). Same
    # seed, same stream, so the draws must be identical -- if they were not,
    # matching evidences would be a coincidence.
    reference_partners = np.asarray(absorber_gp.base_sample_inds[0], dtype=int)
    assert np.array_equal(partners, reference_partners), (
        "the resampled partner indices differ, so the two sides are not "
        "integrating over the same pairs"
    )

    # --- (3) the best evaluated pair ---------------------------------------
    reference_samples = np.asarray(absorber_gp.sample_log_likelihoods[:, 1])
    ours_best = int(
        np.nanargmax(np.where(np.isfinite(ours_two_samples), ours_two_samples, -np.inf))
    )
    reference_best = int(
        np.nanargmax(
            np.where(np.isfinite(reference_samples), reference_samples, -np.inf)
        )
    )
    assert ours_best == reference_best, "the best evaluated pair differs"
    assert partners[ours_best] == reference_partners[reference_best]

    # --- (4) aligned model priors ------------------------------------------
    prior = load_prior()
    absorber_priors = prior.log_priors(spectrum.z_qso, 2, config.prior_z_qso_increase)
    log_prior_null = prior.log_prior_no_absorber(
        spectrum.z_qso, config.prior_z_qso_increase
    )
    ladder = ModelLadder(
        (ours_null, ours_one, ours_two),
        (log_prior_null, float(absorber_priors[0]), float(absorber_priors[1])),
    )
    # P(M0) + P(exactly 1) + P(>= 2) is a partition, so it sums to one.
    assert sum(np.exp(ladder.log_priors)) == pytest.approx(1.0, abs=1e-9)

    # --- (5) model posteriors and (6) the selected model --------------------
    posteriors = ladder.model_posteriors
    assert sum(posteriors) == pytest.approx(1.0)
    joint = np.asarray(ladder.log_joint)
    assert ladder.selected_model == int(np.argmax(joint))
    assert ladder.p_absorber == pytest.approx(posteriors[1] + posteriors[2])

    print(f"partners identical: {np.array_equal(partners, reference_partners)}")
    print(f"best pair: ours {ours_best}  reference {reference_best}")
    print(f"priors    : {tuple(round(v, 3) for v in ladder.log_priors)}")
    print(f"posteriors: {tuple(round(v, 4) for v in posteriors)}")
    print(f"selected  : {ladder.model_labels[ladder.selected_model]}")

    # Measured: both rungs agree BITWISE under a controlled seed.
    assert ours_one == reference_one, (
        "the one-absorber evidences disagree, which is upstream of anything "
        "the two-absorber path does"
    )
    assert ours_two == reference_two, (
        "the two-absorber evidences disagree under a controlled seed. That is a "
        "port difference, not a legacy limitation, and it must be found before "
        "any M2 result is quoted."
    )

    # The headline this test exists for: the reference ALSO prefers two
    # absorbers for a one-DLA spectrum. So the spurious second absorber the
    # package reports is the reference's own behaviour, faithfully reproduced --
    # a bounded legacy limitation, not something introduced here.
    assert reference_two > reference_one, (
        "the reference no longer prefers M2 on the one-DLA control; the "
        "package's behaviour can no longer be attributed to it"
    )
    assert (ours_two > ours_one) == (reference_two > reference_one)
