"""The configuration must not describe a different prior than the grid it uses.

PI ruling N83. ``log_nhi_range`` and ``log_nhi_prior_alpha`` are declarative:
the column-density prior is drawn once into the sample grid, and nothing reads
those fields at inference time. A configuration that disagrees with its grid
therefore changes the recorded science and not the calculation, which is the
one failure mode a reader cannot detect from the output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gp_dla_finder import Config, load_sample_grid
from gp_dla_finder.finder import Finder, SampleGridMismatch
from gp_dla_finder.samples import REQUIRED_GRID_METADATA

ROOT = Path(__file__).resolve().parents[1]


def _finder(config, **kwargs):
    return Finder(config, warn_about_threads=False, **kwargs)


# --- the guard fires on each field it is supposed to check --------------------


def test_a_matching_configuration_runs():
    _finder(Config.desi_y3(max_absorbers=1))


def test_a_narrowed_support_is_refused():
    """The case that motivated the ruling.

    A user narrows the range to search only DLAs. The samples do not move; the
    digest and the recorded preset do.
    """
    with pytest.raises(SampleGridMismatch, match="log_nhi_range"):
        _finder(Config.desi_y3(max_absorbers=1, log_nhi_range=(20.3, 22.5)))


def test_a_different_mixture_weight_is_refused():
    with pytest.raises(SampleGridMismatch, match="log_nhi_prior_alpha"):
        _finder(Config.desi_y3(max_absorbers=1, log_nhi_prior_alpha=0.5))


def test_a_sample_budget_that_does_not_match_the_grid_is_refused():
    """`num_samples` and `sample_grid` are separate fields and can drift apart."""
    with pytest.raises(SampleGridMismatch, match="num_samples"):
        _finder(Config.desi_y3(max_absorbers=1, num_samples=10_000))


def test_the_error_names_every_field_that_disagrees():
    with pytest.raises(SampleGridMismatch) as caught:
        _finder(
            Config.desi_y3(
                max_absorbers=1,
                num_samples=10_000,
                log_nhi_range=(20.3, 22.5),
                log_nhi_prior_alpha=0.5,
            )
        )
    message = str(caught.value)
    for field in ("num_samples", "log_nhi_range", "log_nhi_prior_alpha"):
        assert field in message
    # And it says what to do about it, not only what is wrong.
    assert "build_sample_grid.py" in message
    assert "load_sample_grid(path=" in message


def test_it_is_an_error_and_not_a_warning():
    """A warning would leave the misleading record in place."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SampleGridMismatch):
            _finder(Config.desi_y3(max_absorbers=1, log_nhi_range=(20.3, 22.5)))


@pytest.mark.parametrize("preset", ["desi_y3", "desi_y3_refined", "desi_y3_fast"])
def test_every_packaged_preset_passes_its_own_guard(preset):
    """Every shipped preset keeps working.

    Not the same as "refuses nothing that was previously valid" -- bare grids
    did run before N88 and are now refused on purpose. The guarantee is
    narrower and true: every shipped preset, and every fully provenanced grid
    from the supported builder, continues to run.
    """
    _finder(getattr(Config, preset)(max_absorbers=1))


# --- the public external-grid workflow the guard points users at --------------


@pytest.fixture(scope="module")
def custom_grid(tmp_path_factory):
    """A grid built the way a user would, outside the installed package."""
    out = tmp_path_factory.mktemp("grids")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_sample_grid.py"),
            "--name",
            "dla_only_2000",
            "--num-samples",
            "2000",
            "--log-nhi",
            "20.3",
            "22.5",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return out / "dla_only_2000.npz"


def test_the_builder_writes_the_arrays_and_a_provenance_sidecar(custom_grid):
    assert custom_grid.is_file()
    sidecar = custom_grid.with_suffix(".json")
    assert sidecar.is_file(), "the provenance sidecar is what makes the grid checkable"
    provenance = json.loads(sidecar.read_text())
    assert provenance["name"] == "dla_only_2000"
    assert tuple(provenance["prior"]["support_log_nhi"]) == (20.3, 22.5)


def test_an_external_grid_keeps_its_name_and_provenance(custom_grid):
    """It used to load as '<external>' with no provenance, so nothing could be
    checked against it."""
    grid = load_sample_grid(path=custom_grid)
    assert grid.name == "dla_only_2000"
    assert grid.declared_support == (20.3, 22.5)
    assert grid.declared_prior_alpha == pytest.approx(0.97)
    assert grid.num_samples == 2000
    assert np.all(grid.log_nhi_samples >= 20.3)


def test_a_custom_grid_runs_when_the_configuration_matches_it(custom_grid):
    """The whole workflow, with no file placed inside the package."""
    grid = load_sample_grid(path=custom_grid)
    config = Config.desi_y3(
        max_absorbers=1,
        sample_grid=grid.name,
        num_samples=grid.num_samples,
        log_nhi_range=grid.declared_support,
        log_nhi_prior_alpha=grid.declared_prior_alpha,
        preset="custom-dla-only",
    )
    finder = _finder(config, grid=grid)
    assert finder.grid.name == "dla_only_2000"


def test_a_grid_the_configuration_does_not_name_is_refused(custom_grid):
    """``sample_grid`` enters the digest and the catalogue's GPDLF_SAMPLE_GRID.

    Leaving it at a packaged name while running a custom grid would have the
    file cite a grid the run never read.
    """
    grid = load_sample_grid(path=custom_grid)
    config = Config.desi_y3(
        max_absorbers=1,
        num_samples=grid.num_samples,
        log_nhi_range=grid.declared_support,
        log_nhi_prior_alpha=grid.declared_prior_alpha,
        preset="custom-dla-only",
    )
    with pytest.raises(SampleGridMismatch, match="sample_grid"):
        _finder(config, grid=grid)


def test_a_custom_grid_is_refused_when_the_configuration_does_not_match(custom_grid):
    grid = load_sample_grid(path=custom_grid)
    config = Config.desi_y3(
        max_absorbers=1,
        num_samples=grid.num_samples,
        # Left at the packaged default, which this grid does not cover.
        preset="custom-dla-only",
    )
    with pytest.raises(SampleGridMismatch, match="log_nhi_range"):
        _finder(config, grid=grid)


# --- a grid that cannot be checked cannot be used (PI ruling N88) -------------


@pytest.fixture
def bare_grid(custom_grid, tmp_path):
    """The .npz alone, with its provenance sidecar deliberately absent."""
    bare = tmp_path / "bare.npz"
    bare.write_bytes(custom_grid.read_bytes())
    assert not bare.with_suffix(".json").exists()
    return bare


def test_a_bare_grid_loads_for_inspection(bare_grid):
    """The arrays are all there; it is inference that is refused, not loading."""
    grid = load_sample_grid(path=bare_grid)
    assert grid.num_samples == 2000
    assert np.all(grid.log_nhi_samples >= 20.3)
    # But it makes no claims about itself.
    assert grid.declared_support is None
    assert grid.declared_prior_alpha is None
    assert grid.name.startswith("<external:")


def test_a_bare_grid_knows_it_is_not_inference_ready(bare_grid):
    grid = load_sample_grid(path=bare_grid)
    assert grid.usable_for_inference is False
    assert set(grid.inference_metadata) == set(REQUIRED_GRID_METADATA)


def test_inference_with_a_bare_grid_is_refused(bare_grid):
    """Fail closed.

    The previous behaviour checked the sample count and accepted the grid,
    which left exactly the hole N83 exists to close: a count match is not
    evidence that the prior is what the configuration claims, and the run still
    recorded config.sample_grid -- a packaged name for a grid it never read.
    """
    grid = load_sample_grid(path=bare_grid)
    config = Config.desi_y3(
        max_absorbers=1, num_samples=grid.num_samples, preset="custom-bare"
    )
    with pytest.raises(SampleGridMismatch) as caught:
        _finder(config, grid=grid)

    message = str(caught.value)
    assert "cannot be used for inference" in message
    # It names what is missing, and how the supported builder supplies it.
    for field in ("prior.support_log_nhi", "prior.mixture_weight_pw14", "sha256"):
        assert field in message
    assert "build_sample_grid.py" in message
    assert "load_sample_grid(path=" in message


def test_a_matching_count_does_not_rescue_a_bare_grid(bare_grid):
    """The specific regression: count agreement is not provenance."""
    grid = load_sample_grid(path=bare_grid)
    config = Config.desi_y3(
        max_absorbers=1,
        sample_grid="whatever",
        num_samples=grid.num_samples,  # agrees exactly
        log_nhi_range=(20.3, 22.5),  # and so does this, by luck
        preset="custom-bare",
    )
    with pytest.raises(SampleGridMismatch, match="cannot be used for inference"):
        _finder(config, grid=grid)


def test_restoring_the_sidecar_restores_the_grid(custom_grid, tmp_path):
    """Mutation check in the other direction: removing it cannot silently pass.

    Copying both files gives a usable grid; copying only the .npz does not. If
    a future change made the sidecar optional again, the second half fails.
    """
    both = tmp_path / "with-sidecar"
    both.mkdir()
    (both / "dla_only_2000.npz").write_bytes(custom_grid.read_bytes())
    (both / "dla_only_2000.json").write_text(
        custom_grid.with_suffix(".json").read_text()
    )

    restored = load_sample_grid(path=both / "dla_only_2000.npz")
    assert restored.usable_for_inference is True
    assert restored.name == "dla_only_2000"

    (both / "dla_only_2000.json").unlink()
    without = load_sample_grid(path=both / "dla_only_2000.npz")
    assert without.usable_for_inference is False
    with pytest.raises(SampleGridMismatch):
        _finder(
            Config.desi_y3(
                max_absorbers=1, num_samples=without.num_samples, preset="custom"
            ),
            grid=without,
        )


# --- the effective grid reaches every output (PI ruling N88, correction 1) -----


@pytest.mark.slow
def test_a_custom_grid_run_names_that_grid_in_every_output(custom_grid, tmp_path):
    """external grid -> Finder -> Result -> FITS + JSON, all naming the same grid.

    The failure this closes: provenance took the grid name from the
    configuration string, so a run on a custom grid could file itself under a
    packaged grid it never read.
    """
    pytest.importorskip("astropy")
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from gp_dla_finder.finder import results_to_catalogue
    from gp_dla_finder.gp.spectrum import Spectrum
    from gp_dla_finder.io.fits import read_catalogue_metadata, write_catalogue
    from gp_dla_finder.io.structured import (
        read_structured_results,
        write_structured_results,
    )
    from synthetic import CORPUS, build

    grid = load_sample_grid(path=custom_grid)
    config = Config.desi_y3(
        max_absorbers=1,
        sample_grid=grid.name,
        num_samples=grid.num_samples,
        log_nhi_range=grid.declared_support,
        log_nhi_prior_alpha=grid.declared_prior_alpha,
        enable_tau_eb=False,
        preset="custom-dla-only",
    )
    finder = _finder(config, grid=grid)

    spectrum: Spectrum = build({c.name: c for c in CORPUS}["classical-dla-mid-z"])
    result = finder.run(spectrum, targetid=7)
    assert result.status == "completed"

    # 1. the Result
    assert result.provenance["sample_grid"] == "dla_only_2000"
    assert result.provenance["preset"] == "custom-dla-only"

    catalogue = results_to_catalogue([result], detection_threshold=0.98)

    # 2. the run record
    assert catalogue.run["GPDLF_SAMPLE_GRID"] == "dla_only_2000"

    # 3. the FITS product
    fits_path = tmp_path / "custom.fits"
    write_catalogue(fits_path, catalogue)
    assert read_catalogue_metadata(fits_path)["GPDLF_SAMPLE_GRID"] == "dla_only_2000"

    # 4. the structured JSON
    json_path = tmp_path / "custom.json"
    write_structured_results(json_path, catalogue)
    payload = read_structured_results(json_path)
    assert payload["run"]["GPDLF_SAMPLE_GRID"] == "dla_only_2000"

    # And no output cites a packaged grid.
    assert "pw14" not in fits_path.read_bytes().decode("latin-1")
    assert "pw14" not in json_path.read_text()


def test_provenance_takes_the_grid_name_from_the_grid_not_the_config(custom_grid):
    """Structural, not incidental.

    The guard makes the two agree, so reading either gives the same answer
    today. Taking it from the grid means provenance stays correct even if a
    future path loosens the guard.
    """
    import inspect

    from gp_dla_finder.finder import Finder as FinderClass

    source = inspect.getsource(FinderClass._provenance)
    assert '"sample_grid": self.grid.name' in source, (
        "provenance should name the grid object that ran, not config.sample_grid"
    )
