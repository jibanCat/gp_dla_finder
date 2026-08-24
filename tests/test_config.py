"""Configuration tests."""

from __future__ import annotations

import numpy as np
import pytest

from gp_dla_finder import load_model
from gp_dla_finder.config import SPEED_OF_LIGHT, Config
from gp_dla_finder.model import GPModel


def test_desi_y3_preset_matches_the_deployed_operating_point():
    """Guards the production configuration against silent drift.

    Every value here is from the deployed run configuration. A change to any of
    them changes which absorbers are found, so it is a PI decision, not a tidy-up.
    """
    c = Config.desi_y3()
    assert (c.min_lambda, c.max_lambda) == (911.75, 1250.0)
    assert (c.num_forest_lines, c.num_lines) == (31, 3)
    assert (c.prev_tau_0, c.prev_beta) == (0.00246, 3.62)
    assert c.max_absorbers == 4
    assert c.num_samples == 50_000
    assert c.log_nhi_range == (17.2, 22.5)
    # Full grid, not FILTER: PI ruling N60 made the conservative path the v0.1
    # default. The deployed pipeline used FILTER, so this is a deliberate
    # divergence from it and is recorded as such.
    assert c.filter_low_likelihood is False
    assert c.filter_n_initial_floor == 5000
    assert c.filter_empty_mask_fallthrough is False
    assert c.early_stop_mode == "baseline"
    assert c.enable_tau_eb is True
    assert c.tau_eb_objective == "null"
    assert c.tau_eb_apply_hcd_mask is False
    assert c.min_z_separation_kms == 3000.0


def test_presets_are_mandatory():
    """PI ruling N3: a bare Config() must not silently select production."""
    with pytest.raises(ValueError, match="requires an explicit preset"):
        Config()
    with pytest.raises(ValueError, match="non-empty name"):
        Config(preset=None)  # type: ignore[arg-type]
    # Declaring a non-standard configuration deliberately is allowed.
    assert Config(preset="custom").preset == "custom"


def test_default_is_deterministic():
    """PI ruling D4: reproducible by default, stochastic only on request."""
    assert Config.desi_y3().seed == 0
    assert Config.desi_y3(seed=None).seed is None


def test_kms_to_z_matches_the_reference_formula():
    assert Config.kms_to_z(3000.0) == (3000.0 * 1000) / SPEED_OF_LIGHT
    assert Config.desi_y3().max_z_cut == pytest.approx(0.010006922, abs=1e-9)


def test_model_labels_describe_the_posterior_vector():
    c = Config.desi_y3()
    assert c.n_models == 5
    assert c.model_labels == (
        "null",
        "1_absorber",
        "2_absorbers",
        "3_absorbers",
        "4_absorbers",
    )


def test_early_stop_names_map_onto_the_reference_names():
    assert Config.desi_y3()._reference_early_stop_mode == "baseline"
    assert (
        Config.desi_y3(early_stop_mode="no_null_stop")._reference_early_stop_mode == "A"
    )
    assert Config.desi_y3(early_stop_mode="pre_occam")._reference_early_stop_mode == "D"


def test_convolution_half_width_comes_from_the_kernel():
    assert Config.desi_y3().convolution_half_width == 3


def test_replace_marks_the_config_as_modified():
    c = Config.desi_y3().replace(num_samples=1000)
    assert c.num_samples == 1000
    assert c.preset == "desi_y3+modified"
    assert Config.desi_y3().num_samples == 50_000  # original untouched


def test_config_is_frozen():
    with pytest.raises(AttributeError):
        Config.desi_y3().min_lambda = 900.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_lambda": 1300.0}, "must be <"),
        ({"num_lines": 99}, "num_lines"),
        ({"num_forest_lines": 0}, "num_forest_lines"),
        ({"max_absorbers": 0}, "max_absorbers"),
        ({"num_samples": 0}, "num_samples"),
        ({"log_nhi_range": (22.5, 17.2)}, "increasing"),
        ({"early_stop_mode": "fastest"}, "early_stop_mode"),
        ({"tau_eb_objective": "dla"}, "tau_eb_objective"),
        ({"lsf_kernel": "boss-r2000"}, "unknown LSF kernel"),
    ],
)
def test_invalid_configs_are_rejected(kwargs, match):
    with pytest.raises((ValueError, KeyError), match=match):
        Config(preset="custom", **kwargs)


# --------------------------------------------------------------------------
# Validation against a trained model
# --------------------------------------------------------------------------


def test_deployed_model_satisfies_the_production_config():
    Config.desi_y3().validate_against(load_model())


def test_search_window_outside_the_model_grid_is_rejected():
    with pytest.raises(ValueError, match="search window"):
        Config.desi_y3(min_lambda=800.0).validate_against(load_model())


def test_model_without_a_normalisation_band_is_rejected():
    bare = GPModel(
        name="bare",
        rest_wavelengths=np.linspace(900.0, 1300.0, 50),
        mu=np.ones(50),
        M=np.zeros((50, 2)),
        log_omega=np.zeros(50),
        log_c_0=0.0,
        log_tau_0=0.0,
        log_beta=0.0,
    )
    with pytest.raises(ValueError, match="normalisation band"):
        Config.desi_y3().validate_against(bare)


def test_nan_normalisation_band_is_rejected():
    """A model trained with --no-normalize must not be silently normalised."""
    unnormalised = GPModel(
        name="unnormalised",
        rest_wavelengths=np.linspace(900.0, 1300.0, 50),
        mu=np.ones(50),
        M=np.zeros((50, 2)),
        log_omega=np.zeros(50),
        log_c_0=0.0,
        log_tau_0=0.0,
        log_beta=0.0,
        normalization_min_lambda=float("nan"),
        normalization_max_lambda=float("nan"),
    )
    with pytest.raises(ValueError, match="without flux normalisation"):
        Config.desi_y3(max_lambda=1250.0).validate_against(unnormalised)


def test_normalisation_band_outside_the_gp_grid_is_allowed():
    """The band applies to the input spectrum, not the GP interpolation grid.

    The legacy eBOSS model is the real case: its band sits redward of a grid that
    ends near 1421 A. Rejecting that would reject an approved model.
    """
    legacy_like = GPModel(
        name="legacy_like",
        rest_wavelengths=np.linspace(851.0, 1421.0, 50),
        mu=np.ones(50),
        M=np.zeros((50, 2)),
        log_omega=np.zeros(50),
        log_c_0=0.0,
        log_tau_0=0.0,
        log_beta=0.0,
        normalization_min_lambda=1425.0,
        normalization_max_lambda=1475.0,
    )
    Config.desi_y3(min_lambda=911.75, max_lambda=1250.0).validate_against(legacy_like)


def test_presets_name_their_sample_grid_explicitly():
    """PI ruling N23: an operating point never inherits a grid by accident."""
    assert Config.desi_y3().sample_grid == "pw14_172_225_50000"
    assert Config.desi_y3().num_samples == 50_000
    assert Config.desi_y3_refined().sample_grid == "pw14_172_225_100000"
    assert Config.desi_y3_refined().num_samples == 100_000
    assert Config.desi_y3_fast().sample_grid == "pw14_172_225_10000"
    assert Config.desi_y3_fast().num_samples == 10_000


def test_refined_is_never_selected_silently():
    assert Config.desi_y3().preset == "desi_y3"
    assert Config.desi_y3_refined().preset == "desi_y3_refined"
    assert Config.desi_y3().num_samples != Config.desi_y3_refined().num_samples
