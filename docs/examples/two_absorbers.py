"""Run the public two-absorber workflow on a deterministic injection.

The spectrum is generated locally from the package's Voigt model. It is a
code-path demonstration, not a survey mock and not a performance claim.
"""

import numpy as np

from gp_dla_finder import Config
from gp_dla_finder.finder import Finder
from gp_dla_finder.gp.spectrum import Spectrum
from gp_dla_finder.voigt import (
    PRODUCTION_KERNEL,
    kernel_half_width,
    voigt_absorption,
)

# --8<-- [start:spectrum]
rng = np.random.default_rng(105)
wave = np.arange(3600.0, 6000.0, 0.8)
continuum = 1.0 + 0.3 * np.sin(wave / 180.0) + 0.15 * np.cos(wave / 47.0)

# Pad before applying the same line-spread kernel used by the inference, so the
# convolution has no edge artefact. Both absorbers then multiply the continuum.
half = kernel_half_width(PRODUCTION_KERNEL)
step = wave[1] - wave[0]
padded_wave = np.concatenate(
    [
        wave[0] - step * np.arange(half, 0, -1),
        wave,
        wave[-1] + step * np.arange(1, half + 1),
    ]
)
for z_abs, log_nhi in ((2.20, 20.5), (2.70, 20.6)):
    continuum *= voigt_absorption(
        padded_wave,
        nhi=10.0**log_nhi,
        z_dla=z_abs,
        num_lines=3,
        kernel=PRODUCTION_KERNEL,
    )

ivar = np.full_like(wave, 25.0)
mask = np.zeros_like(wave, dtype=bool)
for centre in np.linspace(0.5, wave.size - 0.5, 8):
    start = int(centre) - 4
    mask[start : start + 8] = True

spectrum = Spectrum(
    wavelength=wave,
    flux=continuum + rng.normal(0.0, 1.0 / np.sqrt(ivar), wave.size),
    ivar=ivar,
    z_qso=2.9,
    mask=mask,
)
# --8<-- [end:spectrum]


# --8<-- [start:run]
config = Config.desi_y3_fast(
    max_absorbers=2,
    experimental_multi_absorber=True,
    enable_tau_eb=False,  # keep this tutorial run short
    seed=0,  # the M2 resampler is deterministic
)
result = Finder(config, warn_about_threads=False).run(spectrum, targetid=2002)

assert result.completed and result.ladder is not None
for label, log_z, posterior in zip(
    result.ladder.model_labels,
    result.ladder.log_evidences,
    result.ladder.model_posteriors,
    strict=True,
):
    print(label, log_z, posterior)

print("selected model:", result.ladder.model_labels[result.ladder.selected_model])
for candidate in result.absorber_candidates:
    print(candidate.model, candidate.grid_z_abs, candidate.grid_log_nhi)
# --8<-- [end:run]
