"""Run the absorber search on one spectrum and write a catalog.

This file is the canonical source for the code shown in ``docs/tutorial.md``.
The page includes regions of it verbatim, and ``tests/test_examples.py`` runs
the complete example.

Run it directly:

    python docs/examples/quickstart.py
"""

# --8<-- [start:imports]
import numpy as np

from gp_dla_finder import Config
from gp_dla_finder.finder import Finder, results_to_catalogue
from gp_dla_finder.gp.spectrum import Spectrum
from gp_dla_finder.io.fits import write_catalogue
from gp_dla_finder.io.structured import write_structured_results

# --8<-- [end:imports]


# --8<-- [start:spectrum]
# A stand-in for your own data: wavelength in angstroms, flux, inverse variance,
# and a boolean mask that is True for pixels to ignore.
rng = np.random.default_rng(20260820)
wave = np.arange(3600.0, 5600.0, 0.8)
flux = 1.0 + 0.3 * np.sin(wave / 180.0) + rng.normal(0.0, 0.2, wave.size)
ivar = np.full_like(wave, 25.0)
mask = np.zeros_like(wave, dtype=bool)
mask[500:520] = True  # e.g. a sky line

spectrum = Spectrum(wavelength=wave, flux=flux, ivar=ivar, z_qso=2.6, mask=mask)
# --8<-- [end:spectrum]


# --8<-- [start:run]
# max_absorbers=1 is the supported path: null versus one absorber. The
# experimental two-absorber ladder needs max_absorbers=2 AND
# experimental_multi_absorber=True; read the README before using it.
#
# The presets declare max_absorbers=4, which this package does not implement, so
# saying 1 here is required. Finder() with no argument makes the same choice for
# you.
#
# enable_tau_eb=False turns OFF the per-spectrum empirical-Bayes mean-flux fit.
# The fit is implemented and is on by default; it is skipped here only to keep
# the example fast. Turning it off is a change to the forward model, so the
# override relabels the preset as desi_y3_fast+modified.
finder = Finder(Config.desi_y3_fast(enable_tau_eb=False, max_absorbers=1))
result = finder.run(spectrum, targetid=1234)

print(result.status)  # "completed", or why it could not run
print(result.p_absorber)  # posterior probability of >= 1 absorber

for candidate in result.absorber_candidates:
    print(candidate.grid_z_abs, candidate.grid_log_nhi)
# --8<-- [end:run]


# --8<-- [start:catalogue]
# `results` is the list you accumulated over your survey; here it is one result.
results = [result]

catalogue = results_to_catalogue(results, detection_threshold=0.98)

# The flat DESI catalogue: one row per absorber, fixed columns.
write_catalogue("absorbers.fits", catalogue)

# The full inference record, including the model ladder when one was evaluated.
# Standard library only, so it works without astropy.
write_structured_results("run.json", catalogue)
# --8<-- [end:catalogue]
