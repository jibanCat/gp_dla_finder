"""Build a custom column-density prior, load it, and run with it.

This file is the canonical source for the "custom grid" section of
``docs/customisation.md``. The page includes regions of it verbatim, and
``tests/test_examples.py`` runs the complete example.

Nothing here writes inside the installed package. The grid lands in a directory
you choose, is loaded through the public loader, and is handed to ``Finder``.

Run it directly:

    python docs/examples/custom_grid.py
"""

# --8<-- [start:imports]
import subprocess
import sys
from pathlib import Path

import numpy as np

from gp_dla_finder import Config, load_sample_grid
from gp_dla_finder.finder import Finder, SampleGridMismatch
from gp_dla_finder.gp.spectrum import Spectrum

# --8<-- [end:imports]

REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER = REPOSITORY / "tools" / "build_sample_grid.py"


# --8<-- [start:build]
# Build the grid wherever you like -- `--out` is a directory of your choosing,
# and nothing is written inside the installed package. `--log-nhi 17.2 23.0`
# keeps the deployed lower limit and extends the historical upper choice from
# 22.5 to 23.0; `--num-samples` is the budget. The builder writes two files:
# `<name>.npz` with the sample arrays, and
# `<name>.json` with the provenance.
grids = Path("grids").resolve()
subprocess.run(
    [
        sys.executable,
        str(BUILDER),
        "--name",
        "extended_nhi_2000",
        "--num-samples",
        "2000",
        "--log-nhi",
        "17.2",
        "23.0",
        "--out",
        str(grids),
    ],
    check=True,
)
# --8<-- [end:build]


# --8<-- [start:load]
# Load it by path. Keep the .json beside the .npz -- it carries the name, prior
# support, mixture weight and hashes. Without it the arrays still load, but the
# grid cannot be checked against your configuration, so `Finder` will not use it
# for inference.
grid = load_sample_grid(path=grids / "extended_nhi_2000.npz")

print(grid.name)  # extended_nhi_2000
print(grid.declared_support)  # (17.2, 23.0)
print(grid.declared_prior_alpha)  # 0.97
print(grid.num_samples)  # 2000
print(float(np.max(grid.log_nhi_samples)))  # <= 23.0
# --8<-- [end:load]


# --8<-- [start:mismatch]
# The configuration has to describe the grid it is given. This one does not --
# it still claims the packaged support of 17.2 to 22.5 -- so the Finder stops
# before recording a prior that the calculation never used.
mismatched = Config.desi_y3(
    max_absorbers=1,
    num_samples=grid.num_samples,
    preset="custom-extended-nhi",
)  # still says sample_grid="pw14_172_225_50000" and the packaged support

try:
    Finder(mismatched, grid=grid)
except SampleGridMismatch as error:
    print("refused, as it should be:")
    print(error)
# --8<-- [end:mismatch]


# --8<-- [start:run]
# Describe the grid you actually built, and it runs.
config = Config.desi_y3(
    max_absorbers=1,
    sample_grid=grid.name,
    num_samples=grid.num_samples,
    log_nhi_range=grid.declared_support,
    log_nhi_prior_alpha=grid.declared_prior_alpha,
    enable_tau_eb=False,  # only to keep the example quick
    preset="custom-extended-nhi",
)

finder = Finder(config, grid=grid)

rng = np.random.default_rng(20260822)
wave = np.arange(3600.0, 5600.0, 0.8)
flux = 1.0 + 0.3 * np.sin(wave / 180.0) + rng.normal(0.0, 0.2, wave.size)
spectrum = Spectrum(
    wavelength=wave,
    flux=flux,
    ivar=np.full_like(wave, 25.0),
    z_qso=2.6,
    mask=np.zeros_like(wave, dtype=bool),
)

result = finder.run(spectrum, targetid=1)
print(result.status, result.p_absorber)

# The run records the prior it actually used, and the digest covers it.
print(result.provenance["preset"])  # custom-extended-nhi
print(result.provenance["sample_grid"])  # extended_nhi_2000
print(result.provenance["config_digest"])  # distinct from any packaged run
# --8<-- [end:run]
