# Notices — attribution and bundled-asset terms

The **source code** of `gp_dla_finder` is MIT licensed (see `LICENSE`). This file
covers everything that licence does *not* settle: prior work the code derives from,
and the provenance and redistribution status of the data and model artifacts
shipped inside the package.

## Derivation and historical attribution

This package is a repackaging of the inference core of the Gaussian-process DLA
detection method and code of Garnett et al. (2017) and Ho, Bird & Garnett (2020,
2021). The historical copyright notice of Roman Garnett is retained in `LICENSE`
alongside the copyright covering the packaging and reimplementation work.

## Development process

Ming-Feng Ho developed the packaging, tests, documentation and release workflow
with substantial assistance from Claude Opus 5 and OpenAI Codex/GPT-5.6-Sol.
The scientific decisions, review and final responsibility remain with
Ming-Feng Ho.

## Bundled model and data assets

Assets under `src/gp_dla_finder/data/` are **not** covered by the MIT licence
above. Each ships with a JSON provenance record giving its source artifact,
checksum, and conversion audit.

### `models/phase2_2lpt_loa124_nohcd_nobal_wide_m`

Trained Gaussian-process quasar-emission model; the default model, and the one
that defines the deployed production catalogues.

| | |
|---|---|
| kind | trained model parameters (mean, low-rank covariance factor, noise terms) |
| derived from | `phase2_result.h5`, sha256 `5e7a2691…c232856` |
| training data | the 2LPT-0 (`loa-124`) Lyman-α mock, HCD-free and BAL-free variant |
| redistribution status | **model parameters only.** See the restriction below. |

**Restriction — training spectra are not redistributed.** The mock spectra this
model was trained on are not redistributable by this project. No 2LPT mock
spectrum, truth catalogue, cutout, private mock identifier, or reconstructable
extract is included in this repository or its distributions. Trained parameters
are a separate artifact from the data used to fit them; shipping the former does
not authorise shipping the latter.

**Scientific caveat.** The training mock is also the calibration mock, so finder
performance quoted on it is in-sample rather than held-out.

**Cite the mock** when using this model:

> M. F. Ruiz-Herrera Bernal et al., *CoLoRe-2LPT: Lyman-alpha mock catalogues for
> the validation of DESI cosmological analyses*, arXiv:2607.27412.

(arXiv preprint, submitted to A&A, as of 2026-08-18 — not a journal publication.
Update to the journal version and DOI if one appears.)

### `models/eboss_dr16q_minus_dr12q`

Legacy Gaussian-process quasar-emission model trained on SDSS eBOSS/DR16Q. Supports
the legacy SDSS/eBOSS inference path.

| | |
|---|---|
| kind | trained model parameters, rank 20, rest grid 850.75–1420.75 A |
| source | `learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat`, sha256 `8e2f0f578e9e923b…` |
| origin | https://github.com/jibanCat/gp_dr12_trained, commit `b9f222f7` |
| licence | MIT. Attribution: Ming-Feng Ho, `gp_dr12_trained`. |

**Normalisation band.** The source artifact embeds **no** normalisation metadata.
The packaged asset records 1425–1475 A, supplied at conversion time as the
historical compatibility convention this model was fitted under. It was **not**
extracted from the file, and the asset provenance says so explicitly.

**Not yet usable as a preset.** The legacy model was trained on BOSS/eBOSS spectra,
whose line-spread function is not the DESI kernel this package ships. See the
conversion notes.

### `samples/` — quasi-Monte-Carlo absorber sample grids

Samples of `(redshift offset, log10 N_HI)` for the evidence integral. Column
densities come from the Prochaska et al. (2014) CDDF spline prior (97% PW14 + 3%
uniform) over log10 N_HI in [17.2, 22.5]; redshift offsets are uniform in [0, 1).
Both are drawn from a 2-D scrambled Halton sequence, seed 42.

| grid | samples | production target | identity |
|---|---:|---|---|
| `pw14_172_225_10000` | 10,000 | none: exploratory/tutorial grid, reproduces no catalogue | regenerated, production-array identity unverified |
| `pw14_172_225_50000` | 50,000 | pw_samples_a3_172_225_50000.mat | regenerated, production-array identity unverified |
| `pw14_172_225_100000` | 100,000 | pw_samples_a3_172_225_100000.mat | regenerated, production-array identity unverified |

| | |
|---|---|
| kind | generated numerical assets; contain no survey or catalogue data |
| built by | `tools/build_sample_grid.py` |

**Identity caveat, applying to all three.** These reproduce the reference
*generator* bitwise, which is tested. None has been shown identical to any deployed
production array, because the historical grids came from a notebook whose QMC
engine was never seeded reproducibly. They must not support a production-equivalence
claim until the deployed arrays or their hashes are compared directly. "Refined"
(100k) describes a denser integration grid only — it establishes no identity and is
not by itself scientifically more accurate.

Cite Prochaska, Madau, O'Meara & Fumagalli (2014), arXiv:1402.0548, for the
column-density prior.

### `priors/dr9q_concordance`

Absorber-existence prior, stored as an exact step table (sorted quasar redshifts
plus a cumulative absorber count). Derived, not copied: it contains counts, not
catalogue rows.

| | |
|---|---|
| kind | derived summary statistic (53322 sightlines, 5633 hosting an absorber) |
| built by | `tools/build_prior_table.py` |
| equivalence | reproduces the reference prior exactly at 337531 probe redshifts |

Sources, both public:

* **quasar catalogue** — `catalog.mat`, sha256 `35ae81f16ae56efd…`
  from https://github.com/jibanCat/gp_dr12_trained
  Licence: MIT. Attribution: Ming-Feng Ho, gp_dr12_trained.
* **DLA / line-of-sight catalogue** — `BOSSLyaDR9_cat.txt`, sha256 `a5aeb90dbf97b997…`
  from http://data.sdss3.org/sas/dr9/boss/lya/cat/BOSSLyaDR9_cat.txt
  Licence: public SDSS-III DR9 data. Attribution: SDSS-III BOSS DR9 Lyman-alpha forest catalogue.

When using this prior, acknowledge the SDSS-III BOSS DR9 Lyman-alpha forest
catalogue per the SDSS-III data-use policy.

### `logo.jpg` — project artwork

| | |
|---|---|
| File | `logo.jpg` at the repository root |
| Dimensions | 1254 × 1254 pixels, JPEG |
| SHA-256 | `10590b53395cfaf58aadd5c8295493c3f0453f5ca9918484bf2e689008b2fc28` |
| Supplied by | the PI, as project artwork |
| Used for | the README banner and documentation; not shipped in the wheel or sdist |
| Origin | generated by Ming-Feng Ho using GPT-5.6-Sol, with substantial manual adjustment, iteration and artistic direction by Ming-Feng Ho |
| Ownership | Ming-Feng Ho |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Public use | approved by the PI for use and distribution as the project logo in this repository and its documentation |

**Provenance.** The image was generated by Ming-Feng Ho using GPT-5.6-Sol and
then substantially adjusted and iterated by hand under his artistic direction.
It is not a stock asset, and OpenAI did not supply a finished project logo —
the generation was one step in the PI's own process.

The PI has approved its inclusion and public distribution as the project logo
here and in the documentation. It is referenced by relative path from
`README.md` and is deliberately **not** included in the built distributions —
it is presentation, not a package asset.

**Licence.**

> The `gp_dla_finder` logo is © Ming-Feng Ho and is licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It may be reused
> with attribution to Ming-Feng Ho. It is **not a trademark**, the licence
> grants no trademark rights, and it must not be used in a way that implies
> endorsement by the project or its authors.

This applies to `logo.jpg` and to every generated variant — light, dark and
universal, raster and SVG alike. The licence covers the artwork only; the
package code is under the repository's own licence, which is a separate grant.

**Derived variants.** `docs/_static/logo-light.png`,
`docs/_static/logo-dark.png`, `docs/_static/logo-universal.png` and
`docs/_static/logo-universal.svg` are generated from `logo.jpg` by
`tools/make_logo_variants.py`. The universal variant is the `<img>` fallback:
GitHub's mobile app ignores `<picture>` media queries, so a theme-specific
fallback rendered dark text on a dark page there. All key out the near-white
background so the artwork does not render as a bright square on a dark page; the
dark variant additionally lightens the dark strokes, preserving hue, so the
title and subtitle stay legible. The original is never modified and remains canonical. The derived
files inherit the original's ownership, its CC BY 4.0 licence, and the PI's
public-use approval, all recorded above.

## Assets not yet bundled

All assets required for single-spectrum inference are now packaged. Public example
spectra for the tutorial and golden-parity tests remain pending their separate
redistribution checks.

## Release checks

The attribution and redistribution terms for each bundled asset are recorded
above, separately from the MIT licence on the source code. The release checks
also scan the current tree and built distributions before publication:

* `tests/test_asset_hygiene.py` scans the built wheel and sdist byte-wise for
  private paths, and checks packaged model provenance carries none;
* packaged model provenance uses an allowlist and withholds path-like training
  attributes;
* no private spectrum, catalogue or reconstructable 2LPT mock product is
  included in the repository or its distributions.
