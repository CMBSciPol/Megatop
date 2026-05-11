# Maps to power spectra

This step estimates the auto- and cross-spectra of the component-separated
maps using [NaMaster](https://github.com/LSSTDESC/NaMaster).

## Workspace

The mode-coupling workspace is built once per run from:

- the analysis mask (sets $M_{\ell\ell'}$),
- the **effective common beam** $B^{\rm c}_\ell\, w^{\rm c}_\ell$ from preprocessing,
- the binning from [binning](binning.md),
- the purification flags `map2cl_pars.purify_e` and `map2cl_pars.purify_b`,
- the SHT iteration count `map2cl_pars.n_iter_namaster` (typically 3).

Reusing one workspace across realisations keeps per-sim spectrum cost cheap;
the coupling-matrix inversion is paid only once.

## $B$-mode purification

`purify_b: true` constructs the pure-$B$ estimator of Smith (2006) inside
NaMaster. Requires an analysis mask whose first and second derivatives vanish
at the edge — hence the apodisation set up in [mask](mask.md). `purify_e`
and `purify_b` are mutually exclusive.

## What is computed

For each pair of input maps, the pipeline produces the binned,
mode-coupling-deconvolved, beam- and pixel-window-corrected spectrum

$$
\widehat{C}_b^{XY}, \qquad XY \in \{TT, TE, EE, EB, BE, BB\}.
$$

The cosmological analysis uses only the CMB-channel $BB$ auto-spectrum.
Output is restricted to the analysis range via `limit_namaster_output`.

## Relevant configuration

```yaml
general_pars:
  lmin: 30
  lmax: 500
  nside: 256

map2cl_pars:
  delta_ell: 10                    # uniform bin width (or list of widths)
  uniform_start: null              # see binning step
  purify_e: false                  # mutually exclusive with purify_b
  purify_b: true
  n_iter_namaster: 3               # SHT iterations inside map2alm

pre_proc_pars:
  common_beam_correction: 100.0    # FWHM (arcmin) of effective CMB beam
```

The same `map2cl_pars` block is reused by the noise-spectra estimator so that
signal and noise are processed with an identical workspace and beam.
