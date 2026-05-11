# Maps to power spectra

This step estimates the auto- and cross-spectra of the component-separated
maps using the pseudo-$C_\ell$ formalism via
[NaMaster](https://github.com/LSSTDESC/NaMaster).

## Pseudo-$C_\ell$ recap

Multiplication by the analysis mask $W(\hat n)$ couples multipoles. The
masked-sky pseudo-power spectrum $\widetilde{C}_\ell$ relates to the true
$C_\ell$ via

$$
\langle \widetilde{C}_\ell^{XY} \rangle \;=\; \sum_{\ell'} M^{XY}_{\ell\ell'} \, B^X_{\ell'} B^Y_{\ell'} \, p^2_{\ell'} \, C_{\ell'}^{XY},
$$

where $M_{\ell\ell'}$ is the mode-coupling matrix determined by $W$, $B_\ell$
is the (common) beam, and $p_\ell$ is the HEALPix pixel window. NaMaster
deconvolves $M_{\ell\ell'}$ at the binned level: it actually inverts the
binned coupling matrix, so the output is a debiased binned spectrum, not an
unbiased per-$\ell$ spectrum.

## Workspace

The mode-coupling workspace is built once per run from:

- the analysis mask (sets $M_{\ell\ell'}$),
- the **effective common beam** $B_\ell\, p_\ell$ from preprocessing,
- the binning from [binning](binning.md),
- the purification flags `map2cl_pars.purify_e` and `map2cl_pars.purify_b`,
- the SHT iteration count `map2cl_pars.n_iter_namaster` (typically 3).

Reusing one workspace across realisations is what makes the per-sim spectrum
cost cheap; the expensive coupling-matrix computation is paid only once.

## $B$-mode purification

`purify_b: true` removes ambiguous modes (linear combinations that are pure
$E$ on the full sky but project onto $B$ on the masked sky) by constructing
the pure-$B$ estimator of Smith (2006) inside NaMaster. This is essential for
ground-based $B$-mode searches where masked-sky $E$-to-$B$ leakage dominates
the unprocessed estimator at $\ell \lesssim 100$. Purification requires an
analysis mask whose first and second derivatives vanish at the edge &mdash;
hence the apodisation set up in [mask](mask.md).

## What is computed

For each pair of input maps the pipeline computes the binned, mode-coupling
deconvolved spectrum

$$
\widehat{C}_b^{XY}, \qquad XY \in \{TT, TE, EE, EB, BE, BB\},
$$

though the cosmological analysis only uses $BB$ of the CMB-channel
auto-spectrum. The output is restricted to the analysis range via
`limit_namaster_output`.

## Beam treatment

The effective beam includes the common beam $B_{\rm c}(\ell)$ from
preprocessing and the pixel window $p_\ell$ at the analysis $n_{\rm side}$.
NaMaster's beam deconvolution is applied inside the workspace; the recovered
$\widehat{C}_b$ is therefore already corrected for beam and pixel window.

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
