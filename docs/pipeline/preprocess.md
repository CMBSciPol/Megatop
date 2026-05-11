# Preprocessing

This step brings all frequency maps onto a **common resolution** so component
separation can combine them linearly. It runs once per sky realisation
(noise realisations are handled by a sibling step;
see [noise_covariance](noise_covariance.md)).

## Common-beam correction

Each frequency map is delivered at its native instrumental beam
$B_\nu(\ell)$ (FWHM from `map_sets[i].beam`). Pixel-domain component
separation assumes all frequencies share the same beam, so the maps are
reconvolved to a common Gaussian beam $B_{\rm c}(\ell)$ with FWHM
`pre_proc_pars.common_beam_correction`. In harmonic space,

$$
a_{\ell m}^{(\nu),{\rm c}} \;=\; \frac{B_{\rm c}(\ell)\, p_{\rm c}(\ell)}{B_\nu(\ell)\, p_\nu(\ell)} \; a_{\ell m}^{(\nu)},
$$

where $p(\ell)$ is the HEALPix pixel window function at the relevant
$n_{\rm side}$. The pixel window is included so the correction also accounts
for any change in $n_{\rm side}$ between input maps and the analysis grid.

The common beam must be **wider** than every native beam at all $\ell$ used
in the analysis &mdash; otherwise the deconvolution
$B_{\rm c}/B_\nu$ diverges and noise blows up. The pipeline raises if this
condition is not met for the requested `lmax`.

## Masking

After reconvolution the maps are multiplied by the binary mask so that pixels
outside the observation footprint are exactly zero. The apodised analysis
mask is **not** applied here &mdash; it is applied later inside NaMaster, which
needs the unmasked-within-footprint maps to compute pseudo-spectra correctly.

## Skip path

When the configured common beam matches every input beam (or
`pre_proc_pars.DEBUGskippreproc` is set) and harmonic-space component
separation is not requested, the step is short-circuited and the input maps
pass through untouched. This avoids an unnecessary harmonic round-trip and
its associated ringing.

## Harmonic-space variant

If `parametric_sep_pars.use_harmonic_compsep` is set, the preprocessor also
saves the spherical-harmonic coefficients $a_{\ell m}^{(\nu),{\rm c}}$ for use
by the harmonic component-separation path, where the mixing matrix is solved
directly on $C_\ell$ rather than on pixels.

## Relevant configuration

```yaml
general_pars:
  nside: 256                       # target Nside of the preprocessed maps
  lmax: 500                        # SHT lmax used for the reconvolution

pre_proc_pars:
  common_beam_correction: 100.0    # FWHM of the common Gaussian beam (arcmin)
  DEBUGskippreproc: false          # bypass reconvolution (debug only)
  correct_for_TF: false            # apply TF correction in preprocessing
  sum_TF_column: true

parametric_sep_pars:
  use_harmonic_compsep: false      # also save alm if true

map_sets:
  - exp_tag: SO-SAT
    freq_tag: 93
    beam: 30.0                     # native per-frequency FWHM (arcmin)
```

`common_beam_correction` must be larger than every `map_sets[i].beam` at the
analysis $\ell_{\max}$; otherwise the deconvolution diverges.
