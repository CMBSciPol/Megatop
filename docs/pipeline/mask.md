# Masks

This step builds the sky masks that define the analysis footprint, downweight
poorly observed pixels, and (optionally) suppress Galactic emission and point
sources. All downstream estimators take these masks as inputs, so getting them
right is critical: they set the effective sky fraction $f_{\rm sky}$, govern
the mode-coupling matrix used to deconvolve the pseudo-$C_\ell$, and control
$E$-to-$B$ leakage when purification is applied.

## Inputs

Per map set, the user supplies either a hit-count map $n_{\rm hits}(p)$ or a
white-noise depth map $\sigma(p)$ (in $\mu$K-arcmin). Optionally:

- a Planck Galactic plane mask (auto-downloaded if missing) at a chosen sky
  fraction (`gal_key`, e.g. `GAL060`);
- a point-source mask, either provided or mocked from random hole positions.

## What the step does

1. **Normalised hit map** &mdash; each per-frequency $n_{\rm hits}$ (or
   $\sigma^{-2}$ derived from depth maps) is smoothed with a Gaussian of FWHM
   `masks_pars.fwhm_arcmin_smooth_nhits` and divided by its maximum to give a
   dimensionless coverage map in $[0, 1]$.
2. **Common hit map** &mdash; the geometric mean of all per-frequency
   normalised maps. The geometric mean (rather than arithmetic) penalises
   frequencies with poor coverage, ensuring the footprint is the *intersection*
   of well-observed regions across the array.
3. **Binary mask** &mdash; the common hit map thresholded at
   `binary_mask_zero_threshold`. This defines the support of the observation;
   pixels below threshold are exactly zero in all downstream steps.
4. **Analysis (apodised) mask** &mdash; the binary mask, optionally multiplied
   by the Galactic and point-source masks, then apodised with the chosen
   scheme (`C1`/`C2`/Smooth) at radius `apod_radius` (and
   `apod_radius_point_source` for sources). Apodisation reduces ringing in the
   mode-coupling kernel and is essential for $B$-mode purification.

## Diagnostics

`mask_checker` reports:

- $f_{\rm sky}$ for the hit map, the binary mask, and the analysis mask;
- the first and second spin-derivatives of the analysis mask &mdash; these
  enter the $E$/$B$ purification operator and must be finite and well-behaved
  for purification to remove ambiguous modes correctly.

It also generates a small set of purified pure-$B$ simulations to verify that
the chosen apodisation does not leak unacceptably between $E$ and $B$ at the
multipoles of interest.

## Why it matters

The mode-coupling kernel $M_{\ell \ell'}$ used by NaMaster depends entirely on
the analysis mask. Insufficient apodisation inflates the kernel off-diagonal
and biases the deconvolved power; aggressive apodisation reduces effective
sky area and inflates sample variance. The default `C1` apodisation at
$\sim 10^\circ$ is the typical compromise used in Simons Observatory SAT-like
analyses.

## Relevant configuration

```yaml
general_pars:
  nside: 256                       # HEALPix Nside of the analysis grid

masks_pars:
  # output names
  nhits_map_name: common_nhits_map
  binary_mask_name: common_binary_mask
  analysis_mask_name: common_analysis_mask

  # hit-map smoothing and binary thresholding
  fwhm_arcmin_smooth_nhits: 60     # Gaussian FWHM for the n_hits smoothing
  binary_mask_zero_threshold: 1e-1 # cut on the normalised common hit map

  # apodisation
  apod_type: C1                    # one of C1, C2, Smooth
  apod_radius: 10.0                # degrees
  apod_radius_point_source: 4.0    # degrees

  # optional masks
  include_galactic: false
  gal_key: GAL060                  # Planck galactic-plane fraction
  galactic_mask_name: galactic_mask
  include_sources: false
  input_sources_mask: null         # path to user-provided source mask
  sources_mask_name: sources_mask
  mock_nsources: 100               # used only if input_sources_mask is null
  mock_sources_hole_radius: 4.0    # arcmin

# per map set: either nhits_map_path or depth_map_path must be given
map_sets:
  - exp_tag: SO-SAT
    freq_tag: 93
    beam: 30.0
    nhits_map_path: SO_nominal     # or a path; "SO_nominal" triggers built-in
    # depth_map_path: ...          # alternative: white-noise depth map
```

Megatop forbids `purify_e: true` and `purify_b: true` simultaneously, so the
apodisation requirements are dominated by the purified channel (here $B$).
