# Noise covariance

This step estimates the **per-pixel noise variance** used as the weighting
matrix $N^{-1}$ inside pixel-domain component separation, and (optionally) the
**harmonic-space noise spectrum** $N_\ell$ used by the harmonic compsep path
and by the cosmological likelihood.

It is fanned out by Snakemake into one job per noise realisation followed by
an aggregator.

## Per-realisation step (`noise_preproc`)

For each noise realisation $i$:

1. Read the simulated noise maps at native beam (one per frequency).
2. Apply the same common-beam / $n_{\rm side}$ correction as the signal
   preprocessor &mdash; this is essential, otherwise the noise covariance
   would not correspond to the maps actually fed to component separation.
3. Save the preprocessed noise maps to disk.
4. If `use_harmonic_compsep` is enabled, run NaMaster on the noise map to
   obtain the auto-spectra, then **deconvolve the transfer function** with
   $T_b^{-1/2}$ applied symmetrically to the $\{EE, EB, BE, BB\}$ block.
   The symmetric application is what keeps the resulting covariance
   positive-definite once realisations are squared and averaged.

## Aggregator (`noisecov`)

Streams the per-realisation outputs and computes:

$$
\sigma^2_\nu \;=\; \frac{1}{N_{\rm sim}} \sum_i \big[m_\nu^{(i,n)}\big]^2,
\qquad
N_\ell^{\nu\nu} \;=\; \frac{1}{N_{\rm sim}} \sum_i \widetilde{N}_\ell^{(i),\nu\nu},
$$

i.e. the per-pixel variance map (per frequency, per Stokes component) and the
mean noise spectrum (per frequency, auto only). Cross-frequency noise is
assumed zero, which is consistent with detector-noise dominated experiments
where independent receivers observe different bands.

In **real-data mode** (`n_sim = None`), a single placeholder iteration is run
&mdash; the user is expected to supply an externally measured covariance or
use jackknife noise sims.

## Use downstream

- $\sigma^2_\nu$ enters the pixel `fgbuster` likelihood as the diagonal of
  $N^{-1}$, giving each pixel weight $\propto 1/\sigma^2$.
- $N_\ell$ is inverted (with a regularisation floor at $\ell <$
  `parametric_sep_pars.harmonic_lmin`) and used as the harmonic-space noise
  weighting in the harmonic compsep path.
- The same $N_\ell$ (after component separation has propagated it into the
  CMB channel) is what the cosmological likelihood debiases the recovered
  $\widehat{C}_\ell^{BB,{\rm CMB}}$ with.

## Relevant configuration

```yaml
noise_sim_pars:
  n_sim: 20                        # number of noise realisations averaged
                                   # (null => real-data mode, single pass)
  include_nhits: true              # rescale per-pixel by sqrt(<nhits>/nhits)

pre_proc_pars:
  common_beam_correction: 100.0    # same beam used on signal preproc

map2cl_pars:
  delta_ell: 10                    # binning used for the harmonic N_ell
  purify_e: false
  purify_b: true
  n_iter_namaster: 3

parametric_sep_pars:
  use_harmonic_compsep: false      # if true, also compute the binned N_ell
  harmonic_lmin: 30                # regularisation floor for 1/N_ell

general_pars:
  nside: 256
  lmax: 500
```

Output paths follow `output_dirs.covar` and the per-frequency noise file
naming derived from `map_sets[i].noise_prefix`.
