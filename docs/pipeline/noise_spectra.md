# Noise spectra

This step propagates the instrument noise model through the **full pipeline**
&mdash; preprocessing, component separation, and pseudo-$C_\ell$ estimation &mdash;
and returns the mean residual noise power $N_\ell$ on the recovered CMB
channel. This is the quantity the cosmological likelihood subtracts as a
noise bias and uses in its variance.

Each noise realisation is pushed through the same preprocessing → compsep →
NaMaster chain as the signal. This automatically accounts for the compsep
mixing weights (which depend on the best-fit spectral parameters), the
mode-coupling and purification, and the transfer-function debiasing — effects
that an analytic $N_\ell$ estimate would miss.

## What this step does

For each noise realisation $i \in [0, N_{\rm noise})$:

1. Take the preprocessed noise maps from
   [noise_covariance](noise_covariance.md).
2. Apply the same component-separation operator (with the spectral parameters
   already estimated from the signal sim) to obtain a CMB-channel noise map.
3. Run NaMaster with the shared workspace to obtain
   $\widehat{N}_b^{(i),\,{\rm CMB \times CMB}}$ for $BB$ (and other auto-pairs
   for diagnostics).

The mean and dispersion across realisations give

$$
\overline{N}_b \;=\; \frac{1}{N_{\rm noise}} \sum_i \widehat{N}_b^{(i)},
\qquad
\Sigma_{b} \;\propto\; \mathrm{Var}_i \big[\widehat{N}_b^{(i)}\big],
$$

both consumed by the cosmological estimator.

## Beam handling

The workspace uses the **CMB-channel common beam** $B_{\rm c}(\ell)\,p_\ell$,
matching the beam of the component-separated CMB map.

## Relevant configuration

```yaml
noise_sim_pars:
  n_sim: 20                        # realisations propagated through the chain

map2cl_pars:
  delta_ell: 10                    # binning shared with map2cl
  purify_e: false
  purify_b: true
  n_iter_namaster: 3

pre_proc_pars:
  common_beam_correction: 100.0    # effective CMB beam after preproc

parametric_sep_pars:
  use_harmonic_compsep: false
  harmonic_lmin: 30

general_pars:
  lmax: 500
  nside: 256
```

This step does not introduce new knobs beyond those already used by
[map2cl](map2cl.md) and [noise_covariance](noise_covariance.md); consistency
of those blocks across steps is what guarantees the noise spectra used by the
likelihood match the signal estimator.
