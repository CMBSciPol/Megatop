# Noise spectra

This step propagates the instrument noise model through the **full pipeline**
&mdash; preprocessing, component separation, and pseudo-$C_\ell$ estimation &mdash;
and returns the mean residual noise power $N_\ell$ on the recovered CMB
channel. This is the quantity the cosmological likelihood subtracts as a
noise bias and uses in its variance.

## Why a separate pipeline pass

A naive analytic noise model would miss several effects that show up only at
the spectrum level:

- the **mixing weights** $\big(\mathbf{A}^{\!\top} N^{-1} \mathbf{A}\big)^{-1}
  \mathbf{A}^{\!\top} N^{-1}$ from component separation, which redistribute
  per-frequency noise into the CMB channel and depend on the best-fit
  spectral parameters;
- the **mode-coupling and purification** done by NaMaster on a cut sky;
- the **transfer-function** debiasing of the filter response.

By running each noise-only realisation through exactly the same chain as the
signal, the resulting noise spectra are consistent by construction with the
estimator applied to the data.

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

The effective beam used for noise spectra is the **CMB-channel common beam**
&mdash; the same beam the component-separated CMB map carries. Other
components have, in principle, different effective beams after compsep
because the weights mix frequencies differently; only the CMB is needed at
this stage, so this simplification is fine.

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
