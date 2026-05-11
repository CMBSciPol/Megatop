# Cosmological parameters

This is the final step: it fits a model to the recovered $BB$ spectrum to
estimate the tensor-to-scalar ratio $r$ and (jointly) foreground-residual
amplitudes.

## Model

The pipeline fits the binned CMB-channel $BB$ auto-spectrum with the model

$$
C_b^{BB,{\rm model}}(\theta) \;=\; r \, C_b^{BB,{\rm prim}|r=1} \;+\;
A_{\rm lens} \, C_b^{BB,{\rm lens}|A=1} \;+\;
A_{\rm dust} \, \widehat{C}_b^{D \times D} \;+\;
A_{\rm sync} \, \widehat{C}_b^{S \times S} \;+\;
\overline{N}_b,
$$

where:

- $C_b^{BB,{\rm prim}|r=1}$ is the primordial tensor template from
  [binning](binning.md) (CAMB unlensed scalar+tensor at $r = 1$);
- $C_b^{BB,{\rm lens}|A=1}$ is the lensing-$B$ template (CAMB lensed scalar);
- $\widehat{C}_b^{D \times D}$, $\widehat{C}_b^{S \times S}$ are the residual
  dust and synchrotron auto-spectra of the **component-separated foreground
  maps** (from [map2cl](map2cl.md)) &mdash; used as templates because their
  shape captures the leakage from compsep errors;
- $\overline{N}_b$ is the mean residual noise on the CMB channel from
  [noise_spectra](noise_spectra.md).

Parameters $A_{\rm dust}$ and $A_{\rm sync}$ can be fixed to zero by setting
`dust_marg` / `sync_marg` to false; otherwise they are sampled jointly with
$(r, A_{\rm lens})$.

## Likelihood

The likelihood is the same form used in Wolz et al. (2023) &mdash; the
single-mode approximation to the Wishart / Hamimeche-Lewis likelihood:

$$
-2 \ln \mathcal{L}(\theta) \;=\; \sum_b (2 \ell_{\rm eff,b} + 1)\, f_{\rm sky}\, \Delta\ell_b
\Bigg[\frac{\widehat{C}_b}{C_b^{\rm model}(\theta)} + \ln C_b^{\rm model}(\theta)\Bigg],
$$

with $\ell_{\rm eff,b}$ and $\Delta\ell_b$ from the binning and $f_{\rm sky}$
from the analysis mask. The Wishart form is appropriate for a single auto-
spectrum at moderate $f_{\rm sky}$; it is positively biased for $C^{\rm model}$
near zero, so the prior must keep the model spectrum positive (the code logs
a warning when it sees negative bins inside the analysis range).

The analysis multipole range is restricted to
`[cl2r_pars.lmin_cosmo_analysis, cl2r_pars.lmax_cosmo_analysis]`, typically
narrower than the spectrum-estimation range.

## Sampler

Sampling is done with [emcee](https://emcee.readthedocs.io) (`EnsembleSampler`):

- `n_walkers` walkers initialised from a small ball around the prior centre;
- `n_steps_burnin` burn-in steps discarded;
- `n_steps` production steps stored.

Flat priors are applied via `prior_bounds`; outside the bounds
$\ln \mathcal{L}$ is set to $-\infty$.

## Outputs

- The full chain (parameters $\times$ steps $\times$ walkers).
- A summary file with marginalised mean/median/credible intervals per
  parameter.
- Diagnostic plots: chain traces, corner plot, and model-vs-data spectrum
  overlays (in the `plot_cl2r` / `plot_mcmc` Snakemake rules).

## Relevant configuration

```yaml
cl2r_pars:
  # analysis range (overrides general_pars.lmin/lmax for the likelihood)
  lmin_cosmo_analysis: null        # null => no lower cut beyond binning
  lmax_cosmo_analysis: 180

  # model freedom on foreground residuals
  dust_marg: false                 # if true, A_dust is sampled
  sync_marg: false                 # if true, A_sync is sampled
  load_model_spectra: true         # reuse cached primordial+lensing templates

  # MCMC settings
  n_walkers: 200
  n_steps_burnin: 2000
  n_steps: 10000

  # flat priors; bounds for parameters not sampled are ignored
  prior_bounds:
    r:         [-0.1, 0.1]
    A_{lens}:  [0.0, 10.0]
    A_{dust}:  [0.0, 1.0]
    A_{sync}:  [0.0, 1.0]
```

The likelihood reads the fiducial templates produced by the
[binning](binning.md) step, the residual foreground spectra from
[map2cl](map2cl.md), and the mean noise spectrum from
[noise_spectra](noise_spectra.md).
