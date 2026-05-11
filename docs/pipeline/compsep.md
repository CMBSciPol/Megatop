# Component separation

This step solves the multi-frequency mixing problem to isolate CMB,
synchrotron, and dust components from the preprocessed frequency maps. It
uses **parametric component separation** as implemented in
[`fgbuster`](https://github.com/fgbuster/fgbuster).

## Data model

After preprocessing, every frequency map sits at the same beam and the data
vector at pixel $p$ (or harmonic mode $\ell m$) is

$$
\mathbf{d}(p) \;=\; \mathbf{A}(\boldsymbol{\beta}) \, \mathbf{s}(p) + \mathbf{n}(p),
$$

with

- $\mathbf{d}$: stacked frequency maps ($N_\nu$ entries),
- $\mathbf{s}$: component amplitudes (CMB, dust, optionally synchrotron) at
  a reference frequency,
- $\mathbf{A}(\boldsymbol{\beta})$: $N_\nu \times N_c$ mixing matrix whose
  columns are the SEDs of each component evaluated at the experiment
  frequencies (and bandpass-integrated if `passband_int` is set),
- $\boldsymbol{\beta}$: spectral parameters $(\beta_{\rm d}, T_{\rm d},
  \beta_{\rm s}, \ldots)$ &mdash; constant over the sky (`d0/s0`) or pixel-
  dependent depending on `fgbuster` component classes.

The components used are `CMB`, `Dust` (modified blackbody), and optionally
`Synchrotron` (power law) &mdash; controlled by
`parametric_sep_pars.include_synchrotron`.

## Pixel-domain solver (default)

The spectral parameters are obtained by maximising the profile likelihood
over component amplitudes,

$$
\widehat{\boldsymbol{\beta}} \;=\; \arg\max_{\boldsymbol{\beta}} \;
\mathbf{d}^{\!\top} N^{-1} \mathbf{A} \big(\mathbf{A}^{\!\top} N^{-1} \mathbf{A}\big)^{-1} \mathbf{A}^{\!\top} N^{-1} \mathbf{d},
$$

with $N^{-1}$ the diagonal per-pixel inverse noise covariance from
[noise_covariance](noise_covariance.md). The minimiser is SciPy `minimize`
with method, tolerances and bounds from `parametric_sep_pars.minimize_*`.

Component amplitudes are then recovered by the standard GLS solution

$$
\widehat{\mathbf{s}}(p) \;=\; \big(\mathbf{A}^{\!\top} N^{-1} \mathbf{A}\big)^{-1} \mathbf{A}^{\!\top} N^{-1} \mathbf{d}(p).
$$

The recovered CMB amplitude map is what enters the spectrum estimator
downstream.

## Harmonic-domain solver

When `use_harmonic_compsep` is set, the same likelihood is evaluated on the
$a_{\ell m}$ produced by the preprocessor with the harmonic noise weighting
$N_\ell^{-1}$ from [noise_covariance](noise_covariance.md). This avoids
the pixel-noise approximation and naturally accommodates correlated noise,
at the cost of relying on the transfer-function deconvolution being clean.
A minimum multipole `parametric_sep_pars.harmonic_lmin` is enforced to avoid
the regime where the noise spectrum estimate is unreliable.

## Outputs

- Best-fit spectral parameters $\widehat{\boldsymbol{\beta}}$ and their
  estimated covariance from the Hessian at the minimum.
- Component amplitude maps $\widehat{\mathbf{s}}(p)$ &mdash; in particular the
  CMB-channel map used for the cosmological pipeline.
- Auxiliary products (e.g. residual maps) used for plotting and diagnostics.

## Caveats

- Errors in $\widehat{\boldsymbol{\beta}}$ translate into **multiplicative
  bias** on $\widehat{\mathbf{s}}^{\rm CMB}$ and produce non-Gaussian
  foreground residuals at the spectrum level; the likelihood downstream
  partially absorbs this through the dust template amplitude $A_{\rm dust}$.
- The fit is unconstrained by default (no priors on $\boldsymbol{\beta}$);
  pathological frequency configurations can produce mixing-matrix
  degeneracies that the optimiser handles poorly.

## Relevant configuration

```yaml
parametric_sep_pars:
  include_synchrotron: true        # add Synchrotron column to the mixing matrix
  passband_int: false              # bandpass-integrate the SEDs

  # solver
  minimize_method: TNC             # any scipy.optimize.minimize method
  minimize_tol: 1.0e-18
  minimize_options:
    maxiter: 200                   # renamed to maxfun for TNC internally
    disp: false
    ftol: 1.0e-12
    gtol: 1.0e-12
    eps: 1.0e-12

  # harmonic-domain variant
  use_harmonic_compsep: false
  harmonic_lmin: 30
  harmonic_lmax: 256               # defaults to 2 * 128
  harmonic_delta_ell: 10
  alm2map: false                   # synthesize maps back from compsep alm
```

Each `map_sets[i].passband_filename` must be set when `passband_int: true`;
the SED of every component is integrated against the provided bandpass before
entering $\mathbf{A}(\boldsymbol{\beta})$.
