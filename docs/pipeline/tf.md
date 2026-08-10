# Transfer function

Map-making and filtering (TOD filters, ground-pickup removal, ...) bias the
recovered power spectrum in a multipole- and polarisation-dependent way. The
**transfer function** quantifies this bias so it can be deconvolved from the
data spectra and folded into the noise model.

## Definition

For each pair of $(p, q) \in \{E, B\}$ and each multipole bin $b$, the
transfer function is a $4 \times 4$ matrix $T_b$ relating input and recovered
binned spectra,

$$
\langle \widetilde{C}_b^{pq} \rangle \;=\; \sum_{rs} T_b^{\,pq, rs} \, C_b^{rs,{\rm in}},
$$

estimated from many filtered Gaussian simulations of pure-$E$ and pure-$B$
skies. Off-diagonal entries encode $E$/$B$ leakage induced by the filter on
top of the leakage already absorbed by the mode-coupling kernel.

## How Megatop computes it

Megatop does not implement the TF estimator itself; it delegates to
[**SOOPERCOOL**](https://github.com/simonsobs/SOOPERCOOL). The interface in
`TF_computation_interface` does three things:

1. Writes a SOOPERCOOL YAML config derived from the Megatop config (matching
   the analysis mask, $n_{\rm side}$, binning, purification flags, and the
   user-supplied filter description).
2. Calls SOOPERCOOL as a subprocess to generate the filtered simulations,
   estimate the mode-coupling-deconvolved pseudo-$C_\ell$, and solve for
   $T_b$.
3. Stores the resulting transfer function on disk in the Megatop output tree
   so that downstream stages (preprocessing and noise covariance) can apply
   $T_b^{-1/2}$ to noise realisations and recover unbiased $C_\ell$ estimates.

## Scientific notes

- The TF depends on the filter, the mask, and the binning &mdash; it must be
  recomputed whenever any of these change.
- Off-diagonal $T_b$ entries capture residual $E\to B$ leakage not removed by
  purification.
- The TF is applied symmetrically through $T_b^{-1/2}$ when used to debias
  noise covariances (see [noise_covariance](noise_covariance.md)); this keeps
  the resulting covariance positive-definite.

## Relevant configuration

The TF step itself reads the analysis mask, binning, and purification flags
from the rest of the config; the dedicated knobs are the TF-simulation
parameters and the per-map-set observation-matrix paths.

```yaml
pre_proc_pars:
  correct_for_TF: false            # apply TF deconvolution in preprocessing
  sum_TF_column: true              # sum TF columns when contracting

map_sim_pars:
  filter_sims: false               # filter sky sims with the obsmat
  generate_sims_for_TF: true       # turn on TF-sim generation
  TF_n_sim: 100                    # number of TF realisations
  TF_power_law_amp: 1.0            # amplitude of the input power law
  TF_power_law_index: 2.0          # absolute value; minus sign added downstream
  TF_power_law_delta_ell: 1

map_sets:
  - exp_tag: SO-SAT
    freq_tag: 93
    obsmat_path: /path/to/obsmat.npz  # required when filter_sims is true
    TF_path: /path/to/TF.npz          # if pre-computed externally
```

The mask, binning, $n_{\rm side}$, and `map2cl_pars.purify_b` settings are
forwarded directly to SOOPERCOOL so the TF is consistent with the spectrum
estimator.
