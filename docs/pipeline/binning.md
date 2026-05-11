# Binning

This step sets up the multipole binning used by every spectrum estimator in
the pipeline and produces the fiducial CMB $C_\ell$ that other stages compare
against.

## Multipole binning

A `NaMaster` binning object is built from the configuration:

- `general_pars.lmin` and `general_pars.lmax` define the analysis range;
- `map2cl_pars.delta_ell` sets the bin width $\Delta\ell$ (linear bins by
  default; weighting inside a bin is $1/(2\ell+1)$).

The same binning is used throughout: spectrum estimation, noise spectra,
transfer function, and the cosmological likelihood, so all quantities are
directly comparable bin-by-bin.

## Fiducial CMB spectra

When `fiducial_cmb.compute_from_camb` is true, CAMB is run twice with the
cosmological parameters in `camb_cosmo_pars`:

1. **Unlensed scalar + tensor at $r = 1$** &mdash; this gives a primordial
   $C_\ell^{BB,{\rm prim}}\big|_{r=1}$ template that the likelihood later
   rescales by $r$.
2. **Lensed scalar at $r = 0$** &mdash; this gives the lensing-$B$ template
   that the likelihood rescales by $A_{\rm lens}$.

The two templates are stored on disk and read back by the mock generator and
by the cosmological estimator. Splitting the model in this way makes the
linear dependence on $(r, A_{\rm lens})$ explicit at evaluation time, so each
MCMC step costs only two scalar multiplications instead of a CAMB call.

## Relevant configuration

```yaml
general_pars:
  lmin: 30                         # lower edge of the analysis range
  lmax: 500                        # upper edge (must satisfy lmax <= 2*nside)

map2cl_pars:
  delta_ell: 10                    # width of uniform bins; can also be a list
  uniform_start: null              # if set: first bin [2, uniform_start-1],
                                   #         uniform delta_ell bins above

fiducial_cmb:
  compute_from_camb: true          # else provide paths below
  fiducial_lensed_scalar: null
  fiducial_unlensed_scalar_tensor_r1: null
  camb_cosmo_pars:
    H0: 67.5
    ombh2: 0.022
    omch2: 0.122
    tau: 0.06
    As: 2.0e-9
    ns: 0.965
    # extra_args: { ... }          # passed through to camb.set_params
```
