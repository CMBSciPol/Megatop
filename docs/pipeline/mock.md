# Mocks

This step generates the simulated frequency maps that the rest of the pipeline
treats as data. Two kinds of realisations are produced, with separate seeds
and separate counts:

- **sky simulations** &mdash; CMB + foregrounds, varied across `map_sim_pars.n_sim`
  realisations;
- **noise simulations** &mdash; instrument noise only, varied across
  `noise_sim_pars.n_sim` realisations.

Keeping the two independent is what allows the pipeline to estimate signal-only
and noise-only spectra (and noise covariances) cleanly.

## Sky component

For each sky realisation the mocker builds, per frequency $\nu$,

$$
m_\nu \;=\; B_\nu \!\ast\!\Big[ s^{\rm CMB} + \sum_c s^{(c)}_\nu \Big],
$$

where $B_\nu$ is the per-frequency Gaussian beam (FWHM from
`map_sets[i].beam`).

- **CMB** &mdash; a Gaussian realisation drawn from the fiducial $C_\ell$ at
  the input tensor-to-scalar ratio `map_sim_pars.r_input` and lensing
  amplitude `map_sim_pars.A_lens`. With `cmb_sim_no_pysm: true` the CMB is
  drawn directly with `healpy.synfast` from the CAMB spectra; otherwise PySM's
  CMB model is used. `single_cmb` reuses one CMB realisation across all sky
  sims (useful for noise-only variance studies).
- **Foregrounds** &mdash; PySM models selected by `map_sim_pars.sky_model`
  (e.g. `d0`, `s0` for the simplest dust + synchrotron). Each frequency map
  is bandpass-integrated when `parametric_sep_pars.passband_int` is set, using
  the bandpasses in `passband.fgbuster_passband`.

## Noise component

For each noise realisation, per frequency:

1. A full-sky noise map is drawn from either the experiment-level noise model
   (`noise_option: noise_spectra`, using V3/V3.1 SO calculators with the
   chosen $1/f$ mode and sensitivity tier) or a white-noise level
   (`noise_option: white`).
2. If `include_nhits: true`, the map is rescaled pixel-by-pixel by
   $\sqrt{\langle n_{\rm hits}\rangle / n_{\rm hits}}$ inside the binary
   mask, so the noise variance tracks the relative hit count of the
   experiment.

Noise and sky are written separately to disk; the linear combination is taken
later in the preprocessing step.

## Seeds

Seeds are deterministic functions of the realisation index and the
configured base seeds (`cmb_seed`, `noise_sim_pars.seed`), so two pipeline
runs with the same configuration produce byte-identical simulations &mdash;
important for diagnosing bias vs. variance differences across pipeline
changes.

## Relevant configuration

```yaml
map_sim_pars:
  n_sim: 10                        # number of sky realisations
  r_input: 0.0                     # tensor-to-scalar ratio in the truth
  A_lens: 1.0                      # lensing amplitude in the truth
  cmb_sim_no_pysm: true            # draw CMB with healpy.synfast (else PySM)
  single_cmb: false                # if true, reuse one CMB across realisations
  cmb_seed: 1234                   # base CMB seed
  sky_model: [d0, s0]              # PySM foreground templates (d*, s*)
  filter_sims: false               # apply per-map-set observation matrix
  passband_int: false              # bandpass-integrate sky maps
  # TF-only flags (see TF step):
  generate_sims_for_TF: false
  TF_n_sim: 1
  TF_power_law_amp: 1.0
  TF_power_law_index: 2.0
  TF_power_law_delta_ell: 1

noise_sim_pars:
  n_sim: 20                        # number of noise realisations
  include_nhits: true              # rescale per pixel by sqrt(<nhits>/nhits)
  seed: 42                         # base noise seed
  experiments:
    SO-SAT:
      usev3p1: true                # use V3.1 noise calculator
      default_bands: [27, 39, 93, 145, 225, 280]
      noise_option: noise_spectra  # or white_noise / no_noise / noise_map
      v3_sensitivity_mode: GOAL    # THRESHOLD | BASELINE | GOAL
      v3_one_over_f_mode: OPTIMISTIC  # WHITE | OPTIMISTIC | PESSIMISTIC | ...
      Ntubes_years: [1.0, 9.0, 5.0]

# per-frequency beam used to convolve the sky before noise is added
map_sets:
  - exp_tag: SO-SAT
    freq_tag: 93
    beam: 30.0                     # FWHM in arcmin
    passband_filename: ""          # required if passband_int is true
    obsmat_path: null              # required if filter_sims is true
```
