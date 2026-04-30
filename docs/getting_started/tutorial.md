# Tutorial and bird's eye view

## Creating a Default Configuration and Launching the Full MEGATOP Pipeline

This will allow you to test the pipeline and see the main steps. **This is however a minimal test case**. For instance, more noise simulations would be needed for the ouptut to make sense.

1. First let's create a configuration file using the default settings

```bash
python get_example_config.py
```

This will save the default configuration file in `./paramfiles/default_config.yaml`

2. A few parameters concerning simulations and outputs must still be provided:

| Location in YAML  | Description                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `data_dirs.root`  | Path where simulations are saved.                                                                                |
| `output_dirs.root`| Path where pipeline outputs are saved (defaults to current directory if not changed).                            |

By default the config uses `fiducial_cmb.compute_from_camb: true`, which computes fiducial CMB spectra on the fly via CAMB.
If you prefer to provide precomputed spectra files, set `compute_from_camb: false` and point `fiducial_cmb.fiducial_lensed_scalar` and `fiducial_cmb.fiducial_unlensed_scalar_tensor_r1` to your FITS files (lensed scalar spectra and unlensed tensor spectra with `r=1` respectively).

3. Then run the full pipeline. Two approaches are available:

### Option A — Snakemake (recommended)

Snakemake is an optional dependency (requires Python >= 3.11), installed via `pip install megatop[snake]`. It handles step ordering, parallelism, and skips steps whose outputs already exist.

```bash
snakemake --cores 4 --configfile paramfiles/default_config.yaml
```

To preview which steps will run without executing them:

```bash
snakemake --cores 4 --configfile paramfiles/default_config.yaml --dry-run
```

On a SLURM allocation, replace `4` with `$SLURM_CPUS_PER_TASK`.

### Option B — Bash script (legacy)

```bash
bash runfiles/run_default.sh
```

**WARNING:** `run_default.sh` expects a conda environment named `megatop` where Megatop is installed.

A few notes:

- This default config generates only one sky and one noise simulation — results will not be statistically meaningful.
- It uses `nside=512`, which implies somewhat slow execution and sizeable memory use. Feel free to lower `general_pars.nside` in the config file; `general_pars.lmax` must be adjusted accordingly.
