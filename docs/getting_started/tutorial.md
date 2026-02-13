# Tutorial and bird's eye view

## Creating a Default Configuration and Launching the Full MEGATOP Pipeline

This will allow you to test the pipeline and see the main steps. **This is however a minimal test case**. For instance, more noise simulations would be needed for the ouptut to make sense.

1. First let's create a configuration file using the default settings

```bash
python get_example_config.py
```

This will save the default configuration file in `./paramfiles/default_config.yaml`

2. A few parameters concerning simulations, outputs, fiducial CMB spectra paths must still be provided:

| Location in YAML    | Description                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| `data_dirs.root`    | Path for simulations to be be saved in.                                                                        |
| `output_dirs.root`  | Path for outputs to be saved in (optional, if not changed it will save outputs where the bash file is running) |
| `fiducial_cmb.root` | Path where to find fiducial CMB spectra                                                                        |

Concerning `fiducial_cmb.root` it is expected to host `lensed_scalar_cl.fits` and `unlensed_scalar_tensor_r1_cl.fits` providing respectively lensed scalar CMB spectra (both in temperature and polarization) and the unlensed tensor CMB spectra with `r=1`. Those file names can also be renamed in the `default_config.yaml` if necessary.

3. Then we can run the full pipeline using:

```bash
bash runfiles/run_default.sh
```

**WARNING:** `run_default.sh` is expecting a conda environment named `megatop` to be present where MEGATOP is installed.

A few notes about this:

- This default config will only generate one sky and more importantly one noise simulations, don't expect the results to make sense.
- It uses `nside=512` which might imply somewhat slow execution and "largish" memory use. Feel free to modify `general_pars.nside` in the config file, `general_pars.lmax` must be modified accordingly.
