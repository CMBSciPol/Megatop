# Megatop

A map-based CMB polarization data analysis pipeline, from maps to tensor-to-scalar ratio estimation.

## Installation

First clone the repository, for example via https, and navigate to the directory:

```bash
git clone https://github.com/CMBSciPol/Megatop.git
cd Megatop
```

`Megatop` depends on `NaMaster`, which is typically installed either from [PyPI](https://pypi.org/project/pymaster/)
or [conda-forge](https://anaconda.org/conda-forge/namaster).
Check the [NaMaster documentation](https://namaster.readthedocs.io/en/latest/source/installation.html) for more information.

If you have the necessary dependencies to install `NaMaster`, you can simply install `Megatop` and its dependencies with pip:

```bash
pip install .
```

Otherwise, we recommend starting with a conda environment and pip-installing the rest of the dependencies:

```bash
conda create -y -p ./conda_env python=3.10 namaster
pip install .
```

If you intend to use functions calling MPI, you will need `mpi4py`, part of the `mpi` optional dependencies:

```bash
pip install .[mpi]
```

Refer to the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html) for more information.

## Developing Megatop

After cloning, install in editable mode and with development dependencies:

```bash
pip install -e .[dev]
```

For detailed installation instructions, refer to the previous section.

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
You can run the tests with:

```bash
pytest
```

To ensure that your code passes the quality checks,
you can use our [pre-commit](https://pre-commit.com/) configuration:

1. Install `pre-commit`, for example via

```bash
pip install pre-commit
```

2. Install the pre-commit hooks with

```bash
pre-commit install
```

3. That's it! Every commit will trigger the code quality checks.


## Creating a Default Configuration and Launching the Full MEGATOP Pipeline
This will allow you to test the pipeline and see the main steps. **This is however a minimal test case**. For instance, more noise simulations would be needed for the ouptut to make sense.
1. First let's create a configuration file using the default settings
```bash
python get_example_config.py
```
This will save the default configuration file in `./paramfiles/default_config.yaml`

2. A few parameters concerning simulations, outputs, fiducial CMB spectra paths must still be provided:

|Location in YAML | Description |
|-----------------|-------------------------|
| `data_dirs.root` | Path for simulations to be be saved in. |
| `output_dirs.root` | Path for outputs to be saved in (optional, if not changed it will save outputs where the bash file is running) |
| `fiducial_cmb.root` | Path where to find fiducial CMB spectra |

Concerning `fiducial_cmb.root` it is expected to host `lensed_scalar_cl.fits` and `unlensed_scalar_tensor_r1_cl.fits` providing respectively lensed scalar CMB spectra (both in temperature and polarization) and the unlensed tensor CMB spectra with `r=1`. Those file names can also be renamed in the `default_config.yaml` if necessary.

3. Then we can run the full pipeline using:

```bash
bash runfiles/run_default.sh
```
**WARNING:** `run_default.sh` is expecting a conda environment named `megatop` to be present where MEGATOP is installed.

A few notes about this:
- This default config will only generate one sky and more importantly one noise simulations, don't expect the results to make sense.
- It uses `nside=512` which might imply somewhat slow execution and "largish" memory use. Feel free to modify `general_pars.nside` in the config file, `general_pars.lmax` must be modified accordingly.
