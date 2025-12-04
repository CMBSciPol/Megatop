# Installation and setup

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

## Developing `Megatop`

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
