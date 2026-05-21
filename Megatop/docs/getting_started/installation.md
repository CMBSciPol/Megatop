# Installation and setup

## Installation

You can install the latest development version of `Megatop` directly from GitHub.

```bash
pip install git+https://github.com/CMBSciPol/Megatop.git
```

!!! note

    Megatop depends on [NaMaster](https://github.com/LSSTDESC/NaMaster).
    The method above will compile that package on your machine, and assumes you have installed its dependencies.
    Otherwise, we suggest installing it from [conda-forge](https://anaconda.org/conda-forge/namaster) beforehand.

    Check the [NaMaster documentation](https://namaster.readthedocs.io/en/latest/source/installation.html) for more information.

Most of the pipeline steps use [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface).
To run those, you will need the `mpi4py` package, an optional dependency.

```bash
pip install megatop[mpi] @ git+https://github.com/CMBSciPol/Megatop.git
```

Refer to the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html) for more information.

## Development

You should clone the [repository](https://github.com/CMBSciPol/Megatop) and install in editable mode with the `dev` dependency group.

```bash
pip install -e .[mpi] --group dev
```

The `dev` group includes [pytest](https://docs.pytest.org/en/stable/) for testing.
You can simply run the tests with

```bash
pytest
```

### Code quality checks

Our pre-commit hooks run automatically via GitHub actions.
To run them locally, you can follow these steps.

1. Install `prek` following the [documentation](https://prek.j178.dev/installation/).
2. Install hooks with

    ```bash
    prek install
    ```

3. That's it! Every commit will trigger the code quality checks.

!!! tip
    You can run hooks on demand with `prek run`.
