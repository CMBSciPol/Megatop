<div align="center">
<img width="1024" height="283" alt="megatop_logo" src="https://github.com/user-attachments/assets/8fcb7af2-2a62-45d3-ab96-9ee68be1e4d0" />
</div>

# A CMB polarization data analysis pipeline

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/CMBSciPol/Megatop/actions/workflows/ci.yml/badge.svg)](https://github.com/CMBSciPol/Megatop/actions/workflows/ci.yml)

[**Docs**](https://megatop.readthedocs.io/en)

Megatop is a map-based data analysis pipeline for CMB polarization experiments.
It takes frequency maps and performs component separation and estimation of cosmological parameters such as the tensor-to-scalar ratio.

## Installation

`Megatop` is pip-installable. Simply clone the repo and run

```bash
pip install .
```

Head to [documentation](https://megatop.readthedocs.io/en) for more detailed instructions.

## Developing Megatop

Install in editable mode and include development dependencies with

```bash
pip install -e . --group dev
```
