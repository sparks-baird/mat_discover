<!-- TODO: add buttons for code ocean and Zenodo DOI [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/3904426/tree)-->
# DiSCoVeR
[![Open In Colab (PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MgV_ZewS6gLm1a3Vyhg33pFHi5uTld_2?usp=sharing)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sparks-baird/mat_discover/main?labpath=mat_discover_pypi.ipynb)
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://doi.org/10.24433/CO.8463578.v1)

[![PyPI version](https://img.shields.io/pypi/v/mat_discover.svg)](https://pypi.org/project/mat_discover/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mat_discover?label=PyPI%20downloads)

![Conda](https://img.shields.io/conda/v/sgbaird/mat_discover)
![Conda](https://img.shields.io/conda/pn/sgbaird/mat_discover)
[![Anaconda-Server Downloads](https://anaconda.org/sgbaird/mat_discover/badges/downloads.svg)](https://anaconda.org/sgbaird/mat_discover)
[![Anaconda-Server Badge](https://anaconda.org/sgbaird/mat_discover/badges/latest_release_relative_date.svg)](https://anaconda.org/sgbaird/mat_discover)
<!-- ![Conda](https://img.shields.io/conda/dn/sgbaird/mat_discover) -->
<!-- [![Anaconda-Server Downloads](https://anaconda.org/sgbaird/mat_discover/badges/downloads.svg)](https://anaconda.org/sgbaird/mat_discover) -->

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) <!-- ![Coveralls](https://img.shields.io/coveralls/github/sparks-baird/mat_discover) -->
[![Coverage Status](https://coveralls.io/repos/github/sparks-baird/mat_discover/badge.svg?branch=main)](https://coveralls.io/github/sparks-baird/mat_discover?branch=main)
![Lines of code](https://img.shields.io/tokei/lines/github/sparks-baird/mat_discover)
![GitHub](https://img.shields.io/github/license/sparks-baird/mat_discover)
[![DOI](https://zenodo.org/badge/392897894.svg)](https://zenodo.org/badge/latestdoi/392897894)
<!-- ![PyPI - License](https://img.shields.io/pypi/l/mat_discover) -->

A materials discovery algorithm geared towards exploring high performance candidates in new chemical spaces using composition-only.

<img src=https://user-images.githubusercontent.com/45469701/139520031-bf4fda18-9be7-4c54-b70b-c9be8e974cea.png width=500>  
<sup>Bulk modulus values overlaid on DensMAP densities (cropped).</sup>

## Citing
The preprint is hosted on ChemRxiv:
> Baird S, Diep T, Sparks T. DiSCoVeR: a Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions. ChemRxiv 2021. [doi:10.33774/chemrxiv-2021-5l2f8-v2](https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8). This content is a preprint and has not been peer-reviewed.

The BibTeX citation is as follows:
```bib
@article{baird_diep_sparks_2021,
place={Cambridge},
title={DiSCoVeR: a Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions},
DOI={10.33774/chemrxiv-2021-5l2f8-v2},
journal={ChemRxiv},
publisher={Cambridge Open Engage},
author={Baird, Sterling and Diep, Tran and Sparks, Taylor},
year={2021}
}
```

## DiSCoVeR Workflow
<img src="https://sparks-baird.github.io/mat_discover/figures/discover-workflow.png" alt="DiSCoVeR Workflow" width=600>

<sup>Figure 1. DiSCoVeR workflow to create chemically homogeneous clusters.  (a) Training and validation data.  (b) ElMD pairwise distances.  (c) DensMAP embeddings and DensMAP densities.  (d) Clustering via HDBSCAN*.  (e) Pareto plot and discovery scores.  (f) Pareto plot of cluster properties</sup>

## Installation
I recommend that you run `mat_discover` in a separate conda environment, at least for initial testing. After installing [Anaconda](https://docs.anaconda.com/anaconda/navigator/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), you can [create a new environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) via:
```python
conda create --name mat_discover
```
There are three ways to install `mat_discover`: Anaconda (`conda`), PyPI (`pip`), and from source. Anaconda is the preferred method.
### Anaconda
To install `mat_discover` using `conda`, first, update `conda` via:
```python
conda update conda
```
The Anaconda `mat_discover` package is hosted on the [@sgbaird channel](https://anaconda.org/sgbaird/repo) and can be installed via:
```python
conda install -c sgbaird mat_discover
```
### Pip
To install via `pip`, first update `pip` via:
```python
pip install -U pip
```

Due to limitations of PyPI distributions of CUDA/PyTorch, you will need to install PyTorch separately via the command that's most relevant to you ([PyTorch Getting Started](https://pytorch.org/get-started/locally/)). For example, for Stable/Windows/Pip/Python/CUDA-11.3:
```python
pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
<!--- ```python
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
``` --->

Finally, install `mat_discover`:
```python
pip install mat_discover
```
### From Source
To install from source, clone the `mat_discover` repository:

```python
git clone --recurse-submodules https://github.com/sparks-baird/mat_discover.git
cd mat_discover
```

To perform the local installation, you can use `pip`, `conda`, or `flit`:
| **pip**            | **conda**                                 | **flit**                           |
| ------------------ | ----------------------------------------- | ---------------------------------- |
| `pip install -e .` | `conda env create --file environment.yml` | `conda install flit; flit install` |

<!-- conda install torch cudatoolkit=11.1 -c pytorch -c conda-forge # or use pip command specific to you from https://pytorch.org/get-started/locally/ -->

## Basic Usage
### Fit/Predict
```python
from mat_discover.mat_discover_ import Discover
disc = Discover()
disc.fit(train_df) # DataFrames should have at minimum "formula" and "target" columns
scores = disc.predict(val_df)
disc.plot()
disc.save()
print(disc.dens_score_df.head(10), disc.peak_score_df.head(10))
```

See [mat_discover_example.py](mat_discover_example.py), [![Open In Colab (PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MgV_ZewS6gLm1a3Vyhg33pFHi5uTld_2?usp=sharing), or [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sparks-baird/mat_discover/main?labpath=mat_discover_pypi.ipynb). On Google Colab and Binder, this may take a few minutes to install and load, respectively. During training and prediction, Google Colab will be faster than Binder since Google Colab has access to a GPU while Binder does not.

### Load Data
If you're using your own dataset, you will need to supply a Pandas DataFrame that contains `formula` and `target` columns. If you have a `train.csv` file (located in current working directory) with these two columns, this can be converted to a DataFrame via:
```python
import pandas as pd
df = pd.read_csv("train.csv")
```
Note that you can load any of the datasets within `CrabNet/data/`, which includes `matbench` data, other datasets from the CrabNet paper, and a recent (as of Oct 2021) snapshot of `K_VRH` bulk modulus data from Materials Project. For example, to load the bulk modulus snapshot:
```python
from crabnet.data.materials_data import elasticity
train_df, val_df = disc.data(elasticity, "train.csv") # note that `val.csv` within `elasticity` is every other Materials Project compound (i.e. "target" column filled with zeros)
```

The built-in data directories are as follows:
> ```python
> {'benchmark_data',
>  'benchmark_data.CritExam__Ed',
>  'benchmark_data.CritExam__Ef',
>  'benchmark_data.OQMD_Bandgap',
>  'benchmark_data.OQMD_Energy_per_atom',
>  'benchmark_data.OQMD_Formation_Enthalpy',
>  'benchmark_data.OQMD_Volume_per_atom',
>  'benchmark_data.aflow__Egap',
>  'benchmark_data.aflow__ael_bulk_modulus_vrh',
>  'benchmark_data.aflow__ael_debye_temperature',
>  'benchmark_data.aflow__ael_shear_modulus_vrh',
>  'benchmark_data.aflow__agl_thermal_conductivity_300K',
>  'benchmark_data.aflow__agl_thermal_expansion_300K',
>  'benchmark_data.aflow__energy_atom',
>  'benchmark_data.mp_bulk_modulus',
>  'benchmark_data.mp_e_hull',
>  'benchmark_data.mp_elastic_anisotropy',
>  'benchmark_data.mp_mu_b',
>  'benchmark_data.mp_shear_modulus',
>  'element_properties',
>  'matbench',
>  'materials_data',
>  'materials_data.elasticity',
>  'materials_data.example_materials_property'}
> ```

To see what `.csv` files are available (e.g. `train.csv`), you will probably need to navigate to [CrabNet/data/](https://github.com/sgbaird/CrabNet/tree/master/data) and explore.

Finally, to download data from Materials Project directly, see [generate_elasticity_data.py](https://github.com/sparks-baird/mat_discover/blob/main/generate_elasticity_data.py).

## Interactive Plots
Interactive plots for several types of Pareto front plots can be found [here](https://sparks-baird.github.io/mat_discover/figures/).

## Developing
This project was developed primarily in "Python in Visual Studio Code" using `black`, `mypy`, `pydocstyle`, `kite`, other tools, and various community extensions. Some other notable tools used in this project are:
- Miniconda
- `pipreqs` was used as a starting point for `requirements.txt`
- `flit` is used to create `pyproject.toml` and publish to PyPI
- `conda env export --from-history -f environment.yml` was used as a starting point for `environment.yml`
- `grayskull` is used to generate `meta.yaml` for publishing to `conda-forge`
- `conda-smithy` is used to create a feedstock for `conda-forge`
- A variety of GitHub actions are used (see [workflows](.github/workflows))
- `pytest` is used for testing
- `numba` is used to accelerate the Wasserstein distance matrix computations via CPU or GPU

To help with development, you will need to [install from source](README.md#from-source). Note that when using a `conda` environment (recommended), you may avoid certain issues down the road by opening VS Code via an Anaconda command prompt and entering the command `code` (at least until the VS Code devs fix some of the issues associated with opening it "normally"). For example, in Windows, press the "Windows" key, type "anaconda", and open "Anaconda Powershell Prompt (miniconda3)" or similar. Then type `code` and press enter.

## Bugs, Questions, and Suggestions
If you find a bug or have suggestions for documentation please [open an issue](https://github.com/sparks-baird/mat_discover/issues/new/choose). If you're reporting a bug, please include a simplified reproducer. If you have questions, have feature suggestions/requests, or are interested in extending/improving `mat_discover` and would like to discuss, please use the Discussions tab and use the appropriate category ("Ideas", "Q&A", etc.). Pull requests are welcome and encouraged.

<!---
Recommended installation through `pip` with python 3.7.

```
pip install python==3.7
pip install ElM2D
```

For the background theory, please read our paper ["The Earth Mover’s Distance as a Metric for the Space of Inorganic Compositions"](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c03381)

## Examples

For more interactive examples please see www.elmd.io/plots

## Usage

### Computing Distance Matrices

### Sorting

### Embedding

### Saving

### Cross Validation

### Available Metrics

## Citing

If you would like to cite this code in your work, please use the following reference

```
@article{doi:10.1021/acs.chemmater.0c03381,
author = {Hargreaves, Cameron J. and Dyer, Matthew S. and Gaultois, Michael W. and Kurlin, Vitaliy A. and Rosseinsky, Matthew J.},
title = {The Earth Mover’s Distance as a Metric for the Space of Inorganic Compositions},
journal = {Chemistry of Materials},
volume = {32},
number = {24},
pages = {10610-10620},
year = {2020},
doi = {10.1021/acs.chemmater.0c03381},
URL = {
        https://doi.org/10.1021/acs.chemmater.0c03381
},
eprint = {
        https://doi.org/10.1021/acs.chemmater.0c03381
}
}
```
--->
