<!-- TODO: add buttons for code ocean and Zenodo DOI [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/3904426/tree)-->
# DiSCoVeR

[![Open In Colab (PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MgV_ZewS6gLm1a3Vyhg33pFHi5uTld_2?usp=sharing)
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://doi.org/10.24433/CO.8463578.v1)
[![Interactive Figures](https://img.shields.io/static/v1?message=Open%20interactive%20figures&logo=github&labelColor=5c5c5c&color=blueviolet&logoColor=white&label=%20)](https://mat-discover.readthedocs.io/en/latest/figures.html)
[![Read the Docs](https://img.shields.io/readthedocs/mat-discover?label=Open%20the%20documentation&logo=readthedocs)](https://mat-discover.readthedocs.io/en/latest/)

[![PyPI version](https://img.shields.io/pypi/v/mat_discover.svg)](https://pypi.org/project/mat_discover/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/sparks-baird/mat_discover/badge.svg?service=github)](https://coveralls.io/github/sparks-baird/mat_discover)
[![Lines of code](https://img.shields.io/tokei/lines/github/sparks-baird/mat_discover)](https://img.shields.io/tokei/lines/github/sparks-baird/mat_discover)
[![License](https://img.shields.io/github/license/sparks-baird/mat_discover?service=github)](https://img.shields.io/github/license/sparks-baird/mat_discover)
[![DOI](https://zenodo.org/badge/392897894.svg?service=github)](https://zenodo.org/badge/latestdoi/392897894)
<!-- ![PyPI - License](https://img.shields.io/pypi/l/mat_discover) -->
<!-- [![Coverage Status](https://coveralls.io/repos/github/sparks-baird/mat_discover/badge.svg?branch=main)](https://coveralls.io/github/sparks-baird/mat_discover?branch=main) -->
<!-- ![Coveralls](https://img.shields.io/coveralls/github/sparks-baird/mat_discover) -->

[![Conda](https://img.shields.io/conda/v/sgbaird/mat_discover)](https://anaconda.org/sgbaird/mat_discover)
[![Conda](https://img.shields.io/conda/pn/sgbaird/mat_discover)](https://anaconda.org/sgbaird/mat_discover)
[![Conda](https://img.shields.io/conda/dn/sgbaird/mat_discover?label=conda%7Cdownloads)](https://anaconda.org/sgbaird/mat_discover)
[![Anaconda-Server Badge](https://anaconda.org/sgbaird/mat_discover/badges/latest_release_relative_date.svg)](https://anaconda.org/sgbaird/mat_discover)
<!-- ![Conda](https://img.shields.io/conda/dn/sgbaird/mat_discover) -->
<!-- [![Anaconda-Server Downloads](https://anaconda.org/sgbaird/mat_discover/badges/downloads.svg)](https://anaconda.org/sgbaird/mat_discover) -->
<!-- [![Anaconda-Server Downloads](https://anaconda.org/sgbaird/mat_discover/badges/downloads.svg?service=github)](https://anaconda.org/sgbaird/mat_discover) -->
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dm/mat_discover?label=PyPI%20downloads) -->

A materials discovery algorithm geared towards exploring high performance candidates in new chemical spaces using composition-only.

<img src=https://user-images.githubusercontent.com/45469701/139520031-bf4fda18-9be7-4c54-b70b-c9be8e974cea.png width=500>  

<sup>Bulk modulus values overlaid on DensMAP densities (cropped).</sup>

We describe the DiSCoVeR algorithm, how to install `mat_discover`, and basic usage (e.g.
`fit`/`predict`, custom or built-in datasets, adaptive design). [Interactive plots](https://mat-discover.readthedocs.io/en/latest/figures.html) for several types of
Pareto front plots are available via [the `mat_discover` documentation](https://mat-discover.readthedocs.io/en/latest/). We also describe how
to contribute, what to do if you run into bugs or have questions, and citation information. The [`mat_discover` docs](https://mat-discover.readthedocs.io/en/latest/) have more, such as [examples](https://mat-discover.readthedocs.io/en/latest/examples.html) (including a [teaching example](https://mat-discover.readthedocs.io/en/latest/examples.html#bare-bones)), the [interactive figures](https://mat-discover.readthedocs.io/en/latest/figures.html#figures) mentioned, and the [Python API](https://mat-discover.readthedocs.io/en/latest/modules.html).

The article ([ChemRxiv](https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8-v3)) has been accepted at [Digital Discovery](https://www.rsc.org/journals-books-databases/about-journals/digital-discovery/) (2021-02-03). See [Citing](README.md#citing).

## DiSCoVeR Workflow

Why you'd want to use this tool, whether it's "any good", alternative tools, and summaries of the workflow.

### Why DiSCoVeR?

The primary anticipated use-case of DiSCoVeR is that you have some training data (chemical formulas and target property), and you would like to determine the "next best experiment" to perform based on a user-defined relative importance of performance vs. chemical novelty. You can even run the model without any training targets which is equivalent to setting the target weight as 0.

### Is it any good?

Take an initial training set of 100 chemical formulas and associated Materials Project bulk moduli followed by 900 adaptive design iterations (x-axis) using random search, novelty-only (performance weighted at 0), a 50/50 weighting split, and performance-only (novelty weighted at 0). These are the columns. The rows are the total number of observed "extraordinary" compounds (top 2%), the total number of _additional_ unique atoms, and total number of additional unique chemical formulae templates. In other words:
1. How many "extraordinary" compounds have been observed so far?
1. How many unique atoms have been explored so far? (not counting atoms already in the starting 100 formulas)
1. How many unique chemical templates (e.g. A2B3, ABC, ABC2) have been explored so far? (not counting templates already in the starting 100 formulas)

The 50/50 weighting split offers a good trade-off between performance and novelty. Click the image to navigate to the interactive figure which includes two additional rows: best so far and current observed.

<a href=https://mat-discover.readthedocs.io/en/latest/figures.html#adaptive-design-comparison>
  <img src=https://user-images.githubusercontent.com/45469701/146947278-6399e996-ce9a-46bd-bda7-e3b4feedc525.png width=675>
</a>

We also ran some benchmarking against `sklearn.neighbors.LocalOutlierFactor` (novelty detection algorithm) using `mat2vec` and `mod_petti` featurizations. The interactive results are available [here](https://mat-discover.readthedocs.io/en/latest/figures.html#adaptive-design-comparison).

### Alternatives

This approach is similar to what you will find with Bayesian optimization
(BO), but with explicit emphasis on chemical novelty. If you're interested in doing
Bayesian optimization, I recommend using [Facebook/Ax](https://ax.dev/docs/bayesopt.html) (not affiliated). I am
working on an [implementation of composition-based Bayesian optimization
using Ax](https://github.com/facebook/Ax/issues/727) (2021-12-10).

For alternative "suggest next experiment" materials discovery tools,
see the [Citrine Platform](https://citrination.com/) (free for non-commercial use), [CAMD](https://github.com/TRI-AMDD/CAMD) ([trihackathon2020 tutorial notebooks](https://github.com/TRI-AMDD/tri-hackathon-2020)), [PyChemia](https://github.com/MaterialsDiscovery/PyChemia),
[Heteroscedastic-BO](https://github.com/Ryan-Rhys/Heteroscedastic-BO), and
[thermo](https://github.com/janosh/thermo).

For materials informatics (MI) and other relevant codebases/links, see:

- [my lists of (total ~200) MI codebases](https://github.com/sgbaird?tab=stars),
  in particular:
  - [materials discovery](https://github.com/stars/sgbaird/lists/materials-discovery)
  - [composition](https://github.com/stars/sgbaird/lists/%EF%B8%8F-composition-predictions)-,
    [crystal structure](https://github.com/stars/sgbaird/lists/structural-predictions)-,
    and [molecule](https://github.com/stars/sgbaird/lists/molecule-predictions)-based predictions
  - [MI databases](https://github.com/stars/sgbaird/lists/materials-databases), especially [NOMAD](https://nomad-lab.eu/) and [MPDS](https://mpds.io/)
  - [MI materials synthesis](https://github.com/stars/sgbaird/lists/materials-synthesis)
  - [MI natural language processing](https://github.com/stars/sgbaird/lists/materials-nlp)
  - [physics-based MI simulations](https://github.com/stars/sgbaird/lists/materials-synthesis)
- Other lists of MI-relevant codebases:
  - [general machine learning codebases](https://github.com/stars/sgbaird/lists/machine-learning-general)
  - [tools to help with scientific publishing](https://github.com/stars/sgbaird/lists/scientific-publishing)
  - [tools to help with your Python coding efforts](https://github.com/stars/sgbaird/lists/python-enhancements)
- [this curated list of "Awesome" materials
  informatics](https://github.com/tilde-lab/awesome-materials-informatics) (~100 as of 2021-12-10)

### Visualization
The DiSCoVeR workflow is visualized as follows:

<img src="https://github.com/sparks-baird/mat_discover/raw/main/figures/discover-workflow.png" alt="DiSCoVeR Workflow" width=600>

<sup>Figure 1: DiSCoVeR workflow to create chemically homogeneous clusters.  (a) Training and validation data are obtained inthe form of chemical formulas and target properties (i.e.  performance).  (b) The training and validation chemical formulasare combined and used to compute ElMD pairwise distances.  (c) ElMD pairwise distance matrices are used to computeDensMAP embeddings and DensMAP densities.  (d) DensMAP embeddings are used to compute HDBSCAN\* clusters.(e) Validation target property predictions are made via CrabNet and plotted against the uniqueness proxy (e.g.  densityproxy) in the form of a Pareto front plot.  Discovery scores are assigned based on the (arbitrarily) weighted sum of scaledperformance and uniqueness proxy.  Higher scores are better.  (f) HDBSCAN* clustering results can be used to obtain acluster-wise performance (e.g.  average target property) plotted against a cluster-wise uniqueness proxy (e.g.  fraction ofvalidation compounds vs.  total compounds within a cluster).</sup>

### Tabular Summary
A summary of the DiSCoVeR methods are given in the following table:

<sup>Table 1: A description of methods used in this work and each method’s role in DiSCoVeR. ∗A Pareto front is more information-dense than a proxy score in that there are no predefined relative weights for performance vs. uniqueness proxy. Compounds that are closer to the Pareto front are better. The upper areas of the plot represent a higher weight towards performance while the right-most areas of the plot represent a higher weight towards uniqueness.</sup>
| Method                                                                   | What is it?                                   | What is its role in DiSCoVeR?           |
| ------------------------------------------------------------------------ | --------------------------------------------- | --------------------------------------- |
| [CrabNet](https://github.com/anthony-wang/CrabNet)                       | Composition-based property regression         | Predict performance for proxy scores    |
| [ElMD](https://github.com/lrcfmd/ElMD)                                   | Composition-based distance metric             | Supply distance matrix to DensMAP       |
| [DensMAP](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html) | Density-aware dimensionality reduction        | Obtain densities for density proxy      |
| [HDBSCAN*](https://hdbscan.readthedocs.io/en/latest/index.html)          | Density-aware clustering                      | Create chemically homogeneous clusters  |
| Peak proxy                                                               | High performance relative to nearby compounds | Proxy for "surprising" high performance |
| Density proxy                                                            | Sparsity relative to nearby compounds         | Proxy for chemical novelty              |
| Peak proxy score                                                         | Weighted sum of performance and peak proxy    | Used to rank compounds                  |
| Density proxy score                                                      | Weighted sum of performance and density proxy | Used to rank compounds                  |
| [Pareto front](https://en.wikipedia.org/wiki/Pareto_front)               | Optimal performance/uniqueness trade-offs     | Visually screen compounds (no weights*) |

## Installation

I recommend that you run `mat_discover` in a separate conda environment, at least for
initial testing. After installing
[Anaconda](https://docs.anaconda.com/anaconda/navigator/index.html) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html), you can [create a new
environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
in Python `3.9` (`mat_discover` is also tested on `3.7` and `3.8`) via:

```python
conda create --name mat_discover python==3.9.*
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
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
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
git clone https://github.com/sparks-baird/mat_discover.git
cd mat_discover
```

To perform the local installation, you can use `pip`, `conda`, or `flit`. If using `flit`, make sure to install it first via `conda install flit` or `pip install flit`.
| **pip**            | **conda**                                 | **flit**                  |
| ------------------ | ----------------------------------------- | ------------------------- |
| `pip install -e .` | `conda env create --file environment.yml` | `flit install --pth-file` |

<!-- conda install torch cudatoolkit=11.1 -c pytorch -c conda-forge # or use pip command specific to you from https://pytorch.org/get-started/locally/ -->

## Basic Usage
How to `fit`/`predict`, use custom or built-in datasets, and perform adaptive design.

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

> ⚠️ ignore the "validation" mean absolute error (MAE) command line output during `disc.fit(train_df)` ⚠️

See
[mat_discover_example.py](https://github.com/sparks-baird/examples/mat_discover_example.py),
[![Open In Colab
(PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MgV_ZewS6gLm1a3Vyhg33pFHi5uTld_2?usp=sharing),
or
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sparks-baird/mat_discover/main?labpath=mat_discover_pypi.ipynb).
On Google Colab and Binder, this may take a few minutes to install and load,
respectively. During training and prediction, Google Colab will be faster than Binder
since Google Colab has access to a GPU while Binder does not. Sometimes Binder takes a long time to load, so please consider using Open In Colab or the normal installation instructions instead.

### Load Data

If you're using your own dataset, you will need to supply a Pandas DataFrame that
contains `formula` (string) and `target` (numeric) columns. If you have a `train.csv` file
(located in current working directory) with these two columns, this can be converted to
a DataFrame via:

```python
import pandas as pd
train_df = pd.read_csv("train.csv")
```

For validation data without known property values to be used with `predict`, dummy
values (all zeros) are assigned internally. In this case, you can read in a CSV file
that contains only the `formula` (string) column:

```python
val_df = pd.read_csv("val.csv")
```

Note that you can load any of the datasets within `CrabNet/data/`, which includes `matbench` data, other datasets from the CrabNet paper, and a recent (as of Oct 2021) snapshot of `K_VRH` bulk modulus data from Materials Project. For example, to load the bulk modulus snapshot:

```python
from crabnet.data.materials_data import elasticity
train_df, val_df = disc.data(elasticity, "train.csv") # note that `val.csv` within `elasticity` is every other Materials Project compound (i.e. "target" column filled with zeros)
```

The built-in data directories are as follows:
>
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

To see what `.csv` files are available (e.g. `train.csv`), you will probably need to navigate to [CrabNet/data/](https://github.com/sgbaird/CrabNet/tree/master/crabnet/data) and explore. For example, to use a snapshot of the Materials Project `e_above_hull` dataset ([`mp_e_hull`](https://github.com/sgbaird/CrabNet/tree/master/crabnet/data/benchmark_data/mp_e_hull)):
```python
from crabnet.data.benchmark_data import mp_e_hull
train_df = disc.data(mp_e_hull, "train.csv", split=False)
val_df = disc.data(mp_e_hull, "val.csv", split=False)
test_df = disc.data(mp_ehull, "test.csv", split=False)
```

Finally, to download data from Materials Project directly, see [generate_elasticity_data.py](https://github.com/sparks-baird/mat_discover/blob/main/mat_discover/utils/generate_elasticity_data.py).

### Adaptive Design
The anticipated end-use of `mat_discover` is in an adaptive design scheme where the objective function (e.g. wetlab synthesis and characterization) is expensive. After loading some data for a validation scenario (or your own data)
```python
from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.adaptive_design import Adapt
train_df, val_df = data(elasticity, "train.csv", dummy=False, random_state=42)
train_df, val_df, extraordinary_thresh = extraordinary_split(
    train_df, val_df, train_size=100, extraordinary_percentile=0.98, random_state=42
)
```
you can then predict your first additional experiment to run via:
```python
adapt = Adapt(train_df, val_df, timed=False)
first_experiment = adapt.suggest_first_experiment() # fit Discover() to train_df, then move top-ranked from val_df to train_df
```
Subsequent experiments are suggested as follows:
```python
second_experiment = adapt.suggest_next_experiment() # refit CrabNet, use existing DensMAP data, move top-ranked from val to train
third_experiment = adapt.suggest_next_experiment()
```

Alternatively, you can do this in a closed loop via:
```python
n_iter = 100
adapt.closed_loop_adaptive_design(n_experiments=n_iter, print_experiment=False)
```
However, as the name suggests, the closed loop approach does not allow you to input data after each suggested experiment.

## Developing and Contributing

This project was developed primarily in [Python in Visual Studio Code](https://code.visualstudio.com/docs/languages/python) using `black`, `mypy`, `pydocstyle`, `kite`, other tools, and various community extensions. Some other notable tools used in this project are:

- Miniconda
- `pipreqs` was used as a starting point for `requirements.txt`
- `flit` is used to create `pyproject.toml` to publish to PyPI
- `conda env export --from-history -f environment.yml` was used as a starting point for `environment.yml`
- `grayskull` and `conda-souschef` are used to generate and tweak `meta.yaml`,
  respectively, for publishing to Anaconda (if you know how to get this up on
  conda-forge, help is welcome 😉)
- A variety of GitHub actions are used (see [workflows](https://github.com/sparks-baird/.github/workflows))
- `pytest` is used for testing
- `numba` is used to accelerate the Wasserstein distance matrix computations via CPU or GPU

<!-- - `conda-smithy` is used to create a feedstock for `conda-forge` -->

For simple changes, navigate to github.com/sparks-baird/mat_discover, click on the
relevant file (e.g. `README.md`), and look for the pencil (✏️). GitHub will walk you
through the rest.

To help with in-depth development, you will need to [install from
source](README.md#from-source). Note that when using a `conda` environment
(recommended), you may avoid certain issues down the road by opening VS Code via an
Anaconda command prompt and entering the command `code` (at least until the VS Code devs
fix some of the issues associated with opening it "normally"). For example, in Windows,
press the "Windows" key, type "anaconda", and open "Anaconda Powershell Prompt
(miniconda3)" or similar. Then type `code` and press enter. To build the docs, first install `sphinx` and `sphinx_rtd_theme`. Then run:

```bash
cd docs/
make html
```

And open `docs/build/index.html` (e.g. via `start index.html` on Windows)

## Bugs, Questions, and Suggestions

If you find a bug or have suggestions for documentation please [open an
issue](https://github.com/sparks-baird/mat_discover/issues/new/choose). If you're
reporting a bug, please include a simplified reproducer. If you have questions, have
feature suggestions/requests, or are interested in extending/improving `mat_discover`
and would like to discuss, please use the Discussions tab and use the appropriate
category ("Ideas", "Q&A", etc.). If you have a
question, please ask! I won't bite. Pull requests are welcome and encouraged.

## Citing

The preprint is hosted on ChemRxiv:
> Baird S, Diep T, Sparks T. DiSCoVeR: a Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions. ChemRxiv 2021. [doi:10.33774/chemrxiv-2021-5l2f8-v3](https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8-v3). This content is a preprint and has not been peer-reviewed.

The BibTeX citation is as follows:

```bib
@article{baird_diep_sparks_2021,
place={Cambridge},
title={DiSCoVeR: a Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions},
DOI={10.33774/chemrxiv-2021-5l2f8-v3},
journal={ChemRxiv},
publisher={Cambridge Open Engage},
author={Baird, Sterling and Diep, Tran and Sparks, Taylor},
year={2021}
}
```

The article is under review at [Digital Discovery](https://www.rsc.org/journals-books-databases/about-journals/digital-discovery/).

## Looking for more?
See [examples](https://mat-discover.readthedocs.io/en/latest/examples.html), including [a teaching example](https://mat-discover.readthedocs.io/en/latest/examples.html#bare-bones), and the [Python API](https://mat-discover.readthedocs.io/en/latest/modules.html).
