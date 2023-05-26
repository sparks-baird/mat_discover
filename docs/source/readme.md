```{include} ../../README.md
:relative-images:
```

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
see the [Citrine Platform](https://citrination.com/) (proprietary), [ChemOS](https://chemos.io/) (proprietary), [Olympus](https://aspuru-guzik-group.github.io/olympus/), [CAMD](https://github.com/TRI-AMDD/CAMD) ([trihackathon2020 tutorial notebooks](https://github.com/TRI-AMDD/tri-hackathon-2020)), [PyChemia](https://github.com/MaterialsDiscovery/PyChemia),
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

There are three ways to install `mat_discover`: Anaconda (`conda`), PyPI (`pip`), and from source. Anaconda is the preferred method.

### Anaconda

After installing
[Anaconda](https://docs.anaconda.com/anaconda/navigator/index.html) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda preferred), first update `conda` via:
```python
conda update conda
```

Then add the following channels to your default channels list:

```python
conda config --add channels conda-forge
conda config --add channels pytorch
```

I recommend that you run `mat_discover` in a separate conda environment, at least for
initial testing. You can [create a new environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) in Python `3.9` (`mat_discover` is also tested on `3.7` and `3.8`), install `mat_discover`, and activate it via:
```python
conda create --name mat_discover --channel sgbaird python==3.9.* mat_discover
conda activate mat_discover
```

In English, this reads as "Create a new environment named `mat_discover` and install a version of Python that matches `3.9.*` (e.g. `3.9.7`) and the `mat_discover` package while looking preferentially in the [@sgbaird Anaconda channel](https://anaconda.org/sgbaird/repo). Activate the `mat_discover` environment."

### Pip

Even if you use `pip` to install `mat_discover`, I still recommend doing so in a fresh `conda` environment, at least for initial testing:
```python
conda create --name mat_discover python==3.9.*
conda activate mat_discover
```

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

The same recommendation about using a fresh `conda` environment for initial testing applies here. To install from source, clone the `mat_discover` repository:

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
disc = Discover(target_unit="GPa")
disc.fit(train_df) # DataFrames should have at minimum ("formula" or "structure") and "target" columns
scores = disc.predict(val_df)
disc.plot()
disc.save()
print(disc.dens_score_df.head(10), disc.peak_score_df.head(10))
```

Note that `target_unit="GPa"` simply appends ` (GPa)` to the end of plotting labels where appropriate.

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

#### From File
If you're using your own dataset, you will need to supply a Pandas DataFrame that
contains `formula` (string) and `target` (numeric) columns (optional for `val_df`). If you have a `train.csv` file
(located in current working directory) with these two columns, this can be converted to
a DataFrame via:

```python
import pandas as pd
train_df = pd.read_csv("train.csv")
```

which might look something like the following:

formula | target
---|---
Tc1V1 | 248.539
Cu1Dy1 | 66.8444
Cd3N2 | 91.5034

For validation data without known property values to be used with `predict`, dummy
values (all zeros) are assigned internally if the `target` column isn't present. In this case, you can read in a CSV file
that contains only the `formula` (string) column:

```python
val_df = pd.read_csv("val.csv")
```

| formula |
| --- |
| Al2O3 |
| SiO2 |

#### Hard-coded
For a quick hard-coded example, you could use:
```python
train_df = pd.DataFrame(dict(formula=["Tc1V1", "Cu1Dy1", "Cd3N2"], target=[248.539, 66.8444, 91.5034]))
val_df = pd.DataFrame(dict(formula=["Al2O3", "SiO2"]))
```

#### CrabNet Datasets (including Matbench)
NOTE: you can load any of the datasets within `CrabNet/data/`, which includes `matbench` data, other datasets from the CrabNet paper, and a recent (as of Oct 2021) snapshot of `K_VRH` bulk modulus data from Materials Project. For example, to load the bulk modulus snapshot:

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

#### Directly via Materials Project
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

### Cluster Plots
To quickly determine ElMD+DensMAP+HDBSCAN* cluster labels, make the following interactive cluster plot for your data, and export a "paper-ready" PNG image, you can [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/sparks-baird/mat_discover/blob/main/examples/elmd_densmap_cluster_colab.ipynb) or see the (nearly identical) example in [`elmd_densmap_cluster.ipynb`](https://github.com/sparks-baird/mat_discover/blob/main/examples/elmd_densmap_cluster.py).

[<img src=https://user-images.githubusercontent.com/45469701/154414034-d4bfbc7c-d7bc-4cdf-9123-c5b5098b786e.png width=850>](https://colab.research.google.com/github/sparks-baird/mat_discover/blob/cluster-example/examples/elmd_densmap_cluster_colab.ipynb#scrollTo=3At5TC0gixl3)

## Bugs, Questions, and Suggestions

If you find a bug or have suggestions for documentation please [open an
issue](https://github.com/sparks-baird/mat_discover/issues/new/choose). If you're
reporting a bug, please include a simplified reproducer. If you have questions, have
feature suggestions/requests, or are interested in extending/improving `mat_discover`
and would like to discuss, please use the Discussions tab and use the appropriate
category ("Ideas", "Q&A", etc.). If you have a
question, please ask! I won't bite. Pull requests are welcome and encouraged.

## Looking for more?
See [examples](https://mat-discover.readthedocs.io/en/latest/examples.html), including [a teaching example](https://mat-discover.readthedocs.io/en/latest/examples.html#bare-bones), and the [Python API](https://mat-discover.readthedocs.io/en/latest/modules.html).
