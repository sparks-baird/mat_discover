# DiSCoVeR
A materials discovery algorithm geared towards exploring high performance candidates in new chemical spaces using composition-only.

## Installation
Updates coming soon, including a PyPI distribution. Anaconda distribution to follow.

The current instructions are:
```bash
conda install flit # or `pip install flit`
git clone --recurse-submodules https://github.com/sparks-baird/mat_discover.git
cd mat_discover
flit install
```

## Usage
The basic usage is:
```python
from mat_discover.discover_ import Discover
disc = Discover()
disc.fit(train_df) # DataFrames should have at minimum "formula" and "target" columns
scores = disc.predict(val_df)
disc.plot()
disc.save()
print(disc.dens_score_df.head(10), disc.peak_score_df.head(10))
```

See [mat_discover_example.py](mat_discover_example.py).

## Citing
The preprint is hosted on ChemRxiv:

> Baird S, Diep T, Sparks T. DiSCoVeR: a Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions. ChemRxiv 2021. [doi:10.33774/chemrxiv-2021-5l2f8](https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8). This content is a preprint and has not been peer-reviewed.

The BibTeX citation is as follows:
```bib
@article{baird_diep_sparks_2021,
place={Cambridge},
title={DiSCoVeR: a Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions},
DOI={10.33774/chemrxiv-2021-5l2f8},
journal={ChemRxiv},
publisher={Cambridge Open Engage},
author={Baird, Sterling and Diep, Tran and Sparks, Taylor},
year={2021}
}
```

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
