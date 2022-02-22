<!-- TODO: add buttons for code ocean and Zenodo DOI [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/3904426/tree)-->
# DiSCoVeR

[![Open In Colab (PyPI)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MgV_ZewS6gLm1a3Vyhg33pFHi5uTld_2?usp=sharing)
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://doi.org/10.24433/CO.8463578.v1)
[![Interactive Figures](https://img.shields.io/static/v1?message=Open%20interactive%20figures&logo=github&labelColor=5c5c5c&color=blueviolet&logoColor=white&label=%20)](https://mat-discover.readthedocs.io/en/latest/figures.html)
[![Read the Docs](https://img.shields.io/readthedocs/mat-discover?label=Read%20the%20docs&logo=readthedocs)](https://mat-discover.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/sparks-baird/mat_discover/Install%20with%20flit%20and%20test%20via%20Pytest?label=main)](https://github.com/sparks-baird/mat_discover/actions/workflows/flit-install-test.yml)

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

[The documentation](https://mat-discover.readthedocs.io/) describes the DiSCoVeR algorithm, how to install `mat_discover`, and basic usage (`fit`/`predict`, custom or built-in datasets, adaptive design, and cluster plots). [Interactive plots](https://mat-discover.readthedocs.io/en/latest/figures.html) for several types of
Pareto front plots are available. We also describe how to contribute, and what to do if you run into bugs or have questions. Various [examples](https://mat-discover.readthedocs.io/en/latest/examples.html) (including a [teaching example](https://mat-discover.readthedocs.io/en/latest/examples.html#bare-bones)), the [interactive figures](https://mat-discover.readthedocs.io/en/latest/figures.html#figures) mentioned, and the [Python API](https://mat-discover.readthedocs.io/en/latest/modules.html) are also hosted at https://mat-discover.readthedocs.io. The article is [published at Digital Discovery](https://dx.doi.org/10.1039/D1DD00028D). If you find this useful, please consider citing as follows:

## Citing
> Baird, S. G.; Diep, T. Q.; Sparks, T. D. DiSCoVeR: A Materials Discovery Screening Tool for High Performance, Unique Chemical Compositions. Digital Discovery 2022. https://doi.org/10.1039/D1DD00028D.

```bib
@article{bairdDiSCoVeRMaterialsDiscovery2022,
  title = {{{DiSCoVeR}}: A {{Materials Discovery Screening Tool}} for {{High Performance}}, {{Unique Chemical Compositions}}},
  shorttitle = {{{DiSCoVeR}}},
  author = {Baird, Sterling Gregory and Diep, Tran Q. and Sparks, Taylor D.},
  year = {2022},
  month = feb,
  journal = {Digital Discovery},
  publisher = {{RSC}},
  issn = {2635-098X},
  doi = {10.1039/D1DD00028D},
  abstract = {We present Descending from Stochastic Clustering Variance Regression (DiSCoVeR) (https://github.com/sparks-baird/mat_discover), a Python tool for identifying and assessing high-performing, chemically unique compositions relative to existing compounds using a combination of a chemical distance metric, density-aware dimensionality reduction, clustering, and a regression model. In this work, we create pairwise distance matrices between compounds via Element Mover's Distance (ElMD) and use these to create 2D density-aware embeddings for chemical compositions via Density-preserving Uniform Manifold Approximation and Projection (DensMAP). Because ElMD assigns distances between compounds that are more chemically intuitive than Euclidean-based distances, the compounds can then be clustered into chemically homogeneous clusters via Hierarchical Density-based Spatial Clustering of Applications with Noise (HDBSCAN*). In combination with performance predictions via Compositionally-Restricted Attention-Based Network (CrabNet), we introduce several new metrics for materials discovery and validate DiSCoVeR on Materials Project bulk moduli using compound-wise and cluster-wise validation methods. We visualize these via multi-objective Pareto front plots and assign a weighted score to each composition that encompasses the trade-off between performance and density-based chemical uniqueness. In addition to density-based metrics, we explore an additional uniqueness proxy related to property gradients in DensMAP space. As a validation study, we use DiSCoVeR to screen materials for both performance and uniqueness to extrapolate to new chemical spaces. Top-10 rankings are provided for the compound-wise density and property gradient uniqueness proxies. Top-ranked compounds can be further curated via literature searches, physics-based simulations, and/or experimental synthesis. Finally, we compare DiSCoVeR against the naive baseline of random search for several parameter combinations in an adaptive design scheme. To our knowledge, this is the first time automated screening has been performed with explicit emphasis on discovering high-performing, novel materials.},
  langid = {english},
}
```

If you use this software, in addition to the above reference, please also cite the Zenodo DOI:
> Sterling Baird. (2022). sparks-baird/mat_discover: Release 2.1.2 (2.1.2). Zenodo. https://doi.org/10.5281/zenodo.6116258

```bib
@software{sterling_baird_2022_6116258,
  author       = {Sterling Baird},
  title        = {sparks-baird/mat\_discover: Release 2.1.2},
  month        = feb,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {2.1.2},
  doi          = {10.5281/zenodo.6116258},
  url          = {https://doi.org/10.5281/zenodo.6116258}
}
```

If you use this software as an installed dependency in another GitHub repository, please add `mat_discover` to a `requirements.txt` file in your repository via e.g.:
```bash
pip install pipreqs
pipreqs .
```
For an example, see [`requirements.txt`](requirements.txt).
