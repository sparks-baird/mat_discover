# ElM2D
A high performance mapping class to construct [ElMD distance matrices](www.github.com/lrcfmd/ElMD) from large datasets of ionic compositions, suitable for single node usage on HPC systems. This includes helper methods to directly embed these datasets as maps of chemical space, as well as sorting lists of compositions, and exporting kernel matrices. 

Recommended installation through `pip` with python 3.7.

```
pip install python==3.7
pip install ElM2D
```

## Examples

125,000 compositions from the inorganic crystal structure database embedded with PCA, plotted with [datashader](https://github.com/holoviz/datashader):
![ICSD Map](https://i.imgur.com/ZPqHxsz.png)

For more interactive examples please see www.elmd.io/plots

## Usage 

### Computing Distance Matrices

The computed distance matrix is accessible through the `dm` attribute and can be saved and loaded as a python binary pickle object.

```python
mapper = ElM2D()
mapper.fit(df["composition"])

print(mapper.dm)

mapper.save_dm("ComputedMatrix.pk")
...
mapper.load_dm("ComputedMatrix.pk")
```

This distance matrix can be used as a lookup table for distances between compositions given their numeric indices (`distance = mapper.dm[i][j]`) or used as a kernel matrix for embedding, regression, and classification tasks directly.

### Sorting

To sort a list of compositions into an ordering of chemical similarity

```python
from ElM2D import ElM2D
...

comps = df["composition"].to_numpy()
sorted_indices = ElM2D().sort_compositions(comps)
sorted_comps = comps[sorted_indices]
```

### Embedding

Embeddings can be constructed through either the [UMAP](https://github.com/lmcinnes/umap) or PCA methods of dimensionality reduction. The most recently embedded points are accessible via the `embedding` property. Higher dimensional embeddings can be created with the `n_components` parameter. 

```python
mapper = ElM2D()
embedding = mapper.fit_transform(df["formula"])
embedding = mapper.fit_transform(df["formula"], how="PCA", n_components=7)
```

Embeddings may also be directed towards a particular chemical property in a pandas DataFrame, to bring known patterns into focus.
```python
embedding = mapper.fit_transform(df["formula"], df["property_of_interest"])
```

By default, the [modified Pettifor scale](https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011/meta) is used as the method of atomic similarity, "atomic", "petti", "mod_petti", and "mendeleev" can be selected through the `metric` attribute. 

```python
embedding = mapper.fit_transform(df["formula"], metric="atomic")
```

These embeddings may be visualized within a jupyter notebook, or exported to HTML to view full page in the web browser.

```python
mapper.fit_transform(df["formula"])

# Returns a figure for viewing in notebooks
mapper.plot() 

# Returns a figure and saves as ElM2D_Plot_UMAP.html
mapper.plot("ElM2D_Plot_UMAP.html")  

# Returns and saves figure, with colouring based on property from a pandas Series
mapper.plot(fp="ElM2D_Plot_UMAP.html", color=df["chemical_property"]) 

# Plotting also works in 3D
mapper.fit_transform(df["formula"], n_components=3)
mapper.plot(color=df["chemical_property"])
```
