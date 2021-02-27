# ElM2D
A high performance mapping class to construct ElM2D plots from large datasets of ionic compositions.

Recommended installation through `pip` and python 3.7.

```
pip install python==3.7
pip install ElM2D
```

## Usage 

### Sorting

To sort a list of compositions into an ordering of chemical similarity

```python
from ElM2D import ElM2D
...

comps = df["formula"].to_numpy()
sorted_indices = ElM2D().sort_compositions(comps)
sorted_comps = comps[sorted_indices]
```

### Embedding

Embeddings can be constructed through either the UMAP or PCA methods of dimensionality reduction. The embedded points are accessible via the `embedding` property. Higher dimensional embeddings can be created with the `n_components` parameter. 

```python
mapper = ElMD()
embedding = mapper.fit_transform(df["formula"])
embedding = mapper.fit_transform(df["formula"], how="PCA", n_components=7)
```

These embeddings may be visualized within a jupyter notebook, or exported to HTML to view in the web browser.

```python
mapper.fit_transform(df["formula"])
mapper.plot()  # Returns a figure for viewing in notebooks
...
mapper.plot("ElM2D_Plot_UMAP.html")  # Returns a figure and saves as ElM2D_Plot_UMAP.html
...
mapper.plot(fp="ElM2D_Plot_UMAP.html", color=df["target"])  # Returns and saves figure, with additional colouring based on a property from an associated pandas dataframe
```


