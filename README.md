# ElM2D
A high performance mapping class to construct [ElMD distance matrices](www.github.com/lrcfmd/ElMD) from large datasets of ionic compositions, suitable for single node usage on HPC systems. This includes helper methods to directly embed these datasets as maps of chemical space, as well as sorting lists of compositions, and exporting kernel matrices. 

Recommended installation through `pip` with python 3.7.

```
pip install python==3.7
pip install ElM2D
```

For the background theory, please read our paper ["The Earth Mover’s Distance as a Metric for the Space of Inorganic Compositions"](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c03381)

## Examples

125,000 compositions from the inorganic crystal structure database embedded with PCA, plotted with [datashader](https://github.com/holoviz/datashader):
![ICSD Map](https://i.imgur.com/ZPqHxsz.png)

For more interactive examples please see www.elmd.io/plots

## Usage 

### Computing Distance Matrices

The computed distance matrix is accessible through the `dm` attribute and can be saved and loaded as a python binary pickle object.

```python
from ElM2D import ElM2D

mapper = ElM2D()
mapper.fit(df["formula"])

print(mapper.dm)

mapper.export_dm("ComputedMatrix.csv")
```

This distance matrix can be used as a lookup table for distances between compositions given their numeric indices (`distance = mapper.dm[i][j]`) or used as a kernel matrix for embedding, regression, and classification tasks directly.

### Sorting

To sort a list of compositions into an ordering of chemical similarity

```python
mapper.fit(df["formula"])

sorted_indices = mapper.sort()
sorted_comps = mapper.sorted_comps
```

### Embedding

Embeddings can be constructed through either the [UMAP](https://github.com/lmcinnes/umap) or PCA methods of dimensionality reduction. The most recently embedded points are accessible via the `embedding` property. Higher dimensional embeddings can be created with the `n_components` parameter. 

```python
mapper = ElM2D()
mapper.fit(df["formula"])
embedding = mapper.transform()
...

# For new data
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

### Saving 

Smaller datasets can be saved directly with the `save(filepath.pk)`/`load(filepath.pk)` methods directly. This is limited to files of size 3GB (the python binary file size limit).

Larger datasets will require importing/exporting the distance matrix and embeddings (`export_embedding(filepath.csv)`/`import_embedding(filepath.csv)` separately as csv files if you require this processed data in future work. 

```python
mapper.fit(small_df["formula"])
mapper.save("small_df_mapper.pk")
...
mapper = ElM2D()
mapper.load("small_df_mapper.pk")
...

mapper.fit(large_df["formula"])
mapper.export_dm("large_df_dm.csv")
mapper.export_dm("large_df_emb_UMAP.csv")
...

mapper = ElM2D()
mapper.import_dm("large_df_dm.csv")
mapper.import_embedding("large_df_emb_UMAP.csv")
mapper.formula_list = df["formula"]
```

### Cross Validation

Perform a K-Folds splitting of the dataset into subsets, to build up training
and testing datasets.

```python
cvs = mapper.cross_validate()
for i, (X_train, X_test) in enumerate(cvs):
    sub_mapper = ElM2D()
    
    sub_mapper.fit(X_train)
    sub_mapper.save(f"train_elm2d_{i}.pk")
    
    sub_mapper.fit(X_test)
    sub_mapper.save(f"test_elm2d_{i}.pk")
...
from sklearn.metrics import mean_average_error as mae

cvs = mapper.cross_validate(y=df["target"])

for X_train, X_test, y_train, y_test in cvs:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    errors.append(mae(y_pred, y_test))

print(np.mean(errors))
```

### Available Metrics
You may use either discrete scales or machine learnt representations for each element. Choose these via the `metric` parameter.

Linear:
- mendeleev
- petti
- atomic
- mod_petti

Machine Learnt:
- oliynyk 
- oliynyk_sc
- cgcnn 
- elemnet 
- jarvis 
- jarvis_sc 
- magpie 
- magpie_sc 
- mat2vec 
- matscholar 
- megnet16 
- random_200

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