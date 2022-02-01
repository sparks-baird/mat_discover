# Predicting Decomposition Energy with CGCNN

A CGCNN model has been trained using the dataset developed by Bartel et al. and presented in their [paper](https://www.nature.com/articles/s41524-020-00362-y). Due to the fluidity of the Materials Project database, several data points within the original dataset are missing or have been assigned new MP-IDS within the current instance of the dataset. These issues were resolved prior to pulling the structure files for training.

## Running Predictions

The following packages are required to run CGCNN:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

To input crystal structures to CGCNN, you will need to define a customized dataset. This takes the form of a directory containing `.cif` files of the structures you want to predict in addition to a `id_prop.csv` file containing filenames and targets and an `atom_init.json` configuration file. Examples are available within the `data/` folder.

The structure of the data directory should be as follows:
```
data_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

A trained regression model for predicting the decomposition enthalpy of crystalline solids is available for use as `Ed_regression.pth.tar`. Before making a prediction, ensure that the dataset you want to predict on matches the previously mentioned format. Then enter the following code into your terminal:

```bash
python predict.py Ed_regression.pth.tar data/data-dir
```

When the model has finished it will return `test_results.csv` file containing the predicted values for each material as it is listed in the `id_prop.csv` file.
