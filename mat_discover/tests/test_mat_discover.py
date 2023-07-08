"""
Test DISCOVER algorithm.

- create distance matrix
- apply densMAP
- create clusters via HDBSCAN*
- search for interesting materials, for example:
     - high-target/low-density
     - materials with high-target surrounded by materials with low targets
     - high mean cluster target/high fraction of validation points within cluster
"""
# %% imports
from pathlib import Path
import pandas as pd
from copy import deepcopy

from crabnet.data.materials_data import elasticity
from sklearn.neighbors import LocalOutlierFactor
from mat_discover.mat_discover_ import Discover
from mat_discover.utils.gridrdf_helper import gridrdf_pdist
from mat_discover.adaptive_design import DummyCrabNet
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
lattice2 = Lattice.from_parameters(a=3.62, b=3.62, c=3.62, alpha=120, beta=90, gamma=60)
lattice3 = Lattice.from_parameters(a=3.62, b=3.62, c=3.62, alpha=90, beta=90, gamma=90)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice2, ["Ni", "Ni"], coords),
    Structure(lattice3, ["Al", "Al"], coords),
]


# %% Test Functions
def test_mat_discover():
    """Perform a simple run of mat_discover to ensure it runs without errors.

    This does not involve checking to verify that the output is correct, and
    additionally it uses a `dummy_run` and `dummy` such that MDS is used on a small
    dataset rather than UMAP on a large dataset (for faster runtimes).
    """
    disc = Discover(dummy_run=True, target_unit="GPa")
    train_df, val_df = disc.data(elasticity, fname="train.csv", dummy=True)
    disc.fit(train_df)

    score = disc.predict(val_df, umap_random_state=42)
    cat_df = pd.concat((train_df, val_df), axis=0)
    disc.group_cross_val(cat_df, umap_random_state=42)
    print("scaled test error = ", disc.scaled_error)
    disc.plot()
    # disc.save() #doesn't work with pytest for some reason (pickle: object not the same)
    # disc.load()


def test_mat_discover_xtal():
    """Perform a simple run of mat_discover with structure to ensure it runs without errors.

    This does not involve checking to verify that the output is correct, and
    additionally it uses a `dummy_run` and `dummy` such that MDS is used on a small
    dataset rather than UMAP on a large dataset (for faster runtimes).
    """
    disc = Discover(
        dummy_run=True, target_unit="GPa", use_structure=True, n_peak_neighbors=3
    )
    train_structures = deepcopy(dummy_structures) * 2
    val_structures = deepcopy(dummy_structures)
    train_df = pd.DataFrame(
        dict(structure=train_structures, target=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
    val_df = pd.DataFrame(dict(structure=val_structures, target=[7.0, 8.0, 9.0]))
    disc.fit(train_df)

    score = disc.predict(val_df, umap_random_state=42)
    cat_df = pd.concat((train_df, val_df), axis=0)
    # disc.group_cross_val(cat_df, umap_random_state=42)
    # print("scaled test error = ", disc.scaled_error)
    disc.plot()


def test_custom_regressor():
    """Test a custom # of epochs with CrabNet."""
    disc = Discover(dummy_run=True, regressor=DummyCrabNet())
    train_df, _ = disc.data(elasticity, fname="train.csv", dummy=True)
    disc.fit(train_df)


def test_plotting():
    """Ensure the individual plotting functions run successfully.

    This does not involve checking to verify that the output is correct, and
    additionally it uses a `dummy_run` and `dummy` such that MDS is used on a small
    dataset rather than UMAP on a large dataset (for faster runtimes).
    """
    disc = Discover(dummy_run=True)
    train_df, val_df = disc.data(elasticity, fname="train.csv", dummy=True)
    disc.fit(train_df)

    score = disc.predict(val_df, umap_random_state=42)

    # create dir https://stackoverflow.com/a/273227/13697228
    Path(disc.figure_dir).mkdir(parents=True, exist_ok=True)

    fig, pk_pareto_ind = disc.pf_peak_proxy()
    fig, frac_pareto_ind = disc.pf_train_contrib_proxy()

    disc.umap_cluster_scatter()
    disc.px_umap_cluster_scatter()

    # Histogram of cluster counts
    fig = disc.cluster_count_hist()

    # Scatter plot colored by target values
    fig = disc.target_scatter()
    disc.px_targ_scatter()

    # PDF evaluated on grid of points
    disc.dens_scatter()
    disc.dens_targ_scatter()

    disc.pf_dens_proxy()
    disc.pf_frac_proxy()


def test_sklearn_modpetti():
    """Perform a simple run of mat_discover to ensure it runs without errors.

    This does not involve checking to verify that the output is correct, and
    additionally it uses a `dummy_run` and `dummy` such that MDS is used on a small
    dataset rather than UMAP on a large dataset (for faster runtimes).
    """
    disc = Discover(
        dummy_run=True,
        novelty_learner=LocalOutlierFactor(novelty=True),
        novelty_prop="mod_petti",
    )
    train_df, val_df = disc.data(elasticity, fname="train.csv", dummy=True)
    disc.fit(train_df)
    score = disc.predict(val_df, umap_random_state=42)
    cat_df = pd.concat((train_df, val_df), axis=0)
    disc.group_cross_val(cat_df, umap_random_state=42)
    print("scaled test error = ", disc.scaled_error)
    # disc.plot() # not functional


def test_sklearn_mat2vec():
    """Perform a simple run of mat_discover to ensure it runs without errors.

    This does not involve checking to verify that the output is correct, and
    additionally it uses a `dummy_run` and `dummy` such that MDS is used on a small
    dataset rather than UMAP on a large dataset (for faster runtimes).
    """
    disc = Discover(
        dummy_run=True,
        novelty_learner=LocalOutlierFactor(novelty=True),
        novelty_prop="mat2vec",
    )
    train_df, val_df = disc.data(elasticity, fname="train.csv", dummy=True)
    disc.fit(train_df)
    score = disc.predict(val_df, umap_random_state=42)
    cat_df = pd.concat((train_df, val_df), axis=0)
    disc.group_cross_val(cat_df, umap_random_state=42)
    print("scaled test error = ", disc.scaled_error)
    # disc.plot() # not functional


def test_gridrdf_helper():
    gridrdf_pdist(dummy_structures)


if __name__ == "__main__":
    test_gridrdf_helper()
    test_mat_discover_xtal()
    test_custom_regressor()
    test_mat_discover()
    test_plotting()
    test_sklearn_modpetti()
    test_sklearn_mat2vec()
