"""Download and partition elasticity data using Materials Project API."""
from os import cpu_count
from os.path import join
from pathlib import Path
import pickle

# from tqdm import tqdm
from pqdm.processes import pqdm

import numpy as np
import pandas as pd


from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition


def structure_from_cif(cif):
    """Create `pymatgen` `Structure` from a crystallographic information file str."""
    return Structure.from_str(cif, fmt="cif")


def generate_elasticity_data(
    download_data=True,
    cif=False,
    train_e_above_hull=0.05,
    val_e_above_hull=0.05,
    theoretical=False,
    folder=join("mat_discover", "data", "elasticity"),
):
    """Download (or reload) elasticity data using MPRester.

    Parameters
    ----------
    download_data : bool, optional
        [description], by default True
    cif : bool, optional
        [description], by default False
    train_e_above_hull : float, optional
        [description], by default 0.5
    val_e_above_hull : float, optional
        [description], by default 0.05
    theoretical : bool, optional
        Whether a compound is theoretical or not. False means experimental compounds, API subject to change. Can take on values
        False, True, None, or a list of the previous. by default False. See
        https://matsci.org/t/how-to-use-has-icsd-exptl-id-property-in-pymatgen-query-function/2550/4
    folder : str, optional
        Which folder to save to, by default join("mat_discover", "data", "elasticity").

    Returns
    -------
    [type]
        [description]
    """
    if type(theoretical) is not list:
        theoretical = [theoretical]
    # download and save Materials Project dataset
    elast_path = join(folder, "elast_results.pkl")
    all_path = join(folder, "all_results.pkl")

    Path(folder).mkdir(parents=True, exist_ok=True)

    # create Python "module" (for loading data)
    with open(join(folder, "__init__.py"), "w") as f:
        pass

    if download_data:
        # download
        props = ["task_id", "pretty_formula", "elasticity", "cif"]
        if not cif:
            props.remove("cif")
        # fmt: off
        excluded_elements = [
            "He", "Ne", "Ar", "Kr", "Xe", "Rn",
            "U", "Th", "Rn", "Tc", "Po", "Pu", "Pa",
            ]
        # fmt: on
        query = {
            "e_above_hull": {"$lt": train_e_above_hull},
            "elasticity": {"$exists": True},
            "theoretical": {"$in": theoretical},
            "elements": {"$nin": excluded_elements},
        }
        with MPRester() as m:
            elast_results = m.query(query, properties=props, chunk_size=2000)

        props = ["task_id", "pretty_formula", "cif"]
        if not cif:
            props.remove("cif")
        with MPRester() as m:
            all_results = m.query(
                {
                    "e_above_hull": {"$lt": val_e_above_hull},
                    "theoretical": {"$in": theoretical},
                    "elements": {"$nin": excluded_elements},
                },
                properties=props,
                chunk_size=2000,
            )

        # save
        with open(elast_path, "wb") as f:
            pickle.dump(elast_results, f)

        with open(all_path, "wb") as f:
            pickle.dump(all_results, f)
    else:
        # load the data
        with open(elast_path, "rb") as f:
            elast_results = pickle.load(f)

        with open(all_path, "rb") as f:
            all_results = pickle.load(f)

    Path(folder).mkdir(parents=True, exist_ok=True)

    def my_path(name):
        """Return a relative path to a data file."""
        return join(".", folder, name)

    # %% separate mpids and other properties for elasticity materials
    elast_mpids = [d["task_id"] for d in elast_results]
    elast_formulas = [d["pretty_formula"] for d in elast_results]
    if cif:
        elast_cifs = [d["cif"] for d in elast_results]
    elasticity = [d["elasticity"] for d in elast_results]
    K_VRH = [d["K_VRH"] for d in elasticity]

    del elast_results

    elast_comp = [Composition(formula) for formula in elast_formulas]

    elast_struct_path = join(folder, "elast_struct_dicts.pkl")
    if download_data:
        if cif:
            elast_structures = [
                Structure.from_str(cif, fmt="cif") for cif in elast_cifs
            ]
            elast_struct_dicts = [structure.as_dict() for structure in elast_structures]
            with open(elast_struct_path, "wb") as f:
                pickle.dump(elast_struct_dicts, f)
    else:
        if cif:
            with open(elast_struct_path, "rb") as f:
                elast_struct_dicts = pickle.load(f)
            elast_structures = [Structure.from_dict(s) for s in elast_struct_dicts]
            del elast_struct_dicts

    if not cif:
        elast_structures = []
    data = {
        "formula": elast_formulas,
        "composition": elast_comp,
        "structure": elast_structures,
        "K_VRH": K_VRH,
        "task_id": elast_mpids,
        "target": K_VRH,
    }
    if not cif:
        data.pop("structure")

    elast_df = pd.DataFrame(data=data)

    elast_df.to_csv(my_path("train.csv"), columns=["formula", "target"], index=False)

    all_formulas = [d["pretty_formula"] for d in all_results]

    # remove rows corresponding to formulas in val_df that overlap with train_df
    # https://stackoverflow.com/questions/11483863/python-intersection-indices-numpy-array
    indices = np.invert(np.in1d(all_formulas, elast_formulas))

    val_results = [all_results[i] for i in np.nonzero(indices)[0]]

    val_formulas = [d["pretty_formula"] for d in val_results]
    val_mpids = [d["task_id"] for d in val_results]

    if cif:
        val_cifs = [d["cif"] for d in val_results]

    del all_results, val_results

    val_comp = [Composition(formula) for formula in val_formulas]

    val_struct_path = join(folder, "val_struct_dicts.pkl")
    if download_data:
        if cif:
            val_structures = pqdm(val_cifs, structure_from_cif, n_jobs=cpu_count())
            val_struct_dicts = [structure.as_dict() for structure in val_structures]
            with open(val_struct_path, "wb") as f:
                pickle.dump(val_struct_dicts, f)
    else:
        if cif:
            with open(val_struct_path, "rb") as f:
                val_struct_dicts = pickle.load(f)
                val_structures = [Structure.from_dict(s) for s in val_struct_dicts]
                del val_struct_dicts

    if not cif:
        val_structures = []
    data = {
        "composition": val_comp,
        "formula": val_formulas,
        "structure": val_structures,
        "task_id": val_mpids,
        "target": np.zeros((len(val_mpids))),
    }
    if not cif:
        data.pop("structure")

    val_df = pd.DataFrame(data=data)

    val_df.to_csv(my_path("val.csv"), columns=["formula", "target"], index=False)


if __name__ == "__main__":
    generate_elasticity_data()
