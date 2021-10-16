"""
Download and partition elasticity data using Materials Project API.

Use pymatgen environment.

Created on Sat Sep 11 17:02:10 2021

@author: sterg
"""
from os import cpu_count
from os.path import join
from pathlib import Path
import pickle

# from tqdm import tqdm
from pqdm.processes import pqdm
from mat_discover.ElM2D.utils.Timer import Timer

import numpy as np
import pandas as pd


from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition


def structure_from_cif(cif):
    return Structure.from_str(cif, fmt="cif")


def generate_elasticity_data(download_data=True):
    """Download (or reload) elasticity data using MPRester."""
    # download and save Materials Project dataset
    data_dir = "data"
    elast_path = join(data_dir, "elast_results.pkl")
    all_path = join(data_dir, "all_results.pkl")

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if download_data:
        # download
        props = ["task_id", "pretty_formula", "elasticity", "cif"]
        with MPRester() as m:
            # TODO: don't download noble gases
            elast_results = m.query(
                {
                    "e_above_hull": {"$lt": 0.5},
                    "elasticity": {"$exists": True},
                    "elements": {"$nin": ["Tc", "He", "Ne", "Ar", "Kr", "Xe", "Rn"]},
                    # "pretty_formula": {"$nin": ["Tc","He", "Ne", "Ar", "Kr", "Xe", "Rn"]},
                },
                properties=props,
                chunk_size=2000,
            )

        props = ["task_id", "pretty_formula", "cif"]
        with MPRester() as m:
            all_results = m.query(
                {
                    "e_above_hull": {"$lt": 0.5},
                    "elements": {"$nin": ["Tc", "He", "Ne", "Ar", "Kr", "Xe", "Rn"]},
                    # "pretty_formula": {"$nin": ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]},
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

    crabnet_folder = join(
        "mat_discover", "CrabNet", "data", "materials_data", "elasticity"
    )
    Path(crabnet_folder).mkdir(parents=False, exist_ok=True)

    def crabnet_path(name):
        """Return a relative path to a CrabNet data file."""
        return join(".", crabnet_folder, name)

    # %% separate mpids and other properties for elasticity materials
    elast_mpids = [d["task_id"] for d in elast_results]
    elast_formulas = [d["pretty_formula"] for d in elast_results]
    elast_cifs = [d["cif"] for d in elast_results]
    elasticity = [d["elasticity"] for d in elast_results]
    K_VRH = [d["K_VRH"] for d in elasticity]

    del elast_results

    elast_comp = [Composition(formula) for formula in elast_formulas]

    elast_struct_path = join("data", "elast_struct_dicts.pkl")
    if download_data:
        elast_structures = [Structure.from_str(cif, fmt="cif") for cif in elast_cifs]
        elast_struct_dicts = [structure.as_dict() for structure in elast_structures]
        with open(elast_struct_path, "wb") as f:
            pickle.dump(elast_struct_dicts, f)
    else:
        with Timer("elast structures loaded"):
            with open(elast_struct_path, "rb") as f:
                elast_struct_dicts = pickle.load(f)
            elast_structures = [Structure.from_dict(s) for s in elast_struct_dicts]
            del elast_struct_dicts

    elast_df = pd.DataFrame(
        data={
            "formula": elast_formulas,
            "composition": elast_comp,
            "structure": elast_structures,
            "K_VRH": K_VRH,
            "task_id": elast_mpids,
            "target": K_VRH,
        }
    )

    elast_df.to_csv(
        crabnet_path("train.csv"), columns=["formula", "target"], index=False
    )

    # TODO: make separate "prediction" df that doesn't include training data
    # separate mpids and other properties for all data
    all_mpids = [d["task_id"] for d in all_results]
    all_cifs = [d["cif"] for d in all_results]
    all_formulas = [d["pretty_formula"] for d in all_results]

    del all_results

    all_comp = [Composition(formula) for formula in all_formulas]

    all_struct_path = join("data", "all_struct_dicts.pkl")
    if download_data:
        # TODO: add waitbar for list comp

        all_structures = pqdm(all_cifs, structure_from_cif, n_jobs=cpu_count())
        all_struct_dicts = [structure.as_dict() for structure in all_structures]
        with open(all_struct_path, "wb") as f:
            pickle.dump(all_struct_dicts, f)
    else:
        with Timer("all-structures-loaded"):
            with open(all_struct_path, "rb") as f:
                all_struct_dicts = pickle.load(f)
                all_structures = [Structure.from_dict(s) for s in all_struct_dicts]
                del all_struct_dicts

    # n = 10000
    # all_comp = all_comp[:n]
    # all_formulas = all_formulas[:n]
    # all_mpids = all_mpids[:n]
    # all_structures = all_structures[:n]

    all_df = pd.DataFrame(
        data={
            "composition": all_comp,
            "formula": all_formulas,
            "structure": all_structures,
            "task_id": all_mpids,
            "target": np.zeros((len(all_mpids))),
        }
    )

    all_df.to_csv(crabnet_path("val.csv"), columns=["formula", "target"], index=False)


if __name__ == "__main__":
    generate_elasticity_data()
