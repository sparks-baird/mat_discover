"""
Download and partition elasticity data using Materials Project API.

Use pymatgen environment.

Created on Sat Sep 11 17:02:10 2021

@author: sterg
"""
from os.path import join
import pickle
from ElM2D.utils.Timer import Timer

import numpy as np
import pandas as pd


from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition


def generate_elasticity_data(download_data=False):
    """Download (or reload) elasticity data using MPRester."""
    # download and save Materials Project dataset
    if download_data:
        # download
        props = ["task_id", "pretty_formula", "elasticity", "cif"]
        with MPRester() as m:
            elast_results = m.query(
                {"e_above_hull": {"$lt": 0.5}, "elasticity": {"$exists": True}},
                properties=props,
                chunk_size=2000,
            )

        props = ["task_id", "pretty_formula", "cif"]
        with MPRester() as m:
            all_results = m.query(
                {"e_above_hull": {"$lt": 0.5}}, properties=props, chunk_size=2000
            )

        # save
        with open("data/elast_results.pkl", "wb") as f:
            pickle.dump(elast_results, f)

        with open("data/all_results.pkl", "wb") as f:
            pickle.dump(all_results, f)
    else:
        # load the data
        with open("data/elast_results.pkl", "rb") as f:
            elast_results = pickle.load(f)

        with open("data/all_results.pkl", "rb") as f:
            all_results = pickle.load(f)

    crabnet_folder = join("CrabNet", "data", "materials_data", "elasticity")

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

    if download_data:
        elast_structures = [Structure.from_str(cif, fmt="cif") for cif in elast_cifs]
        elast_struct_dicts = [structure.as_dict() for structure in elast_structures]
        with open("data/elast_struct_dicts.pkl", "wb") as f:
            pickle.dump(elast_struct_dicts, f)
    else:
        with Timer("elast structures loaded"):
            with open("data/elast_struct_dicts.pkl", "rb") as f:
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

    if download_data:
        all_structures = [Structure.from_str(cif, fmt="cif") for cif in all_cifs]
        all_struct_dicts = [structure.as_dict() for structure in all_structures]
        with open("data/all_struct_dicts.pkl", "wb") as f:
            pickle.dump(all_struct_dicts, f)
    else:
        with Timer("all-structures-loaded"):
            with open("data/all_struct_dicts.pkl", "rb") as f:
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
