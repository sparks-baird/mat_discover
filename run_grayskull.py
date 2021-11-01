"""Touch up the conda recipe from grayskull using conda-souschef."""
from souschef.recipe import Recipe
import os
from os.path import join
from pathlib import Path
import mat_discover

name, version = mat_discover.__name__, mat_discover.__version__
os.system("grayskull pypi {0}=={1}".format(name, version))

Path("scratch").mkdir(exist_ok=True)

fpath = join(name, "meta.yaml")
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
my_recipe["requirements"]["host"].append("flit")
my_recipe["build"].add_section("noarch")
my_recipe["build"]["noarch"] = "python"
del my_recipe["requirements"]["build"]
my_recipe["requirements"]["run"].replace("kaleido", "python-kaleido")
my_recipe["requirements"]["run"].append("pytorch >=1.9.0")
my_recipe["requirements"]["run"].append("cudatoolkit <11.4")
my_recipe.save(fpath)
my_recipe.save(fpath2)
