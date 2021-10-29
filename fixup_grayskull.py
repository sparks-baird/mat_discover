"""Touch up the conda recipe from grayskull using conda-souschef."""
from souschef.recipe import Recipe

my_recipe = Recipe(load_file="meta.yaml")
my_recipe["requirements"]["host"].append("flit")
del my_recipe["requirements"]["build"]
my_recipe["requirements"]["run"].remove("kaleido")
my_recipe["requirements"]["run"].append("python-kaleido")
my_recipe["requirements"]["run"].append("pytorch==1.10.*")
my_recipe["requirements"]["run"].append("cudatoolkit==11.3.*")
my_recipe.save()
