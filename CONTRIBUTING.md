# Contributing
Thank you for your interest in contributing to `mat_discover`! We're glad you found this resource, and we hope it will be of use to you.
We welcome any contributions you might be willing to add, including suggestions, reporting bugs, or contributions to the codebase.

For feature requests and reporting bugs, please [open a GitHub issue](https://github.com/sparks-baird/mat_discover/issues/new/choose).

## Tools and Extensions
This project was developed primarily in [Python in Visual Studio Code](https://code.visualstudio.com/docs/languages/python) using `black`, `mypy`,
`pydocstyle`, `kite`, other tools, and various community extensions. Some other notable tools used in this project are:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (fewer issues during Python development than Anaconda with minimal downsides)
- [`pipreqs`](https://github.com/bndr/pipreqs) was used as a starting point for [`requirements.txt`](https://github.com/sparks-baird/mat_discover/blob/main/requirements.txt)
- [`flit`](https://flit.readthedocs.io/en/latest/) is used as a starting point for [`pyproject.toml`](https://github.com/sparks-baird/mat_discover/blob/main/pyproject.toml), 
to perform local installation, and to publish to PyPI
- [`conda env export --from-history -f environment.yml`](https://stackoverflow.com/a/64288844/13697228) was used as a starting point for [`environment.yml`](https://github.com/sparks-baird/mat_discover/blob/main/environment.yml)
- [`grayskull`](https://github.com/conda-incubator/grayskull) and [`conda-souschef`](https://github.com/marcelotrevisani/souschef) are used to generate and tweak [`meta.yaml`](https://github.com/sparks-baird/mat_discover/blob/main/mat_discover/meta.yaml),
  respectively, for publishing to Anaconda. See [this `conda-souschef` issue](https://github.com/marcelotrevisani/souschef/issues/32) for additional details.
- A variety of GitHub actions are used (see [workflows](https://github.com/sparks-baird/.github/workflows))
- [`pytest`](https://docs.pytest.org/en/7.0.x/) is used for testing
- [`numba`](https://numba.pydata.org/) is used to accelerate the Wasserstein distance matrix computations via CPU or GPU (which now happens in the external dependency, 
[`chem_wasserstein`](https://github.com/sparks-baird/chem_wasserstein) via [`dist-matrix`](https://github.com/sparks-baird/dist-matrix) package).

<!-- - `conda-smithy` is used to create a feedstock for `conda-forge` -->

For simple changes (e.g. quick-fix in documentation), navigate to https://github.com/sparks-baird/mat_discover, click on the
relevant file (e.g. `README.md`), and look for the pencil (✏️). GitHub will walk you through the rest.

## Development
This goes over installation and, if desired, VS Code Settings Sync.

### Installation
To help with development, we suggest that you [install from source](README.md#from-source). Note that when using a `conda` environment
(recommended), you may avoid certain issues down the road by opening VS Code via an
Anaconda command prompt and entering the command `code` (at least until the VS Code devs
fix some of the issues associated with opening it "normally"). For example, in Windows,
press the "Windows" key, type "anaconda", and open "Anaconda Powershell Prompt
(miniconda3)" or similar. Then type `code` and press enter.

The most seamless install that is least likely to cause issues will be using `flit`. For example, starting from the top:
```bash
git clone https://github.com/sparks-baird/mat_discover.git
cd mat_discover
conda create -n python==3.9.*
conda install flit
flit install --pth-file
```

From the [`flit` docs](https://flit.readthedocs.io/en/latest/cmdline.html):
> `--pth-file`
> Create a `.pth` file in site-packages rather than copying the module, so you can test changes without reinstalling.
> This is a less elegant alternative to `--symlink`, but it works on Windows, which typically doesn’t allow symlinks.

### Settings Sync
This should ensure that your setup is similar to the one used for development and testing. `flit` is also used to publish new PyPI versions
and to prep metadata for `grayskull` which creates `conda` recipes that are used to publish to Anaconda. Additionally, you may find it useful to use my
"secret sauce" list of VS Code extensions and settings by using the [Settings Sync](https://marketplace.visualstudio.com/items?itemName=Shan.code-settings-sync)
extension and inputting the GitHub Gist ID ([`98ade0073783c7dd54c50d5c8105d07d`](https://gist.github.com/sgbaird/98ade0073783c7dd54c50d5c8105d07d))
for my Settings Sync file when asked. You can keep it up-to-date with my settings, use these as a starting point
(by disconnecting Settings Sync or removing the Gist ID after the initial download and sync), or pick and choose from the list as you see fit.
By no means is it required, but these have made the Python development experience a lot more enjoyable for me.

## Documentation
To build the docs, if you did a local installation via `flit install --pth-file`, you can ignore the following installation steps.

Ensure that `sphinx`, `sphinx_rtd_theme`, and `sphinx_copy_button` are installed:
```bash
conda install sphinx sphinx_rtd_theme sphinx_copy_button
```

Alternatively, make sure all files from requirements.txt are installed:
```bash
conda install --file requirements.txt
```

Then run:
```bash
cd docs/
make html
```

Open `docs/build/index.html` (e.g. via `start build/index.html` on Windows, or by clicking on `index.html` in a file browser)
to view the documentation on your internet browser (e.g. Google Chrome).
