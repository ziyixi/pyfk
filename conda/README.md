# Build Conda packages for PyFK

## Build conda packages

1. install conda packages as: `mamba install boa conda-verify anaconda-client`.
2. Run `conda mambabuild .` inside this directory.
3. Run `anaconda login`.
4. Upload the packges using `anaconda upload`.

## Note for modifying meta.yaml

1. The package versions can refer to the lowest supported versions in pyproject.toml (As conda-build will try to find the appropriate one).