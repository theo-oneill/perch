# perch
### Persistent Homology for Images &amp; Cubes

[![tests](https://github.com/theo-oneill/perch/actions/workflows/main.yml/badge.svg)](https://github.com/theo-oneill/perch/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/theo-oneill/perch/branch/main/graph/badge.svg)](https://codecov.io/gh/theo-oneill/perch)
[![docs](https://readthedocs.org/projects/perch/badge/?version=latest)](https://perch.readthedocs.org)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

perch is an in-development python package for applying the topological technique of persistent homology to images & cubes.  Persistent Homology identifies "holes" of different dimensions in data.  In the perch framework, this enables the identification, segmentation, and hierarchical mapping of multi-scale structures in astronomical data.

#### Dependencies:

- [cubical ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim) 
   
- [cc3d](https://pypi.org/project/connected-components-3d/) 
    
- [jax](https://jax.readthedocs.io/en/latest/installation.html)
    
- numpy
   
- matplotlib
    
- tqdm


#### Running the tests

After installing the test extras (`pip install -e ".[test]"`), the
test suite runs in a few seconds:

```
pytest perch/tests/
```

Diagnostic plots of the synthetic fixtures (PH on toy peaks, ring, shell,
and the segmentation maps) are written when `--perch-plots` is passed.
The default output directory is `<repo>/test_plots/` (gitignored):

```
pytest perch/tests/ --perch-plots
pytest perch/tests/ --perch-plots=/custom/output/dir
```

Frozen-output regression tests live in `perch/tests/test_regression.py`
and compare against references in `perch/tests/data/*.npz`. After an
intentional change to `PH.compute_hom`, regenerate the references with:

```
python -m perch.tests.data._regenerate
```

The diff in the resulting `.npz` files is the change log for the
regression suite.

