# perch
### Persistent Homology for Images &amp; Cubes

[![tests](https://github.com/theo-oneill/perch/actions/workflows/main.yml/badge.svg)](https://github.com/theo-oneill/perch/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/theo-oneill/perch/branch/main/graph/badge.svg)](https://codecov.io/gh/theo-oneill/perch)
[![docs](https://readthedocs.org/projects/perch/badge/?version=latest)](https://perch.readthedocs.org)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

perch is an in-development python package for applying the topological technique of persistent homology to images & cubes.  Persistent Homology identifies "holes" of different dimensions in data.  In the perch framework, this enables the identification, segmentation, and hierarchical mapping of multi-scale structures in astronomical data.

See examples folder for minimal examples of running perch on 2D images and 3D volumes.  


A few comments:

- perch performs superlevel set filtrations (as opposed to the default behavior of sublevel sets in cubical ripser), so that the zeroth homology (H<sub>0</sub>) represents connected high-valued components and higher level homologies  (H<sub>1</sub> and/or H<sub>2</sub>) represent high-valued rings and low-valued voids.

- The slowest and most memory-intensive step in either engine is typically the computation of the higher level homologies (H<sub>1</sub> and/or H<sub>2</sub>).  By the [Alexander duality](https://arxiv.org/abs/2005.04597), the highest level homology of an image (H<sub>1</sub> in 2D or H<sub>2</sub> in 3D) is equivalent to the zeroth homology (H<sub>0</sub>) of the inverted image.  If the intermediate homology is not of interest, computation of the highest homology with either engine can be sped up by taking advantage of this duality and computing only the zeroth homology of both the original and inverted images.  

- Computing and segmenting only the zeroth homology (H<sub>0</sub>) is equivalent to computing the dendrogram (hierarchical structure) of connected components, a la [astrodendro](https://github.com/dendrograms/astrodendro).

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

