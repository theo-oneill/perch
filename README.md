# perch
### Persistent Homology for Images &amp; Cubes

perch is an in-development python package for applying the topological technique of persistent homology to images & cubes.  Persistent Homology identifies "holes" of different dimensions in data.  In the perch framework, this enables the identification, segmentation, and hierarchical mapping of multi-scale structures in astronomical data.

See examples folder for minimal examples of running perch on 2D images and 3D volumes.  


#### Engines

Two engines are available for the Persistent Homology computation:

- C++ engine:  [cubical ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim) 

- Python engine: py_cripser

The C++ engine is the default engine and is generally recommended for speed, but requires image segmentation to be computed as a separate step in the perch workflow.  

The Python engine is currently not recommended (as it is *much* slower), but performs segmentation of all structures on the fly.  

A few comments:

- perch performs superlevel set filtrations (as opposed to the default behavior of sublevel sets in cubical ripser), so that the zeroth homology (H<sub>0</sub>) represents connected high-valued components and higher level homologies  (H<sub>1</sub> and/or H<sub>2</sub>) represent high-valued rings and low-valued voids.

- The slowest and most memory-intensive step in either engine is typically the computation of the higher level homologies (H<sub>1</sub> and/or H<sub>2</sub>).  By the [Alexander duality](https://arxiv.org/abs/2005.04597), the highest level homology of an image (H<sub>1</sub> in 2D or H<sub>2</sub> in 3D) is equivalent to the zeroth homology (H<sub>0</sub>) of the inverted image.  If the intermediate homology is not of interest, computation of the highest homology with either engine can be sped up by taking advantage of this duality and computing only the zeroth homology of both the original and inverted images (by setting the parameter maxdim=0).  (This is particularly useful for the Python engine, as the computation of H<sub>1</sub> and  H<sub>2</sub> is currently extremely slow!)

- Computing and segmenting only the zeroth homology (H<sub>0</sub>) is equivalent to computing the dendrogram (hierarchical structure) of connected components, a la [astrodendro](https://github.com/dendrograms/astrodendro).

#### Dependencies:

required for C++ engine:
    
- [cubical ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim) 
   
- [cc3d](https://pypi.org/project/connected-components-3d/) 
    
- [jax](https://jax.readthedocs.io/en/latest/installation.html)

for either engine:
    
- numpy
   
- matplotlib
    
- tqdm


