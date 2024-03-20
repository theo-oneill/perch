# perch
### Persistent Homology for Images &amp; Cubes

See examples folder for an example of running perch on 2D images and 3D volumes.  


#### Engines

Two engines are available for the Persistent Homology computation:

– C++ engine: cubical ripser

– Python engine: py_cripser

The C++ engine is generally recommended for speed, but requires image segmentation to be computed as a separate step in the perch workflow.  

The Python engine is much slower, but performs segmentation on the fly.  

#### Dependencies:

recommended but optional, for C++ engine:
    
– [cubical ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim) 
   
 – [cc3d](https://pypi.org/project/connected-components-3d/) 
    
– [jax](https://jax.readthedocs.io/en/latest/installation.html)

required for either engine:
    
– numpy
   
 – matplotlib
    
– tqdm

– astropy (optional, only required for WCS information)



