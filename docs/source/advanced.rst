Advanced Usage
==============

This guide covers advanced perch features including 3D data analysis, FITS file integration, and specialized filtering techniques.

Working with 3D Data
--------------------

The workflow for 3D data cubes is the same as 2D, but you can compute higher-dimensional homology groups:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from perch.ph import PH

   # Load 3D data cube (nz, ny, nx)
   data_cube = np.random.randn(50, 100, 100)

   # Compute H₀, H₁, and H₂
   ph = PH.compute_hom(data_cube, max_Hi=2, verbose=True)

   # Filter H₀ structures (peaks)
   peaks = ph.filter(min_life=5.0, dimension=0)
   peaks.compute_segment_hierarchy(data_cube, verbose=True)

   # Filter H₂ structures (voids)
   voids = ph.filter(min_life=5.0, dimension=2)
   voids.compute_segment_hierarchy(data_cube, verbose=True)

   # Visualize a slice
   slice_idx = data_cube.shape[0] // 2
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   axes[0].imshow(data_cube[slice_idx], cmap='viridis')
   axes[0].set_title('Data Slice')
   axes[1].imshow(peaks.struc_map[slice_idx], cmap='tab10')
   axes[1].set_title('Peaks (H₀)')
   axes[2].imshow(voids.struc_map[slice_idx], cmap='tab10_r')
   axes[2].set_title('Voids (H₂)')
   plt.show()

Working with FITS Files
------------------------

perch integrates with astropy for astronomical FITS files and World Coordinate Systems (WCS):

.. code-block:: python

   from astropy.io import fits
   from astropy.wcs import WCS
   from perch.ph import PH

   # Load FITS file with WCS
   hdulist = fits.open('observation.fits')
   data = hdulist[0].data
   wcs = WCS(hdulist[0].header)

   # Compute with WCS (preserved throughout pipeline)
   ph = PH.compute_hom(data, max_Hi=1, wcs=wcs, verbose=True)

   # Filter and segment
   peaks = ph.filter(min_life=1.0, dimension=0)
   peaks.compute_segment_hierarchy(data, verbose=True)

   # Access WCS coordinates
   print(peaks.centroid_coord)  # Centroids in WCS coordinates (e.g., RA/Dec)

The WCS information is automatically preserved through filtering and segmentation, allowing you to work in physical coordinates rather than pixel coordinates.

Advanced Filtering Techniques
------------------------------

Noise-Normalized Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you have an estimate of the local noise level (e.g., from observational data or error propagation), you can filter structures based on their significance relative to the local noise environment. This is particularly useful for data with spatially varying noise levels.

.. code-block:: python

   # Assume you have a noise map from your observations
   noise_map = np.random.rand(*data.shape) * 0.5  # Example

   # Filter by persistence/noise ratio at death pixel
   gens = ph.generators
   death_pts = gens[:, 6:9].astype(int)  # Death pixel coordinates
   death_noise = noise_map[death_pts[:, 0], death_pts[:, 1]]  # For 2D
   pers_norm = np.abs(gens[:, 1] - gens[:, 2]) / death_noise

   # Select structures with normalized persistence > 10
   peaks = ph.filter(mask=np.where(pers_norm > 10))

This approach identifies structures that stand out significantly above the local noise, rather than using a global threshold that might miss faint but real structures in low-noise regions or include noise in high-noise regions.

Combining Multiple Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most real workflows combine several filtering criteria, since no single cut
captures everything you want. A typical filter picks a homology ``dimension``
(which kind of structure to keep), applies a persistence cut (``min_life``) to
suppress noise, optionally adds bounds on the data values themselves
(``min_birth`` / ``min_death``) to restrict to physically meaningful intensity
ranges, and may add a normalized-persistence cut (``min_life_norm_birth`` /
``min_life_norm_death``) when contrast varies across the field. ``filter``
applies the cuts conjunctively, so any structure that fails any one of them is
dropped.

.. code-block:: python

   # Select structures meeting multiple criteria
   peaks = ph.filter(
       dimension=0,
       min_life=5.0,          # Minimum persistence
       min_birth=10.0,        # Minimum birth value
       min_life_norm_birth=0.1  # Minimum normalized persistence
   )

Parameter reference:

- ``dimension``: homology group to keep (``0`` for peaks/H₀, ``1`` for loops/H₁,
  ``2`` for voids/H₂).
- ``min_life`` / ``max_life``: persistence (lifetime) bounds. This is the most
  common knob — everything below ``min_life`` is treated as noise.
- ``min_birth`` / ``max_birth``: bounds on the birth value. For H₀ this is the
  brightest pixel inside the structure; useful for excluding faint detections
  or capping above a saturation level.
- ``min_death`` / ``max_death``: bounds on the death value (the threshold at
  which the structure merges into a larger one).
- ``min_life_norm_birth`` / ``min_life_norm_death``: persistence normalized by
  birth (resp. death). Useful when absolute persistence varies across the field
  but the relative contrast of a feature is the more meaningful significance
  measure.
- ``mask``: when supplied, all other cuts are ignored and only the given
  boolean / index mask is applied to the generators — handy for plugging in a
  custom selection (e.g. the noise-normalized persistence above).

Alexander Duality for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large 3D datasets (billions of voxels), computing H₂ directly can be
slow. `Alexander duality <https://en.wikipedia.org/wiki/Alexander_duality>`_
tells us that the H₂ classes of a superlevel-set filtration are in one-to-one
correspondence with the H₀ classes of the same filtration on the *negated*
data — intuitively, voids in the original become peaks once the sign is
flipped. H₀ is dramatically cheaper to compute than H₂, so for void-finding at
scale we recommend the inverted-data route:

.. code-block:: python

   # For large datasets: compute voids as H₀ of inverted data
   # (equivalent to H₂ but much faster)
   ph_voids = PH.compute_hom(-data_cube, verbose=True)
   voids = ph_voids.filter(min_life=5.0, dimension=0)
   voids.compute_segment_hierarchy(-data_cube, verbose=True)

This computes H₀ on the inverted data, which is mathematically equivalent to H₂ on the original data but with significantly better performance for massive datasets.

Hierarchical Analysis
----------------------

Analyzing Parent-Child Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After segmentation, you can explore the hierarchical relationships between structures:

.. code-block:: python

   # Get all structures and build full hierarchy
   all_peaks = ph.strucs
   all_peaks.compute_segment_hierarchy(data, verbose=True)

   # Access hierarchy properties
   print("Parent IDs:", all_peaks.parent)
   print("Hierarchy levels:", all_peaks.level)
   print("Trunk (most persistent):", all_peaks.trunk)
   print("Leaves (no children):", all_peaks.leaves)

   # Analyze individual structure hierarchy
   sid = all_peaks.id[0]
   structure = all_peaks.structures[sid]
   print(f"Structure {sid}:")
   print(f"  Parent: {structure.parent}")
   print(f"  Children: {structure.children}")
   print(f"  Descendants: {structure.descendants}")
   print(f"  Level: {structure.level}")


Tips for Large Datasets
------------------------

1. **Memory management**: Use ``clear_indices=True`` (default) in ``compute_segment_hierarchy()``

2. **Alexander duality**: For 3D voids in billion-voxel datasets, use H₀ of inverted data instead of H₂

3. **Filtering before segmentation**: Apply aggressive filtering before segmentation to reduce computation

4. **Export intermediate results**: Save generators after computation to avoid recomputing during exploration

.. code-block:: python

   # Efficient workflow for large datasets
   ph = PH.compute_hom(large_data, max_Hi=0, verbose=True)
   ph.export_generators('large_ph.txt', odir='./output/')

   # Later: load and filter without recomputing
   ph_loaded = PH.load_from('large_ph.txt', odir='./output/',
                            data=large_data, wcs=wcs)
   peaks = ph_loaded.filter(min_life=50.0, dimension=0)
   peaks.compute_segment_hierarchy(large_data, verbose=True, clear_indices=True)

Further Resources
-----------------

- See :doc:`quickstart` for basic 2D workflow
- Check :doc:`concepts` for theoretical background
- Browse :doc:`api` for complete documentation
- Explore ``examples/run_perch_3d.ipynb`` for detailed 3D examples
