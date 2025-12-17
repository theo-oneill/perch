Quickstart Guide
================

This guide walks you through your first perch analysis with a complete 2D example.

Basic Workflow Overview
-----------------------

The typical perch workflow:

1. **Compute persistent homology**

.. code-block:: python

   from perch.ph import PH
   ph = PH.compute_hom(img, max_Hi=1, verbose=True)

2. **Filter significant structures**

.. code-block:: python

   strucs = ph.filter(min_life=1.0, dimension=0)

3. **Segment and analyze**

.. code-block:: python

   strucs.compute_segment_hierarchy(img, verbose=True)
   print(strucs.persistence)  # Access properties

Your First Analysis: 2D Example
--------------------------------

Create Test Data
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from perch.ph import PH
   from perch import pplot

   # Create test image with Gaussian peaks
   np.random.seed(42)
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)
   X, Y = np.meshgrid(x, y)
   data = (3.0 * np.exp(-((X-1)**2 + (Y-1)**2) / 2) +
           2.0 * np.exp(-((X+2)**2 + (Y+1)**2) / 3) +
           1.5 * np.exp(-((X-2)**2 + (Y+2)**2) / 2) +
           0.3 * np.random.randn(100, 100))

Step 1: Compute Persistent Homology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute H₀ (connected components) and H₁ (rings/loops)
   ph = PH.compute_hom(data, max_Hi=1, verbose=True)
   print(f"Found {ph.strucs.n_struc} structures")

Step 2: Filter Significant Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot persistence diagram to see all structures
   pplot.pers_diagram(ph)

The persistence diagram shows all detected structures. Points far from the diagonal have high persistence (signal), while points near the diagonal have low persistence (noise).

.. code-block:: python

   # Filter H₀ structures (peaks) with persistence > 0.5
   peaks = ph.filter(min_life=0.5, dimension=0)
   print(f"Selected {peaks.n_struc} significant peaks")

Step 3: Segment image and Build Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Segment structures and build parent-child hierarchy
   peaks.compute_segment_hierarchy(data, export=False, verbose=True)

Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Visualize segmentation
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   axes[0].imshow(data, origin='lower', cmap='viridis')
   axes[0].set_title('Original Data')
   axes[1].imshow(peaks.struc_map, origin='lower', cmap='tab10')
   axes[1].set_title('Peak Segmentation')
   plt.show()

Understanding Your Results
---------------------------

Accessing Structure Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After segmentation, you can access properties for all structures:

.. code-block:: python

   # Get properties for all structures (vectorized)
   print("Peak IDs:", peaks.id)
   print("Persistence:", peaks.persistence)
   print("Centroids:", peaks.centroid)
   print("Number of pixels:", peaks.npix)

   # Access individual structure
   sid = peaks.id[0]
   structure = peaks.structures[sid]
   print(f"Structure {sid}: birth={structure.birth:.2f}, pers={structure.persistence:.2f}")

Working with Segmented Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``struc_map`` attribute contains the segmented structures. Each pixel is labeled with the ID of the structure it belongs to (or NaN for background).

**Option 1: Create masks for specific structures**

.. code-block:: python

   # Get mask for a structure and its descendants
   struc_id = peaks.id[0]
   mask = peaks.get_mask(s_include=[struc_id], use_descendants=True)

   # Visualize just this structure
   masked_data = np.where(mask, data, np.nan)
   plt.imshow(masked_data, origin='lower')
   plt.title(f'Structure {struc_id} and descendants')
   plt.show()

**Option 2: Access pixel indices directly**

.. code-block:: python

   # Compute with indices retained (uses more memory)
   peaks.compute_segment_hierarchy(data, clear_indices=False, verbose=True)

   # Get pixel coordinates for a structure
   struc_id = peaks.id[0]
   pixel_indices = peaks.structures[struc_id].indices
   structure_values = data[pixel_indices]

Exploring Hierarchies
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access hierarchy information
   print("Parent IDs:", peaks.parent)
   print("Hierarchy levels:", peaks.level)
   print("Trunk (most persistent):", peaks.trunk)
   print("Leaves (no children):", peaks.leaves)

   # Individual structure hierarchy
   sid = peaks.id[0]
   structure = peaks.structures[sid]
   print(f"Parent: {structure.parent}")
   print(f"Children: {structure.children}")

Choosing Parameters
-------------------

max_Hi: Homology Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose based on features to detect:

- ``max_Hi=0``: Only connected components (peaks/clumps)
- ``max_Hi=1``: Components + rings (2D) or tunnels (3D)
- ``max_Hi=2``: All homologies including voids (3D only)

**Recommendation**: Use ``max_Hi=1`` for 2D data or ``max_Hi=2`` for 3D data.

Homology groups are computed in order (H₀, H₁, H₂). Higher dimensions increase computation time and memory usage.

Filtering Thresholds
~~~~~~~~~~~~~~~~~~~~

Common strategies:

1. **Visual inspection**: Plot persistence diagram, identify gap between noise and signal
2. **Fixed threshold**: Use domain knowledge (e.g., 5-sigma above noise)
3. **Normalized persistence**: ``min_life_norm_birth`` for scale-independent selection

.. code-block:: python

   # Absolute persistence
   strucs = ph.filter(min_life=5.0, dimension=0)

   # Normalized persistence (persistence/birth)
   strucs = ph.filter(min_life_norm_birth=0.2, dimension=0)

   # Combine multiple criteria
   strucs = ph.filter(min_life=5.0, min_birth=10.0, dimension=0)

Saving and Loading Results
---------------------------

Export Results
~~~~~~~~~~~~~~

.. code-block:: python

   # Export PH generators
   ph.export_generators('ph_results.txt', odir='./output/')

   # Export segmentation with properties
   peaks.compute_segment_hierarchy(
       data,
       export=True,
       fname='peaks',
       odir='./output/',
       calc_supp_props=True
   )

Load Saved Results
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load PH generators
   ph_loaded = PH.load_from(
       'ph_results.txt',
       odir='./output/',
       data=data,
       wcs=None
   )

   # Load segmentation
   from perch.structures import Structures
   peaks_loaded = Structures.load_from(
       fname='peaks',
       odir='./output/',
       verbose=True
   )

Next Steps
----------

Now that you understand the basics:

- See :doc:`advanced` for 3D data, FITS files, and advanced filtering techniques
- Check :doc:`concepts` for theoretical background on persistent homology
- Browse :doc:`api` for complete method and property documentation
