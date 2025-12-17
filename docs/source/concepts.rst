Conceptual Introduction
=======================

What Can Persistent Homology Do?
---------------------------------

Persistent homology provides a framework for identifying structures in data based on their topological properties. Unlike traditional methods that rely on fixed thresholds, persistent homology analyzes how structures evolve across all possible threshold levels, capturing features that persist over a wide range of scales.

**Key Capabilities of perch:**

- **Multi-scale structure detection**: Automatically identify peaks, filaments, voids, and other topological features across all scales simultaneously
- **Significance ranking**: Quantify which structures are robust signal versus noise using persistence
- **Hierarchical relationships**: Understand how structures nest within each other and merge across scales

This makes persistent homology particularly valuable for analyzing complex datasets where features exist at multiple scales and traditional fixed-threshold methods would miss important structures or introduce bias.

Homology Dimensions
-------------------

Persistent homology identifies topological features characterized by their dimension. The maximum homology group that can be computed depends on the data dimensionality: for 2D images, the maximum is H₁; for 3D cubes, it is H₂.

H₀: Connected Components
~~~~~~~~~~~~~~~~~~~~~~~~~

**H₀** identifies connected high-valued regions:

- **In 2D and 3D Data**: Peaks, clumps, bright sources
- E.g., stars, molecular clouds, galaxies

H₁: Loops and Rings
~~~~~~~~~~~~~~~~~~~

**H₁** identifies one-dimensional "holes":

- **In 2D Images**: Rings or loops around low-valued voids
- **In 3D Cubes**: Tunnels, tube-like voids
- E.g., bubbles (in 2D), tunnels in porous media (in 3D)

H₂: Voids and Cavities
~~~~~~~~~~~~~~~~~~~~~~

**H₂** identifies two-dimensional "holes":

- **In 3D Cubes**: Surfaces surrounding low-valued voids
- E.g., bubbles, cavities, cosmic voids in large-scale structure

How Perch Works: Superlevel Set Filtration
-------------------------------------------

Perch analyzes your data by sweeping through threshold values from **high to low** (superlevel sets):

1. **Start at highest values**: Begin at the maximum data value where nothing is above threshold
2. **Lower threshold progressively**: As the threshold decreases, connected regions of high-valued pixels appear
3. **Track structure evolution**: Monitor when structures first appear and when they merge with other structures
4. **Build complete picture**: Continue until reaching the minimum data value

This approach is more intuitive for scientific data where high values typically indicate features of interest (bright sources, high densities, strong signals).

**Technical note**: Internally, perch negates your data to convert this superlevel set problem into the standard sublevel set formulation used in computational topology.

Birth, Death, and Persistence
------------------------------

As the threshold sweeps from high to low values, structures appear and disappear:

**Birth**: The data value where a structure first appears (for H₀: the maximum value within the component).

**Death**: The data value where a structure disappears or merges with another structure.

**Persistence**: The lifetime of a structure:

.. math::

   \text{persistence} = |\  \text{death} - \text{birth} \ |

**High persistence** = robust feature existing across a wide range of thresholds. **Low persistence** = likely noise or minor fluctuation.

Persistence provides a measure of structure significance.

.. note::

   Because perch uses superlevel sets, birth values are higher than death values (opposite to standard persistent homology convention).

Persistence Diagrams
--------------------

A **persistence diagram** visualizes all structures by plotting birth vs. death:

- Each point represents one structure
- Distance from diagonal = persistence
- Near diagonal = low persistence (noise)
- Far from diagonal = high persistence (signal)

Persistence diagrams help you identify appropriate filtering thresholds and distinguish signal from noise.

Hierarchical Structure
----------------------

Segmentation Order and Parent Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When computing segmentation hierarchies, perch processes structures in a specific order to build parent-child relationships:

**For H₀ structures:**

1. Structures are segmented in **ascending order of death** (earliest deaths first)
2. As each structure is segmented, any previously segmented structures that overlap it become **parent candidates**
3. The parent is selected as the overlapping structure with the **highest death value** (most persistent)
4. This means less persistent structures naturally merge into more persistent ones

**For higher-dimensional structures (H₁, H₂):**

1. Structures are segmented in **descending order of birth** (highest births first)
2. The parent is selected as the overlapping structure with the **lowest birth value**

**Hierarchy levels:**

The level of a structure is determined by the maximum number of overlapping structures at any pixel within it. Higher levels indicate deeper nesting in the hierarchy.

Parent-Child Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~

The segmentation process creates a natural hierarchy, analogous to the structure of dendrograms, and enables multi-scale analysis of nested structures.

- **Child**: A structure that overlaps with and merges into another structure
- **Parent**: The structure that a child merges into
- **Trunk**: The most persistent structure(s) with no parent
- **Leaves**: Structures with no children (lowest in hierarchy)

**Descendants**: All structures that eventually merge into a given structure (children, grandchildren, etc.)


Alexander Duality
-----------------

The highest homology dimension (H₁ in 2D, H₂ in 3D) is mathematically equivalent to H₀ of the inverted data.

For very large datasets (billions of voxels), computing H₂ directly can be slow. In these cases, computing H₀ of inverted data provides the same result with better performance.
