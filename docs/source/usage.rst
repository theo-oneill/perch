Usage
=====

.. _installation:

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

To install perch from source:

.. code-block:: bash

   git clone https://github.com/theo-oneill/perch.git
   cd perch
   pip install -e .

This will install the core dependencies:

- numpy
- matplotlib
- tqdm
- jax

Perch requires the following external libraries for the PH computation and structure segmentation:

.. code-block:: bash

   pip install cripser cc3d

- ``cripser``: Fast C++ library (cubical ripser) for PH computation
- ``cc3d``: Connected components for structure segmentation

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For additional functionality, install these optional packages:

**Astronomical Data Support:**

.. code-block:: bash

   pip install astropy

This enables FITS file I/O and World Coordinate System (WCS) support.

**Data Analysis:**

.. code-block:: bash

   pip install pandas scikit-image

- ``pandas``: Export structures to DataFrame for analysis
- ``scikit-image``: Advanced geometric property calculations

For detailed workflow and examples, see the :doc:`quickstart` guide.

