'''
perch is a python package for applying the topological technique of persistent homology to
images & cubes.  Persistent Homology identifies "holes" of different dimensions in data.
In the perch framework, this enables the identification, segmentation,
and hierarchical mapping of multi-scale structures in astronomical data.
'''

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''
