
import numpy as np
import cc3d
import os
import jax.numpy as jnp
from perch.utils import filter_super_jit, filter_sub_jit

class Structure(object):

    '''
    Class for storing information about a topological structure in an image.

    Parameters
    ----------
    pi : array-like
        Array of persistence information. The first element is the homology type, the second is the birth time, the third is the death time, the next three are the birth pixel coordinates, and the last three are the death pixel coordinates.

    id : int
        Unique identifier for the structure, in the subset of selected structures.

    id_ph : int
        Unique identifier for the structure, in the full persistence diagram.

    img_shape : tuple
        Shape of the image the structure is segmented from.

    sdir : str
        Directory to save/load structure information.

    '''

    ###########################################################################
    ###########################################################################

    def __init__(self, pi = None, id=None, id_ph=None, img_shape = None, sdir='./'):

        if pi is not None:
            self._htype = int(pi[0])
            self._birth = pi[1]
            self._death = pi[2]
            self._birthpix = pi[3:6]
            self._deathpix = pi[6:9]

        if img_shape is not None:
            self._imgshape = img_shape
            self._ndim = len(img_shape)

        if id is not None:
            self._id = int(id)

        if id_ph is not None:
            self._id_ph = int(id_ph)

        self.sdir = sdir
        self._reset_cache()

    def _reset_cache(self):

        '''
        Reset all cached properties.
        '''

        self._indices = None
        self._values = None

        self._npix = None
        self._sum_val = None
        self._volume = None
        self._surface_area = None
        self._sphericity = None

        self._geom_cent = None
        self._weight_cent = None
        self._extreme_cent = None
        self._centroid = None
        self._bbox = None
        self._bbox_min = None
        self._bbox_max = None

        self._mask = None

        self._min_val = None
        self._max_val = None
        self._med_val = None

        ## only use these after all structures have been segemented,
        #   and hierarchy computed
        self._level = None
        self._parent = None
        self._children = []
        self._descendants = []

    ###################################################################

    @property
    def indices(self):
        '''
        Indices of the structure in the image.
        '''
        return self._indices

    @property
    def htype(self):
        '''
        Homology type of the structure.
        '''
        return self._htype

    @property
    def birth(self):
        '''
        Birth time of the structure.
        '''
        return self._birth

    @property
    def death(self):
        '''
        Death time of the structure.
        '''
        return self._death

    @property
    def persistence(self):
        '''
        Persistence of the structure.
        '''
        return np.abs(self._death - self._birth)

    @property
    def birthpix(self):
        '''
        Pixel coordinates of the birth of the structure.
        '''
        return np.array(self._birthpix,dtype='int')

    @property
    def deathpix(self):
        '''
        Pixel coordinates of the death of the structure.
        '''
        return np.array(self._deathpix,dtype='int')

    @property
    def npix(self):
        '''
        Number of pixels in the structure.
        '''
        return self._npix

    @property
    def sum_val(self):
        '''
        Sum of the values of the pixels in the structure.
        '''
        if self._sum_val is None:
            self._calculate_pix_values()
        return self._sum_val

    @property
    def min_val(self):
        '''
        Minimum value of the pixels in the structure.
        '''
        if self._min_val is None:
            self._calculate_pix_values()
        return self._min_val

    @property
    def max_val(self):
        '''
        Maximum value of the pixels in the structure.
        '''
        if self._max_val is None:
            self._calculate_pix_values()
        return self._max_val

    @property
    def med_val(self):
        '''
        Median value of the pixels in the structure.
        '''
        if self._med_val is None:
            self._calculate_pix_values()
        return self._med_val

    @property
    def volume(self):
        '''
        Volume of the structure.
        '''
        if self._volume is None:
            self._calculate_volume()
        return self._volume

    @property
    def surface_area(self):
        '''
        Surface area of the structure.
        '''
        if self._surface_area is None:
            self._calculate_surface_area()
        return self._surface_area

    @property
    def sphericity(self):
        '''
        Sphericity of the structure.
        '''
        if self._sphericity is None:
            self._calculate_sphericity()
        return self._sphericity

    @property
    def id(self):
        '''
        Unique identifier for the structure in the subsection of structures.
        '''
        return self._id

    @property
    def id_ph(self):
        '''
        Unique identifier for the structure in the full persistence diagram.
        '''
        return self._id_ph

    @property
    def geom_cent(self):
        '''
        Geometric center of the structure.
        '''
        if self._geom_cent is None:
            self._calculate_geom_cent()
        return self._geom_cent

    @property
    def centroid(self):
        '''
        Centroid of the structure.
        '''
        if self._centroid is None:
            self._calculate_centroid()
        return self._centroid

    @property
    def weight_cent(self):
        '''
        Weighted center of the structure.
        '''
        if self._weight_cent is None:
            self._calculate_weight_cent()
        return self._weight_cent

    @property
    def bbox(self):
        '''
        Bounding box of the structure in pixel coordinates.
        '''
        if self._bbox is None:
            self._calculate_bbox()
        return np.array([self._bbox_min, self._bbox_max])

    @property
    def bbox_min(self):
        '''
        Minimum coordinates of the bounding box of the structure.
        '''
        if self._bbox_min is None:
            self._calculate_bbox()
        return self._bbox_min

    @property
    def bbox_max(self):
        '''
        Maximum coordinates of the bounding box of the structure.
        '''
        if self._bbox_max is None:
            self._calculate_bbox()
        return self._bbox_max

    @property
    def extreme_pix(self):
        '''
        Pixel with the extreme value in the structure.
        '''
        if self._extreme_pix is None:
            self._calculate_extreme_pix()
        return self._extreme_pix

    @property
    def equiv_radius(self):
        '''
        Equivalent radius of the structure.
        '''
        # sphere
        if self._ndim == 3:
            return (3 * self.npix / (4*np.pi))**(1/3)
        # circle
        if self._ndim == 2:
            return (self.npix/np.pi)**0.5

    @property
    def level(self):
        '''
        Level of the structure in the hierarchy.
        '''
        return self._level

    @property
    def parent(self):
        '''
        Parent of the structure in the hierarchy.
        '''
        return self._parent

    @property
    def children(self):
        '''
        Children of the structure in the hierarchy.
        '''
        return self._children

    @property
    def descendants(self):
        '''
        Descendants of the structure in the hierarchy.
        '''
        return self._descendants

    @property
    def is_leaf(self):
        '''
        Check if the structure is a leaf in the hierarchy.
        '''
        return self.n_children == 0

    @property
    def n_children(self):
        '''
        Number of children of the structure in the hierarchy.
        '''
        return len(self.children)

    @property
    def n_descendants(self):
        '''
        Number of descendants of the structure in the hierarchy.
        '''
        return len(self.descendants)




    ###################################################################

    def compute_segment(self, img):
        '''
        Compute the segmentation of the structure in the image.

        Parameters
        ----------
        img : array-like
            Image to segment the structure from.

        '''

        # check if image is numpy array
        if type(img) is np.ndarray:
            img = jnp.array(img)

        if self.htype == 0:
            filt_img = np.array(filter_super_jit(img, self.death))
            labels_out = cc3d.connected_components(filt_img, connectivity=6)
            if self._ndim == 3:
                comp_use = labels_out[self.birthpix[0],self.birthpix[1],self.birthpix[2]]
            if self._ndim == 2:
                comp_use = labels_out[self.birthpix[0],self.birthpix[1]]

        if (self.htype == 1) and (self._ndim == 2):
            filt_img = np.array(filter_sub_jit(img, self.birth))
            labels_out = cc3d.connected_components(filt_img, connectivity=6)
            comp_use = labels_out[self.deathpix[0],self.deathpix[1]]

        if self.htype == 1 and self._ndim == 3:
            print('Segmentation for 3D H_1 not supported.')
            return

        if (self.htype == 2) and (self._ndim == 3):
            filt_img = np.array(filter_sub_jit(img, self.birth))
            labels_out = cc3d.connected_components(filt_img, connectivity=6)
            comp_use = labels_out[self.deathpix[0],self.deathpix[1],self.deathpix[2]]

        #uinds = np.where(labels_out == comp_use)
        self._indices =  np.where(labels_out == comp_use)#np.ravel_multi_index( np.where(labels_out == comp_use),self._imgshape)
        self._npix = len(self._indices[0])

    def _clear_indices(self):
        '''
        Clear indices.
        '''
        self._indices = None

    ##########################################################

    def get_mask(self):
        '''
        Get mask of the structure.
        '''
        if self.indices is None:
            print('Error: must compute or load segmentation first!')
            return
        mask = np.zeros(self._imgshape, dtype=bool)
        mask[self.indices] = True
        self._mask = mask
        return mask

    def get_values(self, img=None):
        '''
        Get image values of the structure.

        Parameters
        ----------
        img : array-like
            Image to get values from.
        '''
        if img is None:
            print('Error: must input image!')
            return
        if self.indices is None:
            print('Error: must compute or load segmentation first!')
            return
        return img[self.indices]

    def _calculate_pix_values(self, img =None):
        '''
        Calculate the sum of the image values of the structure

        Parameters
        ----------
        img : array-like
            Image to get values from.

        '''
        img_vals = self.get_values(img=img)
        self._sum_val = np.nansum(img_vals)
        self._min_val = np.nanmin(img_vals)
        self._max_val = np.nanmax(img_vals)
        self._med_val = np.nanmedian(img_vals)

    def _calculate_geom_cent(self):
        '''
        Calculate the geometric center of the structure.
        '''
        self._geom_cent = np.mean(self.indices,axis=1)

    def _calculate_centroid(self, img=None):
        from skimage import measure
        if img is None:
            print('Error: must input image!')
            return
        if self.indices is None:
            print('Error: must compute or load segmentation first!')
            return
        if self._mask is None:
            self.get_mask()
        centr = measure.centroid(np.where(self._mask, img, 0))
        self._centroid = centr


    def _calculate_weight_cent(self, img=None):
        '''
        Calculate the weighted center of the structure.

        Parameters
        ----------
        img : array-like
            Image to get values from.

        '''
        if img is None:
            print('Error: must input image!')
            return
        from scipy import ndimage
        if self.htype == 0:
            wcent = ndimage.center_of_mass(np.where(self.get_mask(), img, 0))
        if self.htype == 2:
            wcent = ndimage.center_of_mass(np.where(self.get_mask(), -img, 0))
        self._weight_cent = wcent

    def _calculate_extreme_pix(self, img=None):
        '''
        Calculate the pixel with the extreme value in the structure

        Parameters
        ----------
        img : array-like
            Image to get values from.

        '''
        if img is None:
            print('Error: must input image!')
            return
        if self.htype == 0:
            extr = self.indices[np.argmax(self.get_values(img=img))]
        if self.htype == 2:
            extr = self.indices[np.argmin(self.get_values(img=img))]
        self._extreme_pix = extr

    def _calculate_bbox(self):
        '''
        Calculate the bounding box of the structure in pixel coordinates.
        '''
        if self.indices is None:
            print('Error: must compute or load segmentation first!')
            return
        mins = np.min(self.indices, axis=1)
        maxs = np.max(self.indices, axis=1)
        self._bbox = np.array([mins, maxs])
        self._bbox_min = mins
        self._bbox_max = maxs

    def _calculate_surface_area(self, save_points=True,sdir='./'):
        '''
        Calculate the surface area of the structure using marching cubes.

        Parameters
        ----------
        save_points : bool
            Save the surface points.
        sdir : str
            Directory to save surface points.

        '''
        from skimage.measure import marching_cubes, mesh_surface_area
        smask = self.get_mask()
        march = marching_cubes(smask)
        surf_area = mesh_surface_area(march[0], march[1])
        self._surface_area = surf_area
        if save_points:
            fname = f'struc_{self.id_ph}_verts.txt'
            np.savetxt(f'{sdir}{fname}', march[0])

    def _calculate_volume(self):
        '''
        Calculate the volume of the structure
        '''
        self._volume = self.npix

    def _calculate_sphericity(self,sdir='./',save_points=False):
        '''
        Calculate the sphericity of the structure.

        Parameters
        ----------
        sdir : str
            Directory to save surface points.
        save_points : bool
            Save the surface points.

        '''
        if self.volume is None:
            self._calculate_volume()
        if self.surface_area is None:
            self._calculate_surface_area(save_points=save_points,sdir=sdir)
        self._sphericity = np.pi**(1/3) * (6* self.volume)**(2/3) / self.surface_area


    ###################################################################
