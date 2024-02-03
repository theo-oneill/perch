
import numpy as np
import cc3d
import os
from jax import jit
import jax.numpy as jnp

def filter_super(X,thresh):
    return X > thresh

def filter_sub(X,thresh):
    return X < thresh

filter_super_jit = jit(filter_super)
filter_sub_jit = jit(filter_sub)

class Structure(object):

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

        self._indices = None
        self._values = None

        self._npix = None
        self._sum_values = None
        self._volume = None
        self._surface_area = None
        self._sphericity = None

        self._geom_cent = None
        self._weight_cent = None
        self._extreme_cent = None

        ## only use these after all structures have been segemented,
        #   and hierarchy computed
        self._level = None
        self._parent = None
        self._children = []
        self._descendants = []

    ###################################################################

    @property
    def indices(self):
        return self._indices

    @property
    def htype(self):
        return self._htype

    @property
    def birth(self):
        return self._birth

    @property
    def death(self):
        return self._death

    @property
    def persistence(self):
        return np.abs(self._death - self._birth)

    @property
    def birthpix(self):
        return np.array(self._birthpix,dtype='int')

    @property
    def deathpix(self):
        return np.array(self._deathpix,dtype='int')

    @property
    def npix(self):
        return self._npix

    @property
    def sum_values(self):
        return self._sum_values

    @property
    def volume(self):
        return self._volume

    @property
    def surface_area(self):
        return self._surface_area

    @property
    def sphericity(self):
        return self._sphericity

    @property
    def id(self):
        return self._id

    @property
    def id_ph(self):
        return self._id_ph

    @property
    def geom_cent(self):
        return self._geom_cent

    @property
    def weight_cent(self):
        return self._weight_cent

    @property
    def extreme_pixt(self):
        return self._extreme_pix

    @property
    def equiv_radius(self):
        # sphere
        if self._ndim == 3:
            return (3 * self.npix / (4*np.pi))**(1/3)
        # circle
        if self._ndim == 2:
            return (self.npix/np.pi)**0.5

    @property
    def level(self):
        return self._level

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def descendants(self):
        return self._descendants

    @property
    def is_leaf(self):
        return self.n_children == 0

    @property
    def n_children(self):
        return len(self.children)

    @property
    def n_descendants(self):
        return len(self.descendants)


    ###################################################################

    def compute_segment(self, img):

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
        self._indices = np.ravel_multi_index( np.where(labels_out == comp_use),self._imgshape)
        self._npix = len(self._indices)

    def saved_indices_exist(self):
        fname = f'struc_{self.id_ph}_inds.txt'
        return os.path.isfile(f'{self.sdir}{fname}')

    def load_indices(self):
        fname = f'struc_{self.id_ph}_inds.txt'
        #self._indices = tuple(np.loadtxt(f'{sdir}{fname}').astype('int'))
        self._indices = np.loadtxt(f'{self.sdir}{fname}').astype('int')
        #if len(np.shape(self._indices)) == 0:
        #    i0 =  tuple(np.loadtxt(f'{sdir}{fname}').astype('int'))
        #    self._indices = tuple([np.array([i0[i]]) for i in range(len(i0))])
        self._npix = self._indices.size

    def save_indices(self):
        fname = f'struc_{self.id_ph}_inds.txt'
        np.savetxt(f'{self.sdir}{fname}', self.indices, fmt='%i')

    def clear_indices(self):
        self._indices = None

    def set_indices(self, inds):
        self._indices = inds

    ##########################################################

    def get_mask(self):
        if self.indices is None:
            print('Error: must compute or load segmentation first!')
            return
        mask = np.zeros(self._imgshape, dtype=bool)
        mask[self.indices] = True
        return mask

    def get_values(self, img=None):
        if img is None:
            print('Error: must input image!')
            return
        if self.indices is None:
            print('Error: must compute or load segmentation first!')
            return
        return img[self.indices]

    def calculate_sum_values(self, img =None):
        self._sum_values = np.nansum(self.get_values(img=img))

    def calculate_geom_cent(self):
        self._geom_cent = np.mean(self.indices,axis=1)

    def calculate_weight_cent(self, img=None):
        if img is None:
            print('Error: must input image!')
            return
        from scipy import ndimage
        if self.htype == 0:
            wcent = ndimage.center_of_mass(np.where(self.get_mask(), img, 0))
        if self.htype == 2:
            wcent = ndimage.center_of_mass(np.where(self.get_mask(), -img, 0))
        self._weight_cent = wcent

    def calculate_extreme_pix(self, img=None):
        if img is None:
            print('Error: must input image!')
            return
        if self.htype == 0:
            extr = self.indices[np.argmax(self.get_values(img=img))]
        if self.htype == 2:
            extr = self.indices[np.argmin(self.get_values(img=img))]
        self._extreme_pix = extr

    def calculate_surface_area(self, save_points=True,sdir='./'):
        from skimage.measure import marching_cubes, mesh_surface_area
        smask = self.get_mask()
        march = marching_cubes(smask)
        surf_area = mesh_surface_area(march[0], march[1])
        self._surface_area = surf_area
        if save_points:
            fname = f'struc_{self.id_ph}_verts.txt'
            np.savetxt(f'{sdir}{fname}', march[0])

    def calculate_volume(self):
        self._volume = self.npix

    def calculate_sphericity(self,sdir='./',save_points=False):
        if self.volume is None:
            self.calculate_volume()
        if self.surface_area is None:
            self.calculate_surface_area(save_points=save_points,sdir=sdir)
        self._sphericity = np.pi**(1/3) * (6* self.volume)**(2/3) / self.surface_area


    ###################################################################
