import numpy  as np
from tqdm import tqdm
from perch.perch_structure import *

class Structures(object):

    ###########################################################################
    ###########################################################################

    def __init__(self, structures = None, img_shape = None, wcs = None, struc_map = None, inds_dir='./'):

        if type(structures) == np.ndarray and np.shape(structures)[1] == 10:
            strucdict = {}
            for i in range(len(structures)):
                struc = Structure(pi=structures[i][0:9], img_shape=img_shape, id_ph=structures[i][9], id=i, sdir=inds_dir)
                strucdict[i] = struc
            self.structures = strucdict

        if type(structures) == dict:
            self.structures = structures

        if img_shape is not None:
            self._imgshape = img_shape
            self._ndim = len(img_shape)

        if wcs is not None:
            self.wcs = wcs

        self.struc_map = struc_map
        self.level_map = None

        self.sdir = inds_dir

    ###########################################################################
    ###########################################################################

    @property
    def trunk(self):
        return self._trunk

    @property
    def structure_keys(self):
        return self.structures.keys()

    @property
    def all_structures(self):
        return [self.structures[j] for j in self.structure_keys]

    @property
    def leaves(self):
        return self._leaves

    #@property
    def saved_indices_exist(self):
        return [self.structures[j].saved_indices_exist(sdir=self.sdir) for j in self.structure_keys]

    @property
    def n_struc(self):
        return len(self.structure_keys)

    @property
    def npix(self):
        return np.array([self.structures[i].npix for i in self.structure_keys])

    @property
    def birth(self):
        return np.array([self.structures[i].birth for i in self.structure_keys])

    @property
    def death(self):
        return np.array([self.structures[i].death for i in self.structure_keys])

    @property
    def equiv_radius(self):
        return np.array([self.structures[i].equiv_radius for i in self.structure_keys])

    @property
    def birthpix(self):
        return np.array([self.structures[i].birthpix for i in self.structure_keys])

    @property
    def deathpix(self):
        return np.array([self.structures[i].deathpix for i in self.structure_keys])

    @property
    def htype(self):
        return np.array([self.structures[i].htype for i in self.structure_keys])

    @property
    def persistence(self):
        return np.array([self.structures[i].persistence for i in self.structure_keys])

    @property
    def sphericity(self):
        return np.array([self.structures[i].sphericity for i in self.structure_keys])

    @property
    def id(self):
        return np.array([self.structures[i].id for i in self.structure_keys])

    @property
    def id_ph(self):
        return np.array([self.structures[i].id_ph for i in self.structure_keys])

    @property
    def geom_cent(self):
        return np.array([self.structures[i].geom_cent for i in self.structure_keys])

    @property
    def parent(self):
        return np.array([self.structures[i].parent for i in self.structure_keys])

    @property
    def children(self):
        return np.array([self.structures[i].children for i in self.structure_keys])

    @property
    def descendants(self):
        return np.array([self.structures[i].descendants for i in self.structure_keys])

    @property
    def n_children(self):
        return np.array([self.structures[i].n_children for i in self.structure_keys])

    @property
    def n_descendants(self):
        return np.array([self.structures[i].n_descendants for i in self.structure_keys])

    @property
    def is_leaf(self):
        return np.array([self.structures[i].is_leaf for i in self.structure_keys])

    #####################################################

    @property
    def birthpix_coord(self):
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return self.wcs.pixel_to_world(self.birthpix[:,0],self.birthpix[:,1],self.birthpix[:,2])

    @property
    def deathpix_coord(self):
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return self.wcs.pixel_to_world(self.deathpix)

    @property
    def geom_cent_coord(self):
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return self.wcs.pixel_to_world(self.geom_cent)

    @property
    def equiv_radius_coord(self):
        ## NOTE: ASSUMES EVEN PIXEL SCALES!!!
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return np.diag(self.wcs.pixel_scale_matrix)[0] * self.equiv_radius

    @property
    def volume_coord(self):
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        volpc_pixel = np.product(np.diag(self.wcs.pixel_scale_matrix))  # pc^3 / voxel or pc^2/pixel
        return self.npix * volpc_pixel

    ###########################################################################
    ###########################################################################

    def load_mask(self, s_include = None):
        if s_include is None:
            s_include = self.structure_keys
        if (len(s_include) == 1 )& (type(s_include) != list):
            s_include = list(s_include)
        mask = np.zeros(self._imgshape, dtype=bool)
        for s in s_include:
            struc = self.structures[s]
            struc.load_indices(sdir = self.sdir)
            struc_multi_inds = np.unravel_index(struc.indices, self._imgshape)
            struc.clear_indices()
            mask[struc_multi_inds] = True
        return mask

    def get_mask(self, s_include = None, use_descendants=True):
        if s_include is None:
            s_include = self.structure_keys
        if use_descendants:
            s_include = np.unique(self.s_descendants[s_include])
        mask = np.isin(self.struc_map, s_include)
        return mask

    def sort_keys(self, s_include = None):
        if s_include is None:
            s_include = list(self.structure_keys)
        return np.argsort(self.death[s_include])

    def sort_birth(self, s_include = None):
        if s_include is None:
            s_include = list(self.structure_keys)
        return np.argsort(self.birth[s_include])

    def _set_descendants(self, s = None):
        """
        Set descendants as a flattened list of all child leaves and branches.
        """
        if s is None:
            return

        struc = self.structures[s]
        if len(struc._descendants) == 0:
            struc._descendants = [struc.id]
            to_add = [struc.id]  # branches with children we will need to add to the list
            while True:
                children = []
                list(map(children.extend, [branch.children for branch in [self.structures[j] for j in to_add]]))
                self.structures[s]._descendants.extend(children)
                # Then proceed, essentially recursing through child branches:
                to_add = [b.id for b in [self.structures[j] for j in children] if not b.is_leaf]
                if not to_add:
                    break
        #return self.structures[s]._descendants

    def _clear_hierarchy(self):
        #print('Warning: clearing hierarchy!!!')
        for s in list(self.structure_keys):
            self.structures[s]._parent = None
            self.structures[s]._children = []
            self.structures[s]._descendants = []


    def id_mask(self,s_include=None):
        if s_include is None:
            s_include = self.structure_keys
        if (len(s_include) == 1 )& (type(s_include) != list):
            s_include = list(s_include)
        mask = np.full(self._imgshape, np.nan)
        for s in s_include:
            struc = self.structures[s]
            struc.load_indices(sdir = self.sdir)
            struc_multi_inds = np.unravel_index(struc.indices, self._imgshape)
            mask[struc_multi_inds] = int(s)
            struc.clear_indices()
        return mask

    ########################################################################
    def compute_segment_hierarchy(self, img_jnp=None,  s_include = None, clobber=True):

        mask_s = np.full((self._imgshape),np.nan)

        if clobber:
            self._clear_hierarchy()

        ### calculate parents
        ascend_death = self.sort_keys(s_include=s_include)
        pbar = tqdm(total=len(ascend_death), unit='structures')
        for s in ascend_death:
            struc = self.structures[s]
            struc.compute_segment(img=img_jnp, verbose=False)
            struc_multi_inds = np.unravel_index(struc.indices, self._imgshape)
            parent_cand = np.nanmax(mask_s[struc_multi_inds])
            if np.isfinite(parent_cand):
                self.structures[s]._parent = int(parent_cand)
            mask_s[struc_multi_inds] = int(s)
            struc.clear_indices()
            pbar.update(1)

        ### label children
        has_parent = np.where(self.parent != None)[0]
        for s in has_parent:
            s_parent = self.structures[s].parent
            self.structures[s_parent]._children.extend([int(s)])
        self._trunk = [structure for structure in self.all_structures if structure.parent == None]
        self._leaves = [structure for structure in self.all_structures if structure.is_leaf]

        for s in ascend_death:
            self._set_descendants(s)
        self.struc_map = mask_s

    ########################################################################
    def compute_segment(self, img_jnp=None):
        pbar = tqdm(total=self.n_struc, unit='structures')
        for i in range(self.n_struc):
            struc = self.structures[i]
            if struc.saved_indices_exist():
                struc.load_indices()
            if not struc.saved_indices_exist():
                struc.compute_segment(img=img_jnp)
                struc.save_indices()
            struc.clear_indices()
            pbar.update(1)

    def compute_hierarchy(self,  s_include = None, return_masks=False, clobber=True):
        mask_count = np.full((self._imgshape), 0.)
        mask_s = np.full((self._imgshape),np.nan)

        if clobber:
            self._clear_hierarchy()

        ### calculate parents
        ascend_death = self.sort_keys(s_include=s_include)
        pbar = tqdm(total=len(ascend_death), unit='structures')
        for s in ascend_death:
            struc = self.structures[s]
            if struc.saved_indices_exist():
                struc.load_indices()
                struc_multi_inds = np.unravel_index(struc.indices,self._imgshape)
                mask_count[struc_multi_inds] += 1
                self.structures[s]._level = np.nanmax(mask_count[struc_multi_inds])
                parent_cand = np.nanmax(mask_s[struc_multi_inds])
                if np.isfinite(parent_cand):
                    self.structures[s]._parent = int(parent_cand)
                mask_s[struc_multi_inds] = int(s)
                struc.clear_indices()
            pbar.update(1)

        ### label children
        has_parent = np.where(self.parent != None)[0]
        for s in has_parent:
            s_parent = self.structures[s].parent
            self.structures[s_parent]._children.extend([int(s)])
        self._trunk = [structure for structure in self.all_structures if structure.parent == None]
        self._leaves = [structure for structure in self.all_structures if structure.is_leaf]

        for s in ascend_death:
            self._set_descendants(s)

        self.struc_map = mask_s
        self.level_map = mask_count

        if return_masks:
            return mask_s# mask_count,


    ########################################################################
    def export_struc_map(self, fname='run', odir = './'):
        if self.struc_map is None:
            print('Error: must compute or load hierarchy first!')
            return

        from astropy.io import fits
        if self.wcs is not None:
            head = self.wcs.to_header()
        else:
            head = None
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data=np.array(self.struc_map, dtype='float32'), header=head))
        hdul.writeto(f'{odir}{fname}_struc_map.fits', overwrite=True)

    def load_struc_map(self, fname='run', odir = './'):
        from astropy.io import fits
        hdul = fits.open(f'{odir}{fname}_struc_map.fits')
        self.struc_map = hdul[0].data

    def clear_struc_map(self):
        self.struc_map = None

    def export_collection(self, fname='run', odir='./', include_map=False):
        import pickle
        pickle.dump(self, open(f'{odir}{fname}_structures.p', 'wb'))

















