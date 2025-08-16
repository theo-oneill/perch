import numpy  as np
from tqdm import tqdm
from perch.structure import *
import jax.numpy as jnp


class Structures(object):

    '''
    Class for managing a collection of Structure objects.

    Parameters:
    -----------
    structures : dict
        Dictionary of Structure objects.
    img_shape : tuple
        Shape of the image.
    wcs : astropy.wcs.WCS
        World Coordinate System of the image.
    struc_map : np.ndarray
        Map of structure IDs.
    level_map : np.ndarray
        Map of structure levels.

    '''

    ###########################################################################
    ###########################################################################

    def __init__(self, structures = None, img_shape = None, wcs = None,
                 struc_map = None, level_map=None, inds_dir='./'):

        if type(structures) == np.ndarray and np.shape(structures)[1] == 10:
            strucdict = {}
            for i in range(len(structures)):
                struc = Structure(pi=structures[i][0:9], img_shape=img_shape, id_ph=structures[i][9], id=i, sdir=inds_dir)
                strucdict[i] = struc
            self.structures = strucdict

        if type(structures) == list and type(structures[0]) == Structure:
            strucdict = {}
            for i in range(len(structures)):
                strucdict[i] = structures[i]
            self.structures = strucdict

        if type(structures) == dict:
            self.structures = structures

        if img_shape is not None:
            self._imgshape = img_shape
            self._ndim = len(img_shape)

        self.wcs = wcs

        self.struc_map = struc_map
        self.level_map = level_map

        self._frac_npix_parent = None

        self.sdir = inds_dir

    ###########################################################################
    ###########################################################################

    @property
    def trunk(self):
        '''
        Return the trunk of the structure hierarchy.
        '''
        return self._trunk

    @property
    def structure_keys(self):
        '''
        Return the keys of the structures dictionary.
        '''
        return self.structures.keys()

    @property
    def all_structures(self):
        '''
        Return a list of all structures.
        '''
        return [self.structures[j] for j in self.structure_keys]

    @property
    def leaves(self):
        '''
        Return the leaves of the structure hierarchy.
        '''
        return self._leaves

    @property
    def n_struc(self):
        '''
        Return the number of structures.
        '''
        return len(self.structure_keys)

    @property
    def npix(self):
        '''
        Return the number of pixels in each structure.
        '''
        return np.array([self.structures[i].npix for i in self.structure_keys])

    @property
    def birth(self):
        '''
        Return the birth time of each structure.
        '''
        return np.array([self.structures[i].birth for i in self.structure_keys])

    @property
    def death(self):
        '''
        Return the death time of each structure.
        '''
        return np.array([self.structures[i].death for i in self.structure_keys])

    @property
    def equiv_radius(self):
        '''
        Return the equivalent radius of each structure.
        '''
        return np.array([self.structures[i].equiv_radius for i in self.structure_keys])

    @property
    def birthpix(self):
        '''
        Return the birth pixel of each structure.
        '''
        return np.array([self.structures[i].birthpix for i in self.structure_keys])

    @property
    def deathpix(self):
        '''
        Return the death pixel of each structure.
        '''
        return np.array([self.structures[i].deathpix for i in self.structure_keys])

    @property
    def htype(self):
        '''
        Return the homology type of each structure.
        '''
        return np.array([self.structures[i].htype for i in self.structure_keys])

    @property
    def persistence(self):
        '''
        Return the persistence of each structure.
        '''
        return np.array([self.structures[i].persistence for i in self.structure_keys])

    @property
    def norm_life(self):
        '''
        Return the normalized life of each structure.
        '''
        return self.persistence / self.birth

    @property
    def sphericity(self):
        '''
        Return the sphericity of each structure.
        '''
        return np.array([self.structures[i].sphericity for i in self.structure_keys])

    @property
    def id(self):
        '''
        Return the ID of each structure.
        '''
        return np.array([self.structures[i].id for i in self.structure_keys])

    @property
    def id_ph(self):
        '''
        Return the ID_PH of each structure.
        '''
        return np.array([self.structures[i].id_ph for i in self.structure_keys])

    @property
    def geom_cent(self):
        '''
        Return the geometric center of each structure.
        '''
        return np.array([self.structures[i].geom_cent for i in self.structure_keys])

    @property
    def parent(self):
        '''
        Return the parent of each structure.
        '''
        return np.array([self.structures[i].parent for i in self.structure_keys])

    @property
    def children(self):
        '''
        Return the children of each structure.
        '''
        return [self.structures[i].children for i in self.structure_keys]

    @property
    def descendants(self):
        '''
        Return the descendants of each structure.
        '''
        return [self.structures[i].descendants for i in self.structure_keys]

    @property
    def n_children(self):
        '''
        Return the number of children of each structure.
        '''
        return np.array([self.structures[i].n_children for i in self.structure_keys])

    @property
    def n_descendants(self):
        '''
        Return the number of descendants of each structure.
        '''
        return np.array([self.structures[i].n_descendants for i in self.structure_keys])

    @property
    def is_leaf(self):
        '''
        Return whether each structure is a leaf.
        '''
        return np.array([self.structures[i].is_leaf for i in self.structure_keys])

    @property
    def level(self):
        '''
        Return the level of each structure.
        '''
        return np.array([self.structures[i].level for i in self.structure_keys])

    @property
    def frac_npix_parent(self):
        '''
        Return the fraction of pixels of each structure relative to its parent.
        '''
        if self._frac_npix_parent is None:
            self._calc_frac_npix_parent()
        return  self._frac_npix_parent

    def _calc_frac_npix_parent(self):
        '''
        Calculate the fraction of pixels of each structure relative to its parent.
        '''
        frac_parent = np.full(self.n_struc, np.nan)
        for j in range(self.n_struc):
            if self.parent[j] is not None:
                frac_parent[j] = self.npix[j] / self.npix[self.parent[j]]
        self._frac_npix_parent = frac_parent

    #####################################################

    @property
    def birthpix_coord(self):
        '''
        Return the WCS birth coordinates of each structure.
        '''
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return self.wcs.pixel_to_world(self.birthpix[:,0],self.birthpix[:,1],self.birthpix[:,2])

    @property
    def deathpix_coord(self):
        '''
        Return the  WCS death coordinates of each structure.
        '''
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        if self._ndim == 2:
            return self.wcs.pixel_to_world(self.deathpix[:,0],self.deathpix[:,1])
        if self._ndim == 3:
            return self.wcs.pixel_to_world(self.deathpix[:,0],self.deathpix[:,1],self.deathpix[:,2])

    @property
    def geom_cent_coord(self):
        '''
        Return the WCS geometric center coordinates of each structure.
        '''
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return self.wcs.pixel_to_world(self.geom_cent)

    @property
    def centroid_coord(self):
        '''
        Return the WCS centroid coordinates of each structure.

        TO DO: add centroid calculation
        '''
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        if self.centroid_0 is None:
            print('Error: must calculate centroid!')
        return self.wcs.pixel_to_world(self.centroid_0,self.centroid_1,self.centroid_2)

    @property
    def equiv_radius_coord(self):
        '''
        Return the WCS equivalent radius of each structure.
        '''
        ## NOTE: ASSUMES EVEN PIXEL SCALES!!!
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        return np.abs(np.diag(self.wcs.pixel_scale_matrix)[0]) * self.equiv_radius

    @property
    def volume_coord(self):
        '''
        Return the WCS volume of each structure.
        '''
        if self.wcs is None:
            print('Error: must input wcs!')
            return
        volpc_pixel = np.product(np.diag(self.wcs.pixel_scale_matrix))  # pc^3 / voxel or pc^2/pixel
        return self.npix * volpc_pixel

    ###########################################################################
    ###########################################################################

    def get_mask(self, s_include = None, use_descendants=True):
        ''''
        Get a binary mask of the structures in s_include.

        Parameters:
        -----------
        s_include : list
            List of structure IDs to include.
        use_descendants : bool
            Include descendants of structures in s_include.

        Returns:
        --------
        mask : np.ndarray
            Binary mask.

        '''
        if s_include is None:
            s_include = list(self.structure_keys)
        if use_descendants:
            s_include = np.unique(np.hstack([self.descendants[s_include[i]] for i in range(len(s_include))]))
        mask = np.isin(self.struc_map, s_include)
        return mask

    def get_struc_map_mask(self, s_include = None, use_descendants=True):
        '''
        Get an ID mask of the structures in s_include.

        Parameters:
        -----------
        s_include : list
            List of structure IDs to include.
        use_descendants : bool
            Include descendants of structures in s_include.

        Returns:
        --------
        struc_map_mask : np.ndarray
            Masked structure map.
        '''
        if s_include is None:
            s_include = list(self.structure_keys)
        if use_descendants:
            s_include = np.unique(np.hstack([self.descendants[s_include[i]] for i in range(len(s_include))]))
        mask = np.isin(self.struc_map, s_include)
        struc_map_mask = np.where(mask, self.struc_map, np.nan)
        return struc_map_mask

    def sort_keys(self, s_include = None, invert=False):
        '''
        Sort the structure keys.

        Parameters:
        -----------
        s_include : list
            List of structure IDs to include.
        invert : bool
            Invert the sorting.

        '''
        if s_include is None:
            s_include = list(self.structure_keys)
        if not invert:
            return np.argsort(self.death[s_include])
        if invert:
            return np.argsort(self.death[s_include])[::-1]

    def sort_birth(self, s_include = None):
        '''
        Sort the structures by birth time.

        Parameters:
        -----------
        s_include : list
            List of structure IDs to include.

        '''
        if s_include is None:
            s_include = list(self.structure_keys)
        return np.argsort(self.birth[s_include])

    def _set_descendants(self, s = None):
        """
        Set descendants as a flattened list of all child leaves and branches.

        Parameters:
        -----------
        s : int
            Structure ID.

        From: astrodendro
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
        '''
        Clear the hierarchy.
        '''
        #print('Warning: clearing hierarchy!!!')
        for s in list(self.structure_keys):
            self.structures[s]._parent = None
            self.structures[s]._children = []
            self.structures[s]._descendants = []


    def id_mask(self,s_include=None):
        '''
        Create a masked  structure ID map.

        Parameters:
        -----------
        s_include : list
            List of structure IDs to include.

        Returns:
        --------
        mask : np.ndarray
            Masked structure ID map.

        '''
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
            struc._clear_indices()
        return mask

    ########################################################################

    def load_from(verbose=False,odir='./',fname='run',check_success=True):

        '''
        Load the structure hierarchy.

        Parameters:
        -----------
        verbose : bool
            Verbose output.
        odir : str
            Output directory.
        fname : str
            File name.
        '''

        print('Loading...') if verbose else None

        from astropy.io import fits
        from astropy.table import Table
        import pandas as pd
        from astropy.wcs import WCS


        hdul = fits.open(f'{odir}{fname}_perch_seg.fits')
        parent_table = Table(hdul[2].data)
        parent_df = parent_table.to_pandas()

        # mirror cripser output
        gen_array = parent_df[['Htype', 'Birth', 'Death', 'Birthpix_0', 'Birthpix_1', 'Birthpix_2', 'Deathpix_0', 'Deathpix_1', 'Deathpix_2', 'ID_PH']].values
        # create structures object
        self = Structures(structures=gen_array, img_shape = np.shape(hdul[0].data), wcs=WCS(hdul[0].header), struc_map=hdul[0].data, level_map=hdul[1].data)

        print('Assigning properties...') if verbose else None
        for i in range(self.n_struc):
            parent_val =  parent_df['Parent_ID'].values[parent_df['ID'].values == self.id[i]][0]
            if parent_val >= 0:
                self.structures[i]._parent = int(parent_val)
            self.structures[i]._npix = parent_df['Npix'].values[parent_df['ID'].values == self.id[i]][0]
            self.structures[i]._level = parent_df['Level'].values[parent_df['ID'].values == self.id[i]][0]

        ### label children
        print('Assigning children...') if verbose else None
        has_parent = np.where(self.parent != None)[0]
        for s in has_parent:
            s_parent = self.structures[s].parent
            self.structures[s_parent]._children.extend([int(s)])

        print('Assigning trunk and leaves...')  if verbose else None
        self._trunk = [structure for structure in self.all_structures if structure.parent == None]
        self._leaves = [structure for structure in self.all_structures if structure.is_leaf]

        print('Assigning descendants...') if verbose else None
        for i in range(self.n_struc):
            self._set_descendants(i)

        #'''
        # TO DO: add function to load level and npix instead of reassigning (requires storing in parent_df)
        '''if assign_npix:
            print('Assigning npix...') if verbose else None
            pbar = tqdm(total=self.n_struc, unit='structures') if verbose else None
            for i in range(self.n_struc):
                i_desc = self.structures[i].descendants
                struc_mask = np.isin(self.struc_map, i_desc)
                self.structures[i]._npix = np.nansum(struc_mask)
                if load_level:
                    self.structures[i]._level = np.nanmax(np.where(self.struc_map==self.structures[i].id,self.level_map,np.nan))
                pbar.update(1) if verbose else None
                #'''

        if check_success:
            print('Validating assignments...') if verbose else None
            self._check_segmentation_success(verbose=verbose)

        return self

    def add_attributes(self, prop_df):
        '''
        Add attributes to the structures collection.

        Parameters:
        -----------
        prop_df : pandas.DataFrame
            DataFrame of properties.

        '''
        for col in prop_df.columns[prop_df.columns != 'ID']:
            for i in range(self.n_struc):
                skey = list(self.structure_keys)[i]
                col_i_val = prop_df[col].values[prop_df['ID'].values == self.id[i]][0]
                #self.structures[i]._npix = prop_df[col].values[prop_df['ID'].values == self.id[i]][0]
                setattr(self.structures[skey], col, col_i_val)
            setattr(self, col,  np.array([getattr(self.structures[i],col) for i in self.structure_keys]))



    def compute_segment_hierarchy(self, img_jnp=None,  s_include = None, clobber=True,
                                  export=True,odir='./',fname='run',verbose=False):

        '''
        Compute the segment hierarchy.

        Parameters:
        -----------
        img_jnp : jnp.ndarray
            Image data.
        s_include : list
            List of structure IDs to include.
        clobber : bool
            Clear the hierarchy.
        export_parent : bool
            Export the parent DataFrame.
        odir : str
            Output directory.
        fname : str
            File name.
        verbose : bool
            Verbose output.

        '''

        mask_s = np.full((self._imgshape),np.nan)
        mask_count = np.full((self._imgshape), 0.)

        if type(img_jnp) is np.ndarray:
            img_jnp = jnp.array(img_jnp)

        if clobber:
            self._clear_hierarchy()

        ### calculate parents
        # TO DO: remove
        if len(self._imgshape) > 2:
            dim_invert = 2
        if len(self._imgshape) == 2:
            dim_invert = 1
        flag_invert = False#self.htype[0]!=dim_invert
        if self.htype[0] == 2:
            flag_invert = True
        ascend_death = self.sort_keys(s_include=s_include,invert=flag_invert)
        if verbose:
            pbar = tqdm(total=len(ascend_death), unit='structures')
        for s in ascend_death:
            struc = self.structures[s]
            struc.compute_segment(img=img_jnp)
            # TO DO: move back to normal indices instead of raveled
            struc_multi_inds = np.unravel_index(struc.indices, self._imgshape)
            mask_count[struc_multi_inds] += 1
            self.structures[s]._level = np.nanmax(mask_count[struc_multi_inds])

            # IDs are in order of lowest birth to highest birth
            # so nanmax will find the structure in the index footprint with the highest ID i.e. highest birth
            # so it's the most recent structure in the footprint so i think this definition of parent is good
            # unless want parent to be structure with most recent death?
            # need to think more about this

            # overwriting seems to occur for structures that have death less than death of the large structure
            # i.e., the overwitten structures appeared earlier in the ascending death hierarchy
            # contradiciotn between IDs in order of ascending birth, and segmenting in order of ascending death?
            parent_cand = np.nanmax(mask_s[struc_multi_inds]) ##### TO DO: check if this works in edge cases
            # because it's taking max id, but if multiple structures had pixels there, won't work
            if np.isfinite(parent_cand):
                self.structures[s]._parent = int(parent_cand)

            #mask_s[struc_multi_inds] = int(s)

            unassigned_pixels = np.isnan(mask_s[struc_multi_inds])
            if np.sum(unassigned_pixels) > 0:
                mask_s[struc_multi_inds] = np.where(unassigned_pixels, int(s), mask_s[struc_multi_inds])
            if np.sum(unassigned_pixels) == 0:
                mask_s[struc_multi_inds] = int(s)#'''

            struc._clear_indices()
            if verbose:
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
        self.level_map = np.where(mask_count>0,mask_count,np.nan)

        self._check_segmentation_success(verbose=verbose)

        if export:
            self.export_segmentation(fname=fname, odir=odir)

    ########################################################################

    def _check_segmentation_success(self,verbose=False):

        '''
        Check if the segmentation was successful.

        '''
        unique_struc = self.n_struc
        print(f'{unique_struc} total structures') if verbose else None
        unique_seg = np.unique(self.struc_map)
        unique_seg = unique_seg[~np.isnan(unique_seg)]
        n_unique = len(unique_seg)
        print(f'{n_unique} structures in segmented map') if verbose else None
        if unique_struc == n_unique:
            print('Segmentation successful!') if verbose else None
        if unique_struc != n_unique:
            print('Uh oh!') if verbose else None

    ########################################################################
    def remove_strucs(self, remove_ids, verbose=False):
        for s in remove_ids:
            self.remove_struc(s, verbose=verbose)


    def remove_struc(self, s_remove, verbose=False):
        '''
        Remove a structure.

        NOTE: in development.  need to add recursive removal of structure from all parent descendant lists in hierarchy

        Parameters:
        -----------
        s_remove : int
            Structure ID to remove.
        verbose : bool
            Verbose output.

        '''
        if verbose:
            print(f'Removing structure {s_remove}...')

        if not self.structures[s_remove].is_leaf:
            if verbose:
                print('Structure is a branch.')

            ids_children = self.structures[s_remove].children
            #child_mask = np.isin(self.id, ids_children)

            ids_desc = self.structures[s_remove].descendants

            #desc_mask = np.isin(self.id, ids_desc)
            # reduce level of all descendants by one
            for j in range(len(ids_desc)):
                self.structures[ids_desc[j]]._level = self.structures[ids_desc[j]]._level - 1
            desc_map_mask = np.isin(self.struc_map, ids_desc)
            self.level_map[desc_map_mask] -= 1

            # remove structure as children's parent
            for j in range(len(ids_children)):
                self.structures[ids_children[j]]._parent = None

        # remove structure from list of parent's children and descendants
        if self.structures[s_remove].parent is not None:
            if verbose:
                print('Structure has a parent.')
            parent_id = self.structures[s_remove].parent
            self.structures[parent_id]._children.remove(s_remove)
            self.structures[parent_id]._descendants.remove(s_remove)
            #self.structures[s_remove]._parent = None

        # replace pixels with structure's ID in segmented map with parent ID or nan, depending on parentage
        struc_mask = np.isin(self.struc_map, s_remove)
        self.struc_map[struc_mask] = np.nan if self.structures[s_remove].parent is None else self.structures[s_remove].parent
        self.level_map[struc_mask] = np.nan if self.structures[s_remove].parent is None else self.structures[self.structures[s_remove].parent].level

        # finally, remove structure from dictionary of structures
        self.structures.pop(s_remove)

    ########################################################################

    def generators(self):
        import pandas as pd
        parent_df = pd.DataFrame({ 'Htype': self.htype, 'Birth': self.birth, 'Death': self.death,
                                  'Birthpix_0': self.birthpix[:, 0], 'Birthpix_1': self.birthpix[:, 1],  'Birthpix_2': self.birthpix[:, 2],
                                  'Deathpix_0': self.deathpix[:, 0], 'Deathpix_1': self.deathpix[:, 1],'Deathpix_2': self.deathpix[:, 2],
                                   'ID_PH': self.id_ph})
        return parent_df.values

    def export_segmentation(self, fname='run', odir = './'):
        '''
        Export all information needed to recreate the segmentation (struc_map, level_map, structure properties).

        Parameters:
        -----------
        fname : str
            File name.
        odir : str
            Output directory.

        '''
        from astropy.io import fits
        from astropy.table import Table
        import pandas as pd

        if self.wcs is not None:
            head = self.wcs.to_header()
        else:
            head = None
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data=np.array(self.struc_map, dtype='float32'), header=head))
        hdul.append(fits.PrimaryHDU(data=np.array(self.level_map, dtype='float32'), header=head))

        parent_df = pd.DataFrame({'ID_PH': self.id_ph,  'ID': self.id,
                                  'Htype': self.htype, 'Birth': self.birth, 'Death': self.death,
                                  'Birthpix_0': self.birthpix[:,0], 'Birthpix_1': self.birthpix[:,1], 'Birthpix_2': self.birthpix[:,2],
                                  'Deathpix_0': self.deathpix[:,0], 'Deathpix_1': self.deathpix[:,1], 'Deathpix_2': self.deathpix[:,2],
                                  'Npix': self.npix, 'Parent_ID': np.array(self.parent,dtype='float32'),'Level': np.array(self.level,dtype='float32')})
        hdul.append(fits.BinTableHDU(Table.from_pandas(parent_df)))
        hdul.writeto(f'{odir}{fname}_perch_seg.fits', overwrite=True)

    def export_struc_map(self, fname='run', odir = './'):
        '''
        Export the structure map.

        Parameters:
        -----------
        fname : str
            File name.
        odir : str
            Output directory.
        '''
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

    def export_level_map(self, fname='run', odir = './'):
        '''
        Export the level map.

        Parameters:
        -----------
        fname : str
            File name.
        odir : str
            Output directory.

        '''
        if self.level_map is None:
            print('Error: must compute or load hierarchy first!')
            return

        from astropy.io import fits
        if self.wcs is not None:
            head = self.wcs.to_header()
        else:
            head = None
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data=np.array(self.level_map, dtype='float32'), header=head))
        hdul.writeto(f'{odir}{fname}_level_map.fits', overwrite=True)

    def clear_struc_map(self):
        '''
        Clear the structure map.
        '''
        self.struc_map = None













