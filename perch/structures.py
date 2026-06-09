import warnings

import numpy as np
from tqdm import tqdm
from perch.structure import Structure
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

        self.df = None

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
    def centroid(self):
        '''
        Return the weighted centroid of each structure.
        '''
        return np.array([self.structures[i].centroid for i in self.structure_keys])

    @property
    def centroid_0(self):
        return self.centroid[:, 0]
    @property
    def centroid_1(self):
        return self.centroid[:, 1]
    @property
    def centroid_2(self):
        if self._ndim == 3:
            return self.centroid[:, 2]
        else:
            return None

    @property
    def bbox(self):
        '''
        Return the weighted centroid of each structure.
        '''
        return np.array([self.structures[i].bbox for i in self.structure_keys])

    @property
    def bbox_min(self):
        '''
        Return the minimum bounding box of each structure.
        '''
        return np.array([self.structures[i].bbox_min for i in self.structure_keys])

    @property
    def bbox_max(self):
        '''
        Return the maximum bounding box of each structure.
        '''
        return np.array([self.structures[i].bbox_max for i in self.structure_keys])

    @property
    def bbox_min_0(self):
        return self.bbox[:, 0, 0]
    @property
    def bbox_min_1(self):
        return self.bbox[:, 0, 1]
    @property
    def bbox_min_2(self):
        return self.bbox[:, 0, 2] if self._ndim == 3 else None
    @property
    def bbox_max_0(self):
        return self.bbox[:, 1, 0]
    @property
    def bbox_max_1(self):
        return self.bbox[:, 1, 1]
    @property
    def bbox_max_2(self):
        return self.bbox[:, 1, 2] if self._ndim == 3 else None

    @property
    def sum_val(self):
        '''
        Return the integrated value of pixels assigned to each structure.
        '''
        return np.array([self.structures[i].sum_val for i in self.structure_keys])

    @property
    def min_val(self):
        '''
        Return the minimum values of each structure.
        '''
        return np.array([self.structures[i].min_val for i in self.structure_keys])

    @property
    def max_val(self):
        '''
        Return the maximum values of each structure.
        '''
        return np.array([self.structures[i].max_val for i in self.structure_keys])

    @property
    def median_val(self):
        '''
        Return the median values of each structure.
        '''
        return np.array([self.structures[i].median_val for i in self.structure_keys])

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

        Looks parents up by ID through ``self.structures`` rather than indexing
        the positional ``npix`` array, so it stays correct after removals leave
        the structure IDs non-contiguous.
        '''
        frac_parent = np.full(self.n_struc, np.nan)
        for j, k in enumerate(self.structure_keys):
            struc = self.structures[k]
            if struc.parent is not None:
                frac_parent[j] = struc.npix / self.structures[struc.parent].npix
        self._frac_npix_parent = frac_parent

    #####################################################

    def _pixel_to_world(self, pix):
        """Dispatch ``wcs.pixel_to_world`` over a (n_struc, ndim) pixel array.

        Internal helper used by every per-structure ``*_coord`` property to
        avoid duplicating the 2D/3D switch.
        """
        if self._ndim == 2:
            return self.wcs.pixel_to_world(pix[:, 0], pix[:, 1])
        if self._ndim == 3:
            return self.wcs.pixel_to_world(pix[:, 0], pix[:, 1], pix[:, 2])
        raise NotImplementedError(f"WCS coord dispatch for ndim={self._ndim}")

    @property
    def birthpix_coord(self):
        '''
        Return the WCS birth coordinates of each structure.
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        return self._pixel_to_world(self.birthpix)

    @property
    def deathpix_coord(self):
        '''
        Return the  WCS death coordinates of each structure.
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        return self._pixel_to_world(self.deathpix)

    @property
    def geom_cent_coord(self):
        '''
        Return the WCS geometric center coordinates of each structure.
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        return self._pixel_to_world(self.geom_cent)

    @property
    def centroid_coord(self):
        '''
        Return the WCS centroid coordinates of each structure.
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        if self.centroid is None:
            raise RuntimeError(
                "centroid has not been computed; call compute_segment_hierarchy "
                "(with calc_supp_props=True) or _calculate_centroid first"
            )
        return self._pixel_to_world(self.centroid)

    @property
    def bbox_min_coord(self):
        '''
        Return the WCS bounding box minimum coordinates of each structure.
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        return self._pixel_to_world(self.bbox_min)

    @property
    def bbox_max_coord(self):
        '''
        Return the WCS bounding box maximum coordinates of each structure.
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        return self._pixel_to_world(self.bbox_max)


    @property
    def equiv_radius_coord(self):
        '''
        Return the WCS equivalent radius of each structure.

         NOTE: ASSUMES EVEN PIXEL SCALES!!!
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        return np.abs(np.diag(self.wcs.pixel_scale_matrix)[0]) * self.equiv_radius

    @property
    def volume_coord(self):
        '''
        Return the WCS volume of each structure.

         NOTE: ASSUMES EVEN PIXEL SCALES!!!
        '''
        if self.wcs is None:
            raise ValueError("Structures has no WCS attached")
        volpc_pixel = np.prod(np.abs(np.diag(self.wcs.pixel_scale_matrix)))  # pc^3 / voxel or pc^2/pixel
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
            # index descendants by structure ID, not by array position, so this
            # stays correct after removals leave the IDs non-contiguous.
            s_include = np.unique(np.hstack([self.structures[sid].descendants for sid in s_include]))
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
            # index descendants by structure ID, not by array position, so this
            # stays correct after removals leave the IDs non-contiguous.
            s_include = np.unique(np.hstack([self.structures[sid].descendants for sid in s_include]))
        mask = np.isin(self.struc_map, s_include)
        struc_map_mask = np.where(mask, self.struc_map, np.nan)
        return struc_map_mask

    def sort_death(self, s_include = None, invert=False):
        '''
        Sort the structures by death.

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
            return np.lexsort((self.birth[s_include], self.death[s_include])) # Sort by death from low to high, then by birth from low to high for ties
        if invert:
            return np.lexsort((self.birth[s_include], self.death[s_include]))[::-1]

    def sort_birth(self, s_include = None, invert=False):
        '''
        Sort the structures by birth.

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
            return np.lexsort((self.death[s_include], self.birth[s_include])) # Sort by birth from low to high, then by death from low to high for ties
        if invert:
            return  np.lexsort((self.death[s_include], self.birth[s_include]))[::-1]

    def _set_descendants(self, s = None):
        """
        Set descendants as a flattened list of the structure itself plus
        all of its child leaves and branches. The list is self-inclusive
        — `s` appears as the first element of its own descendants.

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


    def id_mask(self, s_include=None):
        '''
        Return a struc_map-style array carrying only the IDs of the selected
        structures, with NaN elsewhere. Equivalent to
        ``get_struc_map_mask(s_include, use_descendants=False)`` but kept as a
        flatter, descendants-free entry point.

        Parameters
        ----------
        s_include : list, optional
            Structure IDs to keep. Defaults to all.

        Returns
        -------
        mask : np.ndarray
            Image-shaped array. Pixels belonging to a selected structure
            carry that structure's ID; all other pixels are NaN.
        '''
        if self.struc_map is None:
            raise RuntimeError(
                "hierarchy has not been computed; call compute_segment_hierarchy first"
            )
        if s_include is None:
            s_include = list(self.structure_keys)
        else:
            s_include = list(s_include)
        in_set = np.isin(self.struc_map, s_include)
        mask = np.full(self._imgshape, np.nan)
        mask[in_set] = self.struc_map[in_set]
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
        from astropy.wcs import WCS

        with fits.open(f'{odir}{fname}_perch_seg.fits') as hdul:
            parent_table = Table(hdul[2].data)
            parent_df = parent_table.to_pandas()
            struc_map = np.asarray(hdul[0].data).copy()
            level_map = np.asarray(hdul[1].data).copy()
            wcs = WCS(hdul[0].header)

        output_cols = ['Htype', 'Birth', 'Death', 'Birthpix_0', 'Birthpix_1', 'Birthpix_2', 'Deathpix_0', 'Deathpix_1', 'Deathpix_2', 'ID_PH']
        ignore_cols = ['Parent_ID', 'Npix', 'Level']
        supp_cols = parent_df.columns[~np.isin(parent_df.columns,np.hstack([output_cols,ignore_cols])) ]

        ### note: this ssetup assumes taht you are feding in only the same set of structures as when saved.  otherwise ID nubmers etc will get messed up.
        # so don't subsample this dataframe...since stored in fits file I think safe.  but probably revisit this later

        # mirror cripser output
        gen_array = parent_df[output_cols].values
        # create structures object
        self = Structures(structures=gen_array, img_shape=struc_map.shape, wcs=wcs, struc_map=struc_map, level_map=level_map)

        self.df = parent_df

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

        if supp_cols is not None and len(supp_cols) > 0:
            supp_df = parent_df[supp_cols]
            cols_to_update = ['sum_val','min_val','max_val','median_val']
            supp_df.columns = [f"_{c}" if c in cols_to_update else c for c in supp_df.columns]
            coord_cols = ['centroid_0', 'centroid_1', 'centroid_2', 'bbox_min_0', 'bbox_min_1', 'bbox_min_2',
                           'bbox_max_0', 'bbox_max_1', 'bbox_max_2']
            df_exclude_coords = supp_df.columns[~np.isin(supp_df.columns, coord_cols)]
            self.add_attributes(supp_df[df_exclude_coords])

            coord_pref_list = ['centroid', 'bbox_min', 'bbox_max']
            for coord_pref in coord_pref_list:
                if f'{coord_pref}_0' in supp_df.columns:
                    if self._ndim ==3:
                        centroid_stack = np.stack((supp_df[f'{coord_pref}_0'].values, supp_df[f'{coord_pref}_1'].values,supp_df[f'{coord_pref}_2'].values), axis=1)
                    if self._ndim ==2:
                        centroid_stack = np.stack((supp_df[f'{coord_pref}_0'].values, supp_df[f'{coord_pref}_1'].values), axis=1)
                    for i in range(self.n_struc):
                        skey = list(self.structure_keys)[i]
                        col_i_val = centroid_stack[supp_df['ID'].values == self.id[i]][0]
                        setattr(self.structures[skey], f'_{coord_pref}', col_i_val)
                   #setattr(self, f'_{coord_pref}', np.array([getattr(self.structures[i], f'_{coord_pref}') for i in self.structure_keys]))

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
                setattr(self.structures[skey], col, col_i_val)
            setattr(self, col,  np.array([getattr(self.structures[i],col) for i in self.structure_keys]))



    def compute_segment_hierarchy(self, img_jnp=None,  s_include = None, clobber=True,
                                  export=True,odir='./',fname='run',verbose=False, calc_supp_props=True,clear_indices=True):

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
        clear_indices : bool
            Clear the pixel indices after computation (saves memory).

        Examples
        --------
        Segment the H0 generators of a one-peak image and inspect the
        resulting per-pixel structure map:

        >>> import numpy as np
        >>> from perch.ph import PH
        >>> y, x = np.indices((8, 8))
        >>> img = np.exp(-((y - 3)**2 + (x - 4)**2) / 2.0).astype(np.float32)
        >>> ph = PH.compute_hom(data=img, verbose=False)
        >>> h0 = ph.filter(dimension=0)
        >>> h0.compute_segment_hierarchy(img_jnp=img, verbose=False, export=False)
        >>> h0.struc_map.shape
        (8, 8)
        >>> h0.n_struc
        1

        '''

        mask_s = np.full((self._imgshape),np.nan)
        mask_count = np.full((self._imgshape), 0.)

        if type(img_jnp) is np.ndarray:
            # FITS arrays are big-endian; jax >= 0.10 only accepts native byte order.
            if not img_jnp.dtype.isnative:
                img_jnp = img_jnp.astype(img_jnp.dtype.newbyteorder('='))
            img_jnp = jnp.array(img_jnp)

        if clobber:
            self._clear_hierarchy()

        if len(self._imgshape) > 2:
            dim_invert = 2
        if len(self._imgshape) == 2:
            dim_invert = 1

        # Cache property arrays to avoid recomputing in loop
        htype_arr = self.htype
        death_arr = self.death
        birth_arr = self.birth

        if len(htype_arr) == 0:
            self.struc_map = mask_s
            self.level_map = np.where(mask_count > 0, mask_count, np.nan)
            self._trunk = []
            self._leaves = []
            if verbose:
                print('No structures to segment.')
            return

        if len(np.unique(htype_arr)) > 1:
            warnings.warn(
                "multiple homology groups detected; segmentation order may be "
                "incorrect. Please segment each homology group separately.",
                UserWarning,
                stacklevel=2,
            )

        if htype_arr[0] == 0:
            seg_order = self.sort_death(s_include=s_include,invert=False) # ascend death
        if htype_arr[0] == dim_invert:
            seg_order = self.sort_birth(s_include=s_include,invert=True) # descend birth

        # Convert to numpy once for supplementary property calculations
        # (avoids JAX recompilation overhead for variable-sized index arrays)
        img_np = np.asarray(img_jnp) if calc_supp_props else None

        if verbose:
            pbar = tqdm(total=len(seg_order), unit='structures')
        for s in seg_order:
            struc = self.structures[s]
            struc.compute_segment(img=img_jnp)

            struc_multi_inds = struc.indices
            mask_count[struc_multi_inds] += 1
            self.structures[s]._level = np.nanmax(mask_count[struc_multi_inds])

            parent_cands = np.unique(mask_s[struc_multi_inds])
            parent_cands = parent_cands[~np.isnan(parent_cands)]
            if len(parent_cands) > 0:
                if htype_arr[0] == 0:
                    parent_deaths = death_arr[parent_cands.astype(int)]
                    parent_cand = parent_cands[np.argmax(parent_deaths)]
                if htype_arr[0] == dim_invert:
                    parent_births = birth_arr[parent_cands.astype(int)]
                    parent_cand = parent_cands[np.argmin(parent_births)]
                self.structures[s]._parent = int(parent_cand)

            mask_s[struc_multi_inds] = int(s)

            if calc_supp_props:
                struc._calculate_centroid(img=img_np)
                struc._calculate_pix_values(img=img_np)
                struc._calculate_bbox()
                struc._calculate_geom_cent()

            if clear_indices:
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

        for s in seg_order:
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
        '''
        Remove several structures, one at a time. Each removal dissolves the
        structure into its parent (see :meth:`remove_struc`); IDs are stable,
        so order only matters if ``remove_ids`` references a structure that an
        earlier removal already dissolved.
        '''
        for s in remove_ids:
            self.remove_struc(s, verbose=verbose)


    def remove_struc(self, s_remove, verbose=False):
        '''
        Remove a single structure, dissolving it into its parent.

        The structure's exclusive pixels are reassigned to its parent (or set
        to NaN if it was a trunk), its children are re-parented onto its parent
        (becoming new trunks if it had none), and every descendant drops one
        level of nesting. The hierarchy bookkeeping is updated so that the
        removed ID is purged from the children list of its parent and from the
        (self-inclusive) descendant lists of *all* of its ancestors.

        Parameters:
        -----------
        s_remove : int
            Structure ID to remove.
        verbose : bool
            Verbose output.

        '''
        if verbose:
            print(f'Removing structure {s_remove}...')

        struc = self.structures[s_remove]
        parent_id = struc.parent  # None if s_remove is a trunk

        if not struc.is_leaf:
            if verbose:
                print('Structure is a branch.')

            ids_children = list(struc.children)
            # descendants is self-inclusive; exclude s_remove itself, whose
            # own pixels/level are handled by the reassignment step below.
            ids_desc = [d for d in struc.descendants if d != s_remove]

            # every descendant loses one level of nesting (s_remove leaves the
            # containment chain), both per-structure and in the level map.
            for d in ids_desc:
                self.structures[d]._level -= 1
            desc_map_mask = np.isin(self.struc_map, ids_desc)
            self.level_map[desc_map_mask] -= 1

            # re-parent s_remove's children onto s_remove's parent. With no
            # parent they become new trunks; otherwise they are appended to the
            # parent's children list (which then drops s_remove below).
            for c in ids_children:
                self.structures[c]._parent = parent_id
            if parent_id is not None:
                self.structures[parent_id]._children.extend(ids_children)

        # detach s_remove from its parent's children, and from the descendant
        # list of every ancestor (descendants propagate up the whole chain).
        if parent_id is not None:
            if verbose:
                print('Structure has a parent.')
            self.structures[parent_id]._children.remove(s_remove)
            anc = parent_id
            while anc is not None:
                anc_struc = self.structures[anc]
                if s_remove in anc_struc._descendants:
                    anc_struc._descendants.remove(s_remove)
                anc = anc_struc.parent

        # reassign s_remove's exclusive pixels to its parent's ID/level, or to
        # NaN if it was a trunk.
        struc_mask = np.isin(self.struc_map, s_remove)
        if parent_id is None:
            self.struc_map[struc_mask] = np.nan
            self.level_map[struc_mask] = np.nan
        else:
            self.struc_map[struc_mask] = parent_id
            self.level_map[struc_mask] = self.structures[parent_id].level

        # finally, drop s_remove and refresh cached hierarchy summaries.
        self.structures.pop(s_remove)
        self._refresh_hierarchy_cache()

    def _refresh_hierarchy_cache(self):
        '''
        Recompute cached hierarchy summaries (``trunk``/``leaves``) and
        invalidate derived caches after a structural edit such as a removal.
        '''
        self._trunk = [s for s in self.all_structures if s.parent is None]
        self._leaves = [s for s in self.all_structures if s.is_leaf]
        self._frac_npix_parent = None

    ########################################################################

    def generators(self):
        if self.df is None:
            self._make_df()
        return self.df[['Htype', 'Birth', 'Death','Birthpix_0', 'Birthpix_1', 'Birthpix_2',
                        'Deathpix_0', 'Deathpix_1', 'Deathpix_2', 'ID_PH']].values

    def _make_df(self):
        import pandas as pd

        parent_df = pd.DataFrame({'ID_PH': self.id_ph, 'ID': self.id,
                                  'Htype': self.htype, 'Birth': self.birth, 'Death': self.death,
                                  'Birthpix_0': self.birthpix[:, 0], 'Birthpix_1': self.birthpix[:, 1],
                                  'Birthpix_2': self.birthpix[:, 2],
                                  'Deathpix_0': self.deathpix[:, 0], 'Deathpix_1': self.deathpix[:, 1],
                                  'Deathpix_2': self.deathpix[:, 2],
                                  'Npix': self.npix, 'Parent_ID': np.array(self.parent, dtype='float32'),
                                  'Level': np.array(self.level, dtype='float32')})

        # Supplementary properties: include only if at least one structure
        # already has them cached. Read the underlying `_attr` directly so
        # that probing for "is this cached?" never accidentally triggers
        # the image-dependent computation in the public property.
        structures = self.all_structures
        if structures and structures[0]._sum_val is not None:
            parent_df['sum_val'] = [s._sum_val for s in structures]
            parent_df['min_val'] = [s._min_val for s in structures]
            parent_df['max_val'] = [s._max_val for s in structures]
            parent_df['median_val'] = [s._median_val for s in structures]
        if structures and structures[0]._centroid is not None:
            centroids = np.array([s._centroid for s in structures])
            parent_df['centroid_0'] = centroids[:, 0]
            parent_df['centroid_1'] = centroids[:, 1]
            if self._ndim == 3:
                parent_df['centroid_2'] = centroids[:, 2]
            if self._ndim == 2:
                parent_df['centroid_2'] = np.nan
        if structures and structures[0]._bbox is not None:
            bboxes = np.array([s._bbox for s in structures])
            parent_df['bbox_min_0'] = bboxes[:, 0, 0]
            parent_df['bbox_min_1'] = bboxes[:, 0, 1]
            if self._ndim == 3:
                parent_df['bbox_min_2'] = bboxes[:, 0, 2]
            parent_df['bbox_max_0'] = bboxes[:, 1, 0]
            parent_df['bbox_max_1'] = bboxes[:, 1, 1]
            if self._ndim == 3:
                parent_df['bbox_max_2'] = bboxes[:, 1, 2]

        self.df = parent_df

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

        if self.wcs is not None:
            head = self.wcs.to_header()
        else:
            head = None
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data=np.array(self.struc_map, dtype='float32'), header=head))
        hdul.append(fits.PrimaryHDU(data=np.array(self.level_map, dtype='float32'), header=head))

        if self.df is None:
            self._make_df()
        hdul.append(fits.BinTableHDU(Table.from_pandas(self.df)))

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
            raise RuntimeError(
                "hierarchy has not been computed; call compute_segment_hierarchy first"
            )

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
            raise RuntimeError(
                "hierarchy has not been computed; call compute_segment_hierarchy first"
            )

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













