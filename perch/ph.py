import copy
import numpy as np
from perch.structures import Structures

class PH(object):

    '''
    Base persistent homology class.

    Attributes:
    -----------
    data : np.ndarray
        Input data array.
    n_dim : int
        Number of dimensions in image.
    max_Hi : int
        Maximum homology dimension to compute.
    generators : np.ndarray
        Persistent homology generators.
    strucs : perch.Structures
        Structure object.
    ph_fxn : function
        Function for computing PH (cripser or pycripser).
    noise : float
        Noise map.

    '''

    def __init__(self):
        self.data = None
        self.data_prep = None
        self.n_dim = 0
        self.max_Hi = None
        self.generators = None
        self.strucs = None
        self.ph_fxn = None
        self.noise = None

    ####################################################
    ## compute PH (or load from stored)

    def _prep_img(self,buff_pix=False,buff_pix_loc=None,buff_val=None,fill_complete=False,fill_mask=None):
        '''
        Prepare image for PH computation.
        '''
        img_prep = copy.deepcopy(self.data)
        img_prep = -img_prep

        if fill_complete:
            if buff_val is None:
                buff_val = np.nanmin(img_prep) * 2
            if fill_mask is None:
                fill_mask = np.isnan(img_prep)
            img_prep = np.where(fill_mask, buff_val, img_prep)

        if buff_pix:

            if buff_val is None:
                buff_val = np.nanmin(img_prep) * 2

            if self.n_dim == 2:
                if buff_pix_loc is not None:
                    img_prep[buff_pix_loc[0],buff_pix_loc[1]] = buff_val
                else:
                    if np.isfinite(img_prep[0,0]):
                        img_prep[0:1, 0:1] = buff_val
                    else:
                        fin_use = np.where(np.isfinite(img_prep))
                        img_prep[fin_use[0][0],fin_use[1][0]] =buff_val

            if self.n_dim == 3:
                if buff_pix_loc is not None:
                    img_prep[buff_pix_loc[0],buff_pix_loc[1],buff_pix_loc[2]] = buff_val
                else:
                    if np.isfinite(img_prep[0,0,0]):
                        img_prep[0:1, 0:1,0:1] = buff_val
                    else:
                        fin_use = np.where(np.isfinite(img_prep))
                        img_prep[fin_use[0][0],fin_use[1][0], fin_use[2][0]] = buff_val

        return img_prep, buff_val

    def compute_hom(data=None, max_Hi=None, wcs=None, flip_data=True, verbose=True, embedded=False,
                    engine='C', noise=None,prep_img_kwargs={}):

        '''
        Compute persistent homology.

        Parameters:
        -----------
        data : np.ndarray
            Input data array.
        max_Hi : int
            Maximum homology dimension to compute.
        wcs : astropy.wcs.WCS
            WCS object.
        flip_data : bool
            Flip data array.
        verbose : bool
            Print progress.
        embedded : bool
            Compute embedded PH.
        engine : str
            PH computation engine ('C' or 'py').
        noise : np.ndarray
            Noise map of same shape as data.

        Returns:
        --------
        perch.PH
            Persistent homology object.
        '''

        # create PH object
        self = PH()
        self.data = data
        self.n_dim = len(data.shape)
        self.wcs = wcs
        self.img_shape = data.shape
        if flip_data:
            self.data_prep, buff_val = self._prep_img(**prep_img_kwargs)
        if not flip_data:
            self.data_prep = self.data
        if max_Hi is None:
            max_Hi = self.n_dim - 1
        self.max_Hi = max_Hi
        self.noise = noise

        # define PH computation engine
        if engine == 'C':
            import cripser
            self.ph_fxn = cripser.computePH
        if engine == 'py':
            from perch.py_cripser.cubicalripser_pybind import compute_ph
            self.ph_fxn = compute_ph

        if verbose:
            import time
            print('Computing PH... \n')
            t1 = time.time()

        # compute PH
        #ph_all = cripser.computePH(self.data_prep, maxdim=self.max_Hi, embedded=embedded)
        ph_all = self.ph_fxn(self.data_prep, maxdim=self.max_Hi, embedded=embedded)

        if verbose:
            t2 = time.time()
            print(f'\n PH Computation Complete! \n {t2-t1:.1f}s elapsed')

        # flip data back
        if flip_data:
            ph_all[:,1] = -ph_all[:,1]
            ph_all[:,2] = -ph_all[:,2]
            #if buff_val is not None:
            #print('ignoring')
            """base_struc = ph_all[:,1] == -buff_val
                ph_all = ph_all[~base_struc]
                base_struc = ph_all[:, 2] == -buff_val
                ph_all = ph_all[~base_struc]#"""
            #print('flipping observed deaths')

        # remove generators that originate from nans
        """base_struc = ph_all[:,2] < np.nanmin(self.data)
        ph_all = ph_all[~base_struc]
        base_struc = np.isnan(ph_all[:, 1])
        ph_all = ph_all[~base_struc]#"""

        if 'fill_mask' in {}.keys():
            # remove generators that originate from inside of fill mask
            base_struc = prep_img_kwargs['fill_mask'][ph_all[:, 3].astype(int), ph_all[:, 4].astype(int), ph_all[:, 5].astype(int)]
            ph_all = ph_all[~base_struc]

        # add id
        h_id = np.arange(len(ph_all))
        h_all = np.hstack((ph_all, np.array(h_id).reshape(-1, 1)))

        # store generators
        self.generators = h_all
        self.strucs = Structures(structures=h_all, img_shape=self.img_shape, wcs=self.wcs,inds_dir=None)
        #self.data_prep = None # save memory

        return self

    def export_generators(self, fname, odir='./'):
        '''
        Export generators to file.

        Parameters:
        -----------
        fname : str
            File name.
        odir : str
            Output directory.

        '''
        np.savetxt(f'{odir}{fname}', self.generators)

    def load_from(fname, odir='./',data=None,wcs=None, max_Hi=None,conv_fac=None, noise=None):
        '''
        Load generators from file.

        Parameters:
        -----------
        fname : str
            File name.
        odir : str
            Output directory.
        data : np.ndarray
            Input data array.
        wcs : astropy.wcs.WCS
            WCS object.
        max_Hi : int
            Maximum homology dimension to compute.
        conv_fac : float
            Conversion factor.
        noise : np.ndarray
            Noise map of same shape as data.

        Returns:
        --------
        perch.PH
            Persistent homology object.

        '''

        # create PH object
        self = PH()
        self.data = data
        self.n_dim = len(data.shape)
        self.wcs = wcs
        if max_Hi is None:
            max_Hi = self.n_dim - 1
        self.max_Hi = max_Hi
        self.noise = noise

        # load generators
        gens = np.loadtxt(f'{odir}{fname}')
        # convert to physical units if necessary
        if conv_fac is not None:
            gens[:,1:3] *= conv_fac

        # remove generators that originate from nans
        base_struc = gens[:,2] < np.nanmin(self.data)
        gens = gens[~base_struc]
        base_struc = np.isnan(gens[:, 1])
        gens = gens[~base_struc]

        # add homology id if not present
        if np.shape(gens)[1] == 9:
            h_id = np.arange(len(gens))
            gens = np.hstack((gens, np.array(h_id).reshape(-1, 1)))

        # store generators
        self.generators = gens
        self.img_shape = data.shape
        self.strucs = Structures(structures=gens, img_shape=self.img_shape, wcs=self.wcs,inds_dir=None)

        return self

    ####################################################
    ## filtering and segmentation

    def filter(self, dimension=None, min_life=None, max_life=None,
                      min_birth=None, max_birth=None,
                      min_death=None, max_death=None, min_life_norm_birth=None,min_life_norm_death=None,inds_dir=None,
               mask=None):

        '''
        Filter structures.

        Parameters:
        -----------
        dimension : int
            Homology dimension.
        min_life : float
            Minimum lifetime.
        max_life : float
            Maximum lifetime.
        min_birth : float
            Minimum birth.
        max_birth : float
            Maximum birth.
        min_death : float
            Minimum death.
        max_death : float
            Maximum death.
        min_life_norm_birth : float
            Minimum normalized lifetime at birth.
        min_life_norm_death : float
            Minimum normalized lifetime at death.
        inds_dir : str
            Directory for saving/loading indices.
        mask : np.ndarray
            Mask for filtering.

        Returns:
        --------
        perch.Structures
            Filtered structures.

        '''

        ppd = self.generators

        # apply mask if provided
        if mask is not None:
            ppd = ppd[mask]
            return Structures(structures=ppd, img_shape=self.img_shape, wcs=self.wcs,inds_dir=inds_dir)

        # apply filters if provided
        if mask is None:
            if dimension is not None:
                ppd = ppd[ppd[:, 0] == dimension]
            if min_life is not None:
                ppd = ppd[min_life < np.abs(ppd[:, 2] - ppd[:, 1])]
            if max_life is not None:
                ppd = ppd[np.abs(ppd[:, 2] - ppd[:, 1]) < max_life]
            if min_birth is not None:
                ppd = ppd[min_birth < ppd[:, 1]]
            if max_birth is not None:
                ppd = ppd[ppd[:, 1] < max_birth]
            if min_death is not None:
                ppd = ppd[ppd[:,2] > min_death]
            if max_death is not None:
                ppd = ppd[ppd[:, 2] < max_death]
            if min_life_norm_birth is not None:
                ppd = ppd[np.abs(ppd[:, 2] - ppd[:, 1])/np.abs(ppd[:,1]) > min_life_norm_birth]
            if min_life_norm_death is not None:
                ppd = ppd[np.abs(ppd[:, 2] - ppd[:, 1])/np.abs(ppd[:,2]) > min_life_norm_death]

            return Structures(structures=ppd, img_shape=self.img_shape, wcs=self.wcs,inds_dir=inds_dir)


