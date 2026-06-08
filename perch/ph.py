import copy
import warnings

import numpy as np
from perch.structures import Structures
import cripser

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
            warnings.warn(
                "The `buff_pix` path in PH._prep_img is deprecated and will be "
                "removed in a future release. Use `pad_essential=` on "
                "`PH.compute_hom` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

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

    @staticmethod
    def _resolve_pad_mode(pad_essential, data):
        """Resolve ``pad_essential`` to a concrete mode string or ``False``."""
        if pad_essential is False or pad_essential is None:
            return False
        if pad_essential == 'auto':
            return 'dilate' if np.isnan(data).any() else 'bbox'
        if pad_essential in ('dilate', 'bbox'):
            return pad_essential
        raise ValueError(
            f"pad_essential must be one of 'auto', 'dilate', 'bbox', False; "
            f"got {pad_essential!r}"
        )

    @staticmethod
    def _build_padded_data(data, mode, pad_value):
        """Build the padded data array for the second cripser pass.

        Returns ``(padded_data, pixel_offset)`` where ``pixel_offset`` is the
        per-axis shift to subtract from padded-frame pixel coords to recover
        original-frame coords.
        """
        if mode == 'dilate':
            from scipy.ndimage import binary_dilation
            finite = np.isfinite(data)
            if finite.all():
                raise ValueError(
                    "pad_essential='dilate' requires NaN voxels to dilate "
                    "into; the input is fully finite. Use 'bbox' (or 'auto') "
                    "instead."
                )
            if not finite.any():
                raise ValueError(
                    "pad_essential='dilate' requires some finite voxels; "
                    "input is all-NaN."
                )
            new_edges = binary_dilation(finite) & ~finite
            padded = data.copy()
            padded[new_edges] = pad_value
            offset = np.zeros(data.ndim, dtype=int)
            return padded, offset

        if mode == 'bbox':
            padded = np.pad(data, pad_width=1, mode='constant',
                            constant_values=pad_value)
            offset = np.ones(data.ndim, dtype=int)
            return padded, offset

        raise ValueError(f"unknown pad_essential mode: {mode!r}")

    def _pad_and_patch_essential(self, h_all, mode, pad_value, verbose,
                                 embedded=False):
        """Run a padded H_0-only PH and patch the originally-essential row.

        The originally-essential H_0 generator is the row whose birth is
        ``nanmax(data)`` and whose death is the most-negative (``-DBL_MAX``
        sentinel in the un-patched run, or the smallest finite death among
        tied-birth rows). In the padded run, no row carries the sentinel
        because the pad voxels host the new global essential; we locate the
        matching row by birth-pixel equality and copy its death and
        death-pixel columns back into ``h_all`` in place.
        """
        target_birth = np.nanmax(self.data)
        if not np.isfinite(target_birth):
            raise ValueError(
                "pad_essential requires data to have at least one finite "
                "voxel; got an all-NaN/inf input."
            )
        # Default pad value lives here, beside its docstring, and reuses the
        # single nanmax above rather than recomputing it in the caller.
        if pad_value is None:
            pad_value = 10.0 * target_birth

        # Identify the essential row in the original run: among rows with
        # birth==target_birth, the one with the smallest death. With the
        # legacy sentinel that's the -DBL_MAX row; with no ties it's the
        # only such row.
        ess_mask = (h_all[:, 0] == 0) & (h_all[:, 1] == target_birth)
        ess_idx_all = np.where(ess_mask)[0]
        if ess_idx_all.size == 0:
            raise RuntimeError(
                "pad_essential: no H_0 generator in the original run has "
                f"birth == nanmax(data)={target_birth!r}."
            )
        essential_idx = int(ess_idx_all[np.argmin(h_all[ess_idx_all, 2])])
        essential_bp = h_all[essential_idx, 3:3 + self.n_dim].astype(int)

        padded_data, pixel_offset = self._build_padded_data(
            self.data, mode, pad_value
        )

        padded_flipped = -padded_data
        if verbose:
            print(f"  pad_essential={mode!r}: running padded H_0 PH "
                  f"on shape {padded_flipped.shape}...")
        ph_pad = cripser.computePH(padded_flipped, maxdim=0, embedded=embedded)
        ph_pad[:, 1] = -ph_pad[:, 1]
        ph_pad[:, 2] = -ph_pad[:, 2]

        cand_mask = (ph_pad[:, 0] == 0) & (ph_pad[:, 1] == target_birth)
        cand_idx_all = np.where(cand_mask)[0]
        cand_bps = (ph_pad[cand_idx_all, 3:3 + self.n_dim].astype(int)
                    - pixel_offset)
        bp_match = np.all(cand_bps == essential_bp, axis=1)
        matched = cand_idx_all[bp_match]
        if matched.size != 1:
            raise RuntimeError(
                f"pad_essential: could not locate the padded-run counterpart "
                f"of the original essential row (birthpix={tuple(essential_bp)}, "
                f"matched={matched.size}). Pass `pad_essential=False` to fall "
                "back to the legacy sentinel."
            )
        pad_row = ph_pad[matched[0]]

        # cripser always emits 9 columns regardless of dimensionality:
        # [dim, birth, death, bx, by, bz, dx, dy, dz]. The death pixel
        # therefore lives at the fixed columns 6:6+n_dim, NOT at 3+n_dim.
        # The pixel_offset/clip below only does real work for 'bbox' (a
        # 1-voxel shell shifts coords by 1); 'dilate' keeps the original
        # shape and frame, so pixel_offset is all-zero and clip is a no-op.
        shape = np.array(self.data.shape)
        h_all[essential_idx, 2] = pad_row[2]
        death_pix_padded = pad_row[6:6 + self.n_dim].astype(int)
        death_pix_orig = np.clip(
            death_pix_padded - pixel_offset, 0, shape - 1
        )
        h_all[essential_idx, 6:6 + self.n_dim] = death_pix_orig

    def compute_hom(data=None, max_Hi=None, wcs=None, flip_data=True, verbose=True, embedded=False,
                     noise=None,prep_img_kwargs={},
                     pad_essential='auto', pad_value=None):

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
        noise : np.ndarray
            Noise map of same shape as data.
        pad_essential : {'auto', 'dilate', 'bbox', False}, default 'auto'
            Strategy for giving the H_0 generator born at ``nanmax(data)``
            a finite death by running a second, H_0-only cripser pass on a
            padded copy of ``data``. The death and death-pixel columns of
            the originally-essential row are then patched into the primary
            generators table. All other rows are taken from the primary
            run untouched, so H_0/H_1/H_2 counts and finite-row values are
            unchanged.

            ``'auto'`` picks ``'dilate'`` if ``data`` contains NaN voxels,
            else ``'bbox'``. ``'dilate'`` fills the binary-dilation edge of
            the finite-valued region with ``pad_value`` and keeps the
            original array shape. ``'bbox'`` wraps the array in a 1-voxel
            shell of ``pad_value`` and translates padded-frame pixel coords
            back into the original frame. ``False`` skips padding entirely
            (legacy behavior — essential death = ``-DBL_MAX`` sentinel).
            Only the superlevel convention (``flip_data=True``) is supported.
        pad_value : float, optional
            Value placed in pad voxels. Defaults to ``10 * np.nanmax(data)``.
            The infilled essential death is a property of the data, not of
            ``pad_value``; varying ``pad_value`` over a wide range produces
            the same patched death (the regression suite asserts this).

        Returns:
        --------
        perch.PH
            Persistent homology object.

        Examples
        --------
        Compute persistent homology of a small 2D image with a single
        Gaussian peak:

        >>> import numpy as np
        >>> from perch.ph import PH
        >>> y, x = np.indices((8, 8))
        >>> img = np.exp(-((y - 3)**2 + (x - 4)**2) / 2.0).astype(np.float32)
        >>> ph = PH.compute_hom(data=img, verbose=False)
        >>> ph.generators.shape
        (1, 10)
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

        # resolve pad_essential mode (None if disabled / unsupported convention)
        pad_mode = self._resolve_pad_mode(pad_essential, self.data)
        if pad_mode and not flip_data:
            if pad_essential == 'auto':
                pad_mode = False  # silently disable under sublevel convention
            else:
                raise ValueError(
                    "pad_essential is only supported with flip_data=True "
                    "(the superlevel convention). Pass pad_essential=False "
                    "to compute with flip_data=False."
                )
        self.pad_essential = pad_mode

        # define PH computation engine
        self.ph_fxn = cripser.computePH

        if verbose:
            import time
            print('Computing PH... \n')
            t1 = time.time()

        # compute PH
        ph_all = self.ph_fxn(self.data_prep, maxdim=self.max_Hi, embedded=embedded)

        if verbose:
            t2 = time.time()
            print(f'\n PH Computation Complete! \n {t2-t1:.1f}s elapsed')

        # flip data back
        if flip_data:
            ph_all[:,1] = -ph_all[:,1]
            ph_all[:,2] = -ph_all[:,2]

        # add id
        h_id = np.arange(len(ph_all))
        h_all = np.hstack((ph_all, np.array(h_id).reshape(-1, 1)))

        # patch the originally-essential H_0 row with the padded run's death
        # (pad_value=None lets the helper apply its 10*nanmax default)
        if pad_mode:
            self._pad_and_patch_essential(h_all, pad_mode, pad_value, verbose,
                                          embedded=embedded)

        # store generators
        self.generators = h_all
        self.strucs = Structures(structures=h_all, img_shape=self.img_shape, wcs=self.wcs,inds_dir=None)
        self.data_prep = None # save memory

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

        Examples
        --------
        Keep only the H0 generators from a precomputed persistent-homology
        result:

        >>> import numpy as np
        >>> from perch.ph import PH
        >>> y, x = np.indices((8, 8))
        >>> img = np.exp(-((y - 3)**2 + (x - 4)**2) / 2.0).astype(np.float32)
        >>> ph = PH.compute_hom(data=img, verbose=False)
        >>> h0 = ph.filter(dimension=0)
        >>> h0.n_struc
        1

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


