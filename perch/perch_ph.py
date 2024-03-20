import cripser
import copy
import numpy as np
import matplotlib.pyplot as plt
from perch.perch_utils import notify
from perch.perch_structures import Structures

hcol = {0: 'palevioletred', 1: 'mediumpurple', 2: 'deepskyblue'}
hnames = {0: '$H_0$', 1: '$H_1$', 2: '$H_2$'}

class PH(object):

    def __init__(self):
        self.data = None
        self.data_prep = None
        self.n_dim = 0
        self.max_Hi = None
        self.generators = None
        self.strucs = None

    ####################################################
    ## compute PH (or load from stored)

    def prep_img(self):
        img_prep = copy.deepcopy(self.data)
        img_prep = -img_prep
        if self.n_dim == 2:
                img_prep[0:2, 0:2] = np.nanmin(img_prep) * 2
        if self.n_dim == 3:
                img_prep[0:2, 0:2,0:2] = np.nanmin(img_prep) * 2
        return img_prep

    def compute_hom(data, max_Hi=None, wcs=None, flip_data=True, verbose=True, embedded=False):

        self = PH()
        self.data = data
        self.n_dim = len(data.shape)
        self.wcs = wcs
        self.img_shape = data.shape
        if flip_data:
            self.data_prep = self.prep_img()
        if not flip_data:
            self.data_prep = self.data

        if max_Hi is None:
            max_Hi = self.n_dim - 1
        self.max_Hi = max_Hi

        if verbose:
            import time
            print('Computing PH... \n')
            t1 = time.time()

        ph_all = cripser.computePH(self.data_prep, maxdim=self.max_Hi, embedded=embedded)

        if verbose:
            t2 = time.time()
            print(f'\\n PH Computation Complete! \n {t2-t1:.1f}s elapsed')
            #notify("Alert", f"PH Computation Complete!")

        if flip_data:
            ph_all[:,1] = -ph_all[:,1]
            ph_all[:,2] = -ph_all[:,2]
            base_struc = ph_all[:,1] == -2*np.nanmin(-self.data)
            ph_all = ph_all[~base_struc]

        h_id = np.arange(len(ph_all))
        h_all = np.hstack((ph_all, np.array(h_id).reshape(-1, 1)))

        self.generators = h_all
        self.strucs = Structures(structures=h_all, img_shape=self.img_shape, wcs=self.wcs,inds_dir=None)
        self.data_prep = None # save memory

        return self

    def export_generators(self, fname, odir='./'):
        np.savetxt(f'{odir}{fname}', self.generators)

    def load_from(fname, odir='./',data=None,wcs=None, max_Hi=None):
        gens = np.loadtxt(f'{odir}{fname}')
        self = PH()
        self.data = data
        self.n_dim = len(data.shape)
        self.wcs = wcs
        if max_Hi is None:
            max_Hi = self.n_dim - 1
        self.max_Hi = max_Hi
        self.generators = gens

        return self

    ####################################################
    ## filtering and segmentation

    def filter(self, dimension=None, min_life=None, max_life=None,
                      min_birth=None, max_birth=None,
                      min_death=None, max_death=None):

        ppd = self.generators
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

        return ppd

    ####################################################
    ## plotting functions

    def barcode(self,ax=None):#,dimensions=None):

        '''if dimensions == None:
            dimensions = list(np.unique(self.generators[:,0]).astype('int'))
        if type(dimensions) != list:
            print('fixing')
            dimensions = list(dimensions)

        plotcol = [self.filter(dimension=d) for d in dimensions]
        ravpc = self.generators#np.array(plotcol).ravel()'''
        plotcol = self.generators

        #plotcol = self.generators
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,10))
        for k in range(len(plotcol)):
            p_i = plotcol[k]
            ax.plot(np.array([p_i[1], p_i[2]]), [k, k], c=hcol[p_i[0]])
        ax.set_xlabel('Birth –-- Death')
        ax.set_ylabel('Structure Number')
        #ax.set_xscale('linear')
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in list(hcol.values())[0:self.max_Hi+1]]
        ax.legend(markers, hnames.values(), numpoints=1,fontsize=14)
        fig.tight_layout()

    def pers_diagram(self,ax=None,dimensions=None):

        if dimensions == None:
            dimensions = list(np.unique(self.generators[:,0]).astype('int'))
        if type(dimensions) != list:
            print('fixing')
            dimensions = list(dimensions)

        plotcol = [self.filter(dimension=d) for d in dimensions]
        ravpc = self.generators#np.array(plotcol).ravel()

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,8))
        for d in range(len(dimensions)):
            ax.scatter(plotcol[d][:,1],plotcol[d][:,2],s=3,alpha=0.5,c=hcol[dimensions[d]])#,c=hcol[plotcol[:,0]])
        ax.plot([-np.nanmax(np.abs(ravpc[:,1:3])),np.nanmax(np.abs(ravpc[:,1:3]))],
                [-np.nanmax(np.abs(ravpc[:,1:3])),np.nanmax(np.abs(ravpc[:,1:3]))],c='grey',ls='--',alpha=0.5,lw=0.5)
        ax.set_xlim(1.1*np.nanmin(ravpc[:,1]),1.1*np.nanmax(ravpc[:,1]))
        ax.set_ylim(1.1*np.nanmin(ravpc[:,2]),1.1*np.nanmax(ravpc[:,2]))
        ax.set_xlabel('Birth',fontsize=14)
        ax.set_ylabel('Death',fontsize=14)
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in list(hcol.values())[0:self.max_Hi+1]]
        ax.legend(markers, hnames.values(), numpoints=1,fontsize=14)
        fig.tight_layout()

    def lifetime_diagram(self,ax=None,dimensions=None):

        if dimensions == None:
            dimensions = list(np.unique(self.generators[:,0]).astype('int'))
        if type(dimensions) != list:
            print('fixing')
            dimensions = list(dimensions)

        plotcol = [self.filter(dimension=d) for d in dimensions]
        #ravpc = self.generators#np.array(plotcol).ravel()

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,8))
        for d in range(len(dimensions)):
            ax.scatter(plotcol[d][:,1],np.abs(plotcol[d][:,2]-plotcol[d][:,1]),s=3,alpha=0.5,c=hcol[dimensions[d]])#,c=hcol[plotcol[:,0]])
        #ax.set_xlim(1.1*np.min(plotcol[:,1]),1.1*np.max(plotcol[:,1]))
        #ax.set_ylim(1.1*np.min(plotcol[:,2]),1.1*np.max(plotcol[:,2]))
        ax.set_xlabel('Birth',fontsize=14)
        ax.set_ylabel('Lifetime',fontsize=14)
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in list(hcol.values())[0:self.max_Hi+1]]
        ax.legend(markers, hnames.values(), numpoints=1,fontsize=14)
        fig.tight_layout()
