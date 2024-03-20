
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
import jax.numpy as jnp
import os
from perch.perch_ph import PH
from perch.perch_structures import Structures
import matplotlib.pyplot as plt

rname = 'random'
perchdir = f'/Users/toneill/Perch/output/'
odir = f'{perchdir}{rname}/'
if not os.path.isdir(perchdir):
    os.mkdir(perchdir)
if not os.path.isdir(odir):
    os.mkdir(odir)
os.chdir(odir)


import porespy as ps


blob_lev = 1
img2d = ps.generators.blobs(shape=[1000, 1000], porosity=None, blobiness=[blob_lev,blob_lev])

plt.figure()
plt.imshow(img2d)

img3d = ps.generators.blobs(shape=[200, 200,200], porosity=None)#, blobiness=[1, 1])

plt.imshow(img3d[100,:,:])


hom = PH.compute_hom(img2d,verbose=True,engine='C')
hom = PH.compute_hom(img2d,verbose=True,engine='py')

from perch.py_cripser import cubicalripser_pybind


hom.barcode()
hom.pers_diagram()
hom.lifetime_diagram()

hi = hom.generators
h0 = hi[hi[:,0]==0]
h1 = hi[hi[:,0]==1]

img_shape = np.shape(img2d)
img_jnp = jnp.array(img2d)

h_use = h1[h1[:,1]-h1[:,2] >= np.nanquantile(h1[:,1]-h1[:,2],0.95)]
h_use = h1
strucs = Structures(structures=h_use, img_shape=img_shape)


plt.figure()
plt.hist(strucs.persistence,bins=30)

np.nanquantile(strucs.persistence,0.9772)

import time

t1 = time.time()
strucs.compute_segment_hierarchy(img_jnp,clobber=True)
t2 = time.time()
print(f'{t2-t1:.1f}s elapsed')

plt.figure()
plt.imshow(strucs.struc_map)

plt.figure()
plt.imshow(img2d,cmap='Greys_r')
plt.imshow(strucs.struc_map,alpha=0.9,cmap='viridis')

slice_use = 225
plt.figure()
plt.imshow((img[slice_use,:,:]),vmax=np.nanquantile(img[slice_use,:,:],0.9),cmap='Greys_r',origin='lower')
plt.imshow(strucs.struc_map[slice_use,:,:],cmap='tab20',alpha=1,origin='lower')



plt.figure()
plt.scatter(h0[:,1],h0[:,2])

plt.figure()
plt.scatter(h1[:,1],h1[:,2])










