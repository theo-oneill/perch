
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
import jax.numpy as jnp
import os
from perch.perch_ph import PH
from perch.perch_structures import Structures
import matplotlib.pyplot as plt

#rname = 'full'

rname = 'example'
ddir = '/Users/toneill/Repos/dustmaps_maps/edenhofer_2023/'
perchdir = f'/Users/toneill/Perch/output/'
odir = f'{perchdir}{rname}/'

if not os.path.isdir(perchdir):
    os.mkdir(perchdir)
if not os.path.isdir(odir):
    os.mkdir(odir)
os.chdir(odir)

img_hdu = fits.open(ddir+'mean_and_std_xyz.fits')
wcs_img = WCS(img_hdu[1].header)[375:875,375:875,375:875]
img = img_hdu[1].data[375:875,375:875,375:875]


#######################################################
## STEP 1: compute persistent homology candidates

hom = PH.compute_hom(img,verbose=True,wcs=wcs_img)
hom.export_generators(f'{rname}.txt',odir=odir)

#######################################################
## STEP 2: filter candidates by some threshold

dim_use = 2
rel_thresh = 0.1

hom = PH.load_from(f'{rname}.txt',odir=odir,data=img)

h0 = hom.filter(dimension=dim_use,min_death=0,min_birth=0)
h0_gen = h0

all_life = h0_gen[:,1]-h0_gen[:,2]
rel_life = all_life / h0_gen[:,1]


h0_use = np.where((rel_life >=rel_thresh))
h0_filt = h0_gen[h0_use]
h0_save = np.hstack((h0_filt, np.array(h0_use).reshape(-1,1)))

np.savetxt(f'{odir}{rname}_h{dim_use}_filt_rellife{rel_thresh}.txt', h0_save)

plt.figure()
plt.scatter(h0_gen[:,2],rel_life,s=0.1)
plt.gca().set_xscale('log')
plt.gca().set_yscale('linear')
plt.axhline(y=rel_thresh,c='r',ls='--')

#######################################################
## STEP 3: segment original image

hom = PH.load_from(f'{rname}_h{dim_use}_filt_rellife{rel_thresh}.txt',odir=odir,data=img)

h0 = hom.generators

inds_dir = f'{odir}inds/h{dim_use}/'
if not os.path.isdir(f'{odir}inds/'):
    os.mkdir(f'{odir}inds/')
if not os.path.isdir(inds_dir):
    os.mkdir(inds_dir)

img_shape = np.shape(img)
img_jnp = jnp.array(img)

h0_use = h0
strucs = Structures(structures=h0_use[strucs.npix<10**7], img_shape=img_shape, wcs=wcs_img,inds_dir=inds_dir)

import time

t1 = time.time()
strucs.compute_segment(img_jnp)
t2 = time.time()
print(f'{t2-t1:.1f}s elapsed')


t1 = time.time()
strucs.compute_hierarchy(clobber=True)
t2 = time.time()
print(f'{t2-t1:.1f}s elapsed')



t1 = time.time()
strucs.compute_segment_hierarchy(img_jnp,clobber=True)
t2 = time.time()
print(f'{t2-t1:.1f}s elapsed')

slice_use = 225
plt.figure()
plt.imshow((img[slice_use,:,:]),vmax=np.nanquantile(img[slice_use,:,:],0.9),cmap='Greys_r',origin='lower')
plt.imshow(strucs.struc_map[slice_use,:,:],cmap='tab20',alpha=1,origin='lower')

fname = f'{rname}_h{dim_use}_filt_rellife{rel_thresh}_npix10.7'
strucs.export_struc_map(fname=fname, odir=odir)

strucs.clear_struc_map()
strucs.export_collection(fname=fname, odir=odir)
strucs.load_struc_map(fname=fname, odir=odir)

i = 131

imap = strucs.struc_map == i

plt.figure()
plt.imshow((img[slice_use,:,:]),vmax=np.nanquantile(img[slice_use,:,:],0.9),cmap='Greys_r',origin='lower')
plt.imshow(imap[slice_use,:,:],cmap='viridis',alpha=1,origin='lower')


strucs.birthpix[i]
strucs.deathpix[i]



















plt.figure()
plt.scatter(strucs.persistence,strucs.npix)
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')


plt.figure()
plt.scatter(strucs.death,strucs.npix)
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')



#######

import timeit
import time
from jax import jit
import cc3d

img_slice = img#[0,:,:]
img_slice_jnp = jnp.array(img_slice)

def filter_super(X,thresh):
    return X > thresh
filter_super_jit = jit(filter_super)

def filter_sub(X,thresh):
    return X < thresh
filter_sub_jit = jit(filter_sub)

def filter_eq(X,thresh):
    return X == thresh
filter_eq_jit = jit(filter_eq)

def extract_inds(labels_out,comp_use):
    return np.where(labels_out == comp_use)

nstruc = 2000

t1 = time.time()
filt_img = np.array(filter_super_jit(img_jnp, strucs.death[nstruc]))
labels_out = cc3d.connected_components(filt_img, connectivity=6)
comp_use = labels_out[strucs.birthpix[nstruc,:][0],strucs.birthpix[nstruc,:][1],strucs.birthpix[nstruc,:][2]]
uinds = extract_inds(labels_out,comp_use)
t2 = time.time()
print(f'{t2-t1:.1f}s')

nstruc = 2000

t1 = time.time()
filt_img = np.array(filter_super_jit(img_jnp, strucs.death[nstruc]))
labels_out = cc3d.connected_components(filt_img, connectivity=6)
comp_use = labels_out[strucs.birthpix[nstruc,:][0],strucs.birthpix[nstruc,:][1],strucs.birthpix[nstruc,:][2]]
uinds = extract_inds(labels_out,comp_use)
t2 = time.time()
print(f'{t2-t1:.1f}s')



slice_z = img[strucs.birthpix[nstruc,:][0],:,:]
filt_z = slice_z >= strucs.death[nstruc]
plt.imshow(filt_z)
labels_slice = cc3d.connected_components(filt_z, connectivity=4)
comp_slice= labels_slice[strucs.birthpix[nstruc,:][1],strucs.birthpix[nstruc,:][2]]

%timeit img >= 1e-3

%timeit np.array(filter_super_jit(img_slice_jnp,1e-3))


t1 = time.time()
filt_z = img >= 1e-4
t2 = time.time()
print(f'{t2-t1:.1f}s')
