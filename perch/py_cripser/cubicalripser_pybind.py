import numpy as np
from cube import Cube
from write_pairs import WritePairs
from joint_pairs import JointPairs
from compute_pairs import ComputePairs
from config import Config
from dense_cubical_grids import DenseCubicalGrids
import matplotlib.pyplot as plt
import time
import timeit
import cProfile
import pstats
from astropy.io import fits
import copy


ddir = '/Users/toneill/Perch/phangs/'
img_hdu = fits.open(ddir+'ngc0628_miri_f770w_anchored.fits')
img_orig = img_hdu[0].data[500:1500,800:1800]
img = -copy.deepcopy(img_orig)
img[-1,-1] = 2*np.nanmin(img)




img_orig = np.array([
        [4, 4, 4, 4, 4, 4, 4],
       [4, 4, 2, 4, 3, 2, 4],
       [4, 2, 1, 4, 3, 4, 4],
       [4, 4, 2, 4, 4, 4, 4],
       [6, 4, 4, 4, 5, 4, 4],
       [6, 7, 4, 4, 6, 4, 4],
       [6, 6, 4, 5, 5, 4, 4]])

img_orig = np.array([
        [4, 4, 4, 4, 4, 4, 4],
       [4, 4, 2, 4, 3, 2, 4],
       [4, 2, 1, 2, 3, 4, 4],
       [4, 4, 2, 4, 4, 4, 4],
       [6, 4, 4, 4, 5, 4, 4],
       [6, 7, 4, 4, 6, 4, 4],
       [6, 6, 5, 5, 5, 4, 4]])



img_orig = np.array([
        [4, 4, 4, 4, 4, 4, 4],
       [4, 4, 4, 4, 2, 2, 4],
       [4, 1, 1, 4, 4, 4, 4],
       [4, 4, 4, 4, 4, 7, 4],
       [4, 7, 4, 4, 6, 7, 4],
       [7, 7, 4, 4, 4, 4, 4],
       [4, 4, 4, 4, 4, 4, 4]])

import copy

img = -copy.deepcopy(img_orig)
img[-1,-1] = -2*np.nanmax(img_orig)


maxdim = 1
top_dim = False
embedded = False
fortran_order = False

plt.imshow(img_orig,cmap='RdBu')#,vmin=-10)#,vmax=np.max(img_orig),vmin=np.min(img_orig))


def compute_ph(img, maxdim=0, top_dim=False, embedded=False, fortran_order=False):
    config = Config()
    config.format = "NUMPY"
    config.verbose = False

    writepairs = [] # (dim birth death x y z)
    dcg = DenseCubicalGrids(config)
    ctr = []

    shape = img.shape
    dcg.dim = img.ndim
    config.maxdim = min(maxdim, dcg.dim - 1)

    if top_dim and dcg.dim > 1:
        config.method = "ALEXANDER"
        config.embedded = not embedded
    else:
        config.embedded = embedded

    dcg.ax = shape[0]
    dcg.img_x = shape[0]
    if dcg.dim > 1:
        dcg.ay = shape[1]
        dcg.img_y = shape[1]
    else:
        dcg.ay = 1
        dcg.img_y = 1
    if dcg.dim > 2:
        dcg.az = shape[2]
        dcg.img_z = shape[2]
    else:
        dcg.az = 1
        dcg.img_z = 1

    dcg.gridFromArray(img, embedded, fortran_order,orig_method=False)#.flatten()

    if config.tconstruction:
        if dcg.az > 1:
            dcg.az += 1
        dcg.ax += 1
        dcg.ay += 1
    dcg.axy = dcg.ax * dcg.ay
    dcg.ayz = dcg.ay * dcg.az
    dcg.axyz = dcg.ax * dcg.ay * dcg.az

    if config.method == "ALEXANDER":
        jp = JointPairs(dcg, writepairs, config)
        if dcg.dim == 1:
            jp.enum_edges([0], ctr)
            jp.joint_pairs_main(ctr, 0) # dim0
        elif dcg.dim == 2:
            jp.enum_edges([0, 1, 3, 4], ctr)
            jp.joint_pairs_main(ctr, 1) # dim1
        elif dcg.dim == 3:
            jp.enum_edges(list(range(13)), ctr)
            jp.joint_pairs_main(ctr, 2) # dim2
    else:
        cp = ComputePairs(dcg, writepairs, config)
        betti = []
        seg_shape = np.shape(img) if dcg.dim > 2 else np.append(np.shape(img),1)
        seg_map = np.full(seg_shape,0)
        jp = JointPairs(dcg, writepairs, config, seg_map)

        if dcg.dim == 1:
            jp.enum_edges([0], ctr)
        elif dcg.dim == 2:
            jp.enum_edges([0, 1], ctr)
        else:
            jp.enum_edges([0, 1, 2], ctr)

        jp.joint_pairs_main(ctr, 0) # dim 0
        #plt.figure()
        #plt.imshow(jp.seg_map[:,:,0])
        betti.append(len(writepairs))
        print(f'B(H0): {betti[0]}')
        if config.maxdim > 0:
            cp.compute_pairs_main(ctr) # dim 1
            betti.append(len(writepairs) - betti[0])
            print(f'B(H1): {betti[1]}')
            if config.maxdim > 1:
                cp.assemble_columns_to_reduce(ctr, 2)
                cp.compute_pairs_main(ctr) # dim 2
                betti.append(len(writepairs) - betti[0] - betti[1])

    pad_x = (dcg.ax - dcg.img_x) // 2
    pad_y = (dcg.ay - dcg.img_y) // 2
    pad_z = (dcg.az - dcg.img_z) // 2

    result = np.zeros((len(writepairs), 9))

    for i, wp in enumerate(writepairs):
        result[i, 0] = wp.dim
        result[i, 1] = wp.birth
        result[i, 2] = wp.death
        result[i, 3] = wp.birth_x - pad_x
        result[i, 4] = wp.birth_y - pad_y
        result[i, 5] = wp.birth_z - pad_z
        result[i, 6] = wp.death_x - pad_x
        result[i, 7] = wp.death_y - pad_y
        result[i, 8] = wp.death_z - pad_z

    seg_map = jp.seg_map  if dcg.dim > 2 else  jp.seg_map[:,:,0]

    return result, seg_map

t1 = time.time()
res = compute_ph(img, maxdim=1, top_dim=False, embedded=False, fortran_order=False)
t2 = time.time()
print(f'{t2-t1:.4f} s')

odir = '/Users/toneill/Perch/profiling/'

t1 = time.time()
cProfile.run("compute_ph(img, maxdim=0, top_dim=False, embedded=False, fortran_order=False)",f'{odir}prof_pycripser_7x7')
t2 = time.time()
print(f'{t2-t1:.8f} s')
p = pstats.Stats(f"{odir}prof_pycripser_7x7")
p.sort_stats("cumulative").print_stats()







import copy

img_orig = np.array([
        [4, 4, 4, 4, 4, 4, 4],
       [4, 1, 4, 4, 2, 2, 4],
       [4, 1, 4, 4, 2, 4, 4],
       [4, 4, 4, 4, 4, 4, 4],
       [4, 7, 4, 4, 6, 7, 4],
       [7, 7, 4, 4, 6, 4, 4],
       [4, 4, 4, 4, 4, 4, 4]])

flip = True
if flip:
    imguse = copy.deepcopy(img_orig)
    imguse[-1,-1] = -2*np.nanmax(imguse)
else:
    imguse = -copy.deepcopy(img_orig)
    imguse[-1,-1] = 2*np.nanmin(imguse)

result, map = compute_ph(imguse, maxdim=1, top_dim=False, embedded=False, fortran_order=False)
len(result)

plt.figure()
plt.imshow(np.where(map>0,map,np.nan),cmap='tab10')

plt.figure()
plt.imshow(img_orig,cmap='RdBu_r')#,vmin=-10)
for i in np.where(result[:,2]<1e10):
    plt.plot([result[i,7],result[i,4]],[result[i,6],result[i,3]],c='k',ls='--',zorder=1)
plt.scatter(result[:, 4][result[:, 2] < 1e10], result[:, 3][result[:, 2] < 1e10], c='c', label='Birth',zorder=5,s=15)#,s=10)
plt.scatter(result[:,7][result[:,2]<1e10],result[:,6][result[:,2]<1e10],c='r',marker='x',label='Death')
#plt.scatter(result[:, 4][result[:, 2] < 1e10], result[:, 3][result[:, 2] < 1e10], c=result[:,0][result[:,2]<1e10],facecolor='None', marker='*',cmap='PiYG',alpha=1,s=100)
plt.legend(loc='center right')
plt.title('pyCripser')



plt.figure()
plt.scatter(result[:,1][result[:,2]<1e10],result[:,2][result[:,2]<1e10],alpha=0.5)
#plt.plot([1,8],[1,8])
plt.plot([-100,0],[-100,0],c='k',ls='--')


t1 = time.time()
res = compute_ph(img, maxdim=0, top_dim=False, embedded=False, fortran_order=False)
t2 = time.time()
print(f'{t2-t1:.4f} s')

%timeit -n 1000 compute_ph(imguse, maxdim=1, top_dim=False, embedded=False, fortran_order=False)


odir = '/Users/toneill/Perch/profiling/'


t1 = time.time()
cProfile.run("compute_ph(img, maxdim=0, top_dim=False, embedded=False, fortran_order=False)",f'{odir}prof_pycripser_sub1000x1000.phangs_h0')
t2 = time.time()
print(f'{t2-t1:.8f} s')
p = pstats.Stats(f"{odir}prof_pycripser_sub1000x1000.phangs_h0")
p.sort_stats("cumulative").print_stats()

#######

cProfile.run("compute_ph(imguse, maxdim=1, top_dim=False, embedded=False, fortran_order=False)",f'{odir}prof_pycripser')

p = pstats.Stats(f"{odir}prof_pycripser")
p.sort_stats("cumulative").print_stats()


from astropy.io import fits
ddir = '/Users/toneill/Perch/phangs/'
img_hdu = fits.open(ddir+'ngc0628_miri_f770w_anchored.fits')
img = img_hdu[0].data[500:1000,1000:1500]
plt.imshow(imguse,origin='lower',vmin=-10)

imguse = -copy.deepcopy(img)
imguse[-1,-1] = 2*np.nanmin(imguse)





plt.figure()


t1 = time.time()
result = compute_ph(img, maxdim=0, top_dim=False, embedded=False, fortran_order=False)
t2 = time.time()
print(f'{t2-t1:.8f} s')
len(result)






