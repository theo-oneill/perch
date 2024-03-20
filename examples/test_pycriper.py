import matplotlib.pyplot as plt
import time
import timeit
import cProfile
import pstats
from astropy.io import fits
import copy
from perch.py_cripser.cubicalripser_pybind import compute_ph
import numpy as np

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





