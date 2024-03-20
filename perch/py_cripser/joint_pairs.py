from typing import List
from cube import Cube, CubeComparator#, cube_comp, get_cube_comparator
from dense_cubical_grids import DenseCubicalGrids
from union_find import UnionFind
from write_pairs import WritePairs
from config import Config
import pdb
import numba
from functools import cmp_to_key
import concurrent.futures
import time
from multiprocess import Pool
import pdb
import numpy as np

class JointPairs:
    def __init__(self, dcg: DenseCubicalGrids, wp: List[WritePairs], config: Config, seg_map):
        self.dcg = dcg
        self.wp = wp
        self.config = config
        self.seg_map = seg_map

    #@numba.njit()#parallel=True)
    def enum_edges(self, types: List[int], ctr: List[Cube]):
        ctr.clear()
        # the order of loop matters for performance!
        #t1 = time.time()
        for m in types:
            for z in range(self.dcg.az):
                for y in range(self.dcg.ay):
                    for x in range(self.dcg.ax):
                        birth = self.dcg.get_birth_with_cm(x, y, z, m, 1)
                        if birth < self.config.threshold:
                            ctr.append(Cube(birth, x, y, z, m))
        #t2 = time.time()
        #print(f'births: {t2-t1:.4f} s')
        #t1 = time.time()
        ctr.sort(reverse=False)
        #t2 = time.time()
        #print(f'sorting: {t2-t1:.4f} s')
        #ctr.sort(reverse=False, key=CubeComparator())
        #ctr.sort(reverse=False, key=CubeComparator())
        #return ctr.sort(reverse=False, key=cmp_to_key(CubeComparator()))
        #return sorted(ctr,key=cmp_to_key(CubeComparator()))
        #return sorted(ctr, reverse=False, key=cmp_to_key(cube_comp))
        #return ctr'''


    #@numba.jit(nopython=False)
    #def sort_edges(self,ctr):
    #    ctr.sort(reverse=False)
    #@numba.njit(nopython=True,parallel=True)
    '''def enum_edges(self, types: List[int], ctr: List[Cube]):
        ctr.clear()
        n = len(types) * self.dcg.az * self.dcg.ay * self.dcg.ax
        ctr = [None] * n

        i = 0
        for m in types:
            for z in range(self.dcg.az):
                for y in range(self.dcg.ay):
                    for x in range(self.dcg.ax):
                        birth = self.dcg.get_birth_with_cm(x, y, z, m, 1)
                        if birth < self.config.threshold:
                            #ctr.append(Cube(birth, x, y, z, m))
                            ctr[i] = Cube(birth, x, y, z, m)
                        i += 1
        #ctr.sort(reverse=False, key=CubeComparator())
        #print(f'full possible length ctr: {len(ctr)}')
        ctr = [c for c in ctr if c is not None]
        #print(f'len ctr pre sort: {len(ctr)}')
        #self.sort_edges(ctr)
        ctr.sort(reverse=False)
        #print(f'len ctr post sort: {len(ctr)}')
        return ctr#'''

    # compute H0 by union find
    def joint_pairs_main(self, ctr: List[Cube], current_dim: int):

        dset = UnionFind(self.dcg)
        min_birth = self.config.threshold
        min_idx = 0

        struc_id = 1

        for e in reversed(ctr):
            # indexing scheme for union find is DIFFERENT from that of cubes
            uind = e.x + self.dcg.ax * e.y + self.dcg.axy * e.z

            # for each edge e, identify root indices u and v of the end points
            u = dset.find(uind)

            # Determining the index of the adjacent cell based on type
            if e.m == 0:
                vind = uind + 1  # x+1
            elif e.m == 1:
                vind = uind + self.dcg.ax  # y+1
            elif e.m == 2:
                vind = uind + self.dcg.axy  # z+1
            # T-construction
            elif e.m == 3:
                vind = uind + 1 + self.dcg.ax  # x+1,y+1
            elif e.m == 4:
                vind = uind + 1 - self.dcg.ax  # x+1,y-1
            # 3D T-construction only
            elif e.m == 5:
                vind = uind - self.dcg.ax + self.dcg.axy  # y-1,z+1
            elif e.m == 6:
                vind = uind + self.dcg.ax + self.dcg.axy  # y+1,z+1
            elif e.m == 7:
                vind = uind + 1 - self.dcg.ax + self.dcg.axy  # x+1,y-1,z+1
            elif e.m == 8:
                vind = uind + 1 + self.dcg.axy  # x+1,z+1
            elif e.m == 9:
                vind = uind + 1 + self.dcg.ax + self.dcg.axy  # x+1,y+1,z+1
            elif e.m == 10:
                vind = uind + 1 - self.dcg.ax - self.dcg.axy  # x+1,y-1,z-1
            elif e.m == 11:
                vind = uind + 1 - self.dcg.axy  # x+1,z-1
            elif e.m == 12:
                vind = uind + 1 + self.dcg.ax - self.dcg.axy  # x+1,y+1,z-1

            v = dset.find(vind)

            if u != v:
                if dset.birthtime[u] >= dset.birthtime[v]:
                    # Handle the case when the younger component u is killed
                    birth = dset.birthtime[u]
                    if current_dim == 0:
                        death_ind = uind if dset.birthtime[uind] > dset.birthtime[vind] else vind # cell of which the location is recorded
                        birth_ind = u
                    else:
                        death_ind = u
                        birth_ind = uind if dset.birthtime[uind] > dset.birthtime[vind] else vind
                    if dset.birthtime[v] < min_birth:
                        min_birth = dset.birthtime[v]
                        min_idx = v
                else:
                    # Handle the case when the younger component v is killed
                    birth = dset.birthtime[v]
                    if current_dim == 0:
                        death_ind = uind if dset.birthtime[uind] > dset.birthtime[vind] else vind
                        birth_ind = v
                    else:
                        death_ind = v
                        birth_ind = uind if dset.birthtime[uind] > dset.birthtime[vind] else vind
                    if dset.birthtime[u] < min_birth:
                        min_birth = dset.birthtime[u]
                        min_idx = u
                death = e.birth
                dset.link(u, v)
                if birth != death:
                    if self.config.tconstruction:
                        self.wp.append(WritePairs(current_dim, birthC = Cube(birth, birth_ind % self.dcg.ax,
                                                                    (birth_ind // self.dcg.ax) % self.dcg.ay,
                                                                    (birth_ind // self.dcg.axy) % self.dcg.az, 0),
                                                  deathC = Cube(death, death_ind % self.dcg.ax,
                                                       (death_ind // self.dcg.ax) % self.dcg.ay,
                                                       (death_ind // self.dcg.axy) % self.dcg.az, 0), dcg=self.dcg,
                                                  print_flag=0))
                    else:
                        self.wp.append(WritePairs(current_dim, birth=birth, death=death,
                                                  birth_x = birth_ind % self.dcg.ax,
                                                 birth_y= (birth_ind // self.dcg.ax) % self.dcg.ay,
                                                 birth_z= (birth_ind // self.dcg.axy) % self.dcg.az,
                                                  death_x = death_ind % self.dcg.ax,
                                                 death_y = (death_ind // self.dcg.ax) % self.dcg.ay,
                                                 death_z =  (death_ind // self.dcg.axy) % self.dcg.az, print_flag = 0))
                        #pdb.set_trace()
                        #print(f'segmenting struc {struc_id}')
                        seg_union_inds, = np.where((np.array(dset.parent) == birth_ind) & (np.array(dset.time_max) < death) & (np.array(dset.seg)==0))
                        seg_union_inds = np.append(seg_union_inds,birth_ind)
                        self.seg_map[seg_union_inds % self.dcg.ax, (seg_union_inds // self.dcg.ax) % self.dcg.ay,
                                                 (seg_union_inds // self.dcg.axy) % self.dcg.az] = struc_id
                        dset.seg[seg_union_inds] = 1
                        struc_id += 1

                e.index = None # column clearing

        # the base point component
        if current_dim == 0:
            self.wp.append(WritePairs(current_dim, birth = min_birth, death = self.dcg.threshold, birth_x = min_idx % self.dcg.ax,
                                      birth_y = (min_idx // self.dcg.ax) % self.dcg.ay, birth_z = (min_idx // self.dcg.axy) % self.dcg.az,
                                      death_x = 0, death_y = 0, death_z = 0, print_flag = 0))
        #  remove unnecessary edges
        if self.config.maxdim == 0 or current_dim > 0:
            return
        else:
            ctr[:] = [e for e in ctr if e.index is not None]
