from typing import List
from perch.py_cripser.dense_cubical_grids import DenseCubicalGrids
import numpy as np

class UnionFind:
    def __init__(self, dcg: DenseCubicalGrids):
        self.parent = []
        self.birthtime = []
        self.time_max = []

        n = dcg.ax * dcg.ay * dcg.az
        for i in range(n):
            self.parent.append(i)
            self.birthtime.append(dcg.get_birth(i % dcg.ax, (i // dcg.ax) % dcg.ay, (i // (dcg.ax * dcg.ay)) % dcg.az))
            self.time_max.append(self.birthtime[i]) # maximum filtration value for the group'''

        self.seg = np.repeat(0,n)

    #def __init__(self, dcg: DenseCubicalGrids):

        '''n = dcg.ax * dcg.ay * dcg.az
        #n = np.product(np.shape(dcg.dense3))
        self.parent = list(range(n))
        self.birthtime = [dcg.get_birth(x, y, z) for z in range(dcg.az) for y in range(dcg.ay) for x in range(dcg.ax)]
        #self.birthtime = [dcg.get_birth(x, y, z) for z in range(np.shape(dcg.dense3)[0]) for y in range(np.shape(dcg.dense3)[1]) for x in range(np.shape(dcg.dense3)[2])]
        self.time_max = self.birthtime[:]'''

        '''
        ### THIS VERSION DOES NOT WORK!!!!
        n = dcg.ax * dcg.ay * dcg.az

        self.parent = np.arange(n)
        self.birthtime = dcg.get_birth_bulk(self.parent % dcg.ax, (self.parent // dcg.ax) % dcg.ay, (self.parent // (dcg.ax * dcg.ay)) % dcg.az )
        self.time_max = self.birthtime # maximum filtration value for the group'''

    # find the root of a node x (specified by the index)
    def find(self, x: int) -> int:
        y = x
        z = self.parent[y]
        while z != y:
            y = z
            z = self.parent[y]
        # reassign parents to the found root z
        y = self.parent[x]
        while z != y:
            self.parent[x] = z
            x = y
            y = self.parent[x]
        return z

    #  merge nodes x and y (they should be root nodes); older will be the new parent
    def link(self, x: int, y: int):
        #print('beginning link')
        if x == y:
            #print('x = y')
            return
        if self.birthtime[x] >= self.birthtime[y]:
            #print('birth x > birth y')
            #print(f'initial parent x: {self.parent[x]}')
            self.parent[x] = y
            #print(f'updated parent x: {self.parent[x]}')
            self.time_max[y] = max(self.time_max[x], self.time_max[y])
        elif self.birthtime[x] < self.birthtime[y]:
            #print('birth x < birth y')
            #print(f'initial parent y: {self.parent[y]}')
            self.parent[y] = x
            #print(f'updated parent y: {self.parent[y]}')
            self.time_max[x] = max(self.time_max[x], self.time_max[y])
