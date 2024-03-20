from cube import Cube
from dense_cubical_grids import DenseCubicalGrids

class CoboundaryEnumerator:
    def __init__(self, dcg: DenseCubicalGrids, dim: int):
        self.dim = dim
        self.dcg = dcg
        self.nextCoface = Cube()

    def setCoboundaryEnumerator(self, s: Cube):
        self.cube = s
        self.position = 0

    def hasNextCoface(self):
        dcg = self.dcg
        cube = self.cube
        dim = self.dim
        threshold = dcg.threshold
        birth = 0

        # note the shift of indices to account for the boundary
        cx = cube.x#()
        cy = cube.y#()
        cz = cube.z#()

        if dim == 0: # dim0
            for i in range(self.position, 6):
                if i == 0:
                    birth = max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz + 2])
                    self.nextCoface = Cube(birth, cx, cy, cz, 2)
                elif i == 1:
                    birth = max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz])
                    self.nextCoface = Cube(birth, cx, cy, cz - 1, 2)
                elif i == 2:
                    birth = max(cube.birth, dcg.dense3[cx + 1][cy + 2][cz + 1])
                    self.nextCoface = Cube(birth, cx, cy, cz, 1)
                elif i == 3:
                    birth = max(cube.birth, dcg.dense3[cx + 1][cy][cz + 1])
                    self.nextCoface = Cube(birth, cx, cy - 1, cz, 1)
                elif i == 4:
                    birth = max(cube.birth, dcg.dense3[cx + 2][cy + 1][cz + 1])
                    self.nextCoface = Cube(birth, cx, cy, cz, 0)
                elif i == 5:
                    birth = max(cube.birth, dcg.dense3[cx][cy + 1][cz + 1])
                    self.nextCoface = Cube(birth, cx - 1, cy, cz, 0)

                if birth != threshold:
                    self.position = i + 1
                    return True
            return False

        elif dim == 1: # dim1
            cube_m = cube.m
            for i in range(self.position, 4):
                if cube_m == 0: # dim1 type0 (x-axis -> )
                    if i == 0:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz + 2]),
                                    dcg.dense3[cx + 2][cy + 1][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 1)
                    elif i == 1:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz]),
                                    dcg.dense3[cx + 2][cy + 1][cz])
                        self.nextCoface = Cube(birth, cx, cy, cz - 1, 1)
                    elif i == 2:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy + 2][cz + 1]),
                                    dcg.dense3[cx + 2][cy + 2][cz + 1])
                        self.nextCoface = Cube(birth, cx, cy, cz, 0)
                    elif i == 3:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy][cz + 1]),
                                    dcg.dense3[cx + 2][cy][cz + 1])
                        self.nextCoface = Cube(birth, cx, cy - 1, cz, 0)
                elif cube_m == 1: # dim1 type1 (y-axis -> )
                    if i == 0:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz + 2]),
                                    dcg.dense3[cx + 1][cy + 2][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 2)
                    elif i == 1:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz]),
                                    dcg.dense3[cx + 1][cy + 2][cz])
                        self.nextCoface = Cube(birth, cx, cy, cz - 1, 2)
                    elif i == 2:
                        birth = max(max(cube.birth, dcg.dense3[cx + 2][cy + 1][cz + 1]),
                                    dcg.dense3[cx + 2][cy + 2][cz + 1])
                        self.nextCoface = Cube(birth, cx, cy, cz, 0)
                    elif i == 3:
                        birth = max(max(cube.birth, dcg.dense3[cx][cy + 1][cz + 1]),
                                    dcg.dense3[cx][cy + 2][cz + 1])
                        self.nextCoface = Cube(birth, cx - 1, cy, cz, 0)
                elif cube_m == 2: # dim1 type2 (z-axis -> )
                    if i == 0:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy + 2][cz + 1]),
                                    dcg.dense3[cx + 1][cy + 2][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 2)
                    elif i == 1:
                        birth = max(max(cube.birth, dcg.dense3[cx + 1][cy][cz + 1]),
                                    dcg.dense3[cx + 1][cy][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy - 1, cz, 2)
                    elif i == 2:
                        birth = max(max(cube.birth, dcg.dense3[cx + 2][cy + 1][cz + 1]),
                                    dcg.dense3[cx + 2][cy + 1][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 1)
                    elif i == 3:
                        birth = max(max(cube.birth, dcg.dense3[cx][cy + 1][cz + 1]),
                                    dcg.dense3[cx][cy + 1][cz + 2])
                        self.nextCoface = Cube(birth, cx - 1, cy, cz, 1)

                if birth != threshold:
                    self.position = i + 1
                    return True
            return False

        elif dim == 2: # dim 2
            cube_m = cube.m
            for i in range(self.position, 2):
                if cube_m == 0: # dim2 type0 (fix z)
                    if i == 0: # upper
                        birth = max(max(max(max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz + 2]),
                                             dcg.dense3[cx + 2][cy + 1][cz + 2]),
                                      dcg.dense3[cx + 1][cy + 2][cz + 2]), dcg.dense3[cx + 2][cy + 2][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 0)
                    elif i == 1: # lower
                        birth = max(max(max(max(cube.birth, dcg.dense3[cx + 1][cy + 1][cz]),
                                             dcg.dense3[cx + 2][cy + 1][cz]),
                                      dcg.dense3[cx + 1][cy + 2][cz]), dcg.dense3[cx + 2][cy + 2][cz])
                        self.nextCoface = Cube(birth, cx, cy, cz - 1, 0)
                elif cube_m == 1: # dim2 type1 (fix y)
                    if i == 0: # left
                        birth = max(max(max(max(cube.birth, dcg.dense3[cx + 1][cy + 2][cz + 1]),
                                             dcg.dense3[cx + 2][cy + 2][cz + 1]),
                                      dcg.dense3[cx + 1][cy + 2][cz + 2]), dcg.dense3[cx + 2][cy + 2][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 0)
                    elif i == 1: # right
                        birth = max(max(max(max(cube.birth, dcg.dense3[cx + 1][cy][cz + 1]),
                                             dcg.dense3[cx + 2][cy][cz + 1]),
                                      dcg.dense3[cx + 1][cy][cz + 2]), dcg.dense3[cx + 2][cy][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy - 1, cz, 0)
                elif cube_m == 2: #  dim2 type2 (fix x)
                    if i == 0: # left
                        birth = max(max(max(max(cube.birth, dcg.dense3[cx + 2][cy + 1][cz + 1]),
                                             dcg.dense3[cx + 2][cy + 2][cz + 1]),
                                      dcg.dense3[cx + 2][cy + 1][cz + 2]), dcg.dense3[cx + 2][cy + 2][cz + 2])
                        self.nextCoface = Cube(birth, cx, cy, cz, 0)
                    elif i == 1: # right
                        birth = max(max(max(max(cube.birth, dcg.dense3[cx][cy + 1][cz + 1]),
                                             dcg.dense3[cx][cy + 2][cz + 1]),
                                      dcg.dense3[cx][cy + 1][cz + 2]), dcg.dense3[cx][cy + 2][cz + 2])
                        self.nextCoface = Cube(birth, cx - 1, cy, cz, 0)

                if birth != threshold:
                    self.position = i + 1
                    return True
            return False
