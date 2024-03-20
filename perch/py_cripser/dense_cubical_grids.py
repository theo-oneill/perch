


############################################################
# DCG.h


"""
ChatGPT
This code defines a class DenseCubicalGrids and some supporting classes and functions.
Here's a summary of what it does:

3. Class DenseCubicalGrids:
Represents dense cubical grids.
Manages a 3D array (dense3) to store grid data.
Initializes grid properties like dimensions (ax, ay, az), image dimensions (img_x, img_y, img_z), and other related properties.
Provides methods for loading image data from different file formats (loadImage method).
Allocates memory for a 3D array (alloc3d method).
Constructs a volume with boundary conditions based on input data.
Implements methods for getting birth values (getBirth) and finding parent voxels (ParentVoxel).

4. File I/O:
The loadImage method reads image data from different file formats such as DIPHA, PERSEUS, CSV, and NUMPY.
Depending on the file format, it reads the header information and image data, and then constructs the dense cubical grid accordingly.

5. Memory Management:
Dynamically allocates memory for the 3D array (dense3) in the constructor.
Deallocates the memory in the destructor (~DenseCubicalGrids).

6. Boundary Handling:
Handles boundary conditions while constructing the volume from input data.
Inner and outer boundaries are differentiated based on the embedded flag.
"""

############################################################
# DCG.ccp

"""
ChatGPT
This code defines the implementation for the methods of the DenseCubicalGrids class,
which is part of the CubicalRipser project. Here's a breakdown of what the code does:

2. getBirth Method:
Overloaded method to return the filtration value for a cube at the specified voxel coordinates (cx, cy, cz).
Also takes optional arguments cm (corner mode) and dim (dimension) for handling different cases.
Calculates the birth value based on the cube's position in the dense grid.

3. ParentVoxel Method:
Determines the parent voxel coordinates for a given cube.
Takes the cube's dimension _dim and the cube object c as input.
Compares the birth value of the cube with the neighboring voxel values to find its parent voxel.

Overall, this code provides functionality to work with dense cubical grids,
including retrieving birth values for cubes and determining parent voxels.
"""


"""
The codimension of a cube refers to the difference between the dimension of the ambient space and the dimension of the cube itself. 

In other words, it represents the number of dimensions "missing" from the ambient space to reach the dimensionality of the cube.

For example:
If you have a 2-dimensional cube (square) embedded in a 3-dimensional space, the codimension is 3 - 2 = 1.
If you have a 1-dimensional cube (line segment) embedded in a 3-dimensional space, the codimension is 3 - 1 = 2.
If you have a 3-dimensional cube (a regular cube) embedded in a 3-dimensional space, the codimension is 3 - 3 = 0.
In general, the codimension can be calculated as follows:

Codimension = Dimension of Ambient Space −Dimension of Cube

It's a useful concept in mathematics, particularly in areas like topology and geometry, 
where understanding the relationship between objects in different dimensions is important.
"""


from typing import List, Tuple
from cube import Cube
import numpy as np

class DenseCubicalGrids:
    def __init__(self, config):
        self.config = config
        self.threshold = config.threshold
        self.dim = 0
        self.img_x = 0
        self.img_y = 0
        self.img_z = 0
        self.ax = 0
        self.ay = 0
        self.az = 0
        self.axy = 0
        self.axyz = 0
        self.ayz = 0
        self.dense3 = None

    def alloc3d(self, x, y, z):
        d = np.zeros((x, y, z), dtype=np.float64)
        return d

    def gridFromArray(self, arr, embedded, fortran_order,orig_method=False):
        self.img_x, self.img_y, self.img_z = self.ax, self.ay, self.az
        i = 0
        x_shift, y_shift, z_shift = 2, 2, 2
        sgn = 1

        if embedded:
            sgn = -1
            x_shift, y_shift = 4, 4
            if self.az > 1:
                z_shift = 4

        if not orig_method:
            self.dense3 = np.full((self.ax + x_shift, self.ay + y_shift, self.az + z_shift),sgn*self.config.threshold,dtype=np.float64)
            if self.az == 1:
                self.dense3[x_shift//2 : -x_shift//2, y_shift//2:-y_shift//2, z_shift//2] = sgn*arr
            if self.az > 1:
                self.dense3[x_shift//2 : -x_shift//2, y_shift//2:-y_shift//2, z_shift//2:-z_shift//2] = sgn*arr

        if orig_method:
            self.dense3 = self.alloc3d(self.ax + x_shift, self.ay + y_shift, self.az + z_shift)
            arr = arr.flatten()
            if fortran_order:
                for z in range(self.az + z_shift):
                    for y in range(self.ay + y_shift):
                        for x in range(self.ax + x_shift):
                            if x_shift // 2 - 1 < x <= self.ax + x_shift // 2 - 1 and \
                               y_shift // 2 - 1 < y <= self.ay + y_shift // 2 - 1 and \
                               z_shift // 2 - 1 < z <= self.az + z_shift // 2 - 1:
                                self.dense3[x][y][z] = sgn * arr[i]#[z * (self.ax * self.ay) + y * self.ax + x]
                                i += 1
                            elif x == 0 or x == self.ax - 1 + y_shift or \
                                 y == 0 or y == self.ay - 1 + y_shift or \
                                 z == 0 or z == self.az - 1 + z_shift:
                                self.dense3[x][y][z] = self.config.threshold
                            else: # only for embedded; inner boundary
                                self.dense3[x][y][z] = -self.config.threshold
            else:
                for x in range(self.ax + x_shift):
                    for y in range(self.ay + y_shift):
                        for z in range(self.az + z_shift):
                            if x_shift // 2 - 1 < x <= self.ax + x_shift // 2 - 1 and \
                               y_shift // 2 - 1 < y <= self.ay + y_shift // 2 - 1 and \
                               z_shift // 2 - 1 < z <= self.az + z_shift // 2 - 1:
                                self.dense3[x][y][z] = sgn * arr[i]#arr[z * (self.ax * self.ay) + y * self.ax + x]
                                i += 1
                            elif x == 0 or x == self.ax - 1 + y_shift or \
                                 y == 0 or y == self.ay - 1 + y_shift or \
                                 z == 0 or z == self.az - 1 + z_shift:
                                self.dense3[x][y][z] = self.config.threshold
                            else: # only for embedded; inner boundary
                                self.dense3[x][y][z] = -self.config.threshold

        self.ax += x_shift - 2
        self.ay += y_shift - 2
        self.az += z_shift - 2


    '''def gridFromArray(self, arr, embedded, fortran_order):

        #self.img_x = self.ax
        #self.img_y = self.ay
        #self.img_z = self.az
        i = 0
        x_shift = 2
        y_shift = 2
        z_shift = 2
        sgn = 1
        if embedded:
            sgn = -1
            x_shift = 4
            y_shift = 4
            if self.az > 1:
                z_shift = 4
        self.dense3 = self.alloc3d(self.ax + x_shift, self.ay + y_shift, self.az + z_shift)
        if fortran_order:
            for z in range(self.az + z_shift):
                for y in range(self.ay + y_shift):
                    for x in range(self.ax + x_shift):
                        if (x_shift // 2 - 1 < x <= self.ax + x_shift // 2 - 1 and
                            y_shift // 2 - 1 < y <= self.ay + y_shift // 2 - 1 and
                            z_shift // 2 - 1 < z <= self.az + z_shift // 2 - 1):
                            self.dense3[x][y][z] = sgn * arr[np.unravel_index(i, np.shape(arr),order='F')]#arr[i]
                            i += 1
                        elif (x == 0 or x == self.ax - 1 + y_shift or
                              y == 0 or y == self.ay - 1 + y_shift or
                              z == 0 or z == self.az - 1 + z_shift):
                            self.dense3[x][y][z] = self.config.threshold
                        else:
                            self.dense3[x][y][z] = -self.config.threshold  if embedded else self.threshold
        else:
            for x in range(self.ax + x_shift):
                for y in range(self.ay + y_shift):
                    for z in range(self.az + z_shift):
                        if (x_shift // 2 - 1 < x <= self.ax + x_shift // 2 - 1 and
                            y_shift // 2 - 1 < y <= self.ay + y_shift // 2 - 1 and
                            z_shift // 2 - 1 < z <= self.az + z_shift // 2 - 1):
                            self.dense3[x][y][z] = sgn * arr[np.unravel_index(i, np.shape(arr),order='C')]
                            i += 1
                        elif (x == 0 or x == self.ax - 1 + y_shift or
                              y == 0 or y == self.ay - 1 + y_shift or
                              z == 0 or z == self.az - 1 + z_shift):
                            self.dense3[x][y][z] = self.config.threshold
                        else:
                            self.dense3[x][y][z] = -self.config.threshold  if embedded else self.threshold
        self.ax += x_shift - 2
        self.ay += y_shift - 2
        self.az += z_shift - 2'''

    # // return filtration value for a cube
    # // (cx,cy,cz) is the voxel coordinates in the original image
    def get_birth(self, cx: int, cy: int, cz: int) -> float:
        return self.dense3[cx + 1][cy + 1][cz + 1]

    def get_birth_bulk(self, cx, cy, cz):
        return self.dense3[cx + 1,cy + 1,cz + 1]

    def get_birth_with_cm(self, cx: int, cy: int, cz: int, cm: int, dim: int) -> float:

        """
        The get_birth_with_cm function in the DenseCubicalGrids class is responsible for retrieving the birth time of a cube at a given voxel position (x, y, z) with a specified codimension (cm) and dimension (dim). Here's a breakdown of what it does:

        For cubes of dimension 0, it simply retrieves the birth time of the cube at the specified voxel position (x, y, z).
        For cubes of dimension 1, it considers different codimensions (cm) to retrieve the maximum birth time among the cube and its adjacent cubes.
        For cubes of dimension 2, it considers different codimensions (cm) to retrieve the maximum birth time among the cube and its adjacent cubes.
        For cubes of dimension 3, it retrieves the maximum birth time among all the cubes surrounding the specified voxel position.
        """

        # Beware of the shift due to the boundary
        if dim == 0:
            return self.dense3[cx + 1][cy + 1][cz + 1]
        elif dim == 1:
            if cm is None:
                raise ValueError("cm must be provided for dimension 1")
            if cm == 0:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 1][cz + 1])
            elif cm == 1:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 1][cy + 2][cz + 1])
            elif cm == 2:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 1][cy + 1][cz + 2])
            elif cm == 3:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 2][cz + 1])
            elif cm == 4:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 0][cz + 1])
            # for 3d dual only:
            elif cm == 5:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 1][cy + 0][cz + 2])
            elif cm == 6:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 1][cy + 2][cz + 2])
            elif cm == 7:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 0][cz + 2])
            elif cm == 8:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 1][cz + 2])
            elif cm == 9:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 2][cz + 2])
            elif cm == 10:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 0][cz + 0])
            elif cm == 11:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 1][cz + 0])
            elif cm == 12:
                return max(self.dense3[cx + 1][cy + 1][cz + 1], self.dense3[cx + 2][cy + 2][cz + 0])

        elif dim == 2:
            if cm is None:
                raise ValueError("cm must be provided for dimension 2")
            if cm == 0:  # x - y (fix z)
                return max(
                    self.dense3[cx + 1, cy + 1, cz + 1],
                    self.dense3[cx + 2, cy + 1, cz + 1],
                    self.dense3[cx + 2, cy + 2, cz + 1],
                    self.dense3[cx + 1, cy + 2, cz + 1]
                )
            elif cm == 1:  # z - x (fix y)
                return max(
                    self.dense3[cx + 1, cy + 1, cz + 1],
                    self.dense3[cx + 1, cy + 1, cz + 2],
                    self.dense3[cx + 2, cy + 1, cz + 2],
                    self.dense3[cx + 2, cy + 1, cz + 1]
                )
            elif cm == 2:  # y - z (fix x)
                return max(
                    self.dense3[cx + 1, cy + 1, cz + 1],
                    self.dense3[cx + 1, cy + 2, cz + 1],
                    self.dense3[cx + 1, cy + 2, cz + 2],
                    self.dense3[cx + 1, cy + 1, cz + 2]
                )

        elif dim == 3:
            return max(
                self.dense3[cx + 1, cy + 1, cz + 1],
                self.dense3[cx + 2, cy + 1, cz + 1],
                self.dense3[cx + 2, cy + 2, cz + 1],
                self.dense3[cx + 1, cy + 2, cz + 1],
                self.dense3[cx + 1, cy + 1, cz + 2],
                self.dense3[cx + 2, cy + 1, cz + 2],
                self.dense3[cx + 2, cy + 2, cz + 2],
                self.dense3[cx + 1, cy + 2, cz + 2]
            )
        return self.threshold  # dim > 3

    # (x,y,z) of the voxel which defines the birthtime of the cube
    def parent_voxel(self, dim, c):
        """
        The parent_voxel function in the DenseCubicalGrids class is responsible for determining the voxel coordinates of the parent of a given cube. Here's a breakdown of what it does:

        1. Parameters:
        dim: The dimension of the cube.
        c: The cube for which the parent voxel needs to be determined.

        2. Retrieval of Parent Voxel:
        The function checks the birth time of the cube c and compares it with the surrounding cubes to determine its position relative to them.
        Based on the birth time of the cube and its relative position to the surrounding cubes, the function calculates the coordinates of the parent voxel.

        3. Handling Boundary Conditions:
        The function ensures that the cube is not on the boundary of the grid to prevent index out-of-bounds errors.
        It adjusts the voxel coordinates accordingly based on the relative position of the cube.

        4. Returning the Parent Voxel Coordinates:
        Finally, the function returns the coordinates of the parent voxel as a vector [cx, cy, cz].

        In summary, the parent_voxel function determines the coordinates of the parent voxel for a given cube based on its birth time and relative position to surrounding cubes, while also handling boundary conditions to ensure valid voxel coordinates are returned.
        """

        cx = c.x#()
        cy = c.y#()
        cz = c.z#()
        if c.birth == self.dense3[cx + 1][cy + 1][cz + 1]:
            return [cx, cy, cz]
        elif c.birth == self.dense3[cx + 2][cy + 1][cz + 1]:
            return [cx + 1, cy, cz]
        elif c.birth == self.dense3[cx + 2][cy + 2][cz + 1]:
            return [cx + 1, cy + 1, cz]
        elif c.birth == self.dense3[cx + 1][cy + 2][cz + 1]:
            return [cx, cy + 1, cz]
        elif c.birth == self.dense3[cx + 1][cy + 1][cz + 2]:
            return [cx, cy, cz + 1]
        elif c.birth == self.dense3[cx + 2][cy + 1][cz + 2]:
            return [cx + 1, cy, cz + 1]
        elif c.birth == self.dense3[cx + 1][cy + 2][cz + 2]:
            return [cx, cy + 1, cz + 1]
        elif c.birth == self.dense3[cx + 2][cy + 2][cz + 2]:
            return [cx + 1, cy + 1, cz + 1]
        elif c.birth == self.dense3[cx + 2][cy + 0][cz + 1]:  # for 3d dual only
            return [cx + 1, cy - 1, cz]
        elif c.birth == self.dense3[cx + 1][cy + 0][cz + 2]:
            return [cx, cy - 1, cz + 1]
        elif c.birth == self.dense3[cx + 2][cy + 0][cz + 2]:
            return [cx + 1, cy - 1, cz + 1]
        elif c.birth == self.dense3[cx + 2][cy + 0][cz + 0]:
            return [cx + 1, cy - 1, cz - 1]
        elif c.birth == self.dense3[cx + 2][cy + 1][cz + 0]:
            return [cx + 1, cy, cz - 1]
        elif c.birth == self.dense3[cx + 2][cy + 2][cz + 0]:
            return [cx + 1, cy + 1, cz - 1]
        else:
            print("parent voxel not found!")
            return [0, 0, 0]