
from perch.py_cripser.dense_cubical_grids import DenseCubicalGrids


class WritePairs:
    def __init__(self, dim: int, birth=None, death=None, birth_x=None, birth_y=None, birth_z=None,
                 death_x=None, death_y=None, death_z=None, birthC=None, deathC=None, dcg: DenseCubicalGrids = None,
                 print_flag: bool = False):
        self.dim = dim
        if birth is not None and death is not None and birthC is None and deathC is None and dcg is None:
            self.birth = birth
            self.death = death
            self.birth_x, self.birth_y, self.birth_z = birth_x, birth_y, birth_z
            self.death_x, self.death_y, self.death_z = death_x, death_y, death_z
        elif birthC is not None and deathC is not None and dcg is not None:
            self.birth = birthC.birth
            self.death = deathC.birth
            b = dcg.parent_voxel(dim, birthC)
            d = dcg.parent_voxel(dim, deathC)
            self.birth_x, self.birth_y, self.birth_z = b[0], b[1], b[2]
            self.death_x, self.death_y, self.death_z = d[0], d[1], d[2]
        else:
            raise ValueError("Invalid combination of arguments")

        if print_flag:
            print(
                f"[{self.birth},{self.death}) birth loc. ({self.birth_x},{self.birth_y},{self.birth_z}), death loc. ({self.death_x},{self.death_y},{self.death_z})")
