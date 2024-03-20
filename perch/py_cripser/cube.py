
NONE = 0xffffffffffffffff


def cube_comp(o1, o2):
    if o1.birth == o2.birth:
        return -1 if o1.index < o2.index else 1
    else:
        return 1 if o1.birth > o2.birth else -1

def get_cube_comparator():
    return CubeComparator()

class Cube:

    def __init__(self, _b=0, _x=0, _y=0, _z=0, _m=0, _index=NONE):
        self.birth = _b
        self.index = _index if _index != NONE else (_x | (_y << 20) | (_z << 40) | (_m << 60))

    def copyCube(self, other):
        self.birth = other.birth
        self.index = other.index

    def print_cube(self):
        print(f"{self.birth},{self.index},{self.x},{self.y},{self.z},{self.m}")

    @property
    def x(self):
        return self.index & 0xfffff
    @property
    def y(self):
        return (self.index >> 20) & 0xfffff
    @property
    def z(self):
        return (self.index >> 40) & 0xfffff
    @property
    def m(self):
        return (self.index >> 60) & 0xf

    def __eq__(self, other):
        return self.index == other.index# and self.birth == other.birth

    def __lt__(self, other):
        if self.birth == other.birth:
            return self.index < other.index # if births are equal, c1 is less than c2 if ind1 < ind2
        else:
            return self.birth > other.birth # c1 is less than c2 if birth1 is > birth2

    '''def __le__(self, other):
        return self == other or self < other

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)'''



class CubeComparator:
    def __init__(self):
        pass

    def __call__(self, o1, o2):
        if o1.birth == o2.birth:
            return o1.index < o2.index
        else:
            return o1.birth > o2.birth
