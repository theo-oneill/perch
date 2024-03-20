from enum import Enum
import sys

class CalculationMethod(Enum):
    LINKFIND = 1
    COMPUTEPAIRS = 2
    ALEXANDER = 3

class OutputLocation(Enum):
    LOC_NONE = 0
    LOC_YES = 1

class FileFormat(Enum):
    DIPHA = 0
    PERSEUS = 1
    NUMPY = 2
    CSV = 3

class Config:
    def __init__(self):
        self.filename = ""
        self.output_filename = "output.csv"  # default output filename
        self.format = FileFormat.CSV
        self.method = CalculationMethod.LINKFIND
        self.threshold = sys.float_info.max
        self.maxdim = 2  # compute PH for these dimensions
        self.print_pairs = False  # flag for printing persistence pairs to stdout
        self.print = False
        self.verbose = False
        self.tconstruction = False  # T-construction or V-construction
        self.embedded = False  # embed image in the sphere (for alexander duality)
        self.location = OutputLocation.LOC_YES  # flag for saving location
        self.min_recursion_to_cache = 0  # num of minimum recursions for a reduced column to be cached
        self.cache_size = 1 << 31  # the maximum number of reduced columns to be cached
