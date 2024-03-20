import sys
import os
import argparse
import numpy as np
from datetime import datetime
from perch.py_cripser.cube import Cube
from perch.py_cripser.dense_cubical_grids import DenseCubicalGrids
from perch.py_cripser.write_pairs import WritePairs
from perch.py_cripser.joint_pairs import JointPairs
from perch.py_cripser.compute_pairs import ComputePairs
from perch.py_cripser.config import Config


def print_usage_and_exit(exit_code):
    print("""Usage: cubicalripser [options] [input_filename]

Options:

  --help, -h          print this screen
  --verbose, -v       
  --threshold <t>, -t compute cubical complexes up to birth time <t>
  --maxdim <t>, -m    compute persistent homology up to dimension <t>
  --algorithm, -a     algorithm to compute the 0-dim persistent homology:
                        link_find      (default)
                        compute_pairs  (slow in most cases)
  --min_recursion_to_cache, -mc  minimum number of recursion for a reduced column to be cached (the higher the slower but less memory)
  --cache_size, -c    maximum number of reduced columns to be cached (the lower the slower but less memory)
  --output, -o        name of the output file
  --print, -p         print persistence pairs on console
  --top_dim           (not recommended) compute only for top dimension using Alexander duality (setting '--maxdim 0 --embedded' is generally faster for this purpose)
  --embedded, -e      Take the Alexander dual (pad the image boundary with -infty and negate the pixel values)
  --location, -l      whether creator/destroyer location is included in the output:
                        yes      (default)
                        none
""")
    sys.exit(exit_code)


def main():
    config = Config()
    arg_embedded = False

    # command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--threshold", "-t", type=float, help="compute cubical complexes up to birth time <t>")
    parser.add_argument("--maxdim", "-m", type=int, help="compute persistent homology up to dimension <t>")
    parser.add_argument("--algorithm", "-a", choices=['link_find', 'compute_pairs'],
                        help="algorithm to compute the 0-dim persistent homology")
    parser.add_argument("--min_recursion_to_cache", "-mc", type=int,
                        help="minimum number of recursion for a reduced column to be cached (the higher the slower but less memory)")
    parser.add_argument("--cache_size", "-c", type=int,
                        help="maximum number of reduced columns to be cached (the lower the slower but less memory)")
    parser.add_argument("--output", "-o", help="name of the output file")
    parser.add_argument("--print", "-p", action="store_true", help="print persistence pairs on console")
    parser.add_argument("--top_dim", action="store_true",
                        help="(not recommended) compute only for top dimension using Alexander duality (setting '--maxdim 0 --embedded' is generally faster for this purpose)")
    parser.add_argument("--embedded", "-e", action="store_true",
                        help="Take the Alexander dual (pad the image boundary with -infty and negate the pixel values)")
    parser.add_argument("--location", "-l", choices=['yes', 'none'],
                        help="whether creator/destroyer location is included in the output")
    parser.add_argument("input_filename", nargs='?', help="input filename")

    args = parser.parse_args()

    config.verbose = args.verbose
    config.threshold = args.threshold
    config.maxdim = args.maxdim
    config.method = args.algorithm
    config.min_recursion_to_cache = args.min_recursion_to_cache
    config.cache_size = args.cache_size
    config.output_filename = args.output
    config.print = args.print
    config.filename = args.input_filename
    if args.top_dim:
        config.method = "alexander"
    if args.embedded:
        arg_embedded = True
    if args.location:
        config.location = args.location.upper()

    if not config.filename:
        print_usage_and_exit(-1)

    if config.method == "alexander":
        config.embedded = not arg_embedded
    else:
        config.embedded = arg_embedded

    if not os.path.exists(config.filename):
        print("Couldn't open file", config.filename)
        sys.exit(-1)

    # infer input file type from its extension
    if config.filename.endswith(".txt"):
        config.format = "PERSEUS"
    elif config.filename.endswith(".npy"):
        config.format = "NUMPY"
    elif config.filename.endswith(".csv"):
        config.format = "CSV"
    elif config.filename.endswith(".complex"):
        config.format = "DIPHA"
    else:
        print("Unknown input file format! (the filename extension should be one of npy, txt, complex):",
              config.filename)
        sys.exit(-1)

    writepairs = []
    ctr = []

    dcg = DenseCubicalGrids(config)

    # compute PH
    betti = []
    if config.method == "link_find":
        dcg.loadImage(config.embedded)
        config.maxdim = min(config.maxdim, dcg.dim - 1)
        start0 = datetime.now()
        jp = JointPairs(dcg, writepairs, config)
        if dcg.dim == 1:
            jp.enum_edges([0], ctr)
        elif dcg.dim == 2:
            jp.enum_edges([0, 1], ctr)
        else:
            jp.enum_edges([0, 1, 2], ctr)
        jp.joint_pairs_main(ctr, 0)
        msec = (datetime.now() - start0).total_seconds() * 1000
        betti.append(len(writepairs))
        print("the number of pairs in dim 0:", betti[0])
        if config.verbose:
            print("computation took", msec, "[msec]")
        if config.maxdim > 0:
            start1 = datetime.now()
            cp = ComputePairs(dcg, writepairs, config)
            cp.assemble_columns_to_reduce(ctr, 2)
            cp.compute_pairs_main(ctr)
            msec1 = (datetime.now() - start1).total_seconds() * 1000
            betti.append(len(writepairs) - betti[0])
            print("the number of pairs in dim 1:", betti[1])
            if config.verbose:
                print("computation took", msec1, "[msec]")
            if config.maxdim > 1:
                start2 = datetime.now()
                cp.assemble_columns_to_reduce(ctr, 2)
                cp.compute_pairs_main(ctr)
                msec2 = (datetime.now() - start2).total_seconds() * 1000
                betti.append(len(writepairs) - betti[0] - betti[1])
                print("the number of pairs in dim 2:", betti[2])
                if config.verbose:
                    print("computation took", msec2, "[msec]")
        mseca = (datetime.now() - start0).total_seconds() * 1000
        print("the whole computation took", mseca, "[msec]")
    elif config.method == "compute_pairs":
        dcg.loadImage(config.embedded)
        config.maxdim = min(config.maxdim, dcg.dim - 1)
        cp = ComputePairs(dcg, writepairs, config)
        cp.assemble_columns_to_reduce(ctr, 0)
        cp.compute_pairs_main(ctr)
        betti.append(len(writepairs))
        print("the number of pairs in dim 0:", betti[0])
        if config.maxdim > 0:
            cp.assemble_columns_to_reduce(ctr, 1)
            cp.compute_pairs_main(ctr)
            betti.append(len(writepairs) - betti[0])
            print("the number of pairs in dim 1:", betti[1])
            if config.maxdim > 1:
                cp.assemble_columns_to_reduce(ctr, 2)
                cp.compute_pairs_main(ctr)
                betti.append(len(writepairs) - betti[0] - betti[1])
                print("the number of pairs in dim 2:", betti[2])
    elif config.method == "alexander":
        if config.tconstruction:
            print("Alexander duality for T-construction is not implemented yet.")
            sys.exit(-9)
        dcg.loadImage(config.embedded)
        start0 = datetime.now()
        jp = JointPairs(dcg, writepairs, config)
        if dcg.dim == 1:
            jp.enum_edges([0], ctr)
            jp.joint_pairs_main(ctr, 0)
            print("the number of pairs in dim 0:", len(writepairs))
        elif dcg.dim == 2:
            jp.enum_edges([0, 1, 3, 4], ctr)
            jp.joint_pairs_main(ctr, 1)
            print("the number of pairs in dim 1:", len(writepairs))
        elif dcg.dim == 3:
            jp.enum_edges([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ctr)
            jp.joint_pairs_main(ctr, 2)
            print("the number of pairs in dim 2:", len(writepairs))
        mseca = (datetime.now() - start0).total_seconds() * 1000
        print("computation took", mseca, "[msec]")

    # determine shift between dcg and the voxel coordinates
    pad_x = (dcg.ax - dcg.img_x) / 2
    pad_y = (dcg.ay - dcg.img_y) / 2
    pad_z = (dcg.az - dcg.img_z) / 2

    # write to file
    p = len(writepairs)
    print("the number of total pairs:", p)
    if config.output_filename.endswith(".csv"):
        with open(config.output_filename, 'w') as writing_file:
            for i in range(p):
                d = writepairs[i].dim
                writing_file.write(f"{d},{writepairs[i].birth},{writepairs[i].death}")
                if config.location != "NONE":
                    writing_file.write(
                        f",{writepairs[i].birth_x - pad_x},{writepairs[i].birth_y - pad_y},{writepairs[i].birth_z - pad_z}")
                    writing_file.write(
                        f",{writepairs[i].death_x - pad_x},{writepairs[i].death_y - pad_y},{writepairs[i].death_z - pad_z}")
                writing_file.write("\n")
    elif config.output_filename.endswith(".npy"):  # output in npy
        leshape = [p, 9]
        data = np.zeros((p, 9))
        for i in range(p):
            data[i][0] = writepairs[i].dim
            data[i][1] = writepairs[i].birth
            data[i][2] = writepairs[i].death
            data[i][3] = writepairs[i].birth_x - pad_x
            data[i][4] = writepairs[i].birth_y - pad_y
            data[i][5] = writepairs[i].birth_z - pad_z
            data[i][6] = writepairs[i].death_x - pad_x
            data[i][7] = writepairs[i].death_y - pad_y
            data[i][8] = writepairs[i].death_z - pad_z
        try:
            np.save(config.output_filename, data)
        except Exception as e:
            print("error:", e)
    elif config.output_filename == "none":  # no output
        sys.exit(0)
    else:  # output in DIPHA format
        with open(config.output_filename, 'wb') as writing_file:
            mn = 8067171840
            writing_file.write(mn.to_bytes(8, byteorder='little'))  # magic number
            type_ = 2
            writing_file.write(type_.to_bytes(8, byteorder='little'))  # type number of PERSISTENCE_DIAGRAM
            writing_file.write(p.to_bytes(8, byteorder='little'))  # number of points in the diagram p
            for i in range(p):
                writedim = writepairs[i].dim
                writing_file.write(writedim.to_bytes(8, byteorder='little'))
                writing_file.write(writepairs[i].birth.to_bytes(8, byteorder='little', signed=True))
                writing_file.write(writepairs[i].death.to_bytes(8, byteorder='little', signed=True))


if __name__ == "__main__":
    main()
