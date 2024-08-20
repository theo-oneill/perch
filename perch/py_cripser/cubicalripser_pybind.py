import numpy as np
from perch.py_cripser.cube import Cube
from perch.py_cripser.write_pairs import WritePairs
from perch.py_cripser.joint_pairs import JointPairs
from perch.py_cripser.compute_pairs import ComputePairs
from perch.py_cripser.config import Config
from perch.py_cripser.dense_cubical_grids import DenseCubicalGrids


def compute_ph(img, maxdim=0, top_dim=False, embedded=False, fortran_order=False,return_seg=False):
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
    if return_seg:
        return result, seg_map
    if not return_seg:
        return result#, seg_map

