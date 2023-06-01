import copy
import numpy as np


def compute_affinity(central, side, position, rotation, crop=None):
    central = copy.deepcopy(central)
    side = copy.deepcopy(side)
    side.rotate(k=rotation)
    
    if crop is not None:
        Pi = central.data[crop:-crop, crop:-crop, :].copy() / float(256)
        Pj = side.data[crop:-crop, crop:-crop, :].copy() / float(256)
    else:
        Pi = central.data.copy() / float(256)
        Pj = side.data.copy() / float(256)

    if position == 'top':
        Gi = abs(Pi[0, :, :] - Pi[1, :, :])
        Gj = abs(Pj[-1, :, :] - Pj[-2, :, :])
        Gij = abs(Pj[-1, :, :] - Pi[0, :, :])
        
    elif position == 'right':
        Gi = abs(Pi[:, -1, :] - Pi[:, -2, :])
        Gj = abs(Pj[:, 0, :] - Pj[:, 1, :])
        Gij = abs(Pj[:, 0, :] - Pi[:, -1, :])

    elif position == 'bottom':
        Gi = abs(Pi[-1, :, :] - Pi[-2, :, :])
        Gj = abs(Pj[0, :, :] - Pj[1, :, :])
        Gij = abs(Pj[0, :, :] - Pi[-1, :, :])

    elif position == 'left':
        Gi = abs(Pi[:, 0, :] - Pi[:, 1, :])
        Gj = abs(Pj[:, -1, :] - Pj[:, -2, :])
        Gij = abs(Pj[:, -1, :] - Pi[:, 0, :])

    mu_i = np.mean(Gi)
    mu_j = np.mean(Gj)

    Gi_reg = abs(Gij - mu_i)
    Gj_reg = abs(Gij - mu_j)

    cov_i = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    cov_j = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    p1_cov_inv = np.linalg.inv(cov_i)
    p2_cov_inv = np.linalg.inv(cov_j)

    Dij = np.sqrt(np.sum(np.dot(np.dot(Gi_reg, p1_cov_inv), np.transpose(Gi_reg))))
    Dji = np.sqrt(np.sum(np.dot(np.dot(Gj_reg, p2_cov_inv), np.transpose(Gj_reg))))

    return Dij + Dji
