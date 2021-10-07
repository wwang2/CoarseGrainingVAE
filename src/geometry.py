import torch
import numpy as np


def compute_dihedral_vec(dihedrals, xyz):


    # this is wrong, need to redo with padding
    r12 = xyz[dihedrals[:, 0], dihedrals[:, 1], : ] - xyz[:, dihedrals[:, 2], : ]
    r23 = xyz[dihedrals[:, 0], dihedrals[:, 2], : ] - xyz[:, dihedrals[:, 3], : ]
    r34 = xyz[dihedrals[:, 0], dihedrals[:, 3], : ] - xyz[:, dihedrals[:, 4], : ]


    # A = torch.cross(r12, r23, dim=-1)
    # B = torch.cross(r23, r34, dim=-1)
    # C = torch.cross(r34, A, dim=-1)

    # rA = 1 / A.pow(2).sum(-1).sqrt()
    # rB = 1 / B.pow(2).sum(-1).sqrt()
    # rC = 1 / C.pow(2).sum(-1).sqrt()

    # A = A * rA.unsqueeze(-1)
    # B = B * rB.unsqueeze(-1)
    # C = C * rC.unsqueeze(-1)

    return r12, r23, r34 