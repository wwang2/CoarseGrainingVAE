import torch
import numpy as np 
import networkx as nx
import itertools
from data import *
from torch_scatter import scatter_mean, scatter_add
from moleculekit.molecule import Molecule
import glob 
import sys

import mdtraj as md
import mdshare
import pyemma
from sklearn.utils import shuffle
import random



RES2IDX = {'N': 0,
             'H': 1,
             'A': 2,
             'G': 3,
             'R': 4,
             'M': 5,
             'S': 6,
             'I': 7,
             'E': 8,
             'L': 9,
             'Y': 10,
             'D': 11,
             'V': 12,
             'W': 13,
             'Q': 14,
             'K': 15,
             'P': 16,
             'F': 17,
             'C': 18,
             'T': 19}

Z2IDX ={'C': 0,
         'CA': 1,
         'CB': 2,
         'CD': 3,
         'CD1': 4,
         'CD2': 5,
         'CE': 6,
         'CE1': 7,
         'CE2': 8,
         'CE3': 9,
         'CG': 10,
         'CG1': 11,
         'CG2': 12,
         'CH2': 13,
         'CZ': 14,
         'CZ2': 15,
         'CZ3': 16,
         'N': 17,
         'ND1': 18,
         'ND2': 19,
         'NE': 20,
         'NE1': 21,
         'NE2': 22,
         'NH1': 23,
         'NH2': 24,
         'NZ': 25,
         'O': 26,
         'OD1': 27,
         'OD2': 28,
         'OE1': 29,
         'OE2': 30,
         'OG': 31,
         'OG1': 32,
         'OH': 33,
         'SD': 34,
         'SG': 35}


def get_sidechainet_props(data_dict):
    
    '''parse sidechainnet data struct'''

    data = scn.load("debug")

    all_nxyzs = []
    all_cg_nxyz = []
    all_mapping = []
    num_atoms = []
    num_CGs = []

    for i, seq in enumerate(data_dict['seq']):
        cg_type = []
        ca_xyz = []
        mapping = []
        xyzs = []
        atom_zs = []

        xyz = data_dict['crd'][i].reshape(-1, 14, 3)

        for j, res in enumerate(seq): 
            if data_dict['msk'][i][j] == "+":
                cg_type.append(RES2IDX[res])


        for j, res_xyz in enumerate(xyz):

            zmap = ATOM_MAP_14[seq[j]]

            if data_dict['msk'][i][j] == "+":
                ca_xyz.append(res_xyz[1]) # the 2nd atom is the alpha carbon 
                for k, xyz in enumerate(res_xyz):
                    if np.power(xyz, 2).sum() != 0:
                        # also need to retrieve 
                        xyzs.append(xyz)
                        mapping.append(j)
                        atom_zs.append(Z2IDX[zmap[k]])
                        
        

        xyzs = np.vstack(xyzs)
        
        ca_xyz = np.array(ca_xyz)
        atomszs = np.array(atom_zs)
        cg_type = np.array(cg_type)
        mapping = np.array(mapping)

        num_atoms.append(xyzs.shape[0])
        num_CGs.append(ca_xyz.shape[0])
        
        nxyz = np.hstack([np.array(atomszs).reshape(-1, 1), np.array(xyzs)])
        cg_nxyz = np.hstack([np.array(cg_type).reshape(-1, 1), np.array(ca_xyz)])

        all_nxyzs.append(torch.Tensor(nxyz))
        all_cg_nxyz.append(torch.Tensor(cg_nxyz))
        all_mapping.append(torch.LongTensor(mapping))

        
    return {'nxyz': all_nxyzs, 'CG_nxyz': all_cg_nxyz,
            'CG_mapping': all_mapping, 'num_atoms': num_atoms,
            'num_CGs': num_CGs}
