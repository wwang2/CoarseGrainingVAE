import torch
import numpy as np 
import networkx as nx
import itertools
from data import *
from sampling import get_bond_graphs
from torch_scatter import scatter_mean, scatter_add
from moleculekit.molecule import Molecule
import glob 
import sys

import mdtraj as md
import mdshare
import pyemma
from sklearn.utils import shuffle
import random

from sidechainnet.structure.PdbBuilder import PdbBuilder, ATOM_MAP_14

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

ATOM2IDX ={'C': 0,
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

ATOM2Z = {'C': 6,
         'CA': 6,
         'CB': 6,
         'CD': 6,
         'CD1': 6,
         'CD2': 6,
         'CE': 6,
         'CE1': 6,
         'CE2': 6,
         'CE3': 6,
         'CG': 6,
         'CG1': 6,
         'CG2': 6,
         'CH2': 6,
         'CZ': 6,
         'CZ2': 6,
         'CZ3': 6,
         'N': 7,
         'ND1': 7,
         'ND2': 7,
         'NE': 7,
         'NE1': 7,
         'NE2': 7,
         'NH1': 7,
         'NH2': 7,
         'NZ': 7,
         'O': 8,
         'OD1': 8,
         'OD2': 8,
         'OE1': 8,
         'OE2': 8,
         'OG': 8,
         'OG1': 8,
         'OH': 8,
         'SD': 16,
         'SG': 16}


IDX2ATOM = {v: k for k, v in ATOM2IDX.items()}

def idx2z(idxs):
    types = [IDX2ATOM[atom] for atom in idxs]
    z = [ATOM2Z[atom] for atom in types]  
    return z 

def mask_seq(seq, msk):
    return "".join( [seq[i] if el == '+' else "" for (i, el) in enumerate(msk) ] )

def save_pdb(msk, seq, xyz, fn='./junk.pdb'):
    msk_seq = mask_seq(seq, msk)
    pdb = PdbBuilder(msk_seq, pad_crd.reshape(-1, 3).detach().numpy())
    pdb.save_pdb(fn)

def get_sidechainet_props(data_dict):
    
    '''parse sidechainnet data struct'''

    all_nxyzs = []
    all_cg_nxyz = []
    all_mapping = []
    num_atoms = []
    num_CGs = []
    bond_edges_list = []

    for i, seq in enumerate(data_dict['seq']):
        cg_type = []
        ca_xyz = []
        mapping = []
        xyzs = []
        atom_type = []
        atom_num = []

        xyz = data_dict['crd'][i].reshape(-1, 14, 3)

        for j, res in enumerate(seq): 
            if data_dict['msk'][i][j] == "+":
                cg_type.append(RES2IDX[res])

        map = 0
        for j, res_xyz in enumerate(xyz):
            zmap = ATOM_MAP_14[seq[j]]
            if data_dict['msk'][i][j] == "+":
                ca_xyz.append(res_xyz[1]) # the 2nd atom is the alpha carbon 
                for k, xyz in enumerate(res_xyz):
                    if np.power(xyz, 2).sum() != 0:
                        # also need to retrieve 
                        xyzs.append(xyz)
                        mapping.append(map)
                        atom_type.append(ATOM2IDX[zmap[k]])
                        atom_num.append(ATOM2Z[zmap[k]])
                        
                map += 1 

        xyzs = np.vstack(xyzs)
        
        # compute bond_graphs 
        
        atoms = Atoms(positions=xyzs, numbers=atom_num)        
        edges = get_bond_graphs(atoms).nonzero()

        ca_xyz = np.array(ca_xyz)
        atom_type = np.array(atom_type)
        cg_type = np.array(cg_type)
        mapping = np.array(mapping)

        num_atoms.append(torch.LongTensor([xyzs.shape[0]]))
        num_CGs.append(torch.LongTensor([ca_xyz.shape[0]]))
        
        nxyz = np.hstack([np.array(atom_type).reshape(-1, 1), np.array(xyzs)])
        cg_nxyz = np.hstack([np.array(cg_type).reshape(-1, 1), np.array(ca_xyz)])

        all_nxyzs.append(torch.Tensor(nxyz))
        all_cg_nxyz.append(torch.Tensor(cg_nxyz))
        all_mapping.append(torch.LongTensor(mapping))
        bond_edges_list.append(torch.LongTensor(edges))
        # compute bond graphs 
        

        
    return {'nxyz': all_nxyzs, 'CG_nxyz': all_cg_nxyz,
            'CG_mapping': all_mapping, 'num_atoms': num_atoms,
            'num_CGs': num_CGs, 'bond_edge_list': bond_edges_list, 
            'seq': data_dict['seq']}
