import torch
import numpy as np 
import networkx as nx
import itertools
from .data import *
from .datasets import get_high_order_edge
from torch_scatter import scatter_mean, scatter_add
from moleculekit.molecule import Molecule
import glob 
import sys

import mdtraj as md
import mdshare
import pyemma
from sklearn.utils import shuffle
import random
import tqdm 
import os 
import pickle

from sidechainnet.structure.PdbBuilder import PdbBuilder, ATOM_MAP_14


THREE_LETTER_TO_ONE = {
    "ARG": "R", 
    "HIS": "H", 
    "LYS": "K", 
    "ASP": "D", 
    "GLU": "E", 
    "SER": "S", 
    "THR": "T", 
    "ASN": "N", 
    "GLN": "Q", 
    "CYS": "C", 
    "GLY": "G", 
    "PRO": "P", 
    "ALA": "A", 
    "VAL": "V", 
    "ILE": "I", 
    "LEU": "L", 
    "MET": "M", 
    "PHE": "F", 
    "TYR": "Y", 
    "TRP": "W"
}

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

SEQ_BLACKLIST = ['MPEFLEDPSVLTKDKLKSELVANNVTLPAGEQRKDVYVQLYLQHLTARNRPPLPAGTNSKGPPDFSSDEEREPTPVLGSGAAAAGRSRAAVGRKATKKTDKPRQEDKDDLDVTELTNEDLLDQLVKYGVNPGPIVGTTRKLYEKKLLKLREQGTESRSSTPLPTISSS',
                'MDVKPDRVIDARGSYCPGPLMELIKAYKQAKVGEVISVYSTDAGTKKDAPAWIQKSGQELVGVFDRNGYYEIVMKKVK']

IDX2ATOM = {v: k for k, v in ATOM2IDX.items()}


def get_bond_graphs(atoms, device='cpu', scale=1.3):
    dist = compute_distance_mat(atoms, device=device)
    cutoff = compute_bond_cutoff(atoms, scale=scale)
    bond_mat = (dist < cutoff.to(device))
    bond_mat[np.diag_indices(len(atoms))] = 0
    
    del dist, cutoff

    return bond_mat.to(torch.long).to('cpu')
    
def idx2z(idxs):
    types = [IDX2ATOM[atom] for atom in idxs]
    z = [ATOM2Z[atom] for atom in types]  
    return z 

def mask_seq(seq, msk):
    return "".join( [seq[i] if el == '+' else "" for (i, el) in enumerate(msk) ] )

def save_pdb(msk, seq, pad_crd, fn='./junk.pdb'):
    msk_seq = mask_seq(seq, msk)
    pdb = PdbBuilder(msk_seq, pad_crd.reshape(-1, 3).detach().numpy())
    pdb.save_pdb(fn)
    

def CG2ChannelIdx(CG_mapping):
    CG2atomChannel = torch.zeros_like(CG_mapping).to("cpu")
    for cg_type in torch.unique(CG_mapping): 
        cg_filter = CG_mapping == cg_type
        num_contri_atoms = cg_filter.sum().item()
        CG2atomChannel[cg_filter] = torch.LongTensor(list(range(num_contri_atoms)))#.to(CG_mapping.device)
        
    return CG2atomChannel.detach()

def dense2pad_crd(xyz, n_res, mapping):
    pad_crd = torch.zeros(n_res, 14, 3)
    channel_idx = CG2ChannelIdx(mapping).cpu()
    pad_crd[mapping,  channel_idx, :] = xyz.cpu()
    return pad_crd

def get_sidechainet_props(data_dict, params, n_data=10000, split='train', thinning=30):
    
    '''parse sidechainnet data struct'''

    all_nxyzs = []
    all_cg_nxyz = []
    all_mapping = []
    num_atoms = []
    num_CGs = []
    bond_edges_list = []

    all_seqs = []
    all_msks = []
    all_ids = []


    # generate random index and take first n 
    idx = list(range(len(data_dict['seq'])))
    random.shuffle(idx)

    graph_data_path = os.path.join("../data/",  params['dataset'] + '_{}_{}graph.pkl'.format(split, thinning))

    compute_graph = False 
    if os.path.exists(graph_data_path): 
        print("loading graph dictionary from {}".format(graph_data_path))
        graph_data = pickle.load( open(graph_data_path, "rb" ) )
    else:
        compute_graph = True 
        graph_data = {}
        print("computing graph on the fly ")

    compute_graph= True 

    for i in tqdm.tqdm(idx[:n_data]):
        seq = data_dict['seq'][i]
        msk = data_dict['msk'][i]
        crd = data_dict['crd'][i].reshape(-1, 14, 3) 
        id = str(data_dict['ids'][i])

        seq_len = len(seq)
        msk_ratio = len( [1 for pos in msk if pos == '+']) / seq_len

        cg_type = []
        ca_xyzs = []
        mapping = []
        xyzs = []
        atom_type = []
        atom_num = []

        msk_seq = mask_seq(seq, msk)

        # filter_idx = []
        # ca_filter_idx = []
        # for (id, res) in enumerate(seq):
        #     if msk[id] == '+':
        #         ca_filter_idx.append(id)
        #         for atom_id in range(len(ATOM_MAP_14[res])):
        #             if ATOM_MAP_14[res][atom_id] != 'PAD':
        #                 filter_idx.append([id, atom_id])
                            
        # for id, res in enumerate(msk_seq):
        #     zmap = ATOM_MAP_14[res]
        #     # generate types 
        #     atom_type += [ATOM2IDX[atom] for atom in zmap if atom != 'PAD' ]
        #     atom_num += [ATOM2Z[atom] for atom in zmap if atom != 'PAD' ]
        #     mapping += [id for atom in zmap if atom != 'PAD' ]
        #     cg_type += [RES2IDX[res]]

        # atom_type = torch.LongTensor(atom_type)
        # atom_num = torch.LongTensor(atom_num)
        # mapping = torch.LongTensor(mapping)
        # cg_type = torch.LongTensor(cg_type)

        # filter_idx = torch.LongTensor(filter_idx)
        # ca_filter_idx = torch.LongTensor(ca_filter_idx)
        # xyzs = torch.Tensor( crd.reshape(-1, 14 ,3))[filter_idx[:,0 ], filter_idx[:,1]]
        # ca_xyzs = torch.Tensor( crd.reshape(-1, 14 ,3))[ca_filter_idx, 1] 

        if seq not in SEQ_BLACKLIST:

            for j, res in enumerate(seq): 
                if data_dict['msk'][i][j] == "+":
                    cg_type.append(RES2IDX[res])

            all_ids.append(id)

            msk_seq = mask_seq(seq, msk)
            map = 0
            for j, res_xyz in enumerate(crd):
                zmap = ATOM_MAP_14[seq[j]]
                if data_dict['msk'][i][j] == "+":
                    ca_xyzs.append(res_xyz[1]) # the 2nd atom is the alpha carbon 
                    for k, xyz in enumerate(res_xyz):
                        if xyz.sum() != 0:
                            # also need to retrieve 
                            xyzs.append(xyz)
                            mapping.append(map)
                            atom_type.append(ATOM2IDX[zmap[k]])
                            atom_num.append(ATOM2Z[zmap[k]]) 
                    map += 1 

            xyzs = np.vstack(xyzs)
            
            # compute bond_graphs 
            
            if compute_graph:
                atoms = Atoms(positions=xyzs, numbers=atom_num)        
                edges = get_bond_graphs(atoms, device="cpu").nonzero()
                graph_data[i] = edges
            else: 
                edges = graph_data[i]

            if params['edgeorder'] > 1: 
                edges = get_high_order_edge(edges, params['edgeorder'], xyzs.shape[0])

            ca_xyzs = np.array(ca_xyzs)
            atom_type = np.array(atom_type)
            cg_type = np.array(cg_type)
            mapping = np.array(mapping)

            num_atoms.append(torch.LongTensor([xyzs.shape[0]]))
            num_CGs.append(torch.LongTensor([ca_xyzs.shape[0]]))
            
            nxyz = np.hstack([np.array(atom_num).reshape(-1, 1), np.array(xyzs)])
            cg_nxyz = np.hstack([np.array(cg_type).reshape(-1, 1), np.array(ca_xyzs)])

            all_seqs.append(data_dict['seq'][i])
            all_msks.append(data_dict['msk'][i])
            all_nxyzs.append(torch.Tensor(nxyz))
            all_cg_nxyz.append(torch.Tensor(cg_nxyz))
            all_mapping.append(torch.LongTensor(mapping))
            bond_edges_list.append(torch.LongTensor(edges))
            # compute bond graphs 

            #import ipdb; ipdb.set_trace()
        
    if compute_graph:
        pickle.dump( graph_data, open( graph_data_path, "wb" ) )
        
    return {'nxyz': all_nxyzs, 'CG_nxyz': all_cg_nxyz,
            'CG_mapping': all_mapping, 'num_atoms': num_atoms,
            'num_CGs': num_CGs, 'bond_edge_list': bond_edges_list, 
            'seq': all_seqs, 'msk':all_msks, 'id': all_ids}




def get_CASP14_targets():
    
    all_nxyz = []
    all_CG_nxyz = []
    all_mapping = [] 
    all_seq = []
    all_msk = []
    all_id = []
    all_edges = []
    num_atoms = []
    num_CGs = []

    for file in glob.glob("../data/casp14.targets.T.public_11.29.2020/*.pdb"):
        pdb = md.load_pdb(file)

        z = []
        mapping = []
        cg_type = []
        seq = ''
        msk = ''
        id = file.split("/")[-1].split(".pdb")[0]

        # collect CA positions 
        ca_index = pdb.top.select("name CA")

        ca_xyz = pdb.xyz[0, ca_index] * 10.0 
        xyz = pdb.xyz[0] * 10.0 

        # also need to collect mapping 
        pdb.atom_slice(ca_index).save_pdb("ca_test.pdb")

        for i, res in enumerate(pdb.top.residues):
            res_letter = THREE_LETTER_TO_ONE[ res.name ]
            cg_type.append( RES2IDX[res_letter] )
            seq += res_letter
            msk += '+'

            for atom in res.atoms:
                z.append(atom.element.atomic_number)
                mapping.append(i)

        CG_nxyz = torch.Tensor(np.hstack((np.array(cg_type)[:, None], ca_xyz )))
        nxyz = torch.Tensor(np.hstack((np.array(z)[:, None], xyz )))
        mapping = torch.LongTensor(mapping)

        atoms = Atoms(positions=xyz, numbers=z)        
        edges = get_bond_graphs(atoms, device="cpu").nonzero()

        all_edges.append(edges)
        all_nxyz.append(nxyz)
        all_CG_nxyz.append(CG_nxyz)
        all_mapping.append(mapping)
        all_seq.append(seq)
        all_msk.append(msk)
        all_id.append(id)
        num_atoms.append(torch.LongTensor([nxyz.shape[0]]))
        num_CGs.append(torch.LongTensor([CG_nxyz.shape[0]]))

    props = {'nxyz': all_nxyz, 'CG_nxyz': all_CG_nxyz, 'CG_mapping': all_mapping,
             'seq': all_seq, "msk": all_msk, 'id': all_id, 'bond_edge_list': all_edges,
             'num_CGs': num_CGs, 'num_atoms': num_atoms
              }
    return props 