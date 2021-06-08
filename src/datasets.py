import torch
import numpy as np 
import networkx as nx
import itertools
from data import *
from model import * 

import mdtraj as md
import mdshare
import pyemma
from sklearn.utils import shuffle

atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7}


def compute_nbr_list(frame, cutoff):
    
    dist = (frame[None, ...] - frame[:, None, :]).pow(2).sum(-1).sqrt()
    nbr_list = torch.nonzero(dist < cutoff).numpy()
    
    return nbr_list

def parition2mapping(partitions, n_nodes):
    # generate mapping 
    mapping = np.zeros(n_nodes)
    
    for k, group in enumerate(partitions):
        for node in group:
            mapping[node] = k
            
    return mapping.astype(int)

def get_partition(nbr_list, n_partitions):
    
    adj = [tuple(pair) for pair in nbr_list]
    G = nx.Graph()
    G.add_edges_from(adj)

    comp = nx.community.girvan_newman(G)
    for communities in itertools.islice(comp, n_partitions-1):
        partitions = tuple(sorted(c) for c in communities)
        
    return partitions 

def get_mapping(traj, cutoff, n_nodes, n_partitions):

    mappings = []
    for frame in traj:
        nbr_list = compute_nbr_list(torch.Tensor(frame), cutoff)
        paritions = get_partition(nbr_list, n_partitions)     
        mapping = parition2mapping(paritions, n_nodes)
        mappings.append(mapping)

    mappings = torch.Tensor( np.stack(mappings) )
    mappings = mappings.mode(0)[0]
    
    return mappings.long()

def get_alanine_dipeptide_dataset(cutoff, n_frames=20000,CGgraphcutoff=2.0, n_cg=6):
    
    pdb = mdshare.fetch('alanine-dipeptide-nowater.pdb', working_directory='data')
    files = mdshare.fetch('alanine-dipeptide-1-250ns-nowater.xtc', working_directory='data')
    feat = pyemma.coordinates.featurizer(pdb)
    traj = pyemma.coordinates.load(files, features=feat)

    peptide = md.load("data/alanine-dipeptide-nowater.pdb")
    peptide_top = peptide.top.to_dataframe()[0]
    peptide_element = peptide_top['element'].values.tolist()

    traj_reshape = shuffle(traj)[:n_frames].reshape(-1, len(peptide_element),  3) * 10.0 # Change from nanometer to Angstrom 

    # The mapping might be 
    mapping = get_mapping(traj_reshape[::500], CGgraphcutoff, len(peptide_element), n_cg)

    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []

    atomic_nums = np.array( [atomic_num_dict[el] for el in peptide_element] )

    for xyz in traj_reshape:   
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)]))

    # Aggregate CG coorinates 
    for nxyz in nxyz_data:
        xyz = torch.Tensor(nxyz[:, 1:]) 
        CG_xyz = scatter_mean(xyz, mapping, dim=0)
        CG_nxyz = torch.cat((torch.LongTensor(list(range(len(CG_xyz))))[..., None], CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)

        num_CGs_list.append(torch.LongTensor( [len(CG_nxyz)]) )
        CG_mapping_list.append(mapping)

    props = {'nxyz': nxyz_data,
             'CG_nxyz': CG_nxyz_data,
             'num_atoms': num_atoms_list, 
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list
            }

    dataset = CGDataset(props.copy())
    dataset.generate_neighbor_list(cutoff=cutoff)

    return atomic_nums, dataset