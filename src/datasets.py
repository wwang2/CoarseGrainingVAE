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

atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7, 'S':16, 'Se': 34}

DATALABELS = {'dipeptide': 
                            {'pdb': 'alanine-dipeptide-nowater.pdb', 
                            'xtc': 'alanine-dipeptide-*-250ns-nowater.xtc',
                            'n_atoms': 22
                             },
              'pentapeptide': 
                            {'pdb': 'pentapeptide-impl-solv.pdb',
                             'xtc': 'pentapeptide-*-500ns-impl-solv.xtc',
                             'n_atoms': 94
                            }
              }

PROTEINFILES = {'chignolin': {'traj_paths': "../data/filtered/e1*/*.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'}}

def load_protein_traj(label): 
    
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])
    pdb_file = PROTEINFILES[label]['pdb_path']
    
    file_type = PROTEINFILES[label]['file_type']
    
    if file_type == 'xtc':
        trajs = [md.load_xtc(file,
                    top=pdb_file) for file in traj_files[:20]]
    elif file_type == 'dcd':
        trajs = [md.load_dcd(file,
                    top=pdb_file) for file in traj_files]
    else:
        raise ValueError("file type not recognizable")
                
    traj = md.join(trajs)
    
    #protein_index = traj.top.select("protein")
                   
    return traj
    
def get_Calpha(traj):
    
#     protein_index = traj.top.select("protein")
#     protein_top = traj.top.subset(protein_index)
    
#     elements = [atom.element.symbol for atom in protein_top.atoms]
#     atomic_nums = np.array([atomic_num_dict[el] for el in elements] )

    atomic_nums, protein_index = get_atomNum(traj)

    # get alpha carbon only 
    alpha_indices = traj.top.select_atom_indices('alpha')
    
    #ref_xyz = traj.xyz[0]
    mappings = []
    skip = 200
    for i in protein_index:

        dist = traj.xyz[::skip, [i], ] - traj.xyz[::skip, alpha_indices, :]
        map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )

        mappings.append(map_index)


    assert len(list(set(mappings))) == len(alpha_indices)
    
    return mappings


def get_atomNum(traj):
    
    atomic_nums = [atom.element.number for atom in traj.top.atoms]
    
    protein_index = traj.top.select("protein")
    protein_top = traj.top.subset(protein_index)

    atomic_nums = [atom.element.number for atom in protein_top.atoms]
    
    return np.array(atomic_nums), protein_index

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

def compute_mapping(atomic_nums, traj, cutoff, n_atoms, n_cgs, skip):

    traj = shuffle(traj)[::skip].reshape(-1, len(atomic_nums),  3)
    mappings = []
    for frame in traj:
        nbr_list = compute_nbr_list(torch.Tensor(frame), cutoff)
        paritions = get_partition(nbr_list, n_cgs)     
        mapping = parition2mapping(paritions, n_atoms)
        mappings.append(mapping)

    mappings = torch.Tensor( np.stack(mappings) )
    mappings = mappings.mode(0)[0]


    return mappings

def get_mapping(label, cutoff, n_atoms, n_cgs, skip=200):

    peptide = get_peptide_top(label)

    files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')

    atomic_nums, traj = get_traj(peptide, files, n_frames=20000)

    #peptide_top = peptide.top.to_dataframe()[0]
    peptide_element = [atom.element.symbol for atom in peptide.top.atoms]

    if len(traj) < skip:
        skip = len(traj)

    mappings = compute_mapping(atomic_nums, traj,  cutoff,  n_atoms, n_cgs, skip)

    # traj_reshape = shuffle(traj)[::skip].reshape(-1, len(peptide_element),  3)

    # mappings = []
    # for frame in traj_reshape:
    #     nbr_list = compute_nbr_list(torch.Tensor(frame), cutoff)
    #     paritions = get_partition(nbr_list, n_cgs)     
    #     mapping = parition2mapping(paritions, n_atoms)
    #     mappings.append(mapping)

    # mappings = torch.Tensor( np.stack(mappings) )
    # mappings = mappings.mode(0)[0]
    
    return mappings.long()

def get_random_mapping(n_cg, n_atoms):
    
    # todo: need to check generate mapping covers all types 

    return torch.LongTensor(n_atoms).random_(0, n_cg) 

def get_peptide_top(label):

    pdb = mdshare.fetch(DATALABELS[label]['pdb'], working_directory='data')
    peptide = md.load("data/{}".format(DATALABELS[label]['pdb']))

    return peptide

def get_traj(pdb, files, n_frames, shuffle=False):
    feat = pyemma.coordinates.featurizer(pdb)
    traj = pyemma.coordinates.load(files, features=feat)
    traj = np.concatenate(traj)

    peptide_element = [atom.element.symbol for atom in pdb.top.atoms]

    if shuffle: 
        traj = shuffle(traj)
        
    traj_reshape = traj.reshape(-1, len(peptide_element),  3)[:n_frames] * 10.0 # Change from nanometer to Angstrom 
    atomic_nums = np.array([atomic_num_dict[el] for el in peptide_element] )
    
    return atomic_nums, traj_reshape

# need a function to get mapping, and CG coordinates simultanesouly. We can have alpha carbon as the CG site


def build_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, cg_traj=None):
    
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []

    for xyz in traj:   
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)]))

    # Aggregate CG coorinates 
    for i, nxyz in enumerate(nxyz_data):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj == None:
            CG_xyz = scatter_mean(xyz, mapping, dim=0)
        else:
            CG_xyz = cg_traj[i]

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
    #dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)
    
    return dataset

def get_peptide_dataset(atom_cutoff,  cg_cutoff, label, mapping, n_frames=20000, n_cg=6):

    pdb = mdshare.fetch(DATALABELS[label]['pdb'], working_directory='data')
    files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')
    pdb = md.load("data/{}".format(DATALABELS[label]['pdb']))
    
    atomic_nums, traj_reshape = get_traj(pdb, files, n_frames)

    dataset = build_dataset(mapping, traj_reshape, atom_cutoff, cg_cutoff, atomic_nums)

    return atomic_nums, dataset