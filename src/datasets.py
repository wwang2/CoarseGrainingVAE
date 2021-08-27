import torch
import numpy as np 
import networkx as nx
import itertools
from data import *
from torch_scatter import scatter_mean, scatter_add
import glob 
import sys

import mdtraj as md
import mdshare
import pyemma
from sklearn.utils import shuffle

atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7, 'S':16, 'Se': 34}

PROTEINFILES = {'covid': {'traj_paths': "../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA-00*.dcd", 
                              'pdb_path': '../data/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA/sarscov2-11441075-no-water-zinc-glueCA/DESRES-Trajectory_sarscov2-11441075-no-water-zinc-glueCA.pdb', 
                              'file_type': 'dcd'},

                'chignolin': {'traj_paths': "../data/filtered/e1*/*.xtc", 
                              'pdb_path': '../data/filtered/filtered.pdb', 
                              'file_type': 'xtc'}, 
                'dipeptide': 
                            {'pdb_path': '../data/alanine-dipeptide-nowater.pdb', 
                            'traj_paths': '../data/alanine-dipeptide-*-250ns-nowater.xtc',
                            'file_type': 'xtc'
                             },
              'pentapeptide': 
                            {'pdb_path': '../data/pentapeptide-impl-solv.pdb',
                             'traj_paths': '../data/pentapeptide-*-500ns-impl-solv.xtc',
                             'file_type': 'xtc'
                            }}


def get_diffpool_data(N_cg, trajs, frame_skip=1000):
    props = {}

    num_cgs = []
    num_atoms = []

    z_data = []
    xyz_data = []
    graph_data = []

    for traj in trajs:
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = len(atomic_nums)
        frames = traj.xyz[:, protein_index, :] * 10.0 # from nm to Angstrom

        bondgraph = traj.top.to_bondgraph()
        edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 

        for xyz in frames[::frame_skip]: 
            z_data.append(torch.Tensor(atomic_nums))
            coord = torch.Tensor(xyz)
            xyz_data.append(coord)
            graph_data.append(edges)
            num_cgs.append(torch.LongTensor([N_cg]))
            num_atoms.append(torch.LongTensor([n_atoms]))
            
    props = {'z': z_data,
         'xyz': xyz_data,
         'num_atoms': num_atoms, 
         'num_CGs':num_cgs,
         'bond_edge_list': graph_data
        }

    return props

def load_protein_traj(label): 
    
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])[:200]
    pdb_file = PROTEINFILES[label]['pdb_path']
    file_type = PROTEINFILES[label]['file_type']
    
    if file_type == 'xtc':
        trajs = [md.load_xtc(file,
                    top=pdb_file) for file in traj_files]
    elif file_type == 'dcd':
        trajs = [md.load_dcd(file,
                    top=pdb_file) for file in traj_files]
    else:
        raise ValueError("file type {} not recognized".format(file_type))
                
    traj = md.join(trajs)
                   
    return traj

def get_cg_and_xyz(traj, cg_method='backone', n_cgs=None):

    atomic_nums, protein_index = get_atomNum(traj)
    n_atoms = len(atomic_nums)
    skip = 200
    # get alpha carbon only 

    frames = traj.xyz[:, protein_index, :] * 10.0 

    if cg_method in ['minimal', 'alpha']:
        mappings = []
        print("Note, using CG method {}, user-specified N_cg will be overwritten".format(cg_method))

        indices = traj.top.select_atom_indices(cg_method)
        for i in protein_index:
            dist = traj.xyz[::skip, [i], ] - traj.xyz[::skip, indices, :]
            map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )
            mappings.append(map_index)

        cg_coord = traj.xyz[:, indices, :] * 10.0
        mapping = np.array(mappings)

        n_cgs = len(indices)
        frames, cg_coord = shuffle(frames, cg_coord)

    elif cg_method =='newman':

        if n_cgs is None:
            raise ValueError("need to provided number of CG sites")

        protein_top = traj.top.subset(protein_index)
        g = protein_top.to_bondgraph()
        paritions = get_partition(g, n_cgs)
        mapping = parition2mapping(paritions, n_atoms)
        mapping = np.array(mapping)
        cg_coord = None

        frames = shuffle(frames)

    elif cg_method =='random':

        mapping = get_random_mapping(n_cgs, n_atoms)
        cg_coord = None
        frames = shuffle(frames)

    else:
        raise ValueError("{} coarse-graining option not available".format(cg_method))


    # print coarse graining summary 

    print("CG method: {}".format(cg_method))
    print("Number of CG sites: {}".format(mapping.max() + 1))

    assert len(list(set(mapping.tolist()))) == n_cgs

    mapping = torch.LongTensor( mapping)
    
    return mapping, frames, cg_coord


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

def get_partition(G, n_partitions):
    
    # adj = [tuple(pair) for pair in nbr_list]
    # G = nx.Graph()
    # G.add_edges_from(adj)

    G = nx.convert_node_labels_to_integers(G)
    comp = nx.community.girvan_newman(G)

    for communities in itertools.islice(comp, n_partitions-1):
            partitions = tuple(sorted(c) for c in communities)
        
    return partitions 

def compute_mapping(atomic_nums, traj, cutoff, n_atoms, n_cgs, skip):

    # get bond graphs 
    g = traj.top.to_bondgraph()
    paritions = get_partition(g, n_cgs)
    mapping = parition2mapping(paritions, n_atoms)

    return mapping

def get_mapping(label, cutoff, n_atoms, n_cgs, skip=200):

    peptide = get_peptide_top(label)

    files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')

    atomic_nums, traj = get_traj(peptide, files, n_frames=20000)
    peptide_element = [atom.element.symbol for atom in peptide.top.atoms]

    if len(traj) < skip:
        skip = len(traj)

    mappings = compute_mapping(atomic_nums, traj,  cutoff,  n_atoms, n_cgs, skip)

    return mappings.long()

def get_random_mapping(n_cg, n_atoms):

    mapping = torch.LongTensor(n_atoms).random_(0, n_cg)
    i = 1
    while len(mapping.unique()) != n_cg and i <= 10000000:
        i + 1
        mapping = torch.LongTensor(n_atoms).random_(0, n_cg)

    return mapping

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


def build_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, cg_traj=None):
    
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    bondgraph = top.to_bondgraph()
    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 

    for xyz in traj:   
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)]))
        bond_edge_list.append(edges)

    # Aggregate CG coorinates 
    for i, nxyz in enumerate(nxyz_data):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj is not None:
            CG_xyz = torch.Tensor( cg_traj[i] )
        else:
            CG_xyz = scatter_mean(xyz, mapping, dim=0)

        CG_nxyz = torch.cat((torch.LongTensor(list(range(len(CG_xyz))))[..., None], CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)

        num_CGs_list.append(torch.LongTensor( [len(CG_nxyz)]) )
        CG_mapping_list.append(mapping)

    props = {'nxyz': nxyz_data,
             'CG_nxyz': CG_nxyz_data,
             'num_atoms': num_atoms_list, 
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list, 
             'bond_edge_list':  bond_edge_list
            }

    dataset = CGDataset(props.copy())
    #dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)
    
    return dataset

def get_peptide_dataset(atom_cutoff,  cg_cutoff, label, mapping, n_frames=20000, n_cg=6):

    pdb = mdshare.fetch(DATALABELS[label]['pdb'], working_directory='data')
    files = mdshare.fetch(DATALABELS[label]['xtc'], working_directory='data')
    pdb = md.load("data/{}".format(DATALABELS[label]['pdb']))
    
    atomic_nums, traj_reshape = get_traj(pdb, files, n_frames)

    dataset = build_dataset(mapping, traj_reshape, atom_cutoff, cg_cutoff, atomic_nums, pdb.top)

    return atomic_nums, dataset