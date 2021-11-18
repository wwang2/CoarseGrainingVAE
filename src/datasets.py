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
import tqdm

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


def get_backbone(top):
    backbone_index = []
    for atom in top.atoms:
        if atom.is_backbone:
            backbone_index.append(atom.index)
    return np.array(backbone_index)

def random_rotate_xyz_cg(xyz, cg_xyz ): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    cgatoms = Atoms(positions=cg_xyz, numbers=list( range(cg_xyz.shape[0]) ))
    
    # generate rotation paramters 
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    
    # rotate 
    atoms.rotate(angle, nvec)
    cgatoms.rotate(angle, nvec)
    
    return atoms.positions, cgatoms.positions

def random_rotation(xyz): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    atoms.rotate(angle, nvec)
    return atoms.positions

def backbone_partition(traj, n_cgs, skip=100):
    atomic_nums, protein_index = get_atomNum(traj)
    #indices = traj.top.select_atom_indices('minimal')
    indices = get_backbone(traj.top)

    if indices.shape[0] < n_cgs:
        raise ValueError("N_cg = {} is larger than N_backbone = {}".format(n_cgs, indices.shape[0]) )

    if len(indices) == n_cgs:
        partition = list(range(1, n_cgs))
    else:
        partition = random.sample(range(indices.shape[0]), n_cgs - 1 )
        partition = np.sort(partition)
        segment_sizes = (partition[1:] - partition[:-1]).tolist() + [indices.shape[0] - partition[-1]] + [partition[0]]

        while np.std( segment_sizes) > 1.:
            partition = random.sample(range(indices.shape[0]), n_cgs - 1 )
            partition = np.sort(partition)
            segment_sizes = (partition[1:] - partition[:-1]).tolist() + [indices.shape[0] - partition[-1]] + [partition[0]]

#            print('backbone segment sizes:', segment_sizes)

    mapping = np.zeros(indices.shape[0])
    mapping[partition] = 1
    mapping = np.cumsum(mapping)

    backbone_cgxyz = scatter_mean(torch.Tensor(traj.xyz[:, indices]), 
                          index=torch.LongTensor(mapping), dim=1).numpy()

    mappings = []
    for i in protein_index:
        dist = traj.xyz[::skip, [i], ] - backbone_cgxyz[::skip]
        map_index = np.argmin( np.sqrt( np.sum(dist ** 2, -1)).mean(0) )
        mappings.append(map_index)

    cg_coord = None
    mapping = np.array(mappings)

    return mapping 


def get_diffpool_data(N_cg, trajs, n_data, edgeorder=1, recenter=True, pdb=None):
    props = {}

    num_cgs = []
    num_atoms = []

    z_data = []
    xyz_data = []
    bond_data = []
    angle_data = []
    dihedral_data = []
    hyperedge_data = []

    # todo: not quite generalizable to different proteins
    if pdb is not None:
        mol = Molecule(pdb, guess=['bonds', 'angles', 'dihedrals'] )  
        dihedrals = torch.LongTensor(mol.dihedrals.astype(int))
        angles = torch.LongTensor(mol.angles.astype(int))
    else:
        dihedrals = None
        angles = None

    for traj in tqdm.tqdm(trajs):
        atomic_nums, protein_index = get_atomNum(traj)
        n_atoms = len(atomic_nums)
        frames = traj.xyz[:, protein_index, :] * 10.0 # from nm to Angstrom

        bondgraph = traj.top.to_bondgraph()
        bond_edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
        hyper_edges = get_high_order_edge(bond_edges, edgeorder, n_atoms)

        for xyz in frames: 
            if recenter:
                xyz = xyz - xyz.mean(0)[None, ...]

            #xyz = random_rotation(xyz)
            z_data.append(torch.Tensor(atomic_nums))
            coord = torch.Tensor(xyz)

            xyz_data.append(coord)
            bond_data.append(bond_edges)
            hyperedge_data.append(hyper_edges)

            angle_data.append(angles)
            dihedral_data.append(dihedrals)

            num_cgs.append(torch.LongTensor([N_cg]))
            num_atoms.append(torch.LongTensor([n_atoms]))

    #z_data, xyz_data, num_atoms, num_cgs, bond_data, hyperedge_data, angle_data, dihedral_data = shuffle( z_data, xyz_data, num_atoms, num_cgs, bond_data, hyperedge_data, angle_data, dihedral_data)


    props = {'z': z_data[:n_data],
         'xyz': xyz_data[:n_data],
         'num_atoms': num_atoms[:n_data], 
         'num_CGs':num_cgs[:n_data],
         'bond': bond_data[:n_data],
         'hyperedge': hyperedge_data[:n_data],
        }

    return props

def load_protein_traj(label, ntraj=200): 
    
    traj_files = glob.glob(PROTEINFILES[label]['traj_paths'])[:ntraj]
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

def get_cg_and_xyz(traj, cg_method='backone', n_cgs=None, mapshuffle=0.0):

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

        # randomly shuffle map 
        perm_percent = mapshuffle

        if mapshuffle > 0.0:
            ran_idx = random.sample(range(mapping.shape[0]), int(perm_percent * mapping.shape[0])  )
            idx2map = mapping[ran_idx]
            mapping[ran_idx] = shuffle(idx2map)

        frames = shuffle(frames)

    elif cg_method == 'backbonepartition': 
        mapping = backbone_partition(traj, n_cgs)
        cg_coord = None

    elif cg_method == 'seqpartition':
        partition = random.sample(range(n_atoms), n_cgs - 1 )
        partition = np.sort(partition)
        mapping = np.zeros(n_atoms)
        mapping[partition] = 1
        mapping = np.cumsum(mapping)

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

    #assert len(list(set(mapping.tolist()))) == n_cgs

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
        i += 1
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


def get_high_order_edge(edges, order, natoms):

    # get adj 
    adj = torch.zeros(natoms, natoms)
    adj[edges[:,0], edges[:,1]] = 1
    adj[edges[:,1], edges[:,0]] = 1

    # get higher edges 
    edges = torch.triu(get_higher_order_adj_matrix(adj, order=order)).nonzero()

    return edges 

def build_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None):
    
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    bondgraph = top.to_bondgraph()

    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
    edges = get_high_order_edge(edges, order, atomic_nums.shape[0])

    for xyz in traj:

        xyz = random_rotation(xyz)
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