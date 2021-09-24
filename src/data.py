import torch
import numbers
import numpy as np
import copy
from copy import deepcopy
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm 
import copy 
import sys



def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj, order):
    """
    from https://github.com/MinkaiXu/ConfVAE-ICML21/blob/main/utils/transforms.py
    Args:
        adj:        (N, N)
    """
    adj_mats = [torch.eye(adj.size(0)).long(), binarize(adj + torch.eye(adj.size(0)).long())]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    # print(adj_mats)

    order_mat = torch.zeros_like(adj)
    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return order_mat



def get_neighbor_list(xyz, device='cpu', cutoff=5, undirected=True):

    xyz = torch.Tensor(xyz).to(device)
    n = xyz.size(0)

    # calculating distances
    dist = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

    # neighbor list
    mask = (dist <= cutoff)
    mask[np.diag_indices(n)] = 0
    nbr_list = torch.nonzero(mask)

    if undirected:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    return nbr_list

class DiffPoolDataset(TorchDataset):
    
    def __init__(self,
                 props,
                 check_props=True):
        self.props = props

    def __len__(self):
        return len(self.props['xyz'])

    def __getitem__(self, idx):
        single_item = {key: val[idx] for key, val in self.props.items()}
        # recenter geometry 
        #single_item['xyz'] = single_item['xyz'] - single_item['xyz'].mean(0).unsqueeze(0)
        return single_item
    
    def generate_neighbor_list(self, atom_cutoff,  device='cpu',  undirected=False):
        nbr_list = []
        for xyz in tqdm(self.props['xyz'], desc='building nbr list', file=sys.stdout):
            nbr_list.append(get_neighbor_list(xyz, device, atom_cutoff, undirected).to("cpu"))

        self.props['nbr_list'] = nbr_list


def DiffPool_collate(dicts):

    batch = {}
    xyzs, xyz_pad = padding_tensor([dict['xyz'] for dict in dicts])
    zs, z_pad = padding_tensor([dict['z'] for dict in dicts])
    
    bonds_batch = []
    bonds = [dict['bond'] for dict in dicts]
    hyperedge_batch = []
    hyperedges = [dict['hyperedge'] for dict in dicts]
    nbrs_batch = []
    nbrs = [dict['nbr_list'] for dict in dicts]
    
    for i, bond in enumerate(bonds):
        batch_index = torch.LongTensor([i] * bond.shape[0])        
        bonds_batch.append((torch.cat( (batch_index.unsqueeze(-1), bond),dim=-1 ) )) 

    for i, bond in enumerate(hyperedges):
        batch_index = torch.LongTensor([i] * bond.shape[0])        
        hyperedge_batch.append((torch.cat( (batch_index.unsqueeze(-1), bond),dim=-1 ) )) 
        
    for i, nbr in enumerate(nbrs):
        batch_index = torch.LongTensor([i] * nbr.shape[0])        
        nbrs_batch.append((torch.cat( (batch_index.unsqueeze(-1), nbr),dim=-1 ) )) 
        
    nbrs_batch = torch.cat(nbrs_batch)
    bonds_batch = torch.cat(bonds_batch)
    hyperedge_batch = torch.cat(hyperedge_batch)
        
    return {'z':zs, 'xyz': xyzs, 'nbr_list': nbrs_batch, 'bonds': bonds_batch, 'hyperedges': hyperedge_batch,  'pad': z_pad}


def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    
    out_dims = [num, max_len] + list(sequences[0].shape)[1:]
    
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask
    

class CGDataset(TorchDataset):
    
    def __init__(self,
                 props,
                 check_props=True):
        self.props = props

    def __len__(self):
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}

    def generate_aux_edges(self, auxcutoff, device='cpu', undirected=True):
        edge_list = []
        
        for nxyz in tqdm(self.props['nxyz'], desc='building aux edge list', file=sys.stdout):
            edge_list.append(get_neighbor_list(nxyz[:, 1:4], device, auxcutoff, undirected).to("cpu"))

        self.props['bond_edge_list'] = edge_list

    def generate_neighbor_list(self, atom_cutoff, cg_cutoff, device='cpu', undirected=True):

        # todo : create progress bar
        #edge_list = []
        nbr_list = []
        cg_nbr_list = []

        for nxyz in tqdm(self.props['nxyz'], desc='building nbr list', file=sys.stdout):
            nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, atom_cutoff, undirected).to("cpu"))

        # for nxyz in tqdm(self.props['nxyz'], desc='building edge list', file=sys.stdout):
        #     edge_list.append(get_neighbor_list(nxyz[:, 1:4], device, 3.0, undirected).to("cpu"))

        if cg_cutoff is not None:    
            for nxyz in tqdm(self.props['CG_nxyz'], desc='building CG nbr list', file=sys.stdout):
                cg_nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, cg_cutoff, undirected).to("cpu"))

        elif cg_cutoff is None :
            for i, bond in enumerate( self.props['bond_edge_list'] ):
                
                mapping = self.props['CG_mapping'][i]
                n_atoms = self.props['num_atoms'][i]
                n_cgs = self.props['num_CGs'][i]
                adj = torch.zeros(n_atoms, n_atoms)
                adj[bond[:, 0], bond[:,1]] = 1
                adj[bond[:, 1], bond[:,0]] = 1

                # get assignment vector 
                assign = torch.zeros(n_atoms, n_cgs)
                atom_idx = torch.LongTensor(list(range(n_atoms)))

                assign[atom_idx, mapping] = 1
                # compute CG ajacency 
                cg_adj = assign.transpose(0,1).matmul(adj).matmul(assign) 

                cg_nbr = cg_adj.nonzero()
                cg_nbr = cg_nbr[cg_nbr[:, 0] != cg_nbr[:, 1]]

                cg_nbr_list.append( cg_nbr )

        self.props['nbr_list'] = nbr_list
        self.props['CG_nbr_list'] = cg_nbr_list
        #self.props['edge_list'] = edge_list


def CG_collate(dicts):
    # new indices for the batch: the first one is zero and the
    # last does not matter

    cumulative_atoms = np.cumsum([0] + [d['num_atoms'] for d in dicts])[:-1]
    cumulative_CGs = np.cumsum([0] + [d['num_CGs'] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        #if 'nbr_list' in d:
        d['nbr_list'] = d['nbr_list'] + int(n)
        d['bond_edge_list'] = d['bond_edge_list'] + int(n)
            
    for n, d in zip(cumulative_CGs, dicts):
       # if 'CGmapping' in d:
        d['CG_mapping'] = d['CG_mapping'] + int(n)
        d['CG_nbr_list'] = d['CG_nbr_list'] + int(n)
        
    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if hasattr(val, 'shape') and len(val.shape) > 0:
            batch[key] = torch.cat([
                data[key]
                for data in dicts
            ], dim=0)
        else:
            batch[key] = torch.stack(
                [data[key] for data in dicts],
                dim=0
            )

    return batch

def split_train_test(dataset,
                     test_size=0.2):

    idx = list(range(len(dataset)))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, shuffle=True)

    train = get_subset_by_indices(idx_train, dataset)
    test = get_subset_by_indices(idx_test, dataset)

    return train, test


def get_subset_by_indices(indices, dataset):

    if isinstance(dataset, CGDataset):
        subset = CGDataset(
            props={key: [val[i] for i in indices]
                   for key, val in dataset.props.items()},
        )
    elif isinstance(dataset, DiffPoolDataset):
        subset = DiffPoolDataset(
            props={key: [val[i] for i in indices]
                   for key, val in dataset.props.items()},
        )
    else:
        raise ValueError("dataset type {} not recognized".format(dataset.__name__))

    return subset 


def split_train_validation_test(dataset,
                                val_size=0.2,
                                test_size=0.2,
                                **kwargs):

    train, validation = split_train_test(dataset,
                                         test_size=val_size,
                                         **kwargs)
    train, test = split_train_test(train,
                                   test_size=test_size / (1 - val_size),
                                   **kwargs)

    return train, validation, test
