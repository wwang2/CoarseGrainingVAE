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


class CGDataset(TorchDataset):
    
    def __init__(self,
                 props,
                 check_props=True):
        self.props = props

    def __len__(self):
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}
    

    def generate_neighbor_list(self, atom_cutoff, cg_cutoff,  device='cpu',  undirected=True):
        
        if cg_cutoff == None:
            cg_cutoff = atom_cutoff

        # todo : create progress bar
        nbr_list = []
        cg_nbr_list = []

        for nxyz in tqdm(self.props['nxyz'], desc='building nbr list'):
            nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, atom_cutoff, undirected).to("cpu"))

        for nxyz in tqdm(self.props['CG_nxyz'], desc='building CG nbr list'):
            cg_nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, atom_cutoff, undirected).to("cpu"))

        self.props['nbr_list'] = nbr_list
        self.props['CG_nbr_list'] = cg_nbr_list


def CG_collate(dicts):
    # new indices for the batch: the first one is zero and the
    # last does not matter

    cumulative_atoms = np.cumsum([0] + [d['num_atoms'] for d in dicts])[:-1]
    cumulative_CGs = np.cumsum([0] + [d['num_CGs'] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        #if 'nbr_list' in d:
        d['nbr_list'] = d['nbr_list'] + int(n)
            
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
    
    subset = CGDataset(
        props={key: [val[i] for i in indices]
               for key, val in dataset.props.items()},
    )
    
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
