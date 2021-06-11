import torch
import numbers
import numpy as np
import copy
import nff.utils.constants as const
from copy import deepcopy
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import train_test_split
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset as TorchDataset
from nff.data.graphs import (reconstruct_atoms, get_neighbor_list,
                             DISTANCETHRESHOLDICT_Z)

class CGDataset(TorchDataset):
    
    def __init__(self,
                 props,
                 check_props=True):
        self.props = props

    def __len__(self):
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}
    

    def generate_neighbor_list(self, cutoff, undirected=True):
        self.props['nbr_list'] = [
            get_neighbor_list(nxyz[:, 1:4], cutoff, undirected)
            for nxyz in self.props['nxyz']
        ]
        
        self.props['CG_nbr_list'] = [
            get_neighbor_list(nxyz[:, 1:4], cutoff, undirected)
            for nxyz in self.props['CG_nxyz']
        ]


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
